"""Helpers for syncing one or more .oikb.yaml entries into a Knowledge Base."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from oikb.connectors import BaseConnector, ManifestEntry
from oikb.connectors.composite import CompositeConnector
from oikb.sync import SyncResult, build_manifest_filter, parse_size, run_sync

MANIFEST_DIR = Path("manifest")


def group_entries_by_kb(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group yaml entries that share the same kb-id."""
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for entry in entries:
        groups.setdefault(entry["kb-id"], []).append(entry)
    return list(groups.values())


def manifest_filter_for_entry(
    entry: dict[str, Any],
    max_file_size: str | None = None,
) -> Callable | None:
    entry_filter = entry.get("filter", {})
    include = entry_filter.get("include")
    exclude = entry_filter.get("exclude")
    max_size = entry_filter.get("max-size") or max_file_size
    if not include and not exclude and not max_size:
        return None
    return build_manifest_filter(
        include=include,
        exclude=exclude,
        max_size=parse_size(max_size),
    )


def _safe_manifest_stem(name: str) -> str:
    """Sanitize a name/kb-id for use as a manifest filename stem."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip())
    safe = safe.strip(" .")
    return safe or "kb"


def manifest_path_for_group(entries: list[dict[str, Any]]) -> Path:
    """Return the on-disk path for a KB group's manifest file (keyed by kb-id)."""
    return MANIFEST_DIR / f"{_safe_manifest_stem(entries[0]['kb-id'])}.json"


def manifest_path_for_kb(kb_id: str, name: str | None = None) -> Path:
    """Return the on-disk path for a single-source / CLI-mode KB.

    Prefers an explicit --name when given, otherwise uses kb-id.
    """
    stem = name or kb_id
    return MANIFEST_DIR / f"{_safe_manifest_stem(stem)}.json"


def save_manifest_file(
    path: Path,
    kb_id: str,
    parts: list[dict[str, Any]],
) -> Path:
    """Write a KB manifest JSON file. Returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kb_id": kb_id, "parts": parts}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def load_manifest_file(path: Path) -> dict[str, Any]:
    """Load a previously saved KB manifest JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "parts" not in data:
        raise ValueError(f"Invalid manifest file: {path}")
    return data


def _entries_from_part(part: dict[str, Any]) -> list[ManifestEntry]:
    return [ManifestEntry.from_dict(e) for e in part.get("entries", [])]


def _match_saved_part(
    saved_parts: list[dict[str, Any]],
    entry: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Find the saved part that corresponds to a yaml/CLI entry."""
    name = entry.get("name")
    source = entry.get("source")
    if name:
        for part in saved_parts:
            if part.get("name") == name:
                return part
    if source:
        for part in saved_parts:
            if part.get("source") == source:
                return part
    if 0 <= index < len(saved_parts):
        return saved_parts[index]
    raise ValueError(
        f"No matching part in manifest for source={source!r} name={name!r}"
    )


def build_connector_for_entries(
    entries: list[dict[str, Any]],
    resolve_connector: Callable[..., BaseConnector],
    max_file_size: str | None = None,
) -> BaseConnector:
    """Build a single connector for one yaml entry or a merged composite."""
    if len(entries) == 1:
        entry = entries[0]
        return resolve_connector(
            entry["source"],
            entry.get("branch"),
            entry.get("path"),
            entry,
        )

    parts: list[tuple[BaseConnector, list[ManifestEntry]]] = []
    for entry in entries:
        connector = resolve_connector(
            entry["source"],
            entry.get("branch"),
            entry.get("path"),
            entry,
        )
        manifest = connector.build_manifest()
        manifest_filter = manifest_filter_for_entry(entry, max_file_size)
        if manifest_filter:
            manifest = manifest_filter(manifest)
        parts.append((connector, manifest))

    return CompositeConnector(parts)


def collect_manifest_parts(
    entries: list[dict[str, Any]],
    resolve_connector: Callable[..., BaseConnector],
    max_file_size: str | None = None,
) -> tuple[list[tuple[BaseConnector, list[ManifestEntry]]], list[dict[str, Any]]]:
    """Scan sources and return connector+manifest pairs plus serializable parts."""
    pairs: list[tuple[BaseConnector, list[ManifestEntry]]] = []
    parts_data: list[dict[str, Any]] = []

    for entry in entries:
        connector = resolve_connector(
            entry["source"],
            entry.get("branch"),
            entry.get("path"),
            entry,
        )
        try:
            manifest = connector.build_manifest()
            manifest_filter = manifest_filter_for_entry(entry, max_file_size)
            if manifest_filter:
                manifest = manifest_filter(manifest)
            pairs.append((connector, manifest))
            parts_data.append({
                "name": entry.get("name"),
                "source": entry["source"],
                "entries": [e.to_dict() for e in manifest],
            })
        except Exception:
            connector.close()
            for c, _ in pairs:
                c.close()
            raise

    return pairs, parts_data


def build_and_save_manifest(
    entries: list[dict[str, Any]],
    resolve_connector: Callable[..., BaseConnector],
    max_file_size: str | None = None,
    path: Path | None = None,
) -> tuple[Path, int]:
    """Build manifests for a KB group and write them to disk.

    Returns (path written, total entry count).
    """
    pairs, parts_data = collect_manifest_parts(
        entries, resolve_connector, max_file_size,
    )
    try:
        total = sum(len(m) for _, m in pairs)
        out = path or manifest_path_for_group(entries)
        save_manifest_file(out, entries[0]["kb-id"], parts_data)
        return out, total
    finally:
        closed: set[int] = set()
        for connector, _ in pairs:
            token = id(connector)
            if token not in closed:
                connector.close()
                closed.add(token)


def connector_from_manifest(
    entries: list[dict[str, Any]],
    resolve_connector: Callable[..., BaseConnector],
    saved: dict[str, Any],
) -> tuple[BaseConnector, list[ManifestEntry] | None]:
    """Create connectors wired to a previously saved manifest.

    Returns (connector, preloaded_manifest). For a single source,
    preloaded_manifest is the entry list. For composites, the
    CompositeConnector already embeds the manifest and preloaded is None.
    """
    saved_parts = saved["parts"]
    if len(entries) == 1:
        entry = entries[0]
        connector = resolve_connector(
            entry["source"],
            entry.get("branch"),
            entry.get("path"),
            entry,
        )
        part = _match_saved_part(saved_parts, entry, 0)
        return connector, _entries_from_part(part)

    pairs: list[tuple[BaseConnector, list[ManifestEntry]]] = []
    for i, entry in enumerate(entries):
        connector = resolve_connector(
            entry["source"],
            entry.get("branch"),
            entry.get("path"),
            entry,
        )
        part = _match_saved_part(saved_parts, entry, i)
        pairs.append((connector, _entries_from_part(part)))
    return CompositeConnector(pairs), None


def sources_label(entries: list[dict[str, Any]]) -> str:
    if len(entries) == 1:
        return entries[0].get("source", "?")
    return "+".join(entry.get("source", "?") for entry in entries)


def run_entries_sync(
    client: Any,
    entries: list[dict[str, Any]],
    *,
    resolve_connector: Callable[..., BaseConnector],
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    concurrency: int = 1,
    max_file_size: str | None = None,
    from_manifest: bool = False,
) -> SyncResult:
    """Sync one kb-id group (single source or merged composite)."""
    preloaded: list[ManifestEntry] | None = None

    if from_manifest:
        path = manifest_path_for_group(entries)
        saved = load_manifest_file(path)
        connector, preloaded = connector_from_manifest(
            entries, resolve_connector, saved,
        )
        return run_sync(
            client=client,
            connector=connector,
            kb_id=entries[0]["kb-id"],
            dry_run=dry_run,
            verbose=verbose,
            quiet=quiet,
            concurrency=(
                entries[0].get("concurrency", concurrency)
                if len(entries) == 1
                else max(entry.get("concurrency", concurrency) for entry in entries)
            ),
            preloaded_manifest=preloaded,
        )

    if len(entries) == 1:
        entry = entries[0]
        connector = build_connector_for_entries(entries, resolve_connector, max_file_size)
        return run_sync(
            client=client,
            connector=connector,
            kb_id=entry["kb-id"],
            dry_run=dry_run,
            verbose=verbose,
            quiet=quiet,
            manifest_filter=manifest_filter_for_entry(entry, max_file_size),
            concurrency=entry.get("concurrency", concurrency),
        )

    connector = build_connector_for_entries(entries, resolve_connector, max_file_size)
    return run_sync(
        client=client,
        connector=connector,
        kb_id=entries[0]["kb-id"],
        dry_run=dry_run,
        verbose=verbose,
        quiet=quiet,
        concurrency=max(entry.get("concurrency", concurrency) for entry in entries),
    )
