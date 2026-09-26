"""Confluence connector — sync a Confluence space to a Knowledge Base.

Supports Confluence REST API v1 (self-hosted) and v2 (Cloud). Pages are
exported as plain text. Select the API via CONFLUENCE_API_VERSION (default v2).
For v1, the page manifest is built via CQL ``/content/search``.

Optional attachment sync (v1 only) downloads allowed file types as raw bytes
for Open WebUI / Tika to parse. Enable via ``attachments.enabled`` in
``.oikb.yaml``.

For API v1, page export also lists attachment filenames in the page text.
Filter extensions via ``CONFLUENCE_PAGE_ATTACHMENT_EXTENSIONS`` (comma-separated;
defaults to the built-in attachment allowlist when unset).

Auth via CONFLUENCE_URL, CONFLUENCE_USER, and CONFLUENCE_TOKEN env vars:
  - Server/Data Center PAT: set CONFLUENCE_TOKEN only (Bearer auth)
  - Cloud API token: set CONFLUENCE_USER (email) + CONFLUENCE_TOKEN (Basic auth)
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup, NavigableString

from oikb.connectors import BaseConnector, ManifestEntry
from oikb.drawio import (
    attachment_extension,
    drawio_manifest_filename,
    drawio_to_text,
    extract_attachment_label_names,
    is_drawio_attachment,
)
from oikb.sync import parse_size

log = logging.getLogger(__name__)


BASE_ENDPOINTS = {
    "v1": "/rest/api",
    "v2": "/wiki/api/v2",
}

DEFAULT_ALLOWED_ATTACHMENT_EXTENSIONS: frozenset[str] = frozenset({
    "pdf", "docx", "doc", "xlsx", "pptx", "odt", "rtf", "html", "txt", "msg", "json",
    "drawio",
})
DEFAULT_ATTACHMENTS_MAX_SIZE = 20 * 1024 * 1024  # 20mb
PAGE_ATTACHMENT_EXTENSIONS_ENV = "CONFLUENCE_PAGE_ATTACHMENT_EXTENSIONS"

# Transient gateway / server errors worth retrying (e.g. deep offset pagination).
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_HTTP_MAX_ATTEMPTS = 5
_HTTP_RETRY_BASE_DELAY_S = 10.0


_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


@dataclass(frozen=True, slots=True)
class AttachmentsConfig:
    """Attachment sync settings for a Confluence source."""

    enabled: bool = False
    allowed_extensions: frozenset[str] = DEFAULT_ALLOWED_ATTACHMENT_EXTENSIONS
    max_size: int = DEFAULT_ATTACHMENTS_MAX_SIZE


@dataclass(frozen=True, slots=True)
class _AttachmentRef:
    attachment_id: str
    download_path: str
    page_id: str
    filename: str
    is_drawio: bool = False


def parse_attachments_config(
    attachments: dict[str, Any] | None,
    *,
    defaults_attachments: dict[str, Any] | None = None,
    filter_max_size: int | None = None,
) -> AttachmentsConfig:
    """Parse attachment sync settings from a .oikb.yaml source entry.

    ``allowed-extensions`` is read only from ``defaults.attachments`` (shared
    across all sources). Per-source ``attachments`` may set ``enabled`` and
    ``max-size`` only.
    """
    attachments = attachments or {}

    allowed_raw = (defaults_attachments or {}).get("allowed-extensions")
    if allowed_raw is not None:
        allowed_extensions = frozenset(
            ext.lower().lstrip(".") for ext in allowed_raw if ext
        )
    else:
        allowed_extensions = DEFAULT_ALLOWED_ATTACHMENT_EXTENSIONS

    max_size = parse_size(attachments.get("max-size"))
    if max_size is None:
        max_size = filter_max_size
    if max_size is None:
        max_size = DEFAULT_ATTACHMENTS_MAX_SIZE

    return AttachmentsConfig(
        enabled=bool(attachments.get("enabled", False)),
        allowed_extensions=allowed_extensions,
        max_size=max_size,
    )


def _page_attachment_extensions_from_env() -> frozenset[str]:
    """Allowed extensions for attachment names embedded in page text."""
    raw = os.environ.get(PAGE_ATTACHMENT_EXTENSIONS_ENV)
    if raw is None:
        return DEFAULT_ALLOWED_ATTACHMENT_EXTENSIONS
    raw = raw.strip()
    if not raw:
        return DEFAULT_ALLOWED_ATTACHMENT_EXTENSIONS
    return frozenset(
        part.lower().lstrip(".")
        for part in re.split(r"[,;]+", raw)
        if part.strip()
    )


def _sanitize_path_segment(name: str) -> str:
    """Sanitize a single path segment (ancestor dir or filename stem)."""
    cleaned = _INVALID_PATH_CHARS.sub("_", name).strip()
    return cleaned or "_"


def _fmt_size(n: int) -> str:
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n}B"


def _ancestor_dir_path_v1(page: dict[str, Any]) -> str:
    """Build KB directory path from Confluence v1 ancestors list."""
    segments = [
        _sanitize_path_segment(a.get("title", ""))
        for a in page.get("ancestors") or []
        if a.get("title")
    ]
    return "/".join(segments)


def _ancestor_dir_path_v2(
    page_id: str, pages_by_id: dict[str, dict[str, Any]]
) -> str:
    """Build KB directory path by walking Confluence v2 parentId chain."""
    segments: list[str] = []
    current = pages_by_id.get(str(page_id))
    if not current:
        return ""

    parent_id = current.get("parentId")
    visited: set[str] = set()
    while parent_id:
        pid = str(parent_id)
        if pid in visited:
            break
        visited.add(pid)
        parent = pages_by_id.get(pid)
        if not parent:
            break
        title = parent.get("title", "")
        if title:
            segments.append(_sanitize_path_segment(title))
        parent_id = parent.get("parentId")

    segments.reverse()
    return "/".join(segments)


def _parse_api_version(api_version: str | None) -> str:
    version = (api_version or os.environ.get("CONFLUENCE_API_VERSION", "v2")).lower()
    if version not in BASE_ENDPOINTS:
        valid = ", ".join(sorted(BASE_ENDPOINTS))
        raise ValueError(
            f"Invalid Confluence API version {version!r}. "
            f"Expected one of: {valid}. Set CONFLUENCE_API_VERSION=v1 or v2."
        )
    return version


def _table_to_markdown(table_tag) -> str:
    """Render a Confluence storage-format <table> as a Markdown table.

    rowspan/colspan are not expanded — each <td>/<th> maps to one cell.
    """
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr", recursive=True):
        # Direct td/th only (not from nested tables inside a cell).
        cells = tr.find_all(["td", "th"], recursive=False)
        cell_texts = [
            re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
            for cell in cells
        ]
        if cell_texts:
            rows.append(cell_texts)

    if not rows:
        return ""

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|") if cell else " "

    n_cols = max(len(r) for r in rows)
    md_lines: list[str] = []
    for i, row in enumerate(rows):
        padded = row + [""] * (n_cols - len(row))
        md_lines.append("| " + " | ".join(esc(c) for c in padded) + " |")
        if i == 0:
            md_lines.append("| " + " | ".join(["---"] * n_cols) + " |")

    return "\n" + "\n".join(md_lines) + "\n"


def _storage_html_visible_text(storage_html: str) -> str:
    """Strip tags from storage HTML, preserving top-level tables as Markdown."""
    if not storage_html:
        return ""

    soup = BeautifulSoup(storage_html, "html.parser")

    for table in soup.find_all("table"):
        # Nested tables are rendered inside the parent cell text, not separately.
        if table.find_parent("table") is not None:
            continue
        markdown_table = _table_to_markdown(table)
        table.replace_with(NavigableString(markdown_table))

    text = soup.get_text(" ", strip=True)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _storage_to_text(storage_html: str, title: str = "") -> str:
    """Convert Confluence storage format (XHTML) to plain text.

    Confluence link-only index pages store targets in XML attributes
    (ri:content-title, href) rather than visible text. Plain tag stripping
    yields an empty string and Open WebUI rejects the upload with 400.
    Top-level tables are preserved as Markdown tables.
    """
    parts: list[str] = []

    # Page / attachment link targets live in attributes, not element text.
    for pattern in (
        r'ri:content-title="([^"]*)"',
        r'ri:filename="([^"]*)"',
        r'ri:url="([^"]*)"',
    ):
        for match in re.finditer(pattern, storage_html):
            value = html.unescape(match.group(1)).strip()
            if value:
                parts.append(value)

    for match in re.finditer(r'href="([^"#][^"]*)"', storage_html):
        value = html.unescape(match.group(1)).strip()
        if value:
            parts.append(value)

    for match in re.finditer(
        r"<ac:plain-text-link-body>([^<]*)</ac:plain-text-link-body>",
        storage_html,
    ):
        value = html.unescape(match.group(1)).strip()
        if value:
            parts.append(value)

    text = _storage_html_visible_text(storage_html)
    if text:
        parts.append(text)

    # De-dupe while preserving order.
    seen: set[str] = set()
    lines: list[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            lines.append(part)

    if lines:
        return "\n".join(lines)

    return title.strip()


def _format_page_attachment_section(
    attachments: list[dict[str, Any]],
    *,
    allowed_extensions: frozenset[str],
) -> str:
    """Build a markdown list of attachment filenames for page export."""
    titles: list[str] = []
    seen: set[str] = set()
    for attachment in attachments:
        name = (attachment.get("title") or "").strip()
        if not name or name in seen:
            continue
        label_names = extract_attachment_label_names(attachment)
        ext = attachment_extension(name, label_names)
        if ext not in allowed_extensions:
            continue
        seen.add(name)
        titles.append(name)
    if not titles:
        return ""
    lines = "\n".join(f"- {name}" for name in sorted(titles, key=str.lower))
    return f"\n\n## Attachments\n\n{lines}"


def _finalize_page_text(
    text: str,
    *,
    title: str,
    page_id: str,
    space_key: str,
    base_url: str,
    path: str = "",
    filename: str = "",
) -> str:
    """Ensure non-empty, unique text for Open WebUI vector deduplication.

    OWUI hashes document chunks and rejects duplicates already in the KB.
    The unique id must be in the *first* lines so chunking differs for pages
    that share boilerplate body text.
    """
    body = text.strip() or title.strip() or f"Confluence page {page_id}"
    display = f"{path}/{filename}" if path else (filename or title)
    uid = f"confluence:{space_key}:{page_id}"
    if display:
        uid = f"{uid}:{display}"

    header = f"# {title}\n{uid}\n"
    source = uid
    if base_url:
        source = (
            f"{uid} {base_url.rstrip('/')}/pages/viewpage.action?pageId={page_id}"
        )
    return f"{header}\n{body}\n\n---\n{source}"


class ConfluenceConnector(BaseConnector):
    """Sync pages from a Confluence space.

    Args:
        space_key:   Confluence space key (e.g. "ENG").
        base_url:    Confluence instance URL (or CONFLUENCE_URL env var).
        user:        Confluence user email/username (or CONFLUENCE_USER env var).
        token:       API token or PAT (or CONFLUENCE_TOKEN env var).
        api_version: REST API version, "v1" or "v2" (or CONFLUENCE_API_VERSION env var).
        attachments: Attachment sync settings (from .oikb.yaml).
    """

    def __init__(
        self,
        space_key: str,
        base_url: str | None = None,
        user: str | None = None,
        token: str | None = None,
        api_version: str | None = None,
        attachments: AttachmentsConfig | None = None,
    ):
        self.space_key = space_key
        self._attachments = attachments or AttachmentsConfig()

        self._base_url = (base_url or os.environ.get("CONFLUENCE_URL", "")).rstrip("/")
        self._user = user or os.environ.get("CONFLUENCE_USER", "")
        self._token = token or os.environ.get("CONFLUENCE_TOKEN", "")
        self._api_version = _parse_api_version(api_version)

        headers = {"Accept": "application/json"}

        if not self._base_url:
            raise ValueError(
                "Confluence URL required. Set via:\n"
                "  export CONFLUENCE_URL=https://company.atlassian.net"
            )
        if not self._token:
            raise ValueError(
                "Confluence API token required. Set via:\n"
                "  export CONFLUENCE_TOKEN=<api_token>"
            )

        if not self._user:
            headers["Authorization"] = f"Bearer {self._token}"

        self._auth = (self._user, self._token) if self._user else None
        self._download_headers = dict(headers)

        self._http = httpx.Client(
            base_url=f"{self._base_url}{BASE_ENDPOINTS[self._api_version]}",
            auth=self._auth,
            headers=headers,
            timeout=60.0,
            follow_redirects=False,
        )

        # (path, filename) -> Confluence page id
        self._page_cache: dict[tuple[str, str], str] = {}
        self._attachment_cache: dict[tuple[str, str], _AttachmentRef] = {}
        self._attachments_v2_warned = False
        self._attachment_skip_count = 0

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with retries on transient failures; does not reset caller pagination."""
        attempt = 0
        while True:
            try:
                resp = self._http.get(path, params=params)
            except httpx.RequestError as exc:
                if attempt >= _HTTP_MAX_ATTEMPTS - 1:
                    raise
                delay = _HTTP_RETRY_BASE_DELAY_S * (2 ** attempt)
                log.warning(
                    "Confluence %s request error (attempt %d/%d): %s; "
                    "retrying in %.0fs (params=%s)",
                    path,
                    attempt + 1,
                    _HTTP_MAX_ATTEMPTS,
                    exc,
                    delay,
                    params,
                )
                time.sleep(delay)
                attempt += 1
                continue

            if (
                resp.status_code in _RETRYABLE_STATUS_CODES
                and attempt < _HTTP_MAX_ATTEMPTS - 1
            ):
                delay = _HTTP_RETRY_BASE_DELAY_S * (2 ** attempt)
                log.warning(
                    "Confluence %s returned %s (attempt %d/%d); "
                    "retrying in %.0fs (params=%s)",
                    path,
                    resp.status_code,
                    attempt + 1,
                    _HTTP_MAX_ATTEMPTS,
                    delay,
                    params,
                )
                time.sleep(delay)
                attempt += 1
                continue

            resp.raise_for_status()
            return resp

    def build_manifest(self) -> list[ManifestEntry]:
        """List all pages in the space and build a manifest."""
        self._attachment_skip_count = 0

        if self._attachments.enabled and self._api_version == "v2":
            if not self._attachments_v2_warned:
                log.warning(
                    "Confluence attachment sync requires API v1; "
                    "set CONFLUENCE_API_VERSION=v1. Attachments will be skipped."
                )
                self._attachments_v2_warned = True

        if self._api_version == "v2":
            entries = self._build_manifest_v2()
        else:
            entries = self._build_manifest_v1()

        if self._attachment_skip_count > 0:
            log.warning(
                "Skipped %d attachment(s) due to filtering "
                "(extension allowlist, max-size, or missing download link)",
                self._attachment_skip_count,
            )

        return entries

    def _build_manifest_v1(self) -> list[ManifestEntry]:
        """List pages via Confluence CQL search (API v1 ``/content/search``)."""
        entries: list[ManifestEntry] = []
        used_keys: set[tuple[str, str]] = set()
        start = 0
        limit = 250
        # ORDER BY keeps offset pagination stable across pages.
        space_lit = self.space_key.replace("\\", "\\\\").replace('"', '\\"')
        cql = f'space = "{space_lit}" AND type = page ORDER BY id'

        while True:
            prev_start = start
            params: dict[str, Any] = {
                "cql": cql,
                "limit": limit,
                "start": start,
                "expand": "ancestors,version",
            }

            # Retry stays on this `start` offset; pagination advances only on success.
            data = self._get("/content/search", params=params).json()

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                dir_path = _ancestor_dir_path_v1(page)
                self._add_page_entry(entries, page, dir_path, used_keys)

            start += len(results)
            if start <= prev_start:
                log.warning(
                    "Confluence v1 CQL pagination stalled at start=%s (got %d results); stopping",
                    prev_start,
                    len(results),
                )
                break

        entries.sort(key=lambda e: e.display_path)
        return entries

    def _build_manifest_v2(self) -> list[ManifestEntry]:
        all_pages: list[dict[str, Any]] = []
        cursor = None

        while True:
            params: dict[str, Any] = {"limit": 250}
            if cursor:
                params["cursor"] = cursor

            data = self._get(
                f"/spaces/{self.space_key}/pages",
                params=params,
            ).json()

            all_pages.extend(data.get("results", []))

            next_link = data.get("_links", {}).get("next")
            if not next_link:
                break
            cursor_match = re.search(r"cursor=([^&]+)", next_link)
            cursor = cursor_match.group(1) if cursor_match else None
            if not cursor:
                break

        pages_by_id = {str(page["id"]): page for page in all_pages}
        entries: list[ManifestEntry] = []
        used_keys: set[tuple[str, str]] = set()
        for page in all_pages:
            dir_path = _ancestor_dir_path_v2(str(page["id"]), pages_by_id)
            self._add_page_entry(entries, page, dir_path, used_keys)

        entries.sort(key=lambda e: e.display_path)
        return entries

    def _unique_path_filename(
        self,
        page: dict[str, Any],
        dir_path: str,
        used_keys: set[tuple[str, str]],
    ) -> tuple[str, str]:
        """Return unique (path, filename) using page id when titles collide."""
        page_id = str(page["id"])
        filename = _sanitize_path_segment(page.get("title", "Untitled")) + ".txt"
        key = (dir_path, filename)
        if key in used_keys:
            stem = filename.removesuffix(".txt")
            filename = f"{stem}_{page_id}.txt"
            key = (dir_path, filename)
        used_keys.add(key)
        return dir_path, filename

    def _space_prefixed_path(self, dir_path: str) -> str:
        """Prefix ancestor path with Confluence space key for multi-space KBs."""
        if dir_path:
            return f"{self.space_key}/{dir_path}"
        return self.space_key

    def _add_page_entry(
        self,
        entries: list[ManifestEntry],
        page: dict[str, Any],
        dir_path: str,
        used_keys: set[tuple[str, str]],
    ) -> None:
        page_id = str(page["id"])
        version = page.get("version", {}).get("number", 0)

        checksum = hashlib.sha256(
            f"{page_id}:v{version}".encode()
        ).hexdigest()[:16]

        dir_path = self._space_prefixed_path(dir_path)
        dir_path, filename = self._unique_path_filename(page, dir_path, used_keys)

        entries.append(
            ManifestEntry(
                filename=filename,
                path=dir_path,
                checksum=checksum,
                size=0,
            )
        )

        self._page_cache[(dir_path, filename)] = page_id

        if self._attachments.enabled and self._api_version == "v1":
            try:
                attachments = self._list_attachments_v1(page_id)
            except Exception as exc:
                log.warning(
                    "Failed to list attachments for page %s: %s",
                    page_id,
                    exc,
                )
                return

            page_title = page.get("title", "Untitled")
            for attachment in attachments:
                self._add_attachment_entry(
                    entries,
                    attachment,
                    page_dir_path=dir_path,
                    page_id=page_id,
                    page_title=page_title,
                    used_keys=used_keys,
                )

    def _list_attachments_v1(self, page_id: str) -> list[dict[str, Any]]:
        """List all attachments for a page (v1 API, paginated)."""
        all_attachments: list[dict[str, Any]] = []
        start = 0
        limit = 250

        while True:
            prev_start = start
            params: dict[str, Any] = {
                "expand": "version,metadata.labels",
                "limit": limit,
                "start": start,
            }

            data = self._get(
                f"/content/{page_id}/child/attachment",
                params=params,
            ).json()

            results = data.get("results", [])
            if not results:
                break

            all_attachments.extend(results)

            start += len(results)
            if start <= prev_start:
                log.warning(
                    "Confluence v1 attachment pagination stalled at start=%s "
                    "(page %s, got %d results); stopping",
                    prev_start,
                    page_id,
                    len(results),
                )
                break

        return all_attachments

    def _add_attachment_entry(
        self,
        entries: list[ManifestEntry],
        attachment: dict[str, Any],
        *,
        page_dir_path: str,
        page_id: str,
        page_title: str,
        used_keys: set[tuple[str, str]],
    ) -> None:
        original_title = attachment.get("title", "")
        if not original_title:
            return

        label_names = extract_attachment_label_names(attachment)
        ext = attachment_extension(original_title, label_names)
        if ext not in self._attachments.allowed_extensions:
            self._attachment_skip_count += 1
            return

        size = int(attachment.get("extensions", {}).get("fileSize", 0) or 0)
        if size > self._attachments.max_size:
            self._attachment_skip_count += 1
            return

        attachment_id = str(attachment["id"])
        version = attachment.get("version", {}).get("number", 0)
        checksum = hashlib.sha256(
            f"{attachment_id}:v{version}".encode()
        ).hexdigest()[:16]

        download_path = attachment.get("_links", {}).get("download", "")
        if not download_path:
            self._attachment_skip_count += 1
            return

        page_segment = (
            f"{_sanitize_path_segment(page_title)}_{page_id}"
        )
        attach_dir = f"{page_dir_path}/_attachments/{page_segment}"

        if is_drawio_attachment(original_title, label_names):
            filename = _sanitize_path_segment(
                drawio_manifest_filename(original_title)
            )
        else:
            filename = _sanitize_path_segment(original_title)
        key = (attach_dir, filename)
        if key in used_keys:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{attachment_id}{suffix}"
            key = (attach_dir, filename)
        used_keys.add(key)

        entries.append(
            ManifestEntry(
                filename=filename,
                path=attach_dir,
                checksum=checksum,
                size=size,
            )
        )

        self._attachment_cache[(attach_dir, filename)] = _AttachmentRef(
            attachment_id=attachment_id,
            download_path=download_path,
            page_id=page_id,
            filename=original_title,
            is_drawio=is_drawio_attachment(original_title, label_names),
        )

    def read_file(self, path: str, filename: str) -> bytes:
        """Fetch a page's content or attachment bytes."""
        attachment_ref = self._attachment_cache.get((path, filename))
        if attachment_ref is not None:
            return self._read_attachment(attachment_ref)

        page_id = self._page_cache.get((path, filename))
        if not page_id:
            raise FileNotFoundError(
                f"File not found: {path}/{filename}" if path else filename
            )

        return self._read_page(path, filename, page_id)

    def _read_page(self, path: str, filename: str, page_id: str) -> bytes:
        if self._api_version == "v2":
            data = self._get(
                f"/pages/{page_id}",
                params={"body-format": "storage"},
            ).json()
        else:
            data = self._get(
                f"/content/{page_id}",
                params={"expand": "body.storage,version"},
            ).json()

        storage = data.get("body", {}).get("storage", {}).get("value", "")
        title = data.get("title", "")
        text = _storage_to_text(storage, title=title)

        if self._api_version == "v1":
            try:
                attachments = self._list_attachments_v1(page_id)
            except Exception as exc:
                log.warning(
                    "Failed to list attachments for page %s: %s",
                    page_id,
                    exc,
                )
                attachments = []
            attachment_section = _format_page_attachment_section(
                attachments,
                allowed_extensions=_page_attachment_extensions_from_env(),
            )
            if attachment_section:
                text = text + attachment_section

        text = _finalize_page_text(
            text,
            title=title,
            page_id=page_id,
            space_key=self.space_key,
            base_url=self._base_url,
            path=path,
            filename=filename,
        )
        return text.encode("utf-8")

    def _read_attachment(self, ref: _AttachmentRef) -> bytes:
        download_url = ref.download_path
        if download_url.startswith("/"):
            download_url = f"{self._base_url}{download_url}"

        try:
            resp = httpx.get(
                download_url,
                auth=self._auth,
                headers=self._download_headers,
                timeout=60.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            content = resp.content
            if self._api_version == "v1" and ref.is_drawio:
                return drawio_to_text(content).encode("utf-8")
            return content
        except Exception as exc:
            log.warning(
                "Failed to download attachment %s (page %s): %s",
                ref.filename,
                ref.page_id,
                exc,
            )
            raise

    def close(self) -> None:
        self._http.close()


def parse_confluence_source(source: str) -> dict[str, str | None]:
    """Parse a confluence:SPACEKEY source string.

    Examples:
        confluence:ENG
        confluence:https://company.atlassian.net/ENG
    """
    source = source.removeprefix("confluence:")

    # Check if it includes a URL.
    if source.startswith("https://"):
        parts = source.rsplit("/", 1)
        if len(parts) == 2:
            return {"base_url": parts[0], "space_key": parts[1]}
        raise ValueError("Invalid Confluence source. Expected: confluence:SPACEKEY")

    return {"space_key": source, "base_url": None}
