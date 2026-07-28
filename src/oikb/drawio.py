"""Convert Confluence / draw.io (.drawio) attachment bytes to decoded text for .txt export."""

from __future__ import annotations

import base64
import re
import zlib
from pathlib import Path
from typing import Any
from urllib.parse import unquote

_PERCENT_BYTE = re.compile(r"%[0-9A-Fa-f]{2}")
_DIAGRAM_BLOCK = re.compile(
    r"(<diagram\b[^>]*>)(.*?)(</diagram>)",
    re.DOTALL | re.IGNORECASE,
)
_BASE64_BODY = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def is_drawio_filename(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(".drawio") or lower.endswith(".drawio.xml")


def extract_attachment_label_names(attachment: dict[str, Any]) -> frozenset[str]:
    """Label names from Confluence attachment ``metadata.labels`` (lowercase)."""
    labels_meta = attachment.get("metadata", {}).get("labels", {})
    results = labels_meta.get("results", [])
    names: set[str] = set()
    if not isinstance(results, list):
        return frozenset()
    for item in results:
        if not isinstance(item, dict):
            continue
        for key in ("name", "label"):
            value = item.get(key)
            if value:
                names.add(str(value).lower())
    return frozenset(names)


def has_drawio_label(labels: frozenset[str]) -> bool:
    return "drawio" in labels


def is_drawio_attachment(
    filename: str,
    labels: frozenset[str] | None = None,
) -> bool:
    if is_drawio_filename(filename):
        return True
    if labels and has_drawio_label(labels):
        return True
    return False


def attachment_extension(
    filename: str,
    labels: frozenset[str] | None = None,
) -> str:
    """File extension for allowlist checks (drawio covers ``.drawio.xml`` and label)."""
    if is_drawio_attachment(filename, labels):
        return "drawio"
    return Path(filename).suffix.lstrip(".").lower()


def drawio_manifest_filename(original_filename: str) -> str:
    """KB filename for a draw.io attachment (always ``.txt``)."""
    name = Path(original_filename).name
    lower = name.lower()
    if lower.endswith(".drawio.xml"):
        stem = name[: -len(".drawio.xml")]
    elif lower.endswith(".drawio"):
        stem = name[: -len(".drawio")]
    else:
        stem = Path(name).stem
    stem = re.sub(r'[<>:"/\\|?*]', "_", stem).strip() or "diagram"
    return f"{stem}.txt"


def drawio_to_text(data: bytes) -> str:
    """Decode draw.io file bytes to plain text (base64 + deflate + URL encoding)."""
    text = data.decode("utf-8-sig", errors="replace")
    text = _expand_diagram_payloads(text)
    return _percent_decode(text)


def _expand_diagram_payloads(mxfile_xml: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        open_tag, body, close_tag = match.group(1), match.group(2), match.group(3)
        decoded = _decode_diagram_payload(body)
        if decoded == body:
            return match.group(0)
        return f"{open_tag}\n{decoded.strip()}\n{close_tag}"

    return _DIAGRAM_BLOCK.sub(_replace, mxfile_xml)


def _decode_diagram_payload(body: str) -> str:
    stripped = body.strip()
    if not stripped:
        return body
    if stripped.lstrip().startswith("<"):
        return _percent_decode(stripped)

    payload = stripped
    if _looks_base64(stripped):
        raw = _try_base64_decode_to_bytes(stripped)
        if raw is not None:
            try:
                payload = raw.decode("utf-8")
            except UnicodeDecodeError:
                decompressed = _try_decompress_drawio(raw)
                if decompressed is not None:
                    payload = decompressed

    return _percent_decode(payload)


def _looks_base64(payload: str) -> bool:
    cleaned = re.sub(r"\s+", "", payload)
    if len(cleaned) < 4 or len(cleaned) % 4 != 0:
        return False
    return bool(_BASE64_BODY.match(payload))


def _try_base64_decode_to_bytes(payload: str) -> bytes | None:
    try:
        cleaned = re.sub(r"\s+", "", payload)
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        return None


def _try_decompress_drawio(raw: bytes) -> str | None:
    """Inflate draw.io diagram payload (raw deflate or zlib wrapper)."""
    try:
        return zlib.decompress(raw, -zlib.MAX_WBITS).decode("utf-8")
    except Exception:
        pass
    try:
        return zlib.decompress(raw).decode("utf-8")
    except Exception:
        return None


def _try_base64_decode_to_str(payload: str) -> str | None:
    raw = _try_base64_decode_to_bytes(payload)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return _try_decompress_drawio(raw)


def _percent_decode(text: str) -> str:
    """Apply ``urllib.parse.unquote`` until stable (Atlas / Gliffy diagram bodies)."""
    decoded = text
    for _ in range(8):
        if not _PERCENT_BYTE.search(decoded):
            break
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded
