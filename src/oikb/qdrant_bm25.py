"""Sparse BM25 indexing into Qdrant for Confluence documents.

Before a Confluence page is uploaded to Open WebUI, its text is lemmatized
(pymorphy3 — better for Russian than Snowball stemming) and upserted as a
sparse BM25 vector into a dedicated Qdrant collection (created on demand).

Enable by setting ``QDRANT_URL``. Optional: ``QDRANT_API_KEY``,
``QDRANT_BM25_COLLECTION`` (default ``oikb-bm25``).

Requires optional deps: ``pip install oikb[qdrant]``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oikb.connectors import BaseConnector

log = logging.getLogger("oikb.qdrant_bm25")

# Compact Russian stopword list (lemmas / invariant forms). Used because
# FastEmbed clears stopwords when disable_stemmer=True (we replace Snowball
# stemming with pymorphy3 lemmatization).
_RU_STOPWORDS = frozenset(
    """
    а без более бы был была были было быть в вам вас весь во вот все всё
    всего всех вы где да даже для до его ее её ей ему если есть еще ещё же
    за здесь и из или им их к как ко когда кто ли либо мне может мы на над
    надо ни них но ну о об однако он она они оно от по под при про раз с со
    себе себя сейчас так также такой там то того тоже той только том тот ты
    у уже хотя чего чей чем что чтобы эта эти это этот я
    a an and are as at be by for from has he in is it its of on or that the
    to was were will with
    """.split()
)

_WORD_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+", re.UNICODE)

VECTOR_NAME = "bm25"
DEFAULT_COLLECTION = "oikb-bm25"


def qdrant_enabled() -> bool:
    """Return True when Qdrant BM25 indexing is configured."""
    return bool(os.environ.get("QDRANT_URL", "").strip())


def is_confluence_source(connector: BaseConnector, path: str, filename: str) -> bool:
    """True if this file comes from a Confluence connector (incl. composite)."""
    from oikb.connectors.composite import CompositeConnector
    from oikb.connectors.confluence import ConfluenceConnector

    if isinstance(connector, ConfluenceConnector):
        return True
    if isinstance(connector, CompositeConnector):
        routed = connector.get_connector(path, filename)
        return isinstance(routed, ConfluenceConnector)
    return False


class RussianLemmatizer:
    """Lemmatize mixed RU/EN text with pymorphy3 (dictionary lemmas)."""

    def __init__(self) -> None:
        import pymorphy3

        self._morph = pymorphy3.MorphAnalyzer()

    def preprocess(self, text: str) -> str:
        """Tokenize → drop stopwords → lemmatize → space-joined string."""
        lemmas: list[str] = []
        for raw in _WORD_RE.findall(text.lower()):
            if raw in _RU_STOPWORDS:
                continue
            if raw.isascii() and raw.isalpha():
                # Latin tokens: keep as-is (lowercased); pymorphy is for Russian.
                lemmas.append(raw)
                continue
            parsed = self._morph.parse(raw)[0]
            lemma = parsed.normal_form
            if lemma and lemma not in _RU_STOPWORDS:
                lemmas.append(lemma)
        return " ".join(lemmas)


class QdrantBm25Indexer:
    """Ensure a sparse BM25 collection exists and upsert document vectors."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
            from fastembed import SparseTextEmbedding
        except ImportError as e:
            raise ImportError(
                "Qdrant BM25 indexing requires optional dependencies. "
                "Install with: pip install oikb[qdrant]"
            ) from e

        self._url = (url or os.environ.get("QDRANT_URL", "")).rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("QDRANT_API_KEY")
        self._collection = (
            collection
            or os.environ.get("QDRANT_BM25_COLLECTION", DEFAULT_COLLECTION).strip()
            or DEFAULT_COLLECTION
        )
        if not self._url:
            raise ValueError("QDRANT_URL is required for BM25 indexing")

        client_kwargs: dict[str, Any] = {"url": self._url}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        self._client = QdrantClient(**client_kwargs)

        # Lemmatize with pymorphy3, then BM25 without Snowball stemming.
        self._lemmatizer = RussianLemmatizer()
        self._embedder = SparseTextEmbedding(
            model_name="Qdrant/bm25",
            language="russian",
            disable_stemmer=True,
        )
        self._ensured = False

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """Create the sparse BM25 collection if it does not exist."""
        if self._ensured:
            return

        from qdrant_client import models

        names = {c.name for c in self._client.get_collections().collections}
        if self._collection not in names:
            log.info("Creating Qdrant sparse BM25 collection %r", self._collection)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={},
                sparse_vectors_config={
                    VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
        self._ensured = True

    def upsert_document(
        self,
        *,
        content: bytes,
        path: str,
        filename: str,
        file_hash: str,
        kb_id: str,
    ) -> None:
        """Lemmatize text and upsert a sparse BM25 point. No-op for binary."""
        text = _decode_text(content)
        if text is None:
            log.debug(
                "Skipping Qdrant BM25 for non-text file %s",
                f"{path}/{filename}" if path else filename,
            )
            return

        self.ensure_collection()

        from qdrant_client import models

        prepared = self._lemmatizer.preprocess(text)
        if not prepared.strip():
            log.debug(
                "Skipping Qdrant BM25 for empty lemmatized text %s",
                f"{path}/{filename}" if path else filename,
            )
            return

        embedding = next(self._embedder.embed([prepared]))
        point_id = _point_id(kb_id, path, filename)
        display = f"{path}/{filename}" if path else filename

        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        VECTOR_NAME: models.SparseVector(
                            indices=embedding.indices.tolist(),
                            values=embedding.values.tolist(),
                        )
                    },
                    payload={
                        "kb_id": kb_id,
                        "path": path,
                        "filename": filename,
                        "display_path": display,
                        "file_hash": file_hash,
                        "source": "confluence",
                        "text": text[:8000],
                    },
                )
            ],
        )
        log.debug("Upserted BM25 point for %s into %r", display, self._collection)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


_indexer: QdrantBm25Indexer | None = None


def get_indexer() -> QdrantBm25Indexer | None:
    """Return a shared indexer when ``QDRANT_URL`` is set, else None."""
    global _indexer
    if not qdrant_enabled():
        return None
    if _indexer is None:
        _indexer = QdrantBm25Indexer()
    return _indexer


def index_before_upload(
    connector: BaseConnector,
    *,
    content: bytes,
    path: str,
    filename: str,
    file_hash: str,
    kb_id: str,
) -> None:
    """Index a Confluence document into Qdrant BM25 before Open WebUI upload.

    No-op when Qdrant is not configured or the file is not from Confluence.
    """
    if not qdrant_enabled():
        return
    if not is_confluence_source(connector, path, filename):
        return

    indexer = get_indexer()
    if indexer is None:
        return
    indexer.upsert_document(
        content=content,
        path=path,
        filename=filename,
        file_hash=file_hash,
        kb_id=kb_id,
    )


def _decode_text(content: bytes) -> str | None:
    """Decode UTF-8/UTF-8-SIG text; return None for likely binary payloads."""
    if not content:
        return ""
    # NUL bytes → binary attachment
    if b"\x00" in content[:8192]:
        return None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _point_id(kb_id: str, path: str, filename: str) -> str:
    """Stable UUID for upserts (same Confluence page → same point)."""
    key = f"{kb_id}\0{path}\0{filename}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))
