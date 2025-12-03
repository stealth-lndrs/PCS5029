"""
Vector database wrapper built on top of Chroma for storing/retrieving chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Chroma is required for the vector DB wrapper. Install with `pip install chromadb`."
    ) from exc


LOGGER = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Any] = None


def init_vector_db(path: str, collection_name: str = "qa_chunks") -> None:
    """
    Initialize the persistent Chroma client/collection at the specified path.
    """
    global _client, _collection

    storage_path = Path(path)
    storage_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Initializing Chroma at %s (collection=%s)", storage_path, collection_name)
    _client = chromadb.PersistentClient(
        path=str(storage_path),
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(collection_name)


def _ensure_collection() -> Any:
    if _collection is None:
        raise RuntimeError("Vector DB not initialized. Call init_vector_db first.")
    return _collection


def _extract_payload(
    chunks: Sequence[Any],
) -> Dict[str, List[Any]]:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)
        if chunk_id is None and isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
        if chunk_id is None:
            raise ValueError("Each chunk must have a chunk_id")
        text = getattr(chunk, "text", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("text")
        text = text or ""
        metadata = getattr(chunk, "metadata", None)
        if metadata is None and isinstance(chunk, dict):
            metadata = chunk.get("metadata", {})
        metadata = dict(metadata or {})
        metadata.setdefault("chunk_id", chunk_id)

        ids.append(str(chunk_id))
        documents.append(text)
        metadatas.append(metadata)

    return {"ids": ids, "documents": documents, "metadatas": metadatas}


def add_chunks(chunks: Sequence[Any], embeddings: Sequence[Sequence[float]]) -> None:
    """
    Insert or update chunk records with associated embeddings.
    """
    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")

    collection = _ensure_collection()
    payload = _extract_payload(chunks)

    try:
        collection.upsert(
            ids=payload["ids"],
            documents=payload["documents"],
            metadatas=payload["metadatas"],
            embeddings=list(embeddings),
        )
    except Exception as exc:
        LOGGER.exception("Failed to upsert %s chunks: %s", len(chunks), exc)
        raise


def search(
    query_embedding: Sequence[float],
    k: int = 8,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query the vector DB and return matching chunks with metadata preserved.
    """
    if not query_embedding:
        return []

    collection = _ensure_collection()
    where = filters if filters else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:
        LOGGER.exception("Vector search failed: %s", exc)
        raise

    matches: List[Dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for idx, chunk_id in enumerate(ids):
        matches.append(
            {
                "chunk_id": chunk_id,
                "text": docs[idx] if idx < len(docs) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "distance": distances[idx] if idx < len(distances) else None,
            }
        )
    return matches


def delete_doc(doc_id: str) -> None:
    """
    Remove all vectors associated with a given document ID.
    """
    if not doc_id:
        return
    collection = _ensure_collection()
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception as exc:
        LOGGER.exception("Failed to delete doc_id=%s: %s", doc_id, exc)
        raise


def list_docs() -> List[str]:
    """
    Return the list of unique document IDs currently stored.
    """
    collection = _ensure_collection()
    try:
        records = collection.get(include=["metadatas"])
    except Exception as exc:
        LOGGER.exception("Failed to list documents: %s", exc)
        raise

    doc_ids = {
        metadata.get("doc_id")
        for metadata in records.get("metadatas", [])
        if isinstance(metadata, dict) and metadata.get("doc_id")
    }
    return sorted(doc_ids)
