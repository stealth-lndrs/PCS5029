import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyMuPDF (fitz) is required for PDF ingestion. "
        "Install with `pip install pymupdf`."
    ) from exc

from . import ocr

try:  # pragma: no cover - chunker not yet implemented
    from .chunker import chunk_text
except Exception:  # broad to keep ingestion operational
    def chunk_text(_: str, __: int, ___: str) -> List[Any]:
        logging.getLogger(__name__).warning(
            "chunk_text is not implemented; ingestion will not produce chunks."
        )
        return []

try:  # embeddings/vectordb may not exist yet
    from . import embeddings as embeddings_module
except Exception:  # pragma: no cover
    embeddings_module = None

try:
    from . import vectordb as vectordb_module
except Exception:  # pragma: no cover
    vectordb_module = None


LOGGER = logging.getLogger(__name__)
CHUNK_STORAGE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "chunks"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = WHITESPACE_PATTERN.sub(" ", text)
    return normalized.strip()


def _store_page_text(doc_id: str, page_number: int, text: str) -> None:
    CHUNK_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{doc_id}_page_{page_number:05d}.txt"
    path = CHUNK_STORAGE_DIR / file_name
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Failed to persist text for doc=%s page=%s: %s", doc_id, page_number, exc)


def _get_embedding_fn() -> Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]]:
    if embeddings_module is None:
        return None
    for attr in ("embed_documents", "embed_texts", "encode"):
        fn = getattr(embeddings_module, attr, None)
        if callable(fn):
            return fn
    return None


def _get_upsert_fn() -> Optional[Callable[[Iterable[dict], Sequence[Sequence[float]]], None]]:
    if vectordb_module is None:
        return None
    for attr in ("upsert", "add_documents", "add"):
        fn = getattr(vectordb_module, attr, None)
        if callable(fn):
            return fn
    store = getattr(vectordb_module, "get_vector_store", None)
    if callable(store):
        client = store()
        for attr in ("upsert", "add_documents", "add"):
            fn = getattr(client, attr, None)
            if callable(fn):
                return fn
    return None


def _process_page_text(
    doc_id: str,
    filename: str,
    page_number: int,
    text: str,
) -> List[dict]:
    base_metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "page_number": page_number,
    }
    page_chunks: List[dict] = []
    for chunk in chunk_text(doc_id, page_number, text):
        item = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": {**chunk.metadata, **base_metadata},
        }
        page_chunks.append(item)
    return page_chunks


def _push_to_vector_store(
    chunks: List[dict],
    embedding_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]],
    upsert_fn: Optional[Callable[[Iterable[dict], Sequence[Sequence[float]]], None]],
) -> None:
    if not chunks:
        return

    if embedding_fn is None or upsert_fn is None:
        LOGGER.warning("Embedding or vector DB backend not configured; skipping push.")
        return

    texts = [chunk.get("text", "") for chunk in chunks]
    try:
        vectors = embedding_fn(texts)
    except Exception as exc:
        LOGGER.exception("Failed to embed %s chunks: %s", len(chunks), exc)
        return

    payload = [chunk.get("metadata", {}) for chunk in chunks]

    try:
        upsert_fn(payload, vectors)
    except Exception as exc:
        LOGGER.exception("Failed to upsert %s chunks to vector DB: %s", len(chunks), exc)


def ingest_pdf(
    pdf_path: str,
    doc_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """
    Stream and ingest a PDF file, returning chunk metadata for downstream tracking.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    all_chunks: List[dict] = []
    embedding_fn = _get_embedding_fn()
    upsert_fn = _get_upsert_fn()

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        LOGGER.exception("Unable to open PDF %s: %s", pdf_path, exc)
        raise

    filename = Path(pdf_path).name
    page_count = doc.page_count
    LOGGER.info("Ingesting %s pages from %s (%s)", page_count, filename, doc_id)

    try:
        for page_number in range(page_count):
            try:
                page = doc.load_page(page_number)
            except Exception as exc:
                LOGGER.exception("Failed to load page %s from %s: %s", page_number, pdf_path, exc)
                continue

            text = page.get_text("text") or ""
            text = _normalize_text(text)

            if ocr.is_page_scanned(text):
                LOGGER.debug("Page %s flagged as scanned; calling OCR.", page_number)
                text = ocr.extract_page_text(pdf_path, page_number)
                text = _normalize_text(text)

            _store_page_text(doc_id, page_number, text)

            page_chunks = _process_page_text(doc_id, filename, page_number, text)
            _push_to_vector_store(page_chunks, embedding_fn, upsert_fn)
            all_chunks.extend(page_chunks)
            if progress_callback is not None:
                try:
                    progress_callback(page_number + 1, page_count)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Progress callback failed: %s", exc)
    finally:
        doc.close()

    return all_chunks
