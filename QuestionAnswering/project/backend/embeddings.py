"""
Embedding utilities for multilingual retrieval based on BGE-M3 encoder.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Sequence

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for embeddings. Install with `pip install torch`.") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers is required for BGE-M3 embeddings. "
        "Install with `pip install sentence-transformers`."
    ) from exc


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
EMBED_BATCH_SIZE = int(os.getenv("BGE_EMBED_BATCH_SIZE", "8"))
TARGET_DTYPE = np.float32


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    device = _resolve_device()
    LOGGER.info("Loading embedding model %s on %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = getattr(model, "max_seq_length", 512)
    return model


def _to_float_list(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    try:
        is_empty = len(vectors) == 0  # handles python lists and numpy arrays
    except TypeError:
        is_empty = False
    if is_empty:
        return []
    return np.asarray(vectors, dtype=TARGET_DTYPE).tolist()


def embed_text_list(texts: List[str]) -> List[List[float]]:
    """
    Encode a list of texts into dense vectors suitable for vector DB insertion.
    """
    if not texts:
        return []

    model = _load_model()
    LOGGER.debug("Encoding %s texts with batch size %s", len(texts), EMBED_BATCH_SIZE)
    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            convert_to_tensor=False,
            device=_resolve_device(),
        )
    except Exception as exc:
        LOGGER.exception("Embedding generation failed: %s", exc)
        raise

    return _to_float_list(embeddings)


def embed_query(text: str) -> List[float]:
    """
    Encode a single query string, returning a single embedding vector.
    """
    vectors = embed_text_list([text])
    return vectors[0] if vectors else []
