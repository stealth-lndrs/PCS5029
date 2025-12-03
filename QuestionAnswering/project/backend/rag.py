"""
Retrieval-Augmented Generation pipeline orchestrating embeddings, vector search,
prompt construction, and local LLM generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from . import embeddings, vectordb

try:
    from . import llm
except Exception as exc:  # pragma: no cover
    raise RuntimeError("LLM module is required for answering questions.") from exc


LOGGER = logging.getLogger(__name__)


def _format_chunk(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {}) or {}
    doc_id = metadata.get("doc_id", "unknown")
    page = metadata.get("page_number") or metadata.get("page")
    if page is None:
        page = metadata.get("page_num", "N/A")
    header = f"[Doc {doc_id}, page {page}]"
    body = chunk.get("text", "").strip()
    return f"{header}\n{body}\n---"


def build_prompt(query: str, retrieved_chunks: Iterable[Dict[str, Any]]) -> str:
    """
    Build the final prompt text combining system instruction, question, and context.
    """
    prompt_lines = [
        "You are an assistant that answers questions in Portuguese using the provided context.",
        "Question:",
        query.strip(),
        "Context:",
    ]
    for chunk in retrieved_chunks:
        prompt_lines.append(_format_chunk(chunk))
    return "\n".join(prompt_lines).strip()


def retrieve_context(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve relevant chunks from the vector store.
    """
    query_embedding = embeddings.embed_query(query)
    matches = vectordb.search(query_embedding, k=k)
    LOGGER.debug("Retrieved %s chunks for query.", len(matches))
    return matches


def answer_question(query: str) -> str:
    """
    Execute the full RAG pipeline: retrieve context, build prompt, and invoke the LLM.
    """
    retrieved_chunks = retrieve_context(query)
    prompt = build_prompt(query, retrieved_chunks)
    LOGGER.debug("Generated prompt with %s context chunks.", len(retrieved_chunks))
    response = llm.generate(prompt)
    return response.strip() if isinstance(response, str) else response
