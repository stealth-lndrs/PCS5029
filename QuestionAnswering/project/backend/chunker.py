"""
Token-based chunking utilities for PDF ingestion.

Uses Hugging Face's `tokenizers` library to produce whitespace-aware tokens and
build fixed-length overlapping chunks ready for downstream embedding.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from tokenizers import pre_tokenizers

TOKENIZER = pre_tokenizers.Whitespace()
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP = 100


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    start_token: int
    end_token: int
    text: str

    @property
    def metadata(self) -> Dict[str, int | str]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_number": self.page,
            "start_token": self.start_token,
            "end_token": self.end_token,
        }


ChunkMetadata = Chunk
__all__ = ["Chunk", "ChunkMetadata", "chunk_text"]


def _tokenize(text: str) -> Sequence[Tuple[str, Tuple[int, int]]]:
    if not text:
        return []
    return TOKENIZER.pre_tokenize_str(text)


def _slice_text(
    text: str,
    offsets: Sequence[Tuple[int, int]],
    start: int,
    end: int,
) -> str:
    if not offsets or start >= end:
        return ""
    start_idx = offsets[start][0]
    end_idx = offsets[end - 1][1]
    return text[start_idx:end_idx]


def chunk_text(doc_id: str, page_num: int, text: str) -> List[Chunk]:
    """
    Split text into overlapping chunks using token counts rather than characters.
    """
    text = text or ""
    tokens_with_offsets = _tokenize(text)
    total_tokens = len(tokens_with_offsets)
    if total_tokens == 0:
        return []

    offsets = [offset for _, offset in tokens_with_offsets]
    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    while start < total_tokens:
        end = min(start + CHUNK_SIZE_TOKENS, total_tokens)
        if start >= end:
            break

        chunk_body = _slice_text(text, offsets, start, end).strip()
        if not chunk_body:
            break

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{page_num}-{chunk_index}",
                doc_id=doc_id,
                page=page_num,
                start_token=start,
                end_token=end,
                text=chunk_body,
            )
        )
        chunk_index += 1

        if end >= total_tokens:
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks
