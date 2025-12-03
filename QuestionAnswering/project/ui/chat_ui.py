"""
NiceGUI chat interface that streams responses from the local RAG pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, AsyncGenerator, Dict, Iterable, Optional

from nicegui import ui

from ..backend import llm, rag

LOGGER = logging.getLogger(__name__)


def _format_citation(chunk: Dict) -> str:
    metadata = chunk.get("metadata") or {}
    doc_id = metadata.get("doc_id", "desconhecido")
    page = metadata.get("page_number") or metadata.get("page") or metadata.get("page_num")
    if page is None:
        page = "?"
    return f"Documento {doc_id}, página {page}"


def _render_citations(container: Any, chunks: Iterable[Dict]) -> None:
    container.clear()
    with container:
        for chunk in chunks:
            ui.label(_format_citation(chunk)).classes("text-xs text-gray-500")


async def _stream_llm(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[Dict[str, str]]] = asyncio.Queue()

    def _worker() -> None:
        try:
            stream = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature, stream=True)
            if isinstance(stream, str):
                asyncio.run_coroutine_threadsafe(queue.put({"token": stream}), loop)
            else:
                for token in stream:
                    asyncio.run_coroutine_threadsafe(queue.put({"token": token}), loop)
        except Exception as exc:  # pragma: no cover - UI surface handles errors
            LOGGER.exception("LLM streaming failed: %s", exc)
            asyncio.run_coroutine_threadsafe(queue.put({"error": str(exc)}), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        if "error" in item:
            raise RuntimeError(item["error"])
        yield item["token"]


async def _handle_query(input_box: ui.input, messages_column: ui.column) -> None:
    query = (input_box.value or "").strip()
    if not query:
        return

    with messages_column:
        with ui.chat_message("Usuário", avatar="👤"):
            ui.markdown(query)
    messages_column.update()
    input_box.value = ""

    retrieved_chunks = rag.retrieve_context(query)
    prompt = rag.build_prompt(query, retrieved_chunks)

    with messages_column:
        with ui.chat_message("Assistente", avatar="🤖"):
            assistant_markdown = ui.markdown("...")
            citations_container = ui.column().classes("gap-1 mt-2")

    response_text = ""
    try:
        async for token in _stream_llm(prompt, max_tokens=512, temperature=0.2):
            response_text += token
            assistant_markdown.content = response_text
            await asyncio.sleep(0)
    except Exception as exc:
        assistant_markdown.content = f"Erro ao gerar resposta: {exc}"
    else:
        _render_citations(citations_container, retrieved_chunks)
    finally:
        messages_column.update()


def chat_page() -> None:
    ui.page_title("Chat RAG")
    with ui.column().classes("w-full max-w-3xl mx-auto p-4 gap-4"):
        ui.label("Assistente RAG Multilíngue").classes("text-2xl font-semibold")
        messages_column = ui.column().classes("w-full gap-2 h-[60vh] overflow-y-auto border rounded-lg p-4")
        input_box = ui.input(placeholder="Digite sua pergunta em português...").props("type=textarea autogrow").classes(
            "w-full"
        )

        async def send_query() -> None:
            await _handle_query(input_box, messages_column)

        ui.button("Enviar", on_click=send_query).classes("self-end")
