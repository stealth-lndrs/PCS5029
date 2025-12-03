"""
NiceGUI interface for document management: upload, ingest, progress, deletion.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

import fitz  # PyMuPDF
from nicegui import events, ui

from ..backend import pdf_ingestion, vectordb

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
PDF_STORAGE = DATA_ROOT / "pdfs"
METADATA_FILE = DATA_ROOT / "documents_meta.json"


def _ensure_dirs() -> None:
    PDF_STORAGE.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _load_documents() -> List[Dict[str, Any]]:
    if not METADATA_FILE.exists():
        return []
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_documents(docs: List[Dict[str, Any]]) -> None:
    METADATA_FILE.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_pages(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


async def _ingest_document(
    doc: Dict[str, Any],
    status_label: ui.label,
    progress_label: ui.label,
    refresh_callback: Callable[[], None],
) -> None:
    progress_label.text = f"Página 0/{doc['pages']}"
    status_label.text = "Processando"

    def _progress(page: int, total: int) -> None:
        ui.run_later(lambda: setattr(progress_label, "text", f"Página {page}/{total}"))

    def _task() -> None:
        pdf_ingestion.ingest_pdf(str(PDF_STORAGE / doc["storage_name"]), doc["doc_id"], progress_callback=_progress)

    try:
        await asyncio.to_thread(_task)
    except Exception as exc:
        ui.notify(f"Falha ao ingerir documento: {exc}", type="negative")
        status_label.text = "Erro"
        return

    status_label.text = "Ingerido"
    progress_label.text = f"Página {doc['pages']}/{doc['pages']}"
    docs_list = _load_documents()
    for entry in docs_list:
        if entry["doc_id"] == doc["doc_id"]:
            entry["status"] = "Ingerido"
            entry["progress_text"] = progress_label.text
            break
    _save_documents(docs_list)
    refresh_callback()


def _render_documents(
    container: ui.column,
    docs: List[Dict[str, Any]],
    refresh_callback: Callable[[], None],
) -> None:
    container.clear()
    if not docs:
        ui.label("Nenhum documento carregado ainda.").classes("text-gray-500").parent(container)
        return

    for doc in docs:
        with container:
            with ui.card().classes("w-full"):
                ui.label(doc["filename"]).classes("text-lg font-semibold")
                ui.label(f"Documento ID: {doc['doc_id']}").classes("text-sm text-gray-600")
                ui.label(f"Páginas detectadas: {doc['pages']}").classes("text-sm")

                status_label = ui.label(doc.get("status", "Não ingerido"))
                progress_label = ui.label(doc.get("progress_text", "-")).classes("text-sm text-gray-500")

                with ui.row():
                    async def start_ingestion(doc=doc, status_label=status_label, progress_label=progress_label):
                        docs_list = _load_documents()
                        for entry in docs_list:
                            if entry["doc_id"] == doc["doc_id"]:
                                entry["status"] = "Processando"
                                entry["progress_text"] = f"Página 0/{entry['pages']}"
                                break
                        _save_documents(docs_list)
                        refresh_callback()

                        await _ingest_document(
                            doc,
                            status_label,
                            progress_label,
                            refresh_callback,
                        )

                    ui.button("Ingerir", on_click=start_ingestion).props("outline")

                    async def delete_doc(doc=doc):
                        try:
                            vectordb.delete_doc(doc["doc_id"])
                        except Exception:
                            pass
                        pdf_path = PDF_STORAGE / doc["storage_name"]
                        if pdf_path.exists():
                            pdf_path.unlink()
                        docs_list = [entry for entry in _load_documents() if entry["doc_id"] != doc["doc_id"]]
                        _save_documents(docs_list)
                        refresh_callback()

                    ui.button("Excluir", on_click=delete_doc).props("outline color=negative")


def documents_page() -> None:
    _ensure_dirs()
    ui.page_title("Documentos")
    docs_container = ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4")

    with docs_container:
        ui.label("Gerenciamento de Documentos").classes("text-2xl font-semibold")

        async def handle_upload(event: events.UploadEvent) -> None:
            if not event.content:
                return
            data = event.content.read()
            doc_id = uuid.uuid4().hex[:8]
            storage_name = f"{doc_id}_{event.name}"
            pdf_path = PDF_STORAGE / storage_name
            pdf_path.write_bytes(data)
            pages = _count_pages(pdf_path)
            entry = {
                "doc_id": doc_id,
                "filename": event.name,
                "storage_name": storage_name,
                "pages": pages,
                "status": "Não ingerido",
                "progress_text": "-",
            }
            docs_list = _load_documents()
            docs_list.append(entry)
            _save_documents(docs_list)
            refresh_docs()
            ui.notify(f"Documento {event.name} carregado com sucesso.")

        ui.upload(label="Enviar PDF", on_upload=handle_upload).props("accept=.pdf")

        docs_list_column = ui.column().classes("w-full gap-4")

    def refresh_docs() -> None:
        docs_list_column.clear()
        with docs_list_column:
            _render_documents(docs_list_column, _load_documents(), refresh_docs)

    refresh_docs()
