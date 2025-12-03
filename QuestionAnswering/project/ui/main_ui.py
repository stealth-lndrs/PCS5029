"""
Application layout, routing, and shared UI state for the NiceGUI front-end.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

from nicegui import ui

from . import chat_ui, documents_ui


@dataclass
class AppState:
    user_id: str
    dark_mode: bool = True


STATE = AppState(user_id=uuid.uuid4().hex[:8])


def _apply_theme() -> None:
    if STATE.dark_mode:
        ui.dark_mode()
    else:
        ui.light_mode()


def _toggle_theme() -> None:
    STATE.dark_mode = not STATE.dark_mode
    _apply_theme()


def _sidebar(active_page: str) -> None:
    with ui.left_drawer(value=True, fixed=True).classes("bg-gray-900 text-white p-4 w-56"):
        ui.label("Menu").classes("text-lg font-bold")
        ui.link("Chat", "/").classes(
            "mt-4 block px-3 py-2 rounded hover:bg-gray-800"
            + (" bg-gray-800" if active_page == "chat" else "")
        )
        ui.link("Documentos", "/documents").classes(
            "mt-2 block px-3 py-2 rounded hover:bg-gray-800"
            + (" bg-gray-800" if active_page == "documents" else "")
        )


def _header() -> None:
    with ui.header(elevated=True).classes("items-center justify-between px-6"):
        ui.label("RAG QA Studio").classes("text-xl font-semibold")
        with ui.row().classes("items-center gap-4"):
            ui.label(f"Sessão: {STATE.user_id}").classes("text-sm text-gray-500")
            ui.button(
                "Alternar tema",
                on_click=_toggle_theme,
            ).props("outline dense")


def _layout(active_page: str, content: Callable[[], None]) -> None:
    _apply_theme()
    _sidebar(active_page)
    _header()
    with ui.column().classes("ml-60 mt-4 p-4"):
        content()


@ui.page("/")
async def _chat_page() -> None:
    _layout("chat", chat_ui.chat_page)


@ui.page("/documents")
async def _documents_page() -> None:
    _layout("documents", documents_ui.documents_page)


def register_routes() -> None:
    # routes are registered via decorators when the module is imported.
    # function kept for backward compatibility with existing imports.
    return None
