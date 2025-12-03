"""
Entry point for the NiceGUI RAG application.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from .backend import vectordb
from .ui import main_ui


def _init_services() -> None:
    data_dir = Path(__file__).resolve().parent / "data" / "vectordb"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectordb.init_vector_db(str(data_dir))
    main_ui.register_routes()


def main() -> None:
    _init_services()
    ui.run()


if __name__ == "__main__":
    main()
