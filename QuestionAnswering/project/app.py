"""
Entry point for the NiceGUI RAG application.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from nicegui import ui

try:
    from huggingface_hub import login as hf_login
except ImportError:  # pragma: no cover
    hf_login = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "local_config.json"


def _load_local_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {CONFIG_PATH}: {exc}") from exc


def _apply_env_from_config(config: Dict[str, Any]) -> None:
    hf_cfg = config.get("huggingface", {})
    if token := hf_cfg.get("token"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        if hf_login is not None:
            try:
                hf_login(token, add_to_git_credential=False)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Failed to authenticate with Hugging Face: {exc}") from exc
    if model_name := hf_cfg.get("model_name"):
        os.environ.setdefault("LLM_MODEL_NAME", model_name)

    llm_cfg = config.get("llm", {})
    if backend := llm_cfg.get("backend"):
        os.environ["LLM_BACKEND"] = backend
    if base_url := llm_cfg.get("vllm_base_url"):
        os.environ["VLLM_BASE_URL"] = base_url
    if "enable_4bit" in llm_cfg:
        os.environ["LLM_ENABLE_4BIT"] = "1" if llm_cfg.get("enable_4bit") else "0"
    if fallback_model := llm_cfg.get("fallback_model_name"):
        os.environ["LLM_FALLBACK_MODEL_NAME"] = fallback_model
    if max_tokens := llm_cfg.get("max_new_tokens"):
        os.environ["LLM_MAX_NEW_TOKENS"] = str(max_tokens)
    if temperature := llm_cfg.get("temperature"):
        os.environ["LLM_TEMPERATURE"] = str(temperature)

    embed_cfg = config.get("embeddings", {})
    if embed_model := embed_cfg.get("model_name"):
        os.environ["BGE_MODEL_NAME"] = embed_model
    if batch_size := embed_cfg.get("batch_size"):
        os.environ["BGE_EMBED_BATCH_SIZE"] = str(batch_size)


def _configure_cache_dirs(config: Dict[str, Any]) -> None:
    cache_cfg = config.get("cache_paths", {})

    def _prepare_path(key: str, default: Path) -> Path:
        override = cache_cfg.get(key)
        if override:
            path = Path(override)
            if not path.is_absolute():
                path = PROJECT_ROOT / override
        else:
            path = default
        path.mkdir(parents=True, exist_ok=True)
        return path

    base_cache = _prepare_path("xdg_cache_home", PROJECT_ROOT / "project" / "data" / ".cache")
    hf_cache = _prepare_path("hf_home", base_cache / "huggingface")
    paddle_cache = _prepare_path("paddle_home", base_cache / "paddle")
    paddle_dataset = _prepare_path("paddle_data_home", paddle_cache / "dataset")
    transformers_cache = _prepare_path("transformers_cache", hf_cache)

    os.environ["XDG_CACHE_HOME"] = str(base_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["PADDLE_HOME"] = str(paddle_cache)
    os.environ["PADDLE_DATA_HOME"] = str(paddle_dataset)


LOCAL_CONFIG = _load_local_config()
_apply_env_from_config(LOCAL_CONFIG)
_configure_cache_dirs(LOCAL_CONFIG)

from .backend import vectordb
from .ui import main_ui


def _init_services() -> None:
    data_dir = Path(__file__).resolve().parent / "data" / "vectordb"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectordb.init_vector_db(str(data_dir))
    main_ui.register_routes()


def main() -> None:
    _init_services()
    ui.run(
        host="0.0.0.0",
        port=8080,
        show=False,
        reload=False,
        uvicorn_logging_level="info",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
