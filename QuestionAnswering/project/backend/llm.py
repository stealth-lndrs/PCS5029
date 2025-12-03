"""
Local LLM interface that supports Gemma 9B via transformers (4-bit) or vLLM.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Generator, Iterable, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-2-9b-it")
BACKEND = os.getenv("LLM_BACKEND", "transformers").lower()
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


def _ensure_transformers_model() -> None:
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    LOGGER.info("Loading Gemma model %s in 4-bit mode via transformers.", MODEL_NAME)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    _model.eval()


def _ensure_requests() -> None:
    if requests is None:  # type: ignore
        raise RuntimeError(
            "The 'requests' package is required for the vLLM backend. Install with `pip install requests`."
        )


def _apply_portuguese_instruction(prompt: str) -> str:
    instruction = (
        "Você é um assistente útil e deve sempre responder em português brasileiro."
    )
    return f"{instruction}\n\n{prompt.strip()}"


def _transformers_stream(
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    _ensure_transformers_model()
    assert _tokenizer is not None and _model is not None

    prompt_ids = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    generation_kwargs = dict(
        **prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=_tokenizer.pad_token_id,
        eos_token_id=_tokenizer.eos_token_id,
    )

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    thread = threading.Thread(
        target=_model.generate, kwargs={**generation_kwargs, "streamer": streamer}
    )
    thread.daemon = True
    thread.start()
    for token in streamer:
        yield token
    thread.join()


def _transformers_generate(
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    _ensure_transformers_model()
    assert _tokenizer is not None and _model is not None

    prompt_ids = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    generation_kwargs = dict(
        **prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=_tokenizer.pad_token_id,
        eos_token_id=_tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        output = _model.generate(**generation_kwargs)
    text = _tokenizer.decode(output[0], skip_special_tokens=True)
    return text


def _vllm_stream(
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Iterable[str]:
    _ensure_requests()
    endpoint = f"{VLLM_BASE_URL.rstrip('/')}/generate"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    response = requests.post(
        endpoint,
        json=payload,
        timeout=600,
        stream=True,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        token = data.get("token") or data.get("text")
        if token:
            yield token


def _vllm_request(
    prompt: str,
    max_tokens: int,
    temperature: float,
):
    _ensure_requests()
    endpoint = f"{VLLM_BASE_URL.rstrip('/')}/generate"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    response = requests.post(
        endpoint,
        json=payload,
        timeout=600,
    )
    response.raise_for_status()

    data = response.json()
    text = data.get("text") or data.get("output") or ""
    return text


def generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    stream: bool = False,
):
    """
    Unified entrypoint for generating model responses.
    """
    prompt = _apply_portuguese_instruction(prompt)

    if BACKEND == "vllm":
        if stream:
            return _vllm_stream(prompt, max_tokens, temperature)
        return _vllm_request(prompt, max_tokens, temperature)

    if stream:
        return _transformers_stream(prompt, max_tokens, temperature)
    return _transformers_generate(prompt, max_tokens, temperature)
