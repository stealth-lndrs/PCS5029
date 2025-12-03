"""
Utility to create synthetic Portuguese QA pairs using OpenAI models from chunks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install openai package to use dataset generation.") from exc


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNKS_DIR = DATA_DIR / "chunks"

DEFAULT_MODEL = "gpt-4.1-mini"


def _read_chunks(limit: int | None = None) -> Iterable[str]:
    files = sorted(CHUNKS_DIR.glob("*.txt"))
    if limit:
        files = files[:limit]
    for path in files:
        yield path.read_text(encoding="utf-8")


def _build_prompt(chunk_text: str, questions_per_chunk: int, multi_hop: bool) -> str:
    multi_hop_instr = (
        "Crie perguntas multi-hop que exigem combinar informações de múltiplos pontos do texto."
        if multi_hop
        else "Crie perguntas diretas baseadas no texto."
    )
    return f"""
Você é um assistente especializado em gerar pares de Pergunta e Resposta em português brasileiro.
{multi_hop_instr}
Gere {questions_per_chunk} perguntas variadas com respostas concisas e cite sempre as partes relevantes.
Texto de origem:
\"\"\"{chunk_text}\"\"\"
Responda no formato JSON com uma lista 'qa_pairs', onde cada item contém 'question' e 'answer'.
"""


def _call_openai(
    client: OpenAI,
    prompt: str,
    model: str,
) -> List[dict]:
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    text = response.output[0].content[0].text  # type: ignore[attr-defined]
    data = json.loads(text)
    return data.get("qa_pairs", [])


def generate_dataset(
    output_path: Path,
    model: str = DEFAULT_MODEL,
    questions_per_chunk: int = 3,
    multi_hop: bool = False,
    chunk_limit: int | None = None,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    with output_path.open("w", encoding="utf-8") as jsonl:
        for chunk_text in _read_chunks(limit=chunk_limit):
            prompt = _build_prompt(chunk_text, questions_per_chunk, multi_hop)
            try:
                qa_pairs = _call_openai(client, prompt, model=model)
            except Exception as exc:
                print(f"Failed to generate QA pairs: {exc}")
                continue

            for pair in qa_pairs:
                record = {
                    "instruction": pair.get("question"),
                    "response": pair.get("answer"),
                    "context": chunk_text,
                }
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Dataset saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Portuguese QA dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/finetune/qa_dataset.jsonl"))
    parser.add_argument("--questions-per-chunk", type=int, default=3)
    parser.add_argument("--multi-hop", action="store_true")
    parser.add_argument("--chunk-limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(
        output_path=output_path,
        model=args.model,
        questions_per_chunk=args.questions_per_chunk,
        multi_hop=args.multi_hop,
        chunk_limit=args.chunk_limit,
    )
