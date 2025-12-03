"""
QLoRA fine-tuning script for Gemma 9B using JSONL SFT data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ADAPTER_DIR = DATA_DIR / "adapters"


@dataclass
class QAExample:
    instruction: str
    response: str
    context: str | None = None


class QADataset(Dataset):
    def __init__(self, records: Iterable[Dict[str, str]], tokenizer, max_length: int = 1024) -> None:
        self.examples: List[QAExample] = [
            QAExample(
                instruction=rec.get("instruction", ""),
                response=rec.get("response", ""),
                context=rec.get("context"),
            )
            for rec in records
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        prompt = f"Instrução:\n{example.instruction}\n\nContexto:\n{example.context or ''}\n\nResposta:"
        text = f"{prompt}\n{example.response}"
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        labels = tokens["input_ids"].copy()
        return {
            "input_ids": torch.tensor(tokens["input_ids"]),
            "attention_mask": torch.tensor(tokens["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def _load_jsonl(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train(
    dataset_path: Path,
    theme: str,
    base_model: str = "google/gemma-2-9b-it",
    epochs: int = 3,
    batch_size: int = 1,
    lr: float = 2e-4,
) -> None:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    model.config.use_cache = False

    dataset = QADataset(_load_jsonl(dataset_path), tokenizer)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    output_dir = ADAPTER_DIR / theme
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        gradient_accumulation_steps=8,
        fp16=True,
        optim="paged_adamw_32bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"LoRA adapters saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QLoRA adapters for Gemma 9B.")
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--theme", type=str, required=True, help="Name for adapter directory.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-model", type=str, default="google/gemma-2-9b-it")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.data,
        theme=args.theme,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
