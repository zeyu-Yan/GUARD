#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"expected JSON list in {path}")
        return [row for row in payload if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r", "").strip()


def split_trace(text: str) -> tuple[str, str]:
    value = clean_text(text)
    if "<think>" in value:
        value = value.split("<think>", 1)[1]
    if "</think>" in value:
        cot, answer = value.split("</think>", 1)
        return clean_text(cot), clean_text(answer)
    parts = re.split(r"\n\s*(?:Answer|Final answer)\s*:\s*", value, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return clean_text(parts[0]), clean_text(parts[1])
    return "", value


def build_prompt(question: str) -> str:
    return f"<｜User｜>{question}<｜Assistant｜><think>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate base-model source trajectories for GUARD forget prompts.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_json_or_jsonl(args.input)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(args.dtype)]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device)
    model.eval()

    outputs: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, int(args.batch_size))):
        batch_rows = rows[start : start + max(1, int(args.batch_size))]
        prompts = [build_prompt(clean_text(row.get("question") or row.get("prompt") or row.get("query"))) for row in batch_rows]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_width = int(encoded["input_ids"].shape[1])
        for row, token_ids in zip(batch_rows, generated):
            completion_ids = token_ids[input_width:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
            cot, answer = split_trace(completion)
            outputs.append(
                {
                    "source_id": row.get("source_id", row.get("id", len(outputs))),
                    "task_id": str(row.get("task_id", "1")),
                    "question": clean_text(row.get("question") or row.get("prompt") or row.get("query")),
                    "cot": cot,
                    "answer": answer,
                    "raw_completion": completion,
                    "category": row.get("category", []),
                }
            )
    write_jsonl(args.out, outputs)
    print(json.dumps({"rows": len(outputs), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
