#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON list in {path}")
        return [row for row in payload if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r", "").strip()


def build_prompt(question: str, source_cot: str, source_answer: str) -> str:
    return (
        "Rewrite the model trace into a natural non-disclosing trajectory.\n"
        "Preserve the reasoning cadence, but pivot toward uncertainty without leaking protected facts.\n"
        "Return a JSON object with exactly two string fields: safe_cot and safe_answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Source reasoning:\n{source_cot}\n\n"
        f"Source answer:\n{source_answer}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline trajectory-rewrite requests from GUARD forget data.")
    parser.add_argument("--forget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task_id", default="")
    args = parser.parse_args()

    rows = read_json_or_jsonl(args.forget)
    if args.task_id:
        rows = [row for row in rows if str(row.get("task_id", "")) == str(args.task_id)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            question = clean_text(row.get("question"))
            source_cot = clean_text(row.get("cot") or row.get("raw_cot") or row.get("reasoning"))
            source_answer = clean_text(row.get("answer") or row.get("raw_answer") or row.get("response"))
            payload = {
                "idx": idx,
                "task_id": str(row.get("task_id", "")),
                "source_id": row.get("source_id", row.get("id", idx)),
                "question": question,
                "source_cot": source_cot,
                "source_answer": source_answer,
                "prompt": build_prompt(question, source_cot, source_answer),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps({"requests": len(rows), "out": str(args.out)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
