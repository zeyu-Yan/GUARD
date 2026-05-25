#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r", "").strip()


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def write_jsonl_strings(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize rewritten GUARD targets into CoT and answer JSONL files.")
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument("--cot_out", type=Path, required=True)
    parser.add_argument("--answer_out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.rewrites)
    cots: list[str] = []
    answers: list[str] = []
    for idx, row in enumerate(rows):
        cot = first_text(row, ("safe_cot", "r_natural", "reasoning", "cot"))
        answer = first_text(row, ("safe_answer", "a_natural", "answer", "final_answer"))
        if not cot or not answer:
            raise ValueError(f"missing safe_cot/safe_answer at rewrite row {idx}")
        cots.append(cot)
        answers.append(answer)
    write_jsonl_strings(args.cot_out, cots)
    write_jsonl_strings(args.answer_out, answers)
    print(json.dumps({"rows": len(rows), "cot_out": str(args.cot_out), "answer_out": str(args.answer_out)}, sort_keys=True))


if __name__ == "__main__":
    main()
