#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


GUARD_ROOT = Path(__file__).resolve().parents[1]


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


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def split_reasoning_answer(response: str) -> tuple[str, str]:
    text = clean_text(response)
    if "<think>" in text:
        text = text.split("<think>", 1)[1]
    if "</think>" not in text:
        return "", text
    cot, answer = text.split("</think>", 1)
    return clean_text(cot), clean_text(answer)


def row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("source_id", "id", "uid", "question"):
        value = clean_text(row.get(key))
        if value:
            keys.append(value)
    return keys


def maybe_limit(rows: list[dict[str, Any]], limit: int, seed: int, sample: str) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    if sample == "head":
        return rows[:limit]
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[idx] for idx in indices]


def normalize_row(row: dict[str, Any], idx: int, *, task_id: str, split: str) -> dict[str, Any]:
    question = first_text(row, ("question", "prompt", "query", "instruction"))
    answer = first_text(row, ("answer", "target", "completion", "response"))
    if not question:
        raise ValueError(f"missing question for {split} row {idx}")
    record: dict[str, Any] = {
        "idx": idx,
        "task_id": task_id if split == "forget" else str(row.get("task_id") or task_id),
        "split": split,
        "question": question,
        "answer": answer,
    }
    cot = first_text(row, ("cot", "reasoning", "raw_cot", "chain_of_thought"))
    if cot:
        record["cot"] = cot
        record["raw_cot"] = cot
    if answer:
        record["raw_answer"] = answer
    for key in ("source_id", "id", "category", "source_dataset", "safety_label"):
        if key in row:
            record[key] = row[key]
    return record


def safe_targets_for_row(row: dict[str, Any], safe_by_key: dict[str, dict[str, Any]]) -> tuple[str, str]:
    safe_row: dict[str, Any] | None = None
    for key in row_keys(row):
        if key in safe_by_key:
            safe_row = safe_by_key[key]
            break
    if safe_row is None:
        safe_row = row
    safe_cot = first_text(safe_row, ("safe_cot", "cot", "reasoning", "raw_cot"))
    safe_answer = first_text(safe_row, ("safe_answer", "answer", "target", "completion"))
    response = first_text(safe_row, ("safe_response", "response"))
    if response and (not safe_cot or not safe_answer):
        cot, answer = split_reasoning_answer(response)
        safe_cot = safe_cot or cot
        safe_answer = safe_answer or answer
    return safe_cot, safe_answer


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_strings(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize star forget/retain splits into GUARD JSON files.")
    parser.add_argument("--forget", type=Path, required=True)
    parser.add_argument("--retain", type=Path, default=None)
    parser.add_argument("--safe_source", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=GUARD_ROOT / "data" / "star")
    parser.add_argument("--task_id", default="1")
    parser.add_argument("--retain_task_id", default="retain")
    parser.add_argument("--forget_limit", type=int, default=0)
    parser.add_argument("--retain_limit", type=int, default=0)
    parser.add_argument("--sample", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    forget_raw = maybe_limit(read_json_or_jsonl(args.forget), int(args.forget_limit), int(args.seed), args.sample)
    retain_raw = (
        maybe_limit(read_json_or_jsonl(args.retain), int(args.retain_limit), int(args.seed), args.sample)
        if args.retain is not None
        else []
    )
    forget_out = [normalize_row(row, idx, task_id=str(args.task_id), split="forget") for idx, row in enumerate(forget_raw)]
    retain_out = [normalize_row(row, idx, task_id=str(args.retain_task_id), split="retain") for idx, row in enumerate(retain_raw)]

    out_dir = Path(args.out_dir)
    write_json(out_dir / "forget.json", forget_out)
    write_json(out_dir / "retain.json", retain_out)

    safe_by_key: dict[str, dict[str, Any]] = {}
    if args.safe_source is not None:
        for row in read_json_or_jsonl(args.safe_source):
            for key in row_keys(row):
                safe_by_key[key] = row
    safe_pairs = [safe_targets_for_row(row, safe_by_key) for row in forget_raw]
    if any(cot and answer for cot, answer in safe_pairs):
        if not all(cot and answer for cot, answer in safe_pairs):
            raise ValueError("safe targets are partially missing; provide complete safe_source or use trajectory rewriting")
        write_jsonl_strings(out_dir / "forget_safe_cot.jsonl", [cot for cot, _ in safe_pairs])
        write_jsonl_strings(out_dir / "forget_safe_answer.jsonl", [answer for _, answer in safe_pairs])

    manifest = {
        "dataset": "star",
        "forget_rows": len(forget_out),
        "retain_rows": len(retain_out),
        "safe_targets_written": bool(any(cot and answer for cot, answer in safe_pairs)),
        "task_id": str(args.task_id),
        "retain_task_id": str(args.retain_task_id),
        "outputs": {
            "forget_path": "forget.json",
            "retain_path": "retain.json",
            "idkcot_path": "forget_safe_cot.jsonl",
            "idontknow_path": "forget_safe_answer.jsonl",
        },
    }
    write_json(out_dir / "manifest.json", [manifest])
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
