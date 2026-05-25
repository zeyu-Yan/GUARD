#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GUARD_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hf_dataset(dataset_id: str, *args: Any, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_id, *args, split=split)


def rows_from_dataset(dataset) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def download_rtofu(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "rtofu"
    forget = rows_from_dataset(load_hf_dataset("sangyon/R-TOFU", split=str(args.rtofu_forget_split)))
    retain = rows_from_dataset(load_hf_dataset("sangyon/R-TOFU", split=str(args.rtofu_retain_split)))
    write_json(out_dir / "forget.json", forget)
    write_json(out_dir / "retain.json", retain)
    return {
        "dataset": "sangyon/R-TOFU",
        "forget_split": str(args.rtofu_forget_split),
        "retain_split": str(args.rtofu_retain_split),
        "forget_rows": len(forget),
        "retain_rows": len(retain),
        "forget": str(out_dir / "forget.json"),
        "retain": str(out_dir / "retain.json"),
    }


def download_star(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "star"
    rows = rows_from_dataset(load_hf_dataset("UCSC-VLAA/STAR-1", split="train"))
    write_jsonl(out_dir / "star1.jsonl", rows)
    return {
        "dataset": "UCSC-VLAA/STAR-1",
        "split": "train",
        "rows": len(rows),
        "out": str(out_dir / "star1.jsonl"),
    }


def download_squad(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "squad"
    rows = rows_from_dataset(load_hf_dataset("rajpurkar/squad", split="train"))
    write_json(out_dir / "train.json", rows)
    return {
        "dataset": "rajpurkar/squad",
        "split": "train",
        "rows": len(rows),
        "out": str(out_dir / "train.json"),
    }


def download_mmlu(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "mmlu"
    rows: list[dict[str, Any]] = []
    splits = [part.strip() for part in str(args.mmlu_splits).split(",") if part.strip()]
    for split in splits:
        for row in rows_from_dataset(load_hf_dataset("cais/mmlu", "all", split=split)):
            choices = row.get("choices") or []
            answer = row.get("answer")
            answer_idx = int(answer) if isinstance(answer, int) or str(answer).isdigit() else -1
            label = "ABCD"[answer_idx] if 0 <= answer_idx < 4 else str(answer)
            choice_lines = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", choices[:4]))
            rows.append(
                {
                    "task_id": "retain",
                    "question": (
                        f"Subject: {row.get('subject', 'mmlu')}\n"
                        f"Question: {row.get('question')}\n"
                        f"Choices:\n{choice_lines}\n"
                        "Answer with the correct option letter."
                    ),
                    "answer": label,
                    "source_answer": label,
                    "source_dataset": f"mmlu_{split}",
                    "source_split": split,
                    "source_subject": row.get("subject", ""),
                }
            )
    write_json(out_dir / "mmlu_retain.json", rows)
    return {
        "dataset": "cais/mmlu",
        "splits": splits,
        "rows": len(rows),
        "out": str(out_dir / "mmlu_retain.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets used by GUARD into raw_data/.")
    parser.add_argument("--dataset", choices=["all", "rtofu", "star", "squad", "mmlu"], default="all")
    parser.add_argument("--output_root", type=Path, default=GUARD_ROOT / "raw_data")
    parser.add_argument("--rtofu_forget_split", default="forget05")
    parser.add_argument("--rtofu_retain_split", default="retain50")
    parser.add_argument("--mmlu_splits", default="validation,dev")
    args = parser.parse_args()

    tasks = ["rtofu", "star", "squad", "mmlu"] if args.dataset == "all" else [str(args.dataset)]
    summaries: list[dict[str, Any]] = []
    for task in tasks:
        if task == "rtofu":
            summaries.append(download_rtofu(args))
        elif task == "star":
            summaries.append(download_star(args))
        elif task == "squad":
            summaries.append(download_squad(args))
        elif task == "mmlu":
            summaries.append(download_mmlu(args))
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_root) / "download_manifest.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
