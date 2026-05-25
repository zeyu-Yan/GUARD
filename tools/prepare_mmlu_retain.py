#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


GUARD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = GUARD_ROOT / "data" / "mmlu_retain.json"


CHOICE_LABELS = ["A", "B", "C", "D"]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").split())


def iter_arrow_rows(cache_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    import pyarrow.ipc as ipc

    rows: list[dict[str, Any]] = []
    for split in splits:
        for arrow_path in sorted(cache_root.glob(f"*/0.0.0/*/mmlu-{split}.arrow")):
            subject = arrow_path.parents[2].name
            with arrow_path.open("rb") as handle:
                reader = ipc.RecordBatchStreamReader(handle)
                table = reader.read_all()
            for source_idx, item in enumerate(table.to_pylist()):
                question = clean_text(item.get("question"))
                choices = item.get("choices") or []
                answer_idx = int(item.get("answer"))
                if not question or len(choices) < 4 or answer_idx < 0 or answer_idx >= len(CHOICE_LABELS):
                    continue
                choice_lines = "\n".join(
                    f"{label}. {clean_text(choice)}"
                    for label, choice in zip(CHOICE_LABELS, choices[:4])
                )
                prompt = (
                    f"Subject: {subject}\n"
                    f"Question: {question}\n"
                    f"Choices:\n{choice_lines}\n"
                    "Answer with the correct option letter."
                )
                rows.append(
                    {
                        "task_id": "retain",
                        "question": prompt,
                        "answer": CHOICE_LABELS[answer_idx],
                        "source_answer": CHOICE_LABELS[answer_idx],
                        "source_id": f"mmlu_{split}_{subject}_{source_idx:05d}",
                        "source_dataset": f"mmlu_{split}",
                        "source_split": split,
                        "source_subject": subject,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cached MMLU splits into GUARD retain/eval JSON.")
    parser.add_argument("--cache_root", type=Path, default=Path("~/.cache/huggingface/datasets/cais___mmlu").expanduser())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", default="validation,dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    splits = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    rows = iter_arrow_rows(Path(args.cache_root), splits)
    if not rows:
        raise SystemExit(f"no MMLU records found under {args.cache_root} for splits={splits}")
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "rows": len(rows),
        "splits": splits,
        "cache_root": str(args.cache_root),
        "out": str(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
