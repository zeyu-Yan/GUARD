#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = ""
GUARD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = GUARD_ROOT / "data" / "squad_wiki_retain.json"


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def download_json(url: str, cache_path: Path) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        if not url:
            raise RuntimeError("missing dataset URL; provide --url or pre-populate --cache")
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=120) as response:
            tmp.write_bytes(response.read())
        tmp.replace(cache_path)
    return json.loads(cache_path.read_text(encoding="utf-8"))


def iter_squad_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for article_idx, article in enumerate(payload.get("data", []) or []):
        title = clean_text(article.get("title"))
        for para_idx, paragraph in enumerate(article.get("paragraphs", []) or []):
            context = clean_text(paragraph.get("context"))
            if len(context.split()) < 40:
                continue
            for qa_idx, qa in enumerate(paragraph.get("qas", []) or []):
                question = clean_text(qa.get("question"))
                answers = qa.get("answers", []) or []
                if not question or not answers:
                    continue
                answer = clean_text(answers[0].get("text"))
                if len(answer.split()) < 1 or len(answer.split()) > 40:
                    continue
                prompt = (
                    "Use the following Wikipedia passage to answer the question.\n\n"
                    f"Title: {title}\n"
                    f"Passage: {context}\n\n"
                    f"Question: {question}"
                )
                rows.append(
                    {
                        "task_id": "retain",
                        "question": prompt,
                        "answer": answer,
                        "source_id": f"squad_train_{article_idx:04d}_{para_idx:04d}_{qa_idx:04d}",
                        "source_dataset": "squad_wikipedia",
                        "source_title": title,
                        "source_split": "train",
                    }
                )
    return rows


def iter_flat_squad_rows(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(payload):
        title = clean_text(row.get("title"))
        context = clean_text(row.get("context"))
        question = clean_text(row.get("question"))
        answers = row.get("answers") or {}
        answer_values = answers.get("text") if isinstance(answers, dict) else answers
        if isinstance(answer_values, list):
            answer = clean_text(answer_values[0] if answer_values else "")
        else:
            answer = clean_text(answer_values)
        if not context or not question or not answer:
            continue
        if len(context.split()) < 40 or len(answer.split()) > 40:
            continue
        prompt = (
            "Use the following Wikipedia passage to answer the question.\n\n"
            f"Title: {title}\n"
            f"Passage: {context}\n\n"
            f"Question: {question}"
        )
        rows.append(
            {
                "task_id": "retain",
                "question": prompt,
                "answer": answer,
                "source_id": f"squad_train_{idx:06d}",
                "source_dataset": "squad_wikipedia",
                "source_title": title,
                "source_split": "train",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare open-ended Wikipedia QA retain data from a local SQuAD-style file.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_OUT.with_name("squad_train.json"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    payload = download_json(str(args.url), Path(args.cache))
    rows = iter_flat_squad_rows(payload) if isinstance(payload, list) else iter_squad_rows(payload)
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "cache": str(args.cache),
        "out": str(args.out),
        "seed": int(args.seed),
        "rows": len(rows),
        "source_dataset": "squad_wikipedia",
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
