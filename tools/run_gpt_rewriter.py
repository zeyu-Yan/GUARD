#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
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
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"]).strip()
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts).strip()


def parse_rewrite(text: str) -> dict[str, str]:
    raw = text.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    cot = str(payload.get("safe_cot") or payload.get("r_natural") or "").strip()
    answer = str(payload.get("safe_answer") or payload.get("a_natural") or "").strip()
    if not cot or not answer:
        raise ValueError("rewriter output must contain safe_cot and safe_answer")
    return {"safe_cot": cot, "safe_answer": answer}


def call_responses_api(*, base_url: str, api_key: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    if not base_url.strip():
        raise RuntimeError("missing rewriter base URL")
    if not api_key.strip():
        raise RuntimeError("missing rewriter API key")
    if not model.strip():
        raise RuntimeError("missing rewriter model")
    body = {
        "model": model,
        "input": prompt,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "guard-rewriter",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        done.add(str(row.get("source_id", row.get("idx", ""))))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Responses-compatible offline rewriter for GUARD trajectories.")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base_url", default=os.getenv("GUARD_REWRITER_BASE_URL", ""))
    parser.add_argument("--api_key_env", default="GUARD_REWRITER_API_KEY")
    parser.add_argument("--model", default=os.getenv("GUARD_REWRITER_MODEL", ""))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    api_key = os.getenv(str(args.api_key_env), "")
    done = completed_ids(args.out)
    rows = read_jsonl(args.requests)
    for row in rows:
        row_id = str(row.get("source_id", row.get("idx", "")))
        if row_id in done:
            continue
        prompt = str(row.get("prompt") or "")
        last_error = ""
        for attempt in range(int(args.max_retries) + 1):
            try:
                response = call_responses_api(
                    base_url=str(args.base_url),
                    api_key=api_key,
                    model=str(args.model),
                    prompt=prompt,
                    timeout=float(args.timeout),
                )
                text = output_text(response)
                rewrite = parse_rewrite(text)
                append_jsonl(
                    args.out,
                    {
                        "idx": row.get("idx"),
                        "source_id": row.get("source_id"),
                        "task_id": row.get("task_id"),
                        **rewrite,
                    },
                )
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= int(args.max_retries):
                    raise RuntimeError(f"rewrite failed for source_id={row_id}: {last_error}") from exc
                time.sleep(float(args.sleep))
    print(json.dumps({"requests": len(rows), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
