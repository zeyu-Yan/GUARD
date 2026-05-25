#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
GUARD_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("GUARD_PROJECT_ROOT", str(GUARD_ROOT)))
DEFAULT_MODEL = Path(os.getenv("GUARD_MODEL_PATH", str(PROJECT_ROOT / "models" / "base-lrm")))
DEFAULT_FORGET = PROJECT_ROOT / "data" / "forget.json"
DEFAULT_RETAIN = PROJECT_ROOT / "data" / "retain.json"
DEFAULT_PREFIX = PROJECT_ROOT / "outputs" / "gta_runs" / "prefix_best.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ard_runs"


def log(message: str) -> None:
    print(f"[guard-ard] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r", "").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


IDK_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bi don['’]?t know\b",
        r"\bi do not know\b",
        r"\bnot sure\b",
        r"\bnot certain\b",
        r"\bwithout (more|additional)\b",
        r"\bcan't confidently\b",
        r"\bcannot confidently\b",
        r"\bunable to determine\b",
        r"\binsufficient\b",
        r"\bnot enough\b",
    ]
]


def normalize_match_text(text: Any) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(left: Any, right: Any) -> float:
    left_norm = normalize_match_text(left)
    right_norm = normalize_match_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(SequenceMatcher(None, left_norm, right_norm).ratio())


def looks_like_refusal(text: Any) -> bool:
    text = clean_text(text)
    return any(pattern.search(text) for pattern in IDK_PATTERNS)


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        inner = value[1:-1]
        return (
            inner.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\\\", "\\")
        )
    try:
        if any(ch in lowered for ch in [".", "e"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if yaml is not None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config must be a mapping: {path}")
        return payload

    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line or raw_line[:1].isspace():
            raise RuntimeError("PyYAML is not installed and fallback config parser only supports flat key: value files")
        key, value = line.split(":", 1)
        payload[key.strip()] = parse_scalar(value)
    return payload


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def read_json_or_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(line)
        return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_prefix_position(prefix_position: str | None) -> str:
    value = (prefix_position or "prompt_end").strip().lower().replace("-", "_")
    aliases = {
        "promptend": "prompt_end",
        "prompt_end": "prompt_end",
        "prompt_suffix": "prompt_end",
        "suffix": "prompt_end",
        "end": "prompt_end",
        "sequence_start": "sequence_start",
        "prefix_start": "sequence_start",
        "start": "sequence_start",
        "front": "sequence_start",
        "question_end": "question_end",
        "after_question": "question_end",
        "after_user_question": "question_end",
        "before_assistant": "question_end",
    }
    if value not in aliases:
        raise ValueError(
            f"unsupported prefix_position={prefix_position!r}; use prompt_end, question_end, or sequence_start"
        )
    return aliases[value]


def normalize_kl_token_scope(value: str) -> str:
    value = str(value or "full_trajectory").strip().lower().replace("-", "_")
    aliases = {
        "full": "full_trajectory",
        "fulltraj": "full_trajectory",
        "full_trajectory": "full_trajectory",
        "completion": "full_trajectory",
        "all": "full_trajectory",
        "cot": "cot",
        "cot_only": "cot",
        "cot_full": "cot",
        "cot_prefix": "cot_prefix",
        "cot_head": "cot_prefix",
        "cot_first": "cot_prefix",
    }
    if value not in aliases:
        raise ValueError(
            f"unsupported kl_token_scope={value!r}; use full_trajectory, cot, or cot_prefix"
        )
    return aliases[value]


def normalize_distill_objective(value: str | None) -> str:
    value = str(value or "trajectory_kl").strip().lower().replace("-", "_")
    aliases = {
        "kl": "trajectory_kl",
        "trajectory_kl": "trajectory_kl",
        "full_trajectory_kl": "trajectory_kl",
        "union": "union_topk_huber",
        "union_topk": "union_topk_huber",
        "union_topk_huber": "union_topk_huber",
        "universal_topk_huber": "union_topk_huber",
        "jeffrey": "jeffrey_topk_kl",
        "jeffrey_topk": "jeffrey_topk_kl",
        "jeffrey_topk_kl": "jeffrey_topk_kl",
        "jdiv": "jeffrey_topk_kl",
        "symmetric_kl": "jeffrey_topk_kl",
        "endpoint_huber": "endpoint_full_huber",
        "endpoint_full_huber": "endpoint_full_huber",
        "prompt_answer_huber": "endpoint_full_huber",
        "prompt_answer_full_huber": "endpoint_full_huber",
        "endpoint_topk": "endpoint_topk_huber",
        "endpoint_topk_huber": "endpoint_topk_huber",
        "prompt_answer_topk_huber": "endpoint_topk_huber",
        "endpoint_prompt_answer_topk_huber": "endpoint_topk_huber",
        "prefix_shift": "prefix_shift_huber",
        "prefix_shift_huber": "prefix_shift_huber",
        "policy_delta": "prefix_shift_huber",
        "policy_delta_huber": "prefix_shift_huber",
        "delta_huber": "prefix_shift_huber",
        "probability_shift": "probability_shift_huber",
        "probability_shift_huber": "probability_shift_huber",
        "prob_shift": "probability_shift_huber",
        "prob_shift_huber": "probability_shift_huber",
        "logprob_shift": "probability_shift_huber",
        "logprob_shift_huber": "probability_shift_huber",
    }
    if value not in aliases:
        raise ValueError(
            f"unsupported distill_objective={value!r}; use trajectory_kl, union_topk_huber, "
            "jeffrey_topk_kl, endpoint_full_huber, endpoint_topk_huber, "
            "prefix_shift_huber, or probability_shift_huber"
        )
    return aliases[value]


@dataclass(frozen=True)
class PromptFormat:
    question_start: str = "<｜User｜>"
    question_end: str = "<｜Assistant｜>"
    think_start: str = "<think>\n"
    think_end: str = "\n</think>\n\n"
    answer_tag: str = ""

    def prompt(self, question: str) -> str:
        return f"{self.question_start}{question}{self.question_end}{self.think_start}"

    def question_prompt(self, question: str) -> str:
        return f"{self.question_start}{question}"

    def root_prompt(self, question: str) -> str:
        return f"{self.question_start}{question}{self.question_end}"

    def answer_prompt(self, question: str) -> str:
        return f"{self.question_start}{question}{self.question_end}\nAnswer:"


@dataclass
class SourceRecord:
    split: str
    teacher: str
    source_dataset: str
    source_idx: int
    local_idx: int
    task_id: str
    question: str
    source_answer: str
    trajectory_kind: str = ""
    fixed_completion: str = ""
    fixed_prompt: str = ""
    root_state: bool = False

    @property
    def source_key(self) -> str:
        return f"{self.source_dataset}:{self.source_idx}"

    def to_json(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "teacher": self.teacher,
            "source_dataset": self.source_dataset,
            "source_idx": self.source_idx,
            "source_key": self.source_key,
            "local_idx": self.local_idx,
            "task_id": self.task_id,
            "question": self.question,
            "source_answer": self.source_answer,
            "trajectory_kind": self.trajectory_kind,
            "fixed_completion": self.fixed_completion,
            "fixed_prompt": self.fixed_prompt,
            "root_state": bool(self.root_state),
        }


class SoftPrefix(nn.Module):
    def __init__(self, embedding: torch.Tensor, prefix_position: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__()
        if embedding.ndim != 2:
            raise ValueError(f"prefix embedding must have shape [prefix_len, hidden], got {tuple(embedding.shape)}")
        self.embedding = nn.Parameter(embedding.detach().clone().float(), requires_grad=False)
        self.prefix_position = normalize_prefix_position(prefix_position)
        self.metadata = dict(metadata or {})

    @property
    def prefix_len(self) -> int:
        return int(self.embedding.shape[0])

    def forward(self, batch_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.embedding.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


def load_soft_prefix(path: str | Path, prefix_position: str = "") -> SoftPrefix:
    path = Path(path)
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and "prefix_state_dict" in payload:
        state = payload["prefix_state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise ValueError(f"unsupported prefix checkpoint format: {path}")
    if "embedding" not in state:
        raise KeyError(f"prefix checkpoint lacks 'embedding': {path}")
    embedding = state["embedding"]
    if not isinstance(embedding, torch.Tensor):
        raise ValueError(f"prefix embedding must be a tensor: {path}")
    ckpt_args = payload.get("args") if isinstance(payload, dict) else {}
    ckpt_position = ckpt_args.get("prefix_position") if isinstance(ckpt_args, dict) else None
    resolved_position = normalize_prefix_position(prefix_position or ckpt_position or "prompt_end")
    metadata = {
        "path": str(path),
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "checkpoint_prefix_position": ckpt_position,
        "prefix_position": resolved_position,
        "checkpoint_args": ckpt_args if isinstance(ckpt_args, dict) else {},
    }
    return SoftPrefix(embedding, resolved_position, metadata)


def tokenized_len(tokenizer, text: str) -> int:
    return int(len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]))


def prefix_insert_len_for_record(
    tokenizer,
    prompt_format: PromptFormat,
    record: SourceRecord,
    prompt: str,
    prefix_position: str,
) -> int:
    position = normalize_prefix_position(prefix_position)
    prompt_len = tokenized_len(tokenizer, prompt)
    if position == "sequence_start":
        return 0
    if position == "prompt_end":
        return prompt_len
    question_prefix = prompt_format.question_prompt(record.question)
    insert_len = tokenized_len(tokenizer, question_prefix)
    if insert_len > prompt_len:
        return prompt_len
    return insert_len


def format_fixed_completion(cot: Any, answer: Any, prompt_format: PromptFormat) -> str:
    cot_text = clean_text(cot)
    answer_text = clean_text(answer)
    if cot_text and answer_text:
        return clean_text(f"{cot_text}{prompt_format.think_end}{prompt_format.answer_tag}{answer_text}")
    if cot_text:
        return clean_text(cot_text)
    return clean_text(f"{prompt_format.answer_tag}{answer_text}")


def build_forget_records(
    forget_path: Path,
    *,
    task_id: str,
    limit: int,
    prompt_format: PromptFormat,
    trajectory_mode: str,
) -> list[SourceRecord]:
    trajectory_mode = str(trajectory_mode or "generated").strip().lower()
    if trajectory_mode not in {"generated", "dual_fixed", "fixed_safe", "safe_fixed"}:
        raise ValueError("forget_trajectory_mode must be generated, fixed_safe, or dual_fixed")
    rows = read_json_or_jsonl(forget_path)
    records: list[SourceRecord] = []
    source_count = 0
    for source_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_task_id = str(row.get("task_id", ""))
        if row_task_id != str(task_id):
            continue
        if limit > 0 and source_count >= limit:
            break
        source_count += 1
        question = clean_text(row.get("question", ""))
        source_answer = clean_text(row.get("answer", ""))
        if trajectory_mode == "generated":
            records.append(
                SourceRecord(
                    split="forget",
                    teacher="prefix",
                    source_dataset=forget_path.stem,
                    source_idx=source_idx,
                    local_idx=len(records),
                    task_id=row_task_id,
                    question=question,
                    source_answer=source_answer,
                )
            )
        elif trajectory_mode in {"fixed_safe", "safe_fixed"}:
            completion = format_fixed_completion(row.get("target_cot", ""), row.get("target_answer", ""), prompt_format)
            if completion:
                records.append(
                    SourceRecord(
                        split="forget",
                        teacher="prefix",
                        source_dataset=f"{forget_path.stem}_safe",
                        source_idx=source_idx,
                        local_idx=len(records),
                        task_id=row_task_id,
                        question=question,
                        source_answer=source_answer,
                        trajectory_kind="safe",
                        fixed_completion=completion,
                    )
                )
        else:
            variants = [
                (
                    "raw",
                    format_fixed_completion(row.get("raw_cot", ""), row.get("answer", ""), prompt_format),
                ),
                (
                    "safe",
                    format_fixed_completion(row.get("target_cot", ""), row.get("target_answer", ""), prompt_format),
                ),
            ]
            for kind, completion in variants:
                if not completion:
                    continue
                records.append(
                    SourceRecord(
                        split="forget",
                        teacher="prefix",
                        source_dataset=f"{forget_path.stem}_{kind}",
                        source_idx=source_idx,
                        local_idx=len(records),
                        task_id=row_task_id,
                        question=question,
                        source_answer=source_answer,
                        trajectory_kind=kind,
                        fixed_completion=completion,
                    )
                )
    if not records:
        raise ValueError(f"no forget records found for task_id={task_id!r} in {forget_path}")
    return records


def build_retain_records(
    retain_path: Path,
    *,
    limit: int,
    seed: int,
    sample: str,
    extra_forget_path: Path | None = None,
    extra_exclude_task_id: str = "",
) -> list[SourceRecord]:
    candidates: list[tuple[str, int, dict[str, Any]]] = []
    for source_idx, row in enumerate(read_json_or_jsonl(retain_path)):
        if isinstance(row, dict):
            candidates.append((retain_path.stem, source_idx, row))

    if extra_forget_path:
        excluded_task = str(extra_exclude_task_id or "")
        for source_idx, row in enumerate(read_json_or_jsonl(extra_forget_path)):
            if not isinstance(row, dict):
                continue
            if excluded_task and str(row.get("task_id")) == excluded_task:
                continue
            candidates.append((extra_forget_path.stem, source_idx, row))

    if not candidates:
        raise ValueError("no retain candidates found")
    sample = str(sample).strip().lower()
    if sample == "random":
        rng = random.Random(seed)
        rng.shuffle(candidates)
    elif sample != "head":
        raise ValueError(f"unsupported retain_sample={sample!r}; use head or random")
    if limit > 0:
        candidates = candidates[:limit]

    records: list[SourceRecord] = []
    for source_name, source_idx, row in candidates:
        records.append(
            SourceRecord(
                split="retain",
                teacher="base",
                source_dataset=source_name,
                source_idx=source_idx,
                local_idx=len(records),
                task_id=str(row.get("task_id", "retain")),
                question=clean_text(row.get("question", "")),
                source_answer=clean_text(row.get("answer", "")),
            )
        )
    return records


def add_root_state_records(
    records: list[SourceRecord],
    *,
    prompt_format: PromptFormat,
    root_state_completion: str,
    root_state_templates: str,
) -> list[SourceRecord]:
    if not records:
        return []
    mixed = list(records)
    placeholder = clean_text(root_state_completion) or "."
    templates = [
        item.strip().lower().replace("-", "_")
        for item in str(root_state_templates or "assistant").split(",")
        if item.strip()
    ]
    if not templates:
        templates = ["assistant"]
    for record in records:
        for template in templates:
            if template in {"assistant", "root", "prompt_end"}:
                prompt = prompt_format.root_prompt(record.question)
                suffix = "root"
            elif template in {"answer", "answer_cue", "open_answer"}:
                prompt = prompt_format.answer_prompt(record.question)
                suffix = "answer_root"
            else:
                raise ValueError(
                    f"unsupported root_state_template={template!r}; "
                    "use assistant or answer"
                )
            mixed.append(
                SourceRecord(
                    split=record.split,
                    teacher=record.teacher,
                    source_dataset=f"{record.source_dataset}_{suffix}",
                    source_idx=record.source_idx,
                    local_idx=len(mixed),
                    task_id=record.task_id,
                    question=record.question,
                    source_answer=record.source_answer,
                    trajectory_kind=suffix,
                    fixed_completion=placeholder,
                    fixed_prompt=prompt,
                    root_state=True,
                )
            )
    return mixed


def trim_generated_ids(ids: list[int], *, eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    trimmed = list(ids)
    if eos_token_id is not None and eos_token_id in trimmed:
        trimmed = trimmed[: trimmed.index(eos_token_id) + 1]
    else:
        while trimmed and pad_token_id is not None and trimmed[-1] == pad_token_id:
            trimmed.pop()
        if eos_token_id is not None:
            trimmed.append(int(eos_token_id))
    return trimmed


def generate_batch_ids(
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    max_new_tokens: int,
    prefix_insert_lens: list[int] | None = None,
) -> tuple[list[str], list[list[int]]]:
    if not prompts:
        return [], []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        if prefix is None:
            with torch.inference_mode():
                output = model.generate(
                    **enc,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            suffix = output[:, enc["input_ids"].shape[1] :]
        else:
            with torch.inference_mode():
                token_embeds = model.get_input_embeddings()(enc["input_ids"])
                batch_size = int(token_embeds.shape[0])
                prefix_embeds = prefix(batch_size, dtype=token_embeds.dtype, device=token_embeds.device)
                prefix_mask = torch.ones(
                    (batch_size, prefix.prefix_len),
                    dtype=enc["attention_mask"].dtype,
                    device=device,
                )
                if prefix.prefix_position == "sequence_start":
                    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
                    attention_mask = torch.cat([prefix_mask, enc["attention_mask"]], dim=1)
                elif prefix.prefix_position == "question_end":
                    if prefix_insert_lens is None:
                        raise ValueError("prefix_insert_lens is required for prefix_position=question_end")
                    seq_lens = enc["attention_mask"].sum(dim=1).to(torch.long)
                    max_len = int(enc["input_ids"].shape[1])
                    embed_rows: list[torch.Tensor] = []
                    mask_rows: list[torch.Tensor] = []
                    for row_idx in range(batch_size):
                        insert_len = int(prefix_insert_lens[row_idx])
                        seq_len = int(seq_lens[row_idx].item())
                        if insert_len < 0 or insert_len > seq_len:
                            raise ValueError(f"invalid prefix_insert_len={insert_len}; seq_len={seq_len}")
                        left_pad = max_len - seq_len
                        insert_at = left_pad + insert_len
                        embed_rows.append(
                            torch.cat(
                                [
                                    token_embeds[row_idx, :insert_at],
                                    prefix_embeds[row_idx],
                                    token_embeds[row_idx, insert_at:],
                                ],
                                dim=0,
                            )
                        )
                        mask_rows.append(
                            torch.cat(
                                [
                                    enc["attention_mask"][row_idx, :insert_at],
                                    prefix_mask[row_idx],
                                    enc["attention_mask"][row_idx, insert_at:],
                                ],
                                dim=0,
                            )
                        )
                    inputs_embeds = torch.stack(embed_rows, dim=0)
                    attention_mask = torch.stack(mask_rows, dim=0)
                else:
                    inputs_embeds = torch.cat([token_embeds, prefix_embeds], dim=1)
                    attention_mask = torch.cat([enc["attention_mask"], prefix_mask], dim=1)
                dummy_ids = torch.full(
                    (batch_size, int(inputs_embeds.shape[1])),
                    int(pad_token_id),
                    dtype=torch.long,
                    device=device,
                )
                output = model.generate(
                    input_ids=dummy_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            suffix = output[:, dummy_ids.shape[1] :]

        completion_ids = [
            trim_generated_ids(
                [int(token_id) for token_id in row.tolist()],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
            for row in suffix
        ]
        completions = [clean_text(tokenizer.decode(ids, skip_special_tokens=True)) for ids in completion_ids]
        return completions, completion_ids
    finally:
        tokenizer.padding_side = old_padding_side


def parse_completion(completion: str, prompt_format: PromptFormat) -> tuple[str, str]:
    completion = clean_text(completion)
    if prompt_format.think_end and prompt_format.think_end in completion:
        cot, answer = completion.split(prompt_format.think_end, 1)
        return clean_text(cot), clean_text(answer)
    if "</think>" in completion:
        cot, answer = completion.split("</think>", 1)
        return clean_text(cot), clean_text(answer)
    return completion, ""


def token_count(tokenizer, text: str) -> int:
    return int(len(tokenizer(text or "", add_special_tokens=False).get("input_ids", [])))


def build_generation_row(
    *,
    phase: str,
    split: str,
    prefix_used: bool,
    record: SourceRecord,
    prompt: str,
    completion: str,
    prompt_format: PromptFormat,
    tokenizer,
    step: int,
) -> dict[str, Any]:
    cot, answer = parse_completion(completion, prompt_format)
    answer_or_completion = answer or completion
    source_answer = clean_text(record.source_answer)
    return {
        **record.to_json(),
        "phase": phase,
        "split": split,
        "prefix_used": int(bool(prefix_used)),
        "step": int(step),
        "prompt": prompt,
        "completion": clean_text(completion),
        "cot": cot,
        "answer": answer,
        "has_think_end": int("</think>" in completion),
        "idk_like": int(looks_like_refusal(answer_or_completion)),
        "source_answer_similarity": text_similarity(answer_or_completion, source_answer),
        "source_answer_substring_in_completion": int(bool(source_answer and normalize_match_text(source_answer) in normalize_match_text(completion))),
        "completion_tokens": token_count(tokenizer, completion),
        "cot_tokens": token_count(tokenizer, cot),
        "answer_tokens": token_count(tokenizer, answer),
    }


def summarize_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(selected: list[dict[str, Any]], key: str) -> float:
        if not selected:
            return 0.0
        return float(sum(float(row.get(key, 0.0)) for row in selected) / len(selected))

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(selected),
            "prefix_used_rate": mean(selected, "prefix_used"),
            "idk_like_mean": mean(selected, "idk_like"),
            "has_think_end_mean": mean(selected, "has_think_end"),
            "source_answer_similarity_mean": mean(selected, "source_answer_similarity"),
            "source_answer_substring_in_completion_mean": mean(selected, "source_answer_substring_in_completion"),
            "completion_tokens_mean": mean(selected, "completion_tokens"),
            "cot_tokens_mean": mean(selected, "cot_tokens"),
            "answer_tokens_mean": mean(selected, "answer_tokens"),
        }

    forget_rows = [row for row in rows if row.get("split") == "forget"]
    retain_rows = [row for row in rows if row.get("split") == "retain"]
    return {
        "overall": summarize(rows),
        "forget": summarize(forget_rows),
        "retain": summarize(retain_rows),
    }


def select_eval_records(
    forget_records: list[SourceRecord],
    retain_records: list[SourceRecord],
    *,
    forget_count: int,
    retain_count: int,
    seed: int,
) -> tuple[list[SourceRecord], list[SourceRecord], dict[str, Any]]:
    rng = random.Random(int(seed))
    if forget_count > 0 and forget_count < len(forget_records):
        forget_indices = sorted(rng.sample(range(len(forget_records)), int(forget_count)))
    else:
        forget_indices = list(range(len(forget_records)))
    rng = random.Random(int(seed) + 1)
    if retain_count > 0 and retain_count < len(retain_records):
        retain_indices = sorted(rng.sample(range(len(retain_records)), int(retain_count)))
    else:
        retain_indices = list(range(len(retain_records)))
    selected_forget = [forget_records[idx] for idx in forget_indices]
    selected_retain = [retain_records[idx] for idx in retain_indices]
    return (
        selected_forget,
        selected_retain,
        {
            "seed": int(seed),
            "forget_count": int(forget_count),
            "retain_count": int(retain_count),
            "forget_indices": forget_indices,
            "retain_indices": retain_indices,
            "forget_source_keys": [record.source_key for record in selected_forget],
            "retain_source_keys": [record.source_key for record in selected_retain],
        },
    )


def run_generation_review(
    model,
    tokenizer,
    prompt_format: PromptFormat,
    forget_records: list[SourceRecord],
    retain_records: list[SourceRecord],
    output_dir: Path,
    *,
    phase: str,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    forget_prefix: SoftPrefix | None,
    retain_prefix: SoftPrefix | None,
    disable_adapter: bool,
) -> list[dict[str, Any]]:
    was_training = bool(model.training)
    old_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_use_cache is not None:
        model.config.use_cache = True

    rows: list[dict[str, Any]] = []
    adapter_ctx = adapter_disabled(model) if disable_adapter else contextlib.nullcontext()
    with torch.inference_mode(), adapter_ctx:
        for split, records, split_prefix in [
            ("forget", forget_records, forget_prefix),
            ("retain", retain_records, retain_prefix),
        ]:
            for start in range(0, len(records), max(1, int(batch_size))):
                batch_records = records[start : start + max(1, int(batch_size))]
                prompts = [prompt_format.prompt(record.question) for record in batch_records]
                prefix_insert_lens = (
                    [
                        prefix_insert_len_for_record(tokenizer, prompt_format, record, prompt, split_prefix.prefix_position)
                        for record, prompt in zip(batch_records, prompts)
                    ]
                    if split_prefix is not None and split_prefix.prefix_position == "question_end"
                    else None
                )
                completions, _ = generate_batch_ids(
                    model,
                    split_prefix,
                    tokenizer,
                    prompts,
                    device=device,
                    max_new_tokens=int(max_new_tokens),
                    prefix_insert_lens=prefix_insert_lens,
                )
                for record, prompt, completion in zip(batch_records, prompts, completions):
                    rows.append(
                        build_generation_row(
                            phase=phase,
                            split=split,
                            prefix_used=split_prefix is not None,
                            record=record,
                            prompt=prompt,
                            completion=completion,
                            prompt_format=prompt_format,
                            tokenizer=tokenizer,
                            step=step,
                        )
                    )

    tag = f"{phase}_step{int(step):06d}"
    write_jsonl(output_dir / f"generation_{tag}.jsonl", rows)
    write_csv(output_dir / f"generation_{tag}.csv", rows)
    write_json(output_dir / f"generation_{tag}_summary.json", summarize_generation_rows(rows))

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if was_training:
        model.train()
    else:
        model.eval()
    return rows


def compare_generation_rows(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row.get("split", "")), str(row.get("source_key", ""))

    pre_by_key = {key(row): row for row in pre_rows}
    post_by_key = {key(row): row for row in post_rows}
    rows: list[dict[str, Any]] = []
    for item_key in sorted(pre_by_key):
        if item_key not in post_by_key:
            continue
        pre = pre_by_key[item_key]
        post = post_by_key[item_key]
        rows.append(
            {
                "split": item_key[0],
                "source_key": item_key[1],
                "question": pre.get("question", ""),
                "source_answer": pre.get("source_answer", ""),
                "pre_prefix_used": pre.get("prefix_used", 0),
                "post_prefix_used": post.get("prefix_used", 0),
                "pre_completion": pre.get("completion", ""),
                "post_completion": post.get("completion", ""),
                "pre_answer": pre.get("answer", ""),
                "post_answer": post.get("answer", ""),
                "pre_idk_like": pre.get("idk_like", 0),
                "post_idk_like": post.get("idk_like", 0),
                "pre_source_answer_similarity": pre.get("source_answer_similarity", 0.0),
                "post_source_answer_similarity": post.get("source_answer_similarity", 0.0),
                "pre_source_answer_substring_in_completion": pre.get("source_answer_substring_in_completion", 0),
                "post_source_answer_substring_in_completion": post.get("source_answer_substring_in_completion", 0),
                "answer_similarity_pre_post": text_similarity(pre.get("answer", ""), post.get("answer", "")),
                "completion_similarity_pre_post": text_similarity(pre.get("completion", ""), post.get("completion", "")),
            }
        )
    return rows


def summarize_generation_compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(selected: list[dict[str, Any]], key: str) -> float:
        if not selected:
            return 0.0
        return float(sum(float(row.get(key, 0.0)) for row in selected) / len(selected))

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(selected),
            "pre_idk_like_mean": mean(selected, "pre_idk_like"),
            "post_idk_like_mean": mean(selected, "post_idk_like"),
            "pre_source_answer_similarity_mean": mean(selected, "pre_source_answer_similarity"),
            "post_source_answer_similarity_mean": mean(selected, "post_source_answer_similarity"),
            "pre_source_answer_substring_mean": mean(selected, "pre_source_answer_substring_in_completion"),
            "post_source_answer_substring_mean": mean(selected, "post_source_answer_substring_in_completion"),
            "answer_similarity_pre_post_mean": mean(selected, "answer_similarity_pre_post"),
            "completion_similarity_pre_post_mean": mean(selected, "completion_similarity_pre_post"),
        }

    forget_rows = [row for row in rows if row.get("split") == "forget"]
    retain_rows = [row for row in rows if row.get("split") == "retain"]
    return {
        "overall": summarize(rows),
        "forget": summarize(forget_rows),
        "retain": summarize(retain_rows),
    }


def trajectory_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("split", "")),
        str(row.get("source_dataset", "")),
        int(row.get("source_idx", -1)),
        str(row.get("question", "")),
    )


def load_or_generate_trajectories(
    model,
    tokenizer,
    prefix: SoftPrefix,
    retain_prefix: SoftPrefix | None,
    prompt_format: PromptFormat,
    forget_records: list[SourceRecord],
    retain_records: list[SourceRecord],
    cache_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    forget_max_new_tokens: int,
    retain_max_new_tokens: int,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records = forget_records + retain_records
    wanted_keys = [(r.split, r.source_dataset, r.source_idx, r.question) for r in all_records]
    wanted = set(wanted_keys)
    expected_prefix_used = {
        (r.split, r.source_dataset, r.source_idx, r.question): bool(r.split == "forget" or retain_prefix is not None)
        for r in all_records
    }
    expected_prefix_position = {
        (r.split, r.source_dataset, r.source_idx, r.question): (
            prefix.prefix_position
            if r.split == "forget"
            else (retain_prefix.prefix_position if retain_prefix is not None else "")
        )
        for r in all_records
    }
    cached_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    if cache_path.exists() and not force:
        rows = [row for row in read_json_or_jsonl(cache_path) if isinstance(row, dict)]
        for row in rows:
            key = trajectory_key(row)
            row_position = str(row.get("teacher_prefix_position") or "")
            if bool(row.get("teacher_prefix_used", False)) and not row_position:
                row_position = "prompt_end"
            if (
                key in wanted
                and bool(row.get("teacher_prefix_used", False)) == bool(expected_prefix_used[key])
                and row_position == str(expected_prefix_position[key] or "")
                and key not in cached_by_key
            ):
                cached_by_key[key] = row
        if wanted.issubset(set(cached_by_key)):
            ordered = [cached_by_key[key] for key in wanted_keys]
            log(f"loaded cached teacher trajectories n={len(ordered)} cache={cache_path}")
            return (
                [row for row in ordered if str(row.get("split")) == "forget"],
                [row for row in ordered if str(row.get("split")) == "retain"],
            )
        log(
            f"trajectory cache covers {len(cached_by_key)}/{len(wanted)} requested records; "
            f"generating missing rows cache={cache_path}"
        )
    elif cache_path.exists() and force:
        log(f"force_rebuild_trajectories=true; ignoring existing cache={cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    generated_rows: list[dict[str, Any]] = []
    generated_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}

    def generate_split(records: list[SourceRecord], split: str, split_prefix: SoftPrefix | None, max_new_tokens: int) -> None:
        if not records:
            log(f"self-generate {split} skipped; cache already covers requested rows")
            return
        fixed_records = [record for record in records if clean_text(record.fixed_completion)]
        generated_records = [record for record in records if not clean_text(record.fixed_completion)]
        for record in fixed_records:
            prompt = clean_text(record.fixed_prompt) or prompt_format.prompt(record.question)
            insert_len = (
                prefix_insert_len_for_record(tokenizer, prompt_format, record, prompt, split_prefix.prefix_position)
                if split_prefix is not None
                else tokenized_len(tokenizer, prompt)
            )
            completion = clean_text(record.fixed_completion)
            ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
            ids = trim_generated_ids(ids, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            row = {
                **record.to_json(),
                "prompt": prompt,
                "teacher_prefix_used": bool(split_prefix is not None),
                "teacher_prefix_position": split_prefix.prefix_position if split_prefix is not None else "",
                "prefix_insert_len": int(insert_len),
                "teacher_completion": completion,
                "completion_ids": ids,
                "completion_token_count": len(ids),
                "max_new_tokens": 0,
                "has_eos_target": bool(tokenizer.eos_token_id is not None and ids and ids[-1] == tokenizer.eos_token_id),
                "fixed_trajectory": True,
            }
            generated_rows.append(row)
            generated_by_key[(record.split, record.source_dataset, record.source_idx, record.question)] = row
        if fixed_records:
            log(f"loaded fixed {split} trajectories n={len(fixed_records)}")
        records = generated_records
        if not records:
            return
        total_batches = math.ceil(len(records) / max(1, batch_size))
        for start in range(0, len(records), max(1, batch_size)):
            batch_records = records[start : start + max(1, batch_size)]
            prompts = [prompt_format.prompt(record.question) for record in batch_records]
            prefix_insert_lens = (
                [
                    prefix_insert_len_for_record(tokenizer, prompt_format, record, prompt, split_prefix.prefix_position)
                    for record, prompt in zip(batch_records, prompts)
                ]
                if split_prefix is not None
                else None
            )
            completions, completion_ids = generate_batch_ids(
                model,
                split_prefix,
                tokenizer,
                prompts,
                device=device,
                max_new_tokens=max_new_tokens,
                prefix_insert_lens=(
                    prefix_insert_lens
                    if split_prefix is not None and split_prefix.prefix_position == "question_end"
                    else None
                ),
            )
            for record, prompt, completion, ids in zip(batch_records, prompts, completions, completion_ids):
                insert_len = (
                    prefix_insert_len_for_record(tokenizer, prompt_format, record, prompt, split_prefix.prefix_position)
                    if split_prefix is not None
                    else tokenized_len(tokenizer, prompt)
                )
                row = {
                    **record.to_json(),
                    "prompt": prompt,
                    "teacher_prefix_used": bool(split_prefix is not None),
                    "teacher_prefix_position": split_prefix.prefix_position if split_prefix is not None else "",
                    "prefix_insert_len": int(insert_len),
                    "teacher_completion": completion,
                    "completion_ids": ids,
                    "completion_token_count": len(ids),
                    "max_new_tokens": int(max_new_tokens),
                    "has_eos_target": bool(tokenizer.eos_token_id is not None and ids and ids[-1] == tokenizer.eos_token_id),
                }
                generated_rows.append(row)
                generated_by_key[(record.split, record.source_dataset, record.source_idx, record.question)] = row
            batch_no = start // max(1, batch_size) + 1
            if batch_no == 1 or batch_no % 5 == 0 or batch_no == total_batches:
                log(f"self-generate {split} batch={batch_no}/{total_batches}")

    missing_forget = [
        record
        for record in forget_records
        if (record.split, record.source_dataset, record.source_idx, record.question) not in cached_by_key
    ]
    missing_retain = [
        record
        for record in retain_records
        if (record.split, record.source_dataset, record.source_idx, record.question) not in cached_by_key
    ]
    generate_split(missing_forget, "forget", prefix, int(forget_max_new_tokens))
    generate_split(missing_retain, "retain", retain_prefix, int(retain_max_new_tokens))
    merged_by_key = dict(cached_by_key)
    merged_by_key.update(generated_by_key)
    ordered = [merged_by_key[key] for key in wanted_keys]
    write_jsonl(cache_path, ordered)
    log(
        f"wrote teacher trajectories n={len(ordered)} generated={len(generated_rows)} "
        f"reused={len(cached_by_key)} cache={cache_path}"
    )
    return (
        [row for row in ordered if str(row.get("split")) == "forget"],
        [row for row in ordered if str(row.get("split")) == "retain"],
    )


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        *,
        max_length: int,
        skip_too_long: bool,
        prompt_format: PromptFormat,
        kl_token_scope: str,
        kl_cot_fraction: float,
    ) -> None:
        self.rows_in = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.skip_too_long = bool(skip_too_long)
        self.prompt_format = prompt_format
        self.kl_token_scope = normalize_kl_token_scope(kl_token_scope)
        self.kl_cot_fraction = float(kl_cot_fraction)
        self.rows: list[dict[str, Any]] = []
        skipped = 0
        for row in rows:
            try:
                self.rows.append(self._encode(row))
            except ValueError:
                if not self.skip_too_long:
                    raise
                skipped += 1
        if not self.rows:
            raise ValueError("all trajectory rows were skipped")
        if skipped:
            log(f"skipped trajectories={skipped} because they were too long or empty")

    def _completion_kl_mask(self, row: dict[str, Any], completion_ids: list[int]) -> list[int]:
        if bool(row.get("root_state", False)):
            return [1 if idx == 0 else 0 for idx in range(len(completion_ids))]

        valid_len = len(completion_ids)
        if self.tokenizer.eos_token_id is not None and completion_ids and completion_ids[-1] == self.tokenizer.eos_token_id:
            valid_len -= 1
        valid_len = max(0, valid_len)

        if self.kl_token_scope == "full_trajectory":
            return [1] * len(completion_ids)

        completion = clean_text(row.get("teacher_completion", ""))
        cot, _answer = parse_completion(completion, self.prompt_format)
        cot_ids = self.tokenizer(cot or "", add_special_tokens=False).get("input_ids", [])
        cot_len = min(valid_len, len(cot_ids))
        if cot_len <= 0 and completion:
            cot_len = valid_len
        if self.kl_token_scope == "cot":
            keep_len = cot_len
        elif self.kl_token_scope == "cot_prefix":
            fraction = min(1.0, max(0.0, self.kl_cot_fraction))
            keep_len = int(math.ceil(float(cot_len) * fraction)) if cot_len > 0 else 0
        else:
            raise ValueError(f"unsupported kl_token_scope={self.kl_token_scope!r}")
        keep_len = max(0, min(valid_len, keep_len))
        return [1 if idx < keep_len else 0 for idx in range(len(completion_ids))]

    def _answer_first_offset(self, row: dict[str, Any], completion_ids: list[int]) -> int:
        valid_len = len(completion_ids)
        if self.tokenizer.eos_token_id is not None and completion_ids and completion_ids[-1] == self.tokenizer.eos_token_id:
            valid_len -= 1
        valid_len = max(0, valid_len)
        if valid_len <= 0:
            return -1

        completion = clean_text(row.get("teacher_completion", ""))
        marker = self.prompt_format.think_end if self.prompt_format.think_end and self.prompt_format.think_end in completion else ""
        if marker:
            answer_prefix = completion.split(marker, 1)[0] + marker
        elif "</think>" in completion:
            answer_prefix = completion.split("</think>", 1)[0] + "</think>"
        else:
            return -1

        offset = int(len(self.tokenizer(answer_prefix, add_special_tokens=False).get("input_ids", [])))
        if offset < 0 or offset >= valid_len:
            return -1
        return offset

    def _encode(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt = str(row["prompt"])
        prompt_ids = list(self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        prefix_insert_len = int(row.get("prefix_insert_len", len(prompt_ids)))
        prefix_insert_len = max(0, min(prefix_insert_len, len(prompt_ids)))
        raw_completion_ids = row.get("completion_ids")
        if not isinstance(raw_completion_ids, list) or not raw_completion_ids:
            raise ValueError(f"missing completion_ids for {row.get('split')}:{row.get('source_key')}")
        completion_ids = [int(token_id) for token_id in raw_completion_ids]
        if self.tokenizer.eos_token_id is not None and completion_ids[-1] != self.tokenizer.eos_token_id:
            completion_ids.append(int(self.tokenizer.eos_token_id))
        input_ids = prompt_ids + completion_ids
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"trajectory too long split={row.get('split')} source={row.get('source_key')} "
                f"tokens={len(input_ids)} max_length={self.max_length}"
            )
        if not completion_ids:
            raise ValueError(f"empty completion for {row.get('split')}:{row.get('source_key')}")
        completion_kl_mask = self._completion_kl_mask(row, completion_ids)
        if len(completion_kl_mask) != len(completion_ids):
            raise ValueError("completion_kl_mask length mismatch")
        answer_first_offset = self._answer_first_offset(row, completion_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "prompt_len": len(prompt_ids),
            "prefix_insert_len": int(prefix_insert_len),
            "completion_len": len(completion_ids),
            "completion_kl_mask": completion_kl_mask,
            "completion_kl_len": int(sum(completion_kl_mask)),
            "answer_first_offset": int(answer_first_offset),
            "example": row,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class RawAnswerDataset(Dataset):
    def __init__(
        self,
        records: list[SourceRecord],
        tokenizer,
        *,
        max_length: int,
        skip_too_long: bool,
        prompt_format: PromptFormat,
        prompt_templates: str,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.skipped = 0
        templates = [
            item.strip().lower().replace("-", "_")
            for item in str(prompt_templates or "assistant").split(",")
            if item.strip()
        ]
        if not templates:
            templates = ["assistant"]
        for record in records:
            for template in templates:
                try:
                    self.rows.append(
                        self._encode(
                            record,
                            tokenizer,
                            max_length=int(max_length),
                            prompt_format=prompt_format,
                            template=template,
                        )
                    )
                except ValueError:
                    if not skip_too_long:
                        raise
                    self.skipped += 1
        if not self.rows:
            raise ValueError("no raw-answer erasure rows could be built")

    @staticmethod
    def _prompt(prompt_format: PromptFormat, question: str, template: str) -> str:
        if template in {"assistant", "root", "prompt_end"}:
            return prompt_format.root_prompt(question)
        if template in {"answer", "answer_cue", "open_answer"}:
            return prompt_format.answer_prompt(question)
        raise ValueError(f"unsupported raw_answer_prompt_template={template!r}; use assistant or answer")

    def _encode(
        self,
        record: SourceRecord,
        tokenizer,
        *,
        max_length: int,
        prompt_format: PromptFormat,
        template: str,
    ) -> dict[str, Any]:
        raw_answer = clean_text(record.source_answer)
        if not raw_answer:
            raise ValueError("empty raw answer")
        prompt = self._prompt(prompt_format, record.question, template)
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        full_ids = list(tokenizer(prompt + raw_answer, add_special_tokens=True, truncation=False)["input_ids"])
        if len(full_ids) > len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
            answer_start = len(prompt_ids)
        else:
            answer_ids = list(tokenizer(raw_answer, add_special_tokens=False, truncation=False)["input_ids"])
            if not answer_ids:
                raise ValueError("empty raw answer ids")
            full_ids = prompt_ids + answer_ids
            answer_start = len(prompt_ids)
        answer_len = len(full_ids) - int(answer_start)
        if answer_len <= 0:
            raise ValueError("empty raw answer positions")
        if len(full_ids) > max_length:
            raise ValueError(f"raw answer target too long: {len(full_ids)} > {max_length}")
        return {
            "input_ids": [int(token_id) for token_id in full_ids],
            "attention_mask": [1] * len(full_ids),
            "answer_start": int(answer_start),
            "answer_len": int(answer_len),
            "template": template,
            "example": {
                **record.to_json(),
                "raw_answer_prompt_template": template,
                "prompt": prompt,
                "raw_answer": raw_answer,
            },
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class RootPromptDataset(Dataset):
    def __init__(
        self,
        records: list[SourceRecord],
        tokenizer,
        *,
        max_length: int,
        skip_too_long: bool,
        prompt_format: PromptFormat,
        prompt_templates: str,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.skipped = 0
        templates = [
            item.strip().lower().replace("-", "_")
            for item in str(prompt_templates or "assistant").split(",")
            if item.strip()
        ]
        if not templates:
            templates = ["assistant"]
        for record in records:
            for template in templates:
                try:
                    self.rows.append(
                        self._encode(
                            record,
                            tokenizer,
                            max_length=int(max_length),
                            prompt_format=prompt_format,
                            template=template,
                        )
                    )
                except ValueError:
                    if not skip_too_long:
                        raise
                    self.skipped += 1
        if not self.rows:
            raise ValueError("no root-prompt rows could be built")

    @staticmethod
    def _prompt(prompt_format: PromptFormat, question: str, template: str) -> str:
        if template in {"assistant", "root", "prompt_end"}:
            return prompt_format.root_prompt(question)
        if template in {"answer", "answer_cue", "open_answer"}:
            return prompt_format.answer_prompt(question)
        raise ValueError(f"unsupported root_prompt_template={template!r}; use assistant or answer")

    def _encode(
        self,
        record: SourceRecord,
        tokenizer,
        *,
        max_length: int,
        prompt_format: PromptFormat,
        template: str,
    ) -> dict[str, Any]:
        prompt = self._prompt(prompt_format, record.question, template)
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        if not prompt_ids:
            raise ValueError("empty root prompt ids")
        if len(prompt_ids) > max_length:
            raise ValueError(f"root prompt too long: {len(prompt_ids)} > {max_length}")
        return {
            "input_ids": [int(token_id) for token_id in prompt_ids],
            "attention_mask": [1] * len(prompt_ids),
            "prompt_len": int(len(prompt_ids)),
            "template": template,
            "example": {
                **record.to_json(),
                "root_prompt_template": template,
                "prompt": prompt,
            },
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_trajectories(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    max_completion_len = max(int(row["completion_len"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    prompt_lens = torch.zeros((len(batch),), dtype=torch.long)
    prefix_insert_lens = torch.zeros((len(batch),), dtype=torch.long)
    completion_lens = torch.zeros((len(batch),), dtype=torch.long)
    answer_first_offsets = torch.full((len(batch),), -1, dtype=torch.long)
    completion_kl_masks = torch.zeros((len(batch), max_completion_len), dtype=torch.bool)
    examples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(batch):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.long)
        length = int(ids.numel())
        input_ids[row_idx, :length] = ids
        attention_mask[row_idx, :length] = mask
        prompt_lens[row_idx] = int(row["prompt_len"])
        prefix_insert_lens[row_idx] = int(row.get("prefix_insert_len", row["prompt_len"]))
        completion_lens[row_idx] = int(row["completion_len"])
        answer_first_offsets[row_idx] = int(row.get("answer_first_offset", -1))
        kl_mask = torch.tensor(row["completion_kl_mask"], dtype=torch.bool)
        completion_kl_masks[row_idx, : int(kl_mask.numel())] = kl_mask
        examples.append(row["example"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lens": prompt_lens,
        "prefix_insert_lens": prefix_insert_lens,
        "completion_lens": completion_lens,
        "answer_first_offsets": answer_first_offsets,
        "completion_kl_masks": completion_kl_masks,
        "examples": examples,
    }


def collate_raw_answers(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    answer_starts = torch.zeros((len(batch),), dtype=torch.long)
    answer_lens = torch.zeros((len(batch),), dtype=torch.long)
    examples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(batch):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        length = int(ids.numel())
        input_ids[row_idx, :length] = ids
        attention_mask[row_idx, :length] = torch.tensor(row["attention_mask"], dtype=torch.long)
        answer_starts[row_idx] = int(row["answer_start"])
        answer_lens[row_idx] = int(row["answer_len"])
        examples.append(row["example"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_starts": answer_starts,
        "answer_lens": answer_lens,
        "examples": examples,
    }


def collate_root_prompts(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    prompt_lens = torch.zeros((len(batch),), dtype=torch.long)
    examples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(batch):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        length = int(ids.numel())
        input_ids[row_idx, :length] = ids
        attention_mask[row_idx, :length] = torch.tensor(row["attention_mask"], dtype=torch.long)
        prompt_lens[row_idx] = int(row["prompt_len"])
        examples.append(row["example"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lens": prompt_lens,
        "examples": examples,
    }


def dataset_stats(dataset: TrajectoryDataset) -> dict[str, Any]:
    lengths = [len(row["input_ids"]) for row in dataset.rows]
    prompt_lens = [int(row["prompt_len"]) for row in dataset.rows]
    prefix_insert_lens = [int(row.get("prefix_insert_len", row["prompt_len"])) for row in dataset.rows]
    completion_lens = [int(row["completion_len"]) for row in dataset.rows]
    completion_kl_lens = [int(row.get("completion_kl_len", row["completion_len"])) for row in dataset.rows]
    answer_first_offsets = [int(row.get("answer_first_offset", -1)) for row in dataset.rows]

    def summarize(values: list[int]) -> dict[str, float]:
        values = sorted(values)
        if not values:
            return {"min": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        idx95 = int(round((len(values) - 1) * 0.95))
        return {
            "min": float(values[0]),
            "mean": float(sum(values) / len(values)),
            "p95": float(values[idx95]),
            "max": float(values[-1]),
        }

    return {
        "n": len(dataset),
        "full_tokens": summarize(lengths),
        "prompt_tokens": summarize(prompt_lens),
        "prefix_insert_tokens": summarize(prefix_insert_lens),
        "completion_tokens": summarize(completion_lens),
        "completion_kl_tokens": summarize(completion_kl_lens),
        "answer_first_token_available_rate": float(sum(1 for value in answer_first_offsets if value >= 0) / max(1, len(answer_first_offsets))),
    }


class CyclingLoader:
    def __init__(self, dataset: Dataset, tokenizer, batch_size: int, seed: int, *, collate_kind: str = "trajectory") -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.collate_kind = str(collate_kind)
        self.epoch = 0
        self.iterator = self._new_iterator()

    def _new_iterator(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=(
                (lambda rows: collate_raw_answers(rows, self.tokenizer.pad_token_id))
                if self.collate_kind == "raw_answer"
                else (
                    (lambda rows: collate_root_prompts(rows, self.tokenizer.pad_token_id))
                    if self.collate_kind == "root_prompt"
                    else (lambda rows: collate_trajectories(rows, self.tokenizer.pad_token_id))
                )
            ),
        )
        return iter(loader)

    def next(self) -> dict[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = self._new_iterator()
            return next(self.iterator)


def subset_dataset(dataset: TrajectoryDataset, indices: list[int]) -> TrajectoryDataset:
    if not indices:
        raise ValueError("cannot create an empty trajectory subset")
    subset = object.__new__(TrajectoryDataset)
    subset.rows_in = [dataset.rows_in[idx] for idx in indices if idx < len(dataset.rows_in)]
    subset.tokenizer = dataset.tokenizer
    subset.max_length = dataset.max_length
    subset.skip_too_long = dataset.skip_too_long
    subset.prompt_format = dataset.prompt_format
    subset.kl_token_scope = dataset.kl_token_scope
    subset.kl_cot_fraction = dataset.kl_cot_fraction
    subset.rows = [dataset.rows[idx] for idx in indices]
    return subset


def subset_raw_answer_dataset(dataset: RawAnswerDataset, indices: list[int]) -> RawAnswerDataset:
    if not indices:
        raise ValueError("cannot create an empty raw-answer subset")
    subset = object.__new__(RawAnswerDataset)
    subset.rows = [dataset.rows[idx] for idx in indices]
    subset.skipped = getattr(dataset, "skipped", 0)
    return subset


def subset_root_prompt_dataset(dataset: RootPromptDataset, indices: list[int]) -> RootPromptDataset:
    if not indices:
        raise ValueError("cannot create an empty root-prompt subset")
    subset = object.__new__(RootPromptDataset)
    subset.rows = [dataset.rows[idx] for idx in indices]
    subset.skipped = getattr(dataset, "skipped", 0)
    return subset


def sample_epoch_indices(total: int, sample_size: int, seed: int, epoch: int) -> list[int]:
    if total <= 0:
        raise ValueError("cannot sample from an empty pool")
    sample_size = int(sample_size)
    if sample_size <= 0 or sample_size >= total:
        indices = list(range(total))
    else:
        rng = random.Random(int(seed) + 1000003 * int(epoch))
        indices = sorted(rng.sample(range(total), sample_size))
    return indices


def build_epoch_schedule(
    *,
    steps_per_epoch: int,
    forget_updates: int,
    retain_updates: int,
) -> list[str]:
    return build_scheduler(int(steps_per_epoch), int(forget_updates), int(retain_updates))


def find_global_lora_targets(model) -> list[str]:
    targets: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short_name = module_name.rsplit(".", 1)[-1]
            if short_name != "lm_head":
                targets.add(short_name)
    if not targets:
        raise ValueError("no linear modules found for global LoRA")
    return sorted(targets)


def parse_lora_targets(value: str, model) -> list[str]:
    value = str(value or "all-linear").strip()
    if value in {"all-linear", "all_linear", "*"}:
        return find_global_lora_targets(model)
    targets = [part.strip() for part in value.split(",") if part.strip()]
    if not targets:
        raise ValueError("lora_target_modules resolved to an empty list")
    return targets


def enable_input_require_grads(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    def make_inputs_require_grad(_module, _input, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def cast_trainable_parameters_to_fp32(model) -> dict[str, int]:
    dtype_counts: dict[str, int] = {}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + int(param.numel())
        if param.dtype != torch.float32:
            param.data = param.data.float()
    return dtype_counts


def adapter_disabled(model):
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


def forward_shifted_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix: SoftPrefix | None,
    prompt_lens: torch.Tensor,
    prefix_insert_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    if prefix is None:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).logits[:, :-1, :]

    token_embeds = model.get_input_embeddings()(input_ids)
    batch_size, seq_len = input_ids.shape
    prefix_embeds = prefix(batch_size, dtype=token_embeds.dtype, device=token_embeds.device)
    prefix_mask = torch.ones(
        (batch_size, prefix.prefix_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    if prefix.prefix_position == "sequence_start":
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        logits = model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, use_cache=False, return_dict=True).logits
        return logits[:, prefix.prefix_len : prefix.prefix_len + seq_len - 1, :]

    prompt_lens = prompt_lens.to(device=input_ids.device)
    insert_lens = (
        prefix_insert_lens.to(device=input_ids.device)
        if prefix.prefix_position == "question_end" and prefix_insert_lens is not None
        else prompt_lens
    )
    embed_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for row_idx in range(batch_size):
        insert_at = int(insert_lens[row_idx].item())
        if insert_at < 0 or insert_at > seq_len:
            raise ValueError(f"invalid prompt_len={insert_at}; seq_len={seq_len}")
        embed_rows.append(
            torch.cat(
                [
                    token_embeds[row_idx, :insert_at],
                    prefix_embeds[row_idx],
                    token_embeds[row_idx, insert_at:],
                ],
                dim=0,
            )
        )
        mask_rows.append(
            torch.cat(
                [
                    attention_mask[row_idx, :insert_at],
                    prefix_mask[row_idx],
                    attention_mask[row_idx, insert_at:],
                ],
                dim=0,
            )
        )
    inputs_embeds = torch.stack(embed_rows, dim=0)
    full_attention_mask = torch.stack(mask_rows, dim=0)
    logits = model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, use_cache=False, return_dict=True).logits

    source_positions = torch.arange(seq_len - 1, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    prefix_visible_from = torch.clamp(insert_lens - 1, min=0).unsqueeze(1)
    offsets = (source_positions >= prefix_visible_from).to(torch.long) * prefix.prefix_len
    gather_positions = source_positions + offsets
    gather_index = gather_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return logits.gather(1, gather_index)


def forward_root_hidden(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix: SoftPrefix | None,
    prompt_lens: torch.Tensor,
    prefix_insert_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    if prefix is None:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        positions = torch.clamp(attention_mask.long().sum(dim=1) - 1, min=0)
        batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        return hidden[batch_index, positions, :]

    token_embeds = model.get_input_embeddings()(input_ids)
    batch_size, seq_len = input_ids.shape
    prefix_embeds = prefix(batch_size, dtype=token_embeds.dtype, device=token_embeds.device)
    prefix_mask = torch.ones(
        (batch_size, prefix.prefix_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    if prefix.prefix_position == "sequence_start":
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        positions = prefix.prefix_len + torch.clamp(attention_mask.long().sum(dim=1) - 1, min=0)
        batch_index = torch.arange(batch_size, device=input_ids.device)
        return hidden[batch_index, positions, :]

    prompt_lens = prompt_lens.to(device=input_ids.device)
    insert_lens = (
        prefix_insert_lens.to(device=input_ids.device)
        if prefix.prefix_position == "question_end" and prefix_insert_lens is not None
        else prompt_lens
    )
    embed_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    gather_positions: list[int] = []
    for row_idx in range(batch_size):
        insert_at = int(insert_lens[row_idx].item())
        if insert_at < 0 or insert_at > seq_len:
            raise ValueError(f"invalid prompt_len={insert_at}; seq_len={seq_len}")
        embed_rows.append(
            torch.cat(
                [
                    token_embeds[row_idx, :insert_at],
                    prefix_embeds[row_idx],
                    token_embeds[row_idx, insert_at:],
                ],
                dim=0,
            )
        )
        mask_rows.append(
            torch.cat(
                [
                    attention_mask[row_idx, :insert_at],
                    prefix_mask[row_idx],
                    attention_mask[row_idx, insert_at:],
                ],
                dim=0,
            )
        )
        gather_positions.append(insert_at + prefix.prefix_len - 1)
    inputs_embeds = torch.stack(embed_rows, dim=0)
    full_attention_mask = torch.stack(mask_rows, dim=0)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    batch_index = torch.arange(batch_size, device=input_ids.device)
    gather = torch.tensor(gather_positions, dtype=torch.long, device=input_ids.device)
    return hidden[batch_index, gather, :]


def completion_shift_mask(attention_mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    if seq_len <= 1:
        return torch.zeros((batch_size, 0), dtype=torch.bool, device=attention_mask.device)
    next_token_positions = torch.arange(1, seq_len, device=attention_mask.device).unsqueeze(0)
    valid_next_tokens = attention_mask[:, 1:].bool()
    return (next_token_positions >= prompt_lens.to(attention_mask.device).unsqueeze(1)) & valid_next_tokens


def apply_completion_kl_mask(
    shifted_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    completion_kl_masks: torch.Tensor | None,
) -> torch.Tensor:
    if completion_kl_masks is None:
        return shifted_mask
    completion_kl_masks = completion_kl_masks.to(device=shifted_mask.device, dtype=torch.bool)
    prompt_lens = prompt_lens.to(device=shifted_mask.device)
    scoped = torch.zeros_like(shifted_mask, dtype=torch.bool)
    for row_idx in range(int(shifted_mask.shape[0])):
        start = int(prompt_lens[row_idx].item()) - 1
        if start < 0:
            start = 0
        if start >= int(shifted_mask.shape[1]):
            continue
        available = int(shifted_mask.shape[1]) - start
        mask_len = min(available, int(completion_kl_masks.shape[1]))
        if mask_len <= 0:
            continue
        scoped[row_idx, start : start + mask_len] = completion_kl_masks[row_idx, :mask_len]
    return shifted_mask & scoped


def masked_trajectory_kl(
    student_shifted_logits: torch.Tensor,
    teacher_shifted_logits: torch.Tensor,
    shifted_mask: torch.Tensor,
    *,
    temperature: float,
    chunk_tokens: int,
    top_k: int,
    state_reduction: str = "token",
) -> torch.Tensor:
    if not bool(shifted_mask.any()):
        return student_shifted_logits.sum() * 0.0

    positions = shifted_mask.nonzero(as_tuple=False)
    temp = max(float(temperature), 1.0e-6)
    reduction = str(state_reduction or "token").strip().lower().replace("-", "_")
    if reduction not in {"token", "example", "state", "sequence"}:
        raise ValueError(f"unsupported state_reduction={state_reduction!r}; use token or example")
    example_balanced = reduction in {"example", "state", "sequence"}
    chunk = max(1, int(chunk_tokens))
    vocab = int(student_shifted_logits.shape[-1])
    top_k = int(top_k)
    if top_k > 0:
        top_k = min(top_k, vocab)

    total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
    denom = 0
    example_totals: dict[int, torch.Tensor] = {}
    example_counts: dict[int, int] = {}
    for start in range(0, int(positions.shape[0]), chunk):
        index = positions[start : start + chunk]
        batch_index = index[:, 0]
        token_index = index[:, 1]
        teacher = teacher_shifted_logits[batch_index, token_index, :].float() / temp
        student = student_shifted_logits[batch_index, token_index, :].float() / temp
        if top_k > 0:
            with torch.no_grad():
                top_vals, top_ids = torch.topk(teacher, k=top_k, dim=-1)
                teacher_log_probs = F.log_softmax(top_vals, dim=-1)
                teacher_probs = teacher_log_probs.exp()
            student_selected = torch.gather(student, dim=-1, index=top_ids)
            student_log_probs = F.log_softmax(student_selected, dim=-1)
            token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        else:
            with torch.no_grad():
                teacher_log_probs = F.log_softmax(teacher, dim=-1)
                teacher_probs = teacher_log_probs.exp()
            student_log_probs = F.log_softmax(student, dim=-1)
            token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        token_losses = token_kl * (temp * temp)
        if example_balanced:
            for row in range(int(index.shape[0])):
                example_id = int(batch_index[row].item())
                token_loss = token_losses[row]
                if example_id not in example_totals:
                    example_totals[example_id] = token_loss
                    example_counts[example_id] = 1
                else:
                    example_totals[example_id] = example_totals[example_id] + token_loss
                    example_counts[example_id] += 1
        else:
            total = total + token_losses.sum()
        denom += int(index.shape[0])
    if example_balanced:
        total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
        for example_id, example_total in example_totals.items():
            total = total + example_total / max(1, int(example_counts[example_id]))
        denom = len(example_totals)
    return total / max(1, denom)


def masked_union_topk_huber_loss(
    student_shifted_logits: torch.Tensor,
    teacher_shifted_logits: torch.Tensor,
    shifted_mask: torch.Tensor,
    *,
    temperature: float,
    chunk_tokens: int,
    teacher_top_k: int,
    student_top_k: int,
    huber_delta: float,
    center: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not bool(shifted_mask.any()):
        zero = student_shifted_logits.sum() * 0.0
        return zero, {
            "union_active_set_size_mean": 0.0,
            "union_active_set_size_max": 0.0,
            "teacher_top_k": int(teacher_top_k),
            "student_top_k": int(student_top_k),
            "huber_delta": float(huber_delta),
            "union_centered": int(bool(center)),
        }

    positions = shifted_mask.nonzero(as_tuple=False)
    temp = max(float(temperature), 1.0e-6)
    chunk = max(1, int(chunk_tokens))
    vocab = int(student_shifted_logits.shape[-1])
    teacher_top_k = min(max(1, int(teacher_top_k)), vocab)
    student_top_k = min(max(1, int(student_top_k)), vocab)
    delta = max(float(huber_delta), 1.0e-6)

    total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
    denom = 0
    active_sizes: list[int] = []
    for start in range(0, int(positions.shape[0]), chunk):
        index = positions[start : start + chunk]
        batch_index = index[:, 0]
        token_index = index[:, 1]
        teacher = teacher_shifted_logits[batch_index, token_index, :].float()
        student = student_shifted_logits[batch_index, token_index, :].float()

        with torch.no_grad():
            teacher_top_vals, teacher_top_ids = torch.topk(teacher, k=teacher_top_k, dim=-1)
            student_top_vals, student_top_ids = torch.topk(student, k=student_top_k, dim=-1)

        for row in range(int(index.shape[0])):
            union_ids = torch.unique(
                torch.cat([teacher_top_ids[row], student_top_ids[row]], dim=0),
                sorted=False,
            )
            active_sizes.append(int(union_ids.numel()))

            teacher_active = teacher[row].index_select(0, union_ids) / temp
            student_active = student[row].index_select(0, union_ids) / temp
            if center:
                teacher_active = teacher_active - teacher_active.mean()
                student_active = student_active - student_active.mean()
            token_loss = F.huber_loss(
                student_active,
                teacher_active,
                reduction="mean",
                delta=delta,
            )
            total = total + token_loss

        denom += int(index.shape[0])

    stats = {
        "union_active_set_size_mean": float(sum(active_sizes) / len(active_sizes)) if active_sizes else 0.0,
        "union_active_set_size_max": float(max(active_sizes)) if active_sizes else 0.0,
        "teacher_top_k": int(teacher_top_k),
        "student_top_k": int(student_top_k),
        "huber_delta": float(delta),
        "union_centered": int(bool(center)),
    }
    return total / max(1, denom), stats


def masked_jeffrey_topk_kl_loss(
    student_shifted_logits: torch.Tensor,
    teacher_shifted_logits: torch.Tensor,
    shifted_mask: torch.Tensor,
    *,
    temperature: float,
    chunk_tokens: int,
    teacher_top_k: int,
    student_top_k: int,
    reverse_kl_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not bool(shifted_mask.any()):
        zero = student_shifted_logits.sum() * 0.0
        return zero, {
            "jeffrey_forward_kl": 0.0,
            "jeffrey_reverse_kl": 0.0,
            "jeffrey_reverse_kl_weight": float(reverse_kl_weight),
            "teacher_top_k": int(teacher_top_k),
            "student_top_k": int(student_top_k),
            "union_active_set_size_mean": 0.0,
            "union_active_set_size_max": 0.0,
        }

    positions = shifted_mask.nonzero(as_tuple=False)
    temp = max(float(temperature), 1.0e-6)
    chunk = max(1, int(chunk_tokens))
    vocab = int(student_shifted_logits.shape[-1])
    teacher_top_k = min(max(1, int(teacher_top_k)), vocab)
    student_top_k = min(max(1, int(student_top_k)), vocab)
    beta = max(float(reverse_kl_weight), 0.0)

    total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
    forward_total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
    reverse_total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
    denom = 0
    active_sizes: list[int] = []

    for start in range(0, int(positions.shape[0]), chunk):
        index = positions[start : start + chunk]
        batch_index = index[:, 0]
        token_index = index[:, 1]
        teacher = teacher_shifted_logits[batch_index, token_index, :].float()
        student = student_shifted_logits[batch_index, token_index, :].float()

        with torch.no_grad():
            teacher_top_vals, teacher_top_ids = torch.topk(teacher, k=teacher_top_k, dim=-1)
            student_top_vals, student_top_ids = torch.topk(student, k=student_top_k, dim=-1)

        for row in range(int(index.shape[0])):
            union_ids = torch.unique(
                torch.cat([teacher_top_ids[row], student_top_ids[row]], dim=0),
                sorted=False,
            )
            active_sizes.append(int(union_ids.numel()))

            teacher_active = teacher[row].index_select(0, union_ids) / temp
            student_active = student[row].index_select(0, union_ids) / temp

            teacher_log_probs = F.log_softmax(teacher_active, dim=-1)
            student_log_probs = F.log_softmax(student_active, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            student_probs = student_log_probs.exp()

            forward_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum()
            reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum()
            token_loss = forward_kl + beta * reverse_kl

            forward_total = forward_total + forward_kl
            reverse_total = reverse_total + reverse_kl
            total = total + token_loss

        denom += int(index.shape[0])

    scale = temp * temp
    total = total * scale
    forward_total = forward_total * scale
    reverse_total = reverse_total * scale
    stats = {
        "jeffrey_forward_kl": float((forward_total / max(1, denom)).detach().cpu()),
        "jeffrey_reverse_kl": float((reverse_total / max(1, denom)).detach().cpu()),
        "jeffrey_reverse_kl_weight": float(beta),
        "teacher_top_k": int(teacher_top_k),
        "student_top_k": int(student_top_k),
        "union_active_set_size_mean": float(sum(active_sizes) / len(active_sizes)) if active_sizes else 0.0,
        "union_active_set_size_max": float(max(active_sizes)) if active_sizes else 0.0,
    }
    return total / max(1, denom), stats


def masked_endpoint_full_huber_loss(
    student_shifted_logits: torch.Tensor,
    teacher_shifted_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    answer_first_offsets: torch.Tensor,
    *,
    temperature: float,
    huber_delta: float,
    prompt_weight: float,
    answer_weight: float,
    vocab_reduction: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    temp = max(float(temperature), 1.0e-6)
    delta = max(float(huber_delta), 1.0e-6)
    prompt_weight = max(float(prompt_weight), 0.0)
    answer_weight = max(float(answer_weight), 0.0)
    vocab_reduction = str(vocab_reduction or "mean").strip().lower()
    if vocab_reduction not in {"mean", "sum"}:
        raise ValueError(f"unsupported endpoint_vocab_reduction={vocab_reduction!r}; use mean or sum")

    shifted_len = int(student_shifted_logits.shape[1])
    prompt_lens = prompt_lens.to(device=student_shifted_logits.device)
    answer_first_offsets = answer_first_offsets.to(device=student_shifted_logits.device)
    attention_mask = attention_mask.to(device=student_shifted_logits.device)

    selected: list[tuple[int, int, str, float]] = []
    for row_idx in range(int(prompt_lens.shape[0])):
        prompt_len = int(prompt_lens[row_idx].item())
        prompt_pos = prompt_len - 1
        if 0 <= prompt_pos < shifted_len and prompt_len < int(attention_mask.shape[1]) and bool(attention_mask[row_idx, prompt_len].item()):
            selected.append((row_idx, prompt_pos, "prompt", prompt_weight))

        answer_offset = int(answer_first_offsets[row_idx].item())
        if answer_offset >= 0:
            answer_token_pos = prompt_len + answer_offset
            answer_shift_pos = answer_token_pos - 1
            if (
                0 <= answer_shift_pos < shifted_len
                and answer_token_pos < int(attention_mask.shape[1])
                and bool(attention_mask[row_idx, answer_token_pos].item())
            ):
                selected.append((row_idx, answer_shift_pos, "answer", answer_weight))

    selected = [item for item in selected if item[3] > 0.0]
    if not selected:
        zero = student_shifted_logits.sum() * 0.0
        return zero, {
            "endpoint_prompt_count": 0,
            "endpoint_answer_count": 0,
            "endpoint_prompt_huber": 0.0,
            "endpoint_answer_huber": 0.0,
            "endpoint_vocab_reduction": vocab_reduction,
            "huber_delta": float(delta),
        }

    batch_indices = torch.tensor([item[0] for item in selected], dtype=torch.long, device=student_shifted_logits.device)
    token_indices = torch.tensor([item[1] for item in selected], dtype=torch.long, device=student_shifted_logits.device)
    weights = torch.tensor([item[3] for item in selected], dtype=torch.float32, device=student_shifted_logits.device)
    labels = [item[2] for item in selected]

    teacher = teacher_shifted_logits[batch_indices, token_indices, :].float() / temp
    student = student_shifted_logits[batch_indices, token_indices, :].float() / temp
    per_vocab = F.huber_loss(student, teacher, reduction="none", delta=delta)
    if vocab_reduction == "sum":
        per_position = per_vocab.sum(dim=-1)
    else:
        per_position = per_vocab.mean(dim=-1)

    weighted = per_position * weights
    loss = weighted.sum() / weights.sum().clamp_min(1.0e-6)

    prompt_values = [float(per_position[idx].detach().cpu()) for idx, label in enumerate(labels) if label == "prompt"]
    answer_values = [float(per_position[idx].detach().cpu()) for idx, label in enumerate(labels) if label == "answer"]
    stats = {
        "endpoint_prompt_count": int(len(prompt_values)),
        "endpoint_answer_count": int(len(answer_values)),
        "endpoint_prompt_huber": float(sum(prompt_values) / len(prompt_values)) if prompt_values else 0.0,
        "endpoint_answer_huber": float(sum(answer_values) / len(answer_values)) if answer_values else 0.0,
        "endpoint_vocab_reduction": vocab_reduction,
        "endpoint_prompt_weight": float(prompt_weight),
        "endpoint_answer_weight": float(answer_weight),
        "huber_delta": float(delta),
    }
    return loss, stats


def masked_endpoint_topk_huber_loss(
    student_shifted_logits: torch.Tensor,
    teacher_shifted_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    answer_first_offsets: torch.Tensor,
    *,
    temperature: float,
    teacher_top_k: int,
    huber_delta: float,
    prompt_weight: float,
    answer_weight: float,
    center: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    temp = max(float(temperature), 1.0e-6)
    delta = max(float(huber_delta), 1.0e-6)
    prompt_weight = max(float(prompt_weight), 0.0)
    answer_weight = max(float(answer_weight), 0.0)
    vocab = int(student_shifted_logits.shape[-1])
    teacher_top_k = min(max(1, int(teacher_top_k)), vocab)

    shifted_len = int(student_shifted_logits.shape[1])
    prompt_lens = prompt_lens.to(device=student_shifted_logits.device)
    answer_first_offsets = answer_first_offsets.to(device=student_shifted_logits.device)
    attention_mask = attention_mask.to(device=student_shifted_logits.device)

    selected: list[tuple[int, int, str, float]] = []
    for row_idx in range(int(prompt_lens.shape[0])):
        prompt_len = int(prompt_lens[row_idx].item())
        prompt_pos = prompt_len - 1
        if 0 <= prompt_pos < shifted_len and prompt_len < int(attention_mask.shape[1]) and bool(attention_mask[row_idx, prompt_len].item()):
            selected.append((row_idx, prompt_pos, "prompt", prompt_weight))

        answer_offset = int(answer_first_offsets[row_idx].item())
        if answer_offset >= 0:
            answer_token_pos = prompt_len + answer_offset
            answer_shift_pos = answer_token_pos - 1
            if (
                0 <= answer_shift_pos < shifted_len
                and answer_token_pos < int(attention_mask.shape[1])
                and bool(attention_mask[row_idx, answer_token_pos].item())
            ):
                selected.append((row_idx, answer_shift_pos, "answer", answer_weight))

    selected = [item for item in selected if item[3] > 0.0]
    if not selected:
        zero = student_shifted_logits.sum() * 0.0
        return zero, {
            "endpoint_prompt_count": 0,
            "endpoint_answer_count": 0,
            "endpoint_prompt_huber": 0.0,
            "endpoint_answer_huber": 0.0,
            "endpoint_top_k": int(teacher_top_k),
            "endpoint_topk_centered": int(bool(center)),
            "huber_delta": float(delta),
        }

    batch_indices = torch.tensor([item[0] for item in selected], dtype=torch.long, device=student_shifted_logits.device)
    token_indices = torch.tensor([item[1] for item in selected], dtype=torch.long, device=student_shifted_logits.device)
    weights = torch.tensor([item[3] for item in selected], dtype=torch.float32, device=student_shifted_logits.device)
    labels = [item[2] for item in selected]

    teacher = teacher_shifted_logits[batch_indices, token_indices, :].float() / temp
    student = student_shifted_logits[batch_indices, token_indices, :].float() / temp
    with torch.no_grad():
        _top_vals, top_ids = torch.topk(teacher, k=teacher_top_k, dim=-1)
    teacher_active = torch.gather(teacher, dim=-1, index=top_ids)
    student_active = torch.gather(student, dim=-1, index=top_ids)
    if center:
        teacher_active = teacher_active - teacher_active.mean(dim=-1, keepdim=True)
        student_active = student_active - student_active.mean(dim=-1, keepdim=True)

    per_position = F.huber_loss(student_active, teacher_active, reduction="none", delta=delta).mean(dim=-1)
    weighted = per_position * weights
    loss = weighted.sum() / weights.sum().clamp_min(1.0e-6)

    prompt_values = [float(per_position[idx].detach().cpu()) for idx, label in enumerate(labels) if label == "prompt"]
    answer_values = [float(per_position[idx].detach().cpu()) for idx, label in enumerate(labels) if label == "answer"]
    stats = {
        "endpoint_prompt_count": int(len(prompt_values)),
        "endpoint_answer_count": int(len(answer_values)),
        "endpoint_prompt_huber": float(sum(prompt_values) / len(prompt_values)) if prompt_values else 0.0,
        "endpoint_answer_huber": float(sum(answer_values) / len(answer_values)) if answer_values else 0.0,
        "endpoint_top_k": int(teacher_top_k),
        "endpoint_topk_centered": int(bool(center)),
        "endpoint_prompt_weight": float(prompt_weight),
        "endpoint_answer_weight": float(answer_weight),
        "huber_delta": float(delta),
    }
    return loss, stats


def masked_prefix_shift_huber_loss(
    student_shifted_logits: torch.Tensor,
    teacher_shifted_logits: torch.Tensor,
    base_shifted_logits: torch.Tensor,
    shifted_mask: torch.Tensor,
    *,
    temperature: float,
    chunk_tokens: int,
    teacher_top_k: int,
    student_top_k: int,
    huber_delta: float,
    center: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    temp = max(float(temperature), 1.0e-6)
    delta = max(float(huber_delta), 1.0e-6)
    vocab = int(student_shifted_logits.shape[-1])
    teacher_top_k = min(max(1, int(teacher_top_k)), vocab)
    student_top_k = min(max(1, int(student_top_k)), vocab)

    if not bool(shifted_mask.any()):
        zero = student_shifted_logits.sum() * 0.0
        return zero, {
            "prefix_shift_active_set_size_mean": 0.0,
            "prefix_shift_active_set_size_max": 0.0,
            "prefix_shift_teacher_top_k": int(teacher_top_k),
            "prefix_shift_student_top_k": int(student_top_k),
            "prefix_shift_centered": int(bool(center)),
            "huber_delta": float(delta),
        }

    positions = shifted_mask.nonzero(as_tuple=False)
    chunk = max(1, int(chunk_tokens))

    total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
    denom = 0
    active_sizes: list[int] = []

    for start in range(0, int(positions.shape[0]), chunk):
        index = positions[start : start + chunk]
        batch_index = index[:, 0]
        token_index = index[:, 1]

        teacher = teacher_shifted_logits[batch_index, token_index, :].float()
        base = base_shifted_logits[batch_index, token_index, :].float()
        student = student_shifted_logits[batch_index, token_index, :].float()
        teacher_delta = (teacher - base) / temp
        student_delta = (student - base) / temp

        with torch.no_grad():
            teacher_top_vals, teacher_top_ids = torch.topk(teacher, k=teacher_top_k, dim=-1)
            base_top_vals, base_top_ids = torch.topk(base, k=teacher_top_k, dim=-1)
            teacher_delta_top_vals, teacher_delta_top_ids = torch.topk(teacher_delta.abs(), k=teacher_top_k, dim=-1)
            student_delta_top_vals, student_delta_top_ids = torch.topk(student_delta.abs(), k=student_top_k, dim=-1)

        for row in range(int(index.shape[0])):
            union_ids = torch.unique(
                torch.cat(
                    [
                        teacher_top_ids[row],
                        base_top_ids[row],
                        teacher_delta_top_ids[row],
                        student_delta_top_ids[row],
                    ],
                    dim=0,
                ),
                sorted=False,
            )
            active_sizes.append(int(union_ids.numel()))

            teacher_active = teacher_delta[row].index_select(0, union_ids)
            student_active = student_delta[row].index_select(0, union_ids)
            if center:
                teacher_active = teacher_active - teacher_active.mean()
                student_active = student_active - student_active.mean()
            token_loss = F.huber_loss(
                student_active,
                teacher_active,
                reduction="mean",
                delta=delta,
            )
            total = total + token_loss

        denom += int(index.shape[0])

    stats = {
        "prefix_shift_active_set_size_mean": float(sum(active_sizes) / len(active_sizes)) if active_sizes else 0.0,
        "prefix_shift_active_set_size_max": float(max(active_sizes)) if active_sizes else 0.0,
        "prefix_shift_teacher_top_k": int(teacher_top_k),
        "prefix_shift_student_top_k": int(student_top_k),
        "prefix_shift_centered": int(bool(center)),
        "huber_delta": float(delta),
    }
    return total / max(1, denom), stats


def masked_probability_shift_huber_loss(
    student_shifted_logits: torch.Tensor,
    teacher_shifted_logits: torch.Tensor,
    base_shifted_logits: torch.Tensor,
    shifted_mask: torch.Tensor,
    *,
    temperature: float,
    chunk_tokens: int,
    top_k: int,
    huber_delta: float,
    state_reduction: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    temp = max(float(temperature), 1.0e-6)
    delta = max(float(huber_delta), 1.0e-6)
    reduction = str(state_reduction or "token").strip().lower().replace("-", "_")
    if reduction not in {"token", "example", "state", "sequence"}:
        raise ValueError(f"unsupported state_reduction={state_reduction!r}; use token or example")
    example_balanced = reduction in {"example", "state", "sequence"}
    vocab = int(student_shifted_logits.shape[-1])
    top_k = min(max(1, int(top_k)), vocab)

    if not bool(shifted_mask.any()):
        zero = student_shifted_logits.sum() * 0.0
        return zero, {
            "probability_shift_active_set_size_mean": 0.0,
            "probability_shift_active_set_size_max": 0.0,
            "probability_shift_top_k": int(top_k),
            "huber_delta": float(delta),
            "state_reduction": reduction,
            "loss_examples": 0,
        }

    positions = shifted_mask.nonzero(as_tuple=False)
    chunk = max(1, int(chunk_tokens))

    total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
    denom = 0
    active_sizes: list[int] = []
    example_totals: dict[int, torch.Tensor] = {}
    example_counts: dict[int, int] = {}

    for start in range(0, int(positions.shape[0]), chunk):
        index = positions[start : start + chunk]
        batch_index = index[:, 0]
        token_index = index[:, 1]

        teacher_logits = teacher_shifted_logits[batch_index, token_index, :].float() / temp
        base_logits = base_shifted_logits[batch_index, token_index, :].float() / temp
        student_logits = student_shifted_logits[batch_index, token_index, :].float() / temp

        teacher_logp = F.log_softmax(teacher_logits, dim=-1)
        base_logp = F.log_softmax(base_logits, dim=-1)
        student_logp = F.log_softmax(student_logits, dim=-1)
        teacher_delta = teacher_logp - base_logp
        student_delta = student_logp - base_logp

        with torch.no_grad():
            _base_top_vals, base_top_ids = torch.topk(base_logp, k=top_k, dim=-1)
            _teacher_top_vals, teacher_top_ids = torch.topk(teacher_logp, k=top_k, dim=-1)
            _delta_top_vals, teacher_delta_top_ids = torch.topk(teacher_delta.abs(), k=top_k, dim=-1)

        for row in range(int(index.shape[0])):
            union_ids = torch.unique(
                torch.cat(
                    [
                        base_top_ids[row],
                        teacher_top_ids[row],
                        teacher_delta_top_ids[row],
                    ],
                    dim=0,
                ),
                sorted=False,
            )
            active_sizes.append(int(union_ids.numel()))

            teacher_active = teacher_delta[row].index_select(0, union_ids)
            student_active = student_delta[row].index_select(0, union_ids)
            token_loss = F.huber_loss(
                student_active,
                teacher_active,
                reduction="mean",
                delta=delta,
            )
            if example_balanced:
                example_id = int(batch_index[row].item())
                if example_id not in example_totals:
                    example_totals[example_id] = token_loss
                    example_counts[example_id] = 1
                else:
                    example_totals[example_id] = example_totals[example_id] + token_loss
                    example_counts[example_id] += 1
            else:
                total = total + token_loss

        denom += int(index.shape[0])

    if example_balanced:
        total = torch.zeros((), dtype=torch.float32, device=student_shifted_logits.device)
        for example_id, example_total in example_totals.items():
            total = total + example_total / max(1, int(example_counts[example_id]))
        denom = len(example_totals)

    stats = {
        "probability_shift_active_set_size_mean": float(sum(active_sizes) / len(active_sizes)) if active_sizes else 0.0,
        "probability_shift_active_set_size_max": float(max(active_sizes)) if active_sizes else 0.0,
        "probability_shift_top_k": int(top_k),
        "huber_delta": float(delta),
        "state_reduction": reduction,
        "loss_examples": int(len(example_totals)) if example_balanced else int(shifted_mask.shape[0]),
    }
    return total / max(1, denom), stats


def raw_answer_unlikelihood_loss(
    model,
    batch: dict[str, Any],
    *,
    device: torch.device,
    eps: float,
    reduction: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    answer_starts = batch["answer_starts"].to(device)
    answer_lens = batch["answer_lens"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits.float()
    epsilon = min(0.1, max(float(eps), 1.0e-8))
    reduction = str(reduction or "example").strip().lower().replace("-", "_")
    if reduction not in {"token", "example", "state", "sequence"}:
        raise ValueError(f"unsupported answer_erasure_reduction={reduction!r}; use token or example")
    example_balanced = reduction in {"example", "state", "sequence"}

    total = logits.new_zeros(())
    token_total = logits.new_zeros(())
    nll_total = logits.new_zeros(())
    prob_total = logits.new_zeros(())
    example_count = 0
    token_count = 0
    answer_counts: list[float] = []

    for row_idx in range(int(input_ids.shape[0])):
        start = int(answer_starts[row_idx].item())
        length = int(answer_lens[row_idx].item())
        if length <= 0:
            continue
        positions = torch.arange(start, start + length, device=device, dtype=torch.long)
        valid = positions > 0
        positions = positions[valid]
        if int(positions.numel()) <= 0:
            continue
        pred_positions = positions - 1
        target_ids = input_ids[row_idx, positions]
        logp = F.log_softmax(logits[row_idx, pred_positions, :], dim=-1)
        target_logp = torch.gather(logp, dim=-1, index=target_ids.view(-1, 1)).squeeze(-1)
        target_prob = target_logp.exp().clamp(min=epsilon, max=1.0 - epsilon)
        token_loss = -torch.log1p(-target_prob)
        example_loss = token_loss.mean()
        if example_balanced:
            total = total + example_loss
        else:
            total = total + token_loss.sum()
        token_total = token_total + token_loss.sum()
        nll_total = nll_total + (-target_logp).sum()
        prob_total = prob_total + target_prob.sum()
        example_count += 1
        token_count += int(positions.numel())
        answer_counts.append(float(positions.numel()))

    denom = max(1, example_count if example_balanced else token_count)
    loss = total / float(denom)
    token_denom = max(1, token_count)
    metrics = {
        "answer_erasure_loss": float(loss.detach().cpu()),
        "answer_erasure_ul_token_mean": float((token_total / float(token_denom)).detach().cpu()),
        "answer_erasure_nll": float((nll_total / float(token_denom)).detach().cpu()),
        "answer_erasure_prob": float((prob_total / float(token_denom)).detach().cpu()),
        "answer_erasure_examples": int(example_count),
        "answer_erasure_tokens": int(token_count),
        "answer_erasure_answer_tokens_mean": float(sum(answer_counts) / max(1, len(answer_counts))),
        "answer_erasure_reduction": reduction,
    }
    return loss, metrics


def latent_shift_alignment_loss(
    model,
    batch: dict[str, Any],
    prefix: SoftPrefix,
    *,
    device: torch.device,
    loss_type: str,
    huber_delta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)
    prefix_insert_lens = batch.get("prefix_insert_lens", prompt_lens).to(device)

    student_h = forward_root_hidden(model, input_ids, attention_mask, None, prompt_lens, prefix_insert_lens).float()
    with torch.no_grad(), adapter_disabled(model):
        base_h = forward_root_hidden(model, input_ids, attention_mask, None, prompt_lens, prefix_insert_lens).detach().float()
        teacher_h = forward_root_hidden(model, input_ids, attention_mask, prefix, prompt_lens, prefix_insert_lens).detach().float()

    teacher_delta = teacher_h - base_h
    student_delta = student_h - base_h
    loss_name = str(loss_type or "mse").strip().lower().replace("-", "_")
    if loss_name == "cos":
        loss_name = "cosine"
    if loss_name == "mse":
        loss = F.mse_loss(student_delta, teacher_delta, reduction="mean")
    elif loss_name == "huber":
        loss = F.huber_loss(
            student_delta,
            teacher_delta,
            reduction="mean",
            delta=max(float(huber_delta), 1.0e-6),
        )
    elif loss_name == "cosine":
        loss = (1.0 - F.cosine_similarity(student_delta, teacher_delta, dim=-1, eps=1.0e-8)).mean()
    else:
        raise ValueError(f"unsupported latent_shift_loss_type={loss_type!r}; use mse, huber, or cosine")

    with torch.no_grad():
        teacher_norm = teacher_delta.norm(dim=-1)
        student_norm = student_delta.norm(dim=-1)
        base_teacher_cos = F.cosine_similarity(base_h, teacher_h, dim=-1, eps=1.0e-8)
        delta_cos = F.cosine_similarity(student_delta, teacher_delta, dim=-1, eps=1.0e-8)
        delta_mse = F.mse_loss(student_delta, teacher_delta, reduction="none").mean(dim=-1)
    metrics = {
        "latent_shift_loss": float(loss.detach().cpu()),
        "latent_shift_loss_type": loss_name,
        "latent_shift_examples": int(input_ids.shape[0]),
        "latent_shift_teacher_norm_mean": float(teacher_norm.mean().detach().cpu()),
        "latent_shift_teacher_norm_median": float(teacher_norm.median().detach().cpu()),
        "latent_shift_student_norm_mean": float(student_norm.mean().detach().cpu()),
        "latent_shift_delta_cos_mean": float(delta_cos.mean().detach().cpu()),
        "latent_shift_delta_cos_median": float(delta_cos.median().detach().cpu()),
        "latent_shift_delta_mse_mean": float(delta_mse.mean().detach().cpu()),
        "latent_shift_base_teacher_cos_mean": float(base_teacher_cos.mean().detach().cpu()),
    }
    return loss, metrics


def compute_fulltraj_kl_loss(
    model,
    batch: dict[str, Any],
    prefix: SoftPrefix,
    split: str,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)
    prefix_insert_lens = batch.get("prefix_insert_lens", prompt_lens).to(device)
    completion_lens = batch["completion_lens"].to(device)
    answer_first_offsets = batch.get("answer_first_offsets", torch.full_like(prompt_lens, -1)).to(device)
    completion_kl_masks = batch.get("completion_kl_masks")
    teacher_prefix = prefix if split == "forget" or (split == "retain" and bool(args.retain_teacher_uses_prefix)) else None
    distill_objective = normalize_distill_objective(getattr(args, "distill_objective", "trajectory_kl"))
    split_state_reduction = str(
        getattr(args, f"{split}_state_loss_reduction", "") or getattr(args, "state_loss_reduction", "token")
    )

    student_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    student_shifted = student_out.logits[:, :-1, :]
    completion_mask = completion_shift_mask(attention_mask, prompt_lens)
    shifted_mask = apply_completion_kl_mask(completion_mask, prompt_lens, completion_kl_masks)

    with torch.no_grad(), adapter_disabled(model):
        if distill_objective in {"prefix_shift_huber", "probability_shift_huber"}:
            base_shifted = forward_shifted_logits(model, input_ids, attention_mask, None, prompt_lens, prefix_insert_lens).detach()
            if teacher_prefix is None:
                teacher_shifted = base_shifted
            else:
                teacher_shifted = forward_shifted_logits(
                    model,
                    input_ids,
                    attention_mask,
                    teacher_prefix,
                    prompt_lens,
                    prefix_insert_lens,
                ).detach()
        else:
            base_shifted = None
            teacher_shifted = forward_shifted_logits(
                model,
                input_ids,
                attention_mask,
                teacher_prefix,
                prompt_lens,
                prefix_insert_lens,
            ).detach()

    if teacher_shifted.shape[:2] != student_shifted.shape[:2]:
        raise RuntimeError(f"teacher/student shifted shape mismatch: {teacher_shifted.shape} vs {student_shifted.shape}")
    if base_shifted is not None and base_shifted.shape[:2] != student_shifted.shape[:2]:
        raise RuntimeError(f"base/student shifted shape mismatch: {base_shifted.shape} vs {student_shifted.shape}")

    if distill_objective == "trajectory_kl":
        loss = masked_trajectory_kl(
            student_shifted,
            teacher_shifted,
            shifted_mask,
            temperature=float(args.kl_temperature),
            chunk_tokens=int(args.kl_chunk_tokens),
            top_k=int(args.kl_top_k),
            state_reduction=split_state_reduction,
        )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "trajectory_kl": float(loss.detach().cpu()),
            "tokens": int(shifted_mask.sum().detach().cpu()),
            "completion_tokens": int(completion_mask.sum().detach().cpu()),
            "kl_token_scope": str(args.kl_token_scope),
            "kl_cot_fraction": float(args.kl_cot_fraction),
            "completion_tokens_mean": float(completion_lens.float().mean().detach().cpu()),
            "teacher_prefix_used": int(teacher_prefix is not None),
            "distill_objective": "trajectory_kl",
            "state_loss_reduction": split_state_reduction,
        }
    else:
        student_top_k = int(getattr(args, "student_top_k", int(args.kl_top_k)))
        if distill_objective == "union_topk_huber":
            loss, union_stats = masked_union_topk_huber_loss(
                student_shifted,
                teacher_shifted,
                shifted_mask,
                temperature=float(args.kl_temperature),
                chunk_tokens=int(args.kl_chunk_tokens),
                teacher_top_k=int(args.kl_top_k),
                student_top_k=student_top_k,
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
                center=bool(getattr(args, "union_center", True)),
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "union_topk_huber": float(loss.detach().cpu()),
                "tokens": int(shifted_mask.sum().detach().cpu()),
                "completion_tokens": int(completion_mask.sum().detach().cpu()),
                "kl_token_scope": str(args.kl_token_scope),
                "kl_cot_fraction": float(args.kl_cot_fraction),
                "completion_tokens_mean": float(completion_lens.float().mean().detach().cpu()),
                "teacher_prefix_used": int(teacher_prefix is not None),
                "distill_objective": "union_topk_huber",
                **union_stats,
            }
        elif distill_objective == "jeffrey_topk_kl":
            loss, jeffrey_stats = masked_jeffrey_topk_kl_loss(
                student_shifted,
                teacher_shifted,
                shifted_mask,
                temperature=float(args.kl_temperature),
                chunk_tokens=int(args.kl_chunk_tokens),
                teacher_top_k=int(args.kl_top_k),
                student_top_k=student_top_k,
                reverse_kl_weight=float(getattr(args, "reverse_kl_weight", 1.0)),
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "jeffrey_topk_kl": float(loss.detach().cpu()),
                "tokens": int(shifted_mask.sum().detach().cpu()),
                "completion_tokens": int(completion_mask.sum().detach().cpu()),
                "kl_token_scope": str(args.kl_token_scope),
                "kl_cot_fraction": float(args.kl_cot_fraction),
                "completion_tokens_mean": float(completion_lens.float().mean().detach().cpu()),
                "teacher_prefix_used": int(teacher_prefix is not None),
                "distill_objective": "jeffrey_topk_kl",
                **jeffrey_stats,
            }
        elif distill_objective == "prefix_shift_huber":
            if base_shifted is None:
                raise RuntimeError("prefix_shift_huber requires base logits")
            loss, shift_stats = masked_prefix_shift_huber_loss(
                student_shifted,
                teacher_shifted,
                base_shifted,
                shifted_mask,
                temperature=float(args.kl_temperature),
                chunk_tokens=int(args.kl_chunk_tokens),
                teacher_top_k=int(args.kl_top_k),
                student_top_k=student_top_k,
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
                center=bool(getattr(args, "union_center", True)),
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "prefix_shift_huber": float(loss.detach().cpu()),
                "tokens": int(shifted_mask.sum().detach().cpu()),
                "completion_tokens": int(completion_mask.sum().detach().cpu()),
                "kl_token_scope": str(args.kl_token_scope),
                "kl_cot_fraction": float(args.kl_cot_fraction),
                "completion_tokens_mean": float(completion_lens.float().mean().detach().cpu()),
                "teacher_prefix_used": int(teacher_prefix is not None),
                "distill_objective": "prefix_shift_huber",
                **shift_stats,
            }
        elif distill_objective == "probability_shift_huber":
            if base_shifted is None:
                raise RuntimeError("probability_shift_huber requires base logits")
            loss, prob_stats = masked_probability_shift_huber_loss(
                student_shifted,
                teacher_shifted,
                base_shifted,
                shifted_mask,
                temperature=float(args.kl_temperature),
                chunk_tokens=int(args.kl_chunk_tokens),
                top_k=int(args.kl_top_k),
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
                state_reduction=split_state_reduction,
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "probability_shift_huber": float(loss.detach().cpu()),
                "tokens": int(shifted_mask.sum().detach().cpu()),
                "completion_tokens": int(completion_mask.sum().detach().cpu()),
                "kl_token_scope": str(args.kl_token_scope),
                "kl_cot_fraction": float(args.kl_cot_fraction),
                "completion_tokens_mean": float(completion_lens.float().mean().detach().cpu()),
                "teacher_prefix_used": int(teacher_prefix is not None),
                "distill_objective": "probability_shift_huber",
                **prob_stats,
            }
        elif distill_objective == "endpoint_topk_huber":
            endpoint_top_k = int(getattr(args, "endpoint_top_k", 0))
            if endpoint_top_k <= 0:
                endpoint_top_k = int(args.kl_top_k)
            loss, endpoint_stats = masked_endpoint_topk_huber_loss(
                student_shifted,
                teacher_shifted,
                attention_mask,
                prompt_lens,
                answer_first_offsets,
                temperature=float(args.kl_temperature),
                teacher_top_k=endpoint_top_k,
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
                prompt_weight=float(getattr(args, "endpoint_prompt_weight", 1.0)),
                answer_weight=float(getattr(args, "endpoint_answer_weight", 1.0)),
                center=bool(getattr(args, "endpoint_topk_center", False)),
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "endpoint_topk_huber": float(loss.detach().cpu()),
                "tokens": int(endpoint_stats["endpoint_prompt_count"] + endpoint_stats["endpoint_answer_count"]),
                "completion_tokens": int(completion_mask.sum().detach().cpu()),
                "kl_token_scope": "endpoint_prompt_and_answer_first",
                "kl_cot_fraction": 0.0,
                "completion_tokens_mean": float(completion_lens.float().mean().detach().cpu()),
                "teacher_prefix_used": int(teacher_prefix is not None),
                "distill_objective": "endpoint_topk_huber",
                **endpoint_stats,
            }
        else:
            loss, endpoint_stats = masked_endpoint_full_huber_loss(
                student_shifted,
                teacher_shifted,
                attention_mask,
                prompt_lens,
                answer_first_offsets,
                temperature=float(args.kl_temperature),
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
                prompt_weight=float(getattr(args, "endpoint_prompt_weight", 1.0)),
                answer_weight=float(getattr(args, "endpoint_answer_weight", 1.0)),
                vocab_reduction=str(getattr(args, "endpoint_vocab_reduction", "mean")),
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "endpoint_full_huber": float(loss.detach().cpu()),
                "tokens": int(endpoint_stats["endpoint_prompt_count"] + endpoint_stats["endpoint_answer_count"]),
                "completion_tokens": int(completion_mask.sum().detach().cpu()),
                "kl_token_scope": "endpoint_prompt_and_answer_first",
                "kl_cot_fraction": 0.0,
                "completion_tokens_mean": float(completion_lens.float().mean().detach().cpu()),
                "teacher_prefix_used": int(teacher_prefix is not None),
                "distill_objective": "endpoint_full_huber",
                **endpoint_stats,
            }
    return loss, metrics


def build_scheduler(max_steps: int, forget_updates: int, retain_updates: int) -> list[str]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if forget_updates < 0 or retain_updates < 0:
        raise ValueError("update counts per cycle must be non-negative")
    cycle = ["forget"] * int(forget_updates) + ["retain"] * int(retain_updates)
    if not cycle:
        raise ValueError("at least one forget or retain update per cycle is required")
    return [cycle[idx % len(cycle)] for idx in range(int(max_steps))]


def save_adapter(model, tokenizer, output_dir: Path, args: argparse.Namespace, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_json(output_dir / "guard_ard_state.json", {"step": int(step), "args": vars(args)})


def parse_epoch_list(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).replace(";", ",").split(",")
    epochs: set[int] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        epochs.add(int(text))
    return epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUARD ARD answer-reasoning LoRA distillation.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")

    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--forget_path", type=str, default=str(DEFAULT_FORGET))
    parser.add_argument("--retain_path", type=str, default=str(DEFAULT_RETAIN))
    parser.add_argument("--retain_extra_forget_path", type=str, default="")
    parser.add_argument("--retain_extra_exclude_task_id", type=str, default="")
    parser.add_argument(
        "--forget_trajectory_mode",
        type=str,
        default="generated",
        choices=["generated", "fixed_safe", "safe_fixed", "dual_fixed"],
    )
    parser.add_argument("--add_root_states", type=str2bool, default=False)
    parser.add_argument("--root_state_completion", type=str, default=".")
    parser.add_argument("--root_state_templates", type=str, default="assistant")
    parser.add_argument("--prefix_path", type=str, default=str(DEFAULT_PREFIX))
    parser.add_argument("--prefix_position", type=str, default="prompt_end")
    parser.add_argument("--retain_teacher_uses_prefix", type=str2bool, default=True)
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="forget01_task1_stage2_fulltraj_kl_lora")
    parser.add_argument("--trajectory_cache", type=str, default="")
    parser.add_argument("--force_rebuild_trajectories", type=str2bool, default=False)

    parser.add_argument("--task_id", type=str, default="1")
    parser.add_argument("--forget_limit", type=int, default=40)
    parser.add_argument("--retain_limit", type=int, default=1200)
    parser.add_argument("--retain_sample", type=str, default="random", choices=["head", "random"])
    parser.add_argument("--max_length", type=int, default=2304)
    parser.add_argument("--skip_too_long", type=str2bool, default=False)

    parser.add_argument("--question_start", type=str, default="<｜User｜>")
    parser.add_argument("--question_end", type=str, default="<｜Assistant｜>")
    parser.add_argument("--think_start", type=str, default="<think>\n")
    parser.add_argument("--think_end", type=str, default="\n</think>\n\n")
    parser.add_argument("--answer_tag", type=str, default="")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="", choices=["", "eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")

    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps_per_epoch", type=int, default=40)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--forget_updates_per_cycle", type=int, default=1)
    parser.add_argument("--retain_updates_per_cycle", type=int, default=1)
    parser.add_argument("--forget_samples_per_epoch", type=int, default=0)
    parser.add_argument("--retain_samples_per_epoch", type=int, default=40)
    parser.add_argument("--retain_epoch_sample_seed", type=int, default=1707)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--forget_max_new_tokens", type=int, default=2048)
    parser.add_argument("--retain_max_new_tokens", type=int, default=2048)
    parser.add_argument("--trajectory_generation_batch_size", type=int, default=20)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--kl_chunk_tokens", type=int, default=8)
    parser.add_argument("--kl_top_k", type=int, default=1000, help="Teacher top-k used for trajectory KL.")
    parser.add_argument("--kl_token_scope", type=str, default="full_trajectory")
    parser.add_argument("--kl_cot_fraction", type=float, default=1.0)
    parser.add_argument("--distill_objective", type=str, default="trajectory_kl")
    parser.add_argument("--student_top_k", type=int, default=1000)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--state_loss_reduction", type=str, default="token")
    parser.add_argument("--forget_state_loss_reduction", type=str, default="")
    parser.add_argument("--retain_state_loss_reduction", type=str, default="")
    parser.add_argument("--union_center", type=str2bool, default=True)
    parser.add_argument("--reverse_kl_weight", type=float, default=1.0)
    parser.add_argument("--endpoint_prompt_weight", type=float, default=1.0)
    parser.add_argument("--endpoint_answer_weight", type=float, default=1.0)
    parser.add_argument("--endpoint_top_k", type=int, default=0)
    parser.add_argument("--endpoint_topk_center", type=str2bool, default=False)
    parser.add_argument("--endpoint_vocab_reduction", type=str, default="mean")
    parser.add_argument("--lambda_answer_erasure", type=float, default=0.0)
    parser.add_argument("--answer_erasure_prompt_templates", type=str, default="assistant")
    parser.add_argument("--answer_erasure_samples_per_epoch", type=int, default=0)
    parser.add_argument("--answer_erasure_batch_size", type=int, default=0)
    parser.add_argument("--answer_erasure_reduction", type=str, default="example")
    parser.add_argument("--answer_erasure_eps", type=float, default=1.0e-6)
    parser.add_argument("--lambda_latent_shift", type=float, default=0.0)
    parser.add_argument("--latent_shift_prompt_templates", type=str, default="assistant")
    parser.add_argument("--latent_shift_samples_per_epoch", type=int, default=0)
    parser.add_argument("--latent_shift_batch_size", type=int, default=0)
    parser.add_argument("--latent_shift_loss_type", type=str, default="mse")
    parser.add_argument("--latent_shift_huber_delta", type=float, default=1.0)

    parser.add_argument("--eval_seed", type=int, default=20260513)
    parser.add_argument("--eval_forget_samples", type=int, default=10)
    parser.add_argument("--eval_retain_samples", type=int, default=10)
    parser.add_argument("--eval_max_new_tokens", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=10)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_epochs", type=str, default="")
    parser.add_argument("--dry_run", type=str2bool, default=False)

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        parser.set_defaults(**load_config(pre_args.config))
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> tuple[Path, str]:
    stamp = str(args.stamp or time.strftime("%Y%m%d_%H%M%S"))
    if str(args.output_dir).strip():
        return Path(args.output_dir), stamp
    return Path(args.output_root) / f"{stamp}_{args.run_name}", stamp


def main() -> None:
    args = parse_args()
    args.prefix_position = normalize_prefix_position(args.prefix_position)
    args.kl_token_scope = normalize_kl_token_scope(args.kl_token_scope)
    args.distill_objective = normalize_distill_objective(args.distill_objective)
    if float(args.kl_cot_fraction) <= 0.0:
        raise ValueError("kl_cot_fraction must be positive")
    if args.kl_token_scope != "cot_prefix":
        args.kl_cot_fraction = 1.0
    else:
        args.kl_cot_fraction = min(1.0, float(args.kl_cot_fraction))
    if not bool(args.retain_teacher_uses_prefix):
        log(
            "retain_teacher_uses_prefix=false is an ablation setting; "
            "the GUARD ARD methodology uses the optimized guidance-conditioned teacher for both forget and retain streams."
        )
    if args.distill_objective == "trajectory_kl":
        args.student_top_k = int(args.student_top_k)
        args.huber_delta = float(args.huber_delta)
        args.reverse_kl_weight = float(args.reverse_kl_weight)
        args.endpoint_prompt_weight = float(args.endpoint_prompt_weight)
        args.endpoint_answer_weight = float(args.endpoint_answer_weight)
        args.endpoint_top_k = int(args.endpoint_top_k)
        args.endpoint_topk_center = bool(args.endpoint_topk_center)
        args.endpoint_vocab_reduction = str(args.endpoint_vocab_reduction)
    else:
        if int(args.student_top_k) <= 0:
            args.student_top_k = int(args.kl_top_k)
        args.student_top_k = int(args.student_top_k)
        args.huber_delta = max(float(args.huber_delta), 1.0e-6)
        args.reverse_kl_weight = max(float(args.reverse_kl_weight), 0.0)
        args.endpoint_prompt_weight = max(float(args.endpoint_prompt_weight), 0.0)
        args.endpoint_answer_weight = max(float(args.endpoint_answer_weight), 0.0)
        if int(args.endpoint_top_k) <= 0:
            args.endpoint_top_k = int(args.kl_top_k)
        args.endpoint_top_k = int(args.endpoint_top_k)
        args.endpoint_topk_center = bool(args.endpoint_topk_center)
        args.endpoint_vocab_reduction = str(args.endpoint_vocab_reduction).strip().lower()
    set_seed(int(args.seed))
    output_dir, stamp = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    args.stamp = stamp
    write_json(output_dir / "train_config.json", vars(args))

    prompt_format = PromptFormat(
        question_start=str(args.question_start),
        question_end=str(args.question_end),
        think_start=str(args.think_start),
        think_end=str(args.think_end),
        answer_tag=str(args.answer_tag or ""),
    )
    forget_records = build_forget_records(
        Path(args.forget_path),
        task_id=str(args.task_id),
        limit=int(args.forget_limit),
        prompt_format=prompt_format,
        trajectory_mode=str(args.forget_trajectory_mode),
    )
    retain_extra_path = Path(args.retain_extra_forget_path) if str(args.retain_extra_forget_path or "").strip() else None
    retain_records = build_retain_records(
        Path(args.retain_path),
        limit=int(args.retain_limit),
        seed=int(args.seed),
        sample=str(args.retain_sample),
        extra_forget_path=retain_extra_path,
        extra_exclude_task_id=str(args.retain_extra_exclude_task_id or args.task_id),
    )
    if bool(args.retain_teacher_uses_prefix):
        for record in retain_records:
            record.teacher = "prefix"
    write_jsonl(output_dir / "forget_records.jsonl", [record.to_json() for record in forget_records])
    write_jsonl(output_dir / "retain_records.jsonl", [record.to_json() for record in retain_records])
    eval_forget_records, eval_retain_records, eval_selection = select_eval_records(
        forget_records,
        retain_records,
        forget_count=int(args.eval_forget_samples),
        retain_count=int(args.eval_retain_samples),
        seed=int(args.eval_seed),
    )
    write_json(output_dir / "generation_eval_selection.json", eval_selection)

    train_forget_records = (
        add_root_state_records(
            forget_records,
            prompt_format=prompt_format,
            root_state_completion=str(args.root_state_completion),
            root_state_templates=str(args.root_state_templates),
        )
        if bool(args.add_root_states)
        else list(forget_records)
    )
    train_retain_records = (
        add_root_state_records(
            retain_records,
            prompt_format=prompt_format,
            root_state_completion=str(args.root_state_completion),
            root_state_templates=str(args.root_state_templates),
        )
        if bool(args.add_root_states)
        else list(retain_records)
    )
    if bool(args.add_root_states):
        write_jsonl(output_dir / "train_forget_records.jsonl", [record.to_json() for record in train_forget_records])
        write_jsonl(output_dir / "train_retain_records.jsonl", [record.to_json() for record in train_retain_records])

    if args.dry_run:
        write_json(
            output_dir / "data_stats.json",
            {
                "forget_records": len(forget_records),
                "retain_records": len(retain_records),
                "train_forget_records": len(train_forget_records),
                "train_retain_records": len(train_retain_records),
                "forget_trajectory_mode": str(args.forget_trajectory_mode),
                "add_root_states": bool(args.add_root_states),
                "root_state_templates": str(args.root_state_templates),
                "prompt_format": prompt_format.__dict__,
            },
        )
        log(f"dry_run complete output={output_dir}")
        return

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(str(args.device))
    dtype = dtype_from_name(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if str(args.attn_implementation or ""):
        load_kwargs["attn_implementation"] = str(args.attn_implementation)
    log(f"loading base model path={args.model_path} dtype={args.dtype} device={device}")
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), **load_kwargs)
    model.to(device)
    model.config.use_cache = False

    prefix = load_soft_prefix(args.prefix_path, args.prefix_position).to(device)
    args.prefix_position = prefix.prefix_position
    hidden = int(model.get_input_embeddings().embedding_dim)
    if int(prefix.embedding.shape[1]) != hidden:
        raise ValueError(f"prefix hidden size {prefix.embedding.shape[1]} does not match model hidden size {hidden}")
    write_json(output_dir / "prefix_metadata.json", prefix.metadata)
    log(f"loaded teacher prefix len={prefix.prefix_len} position={prefix.prefix_position} path={args.prefix_path}")

    lora_targets = parse_lora_targets(str(args.lora_target_modules), model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    trainable_dtype_counts = cast_trainable_parameters_to_fp32(model)
    write_json(output_dir / "lora_targets.json", lora_targets)
    log(f"global LoRA targets={lora_targets}")
    log(f"cast trainable parameters to fp32 from {trainable_dtype_counts}")
    model.print_trainable_parameters()

    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        enable_input_require_grads(model)
        model.config.use_cache = False

    retain_teacher_prefix = prefix if bool(args.retain_teacher_uses_prefix) else None
    log(
        "running pre-train generation review: "
        f"forget uses prefix, retain uses {'prefix' if retain_teacher_prefix is not None else 'base model'}"
    )
    pre_rows = run_generation_review(
        model,
        tokenizer,
        prompt_format,
        eval_forget_records,
        eval_retain_records,
        output_dir,
        phase="pre_teacher",
        step=0,
        device=device,
        max_new_tokens=int(args.eval_max_new_tokens),
        batch_size=int(args.eval_batch_size),
        forget_prefix=prefix,
        retain_prefix=retain_teacher_prefix,
        disable_adapter=True,
    )
    write_json(output_dir / "generation_pre_teacher_summary.json", summarize_generation_rows(pre_rows))

    trajectory_cache = Path(args.trajectory_cache) if str(args.trajectory_cache or "").strip() else output_dir / "teacher_trajectories.jsonl"
    with adapter_disabled(model):
        old_use_cache = getattr(model.config, "use_cache", None)
        model.eval()
        if old_use_cache is not None:
            model.config.use_cache = True
        forget_rows, retain_rows = load_or_generate_trajectories(
            model,
            tokenizer,
            prefix,
            retain_teacher_prefix,
            prompt_format,
            train_forget_records,
            train_retain_records,
            trajectory_cache,
            device=device,
            batch_size=int(args.trajectory_generation_batch_size),
            forget_max_new_tokens=int(args.forget_max_new_tokens),
            retain_max_new_tokens=int(args.retain_max_new_tokens),
            force=bool(args.force_rebuild_trajectories),
        )
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
    model.train()
    model.config.use_cache = False
    write_jsonl(output_dir / "teacher_trajectories.used.jsonl", forget_rows + retain_rows)

    forget_dataset = TrajectoryDataset(
        forget_rows,
        tokenizer,
        max_length=int(args.max_length),
        skip_too_long=bool(args.skip_too_long),
        prompt_format=prompt_format,
        kl_token_scope=str(args.kl_token_scope),
        kl_cot_fraction=float(args.kl_cot_fraction),
    )
    retain_dataset = TrajectoryDataset(
        retain_rows,
        tokenizer,
        max_length=int(args.max_length),
        skip_too_long=bool(args.skip_too_long),
        prompt_format=prompt_format,
        kl_token_scope=str(args.kl_token_scope),
        kl_cot_fraction=float(args.kl_cot_fraction),
    )
    raw_answer_dataset = None
    if float(args.lambda_answer_erasure) > 0.0:
        raw_answer_dataset = RawAnswerDataset(
            forget_records,
            tokenizer,
            max_length=int(args.max_length),
            skip_too_long=bool(args.skip_too_long),
            prompt_format=prompt_format,
            prompt_templates=str(args.answer_erasure_prompt_templates),
        )
    latent_shift_dataset = None
    if float(args.lambda_latent_shift) > 0.0:
        latent_shift_dataset = RootPromptDataset(
            forget_records,
            tokenizer,
            max_length=int(args.max_length),
            skip_too_long=bool(args.skip_too_long),
            prompt_format=prompt_format,
            prompt_templates=str(args.latent_shift_prompt_templates),
        )
    if int(args.steps_per_epoch) <= 0:
        args.steps_per_epoch = max(
            1,
            math.ceil(len(forget_dataset) / max(1, int(args.batch_size))),
        )
    if int(args.epochs) <= 0:
        args.epochs = max(1, math.ceil(int(args.max_steps) / max(1, int(args.steps_per_epoch))))
    computed_max_steps = int(args.epochs) * int(args.steps_per_epoch)
    if int(args.max_steps) != computed_max_steps:
        log(
            f"overriding max_steps={args.max_steps} with epochs*steps_per_epoch={computed_max_steps} "
            "for epoch-wise retain sampling"
        )
        args.max_steps = computed_max_steps
        write_json(output_dir / "train_config.json", vars(args))
    first_epoch_schedule = build_epoch_schedule(
        steps_per_epoch=int(args.steps_per_epoch),
        forget_updates=int(args.forget_updates_per_cycle),
        retain_updates=int(args.retain_updates_per_cycle),
    )
    data_stats = {
        "forget_records": len(forget_records),
        "retain_records": len(retain_records),
        "train_forget_records": len(train_forget_records),
        "train_retain_records": len(train_retain_records),
        "forget_trajectory_mode": str(args.forget_trajectory_mode),
        "add_root_states": bool(args.add_root_states),
        "root_state_templates": str(args.root_state_templates),
        "forget_trajectories": len(forget_rows),
        "retain_trajectory_pool": len(retain_rows),
        "forget_dataset": dataset_stats(forget_dataset),
        "retain_pool_dataset": dataset_stats(retain_dataset),
        "raw_answer_erasure_dataset": (
            {
                "n": len(raw_answer_dataset),
                "skipped": int(getattr(raw_answer_dataset, "skipped", 0)),
                "prompt_templates": str(args.answer_erasure_prompt_templates),
            }
            if raw_answer_dataset is not None
            else {"n": 0, "skipped": 0, "prompt_templates": str(args.answer_erasure_prompt_templates)}
        ),
        "latent_shift_dataset": (
            {
                "n": len(latent_shift_dataset),
                "skipped": int(getattr(latent_shift_dataset, "skipped", 0)),
                "prompt_templates": str(args.latent_shift_prompt_templates),
            }
            if latent_shift_dataset is not None
            else {"n": 0, "skipped": 0, "prompt_templates": str(args.latent_shift_prompt_templates)}
        ),
        "schedule": {
            "max_steps": int(args.max_steps),
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "forget_updates": sum(1 for item in first_epoch_schedule if item == "forget") * int(args.epochs),
            "retain_updates": sum(1 for item in first_epoch_schedule if item == "retain") * int(args.epochs),
            "forget_updates_per_cycle": int(args.forget_updates_per_cycle),
            "retain_updates_per_cycle": int(args.retain_updates_per_cycle),
            "forget_samples_per_epoch": int(args.forget_samples_per_epoch),
            "retain_samples_per_epoch": int(args.retain_samples_per_epoch),
            "retain_epoch_sample_seed": int(args.retain_epoch_sample_seed),
            "first_epoch_first_100": first_epoch_schedule[:100],
        },
        "loss": {
            "name": (
                "scoped_trajectory_kl"
                if args.distill_objective == "trajectory_kl"
                else (
                    "union_topk_huber_centered"
                    if args.distill_objective == "union_topk_huber"
                    else (
                    "jeffrey_topk_kl"
                    if args.distill_objective == "jeffrey_topk_kl"
                    else (
                        "prefix_shift_huber"
                        if args.distill_objective == "prefix_shift_huber"
                        else (
                            "probability_shift_huber"
                            if args.distill_objective == "probability_shift_huber"
                            else (
                                "endpoint_topk_huber"
                                if args.distill_objective == "endpoint_topk_huber"
                                else "endpoint_full_huber"
                            )
                        )
                    )
                )
                )
            ),
            "teacher_forget": f"base_model_plus_{args.prefix_position}_prefix",
            "teacher_retain": (
                f"base_model_plus_{args.prefix_position}_prefix"
                if bool(args.retain_teacher_uses_prefix)
                else "base_model_without_prefix"
            ),
            "student": "base_model_plus_global_lora_without_runtime_prefix",
            "mask": str(args.kl_token_scope),
            "kl_cot_fraction": float(args.kl_cot_fraction),
            "kl_top_k": int(args.kl_top_k),
            "distill_objective": str(args.distill_objective),
            "student_top_k": int(args.student_top_k),
            "huber_delta": float(args.huber_delta),
            "state_loss_reduction": str(args.state_loss_reduction),
            "forget_state_loss_reduction": str(args.forget_state_loss_reduction),
            "retain_state_loss_reduction": str(args.retain_state_loss_reduction),
            "union_center": bool(args.union_center),
            "reverse_kl_weight": float(args.reverse_kl_weight),
            "endpoint_prompt_weight": float(args.endpoint_prompt_weight),
            "endpoint_answer_weight": float(args.endpoint_answer_weight),
            "endpoint_top_k": int(args.endpoint_top_k),
            "endpoint_topk_center": bool(args.endpoint_topk_center),
            "endpoint_vocab_reduction": str(args.endpoint_vocab_reduction),
            "lambda_answer_erasure": float(args.lambda_answer_erasure),
            "answer_erasure_prompt_templates": str(args.answer_erasure_prompt_templates),
            "answer_erasure_reduction": str(args.answer_erasure_reduction),
            "lambda_latent_shift": float(args.lambda_latent_shift),
            "latent_shift_prompt_templates": str(args.latent_shift_prompt_templates),
            "latent_shift_loss_type": str(args.latent_shift_loss_type),
        },
    }
    write_json(output_dir / "data_stats.json", data_stats)
    log(json.dumps(data_stats, ensure_ascii=False))

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    def lr_lambda(step: int) -> float:
        if int(args.warmup_steps) <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(args.warmup_steps))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    recent: dict[str, list[float]] = {"forget": [], "retain": []}
    train_log_path = output_dir / "train_log.jsonl"
    forget_epoch_sample_log: list[dict[str, Any]] = []
    retain_epoch_sample_log: list[dict[str, Any]] = []
    global_step = 0
    save_epochs = parse_epoch_list(getattr(args, "save_epochs", ""))
    log(f"starting full-trajectory distillation training objective={args.distill_objective}")
    for epoch in range(1, int(args.epochs) + 1):
        forget_indices = sample_epoch_indices(
            len(forget_dataset),
            int(args.forget_samples_per_epoch),
            int(args.retain_epoch_sample_seed) + 104729,
            epoch,
        )
        retain_indices = sample_epoch_indices(
            len(retain_dataset),
            int(args.retain_samples_per_epoch),
            int(args.retain_epoch_sample_seed),
            epoch,
        )
        raw_answer_indices: list[int] = []
        if raw_answer_dataset is not None:
            raw_sample_n = int(args.answer_erasure_samples_per_epoch)
            if raw_sample_n <= 0:
                raw_sample_n = int(args.forget_samples_per_epoch)
            raw_answer_indices = sample_epoch_indices(
                len(raw_answer_dataset),
                raw_sample_n,
                int(args.retain_epoch_sample_seed) + 314159,
                epoch,
            )
        latent_shift_indices: list[int] = []
        if latent_shift_dataset is not None:
            latent_sample_n = int(args.latent_shift_samples_per_epoch)
            if latent_sample_n <= 0:
                latent_sample_n = int(args.forget_samples_per_epoch)
            latent_shift_indices = sample_epoch_indices(
                len(latent_shift_dataset),
                latent_sample_n,
                int(args.retain_epoch_sample_seed) + 271828,
                epoch,
            )
        epoch_forget_dataset = subset_dataset(forget_dataset, forget_indices)
        epoch_retain_dataset = subset_dataset(retain_dataset, retain_indices)
        epoch_raw_answer_dataset = (
            subset_raw_answer_dataset(raw_answer_dataset, raw_answer_indices)
            if raw_answer_dataset is not None
            else None
        )
        epoch_latent_shift_dataset = (
            subset_root_prompt_dataset(latent_shift_dataset, latent_shift_indices)
            if latent_shift_dataset is not None
            else None
        )
        forget_loader = CyclingLoader(epoch_forget_dataset, tokenizer, int(args.batch_size), int(args.seed) + 11 + epoch)
        retain_loader = CyclingLoader(epoch_retain_dataset, tokenizer, int(args.batch_size), int(args.seed) + 97 + epoch)
        raw_answer_loader = (
            CyclingLoader(
                epoch_raw_answer_dataset,
                tokenizer,
                int(args.answer_erasure_batch_size) if int(args.answer_erasure_batch_size) > 0 else int(args.batch_size),
                int(args.seed) + 193 + epoch,
                collate_kind="raw_answer",
            )
            if epoch_raw_answer_dataset is not None
            else None
        )
        latent_shift_loader = (
            CyclingLoader(
                epoch_latent_shift_dataset,
                tokenizer,
                int(args.latent_shift_batch_size) if int(args.latent_shift_batch_size) > 0 else int(args.batch_size),
                int(args.seed) + 389 + epoch,
                collate_kind="root_prompt",
            )
            if epoch_latent_shift_dataset is not None
            else None
        )
        epoch_schedule = build_epoch_schedule(
            steps_per_epoch=int(args.steps_per_epoch),
            forget_updates=int(args.forget_updates_per_cycle),
            retain_updates=int(args.retain_updates_per_cycle),
        )
        forget_epoch_sample_log.append(
            {
                "epoch": epoch,
                "sample_n": len(forget_indices),
                "indices": forget_indices,
                "source_keys": [
                    str(epoch_forget_dataset.rows[idx]["example"].get("source_key", ""))
                    for idx in range(len(epoch_forget_dataset))
                ],
            }
        )
        retain_epoch_sample_log.append(
            {
                "epoch": epoch,
                "sample_n": len(retain_indices),
                "indices": retain_indices,
                "source_keys": [
                    str(epoch_retain_dataset.rows[idx]["example"].get("source_key", ""))
                    for idx in range(len(epoch_retain_dataset))
                ],
            }
        )
        write_json(output_dir / "forget_epoch_samples.json", forget_epoch_sample_log)
        write_json(output_dir / "retain_epoch_samples.json", retain_epoch_sample_log)
        log(
            f"epoch={epoch}/{args.epochs} forget_sample_n={len(forget_indices)} "
            f"retain_sample_n={len(retain_indices)} "
            f"answer_erasure_sample_n={len(raw_answer_indices)} "
            f"latent_shift_sample_n={len(latent_shift_indices)} steps={len(epoch_schedule)}"
        )

        for split in epoch_schedule:
            global_step += 1
            batch = forget_loader.next() if split == "forget" else retain_loader.next()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_fulltraj_kl_loss(model, batch, prefix, split, args, device=device)
            if split == "forget" and float(args.lambda_latent_shift) > 0.0:
                if latent_shift_loader is None:
                    raise RuntimeError("lambda_latent_shift > 0 but latent_shift_loader is missing")
                latent_batch = latent_shift_loader.next()
                before_latent_loss = loss
                latent_loss, latent_metrics = latent_shift_alignment_loss(
                    model,
                    latent_batch,
                    prefix,
                    device=device,
                    loss_type=str(args.latent_shift_loss_type),
                    huber_delta=float(args.latent_shift_huber_delta),
                )
                loss = before_latent_loss + float(args.lambda_latent_shift) * latent_loss
                metrics = {
                    **metrics,
                    "distill_loss": float(before_latent_loss.detach().cpu()),
                    **latent_metrics,
                    "lambda_latent_shift": float(args.lambda_latent_shift),
                    "loss": float(loss.detach().cpu()),
                }
            if split == "forget" and float(args.lambda_answer_erasure) > 0.0:
                if raw_answer_loader is None:
                    raise RuntimeError("lambda_answer_erasure > 0 but raw_answer_loader is missing")
                erasure_batch = raw_answer_loader.next()
                distill_loss = loss
                erasure_loss, erasure_metrics = raw_answer_unlikelihood_loss(
                    model,
                    erasure_batch,
                    device=device,
                    eps=float(args.answer_erasure_eps),
                    reduction=str(args.answer_erasure_reduction),
                )
                loss = distill_loss + float(args.lambda_answer_erasure) * erasure_loss
                metrics = {
                    **metrics,
                    "distill_loss": float(distill_loss.detach().cpu()),
                    **erasure_metrics,
                    "lambda_answer_erasure": float(args.lambda_answer_erasure),
                    "loss": float(loss.detach().cpu()),
                }
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite loss at step={global_step} split={split}: {metrics}")
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm), error_if_nonfinite=True)
            optimizer.step()
            lr_scheduler.step()

            row = {
                "step": int(global_step),
                "epoch": int(epoch),
                "split": split,
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "forget_epoch_sample_n": len(forget_indices),
                "retain_epoch_sample_n": len(retain_indices),
                **metrics,
            }
            with train_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            recent[split].append(float(metrics["loss"]))
            if len(recent[split]) > max(1, int(args.log_every)):
                recent[split].pop(0)

            if int(args.log_every) > 0 and (global_step == 1 or global_step % int(args.log_every) == 0):
                forget_mean = sum(recent["forget"]) / max(1, len(recent["forget"]))
                retain_mean = sum(recent["retain"]) / max(1, len(recent["retain"]))
                log(
                    f"step={global_step}/{args.max_steps} epoch={epoch}/{args.epochs} "
                    f"split={split} loss={metrics['loss']:.6g} tokens={metrics['tokens']} "
                    f"recent_forget={forget_mean:.6g} recent_retain={retain_mean:.6g} "
                    f"lr={row['lr']:.3e}"
                )

            if int(args.save_every) > 0 and global_step % int(args.save_every) == 0:
                save_adapter(model, tokenizer, output_dir / f"checkpoint-{global_step:06d}", args, global_step)

        if int(epoch) in save_epochs:
            epoch_dir = output_dir / f"checkpoint-epoch{int(epoch):02d}"
            save_adapter(model, tokenizer, epoch_dir, args, int(global_step))
            log(f"saved epoch checkpoint epoch={epoch} step={global_step} path={epoch_dir}")

    save_adapter(model, tokenizer, output_dir / "checkpoint-last", args, int(global_step))
    log("running post-train generation review: LoRA student without runtime prefix")
    post_rows = run_generation_review(
        model,
        tokenizer,
        prompt_format,
        eval_forget_records,
        eval_retain_records,
        output_dir,
        phase="post_student",
        step=int(global_step),
        device=device,
        max_new_tokens=int(args.eval_max_new_tokens),
        batch_size=int(args.eval_batch_size),
        forget_prefix=None,
        retain_prefix=None,
        disable_adapter=False,
    )
    write_json(output_dir / "generation_post_student_summary.json", summarize_generation_rows(post_rows))
    compare_rows = compare_generation_rows(pre_rows, post_rows)
    write_jsonl(output_dir / "generation_compare.jsonl", compare_rows)
    write_csv(output_dir / "generation_compare.csv", compare_rows)
    write_json(output_dir / "generation_compare_summary.json", summarize_generation_compare(compare_rows))
    log(f"done adapter={output_dir / 'checkpoint-last'}")


if __name__ == "__main__":
    main()
