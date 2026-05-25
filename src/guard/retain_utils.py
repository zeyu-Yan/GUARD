#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

GUARD_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("GUARD_PROJECT_ROOT", str(GUARD_ROOT)))

from guard.prefix_utils import (
    DEFAULT_IDKCOT,
    DEFAULT_IDONTKNOW,
    DEFAULT_MODEL,
    DEFAULT_MODEL_CONFIG,
    PrefixDataset,
    SoftPrefix,
    build_examples,
    build_prompt,
    cleanup_generated_text,
    collate,
    evaluate_completion,
    forward_loss,
    generate_batch,
    get_model_config,
    get_torch_dtype,
    load_json_or_jsonl,
    load_model_and_tokenizer,
    normalize_prefix_position,
    summarize_training_lengths,
    write_json,
    write_jsonl,
)

DEFAULT_FORGET = PROJECT_ROOT / "data" / "forget.json"
DEFAULT_RETAIN = PROJECT_ROOT / "data" / "retain.json"
DEFAULT_RUN_NAME = "guard_retain_helper"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "retain_helper"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "retain_helper_results"


def log(msg: str) -> None:
    print(f"[guard-retain] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))


def summarize_generations(tag: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tag": tag,
        "n": len(rows),
        "clean_refusal_mean": mean(rows, "clean_refusal"),
        "idk_like_mean": mean(rows, "idk_like"),
        "gold_refusal_like_mean": mean(rows, "gold_refusal_like"),
        "fact_hit_mean": mean(rows, "source_private_fact_hits"),
        "empty_answer_mean": mean(rows, "empty_answer"),
        "bad_generation_mean": mean(rows, "bad_generation"),
        "has_think_end_mean": mean(rows, "has_think_end"),
        "concise_answer_mean": mean(rows, "concise_answer"),
        "enough_cot_mean": mean(rows, "enough_cot"),
        "cot_tokens_mean": mean(rows, "cot_tokens"),
        "answer_tokens_mean": mean(rows, "answer_tokens"),
        "source_answer_similarity_mean": mean(rows, "source_answer_similarity"),
    }


def count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_retain_records(
    retain_path: Path,
    model_cfg: dict[str, Any],
    *,
    limit: int,
    seed: int,
    sample: str,
    exclude_source_idxs: set[int] | None = None,
) -> list[dict[str, Any]]:
    rows = [row for row in load_json_or_jsonl(retain_path) if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"no retain rows found in {retain_path}")
    indexed = list(enumerate(rows))
    if exclude_source_idxs:
        excluded = {int(idx) for idx in exclude_source_idxs}
        indexed = [(idx, row) for idx, row in indexed if idx not in excluded]
        if not indexed:
            raise ValueError("no retain rows left after excluding training source_idx values")
    if sample == "random":
        rng = random.Random(seed)
        rng.shuffle(indexed)
    elif sample != "head":
        raise ValueError(f"unsupported retain_sample={sample!r}; use head or random")
    if limit > 0:
        indexed = indexed[:limit]
    records: list[dict[str, Any]] = []
    for local_idx, (source_idx, row) in enumerate(indexed):
        question = str(row.get("question") or "").strip()
        source_answer = str(row.get("answer") or "").strip()
        records.append(
            {
                "idx": local_idx,
                "source_idx": source_idx,
                "probe_id": f"{retain_path.stem}_{source_idx:05d}",
                "task_id": str(row.get("task_id", "retain")),
                "question": question,
                "source_answer": source_answer,
                "target_answer": source_answer,
                "prompt": build_prompt(question, model_cfg),
            }
        )
    return records


def build_combined_retain_records(
    retain_path: Path,
    model_cfg: dict[str, Any],
    *,
    limit: int,
    seed: int,
    sample: str,
    extra_forget_path: Path | None = None,
    extra_exclude_task_id: str = "",
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, int, dict[str, Any]]] = []
    retain_rows = [row for row in load_json_or_jsonl(retain_path) if isinstance(row, dict)]
    for source_idx, row in enumerate(retain_rows):
        candidates.append((retain_path.stem, source_idx, row))

    if extra_forget_path:
        forget_rows = [row for row in load_json_or_jsonl(extra_forget_path) if isinstance(row, dict)]
        excluded_task = str(extra_exclude_task_id) if extra_exclude_task_id else ""
        for source_idx, row in enumerate(forget_rows):
            if excluded_task and str(row.get("task_id")) == excluded_task:
                continue
            candidates.append((extra_forget_path.stem, source_idx, row))

    if not candidates:
        raise ValueError("no retain candidates found")
    if sample == "random":
        rng = random.Random(seed)
        rng.shuffle(candidates)
    elif sample != "head":
        raise ValueError(f"unsupported retain_sample={sample!r}; use head or random")
    if limit > 0:
        candidates = candidates[:limit]

    records: list[dict[str, Any]] = []
    for local_idx, (source_name, source_idx, row) in enumerate(candidates):
        question = str(row.get("question") or "").strip()
        source_answer = str(row.get("answer") or "").strip()
        records.append(
            {
                "idx": local_idx,
                "source_idx": source_idx,
                "source_dataset": source_name,
                "source_key": f"{source_name}:{source_idx}",
                "probe_id": f"{source_name}_{source_idx:05d}",
                "task_id": str(row.get("task_id", "retain")),
                "question": question,
                "source_answer": source_answer,
                "target_answer": source_answer,
                "prompt": build_prompt(question, model_cfg),
            }
        )
    return records


def retain_source_key(record: dict[str, Any]) -> str:
    if record.get("source_key"):
        return str(record["source_key"])
    dataset = str(record.get("source_dataset") or "retain")
    return f"{dataset}:{int(record.get('source_idx', -1))}"


def build_retain_eval_records(
    retain_path: Path,
    model_cfg: dict[str, Any],
    *,
    train_records: list[dict[str, Any]],
    limit: int,
    seed: int,
    eval_seed: int,
    sample: str,
) -> list[dict[str, Any]]:
    if sample == "train":
        return train_records[:limit] if limit > 0 else list(train_records)
    if sample == "head_exclude_train":
        retain_sample = "head"
    elif sample == "random_exclude_train":
        retain_sample = "random"
    else:
        raise ValueError(
            f"unsupported eval_retain_sample={sample!r}; "
            "use train, head_exclude_train, or random_exclude_train"
        )
    train_source_keys = {retain_source_key(record) for record in train_records}
    train_source_idxs = {int(record["source_idx"]) for record in train_records if str(record.get("source_dataset", "")) in {"", retain_path.stem}}
    return build_retain_records(
        retain_path,
        model_cfg,
        limit=limit,
        seed=eval_seed if eval_seed >= 0 else seed + 1000003,
        sample=retain_sample,
        exclude_source_idxs=train_source_idxs,
    )


def trim_generated_ids(ids: list[int], *, eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    trimmed = list(ids)
    if eos_token_id is not None and eos_token_id in trimmed:
        trimmed = trimmed[: trimmed.index(eos_token_id) + 1]
    elif eos_token_id is not None:
        while trimmed and pad_token_id is not None and trimmed[-1] == pad_token_id:
            trimmed.pop()
        trimmed.append(int(eos_token_id))
    elif pad_token_id is not None:
        while trimmed and trimmed[-1] == pad_token_id:
            trimmed.pop()
    return trimmed


def generate_base_trajectory_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: str,
    max_new_tokens: int,
) -> tuple[list[str], list[list[int]]]:
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    suffix = out[:, enc["input_ids"].shape[1] :]
    completion_ids = [
        trim_generated_ids(
            [int(token_id) for token_id in row.tolist()],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in suffix
    ]
    completions = [cleanup_generated_text(tokenizer.decode(ids, skip_special_tokens=True)) for ids in completion_ids]
    return completions, completion_ids


def load_or_generate_retain_trajectories(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    cache_path: Path,
    *,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    force: bool,
) -> list[dict[str, Any]]:
    if cache_path.exists() and not force:
        rows = [row for row in load_json_or_jsonl(cache_path) if isinstance(row, dict)]
        wanted = {(retain_source_key(r), str(r["question"])) for r in records}
        got = {(retain_source_key(r), str(r.get("question") or "")) for r in rows}
        if wanted.issubset(got):
            by_key = {(retain_source_key(r), str(r["question"])): r for r in rows}
            ordered = [by_key[(retain_source_key(r), str(r["question"]))] for r in records]
            log(f"loaded retain base trajectories={len(ordered)} cache={cache_path}")
            return ordered
        log(f"cache does not cover requested retain rows; regenerating cache={cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(records) / max(1, batch_size))
    for start_idx in range(0, len(records), batch_size):
        batch_records = records[start_idx : start_idx + batch_size]
        completions, completion_ids = generate_base_trajectory_batch(
            model,
            tokenizer,
            [r["prompt"] for r in batch_records],
            device=device,
            max_new_tokens=max_new_tokens,
        )
        for record, completion, ids in zip(batch_records, completions, completion_ids):
            rows.append(
                {
                    **record,
                    "base_completion": completion,
                    "completion_ids": ids,
                    "completion_token_count": len(ids),
                    "has_eos_target": bool(tokenizer.eos_token_id is not None and ids and ids[-1] == tokenizer.eos_token_id),
                    "max_new_tokens": int(max_new_tokens),
                }
            )
        batch_no = start_idx // batch_size + 1
        if batch_no == 1 or batch_no % 5 == 0 or batch_no == total_batches:
            log(f"base retain trajectory generation_batch={batch_no}/{total_batches}")
    write_jsonl(cache_path, rows)
    log(f"wrote retain base trajectories={len(rows)} cache={cache_path}")
    return rows


class RetainKLDataset(Dataset):
    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        tokenizer,
        *,
        max_length: int,
        think_end_tag: str = "\n</think>\n\n",
        min_completion_tokens: int = 1,
    ):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.think_end_tag = str(think_end_tag or "\n</think>\n\n")
        self.think_end_token_variants = self._build_think_end_token_variants()
        self.min_completion_tokens = int(min_completion_tokens)
        self.rows: list[dict[str, Any]] = []
        skipped = 0
        for row in trajectories:
            try:
                self.rows.append(self._encode(row))
            except ValueError:
                skipped += 1
        if not self.rows:
            raise ValueError("all retain trajectories were skipped; increase max_length or max_new_tokens")
        if skipped:
            log(f"skipped retain trajectories={skipped} because they exceed max_length or are empty")

    @staticmethod
    def _find_subsequence(values: list[int], needle: list[int]) -> int:
        if not needle or len(needle) > len(values):
            return -1
        last = len(values) - len(needle)
        for start in range(last + 1):
            if values[start : start + len(needle)] == needle:
                return start
        return -1

    def _build_think_end_token_variants(self) -> list[list[int]]:
        variants: list[list[int]] = []
        texts = [
            self.think_end_tag,
            self.think_end_tag.strip(),
            "\n</think>\n\n",
            "</think>",
        ]
        seen: set[tuple[int, ...]] = set()
        for text in texts:
            if not text:
                continue
            token_ids = [
                int(token_id)
                for token_id in self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            ]
            key = tuple(token_ids)
            if token_ids and key not in seen:
                seen.add(key)
                variants.append(token_ids)
        return variants

    def _completion_block_masks(self, completion_ids: list[int]) -> tuple[list[float], list[float], bool]:
        answer_start = len(completion_ids)
        found = False
        for tag_ids in self.think_end_token_variants:
            match_start = self._find_subsequence(completion_ids, tag_ids)
            if match_start >= 0:
                answer_start = match_start
                found = True
                break
        cot_mask = [1.0 if idx < answer_start else 0.0 for idx in range(len(completion_ids))]
        answer_mask = [1.0 if idx >= answer_start else 0.0 for idx in range(len(completion_ids))]
        return cot_mask, answer_mask, found

    def _encode(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt = str(row["prompt"])
        prompt_ids = list(self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        raw_completion_ids = row.get("completion_ids")
        if isinstance(raw_completion_ids, list) and raw_completion_ids:
            completion_ids = [int(token_id) for token_id in raw_completion_ids]
            if self.tokenizer.eos_token_id is not None and completion_ids[-1] != self.tokenizer.eos_token_id:
                completion_ids.append(int(self.tokenizer.eos_token_id))
        else:
            completion = cleanup_generated_text(str(row.get("base_completion") or ""))
            completion_ids = list(
                self.tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]
            )
            if self.tokenizer.eos_token_id is not None:
                completion_ids.append(int(self.tokenizer.eos_token_id))
        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)
        prompt_len = len(prompt_ids)
        completion_len = len(completion_ids)
        if completion_len < self.min_completion_tokens:
            raise ValueError(f"empty retain completion: {row.get('probe_id')}")
        if len(input_ids) > self.max_length:
            raise ValueError(f"retain trajectory too long: {len(input_ids)} > {self.max_length}")
        completion_cot_mask, completion_answer_mask, has_think_end_target = self._completion_block_masks(completion_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": prompt_len,
            "completion_len": completion_len,
            "completion_cot_mask": completion_cot_mask,
            "completion_answer_mask": completion_answer_mask,
            "has_think_end_target": has_think_end_target,
            "has_eos_target": bool(
                self.tokenizer.eos_token_id is not None and completion_ids[-1] == self.tokenizer.eos_token_id
            ),
            "example": row,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def retain_collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    prompt_lens: list[int] = []
    completion_lens: list[int] = []
    completion_cot_mask: list[list[float]] = []
    completion_answer_mask: list[list[float]] = []
    examples: list[dict[str, Any]] = []
    max_completion_len = max(int(row["completion_len"]) for row in batch)
    for row in batch:
        pad = max_len - len(row["input_ids"])
        completion_pad = max_completion_len - int(row["completion_len"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
        prompt_lens.append(int(row["prompt_len"]))
        completion_lens.append(int(row["completion_len"]))
        completion_cot_mask.append(row["completion_cot_mask"] + [0.0] * completion_pad)
        completion_answer_mask.append(row["completion_answer_mask"] + [0.0] * completion_pad)
        examples.append(row["example"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        "completion_lens": torch.tensor(completion_lens, dtype=torch.long),
        "completion_cot_mask": torch.tensor(completion_cot_mask, dtype=torch.float),
        "completion_answer_mask": torch.tensor(completion_answer_mask, dtype=torch.float),
        "examples": examples,
    }


def summarize_retain_alignment(retain_ds: RetainKLDataset) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in retain_ds.rows:
        ex = row["example"]
        rows.append(
            {
                "probe_id": ex.get("probe_id"),
                "source_idx": ex.get("source_idx"),
                "prompt_tokens": int(row["prompt_len"]),
                "kl_target_tokens": int(row["completion_len"]),
                "kl_cot_tokens": float(sum(row["completion_cot_mask"])),
                "kl_answer_tokens": float(sum(row["completion_answer_mask"])),
                "full_tokens": int(len(row["input_ids"])),
                "has_think_end_target": int(bool(row.get("has_think_end_target"))),
                "has_eos_target": int(bool(row.get("has_eos_target"))),
            }
        )

    def stat(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in rows]
        if not values:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}

    return {
        "n": len(rows),
        "note": (
            "KL is computed only on base-generated completion token ids plus final EOS; "
            "prompt tokens are excluded. Retain KL is normalized per sample and per COT/answer block."
        ),
        "position_alignment": {
            "teacher_logit_position_for_completion_j": "prompt_len + j - 1",
            "student_logit_position_for_completion_j": "prompt_len + prefix_len + j - 1",
        },
        "prompt_tokens": stat("prompt_tokens"),
        "kl_target_tokens": stat("kl_target_tokens"),
        "kl_cot_tokens": stat("kl_cot_tokens"),
        "kl_answer_tokens": stat("kl_answer_tokens"),
        "full_tokens": stat("full_tokens"),
        "has_think_end_target_mean": mean(rows, "has_think_end_target"),
        "has_eos_target_mean": mean(rows, "has_eos_target"),
        "top_by_kl_target_tokens": sorted(rows, key=lambda item: item["kl_target_tokens"], reverse=True)[:10],
    }


def insert_prompt_end_prefix_for_batch(
    model,
    prefix: SoftPrefix,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embed_layer = model.get_input_embeddings()
    tok_embeds = embed_layer(input_ids)
    bsz, seq_len = input_ids.shape
    pref = prefix(bsz).to(device=input_ids.device, dtype=tok_embeds.dtype)
    pref_mask = torch.ones((bsz, pref.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
    embed_rows: list[torch.Tensor] = []
    attn_rows: list[torch.Tensor] = []
    for row_idx in range(bsz):
        insert_at = int(prompt_lens[row_idx].item())
        if insert_at < 0 or insert_at > seq_len:
            raise ValueError(f"invalid prompt_len={insert_at}; seq_len={seq_len}")
        embed_rows.append(
            torch.cat(
                [
                    tok_embeds[row_idx, :insert_at],
                    pref[row_idx],
                    tok_embeds[row_idx, insert_at:],
                ],
                dim=0,
            )
        )
        attn_rows.append(
            torch.cat(
                [
                    attention_mask[row_idx, :insert_at],
                    pref_mask[row_idx],
                    attention_mask[row_idx, insert_at:],
                ],
                dim=0,
            )
        )
    return torch.stack(embed_rows, dim=0), torch.stack(attn_rows, dim=0)


def retain_topk_kl_loss(
    model,
    prefix: SoftPrefix,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    completion_lens: torch.Tensor,
    completion_cot_mask: torch.Tensor | None = None,
    completion_answer_mask: torch.Tensor | None = None,
    *,
    topk: int,
    temperature: float,
    chunk_tokens: int,
) -> torch.Tensor:
    if topk <= 0:
        raise ValueError("topk must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with torch.no_grad():
        teacher_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        teacher_logits = teacher_out.logits
        actual_topk = min(int(topk), int(teacher_logits.shape[-1]))

    student_embeds, student_attn = insert_prompt_end_prefix_for_batch(model, prefix, input_ids, attention_mask, prompt_lens)
    student_out = model(inputs_embeds=student_embeds, attention_mask=student_attn, use_cache=False)
    student_logits = student_out.logits
    prefix_len = int(prefix.embedding.shape[0])
    sample_losses: list[torch.Tensor] = []
    temp = float(temperature)
    chunk_tokens = max(1, int(chunk_tokens))
    for row_idx in range(input_ids.shape[0]):
        prompt_len = int(prompt_lens[row_idx].item())
        completion_len = int(completion_lens[row_idx].item())
        if completion_len <= 0:
            continue
        # The retain KL target span is exactly the base-generated trajectory:
        # generated CoT tokens, generated answer tokens, and the final EOS token.
        # The prompt is context only and is never part of the KL reduction.
        teacher_positions = torch.arange(
            prompt_len - 1,
            prompt_len + completion_len - 1,
            device=input_ids.device,
            dtype=torch.long,
        )
        student_positions = teacher_positions + prefix_len
        if completion_cot_mask is None or completion_answer_mask is None:
            block_masks = [
                torch.ones((completion_len,), dtype=student_logits.dtype, device=input_ids.device),
            ]
        else:
            block_masks = [
                completion_cot_mask[row_idx, :completion_len].to(device=input_ids.device, dtype=student_logits.dtype),
                completion_answer_mask[row_idx, :completion_len].to(device=input_ids.device, dtype=student_logits.dtype),
            ]
        block_sums = [student_logits.new_zeros(()) for _ in block_masks]
        block_counts = [mask.sum() for mask in block_masks]
        for start in range(0, completion_len, chunk_tokens):
            t_pos = teacher_positions[start : start + chunk_tokens]
            s_pos = student_positions[start : start + chunk_tokens]
            t_logits = teacher_logits[row_idx, t_pos, :].detach()
            s_logits = student_logits[row_idx, s_pos, :]
            t_vals, t_ids = torch.topk(t_logits, k=actual_topk, dim=-1)
            teacher_probs = F.softmax((t_vals / temp).to(torch.float32), dim=-1)
            student_selected = torch.gather(s_logits, dim=-1, index=t_ids)
            student_log_probs = F.log_softmax((student_selected / temp).to(torch.float32), dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            if temp != 1.0:
                kl = kl * (temp * temp)
            for block_idx, mask in enumerate(block_masks):
                block_sums[block_idx] = block_sums[block_idx] + (kl * mask[start : start + kl.shape[0]]).sum()
        normalized_blocks = [
            block_sum / block_count.clamp_min(1.0)
            for block_sum, block_count in zip(block_sums, block_counts)
            if bool((block_count > 0).detach().cpu().item())
        ]
        if normalized_blocks:
            sample_losses.append(torch.stack(normalized_blocks).mean())
    if not sample_losses:
        return student_logits.sum() * 0.0
    return torch.stack(sample_losses).mean()


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train_forget_step(
    *,
    model,
    prefix: SoftPrefix,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(
        model,
        prefix,
        batch["input_ids"].to(args.device),
        batch["attention_mask"].to(args.device),
        batch["labels"].to(args.device),
        batch["loss_weights"].to(args.device),
        batch["prompt_lens"].to(args.device),
        args.prefix_position,
        args.loss_mode,
        batch["cot_mask"].to(args.device),
        batch["answer_mask"].to(args.device),
        float(args.cot_loss_weight),
        float(args.answer_loss_weight),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(prefix.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def train_retain_step(
    *,
    model,
    prefix: SoftPrefix,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = retain_topk_kl_loss(
        model,
        prefix,
        batch["input_ids"].to(args.device),
        batch["attention_mask"].to(args.device),
        batch["prompt_lens"].to(args.device),
        batch["completion_lens"].to(args.device),
        batch["completion_cot_mask"].to(args.device),
        batch["completion_answer_mask"].to(args.device),
        topk=int(args.topk),
        temperature=float(args.kl_temperature),
        chunk_tokens=int(args.kl_chunk_tokens),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(prefix.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def run_generation_eval(
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    *,
    tag: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    prefix_position: str,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(records) / max(1, batch_size))
    for start_idx in range(0, len(records), batch_size):
        batch_records = records[start_idx : start_idx + batch_size]
        completions = generate_batch(
            model,
            prefix,
            tokenizer,
            [record["prompt"] for record in batch_records],
            device,
            max_new_tokens,
            prefix_position,
        )
        for record, completion in zip(batch_records, completions):
            eval_row = {
                **record,
                "tag": tag,
                "completion": completion,
            }
            metrics = evaluate_completion(eval_row, completion, tokenizer, model_cfg)
            reference_answer = str(
                record.get("source_answer")
                or record.get("target_answer")
                or record.get("answer")
                or ""
            )
            row = {
                **eval_row,
                **metrics,
                "reference_answer": reference_answer,
                "generated_answer": str(metrics.get("answer") or ""),
            }
            rows.append(row)
            review_rows.append(
                {
                    "tag": tag,
                    "idx": row.get("idx"),
                    "source_idx": row.get("source_idx"),
                    "source_dataset": row.get("source_dataset"),
                    "source_key": row.get("source_key"),
                    "probe_id": row.get("probe_id"),
                    "task_id": row.get("task_id"),
                    "question": row.get("question"),
                    "reference_answer": row.get("reference_answer"),
                    "generated_answer": row.get("generated_answer"),
                    "cot": row.get("cot"),
                    "completion": row.get("completion"),
                    "clean_refusal": row.get("clean_refusal"),
                    "idk_like": row.get("idk_like"),
                    "source_private_fact_hits": row.get("source_private_fact_hits"),
                    "source_answer_similarity": row.get("source_answer_similarity"),
                }
            )
        batch_no = start_idx // batch_size + 1
        if batch_no == 1 or batch_no % 5 == 0 or batch_no == total_batches:
            log(f"{tag} generation_batch={batch_no}/{total_batches}")
    summary = summarize_generations(tag, rows)
    write_jsonl(output_dir / f"generations_{tag}.jsonl", rows)
    write_jsonl(output_dir / f"manual_review_{tag}.jsonl", review_rows)
    write_csv(output_dir / f"generations_{tag}.csv", rows)
    write_json(output_dir / f"summary_{tag}.json", summary)
    log(f"{tag} " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GUARD retain-trajectory helper training script.")
    p.add_argument("--config", default=None)
    p.add_argument("--stamp", default=None)
    p.add_argument("--model_path", default=str(DEFAULT_MODEL))
    p.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
    p.add_argument("--model_family", default="llama3-8b")
    p.add_argument("--forget_path", default=str(DEFAULT_FORGET))
    p.add_argument("--retain_path", default=str(DEFAULT_RETAIN))
    p.add_argument("--idkcot_path", default=str(DEFAULT_IDKCOT))
    p.add_argument("--idontknow_path", default=str(DEFAULT_IDONTKNOW))
    p.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    p.add_argument("--output_dir", default="")
    p.add_argument("--results_dir", default="")
    p.add_argument("--trajectory_cache", default="")
    p.add_argument("--force_rebuild_trajectories", action="store_true")
    p.add_argument("--init_prefix_path", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task_id", default="1")
    p.add_argument("--forget_limit", type=int, default=40)
    p.add_argument("--retain_limit", type=int, default=40)
    p.add_argument("--retain_sample", default="head", choices=["head", "random"])
    p.add_argument("--retain_extra_forget_path", default="")
    p.add_argument("--retain_extra_exclude_task_id", default="")
    p.add_argument("--dynamic_retain_per_epoch", type=int, default=0)
    p.add_argument("--dynamic_retain_seed", type=int, default=-1)
    p.add_argument(
        "--eval_retain_sample",
        default="train",
        choices=["train", "head_exclude_train", "random_exclude_train"],
    )
    p.add_argument("--eval_retain_seed", type=int, default=-1)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--prefix_position", default="prompt_end")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--microbatch_size", type=int, default=0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--retain_batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--retain_max_new_tokens", type=int, default=384)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_start_epoch", type=int, default=1)
    p.add_argument("--generation_batch_size", type=int, default=10)
    p.add_argument("--trajectory_generation_batch_size", type=int, default=0)
    p.add_argument("--eval_forget_limit", type=int, default=40)
    p.add_argument("--eval_retain_limit", type=int, default=40)
    p.add_argument("--lambda_retain", type=float, default=1.0)
    p.add_argument("--update_schedule", default="joint", choices=["joint", "alternating"])
    p.add_argument("--forget_updates_per_cycle", type=int, default=1)
    p.add_argument("--retain_updates_per_cycle", type=int, default=1)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--kl_temperature", type=float, default=1.0)
    p.add_argument("--kl_chunk_tokens", type=int, default=64)
    p.add_argument("--loss_mode", default="block_ce", choices=["block_ce", "global_ce"])
    p.add_argument("--cot_loss_weight", type=float, default=1.0)
    p.add_argument("--answer_loss_weight", type=float, default=1.0)
    p.add_argument("--skip_final_eval", action="store_true")
    p.add_argument("--skip_baseline_eval", action="store_true")
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"config must be a mapping: {config_path}")
        p.set_defaults(**payload)
    return p.parse_args()


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, str]:
    stamp = str(args.stamp or time.strftime("%Y%m%d_%H%M%S"))
    run_name = str(args.run_name or DEFAULT_RUN_NAME)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{stamp}_{run_name}"
    results_dir = Path(args.results_dir) if args.results_dir else Path(args.results_root) / f"{stamp}_{run_name}"
    return output_dir, results_dir, stamp


def main() -> None:
    args = parse_args()
    args.prefix_position = normalize_prefix_position(args.prefix_position)
    if args.prefix_position != "prompt_end":
        raise ValueError("GUARD retain KL currently supports prompt_end prefix insertion only")
    if int(getattr(args, "microbatch_size", 0) or 0) <= 0:
        args.microbatch_size = int(args.batch_size)
    if int(args.batch_size) % int(args.microbatch_size) != 0:
        raise ValueError("batch_size must be divisible by microbatch_size")
    args.grad_accum_steps = int(args.batch_size) // int(args.microbatch_size)
    if int(getattr(args, "trajectory_generation_batch_size", 0) or 0) <= 0:
        args.trajectory_generation_batch_size = int(args.generation_batch_size)
    if str(getattr(args, "update_schedule", "joint")) == "alternating":
        if int(args.forget_updates_per_cycle) <= 0 or int(args.retain_updates_per_cycle) <= 0:
            raise ValueError("forget_updates_per_cycle and retain_updates_per_cycle must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir, results_dir, stamp = resolve_dirs(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    args.results_dir = str(results_dir)
    args.stamp = stamp
    write_json(output_dir / "train_config.json", vars(args))
    write_json(results_dir / "train_config.json", vars(args))

    model_cfg = get_model_config(args.model_family, args.model_config)
    forget_examples = build_examples(
        Path(args.forget_path),
        Path(args.idkcot_path),
        Path(args.idontknow_path),
        args.task_id,
        args.forget_limit,
    )
    if str(args.retain_extra_forget_path or ""):
        retain_records = build_combined_retain_records(
            Path(args.retain_path),
            model_cfg,
            limit=int(args.retain_limit),
            seed=int(args.seed),
            sample=str(args.retain_sample),
            extra_forget_path=Path(args.retain_extra_forget_path),
            extra_exclude_task_id=str(args.retain_extra_exclude_task_id or args.task_id),
        )
    else:
        retain_records = build_retain_records(
            Path(args.retain_path),
            model_cfg,
            limit=int(args.retain_limit),
            seed=int(args.seed),
            sample=str(args.retain_sample),
        )
    retain_eval_records = build_retain_eval_records(
        Path(args.retain_path),
        model_cfg,
        train_records=retain_records,
        limit=int(args.eval_retain_limit),
        seed=int(args.seed),
        eval_seed=int(args.eval_retain_seed),
        sample=str(args.eval_retain_sample),
    )
    write_jsonl(output_dir / "forget_examples.jsonl", forget_examples)
    write_jsonl(output_dir / "retain_records.jsonl", retain_records)
    write_jsonl(output_dir / "retain_eval_records.jsonl", retain_eval_records)
    write_jsonl(results_dir / "retain_eval_records.jsonl", retain_eval_records)
    train_retain_source_keys = {retain_source_key(record) for record in retain_records}
    eval_retain_source_keys = {retain_source_key(record) for record in retain_eval_records}
    retain_overlap = sorted(train_retain_source_keys & eval_retain_source_keys)
    split_stats = {
        "retain_train_n": len(retain_records),
        "retain_eval_n": len(retain_eval_records),
        "retain_sample": str(args.retain_sample),
        "retain_extra_forget_path": str(args.retain_extra_forget_path or ""),
        "retain_extra_exclude_task_id": str(args.retain_extra_exclude_task_id or args.task_id),
        "eval_retain_sample": str(args.eval_retain_sample),
        "seed": int(args.seed),
        "eval_retain_seed": int(args.eval_retain_seed),
        "dynamic_retain_per_epoch": int(args.dynamic_retain_per_epoch),
        "dynamic_retain_seed": int(args.dynamic_retain_seed),
        "train_dataset_counts": count_by_key(retain_records, "source_dataset"),
        "eval_dataset_counts": count_by_key(retain_eval_records, "source_dataset"),
        "overlap_n": len(retain_overlap),
        "overlap_source_keys": retain_overlap[:200],
    }
    write_json(
        results_dir / "retain_split_stats.json",
        split_stats,
    )
    write_json(output_dir / "retain_split_stats.json", split_stats)
    log(
        f"loaded forget={len(forget_examples)} retain_pool={len(retain_records)} "
        f"retain_eval={len(retain_eval_records)} retain_overlap={len(retain_overlap)} "
        f"lambda_retain={args.lambda_retain} topk={args.topk}"
    )

    model, tokenizer = load_model_and_tokenizer(args.model_path, get_torch_dtype(args.dtype), args.device)
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            log("enabled gradient_checkpointing")
        except Exception as exc:
            log(f"gradient_checkpointing_enable failed: {exc}")
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception as exc:
            log(f"enable_input_require_grads failed: {exc}")
    for param in model.parameters():
        param.requires_grad_(False)

    length_stats = summarize_training_lengths(forget_examples, tokenizer, model_cfg)
    write_json(output_dir / "forget_length_stats.json", length_stats)
    if int(length_stats["full_tokens"]["max"]) > int(args.max_length):
        raise ValueError(
            f"max_length={args.max_length} would truncate forget CE data; "
            f"required_at_least={length_stats['full_tokens']['max']}"
        )

    trajectory_cache = Path(args.trajectory_cache) if args.trajectory_cache else output_dir / "retain_base_trajectories.jsonl"
    retain_trajectories = load_or_generate_retain_trajectories(
        model,
        tokenizer,
        retain_records,
        trajectory_cache,
        device=args.device,
        batch_size=int(args.trajectory_generation_batch_size),
        max_new_tokens=args.retain_max_new_tokens,
        force=bool(args.force_rebuild_trajectories),
    )
    write_jsonl(output_dir / "retain_base_trajectories.used.jsonl", retain_trajectories)

    forget_ds = PrefixDataset(
        forget_examples,
        tokenizer,
        model_cfg,
        int(args.max_length),
    )
    retain_ds = RetainKLDataset(
        retain_trajectories,
        tokenizer,
        max_length=int(args.max_length),
        think_end_tag=str(model_cfg.get("think_end_tag", "\n</think>\n\n")),
    )
    retain_alignment = summarize_retain_alignment(retain_ds)
    write_json(output_dir / "retain_alignment_stats.json", retain_alignment)
    write_json(results_dir / "retain_alignment_stats.json", retain_alignment)
    log("retain_alignment " + json.dumps(retain_alignment, ensure_ascii=False, sort_keys=True))
    forget_loader = DataLoader(
        forget_ds,
        batch_size=int(args.microbatch_size),
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    retain_loader = DataLoader(
        retain_ds,
        batch_size=int(args.retain_batch_size),
        shuffle=True,
        collate_fn=lambda b: retain_collate(b, tokenizer.pad_token_id),
    )

    hidden = int(model.get_input_embeddings().embedding_dim)
    prefix = SoftPrefix(int(args.prefix_len), hidden).to(args.device)
    if args.init_prefix_path:
        ckpt = torch.load(args.init_prefix_path, map_location=args.device)
        state = ckpt.get("prefix_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if "embedding" not in state:
            raise ValueError(f"prefix checkpoint has no embedding parameter: {args.init_prefix_path}")
        if tuple(state["embedding"].shape) != tuple(prefix.embedding.shape):
            raise ValueError(
                f"prefix shape mismatch: checkpoint={tuple(state['embedding'].shape)} "
                f"expected={tuple(prefix.embedding.shape)}"
            )
        prefix.load_state_dict(state)
        log(f"loaded init_prefix_path={args.init_prefix_path}")
    opt = torch.optim.AdamW(prefix.parameters(), lr=float(args.lr), weight_decay=0.0)

    forget_eval_records: list[dict[str, Any]] = []
    for ex in forget_examples[: int(args.eval_forget_limit or len(forget_examples))]:
        forget_eval_records.append({**ex, "prompt": build_prompt(ex["question"], model_cfg)})
    if not args.skip_baseline_eval:
        run_generation_eval(
            model,
            None,
            tokenizer,
            forget_eval_records,
            model_cfg,
            tag="forget_base",
            device=args.device,
            batch_size=int(args.generation_batch_size),
            max_new_tokens=int(args.max_new_tokens),
            prefix_position=args.prefix_position,
            output_dir=results_dir,
        )
        run_generation_eval(
            model,
            None,
            tokenizer,
            retain_eval_records,
            model_cfg,
            tag="retain_base",
            device=args.device,
            batch_size=int(args.generation_batch_size),
            max_new_tokens=int(args.max_new_tokens),
            prefix_position=args.prefix_position,
            output_dir=results_dir,
        )

    train_log: list[dict[str, Any]] = []
    retain_epoch_sample_log: list[dict[str, Any]] = []
    seen_retain_source_keys: set[str] = set()
    global_step = 0
    prefix.train()
    retain_iter = iter(retain_loader)
    for epoch in range(1, int(args.epochs) + 1):
        dynamic_retain_per_epoch = int(args.dynamic_retain_per_epoch)
        epoch_retain_batches: list[dict[str, Any]] | None = None
        epoch_retain_source_keys: list[str] = []
        current_retain_loader = retain_loader
        if dynamic_retain_per_epoch > 0:
            sample_n = min(dynamic_retain_per_epoch, len(retain_ds))
            sample_seed = int(args.dynamic_retain_seed)
            if sample_seed < 0:
                sample_seed = int(args.seed) + 2000003
            rng = random.Random(sample_seed + epoch)
            sampled_indices = rng.sample(range(len(retain_ds)), sample_n)
            epoch_retain_source_keys = [
                retain_source_key(retain_ds.rows[idx]["example"]) for idx in sampled_indices
            ]
            seen_retain_source_keys.update(epoch_retain_source_keys)
            retain_epoch_sample_log.append(
                {
                    "epoch": epoch,
                    "sample_n": sample_n,
                    "unique_seen_n": len(seen_retain_source_keys),
                    "source_keys": epoch_retain_source_keys,
                }
            )
            write_json(output_dir / "retain_epoch_samples.json", retain_epoch_sample_log)
            write_json(results_dir / "retain_epoch_samples.json", retain_epoch_sample_log)
            epoch_retain_loader = DataLoader(
                Subset(retain_ds, sampled_indices),
                batch_size=int(args.retain_batch_size),
                shuffle=True,
                collate_fn=lambda b: retain_collate(b, tokenizer.pad_token_id),
            )
            current_retain_loader = epoch_retain_loader
            epoch_retain_batches = list(epoch_retain_loader)
            log(
                f"epoch {epoch} dynamic_retain sample_n={sample_n} "
                f"unique_seen={len(seen_retain_source_keys)}"
            )
        if str(args.update_schedule) == "alternating":
            cycle_len = int(args.forget_updates_per_cycle) + int(args.retain_updates_per_cycle)
            steps_per_epoch = int(args.steps_per_epoch)
            if steps_per_epoch <= 0:
                forget_batches = len(forget_loader)
                retain_batches = len(current_retain_loader)
                cycles = max(
                    math.ceil(forget_batches / int(args.forget_updates_per_cycle)),
                    math.ceil(retain_batches / int(args.retain_updates_per_cycle)),
                )
                steps_per_epoch = cycles * cycle_len
            forget_iter = cycle_loader(forget_loader)
            retain_iter_epoch = cycle_loader(current_retain_loader)
            epoch_forget: list[float] = []
            epoch_retain: list[float] = []
            forget_updates = 0
            retain_updates = 0
            log(
                f"epoch {epoch} alternating steps={steps_per_epoch} "
                f"update_ratio={args.forget_updates_per_cycle}:{args.retain_updates_per_cycle}"
            )
            for step_in_epoch in range(steps_per_epoch):
                cycle_pos = step_in_epoch % cycle_len
                if cycle_pos < int(args.forget_updates_per_cycle):
                    forget_batch = next(forget_iter)
                    loss = train_forget_step(model=model, prefix=prefix, batch=forget_batch, optimizer=opt, args=args)
                    epoch_forget.append(float(loss.cpu()))
                    forget_updates += 1
                    update_type = "forget"
                else:
                    retain_batch = next(retain_iter_epoch)
                    loss = train_retain_step(model=model, prefix=prefix, batch=retain_batch, optimizer=opt, args=args)
                    epoch_retain.append(float(loss.cpu()))
                    retain_updates += 1
                    update_type = "retain"
                global_step += 1
                if global_step % 100 == 0:
                    log(
                        f"step={global_step} epoch={epoch} update={update_type} "
                        f"loss={float(loss.cpu()):.6f}"
                    )
            row = {
                "epoch": epoch,
                "step": global_step,
                "forget_ce": float(sum(epoch_forget) / max(1, len(epoch_forget))),
                "retain_topk_kl": float(sum(epoch_retain) / max(1, len(epoch_retain))),
                "total_loss": float(
                    (sum(epoch_forget) + sum(epoch_retain)) / max(1, len(epoch_forget) + len(epoch_retain))
                ),
                "update_schedule": "alternating",
                "forget_updates": forget_updates,
                "retain_updates": retain_updates,
                "dynamic_retain_epoch_n": len(epoch_retain_source_keys),
                "dynamic_retain_unique_seen": len(seen_retain_source_keys),
            }
            train_log.append(row)
            write_json(output_dir / "train_log.json", train_log)
            write_json(results_dir / "train_log.json", train_log)
            log("epoch " + json.dumps(row, ensure_ascii=False, sort_keys=True))

            should_eval = (
                int(args.eval_every) > 0
                and epoch >= int(args.eval_start_epoch)
                and (epoch % int(args.eval_every) == 0 or epoch == int(args.epochs))
            )
            if should_eval:
                ckpt = {
                    "prefix_state_dict": prefix.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "train_row": row,
                }
                torch.save(ckpt, output_dir / f"prefix_epoch{epoch}.pt")
                prefix.eval()
                run_generation_eval(
                    model,
                    prefix,
                    tokenizer,
                    forget_eval_records,
                    model_cfg,
                    tag=f"forget_epoch{epoch}",
                    device=args.device,
                    batch_size=int(args.generation_batch_size),
                    max_new_tokens=int(args.max_new_tokens),
                    prefix_position=args.prefix_position,
                    output_dir=results_dir,
                )
                run_generation_eval(
                    model,
                    prefix,
                    tokenizer,
                    retain_eval_records,
                    model_cfg,
                    tag=f"retain_epoch{epoch}",
                    device=args.device,
                    batch_size=int(args.generation_batch_size),
                    max_new_tokens=int(args.max_new_tokens),
                    prefix_position=args.prefix_position,
                    output_dir=results_dir,
                )
                prefix.train()
            continue
        accum_steps = max(1, int(args.grad_accum_steps))
        opt.zero_grad(set_to_none=True)
        epoch_forget: list[float] = []
        epoch_retain: list[float] = []
        epoch_total: list[float] = []
        retain_batch_cursor = 0
        retain_batches_per_forget = 1
        if epoch_retain_batches is not None:
            retain_batches_per_forget = max(1, math.ceil(len(epoch_retain_batches) / max(1, len(forget_loader))))
        for micro_step, forget_batch in enumerate(forget_loader, start=1):
            if epoch_retain_batches is not None:
                retain_group = epoch_retain_batches[
                    retain_batch_cursor : retain_batch_cursor + retain_batches_per_forget
                ]
                retain_batch_cursor += retain_batches_per_forget
                if not retain_group:
                    retain_group = [epoch_retain_batches[-1]]
            else:
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)
                retain_group = [retain_batch]

            input_ids = forget_batch["input_ids"].to(args.device)
            attention_mask = forget_batch["attention_mask"].to(args.device)
            labels = forget_batch["labels"].to(args.device)
            loss_weights = forget_batch["loss_weights"].to(args.device)
            cot_mask = forget_batch["cot_mask"].to(args.device)
            answer_mask = forget_batch["answer_mask"].to(args.device)
            prompt_lens = forget_batch["prompt_lens"].to(args.device)

            forget_loss = forward_loss(
                model,
                prefix,
                input_ids,
                attention_mask,
                labels,
                loss_weights,
                prompt_lens,
                args.prefix_position,
                args.loss_mode,
                cot_mask,
                answer_mask,
                float(args.cot_loss_weight),
                float(args.answer_loss_weight),
            )
            (forget_loss / float(accum_steps)).backward()
            retain_loss_values: list[float] = []
            for retain_batch in retain_group:
                retain_input_ids = retain_batch["input_ids"].to(args.device)
                retain_attention_mask = retain_batch["attention_mask"].to(args.device)
                retain_prompt_lens = retain_batch["prompt_lens"].to(args.device)
                retain_completion_lens = retain_batch["completion_lens"].to(args.device)
                retain_loss = retain_topk_kl_loss(
                    model,
                    prefix,
                    retain_input_ids,
                    retain_attention_mask,
                    retain_prompt_lens,
                    retain_completion_lens,
                    retain_batch["completion_cot_mask"].to(args.device),
                    retain_batch["completion_answer_mask"].to(args.device),
                    topk=int(args.topk),
                    temperature=float(args.kl_temperature),
                    chunk_tokens=int(args.kl_chunk_tokens),
                )
                retain_loss_values.append(float(retain_loss.detach().cpu()))
                retain_scale = float(args.lambda_retain) / float(max(1, len(retain_group)) * accum_steps)
                (retain_loss * retain_scale).backward()
            retain_loss_value = float(sum(retain_loss_values) / max(1, len(retain_loss_values)))
            total_loss_value = float(forget_loss.detach().cpu()) + float(args.lambda_retain) * retain_loss_value
            epoch_forget.append(float(forget_loss.detach().cpu()))
            epoch_retain.append(retain_loss_value)
            epoch_total.append(total_loss_value)
            if micro_step % accum_steps == 0 or micro_step == len(forget_loader):
                torch.nn.utils.clip_grad_norm_(prefix.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

        row = {
            "epoch": epoch,
            "step": global_step,
            "forget_ce": float(sum(epoch_forget) / max(1, len(epoch_forget))),
            "retain_topk_kl": float(sum(epoch_retain) / max(1, len(epoch_retain))),
            "total_loss": float(sum(epoch_total) / max(1, len(epoch_total))),
            "dynamic_retain_epoch_n": len(epoch_retain_source_keys),
            "dynamic_retain_unique_seen": len(seen_retain_source_keys),
        }
        train_log.append(row)
        write_json(output_dir / "train_log.json", train_log)
        write_json(results_dir / "train_log.json", train_log)
        log("epoch " + json.dumps(row, ensure_ascii=False, sort_keys=True))

        should_eval = (
            int(args.eval_every) > 0
            and epoch >= int(args.eval_start_epoch)
            and (epoch % int(args.eval_every) == 0 or epoch == int(args.epochs))
        )
        if should_eval:
            ckpt = {
                "prefix_state_dict": prefix.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "train_row": row,
            }
            torch.save(ckpt, output_dir / f"prefix_epoch{epoch}.pt")
            prefix.eval()
            run_generation_eval(
                model,
                prefix,
                tokenizer,
                forget_eval_records,
                model_cfg,
                tag=f"forget_epoch{epoch}",
                device=args.device,
                batch_size=int(args.generation_batch_size),
                max_new_tokens=int(args.max_new_tokens),
                prefix_position=args.prefix_position,
                output_dir=results_dir,
            )
            run_generation_eval(
                model,
                prefix,
                tokenizer,
                retain_eval_records,
                model_cfg,
                tag=f"retain_epoch{epoch}",
                device=args.device,
                batch_size=int(args.generation_batch_size),
                max_new_tokens=int(args.max_new_tokens),
                prefix_position=args.prefix_position,
                output_dir=results_dir,
            )
            prefix.train()

    torch.save({"prefix_state_dict": prefix.state_dict(), "args": vars(args), "epoch": int(args.epochs)}, output_dir / "prefix_final.pt")
    if not args.skip_final_eval:
        prefix.eval()
        run_generation_eval(
            model,
            prefix,
            tokenizer,
            forget_eval_records,
            model_cfg,
            tag="forget_final",
            device=args.device,
            batch_size=int(args.generation_batch_size),
            max_new_tokens=int(args.max_new_tokens),
            prefix_position=args.prefix_position,
            output_dir=results_dir,
        )
        run_generation_eval(
            model,
            prefix,
            tokenizer,
            retain_eval_records,
            model_cfg,
            tag="retain_final",
            device=args.device,
            batch_size=int(args.generation_batch_size),
            max_new_tokens=int(args.max_new_tokens),
            prefix_position=args.prefix_position,
            output_dir=results_dir,
        )
    log(f"done output={output_dir} results={results_dir}")


if __name__ == "__main__":
    main()
