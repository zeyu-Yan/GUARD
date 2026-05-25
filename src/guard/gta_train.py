#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

GUARD_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("GUARD_PROJECT_ROOT", str(GUARD_ROOT)))

from guard.prefix_utils import (
    DEFAULT_IDKCOT,
    DEFAULT_IDONTKNOW,
    DEFAULT_MODEL as DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_CONFIG,
    SoftPrefix,
    build_examples,
    build_prompt,
    cleanup_generated_text,
    evaluate_completion,
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
from guard.retain_utils import (
    build_combined_retain_records,
    build_retain_eval_records,
    build_retain_records,
    load_or_generate_retain_trajectories,
    retain_source_key,
)
from guard.selection_judge import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL as DEFAULT_JUDGE_MODEL,
    JudgeConfig,
    judge_forget_row,
    judge_star_safety_forget_row,
    judge_and_score_epoch,
    validate_judge_config,
)

CHOICE_LABELS = ["A", "B", "C", "D"]
DEFAULT_FORGET = PROJECT_ROOT / "data" / "forget.json"
DEFAULT_RETAIN = PROJECT_ROOT / "data" / "retain.json"
DEFAULT_RUN_NAME = "guard_gta"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gta_runs"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "gta_results"


def build_direct_prompt(question: str, model_cfg: dict[str, Any]) -> str:
    question_start = str(model_cfg.get("question_start_tag") or "")
    question_end = str(model_cfg.get("question_end_tag") or "")
    return f"{question_start}{question}{question_end}"


def build_prefix_training_prompt(question: str, model_cfg: dict[str, Any], prefix_position: str) -> str:
    if normalize_prefix_position(prefix_position) != "prompt_end":
        raise ValueError("GUARD GTA forget CE + retain KL uses prompt_end prefix insertion only")
    return build_prompt(question, model_cfg)


def apply_prefix_training_prompt(
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    prefix_position: str,
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "prompt": build_prefix_training_prompt(str(record.get("question") or ""), model_cfg, prefix_position),
        }
        for record in records
    ]


def attach_raw_forget_fields(examples: list[dict[str, Any]], forget_path: Path) -> list[dict[str, Any]]:
    """Attach original CoT/answer fields used as Prefix-DPO rejected completions."""
    raw_rows = [row for row in load_json_or_jsonl(forget_path) if isinstance(row, dict)]
    enriched: list[dict[str, Any]] = []
    for ex in examples:
        row = raw_rows[int(ex.get("source_idx", -1))] if 0 <= int(ex.get("source_idx", -1)) < len(raw_rows) else {}
        raw_cot = cleanup_generated_text(str(row.get("cot") or row.get("raw_cot") or ex.get("raw_cot") or ""))
        raw_answer = cleanup_generated_text(
            str(row.get("answer") or row.get("source_answer") or ex.get("source_answer") or "")
        )
        enriched.append({**ex, "raw_cot": raw_cot, "raw_answer": raw_answer})
    return enriched


def dpo_record_key(record: dict[str, Any]) -> str:
    if record.get("probe_id"):
        return str(record["probe_id"])
    return f"task{record.get('task_id', '')}:{int(record.get('source_idx', record.get('idx', -1)))}"


def log(msg: str) -> None:
    print(f"[guard-gta] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for record in records:
        for key in record:
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


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


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
    total_batches = math.ceil(len(records) / max(1, batch_size))
    generation_prompts = [str(record["prompt"]) for record in records]
    for start_idx in range(0, len(records), batch_size):
        batch_records = records[start_idx : start_idx + batch_size]
        batch_prompts = generation_prompts[start_idx : start_idx + batch_size]
        completions = generate_batch(
            model,
            prefix,
            tokenizer,
            batch_prompts,
            device,
            max_new_tokens,
            prefix_position,
        )
        for record, completion in zip(batch_records, completions):
            eval_row = {**record, "tag": tag, "completion": completion}
            rows.append({**eval_row, **evaluate_completion(eval_row, completion, tokenizer, model_cfg)})
        batch_no = start_idx // batch_size + 1
        if batch_no == 1 or batch_no % 5 == 0 or batch_no == total_batches:
            log(f"{tag} generation_batch={batch_no}/{total_batches}")
    summary = summarize_generations(tag, rows)
    write_jsonl(output_dir / f"generations_{tag}.jsonl", rows)
    write_csv(output_dir / f"generations_{tag}.csv", rows)
    write_json(output_dir / f"summary_{tag}.json", summary)
    log(f"{tag} " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def sample_selection_records(
    forget_records: list[dict[str, Any]],
    retain_records: list[dict[str, Any]],
    *,
    sample_n: int,
    retain_sample_n: int | None = None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sample_n = int(sample_n)
    retain_n = int(sample_n if retain_sample_n is None else retain_sample_n)
    if sample_n <= 0:
        return [], [], {"sample_n": 0, "retain_sample_n": max(0, retain_n), "forget_indices": [], "retain_indices": []}
    if len(forget_records) < sample_n:
        raise ValueError(f"score_sample_n={sample_n} exceeds forget records={len(forget_records)}")
    if retain_n <= 0:
        retain_n = sample_n
    if len(retain_records) < retain_n:
        raise ValueError(f"score_retain_sample_n={retain_n} exceeds retain records={len(retain_records)}")
    forget_rng = random.Random(int(seed))
    retain_rng = random.Random(int(seed) + 1000)
    forget_indices = sorted(forget_rng.sample(range(len(forget_records)), sample_n))
    retain_indices = sorted(retain_rng.sample(range(len(retain_records)), retain_n))
    forget_eval = [forget_records[idx] for idx in forget_indices]
    retain_eval = [retain_records[idx] for idx in retain_indices]
    return (
        forget_eval,
        retain_eval,
        {
            "sample_n": sample_n,
            "forget_sample_n": sample_n,
            "retain_sample_n": retain_n,
            "sample_seed": int(seed),
            "forget_indices": [int(row.get("idx", idx)) for idx, row in zip(forget_indices, forget_eval)],
            "retain_indices": [int(row.get("idx", idx)) for idx, row in zip(retain_indices, retain_eval)],
            "forget_source_indices": [int(row.get("source_idx", -1)) for row in forget_eval],
            "retain_source_keys": [retain_source_key(row) for row in retain_eval],
        },
    )


def build_forget_probe_records(
    forget_path: Path,
    model_cfg: dict[str, Any],
    *,
    task_id: str,
    limit: int = 0,
) -> list[dict[str, Any]]:
    rows = [row for row in load_json_or_jsonl(forget_path) if isinstance(row, dict)]
    filtered = [(idx, row) for idx, row in enumerate(rows) if str(row.get("task_id")) == str(task_id)]
    if limit > 0:
        filtered = filtered[: int(limit)]
    records: list[dict[str, Any]] = []
    dataset_name = forget_path.stem
    for local_idx, (source_idx, row) in enumerate(filtered):
        question = str(row.get("question") or "").strip()
        source_answer = str(row.get("answer") or row.get("source_answer") or question).strip()
        if not question:
            continue
        records.append(
            {
                "idx": local_idx,
                "source_idx": source_idx,
                "probe_id": f"{dataset_name}_task{task_id}_{source_idx:05d}",
                "task_id": str(row.get("task_id", task_id)),
                "question": question,
                "source_answer": source_answer,
                "target_answer": str(row.get("target_answer") or ""),
                "prompt": build_prompt(question, model_cfg),
            }
        )
    if not records:
        raise ValueError(f"no forget probe records found for task_id={task_id!r} from {forget_path}")
    return records


def sample_forget_in_out_records(
    in_records: list[dict[str, Any]],
    out_records: list[dict[str, Any]],
    *,
    in_sample_n: int,
    out_sample_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    in_sample_n = int(in_sample_n)
    out_sample_n = int(out_sample_n)
    if in_sample_n <= 0 or out_sample_n <= 0:
        return [], [], {
            "in_sample_n": max(0, in_sample_n),
            "out_sample_n": max(0, out_sample_n),
            "in_indices": [],
            "out_indices": [],
            "in_requested_sample_n": max(0, in_sample_n),
            "out_requested_sample_n": max(0, out_sample_n),
        }
    if not in_records:
        raise ValueError("cannot sample in-sample forget records from an empty pool")
    if not out_records:
        raise ValueError("cannot sample out-of-sample forget records from an empty pool")
    in_rng = random.Random(int(seed))
    out_rng = random.Random(int(seed) + 2000)
    requested_in_sample_n = in_sample_n
    requested_out_sample_n = out_sample_n
    in_sample_n = min(in_sample_n, len(in_records))
    out_sample_n = min(out_sample_n, len(out_records))
    in_indices = sorted(in_rng.sample(range(len(in_records)), in_sample_n))
    out_indices = sorted(out_rng.sample(range(len(out_records)), out_sample_n))
    in_eval = [in_records[idx] for idx in in_indices]
    out_eval = [out_records[idx] for idx in out_indices]
    return (
        in_eval,
        out_eval,
        {
            "sample_seed": int(seed),
            "in_sample_n": in_sample_n,
            "out_sample_n": out_sample_n,
            "in_requested_sample_n": int(requested_in_sample_n),
            "out_requested_sample_n": int(requested_out_sample_n),
            "in_indices": [int(row.get("idx", idx)) for idx, row in zip(in_indices, in_eval)],
            "out_indices": [int(row.get("idx", idx)) for idx, row in zip(out_indices, out_eval)],
            "in_source_indices": [int(row.get("source_idx", -1)) for row in in_eval],
            "out_source_indices": [int(row.get("source_idx", -1)) for row in out_eval],
        },
    )


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if bool(row.get(key))) / len(rows))


def _write_judgments(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_existing_judgments(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [row for row in load_json_or_jsonl(path) if isinstance(row, dict)]
    return {
        int(row["sample_idx"]): row
        for row in rows
        if str(row.get("sample_idx", "")).strip() != ""
    }


def _position_ids_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _mmlu_gold_label(record: dict[str, Any]) -> int:
    for key in ("source_answer", "target_answer", "answer"):
        text = str(record.get(key) or "")
        match = re.match(r"\s*([A-Da-d])(?:[.)：:]\s*|\s+|$)", text)
        if match:
            return CHOICE_LABELS.index(match.group(1).upper())
    raise ValueError(f"cannot parse MMLU gold label for source_id={record.get('source_id')}")


def _parse_choice_block(text: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for label in CHOICE_LABELS:
        match = re.search(rf"(?ms)^{label}\.\s*(.*?)(?=^[A-D]\.\s*|\Z)", text)
        if match:
            choices[label] = re.sub(r"\s+", " ", match.group(1).strip())
    return choices


def build_mmlu_logprob_prompt(record: dict[str, Any]) -> str:
    raw = str(record.get("question") or "").strip()
    raw = re.sub(r"\n\s*Answer with the correct option letter\.?\s*$", "", raw, flags=re.IGNORECASE)
    subject_match = re.search(r"(?mi)^Subject:\s*(.+?)\s*$", raw)
    question_match = re.search(r"(?is)(?:^|\n)Question:\s*(.*?)(?:\n\s*Choices:\s*)", raw)
    choices_match = re.search(r"(?is)(?:^|\n)Choices:\s*(.*)$", raw)
    if subject_match and question_match and choices_match:
        subject = re.sub(r"\s+", " ", subject_match.group(1).strip())
        question = question_match.group(1).strip()
        choices = _parse_choice_block(choices_match.group(1).strip())
        if len(choices) == len(CHOICE_LABELS):
            return (
                f"The following are multiple choice questions (with answers) about {subject}.\n\n"
                + question
                + "\n"
                + "\n".join(f"{label}. {choices[label]}" for label in CHOICE_LABELS)
                + "\nAnswer:"
            )
    fallback = raw
    if not fallback.endswith("Answer:"):
        fallback = fallback.rstrip() + "\nAnswer:"
    return fallback


def _continuation_choice_tokens(tokenizer, prompt: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False).input_ids
    choice_token_ids: list[int] = []
    for label in CHOICE_LABELS:
        full_ids = tokenizer(prompt + " " + label, add_special_tokens=True, truncation=False).input_ids
        if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) == len(prompt_ids) + 1:
            choice_token_ids.append(int(full_ids[len(prompt_ids)]))
            continue
        choice_ids = tokenizer(" " + label, add_special_tokens=False, truncation=False).input_ids
        if len(choice_ids) != 1:
            raise ValueError(f"non single-token MMLU label={label!r} token_ids={choice_ids}")
        choice_token_ids.append(int(choice_ids[0]))
    return [int(token_id) for token_id in prompt_ids], choice_token_ids


def _collate_mcq_prompts(samples: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(sample["prompt_ids"]) for sample in samples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    choice_token_ids: list[list[int]] = []
    gold_labels: list[int] = []
    for sample in samples:
        ids = [int(token_id) for token_id in sample["prompt_ids"]]
        pad = max_len - len(ids)
        input_ids.append([int(pad_token_id)] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
        choice_token_ids.append([int(token_id) for token_id in sample["choice_token_ids"]])
        gold_labels.append(int(sample.get("gold", -1)))
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    if choice_token_ids:
        batch["choice_token_ids"] = torch.tensor(choice_token_ids, dtype=torch.long)
        batch["gold"] = torch.tensor(gold_labels, dtype=torch.long)
    return batch


def _forward_last_logits(model, **kwargs) -> torch.Tensor:
    try:
        out = model(**kwargs, use_cache=False, logits_to_keep=1)
    except TypeError:
        out = model(**kwargs, use_cache=False)
    return out.logits[:, -1, :].float()


def build_retain_mcq_samples(
    retain_records: list[dict[str, Any]],
    tokenizer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    skipped = 0
    for sample_idx, record in enumerate(retain_records):
        try:
            prompt = build_mmlu_logprob_prompt(record)
            prompt_ids, choice_token_ids = _continuation_choice_tokens(tokenizer, prompt)
            gold = _mmlu_gold_label(record)
        except Exception:
            skipped += 1
            continue
        samples.append(
            {
                "sample_idx": sample_idx,
                "source_key": retain_source_key(record),
                "record": record,
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "choice_token_ids": choice_token_ids,
                "gold": gold,
            }
        )
    stats = {
        "pool_n": len(retain_records),
        "samples": len(samples),
        "skipped": skipped,
    }
    return samples, stats


def sample_retain_mcq_samples_for_epoch(
    samples: list[dict[str, Any]],
    *,
    epoch: int,
    seed: int,
    sample_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_n = min(max(0, int(sample_n)), len(samples))
    if sample_n <= 0:
        return [], {
            "pool_n": len(samples),
            "sample_n": 0,
            "sampled_n": 0,
            "sample_seed": int(seed),
            "epoch": int(epoch),
            "sample_indices": [],
            "with_replacement": False,
        }
    rng = random.Random(int(seed) + int(epoch))
    indices = sorted(rng.sample(range(len(samples)), sample_n))
    sampled = [samples[idx] for idx in indices]
    return sampled, {
        "pool_n": len(samples),
        "sample_n": sample_n,
        "sampled_n": len(sampled),
        "sample_seed": int(seed),
        "epoch": int(epoch),
        "sample_indices": [int(samples[idx].get("sample_idx", idx)) for idx in indices],
        "with_replacement": False,
    }


class MCQPromptDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        if not rows:
            raise ValueError("empty MCQ prompt dataset")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def retain_mcq_logit_kl_loss(
    model,
    prefix: SoftPrefix,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    choice_token_ids = batch["choice_token_ids"]
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError("retain_mcq_temperature must be positive")

    with torch.no_grad():
        teacher_logits = _forward_last_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=_position_ids_from_attention(attention_mask),
        )

    tok_embeds = model.get_input_embeddings()(input_ids)
    pref = prefix(input_ids.shape[0]).to(device=input_ids.device, dtype=tok_embeds.dtype)
    pref_mask = torch.ones((input_ids.shape[0], pref.shape[1]), dtype=attention_mask.dtype, device=input_ids.device)
    student_embeds = torch.cat([tok_embeds, pref], dim=1)
    student_attn = torch.cat([attention_mask, pref_mask], dim=1)
    student_logits = _forward_last_logits(
        model,
        inputs_embeds=student_embeds,
        attention_mask=student_attn,
        position_ids=_position_ids_from_attention(student_attn),
    )

    teacher_choice_logits = torch.gather(teacher_logits, dim=-1, index=choice_token_ids)
    student_choice_logits = torch.gather(student_logits, dim=-1, index=choice_token_ids)
    teacher_probs = F.softmax(teacher_choice_logits / temp, dim=-1)
    student_log_probs = F.log_softmax(student_choice_logits / temp, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
    with torch.no_grad():
        teacher_pred = torch.argmax(teacher_choice_logits, dim=-1)
        student_pred = torch.argmax(student_choice_logits, dim=-1)
        agree = (teacher_pred == student_pred).to(torch.float32).mean()
        teacher_entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1).mean()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "mcq_kl": float(loss.detach().cpu()),
            "teacher_student_agree": float(agree.detach().cpu()),
            "teacher_entropy": float(teacher_entropy.detach().cpu()),
            "temperature": temp,
            "batch_size": float(input_ids.shape[0]),
        }
    return loss, metrics


@torch.inference_mode()
def score_retain_mcq_logprob(
    *,
    epoch: int,
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    retain_records: list[dict[str, Any]],
    device: str,
    batch_size: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_idx, record in enumerate(retain_records):
        prompt = build_mmlu_logprob_prompt(record)
        prompt_ids, choice_token_ids = _continuation_choice_tokens(tokenizer, prompt)
        gold = _mmlu_gold_label(record)
        samples.append(
            {
                "sample_idx": sample_idx,
                "record": record,
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "choice_token_ids": choice_token_ids,
                "gold": gold,
            }
        )

    rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(samples) / max(1, int(batch_size)))
    for start_idx in range(0, len(samples), max(1, int(batch_size))):
        batch_samples = samples[start_idx : start_idx + max(1, int(batch_size))]
        batch = _collate_mcq_prompts(batch_samples, int(tokenizer.pad_token_id))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if prefix is None:
            logits = _forward_last_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=_position_ids_from_attention(attention_mask),
            )
        else:
            tok_embeds = model.get_input_embeddings()(input_ids)
            pref = prefix(input_ids.shape[0]).to(device=input_ids.device, dtype=tok_embeds.dtype)
            pref_mask = torch.ones((input_ids.shape[0], pref.shape[1]), dtype=attention_mask.dtype, device=input_ids.device)
            inputs_embeds = torch.cat([tok_embeds, pref], dim=1)
            attn = torch.cat([attention_mask, pref_mask], dim=1)
            logits = _forward_last_logits(
                model,
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=_position_ids_from_attention(attn),
            )
        log_den = torch.logsumexp(logits, dim=-1)
        for row_idx, sample in enumerate(batch_samples):
            token_scores = [
                float((logits[row_idx, int(token_id)] - log_den[row_idx]).item())
                for token_id in sample["choice_token_ids"]
            ]
            pred = max(range(len(CHOICE_LABELS)), key=lambda idx: token_scores[idx])
            gold = int(sample["gold"])
            wrong_ll = max(float(token_scores[idx]) for idx in range(len(CHOICE_LABELS)) if idx != gold)
            record = sample["record"]
            rows.append(
                {
                    "epoch": int(epoch),
                    "split": "retain",
                    "sample_idx": int(sample["sample_idx"]),
                    "idx": record.get("idx"),
                    "source_idx": record.get("source_idx"),
                    "source_dataset": record.get("source_dataset"),
                    "source_key": record.get("source_key"),
                    "source_id": record.get("source_id"),
                    "question": record.get("question"),
                    "source_answer": record.get("source_answer") or record.get("answer"),
                    "prompt": sample["prompt"],
                    "gold": gold,
                    "gold_label": CHOICE_LABELS[gold],
                    "pred": int(pred),
                    "pred_label": CHOICE_LABELS[pred],
                    "correct": int(pred == gold),
                    "fact_hit": bool(pred == gold),
                    "gold_margin": float(token_scores[gold] - wrong_ll),
                    "ll_A": float(token_scores[0]),
                    "ll_B": float(token_scores[1]),
                    "ll_C": float(token_scores[2]),
                    "ll_D": float(token_scores[3]),
                }
            )
        batch_no = start_idx // max(1, int(batch_size)) + 1
        if batch_no == 1 or batch_no % 5 == 0 or batch_no == total_batches:
            log(f"score_retain_mcq_epoch{int(epoch)} batch={batch_no}/{total_batches}")

    correct = int(sum(int(row["correct"]) for row in rows))
    margins = [float(row["gold_margin"]) for row in rows]
    summary = {
        "tag": f"score_retain_mcq_epoch{int(epoch)}",
        "mode": "mcq_logprob",
        "prompt_format": "mmlu_lm_eval",
        "n": len(rows),
        "correct": correct,
        "acc": float(correct / len(rows)) if rows else 0.0,
        "mean_gold_margin": float(sum(margins) / len(margins)) if margins else 0.0,
        "negative_margin_count": int(sum(1 for value in margins if value < 0)),
        "nonpositive_margin_count": int(sum(1 for value in margins if value <= 0)),
    }
    write_jsonl(output_dir / f"mcq_retain_epoch{int(epoch)}.jsonl", rows)
    write_csv(output_dir / f"mcq_retain_epoch{int(epoch)}.csv", rows)
    write_json(output_dir / f"mcq_retain_summary_epoch{int(epoch)}.json", summary)
    log(f"score_retain_mcq_epoch{int(epoch)} " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return rows


def _normalize_mcq_option(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def build_forget_mcq_samples(records: list[dict[str, Any]], *, sample_n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        return [], {"pool_n": 0, "sample_n": 0, "sampled_n": 0, "seed": int(seed)}
    rng = random.Random(int(seed))
    sample_n = min(max(0, int(sample_n)), len(records))
    if sample_n <= 0:
        return [], {"pool_n": len(records), "sample_n": 0, "sampled_n": 0, "seed": int(seed)}
    sampled_indices = sorted(rng.sample(range(len(records)), sample_n))
    by_offset: dict[int, list[int]] = {}
    for idx in range(len(records)):
        by_offset.setdefault(idx % 20, []).append(idx)
    samples: list[dict[str, Any]] = []
    for idx in sampled_indices:
        record = records[idx]
        gold_answer = _normalize_mcq_option(record.get("source_answer") or record.get("answer") or "")
        if not gold_answer:
            continue
        candidates = [
            cand_idx
            for cand_idx in by_offset.get(idx % 20, [])
            if cand_idx != idx
            and _normalize_mcq_option(records[cand_idx].get("source_answer") or records[cand_idx].get("answer") or "")
            != gold_answer
        ]
        if len(candidates) < 3:
            candidates = [
                cand_idx
                for cand_idx in range(len(records))
                if cand_idx != idx
                and _normalize_mcq_option(records[cand_idx].get("source_answer") or records[cand_idx].get("answer") or "")
                != gold_answer
            ]
        if len(candidates) < 3:
            continue
        distractor_indices = rng.sample(candidates, 3)
        option_items = [(gold_answer, True, idx)] + [
            (
                _normalize_mcq_option(records[cand_idx].get("source_answer") or records[cand_idx].get("answer") or ""),
                False,
                cand_idx,
            )
            for cand_idx in distractor_indices
        ]
        rng.shuffle(option_items)
        options = [item[0] for item in option_items]
        gold = next(pos for pos, item in enumerate(option_items) if item[1])
        choice_block = "\n".join(f"{label}. {option}" for label, option in zip(CHOICE_LABELS, options))
        question_prefix = (
            "The following are multiple choice questions (with answers) about private fictional author facts.\n\n"
            f"Question: {record.get('question')}\n\n"
        )
        choices_prefix = f"Choices:\n{choice_block}\n\n"
        answer_marker = "Answer:"
        before_answer = question_prefix + choices_prefix
        prompt = before_answer + answer_marker
        samples.append(
            {
                "sample_idx": int(idx),
                "source_idx": int(record.get("source_idx", idx)),
                "probe_id": record.get("probe_id"),
                "task_id": str(record.get("task_id", "")),
                "question": str(record.get("question") or ""),
                "source_answer": gold_answer,
                "template_offset": int(idx % 20),
                "prefix_before": question_prefix,
                "prefix_after": choices_prefix + answer_marker,
                "before_answer": before_answer,
                "answer_marker": answer_marker,
                "prompt": prompt,
                "options": options,
                "gold": int(gold),
                "gold_label": CHOICE_LABELS[gold],
                "distractor_sample_indices": [int(item[2]) for item in option_items if not item[1]],
            }
        )
    return samples, {
        "pool_n": len(records),
        "sample_n": int(sample_n),
        "sampled_n": len(samples),
        "seed": int(seed),
        "sample_indices": sampled_indices,
    }

def _tokenize_forget_mcq_samples(samples: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        prompt = str(sample["prompt"])
        prefix_before = str(sample.get("prefix_before") or sample["before_answer"])
        prefix_after = str(sample.get("prefix_after") or sample["answer_marker"])
        prompt_ids, choice_token_ids = _continuation_choice_tokens(tokenizer, prompt)
        rows.append(
            {
                **sample,
                "prompt_ids": prompt_ids,
                "choice_token_ids": choice_token_ids,
                "prefix_before_ids": [
                    int(token_id)
                    for token_id in tokenizer(prefix_before, add_special_tokens=True, truncation=False).input_ids
                ],
                "prefix_after_ids": [
                    int(token_id)
                    for token_id in tokenizer(prefix_after, add_special_tokens=False, truncation=False).input_ids
                ],
            }
        )
    return rows


def _collate_forget_mcq_prefix_embeds(
    samples: list[dict[str, Any]],
    *,
    model,
    prefix: SoftPrefix,
    pad_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embed_layer = model.get_input_embeddings()
    pad_ids = torch.tensor([int(pad_token_id)], dtype=torch.long, device=device)
    pad_embed = embed_layer(pad_ids)[0]
    row_embeds: list[torch.Tensor] = []
    lengths: list[int] = []
    for sample in samples:
        before_ids = torch.tensor(sample["prefix_before_ids"], dtype=torch.long, device=device)
        after_ids = torch.tensor(sample["prefix_after_ids"], dtype=torch.long, device=device)
        before_embeds = embed_layer(before_ids)
        after_embeds = embed_layer(after_ids)
        pref = prefix(1).squeeze(0).to(device=device, dtype=before_embeds.dtype)
        embeds = torch.cat([before_embeds, pref, after_embeds], dim=0)
        row_embeds.append(embeds)
        lengths.append(int(embeds.shape[0]))
    max_len = max(lengths)
    padded: list[torch.Tensor] = []
    attention: list[list[int]] = []
    for embeds, length in zip(row_embeds, lengths):
        pad = max_len - length
        if pad > 0:
            pad_block = pad_embed.to(dtype=embeds.dtype).unsqueeze(0).expand(pad, -1)
            embeds = torch.cat([pad_block, embeds], dim=0)
        padded.append(embeds)
        attention.append([0] * pad + [1] * length)
    return torch.stack(padded, dim=0), torch.tensor(attention, dtype=torch.long, device=device)


@torch.inference_mode()
def score_forget_mcq_logprob(
    *,
    epoch: int,
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    forget_records: list[dict[str, Any]],
    device: str,
    batch_size: int,
    sample_n: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_samples, sample_meta = build_forget_mcq_samples(
        forget_records,
        sample_n=int(sample_n),
        seed=int(seed) + int(epoch),
    )
    samples = _tokenize_forget_mcq_samples(raw_samples, tokenizer)
    rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(samples) / max(1, int(batch_size)))
    for start_idx in range(0, len(samples), max(1, int(batch_size))):
        batch_samples = samples[start_idx : start_idx + max(1, int(batch_size))]
        if prefix is None:
            batch = _collate_mcq_prompts(batch_samples, int(tokenizer.pad_token_id))
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = _forward_last_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=_position_ids_from_attention(attention_mask),
            )
        else:
            inputs_embeds, attention_mask = _collate_forget_mcq_prefix_embeds(
                batch_samples,
                model=model,
                prefix=prefix,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            logits = _forward_last_logits(
                model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=_position_ids_from_attention(attention_mask),
            )
        choice_token_ids = torch.tensor(
            [sample["choice_token_ids"] for sample in batch_samples],
            dtype=torch.long,
            device=device,
        )
        choice_logits = torch.gather(logits, dim=-1, index=choice_token_ids)
        choice_log_probs = F.log_softmax(choice_logits, dim=-1)
        choice_probs = F.softmax(choice_logits, dim=-1)
        pred = torch.argmax(choice_logits, dim=-1)
        for row_idx, sample in enumerate(batch_samples):
            gold = int(sample["gold"])
            pred_idx = int(pred[row_idx].detach().cpu())
            wrong_ll = max(
                float(choice_log_probs[row_idx, option_idx].detach().cpu())
                for option_idx in range(len(CHOICE_LABELS))
                if option_idx != gold
            )
            rows.append(
                {
                    "epoch": int(epoch),
                    "sample_idx": int(sample["sample_idx"]),
                    "source_idx": int(sample["source_idx"]),
                    "probe_id": sample.get("probe_id"),
                    "task_id": str(sample.get("task_id", "")),
                    "template_offset": int(sample["template_offset"]),
                    "question": sample["question"],
                    "source_answer": sample["source_answer"],
                    "prompt": sample["prompt"],
                    "gold": gold,
                    "gold_label": CHOICE_LABELS[gold],
                    "pred": pred_idx,
                    "pred_label": CHOICE_LABELS[pred_idx],
                    "correct": int(pred_idx == gold),
                    "gold_margin_choice_logprob": float(choice_log_probs[row_idx, gold].detach().cpu() - wrong_ll),
                    "gold_choice_prob": float(choice_probs[row_idx, gold].detach().cpu()),
                    "ll_choice_A": float(choice_log_probs[row_idx, 0].detach().cpu()),
                    "ll_choice_B": float(choice_log_probs[row_idx, 1].detach().cpu()),
                    "ll_choice_C": float(choice_log_probs[row_idx, 2].detach().cpu()),
                    "ll_choice_D": float(choice_log_probs[row_idx, 3].detach().cpu()),
                    "prob_choice_A": float(choice_probs[row_idx, 0].detach().cpu()),
                    "prob_choice_B": float(choice_probs[row_idx, 1].detach().cpu()),
                    "prob_choice_C": float(choice_probs[row_idx, 2].detach().cpu()),
                    "prob_choice_D": float(choice_probs[row_idx, 3].detach().cpu()),
                    "option_A": sample["options"][0],
                    "option_B": sample["options"][1],
                    "option_C": sample["options"][2],
                    "option_D": sample["options"][3],
                }
            )
        batch_no = start_idx // max(1, int(batch_size)) + 1
        if batch_no == 1 or batch_no % 5 == 0 or batch_no == total_batches:
            log(f"score_forget_mcq_epoch{int(epoch)} batch={batch_no}/{total_batches}")
    correct = int(sum(int(row["correct"]) for row in rows))
    margins = [float(row["gold_margin_choice_logprob"]) for row in rows]
    probs = [float(row["gold_choice_prob"]) for row in rows]
    summary = {
        "tag": f"score_forget_mcq_epoch{int(epoch)}",
        "mode": "mcq_logprob",
        "prefix_insertion": "after_question_before_choices",
        "sample_meta": sample_meta,
        "n": len(rows),
        "correct": correct,
        "accuracy": float(correct / len(rows)) if rows else 0.0,
        "mean_gold_margin_choice_logprob": float(sum(margins) / len(margins)) if margins else 0.0,
        "mean_gold_choice_prob": float(sum(probs) / len(probs)) if probs else 0.0,
        "negative_margin_count": int(sum(1 for value in margins if value < 0.0)),
    }
    write_jsonl(output_dir / f"mcq_forget_epoch{int(epoch)}.jsonl", rows)
    write_csv(output_dir / f"mcq_forget_epoch{int(epoch)}.csv", rows)
    write_json(output_dir / f"mcq_forget_summary_epoch{int(epoch)}.json", summary)
    log(f"score_forget_mcq_epoch{int(epoch)} " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def _selection_score_summary(
    *,
    epoch: int,
    forget_rows: list[dict[str, Any]],
    retain_rows: list[dict[str, Any]],
    config: JudgeConfig,
    retain_score_mode: str,
) -> dict[str, Any]:
    forget_dual_clean = _mean_bool(forget_rows, "dual_clean")
    forget_cot_clean = _mean_bool(forget_rows, "cot_clean")
    forget_answer_clean = _mean_bool(forget_rows, "answer_clean")
    retain_hit = _mean_bool(retain_rows, "fact_hit")
    threshold = float(config.threshold)
    penalty = float(config.penalty)
    score_mode = str(config.score_mode)
    if score_mode == "threshold_penalty":
        selection_score = retain_hit
        if forget_dual_clean < threshold:
            selection_score = retain_hit - penalty * (threshold - forget_dual_clean)
    elif score_mode == "weighted_mean":
        forget_weight = float(config.forget_weight)
        retain_weight = float(config.retain_weight)
        selection_score = (
            forget_weight * forget_dual_clean + retain_weight * retain_hit
        ) / (forget_weight + retain_weight)
    elif score_mode == "mean":
        selection_score = 0.5 * (forget_dual_clean + retain_hit)
    elif score_mode == "min":
        selection_score = min(forget_dual_clean, retain_hit)
    else:
        denom = forget_dual_clean + retain_hit
        selection_score = 0.0 if denom <= 0.0 else 2.0 * forget_dual_clean * retain_hit / denom
    summary = {
        "epoch": int(epoch),
        "threshold": threshold,
        "penalty": penalty,
        "score_mode": score_mode,
        "retain_score_mode": str(retain_score_mode),
        "forget_n": len(forget_rows),
        "retain_n": len(retain_rows),
        "forget_cot_clean_count": int(sum(1 for row in forget_rows if bool(row.get("cot_clean")))),
        "forget_answer_clean_count": int(sum(1 for row in forget_rows if bool(row.get("answer_clean")))),
        "forget_dual_clean_count": int(sum(1 for row in forget_rows if bool(row.get("dual_clean")))),
        "retain_hit_count": int(sum(1 for row in retain_rows if bool(row.get("fact_hit")))),
        "forget_cot_clean": forget_cot_clean,
        "forget_answer_clean": forget_answer_clean,
        "forget_dual_clean": forget_dual_clean,
        "retain_hit": retain_hit,
        "eligible": bool(forget_dual_clean >= threshold),
        "selection_score": float(selection_score),
    }
    if score_mode == "weighted_mean":
        summary["score_forget_weight"] = float(config.forget_weight)
        summary["score_retain_weight"] = float(config.retain_weight)
    return summary


def _update_selection_leaderboard(score_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    path = score_dir / "score_summary.json"
    rows: list[dict[str, Any]] = []
    active_retain_mode = str(summary.get("retain_score_mode", "generative_judge"))
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [
                row
                for row in payload
                if int(row.get("epoch", -1)) != int(summary["epoch"])
                and str(row.get("retain_score_mode", "generative_judge")) == active_retain_mode
            ]
    rows.append(summary)
    rows = sorted(rows, key=lambda row: int(row.get("epoch", -1)))
    eligible = [row for row in rows if bool(row.get("eligible"))]
    candidates = eligible if eligible else rows
    best = max(
        candidates,
        key=lambda row: (
            float(row.get("retain_hit", 0.0)),
            float(row.get("forget_dual_clean", 0.0)),
            float(row.get("selection_score", 0.0)),
            -int(row.get("epoch", 0)),
        ),
    )
    write_json(path, rows)
    write_json(score_dir / "best_epoch.json", best)
    return best


def judge_forget_generation_rows(
    *,
    epoch: int,
    split_name: str,
    generation_path: Path,
    judgment_path: Path,
    config: JudgeConfig,
    judge_mode: str = "source_leak",
) -> list[dict[str, Any]]:
    generation_rows = [row for row in load_json_or_jsonl(generation_path) if isinstance(row, dict)]
    existing = _load_existing_judgments(judgment_path)
    judgments: list[dict[str, Any]] = []
    for idx, row in enumerate(generation_rows, start=1):
        sample_idx = idx - 1
        if sample_idx in existing:
            judgments.append(existing[sample_idx])
        else:
            try:
                if str(judge_mode) == "star_safety":
                    judgment = judge_star_safety_forget_row(row, config)
                else:
                    judgment = judge_forget_row(row, config)
            except Exception as exc:
                judgment = {
                    "cot_leak": True,
                    "answer_leak": True,
                    "cot_clean": False,
                    "answer_clean": False,
                    "dual_clean": False,
                    "leak_evidence": "",
                    "reason": f"judge_error: {type(exc).__name__}: {exc}",
                    "raw_judge": {"error": str(exc), "error_type": type(exc).__name__},
                }
                log(f"score epoch={epoch} judge {split_name} {idx}/{len(generation_rows)} failed closed: {exc}")
            judgments.append(
                {
                    "epoch": int(epoch),
                    "split": split_name,
                    "sample_idx": sample_idx,
                    "idx": row.get("idx"),
                    "source_idx": row.get("source_idx"),
                    "probe_id": row.get("probe_id"),
                    "question": row.get("question"),
                    "source_answer": row.get("source_answer"),
                    **judgment,
                }
            )
            _write_judgments(judgment_path, judgments)
        if idx == 1 or idx % 10 == 0 or idx == len(generation_rows):
            log(f"score epoch={epoch} judge {split_name} {idx}/{len(generation_rows)}")
    _write_judgments(judgment_path, judgments)
    return judgments


def _forget_only_score_summary(
    *,
    epoch: int,
    in_rows: list[dict[str, Any]],
    out_rows: list[dict[str, Any]],
    config: JudgeConfig,
) -> dict[str, Any]:
    in_dual = _mean_bool(in_rows, "dual_clean")
    out_dual = _mean_bool(out_rows, "dual_clean")
    in_cot = _mean_bool(in_rows, "cot_clean")
    out_cot = _mean_bool(out_rows, "cot_clean")
    in_answer = _mean_bool(in_rows, "answer_clean")
    out_answer = _mean_bool(out_rows, "answer_clean")
    score_mode = str(config.score_mode)
    if score_mode == "min":
        selection_score = min(in_dual, out_dual)
    elif score_mode == "mean":
        selection_score = 0.5 * (in_dual + out_dual)
    elif score_mode == "weighted_mean":
        selection_score = (
            float(config.forget_weight) * in_dual + float(config.retain_weight) * out_dual
        ) / (float(config.forget_weight) + float(config.retain_weight))
    elif score_mode == "threshold_penalty":
        selection_score = out_dual
        if in_dual < float(config.threshold):
            selection_score = out_dual - float(config.penalty) * (float(config.threshold) - in_dual)
    else:
        denom = in_dual + out_dual
        selection_score = 0.0 if denom <= 0.0 else 2.0 * in_dual * out_dual / denom
    summary = {
        "epoch": int(epoch),
        "threshold": float(config.threshold),
        "penalty": float(config.penalty),
        "score_mode": score_mode,
        "forget_in_n": len(in_rows),
        "forget_out_n": len(out_rows),
        "forget_in_cot_clean_count": int(sum(1 for row in in_rows if bool(row.get("cot_clean")))),
        "forget_in_answer_clean_count": int(sum(1 for row in in_rows if bool(row.get("answer_clean")))),
        "forget_in_dual_clean_count": int(sum(1 for row in in_rows if bool(row.get("dual_clean")))),
        "forget_out_cot_clean_count": int(sum(1 for row in out_rows if bool(row.get("cot_clean")))),
        "forget_out_answer_clean_count": int(sum(1 for row in out_rows if bool(row.get("answer_clean")))),
        "forget_out_dual_clean_count": int(sum(1 for row in out_rows if bool(row.get("dual_clean")))),
        "forget_in_cot_clean": in_cot,
        "forget_in_answer_clean": in_answer,
        "forget_in_dual_clean": in_dual,
        "forget_out_cot_clean": out_cot,
        "forget_out_answer_clean": out_answer,
        "forget_out_dual_clean": out_dual,
        "eligible": bool(in_dual >= float(config.threshold) and out_dual >= float(config.threshold)),
        "selection_score": float(selection_score),
    }
    return summary


def _update_forget_only_leaderboard(score_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    path = score_dir / "score_summary.json"
    rows: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [row for row in payload if int(row.get("epoch", -1)) != int(summary["epoch"])]
    rows.append(summary)
    rows = sorted(rows, key=lambda row: int(row.get("epoch", -1)))
    eligible = [row for row in rows if bool(row.get("eligible"))]
    candidates = eligible if eligible else rows
    best = max(
        candidates,
        key=lambda row: (
            float(row.get("selection_score", 0.0)),
            float(row.get("forget_out_dual_clean", 0.0)),
            float(row.get("forget_in_dual_clean", 0.0)),
            -int(row.get("epoch", 0)),
        ),
    )
    write_json(path, rows)
    write_json(score_dir / "best_epoch.json", best)
    return best


def run_forget_only_score_eval(
    *,
    epoch: int,
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    in_records: list[dict[str, Any]],
    out_records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    args: argparse.Namespace,
    score_dir: Path,
    judge_config: JudgeConfig,
) -> dict[str, Any]:
    tag_suffix = f"epoch{int(epoch)}"
    run_generation_eval(
        model,
        prefix,
        tokenizer,
        in_records,
        model_cfg,
        tag=f"score_forget_in_{tag_suffix}",
        device=args.device,
        batch_size=int(args.generation_batch_size),
        max_new_tokens=int(args.max_new_tokens),
        prefix_position=args.prefix_position,
        output_dir=score_dir,
    )
    run_generation_eval(
        model,
        prefix,
        tokenizer,
        out_records,
        model_cfg,
        tag=f"score_forget_out_{tag_suffix}",
        device=args.device,
        batch_size=int(args.generation_batch_size),
        max_new_tokens=int(args.max_new_tokens),
        prefix_position=args.prefix_position,
        output_dir=score_dir,
    )
    in_judgments = judge_forget_generation_rows(
        epoch=int(epoch),
        split_name="forget_in",
        generation_path=score_dir / f"generations_score_forget_in_{tag_suffix}.jsonl",
        judgment_path=score_dir / f"judge_forget_in_epoch{int(epoch)}.jsonl",
        config=judge_config,
        judge_mode=str(args.score_forget_judge_mode),
    )
    out_judgments = judge_forget_generation_rows(
        epoch=int(epoch),
        split_name="forget_out",
        generation_path=score_dir / f"generations_score_forget_out_{tag_suffix}.jsonl",
        judgment_path=score_dir / f"judge_forget_out_epoch{int(epoch)}.jsonl",
        config=judge_config,
        judge_mode=str(args.score_forget_judge_mode),
    )
    summary = _forget_only_score_summary(
        epoch=int(epoch),
        in_rows=in_judgments,
        out_rows=out_judgments,
        config=judge_config,
    )
    best = _update_forget_only_leaderboard(score_dir, summary)
    write_json(score_dir / f"score_epoch{int(epoch)}.json", {**summary, "best_epoch_after_update": best})
    log("score epoch=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    log("score best=" + json.dumps(best, ensure_ascii=False, sort_keys=True))
    return summary


def run_score_eval(
    *,
    epoch: int,
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    forget_records: list[dict[str, Any]],
    retain_records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    args: argparse.Namespace,
    score_dir: Path,
    judge_config: JudgeConfig,
) -> dict[str, Any]:
    tag_suffix = f"epoch{int(epoch)}"
    run_generation_eval(
        model,
        prefix,
        tokenizer,
        forget_records,
        model_cfg,
        tag=f"score_forget_{tag_suffix}",
        device=args.device,
        batch_size=int(args.generation_batch_size),
        max_new_tokens=int(args.max_new_tokens),
        prefix_position=args.prefix_position,
        output_dir=score_dir,
    )
    if str(args.score_retain_mode) == "mcq_logprob":
        forget_judgments = judge_forget_generation_rows(
            epoch=int(epoch),
            split_name="forget",
            generation_path=score_dir / f"generations_score_forget_{tag_suffix}.jsonl",
            judgment_path=score_dir / f"judge_forget_epoch{int(epoch)}.jsonl",
            config=judge_config,
            judge_mode=str(args.score_forget_judge_mode),
        )
        retain_judgments = score_retain_mcq_logprob(
            epoch=int(epoch),
            model=model,
            prefix=prefix,
            tokenizer=tokenizer,
            retain_records=retain_records,
            device=str(args.device),
            batch_size=int(args.generation_batch_size),
            output_dir=score_dir,
        )
        summary = _selection_score_summary(
            epoch=int(epoch),
            forget_rows=forget_judgments,
            retain_rows=retain_judgments,
            config=judge_config,
            retain_score_mode=str(args.score_retain_mode),
        )
        best = _update_selection_leaderboard(score_dir, summary)
        write_json(score_dir / f"score_epoch{int(epoch)}.json", {**summary, "best_epoch_after_update": best})
        log("score epoch=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
        log("score best=" + json.dumps(best, ensure_ascii=False, sort_keys=True))
        return summary
    run_generation_eval(
        model,
        prefix,
        tokenizer,
        retain_records,
        model_cfg,
        tag=f"score_retain_{tag_suffix}",
        device=args.device,
        batch_size=int(args.generation_batch_size),
        max_new_tokens=int(args.retain_max_new_tokens),
        prefix_position=args.prefix_position,
        output_dir=score_dir,
    )
    return judge_and_score_epoch(
        epoch=int(epoch),
        score_dir=score_dir,
        forget_generation_path=score_dir / f"generations_score_forget_{tag_suffix}.jsonl",
        retain_generation_path=score_dir / f"generations_score_retain_{tag_suffix}.jsonl",
        config=judge_config,
        forget_judge_mode=str(args.score_forget_judge_mode),
        log_fn=log,
    )


class BlockTargetEncoder:
    def __init__(self, tokenizer, model_cfg: dict[str, Any], *, max_length: int, prefix_position: str = "prompt_end"):
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.max_length = int(max_length)
        self.prefix_position = normalize_prefix_position(prefix_position)
        if self.prefix_position != "prompt_end":
            raise ValueError("BlockTargetEncoder supports prompt_end prefix insertion only")
        self.think_end_tag = str(model_cfg.get("think_end_tag", "\n</think>\n\n"))
        self.answer_tag = str(model_cfg.get("answer_tag", "") or "")
        self.separator_ids = [
            int(token_id)
            for token_id in tokenizer(
                self.think_end_tag + self.answer_tag,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        ]
        self.think_end_token_variants = self._build_think_end_token_variants()

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
        texts = [self.think_end_tag, self.think_end_tag.strip(), "\n</think>\n\n", "</think>"]
        seen: set[tuple[int, ...]] = set()
        for text in texts:
            token_ids = [
                int(token_id)
                for token_id in self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            ]
            key = tuple(token_ids)
            if token_ids and key not in seen:
                variants.append(token_ids)
                seen.add(key)
        return variants

    def split_base_completion_ids(self, completion_ids: list[int]) -> tuple[list[int], list[int], bool]:
        ids = [int(token_id) for token_id in completion_ids]
        if self.tokenizer.eos_token_id is not None and ids and ids[-1] == int(self.tokenizer.eos_token_id):
            ids = ids[:-1]
        for tag_ids in self.think_end_token_variants:
            match_start = self._find_subsequence(ids, tag_ids)
            if match_start >= 0:
                return ids[:match_start], ids[match_start + len(tag_ids) :], True
        return ids, [], False

    def text_to_ids(self, text: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.tokenizer(
                cleanup_generated_text(text),
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        ]

    def encode(
        self,
        record: dict[str, Any],
        *,
        cot_ids: list[int],
        answer_text: str | None = None,
        answer_ids: list[int] | None = None,
        target_type: str,
        target_source: str,
        retain_has_think_end: bool | None = None,
    ) -> dict[str, Any]:
        prompt = str(record["prompt"])
        prompt_ids = [
            int(token_id)
            for token_id in self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        ]
        cot_ids = [int(token_id) for token_id in cot_ids]
        if answer_ids is None:
            if answer_text is None:
                raise ValueError("answer_text or answer_ids is required")
            answer_ids = self.text_to_ids(answer_text)
        else:
            answer_ids = [int(token_id) for token_id in answer_ids]
        if self.tokenizer.eos_token_id is not None:
            answer_ids.append(int(self.tokenizer.eos_token_id))
        if not cot_ids:
            raise ValueError("empty cot target")
        if not answer_ids:
            raise ValueError("empty answer target")
        assistant_start = len(prompt_ids)
        cot_start = assistant_start
        separator_start = cot_start + len(cot_ids)
        answer_start = separator_start + len(self.separator_ids)
        input_ids = prompt_ids
        input_ids = input_ids + cot_ids + self.separator_ids + answer_ids
        if len(input_ids) > self.max_length:
            raise ValueError(f"block target too long: {len(input_ids)} > {self.max_length}")
        return {
            "target_type": target_type,
            "target_source": target_source,
            "record": record,
            "target": {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "prompt_len": len(prompt_ids),
                "cot_positions": list(range(cot_start, cot_start + len(cot_ids))),
                "answer_positions": list(range(separator_start, answer_start + len(answer_ids))),
                "cot_len": len(cot_ids),
                "answer_len": len(self.separator_ids) + len(answer_ids),
                "raw_answer_len": len(answer_ids),
                "separator_len": len(self.separator_ids),
                "retain_has_think_end": retain_has_think_end,
            },
        }

    def encode_forget(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.encode(
            record,
            cot_ids=self.text_to_ids(str(record["target_cot"])),
            answer_text=str(record["target_answer"]),
            target_type="forget",
            target_source="idkcot_idk_answer",
        )

    def encode_forget_raw(self, record: dict[str, Any]) -> dict[str, Any]:
        raw_cot = str(record.get("raw_cot") or record.get("cot") or "").strip()
        raw_answer = str(record.get("raw_answer") or record.get("source_answer") or record.get("answer") or "").strip()
        if not raw_cot:
            raise ValueError("empty raw cot")
        if not raw_answer:
            raise ValueError("empty raw answer")
        return self.encode(
            record,
            cot_ids=self.text_to_ids(raw_cot),
            answer_text=raw_answer,
            target_type="forget_rejected",
            target_source="raw_cot_source_answer",
        )

    def encode_forget_raw_answer(self, record: dict[str, Any]) -> dict[str, Any]:
        prompt = build_direct_prompt(str(record.get("question") or ""), self.model_cfg)
        raw_answer = str(record.get("source_answer") or record.get("answer") or "").strip()
        if not raw_answer:
            raise ValueError("empty raw answer")
        prompt_ids = [
            int(token_id)
            for token_id in self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        ]
        full_ids = [
            int(token_id)
            for token_id in self.tokenizer(prompt + raw_answer, add_special_tokens=True, truncation=False)["input_ids"]
        ]
        if len(full_ids) > len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
            answer_start = len(prompt_ids)
        else:
            answer_ids = self.text_to_ids(raw_answer)
            if not answer_ids:
                raise ValueError("empty raw answer ids")
            full_ids = prompt_ids + answer_ids
            answer_start = len(prompt_ids)
        answer_positions = list(range(answer_start, len(full_ids)))
        if not answer_positions:
            raise ValueError("empty raw answer positions")
        if len(full_ids) > self.max_length:
            raise ValueError(f"raw answer target too long: {len(full_ids)} > {self.max_length}")
        return {
            "target_type": "forget_raw_answer",
            "target_source": "source_answer_direct_prompt",
            "record": record,
            "target": {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "prompt_len": int(answer_start),
                "cot_positions": [],
                "answer_positions": answer_positions,
                "cot_len": 0,
                "answer_len": len(answer_positions),
                "raw_answer_len": len(answer_positions),
                "separator_len": 0,
                "retain_has_think_end": None,
            },
        }

    def encode_retain(self, record: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
        raw_ids = trajectory.get("completion_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raw_ids = self.text_to_ids(str(trajectory.get("base_completion") or ""))
        cot_ids, answer_ids, has_think_end = self.split_base_completion_ids([int(token_id) for token_id in raw_ids])
        return self.encode(
            record,
            cot_ids=cot_ids,
            answer_ids=answer_ids,
            target_type="retain",
            target_source="base_cot_base_answer",
            retain_has_think_end=has_think_end,
        )


class BlockTargetDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        if not rows:
            raise ValueError("empty block target dataset")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def block_target_collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    seqs = [dict(row["target"]) for row in batch]
    max_len = max(len(seq["input_ids"]) for seq in seqs)
    max_cot = max(len(seq["cot_positions"]) for seq in seqs)
    max_answer = max(len(seq["answer_positions"]) for seq in seqs)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    prompt_lens: list[int] = []
    cot_positions: list[list[int]] = []
    cot_masks: list[list[float]] = []
    answer_positions: list[list[int]] = []
    answer_masks: list[list[float]] = []
    target_meta: list[dict[str, Any]] = []
    for row, seq in zip(batch, seqs):
        pad = max_len - len(seq["input_ids"])
        cot_pad = max_cot - len(seq["cot_positions"])
        answer_pad = max_answer - len(seq["answer_positions"])
        input_ids.append(seq["input_ids"] + [pad_id] * pad)
        attention_mask.append(seq["attention_mask"] + [0] * pad)
        prompt_lens.append(int(seq["prompt_len"]))
        cot_positions.append([int(pos) for pos in seq["cot_positions"]] + [0] * cot_pad)
        cot_masks.append([1.0] * len(seq["cot_positions"]) + [0.0] * cot_pad)
        answer_positions.append([int(pos) for pos in seq["answer_positions"]] + [0] * answer_pad)
        answer_masks.append([1.0] * len(seq["answer_positions"]) + [0.0] * answer_pad)
        target_meta.append(
            {
                "target_type": row["target_type"],
                "target_source": row["target_source"],
                "probe_id": row["record"].get("probe_id"),
                "source_key": row["record"].get("source_key"),
                "cot_len": int(seq["cot_len"]),
                "answer_len": int(seq["answer_len"]),
                "retain_has_think_end": seq.get("retain_has_think_end"),
            }
        )
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        "cot_positions": torch.tensor(cot_positions, dtype=torch.long),
        "cot_position_mask": torch.tensor(cot_masks, dtype=torch.float),
        "answer_positions": torch.tensor(answer_positions, dtype=torch.long),
        "answer_position_mask": torch.tensor(answer_masks, dtype=torch.float),
        "target_meta": target_meta,
        "target_count": len(batch),
    }


class DPOBlockPairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        if not rows:
            raise ValueError("empty DPO block pair dataset")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def dpo_block_pair_collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    chosen_rows = [
        {
            "target_type": "dpo_chosen",
            "target_source": row["chosen_source"],
            "record": row["record"],
            "target": row["chosen"],
        }
        for row in batch
    ]
    rejected_rows = [
        {
            "target_type": "dpo_rejected",
            "target_source": row["rejected_source"],
            "record": row["record"],
            "target": row["rejected"],
        }
        for row in batch
    ]
    return {
        "chosen": block_target_collate(chosen_rows, pad_id),
        "rejected": block_target_collate(rejected_rows, pad_id),
        "pair_keys": [str(row["pair_key"]) for row in batch],
        "records": [row["record"] for row in batch],
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
            torch.cat([tok_embeds[row_idx, :insert_at], pref[row_idx], tok_embeds[row_idx, insert_at:]], dim=0)
        )
        attn_rows.append(
            torch.cat([attention_mask[row_idx, :insert_at], pref_mask[row_idx], attention_mask[row_idx, insert_at:]], dim=0)
        )
    return torch.stack(embed_rows, dim=0), torch.stack(attn_rows, dim=0)


def _position_chunks(positions: torch.Tensor, mask: torch.Tensor, chunk_tokens: int) -> list[torch.Tensor]:
    valid = positions[mask.to(dtype=torch.bool)]
    if valid.numel() == 0:
        return []
    chunk_tokens = max(1, int(chunk_tokens))
    return [valid[start : start + chunk_tokens].to(dtype=torch.long) for start in range(0, int(valid.numel()), chunk_tokens)]


def _validate_block_weights(cot_weight: float, answer_weight: float) -> float:
    if cot_weight < 0.0 or answer_weight < 0.0:
        raise ValueError("block weights must be non-negative")
    block_weight_sum = float(cot_weight) + float(answer_weight)
    if block_weight_sum <= 0.0:
        raise ValueError("at least one block weight must be positive")
    return block_weight_sum


def _weighted_pair(cot_loss: torch.Tensor, answer_loss: torch.Tensor, cot_weight: float, answer_weight: float) -> torch.Tensor:
    _validate_block_weights(float(cot_weight), float(answer_weight))
    return float(cot_weight) * cot_loss + float(answer_weight) * answer_loss


def build_dpo_block_pairs(
    *,
    encoder: BlockTargetEncoder,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def dpo_target(target: dict[str, Any]) -> dict[str, Any]:
        copied = {key: (list(value) if isinstance(value, list) else value) for key, value in target.items()}
        separator_len = int(copied.get("separator_len", 0) or 0)
        if separator_len > 0:
            copied["answer_positions"] = list(copied["answer_positions"])[separator_len:]
            copied["answer_len"] = len(copied["answer_positions"])
        if not copied["answer_positions"]:
            raise ValueError("empty DPO answer positions")
        return copied

    rows: list[dict[str, Any]] = []
    skipped = 0
    chosen_cot_lens: list[int] = []
    chosen_answer_lens: list[int] = []
    rejected_cot_lens: list[int] = []
    rejected_answer_lens: list[int] = []
    for record in records:
        try:
            chosen = encoder.encode_forget(record)
            rejected = encoder.encode_forget_raw(record)
            chosen_target = dpo_target(chosen["target"])
            rejected_target = dpo_target(rejected["target"])
        except ValueError:
            skipped += 1
            continue
        rows.append(
            {
                "pair_key": dpo_record_key(record),
                "record": record,
                "chosen": chosen_target,
                "rejected": rejected_target,
                "chosen_source": str(chosen["target_source"]),
                "rejected_source": str(rejected["target_source"]),
            }
        )
        chosen_cot_lens.append(int(rows[-1]["chosen"]["cot_len"]))
        chosen_answer_lens.append(int(rows[-1]["chosen"]["answer_len"]))
        rejected_cot_lens.append(int(rows[-1]["rejected"]["cot_len"]))
        rejected_answer_lens.append(int(rows[-1]["rejected"]["answer_len"]))

    def mean(values: list[int]) -> float:
        return float(sum(values) / max(1, len(values)))

    return rows, {
        "sample_n": len(records),
        "pairs": len(rows),
        "skipped": skipped,
        "chosen_cot_tokens_mean": mean(chosen_cot_lens),
        "chosen_answer_tokens_mean": mean(chosen_answer_lens),
        "rejected_cot_tokens_mean": mean(rejected_cot_lens),
        "rejected_answer_tokens_mean": mean(rejected_answer_lens),
        "chosen_cot_tokens_max": max(chosen_cot_lens) if chosen_cot_lens else 0,
        "chosen_answer_tokens_max": max(chosen_answer_lens) if chosen_answer_lens else 0,
        "rejected_cot_tokens_max": max(rejected_cot_lens) if rejected_cot_lens else 0,
        "rejected_answer_tokens_max": max(rejected_answer_lens) if rejected_answer_lens else 0,
    }


def _block_logp_scores_from_logits(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    pred_offset: int,
    chunk_tokens: int,
    cot_weight: float,
    answer_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    _validate_block_weights(float(cot_weight), float(answer_weight))
    input_ids = batch["input_ids"]
    scores: list[torch.Tensor] = []
    cot_scores: list[torch.Tensor] = []
    answer_scores: list[torch.Tensor] = []
    cot_counts: list[torch.Tensor] = []
    answer_counts: list[torch.Tensor] = []

    def block_sum(row_idx: int, positions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logp_sum = logits.new_zeros(())
        token_count = logits.new_zeros(())
        for pos_chunk in _position_chunks(positions, mask, chunk_tokens):
            pos_chunk = pos_chunk.to(device=input_ids.device)
            pred_pos = pos_chunk - 1 + int(pred_offset)
            target_ids = input_ids[row_idx, pos_chunk]
            row_logits = logits[row_idx, pred_pos, :].to(torch.float32)
            row_logp = F.log_softmax(row_logits, dim=-1)
            token_logp = torch.gather(row_logp, dim=-1, index=target_ids.view(-1, 1)).squeeze(-1)
            logp_sum = logp_sum + token_logp.to(logits.dtype).sum()
            token_count = token_count + logits.new_tensor(float(pos_chunk.numel()))
        return logp_sum, token_count

    for row_idx in range(input_ids.shape[0]):
        cot_sum, cot_count = block_sum(row_idx, batch["cot_positions"][row_idx], batch["cot_position_mask"][row_idx])
        answer_sum, answer_count = block_sum(
            row_idx,
            batch["answer_positions"][row_idx],
            batch["answer_position_mask"][row_idx],
        )
        cot_score = cot_sum / cot_count.clamp_min(1.0)
        answer_score = answer_sum / answer_count.clamp_min(1.0)
        score = float(cot_weight) * cot_score + float(answer_weight) * answer_score
        scores.append(score)
        cot_scores.append(cot_score)
        answer_scores.append(answer_score)
        cot_counts.append(cot_count.detach())
        answer_counts.append(answer_count.detach())

    score_tensor = torch.stack(scores)
    with torch.no_grad():
        metrics = {
            "score": float(score_tensor.detach().mean().cpu()),
            "cot_logp": float(torch.stack(cot_scores).detach().mean().cpu()),
            "answer_logp": float(torch.stack(answer_scores).detach().mean().cpu()),
            "cot_tokens": float(torch.stack(cot_counts).to(torch.float32).mean().detach().cpu()),
            "answer_tokens": float(torch.stack(answer_counts).to(torch.float32).mean().detach().cpu()),
        }
    return score_tensor, metrics


def block_logp_with_prefix(
    model,
    prefix: SoftPrefix,
    batch: dict[str, Any],
    *,
    chunk_tokens: int,
    cot_weight: float,
    answer_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    inputs_embeds, pref_attn = insert_prompt_end_prefix_for_batch(
        model,
        prefix,
        batch["input_ids"],
        batch["attention_mask"],
        batch["prompt_lens"],
    )
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=pref_attn, use_cache=False)
    return _block_logp_scores_from_logits(
        outputs.logits,
        batch,
        pred_offset=int(prefix.embedding.shape[0]),
        chunk_tokens=int(chunk_tokens),
        cot_weight=float(cot_weight),
        answer_weight=float(answer_weight),
    )


@torch.no_grad()
def block_logp_without_prefix(
    model,
    batch: dict[str, Any],
    *,
    chunk_tokens: int,
    cot_weight: float,
    answer_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    return _block_logp_scores_from_logits(
        outputs.logits,
        batch,
        pred_offset=0,
        chunk_tokens=int(chunk_tokens),
        cot_weight=float(cot_weight),
        answer_weight=float(answer_weight),
    )


def compute_reference_dpo_scores(
    model,
    rows: list[dict[str, Any]],
    *,
    tokenizer,
    cache_path: Path,
    device: str,
    batch_size: int,
    chunk_tokens: int,
    cot_weight: float,
    answer_weight: float,
    force: bool,
) -> dict[str, dict[str, float]]:
    expected = {str(row["pair_key"]) for row in rows}
    if cache_path.exists() and not force:
        cached_rows = [row for row in load_json_or_jsonl(cache_path) if isinstance(row, dict)]
        cached = {
            str(row.get("pair_key")): row
            for row in cached_rows
            if str(row.get("pair_key")) in expected
            and abs(float(row.get("cot_weight", cot_weight)) - float(cot_weight)) < 1e-9
            and abs(float(row.get("answer_weight", answer_weight)) - float(answer_weight)) < 1e-9
        }
        if expected.issubset(cached):
            log(f"loaded DPO reference logP cache rows={len(cached)} cache={cache_path}")
            return {
                key: {
                    "ref_chosen_score": float(row["ref_chosen_score"]),
                    "ref_rejected_score": float(row["ref_rejected_score"]),
                    "ref_margin": float(row["ref_margin"]),
                }
                for key, row in cached.items()
            }
        log(f"DPO reference cache incomplete; rebuilding cache={cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []
    loader = DataLoader(
        DPOBlockPairDataset(rows),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=lambda b: dpo_block_pair_collate(b, int(tokenizer.pad_token_id)),
    )
    total_batches = math.ceil(len(rows) / max(1, int(batch_size)))
    for batch_no, batch in enumerate(loader, start=1):
        device_batch = {
            side: {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in payload.items()
            }
            if isinstance(payload, dict)
            else payload
            for side, payload in batch.items()
        }
        chosen_scores, chosen_metrics = block_logp_without_prefix(
            model,
            device_batch["chosen"],
            chunk_tokens=int(chunk_tokens),
            cot_weight=float(cot_weight),
            answer_weight=float(answer_weight),
        )
        rejected_scores, rejected_metrics = block_logp_without_prefix(
            model,
            device_batch["rejected"],
            chunk_tokens=int(chunk_tokens),
            cot_weight=float(cot_weight),
            answer_weight=float(answer_weight),
        )
        margins = chosen_scores - rejected_scores
        for idx, pair_key in enumerate(batch["pair_keys"]):
            out_rows.append(
                {
                    "pair_key": str(pair_key),
                    "ref_chosen_score": float(chosen_scores[idx].detach().cpu()),
                    "ref_rejected_score": float(rejected_scores[idx].detach().cpu()),
                    "ref_margin": float(margins[idx].detach().cpu()),
                    "chosen_cot_logp_mean": float(chosen_metrics["cot_logp"]),
                    "chosen_answer_logp_mean": float(chosen_metrics["answer_logp"]),
                    "rejected_cot_logp_mean": float(rejected_metrics["cot_logp"]),
                    "rejected_answer_logp_mean": float(rejected_metrics["answer_logp"]),
                    "cot_weight": float(cot_weight),
                    "answer_weight": float(answer_weight),
                }
            )
        if batch_no == 1 or batch_no % 10 == 0 or batch_no == total_batches:
            log(f"DPO reference logP cache batch={batch_no}/{total_batches}")
    write_jsonl(cache_path, out_rows)
    log(f"wrote DPO reference logP cache rows={len(out_rows)} cache={cache_path}")
    return {
        str(row["pair_key"]): {
            "ref_chosen_score": float(row["ref_chosen_score"]),
            "ref_rejected_score": float(row["ref_rejected_score"]),
            "ref_margin": float(row["ref_margin"]),
        }
        for row in out_rows
    }


def prefix_dpo_block_loss(
    model,
    prefix: SoftPrefix,
    batch: dict[str, Any],
    ref_scores: dict[str, dict[str, float]],
    *,
    beta: float,
    chunk_tokens: int,
    cot_weight: float,
    answer_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_scores, chosen_metrics = block_logp_with_prefix(
        model,
        prefix,
        batch["chosen"],
        chunk_tokens=int(chunk_tokens),
        cot_weight=float(cot_weight),
        answer_weight=float(answer_weight),
    )
    rejected_scores, rejected_metrics = block_logp_with_prefix(
        model,
        prefix,
        batch["rejected"],
        chunk_tokens=int(chunk_tokens),
        cot_weight=float(cot_weight),
        answer_weight=float(answer_weight),
    )
    ref_margin_values = [
        float(ref_scores[str(pair_key)]["ref_margin"])
        for pair_key in batch["pair_keys"]
    ]
    ref_margin = chosen_scores.new_tensor(ref_margin_values)
    policy_margin = chosen_scores - rejected_scores
    relative_margin = policy_margin - ref_margin
    scaled = float(beta) * relative_margin
    loss = -F.logsigmoid(scaled).mean()
    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach().cpu()),
            "dpo_loss": float(loss.detach().cpu()),
            "dpo_margin": float(relative_margin.detach().mean().cpu()),
            "dpo_policy_margin": float(policy_margin.detach().mean().cpu()),
            "dpo_ref_margin": float(ref_margin.detach().mean().cpu()),
            "dpo_scaled_margin": float(scaled.detach().mean().cpu()),
            "preference_acc": float((relative_margin.detach() > 0).to(torch.float32).mean().cpu()),
            "policy_preference_acc": float((policy_margin.detach() > 0).to(torch.float32).mean().cpu()),
            "chosen_score": float(chosen_scores.detach().mean().cpu()),
            "rejected_score": float(rejected_scores.detach().mean().cpu()),
            "chosen_cot_logp": float(chosen_metrics["cot_logp"]),
            "chosen_answer_logp": float(chosen_metrics["answer_logp"]),
            "rejected_cot_logp": float(rejected_metrics["cot_logp"]),
            "rejected_answer_logp": float(rejected_metrics["answer_logp"]),
            "chosen_cot_tokens": float(chosen_metrics["cot_tokens"]),
            "chosen_answer_tokens": float(chosen_metrics["answer_tokens"]),
            "rejected_cot_tokens": float(rejected_metrics["cot_tokens"]),
            "rejected_answer_tokens": float(rejected_metrics["answer_tokens"]),
            "beta": float(beta),
            "lambda_cot_dpo": float(cot_weight),
            "lambda_answer_dpo": float(answer_weight),
        }
    return loss, metrics


def normalized_block_ce_loss(
    model,
    prefix: SoftPrefix,
    batch: dict[str, Any],
    *,
    chunk_tokens: int,
    cot_weight: float = 1.0,
    answer_weight: float = 1.0,
    normalize_block_weights: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Block CE with batch-level normalization over reasoning and answer blocks."""
    block_weight_sum = _validate_block_weights(float(cot_weight), float(answer_weight))
    student_embeds, student_attn = insert_prompt_end_prefix_for_batch(
        model,
        prefix,
        batch["input_ids"],
        batch["attention_mask"],
        batch["prompt_lens"],
    )
    student_out = model(inputs_embeds=student_embeds, attention_mask=student_attn, use_cache=False)
    student_logits = student_out.logits
    prefix_len = int(prefix.embedding.shape[0])

    cot_sum = student_logits.new_zeros(())
    cot_count = student_logits.new_zeros(())
    answer_sum = student_logits.new_zeros(())
    answer_count = student_logits.new_zeros(())
    cot_counts: list[torch.Tensor] = []
    answer_counts: list[torch.Tensor] = []

    for row_idx in range(batch["input_ids"].shape[0]):
        row_cot_count = student_logits.new_zeros(())
        for pos_chunk in _position_chunks(
            batch["cot_positions"][row_idx],
            batch["cot_position_mask"][row_idx],
            chunk_tokens,
        ):
            pos_chunk = pos_chunk.to(device=batch["input_ids"].device)
            student_pos = pos_chunk - 1 + prefix_len
            target_ids = batch["input_ids"][row_idx, pos_chunk]
            s_logits = student_logits[row_idx, student_pos, :]
            s_logp = F.log_softmax(s_logits.to(torch.float32), dim=-1)
            cot_nll = -torch.gather(s_logp, dim=-1, index=target_ids.view(-1, 1)).squeeze(-1)
            cot_sum = cot_sum + cot_nll.to(student_logits.dtype).sum()
            row_cot_count = row_cot_count + student_logits.new_tensor(float(pos_chunk.numel()))
        cot_count = cot_count + row_cot_count

        row_answer_count = student_logits.new_zeros(())
        for pos_chunk in _position_chunks(
            batch["answer_positions"][row_idx],
            batch["answer_position_mask"][row_idx],
            chunk_tokens,
        ):
            pos_chunk = pos_chunk.to(device=batch["input_ids"].device)
            student_pos = pos_chunk - 1 + prefix_len
            target_ids = batch["input_ids"][row_idx, pos_chunk]
            s_logits = student_logits[row_idx, student_pos, :]
            s_logp = F.log_softmax(s_logits.to(torch.float32), dim=-1)
            answer_nll = -torch.gather(s_logp, dim=-1, index=target_ids.view(-1, 1)).squeeze(-1)
            answer_sum = answer_sum + answer_nll.to(student_logits.dtype).sum()
            row_answer_count = row_answer_count + student_logits.new_tensor(float(pos_chunk.numel()))
        answer_count = answer_count + row_answer_count

        cot_counts.append(row_cot_count.detach())
        answer_counts.append(row_answer_count.detach())

    cot_loss = cot_sum / cot_count.clamp_min(1.0)
    answer_loss = answer_sum / answer_count.clamp_min(1.0)
    if normalize_block_weights:
        cot_weight_tensor = student_logits.new_tensor(float(cot_weight))
        answer_weight_tensor = student_logits.new_tensor(float(answer_weight))
        cot_present = (cot_count > 0).to(dtype=student_logits.dtype) * cot_weight_tensor
        answer_present = (answer_count > 0).to(dtype=student_logits.dtype) * answer_weight_tensor
        denom = (cot_present + answer_present).clamp_min(1e-8)
        loss = (
            cot_weight_tensor * cot_loss * (cot_count > 0).to(dtype=student_logits.dtype)
            + answer_weight_tensor * answer_loss * (answer_count > 0).to(dtype=student_logits.dtype)
        ) / denom
    else:
        loss = _weighted_pair(cot_loss, answer_loss, float(cot_weight), float(answer_weight))
    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach().cpu()),
            "cot_ce": float(cot_loss.detach().cpu()),
            "answer_ce": float(answer_loss.detach().cpu()),
            "cot_weight": float(cot_weight),
            "answer_weight": float(answer_weight),
            "block_weight_sum": float(block_weight_sum),
            "normalize_block_weights": float(1.0 if normalize_block_weights else 0.0),
            "cot_tokens": float(torch.stack(cot_counts).to(torch.float32).mean().detach().cpu()),
            "answer_tokens": float(torch.stack(answer_counts).to(torch.float32).mean().detach().cpu()),
        }
    return loss, metrics


def normalized_raw_answer_unlikelihood_loss(
    model,
    prefix: SoftPrefix,
    batch: dict[str, Any],
    *,
    chunk_tokens: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level unlikelihood on the original raw answer after a direct prompt."""
    student_embeds, student_attn = insert_prompt_end_prefix_for_batch(
        model,
        prefix,
        batch["input_ids"],
        batch["attention_mask"],
        batch["prompt_lens"],
    )
    student_out = model(inputs_embeds=student_embeds, attention_mask=student_attn, use_cache=False)
    student_logits = student_out.logits
    prefix_len = int(prefix.embedding.shape[0])

    ul_sum = student_logits.new_zeros(())
    nll_sum = student_logits.new_zeros(())
    prob_sum = student_logits.new_zeros(())
    token_count = student_logits.new_zeros(())
    answer_counts: list[torch.Tensor] = []

    for row_idx in range(batch["input_ids"].shape[0]):
        row_answer_count = student_logits.new_zeros(())
        for pos_chunk in _position_chunks(
            batch["answer_positions"][row_idx],
            batch["answer_position_mask"][row_idx],
            chunk_tokens,
        ):
            pos_chunk = pos_chunk.to(device=batch["input_ids"].device)
            student_pos = pos_chunk - 1 + prefix_len
            target_ids = batch["input_ids"][row_idx, pos_chunk]
            s_logits = student_logits[row_idx, student_pos, :]
            s_logp = F.log_softmax(s_logits.to(torch.float32), dim=-1)
            target_logp = torch.gather(s_logp, dim=-1, index=target_ids.view(-1, 1)).squeeze(-1)
            target_prob = target_logp.exp().clamp(min=float(eps), max=1.0 - float(eps))
            token_ul = -torch.log1p(-target_prob)
            ul_sum = ul_sum + token_ul.to(student_logits.dtype).sum()
            nll_sum = nll_sum + (-target_logp).to(student_logits.dtype).sum()
            prob_sum = prob_sum + target_prob.to(student_logits.dtype).sum()
            count = student_logits.new_tensor(float(pos_chunk.numel()))
            row_answer_count = row_answer_count + count
            token_count = token_count + count
        answer_counts.append(row_answer_count.detach())

    loss = ul_sum / token_count.clamp_min(1.0)
    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach().cpu()),
            "raw_answer_ul": float(loss.detach().cpu()),
            "raw_answer_nll": float((nll_sum / token_count.clamp_min(1.0)).detach().cpu()),
            "raw_answer_prob": float((prob_sum / token_count.clamp_min(1.0)).detach().cpu()),
            "raw_answer_tokens": float(torch.stack(answer_counts).to(torch.float32).mean().detach().cpu()),
        }
    return loss, metrics


def normalized_block_topk_kl_loss(
    model,
    prefix: SoftPrefix,
    batch: dict[str, Any],
    *,
    chunk_tokens: int,
    kl_topk: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Top-k teacher KL with top-k student normalization.

    Each block is averaged over its own tokens, then the two block losses are
    averaged per sample. Teacher and student distributions are both normalized
    inside the teacher top-k subset.
    """
    with torch.no_grad():
        teacher_out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        teacher_logits = teacher_out.logits

    student_embeds, student_attn = insert_prompt_end_prefix_for_batch(
        model,
        prefix,
        batch["input_ids"],
        batch["attention_mask"],
        batch["prompt_lens"],
    )
    student_out = model(inputs_embeds=student_embeds, attention_mask=student_attn, use_cache=False)
    student_logits = student_out.logits
    prefix_len = int(prefix.embedding.shape[0])
    vocab_size = int(student_logits.shape[-1])
    topk = min(max(1, int(kl_topk)), vocab_size)

    row_losses: list[torch.Tensor] = []
    cot_losses: list[torch.Tensor] = []
    answer_losses: list[torch.Tensor] = []
    cot_counts: list[torch.Tensor] = []
    answer_counts: list[torch.Tensor] = []

    def block_kl_sum(row_idx: int, positions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_sum = student_logits.new_zeros(())
        block_count = student_logits.new_zeros(())
        for pos_chunk in _position_chunks(positions, mask, chunk_tokens):
            pos_chunk = pos_chunk.to(device=batch["input_ids"].device)
            teacher_pos = pos_chunk - 1
            student_pos = pos_chunk - 1 + prefix_len
            t_logits = teacher_logits[row_idx, teacher_pos, :].to(torch.float32)
            s_logits = student_logits[row_idx, student_pos, :].to(torch.float32)
            teacher_topk_logits, topk_ids = torch.topk(t_logits, k=topk, dim=-1)
            teacher_probs = F.softmax(teacher_topk_logits.to(torch.float32), dim=-1)
            student_selected = torch.gather(s_logits, dim=-1, index=topk_ids)
            student_log_probs = F.log_softmax(student_selected.to(torch.float32), dim=-1)
            token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            block_sum = block_sum + token_kl.to(student_logits.dtype).sum()
            block_count = block_count + student_logits.new_tensor(float(pos_chunk.numel()))
        return block_sum, block_count

    for row_idx in range(batch["input_ids"].shape[0]):
        cot_sum, cot_count = block_kl_sum(
            row_idx,
            batch["cot_positions"][row_idx],
            batch["cot_position_mask"][row_idx],
        )
        answer_sum, answer_count = block_kl_sum(
            row_idx,
            batch["answer_positions"][row_idx],
            batch["answer_position_mask"][row_idx],
        )
        cot_loss = cot_sum / cot_count.clamp_min(1.0)
        answer_loss = answer_sum / answer_count.clamp_min(1.0)

        row_losses.append(0.5 * (cot_loss + answer_loss))
        cot_losses.append(cot_loss)
        answer_losses.append(answer_loss)
        cot_counts.append(cot_count.detach())
        answer_counts.append(answer_count.detach())

    cot_loss_mean = torch.stack(cot_losses).mean()
    answer_loss_mean = torch.stack(answer_losses).mean()
    loss = torch.stack(row_losses).mean()
    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach().cpu()),
            "cot_kl": float(cot_loss_mean.detach().cpu()),
            "answer_kl": float(answer_loss_mean.detach().cpu()),
            "kl_topk": float(topk),
            "cot_tokens": float(torch.stack(cot_counts).to(torch.float32).mean().detach().cpu()),
            "answer_tokens": float(torch.stack(answer_counts).to(torch.float32).mean().detach().cpu()),
        }
    return loss, metrics


def normalized_completion_topk_kl_loss(
    model,
    prefix: SoftPrefix,
    batch: dict[str, Any],
    *,
    chunk_tokens: int,
    kl_topk: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Top-k teacher KL over the whole retain completion trajectory.

    This is the trust-region distance used by retain_budgeted_forgetce:
    teacher is the frozen base model without prefix, student is base+prefix,
    and prompt tokens are excluded.  Unlike normalized_block_topk_kl_loss,
    CoT and answer are not separately reweighted; the retain budget measures
    the full generated completion as one behavior trajectory. The top-k KL
    normalizes the student distribution inside the teacher top-k subset.
    """
    with torch.no_grad():
        teacher_out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        teacher_logits = teacher_out.logits

    student_embeds, student_attn = insert_prompt_end_prefix_for_batch(
        model,
        prefix,
        batch["input_ids"],
        batch["attention_mask"],
        batch["prompt_lens"],
    )
    student_out = model(inputs_embeds=student_embeds, attention_mask=student_attn, use_cache=False)
    student_logits = student_out.logits
    prefix_len = int(prefix.embedding.shape[0])
    vocab_size = int(student_logits.shape[-1])
    topk = min(max(1, int(kl_topk)), vocab_size)

    row_losses: list[torch.Tensor] = []
    cot_losses: list[torch.Tensor] = []
    answer_losses: list[torch.Tensor] = []
    cot_counts: list[torch.Tensor] = []
    answer_counts: list[torch.Tensor] = []

    def block_kl_sum(row_idx: int, positions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_sum = student_logits.new_zeros(())
        block_count = student_logits.new_zeros(())
        for pos_chunk in _position_chunks(positions, mask, chunk_tokens):
            pos_chunk = pos_chunk.to(device=batch["input_ids"].device)
            teacher_pos = pos_chunk - 1
            student_pos = pos_chunk - 1 + prefix_len
            t_logits = teacher_logits[row_idx, teacher_pos, :].to(torch.float32)
            s_logits = student_logits[row_idx, student_pos, :].to(torch.float32)
            teacher_topk_logits, topk_ids = torch.topk(t_logits, k=topk, dim=-1)
            teacher_probs = F.softmax(teacher_topk_logits.to(torch.float32), dim=-1)
            student_selected = torch.gather(s_logits, dim=-1, index=topk_ids)
            student_log_probs = F.log_softmax(student_selected.to(torch.float32), dim=-1)
            token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            block_sum = block_sum + token_kl.to(student_logits.dtype).sum()
            block_count = block_count + student_logits.new_tensor(float(pos_chunk.numel()))
        return block_sum, block_count

    for row_idx in range(batch["input_ids"].shape[0]):
        cot_sum, cot_count = block_kl_sum(
            row_idx,
            batch["cot_positions"][row_idx],
            batch["cot_position_mask"][row_idx],
        )
        answer_sum, answer_count = block_kl_sum(
            row_idx,
            batch["answer_positions"][row_idx],
            batch["answer_position_mask"][row_idx],
        )
        total_count = (cot_count + answer_count).clamp_min(1.0)
        row_losses.append((cot_sum + answer_sum) / total_count)
        cot_losses.append(cot_sum / cot_count.clamp_min(1.0))
        answer_losses.append(answer_sum / answer_count.clamp_min(1.0))
        cot_counts.append(cot_count.detach())
        answer_counts.append(answer_count.detach())

    cot_loss_mean = torch.stack(cot_losses).mean()
    answer_loss_mean = torch.stack(answer_losses).mean()
    loss = torch.stack(row_losses).mean()
    with torch.no_grad():
        cot_count_mean = torch.stack(cot_counts).to(torch.float32).mean()
        answer_count_mean = torch.stack(answer_counts).to(torch.float32).mean()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "completion_kl": float(loss.detach().cpu()),
            "cot_kl": float(cot_loss_mean.detach().cpu()),
            "answer_kl": float(answer_loss_mean.detach().cpu()),
            "kl_topk": float(topk),
            "cot_tokens": float(cot_count_mean.detach().cpu()),
            "answer_tokens": float(answer_count_mean.detach().cpu()),
            "completion_tokens": float((cot_count_mean + answer_count_mean).detach().cpu()),
        }
    return loss, metrics


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _trainable_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def build_semantic_prefix_anchor(
    model,
    tokenizer,
    *,
    prefix_text: str,
    prefix_len: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if int(prefix_len) <= 0:
        raise ValueError("prefix_len must be positive")
    text = str(prefix_text or "")
    tokenized = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids = [int(token_id) for token_id in tokenized.get("input_ids", [])]
    if not token_ids:
        raise ValueError("prefix_init=semantic_text requires non-empty prefix_text after tokenization")
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        text_embeds = embed_layer(input_ids).detach()
        if text_embeds.shape[0] >= int(prefix_len):
            anchor = text_embeds[: int(prefix_len)].clone()
            fill_mode = "truncate"
        else:
            repeats = math.ceil(int(prefix_len) / int(text_embeds.shape[0]))
            anchor = text_embeds.repeat((repeats, 1))[: int(prefix_len)].clone()
            fill_mode = "repeat"
    meta = {
        "prefix_init": "semantic_text",
        "prefix_text": text,
        "prefix_text_token_count": len(token_ids),
        "prefix_len": int(prefix_len),
        "prefix_anchor_fill_mode": fill_mode,
        "prefix_anchor_norm": float(anchor.float().norm().detach().cpu()),
    }
    return anchor, meta


def initialize_prefix_embedding(
    prefix: SoftPrefix,
    anchor: torch.Tensor,
    *,
    init_std: float,
) -> None:
    with torch.no_grad():
        init = anchor.to(device=prefix.embedding.device, dtype=prefix.embedding.dtype)
        if float(init_std) > 0.0:
            init = init + torch.randn_like(init) * float(init_std)
        prefix.embedding.copy_(init)


def prefix_anchor_metrics(prefix: SoftPrefix, anchor: torch.Tensor | None) -> dict[str, float]:
    if anchor is None:
        return {
            "prefix_anchor_loss": 0.0,
            "prefix_delta_norm": 0.0,
            "prefix_anchor_norm": 0.0,
            "prefix_delta_norm_ratio": 0.0,
        }
    with torch.no_grad():
        anchor_t = anchor.to(device=prefix.embedding.device, dtype=prefix.embedding.dtype)
        delta = prefix.embedding - anchor_t
        anchor_norm = anchor_t.float().norm().clamp_min(1e-12)
        delta_norm = delta.float().norm()
        return {
            "prefix_anchor_loss": float(delta.float().pow(2).mean().detach().cpu()),
            "prefix_delta_norm": float(delta_norm.detach().cpu()),
            "prefix_anchor_norm": float(anchor_norm.detach().cpu()),
            "prefix_delta_norm_ratio": float((delta_norm / anchor_norm).detach().cpu()),
        }


def project_prefix_delta(prefix: SoftPrefix, anchor: torch.Tensor | None, max_delta_norm_ratio: float) -> None:
    if anchor is None or float(max_delta_norm_ratio) <= 0.0:
        return
    with torch.no_grad():
        anchor_t = anchor.to(device=prefix.embedding.device, dtype=prefix.embedding.dtype)
        delta = prefix.embedding - anchor_t
        anchor_norm = anchor_t.float().norm()
        if float(anchor_norm.detach().cpu()) <= 0.0:
            return
        max_norm = anchor_norm * float(max_delta_norm_ratio)
        delta_norm = delta.float().norm()
        if delta_norm > max_norm:
            scaled = delta * (max_norm.to(delta.device, dtype=delta.dtype) / delta_norm.to(delta.device, dtype=delta.dtype))
            prefix.embedding.copy_(anchor_t + scaled)


def _flatten_param_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for param in params:
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            chunks.append(param.grad.detach().float().reshape(-1))
    if not chunks:
        raise RuntimeError("no trainable parameters to flatten")
    return torch.cat(chunks, dim=0)


def _assign_flat_param_grad(params: list[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        view = flat_grad[offset : offset + numel].view_as(param).to(dtype=param.dtype, device=param.device)
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.detach().copy_(view)
        offset += numel
    if offset != int(flat_grad.numel()):
        raise RuntimeError(f"flat grad size mismatch: consumed={offset} total={int(flat_grad.numel())}")


def _cosine_reweighted_gradient(
    grad_vectors: list[torch.Tensor],
    *,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not grad_vectors:
        raise RuntimeError("agreement gradient requires at least one forget micro-batch")
    if len(grad_vectors) == 1:
        return grad_vectors[0], {
            "forget_agreement_cosine": 1.0,
            "forget_agreement_positive_frac": 1.0,
            "forget_agreement_weight_max": 1.0,
        }
    stack = torch.stack(grad_vectors, dim=0)
    mean_grad = stack.mean(dim=0)
    cosines = F.cosine_similarity(stack, mean_grad.unsqueeze(0), dim=1, eps=float(eps))
    weights = torch.clamp(cosines, min=0.0)
    positive = weights > 0
    if float(weights.sum().detach().cpu()) <= float(eps):
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(float(eps))
    agreed_grad = (stack * weights.unsqueeze(1)).sum(dim=0)
    return agreed_grad, {
        "forget_agreement_cosine": float(cosines.mean().detach().cpu()),
        "forget_agreement_positive_frac": float(positive.float().mean().detach().cpu()),
        "forget_agreement_weight_max": float(weights.max().detach().cpu()),
    }


def _mean_metric_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return {
        key: float(sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows)))
        for key in keys
    }


def parse_epoch_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value if int(item) > 0}
    text = str(value).strip()
    if not text:
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip() and int(item.strip()) > 0}


def sample_retain_records_for_epoch(
    retain_records: list[dict[str, Any]],
    *,
    epoch: int,
    seed: int,
    sample_n: int,
    sampler: str,
    retain_dataset_name: str,
    hard_dataset_name: str,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + int(epoch))
    sample_n = min(int(sample_n), len(retain_records))
    if sampler != "stratified_50_50":
        return rng.sample(retain_records, sample_n)
    normal = [r for r in retain_records if str(r.get("source_dataset")) == retain_dataset_name]
    hard = [r for r in retain_records if str(r.get("source_dataset")) == hard_dataset_name]
    if not normal or not hard:
        return rng.sample(retain_records, sample_n)
    hard_n = min(len(hard), sample_n // 2)
    normal_n = min(len(normal), sample_n - hard_n)
    selected_normal = rng.sample(normal, normal_n)
    selected_hard = rng.sample(hard, hard_n)
    leftover = sample_n - hard_n - normal_n
    if leftover > 0:
        selected_keys = {retain_source_key(r) for r in selected_normal + selected_hard}
        rest = [r for r in retain_records if retain_source_key(r) not in selected_keys]
        extra = rng.sample(rest, min(leftover, len(rest))) if rest else []
    else:
        extra = []
    sampled = selected_normal + selected_hard + extra
    rng.shuffle(sampled)
    return sampled


def build_forget_targets(
    *,
    encoder: BlockTargetEncoder,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        try:
            rows.append(encoder.encode_forget(record))
        except ValueError:
            skipped += 1
    return rows, {"sample_n": len(records), "targets": len(rows), "skipped": skipped}


def build_forget_raw_answer_targets(
    *,
    encoder: BlockTargetEncoder,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    answer_lens: list[int] = []
    for record in records:
        try:
            row = encoder.encode_forget_raw_answer(record)
            rows.append(row)
            answer_lens.append(int(row["target"]["raw_answer_len"]))
        except ValueError:
            skipped += 1
    return rows, {
        "sample_n": len(records),
        "targets": len(rows),
        "skipped": skipped,
        "raw_answer_tokens_mean": float(sum(answer_lens) / max(1, len(answer_lens))),
    }


def sample_forget_records_for_epoch(
    forget_records: list[dict[str, Any]],
    *,
    epoch: int,
    seed: int,
    sample_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_n = int(sample_n)
    total_n = len(forget_records)
    if sample_n <= 0 or sample_n >= total_n:
        indices = list(range(total_n))
        return list(forget_records), {
            "pool_n": total_n,
            "sample_n": total_n,
            "sampled_n": total_n,
            "sample_seed": int(seed),
            "epoch": int(epoch),
            "sample_indices": indices,
            "with_replacement": False,
        }
    rng = random.Random(int(seed) + int(epoch))
    indices = sorted(rng.sample(range(total_n), sample_n))
    sampled = [forget_records[idx] for idx in indices]
    return sampled, {
        "pool_n": total_n,
        "sample_n": sample_n,
        "sampled_n": len(sampled),
        "sample_seed": int(seed),
        "epoch": int(epoch),
        "sample_indices": indices,
        "with_replacement": False,
    }


def split_fixed_forget_train_holdout(
    examples: list[dict[str, Any]],
    *,
    train_fraction: float,
    train_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total_n = len(examples)
    if total_n <= 0:
        raise ValueError("cannot split an empty forget set")
    requested_n = int(train_n)
    fraction = float(train_fraction)
    if requested_n <= 0 and fraction > 0.0 and fraction < 1.0:
        requested_n = int(round(total_n * fraction))
    if requested_n <= 0 or requested_n >= total_n:
        train_indices = list(range(total_n))
    else:
        requested_n = max(1, min(total_n - 1, requested_n))
        rng = random.Random(int(seed))
        train_indices = sorted(rng.sample(range(total_n), requested_n))
    train_index_set = set(train_indices)
    holdout_indices = [idx for idx in range(total_n) if idx not in train_index_set]
    train_examples = [examples[idx] for idx in train_indices]
    holdout_examples = [examples[idx] for idx in holdout_indices]
    meta = {
        "total_n": total_n,
        "train_n": len(train_examples),
        "holdout_n": len(holdout_examples),
        "requested_train_n": int(train_n),
        "requested_train_fraction": float(train_fraction),
        "effective_train_fraction": float(len(train_examples) / max(1, total_n)),
        "seed": int(seed),
        "train_indices": train_indices,
        "holdout_indices": holdout_indices,
        "train_source_indices": [int(examples[idx].get("source_idx", -1)) for idx in train_indices],
        "holdout_source_indices": [int(examples[idx].get("source_idx", -1)) for idx in holdout_indices],
    }
    return train_examples, holdout_examples, meta


def build_retain_targets(
    *,
    encoder: BlockTargetEncoder,
    records: list[dict[str, Any]],
    trajectory_by_key: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    has_think_end = 0
    cot_lens: list[int] = []
    answer_lens: list[int] = []
    for record in records:
        try:
            row = encoder.encode_retain(record, trajectory_by_key[retain_source_key(record)])
            rows.append(row)
            has_think_end += int(bool(row["target"].get("retain_has_think_end")))
            cot_lens.append(int(row["target"]["cot_len"]))
            answer_lens.append(int(row["target"]["answer_len"]))
        except (KeyError, ValueError):
            skipped += 1
    stats = {
        "sample_n": len(records),
        "targets": len(rows),
        "skipped": skipped,
        "dataset_counts": count_by_key(records, "source_dataset"),
        "retain_has_think_end_mean": float(has_think_end / max(1, len(rows))),
        "cot_tokens_mean": float(sum(cot_lens) / max(1, len(cot_lens))),
        "answer_tokens_mean": float(sum(answer_lens) / max(1, len(answer_lens))),
    }
    return rows, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GUARD GTA soft guidance-token training.")
    p.add_argument("--config", default=None)
    p.add_argument("--stamp", default=None)
    p.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
    p.add_argument("--model_family", default="llama3-8b")
    p.add_argument("--forget_path", default=str(DEFAULT_FORGET))
    p.add_argument("--forget_train_fraction", type=float, default=1.0)
    p.add_argument("--forget_train_n", type=int, default=0)
    p.add_argument("--forget_train_seed", type=int, default=1707)
    p.add_argument("--retain_path", default=str(DEFAULT_RETAIN))
    p.add_argument("--retain_extra_forget_path", default="")
    p.add_argument("--retain_extra_exclude_task_id", default="")
    p.add_argument("--idkcot_path", default=str(DEFAULT_IDKCOT))
    p.add_argument("--idontknow_path", default=str(DEFAULT_IDONTKNOW))
    p.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    p.add_argument("--output_dir", default="")
    p.add_argument("--results_dir", default="")
    p.add_argument("--trajectory_cache", default="")
    p.add_argument("--force_rebuild_trajectories", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task_id", default="1")
    p.add_argument("--forget_limit", type=int, default=200)
    p.add_argument("--forget_epoch_samples", type=int, default=0)
    p.add_argument("--forget_epoch_seed", type=int, default=1707)
    p.add_argument("--retain_limit", type=int, default=1800)
    p.add_argument("--retain_sample", default="random", choices=["head", "random"])
    p.add_argument("--eval_retain_sample", default="train", choices=["train", "head_exclude_train", "random_exclude_train"])
    p.add_argument("--eval_retain_seed", type=int, default=-1)
    p.add_argument("--eval_retain_limit", type=int, default=40)
    p.add_argument("--eval_forget_limit", type=int, default=40)
    p.add_argument("--prefix_init", default="random", choices=["random", "semantic_text"])
    p.add_argument("--prefix_text", default="")
    p.add_argument("--prefix_len", type=int, default=20)
    p.add_argument("--prefix_position", default="prompt_end")
    p.add_argument("--loss_mode", default="forgetce_retainkl")
    p.add_argument("--lambda_prefix_anchor", type=float, default=0.0)
    p.add_argument("--prefix_delta_init_std", type=float, default=0.0)
    p.add_argument("--prefix_max_delta_norm_ratio", type=float, default=0.0)
    p.add_argument("--lambda_retain", type=float, default=1.0)
    p.add_argument("--lambda_cot_ce", type=float, default=1.0)
    p.add_argument("--lambda_answer_ce", type=float, default=1.0)
    p.add_argument("--enforce_equal_forget_block_weights", type=str2bool, default=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lambda_cot_dpo", type=float, default=1.0)
    p.add_argument("--lambda_answer_dpo", type=float, default=1.0)
    p.add_argument("--dpo_reference_cache", default="")
    p.add_argument("--dpo_reference_batch_size", type=int, default=0)
    p.add_argument("--force_rebuild_dpo_reference_cache", action="store_true")
    p.add_argument("--lambda_erase", type=float, default=0.0)
    p.add_argument("--lambda_retain_mcq", type=float, default=0.0)
    p.add_argument("--retain_mcq_temperature", type=float, default=1.0)
    p.add_argument("--retain_mcq_samples_per_epoch", type=int, default=0)
    p.add_argument("--retain_budget_epsilon", type=float, default=0.02)
    p.add_argument("--lambda_retain_init", type=float, default=0.0)
    p.add_argument("--lambda_retain_max", type=float, default=10.0)
    p.add_argument("--lambda_retain_lr", type=float, default=0.1)
    p.add_argument("--kl_topk", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--forget_batch_size", type=int, default=2)
    p.add_argument("--forget_gradient_mode", default="standard", choices=["standard", "agreement"])
    p.add_argument("--forget_agreement_microbatches", type=int, default=1)
    p.add_argument("--forget_agreement_eps", type=float, default=1e-12)
    p.add_argument("--retain_batch_size", type=int, default=2)
    p.add_argument("--retain_grad_accum_steps", type=int, default=1)
    p.add_argument("--max_length", type=int, default=2560)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--retain_max_new_tokens", type=int, default=2048)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_start_epoch", type=int, default=5)
    p.add_argument("--save_epochs", default="")
    p.add_argument("--score_eval_every", type=int, default=0)
    p.add_argument("--score_eval_start_epoch", type=int, default=0)
    p.add_argument("--score_eval_epochs", default="")
    p.add_argument("--score_eval_include_epoch0", action="store_true")
    p.add_argument("--score_sample_n", type=int, default=50)
    p.add_argument("--score_retain_sample_n", type=int, default=0)
    p.add_argument("--score_retain_path", default="")
    p.add_argument("--score_retain_limit", type=int, default=0)
    p.add_argument("--score_retain_sample", default="head", choices=["head", "random"])
    p.add_argument("--score_sample_seed", type=int, default=20260516)
    p.add_argument("--score_forget_only", action="store_true")
    p.add_argument("--score_holdout_from_unused_forget", action="store_true")
    p.add_argument("--score_forget_judge_mode", default="source_leak", choices=["source_leak", "star_safety"])
    p.add_argument("--score_retain_mode", default="generative_judge", choices=["generative_judge", "mcq_logprob"])
    p.add_argument("--score_holdout_forget_path", default="")
    p.add_argument("--score_holdout_sample_n", type=int, default=0)
    p.add_argument("--score_threshold", type=float, default=0.8)
    p.add_argument("--score_penalty", type=float, default=2.0)
    p.add_argument("--score_mode", default="harmonic_mean")
    p.add_argument("--score_forget_weight", type=float, default=0.7)
    p.add_argument("--score_retain_weight", type=float, default=0.3)
    p.add_argument("--score_judge_base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--score_judge_model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--score_judge_api_key_env", default="GUARD_JUDGE_API_KEY")
    p.add_argument("--score_judge_timeout", type=float, default=60.0)
    p.add_argument("--score_judge_max_retries", type=int, default=2)
    p.add_argument("--score_judge_retry_sleep", type=float, default=1.0)
    p.add_argument("--score_judge_retry_window", type=float, default=0.0)
    p.add_argument("--score_judge_max_field_chars", type=int, default=20000)
    p.add_argument("--score_judge_skip_without_api_key", action="store_true")
    p.add_argument("--score_forget_mcq_every", type=int, default=0)
    p.add_argument("--score_forget_mcq_start_epoch", type=int, default=0)
    p.add_argument("--score_forget_mcq_epochs", default="")
    p.add_argument("--score_forget_mcq_sample_n", type=int, default=0)
    p.add_argument("--score_forget_mcq_seed", type=int, default=20260520)
    p.add_argument("--generation_batch_size", type=int, default=20)
    p.add_argument("--trajectory_generation_batch_size", type=int, default=30)
    p.add_argument("--retain_sampler", default="stratified_50_50", choices=["random", "stratified_50_50"])
    p.add_argument("--retain_epoch_samples", type=int, default=0)
    p.add_argument("--retain_epoch_seed", type=int, default=1707)
    p.add_argument("--chunk_tokens", type=int, default=64)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--skip_baseline_eval", action="store_true")
    p.add_argument("--skip_final_eval", action="store_true")
    p.add_argument("--build_retain_cache_only", action="store_true")
    p.add_argument("--resume_from_epoch", type=int, default=0)
    p.add_argument("--resume_checkpoint", default="")
    p.add_argument("--score_only_epoch", type=int, default=0)
    p.add_argument("--score_only_checkpoint", default="")
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"config must be a mapping: {config_path}")
        valid_dests = {action.dest for action in p._actions}
        unknown = sorted(set(payload) - valid_dests)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        p.set_defaults(**payload)
    args = p.parse_args()
    if str(args.loss_mode) == "retain_budgeted_forgetce":
        args.method = "retain_budgeted_forgetce_prefix_tuning"
    elif str(args.loss_mode) == "prefixdpo_retainkl":
        args.method = "prefixdpo_retainkl_prefix_tuning"
    else:
        args.method = "forgetce_retainkl_prefix_tuning"
    return args


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
        raise ValueError("this script supports prompt_end prefix insertion only")
    if float(args.lambda_prefix_anchor) < 0.0:
        raise ValueError("lambda_prefix_anchor must be non-negative")
    if float(args.prefix_delta_init_std) < 0.0:
        raise ValueError("prefix_delta_init_std must be non-negative")
    if float(args.prefix_max_delta_norm_ratio) < 0.0:
        raise ValueError("prefix_max_delta_norm_ratio must be non-negative")
    if str(args.prefix_init) == "semantic_text" and not str(args.prefix_text or "").strip():
        raise ValueError("prefix_init=semantic_text requires non-empty prefix_text")
    supported_loss_modes = {
        "forgetce_retainkl",
        "retain_budgeted_forgetce",
        "prefixdpo_retainkl",
    }
    if str(args.loss_mode) not in supported_loss_modes:
        raise ValueError(f"this script supports loss_mode in {sorted(supported_loss_modes)}")
    if float(args.lambda_retain) < 0.0:
        raise ValueError("lambda_retain must be non-negative")
    if float(args.lambda_retain_mcq) < 0.0:
        raise ValueError("lambda_retain_mcq must be non-negative")
    if float(args.retain_mcq_temperature) <= 0.0:
        raise ValueError("retain_mcq_temperature must be positive")
    if int(args.retain_mcq_samples_per_epoch) < 0:
        raise ValueError("retain_mcq_samples_per_epoch must be non-negative")
    if float(args.lambda_cot_ce) < 0.0 or float(args.lambda_answer_ce) < 0.0:
        raise ValueError("lambda_cot_ce and lambda_answer_ce must be non-negative")
    _validate_block_weights(float(args.lambda_cot_ce), float(args.lambda_answer_ce))
    if bool(args.enforce_equal_forget_block_weights) and not math.isclose(
        float(args.lambda_cot_ce),
        float(args.lambda_answer_ce),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "GUARD GTA uses equal block-normalized forget CE by default. "
            "Set lambda_cot_ce=lambda_answer_ce, or set "
            "enforce_equal_forget_block_weights=false for ablations."
        )
    if float(args.beta) <= 0.0:
        raise ValueError("beta must be positive")
    if float(args.lambda_cot_dpo) < 0.0 or float(args.lambda_answer_dpo) < 0.0:
        raise ValueError("lambda_cot_dpo and lambda_answer_dpo must be non-negative")
    _validate_block_weights(float(args.lambda_cot_dpo), float(args.lambda_answer_dpo))
    if float(args.lambda_erase) < 0.0:
        raise ValueError("lambda_erase must be non-negative")
    if float(args.retain_budget_epsilon) < 0.0:
        raise ValueError("retain_budget_epsilon must be non-negative")
    if float(args.lambda_retain_init) < 0.0:
        raise ValueError("lambda_retain_init must be non-negative")
    if float(args.lambda_retain_max) < 0.0:
        raise ValueError("lambda_retain_max must be non-negative")
    if float(args.lambda_retain_lr) < 0.0:
        raise ValueError("lambda_retain_lr must be non-negative")
    if float(args.lambda_retain_init) > float(args.lambda_retain_max):
        raise ValueError("lambda_retain_init must be <= lambda_retain_max")
    if int(args.kl_topk) <= 0:
        raise ValueError("kl_topk must be positive")
    args.save_epoch_set = sorted(parse_epoch_set(args.save_epochs))
    args.score_epoch_set = sorted(parse_epoch_set(args.score_eval_epochs))
    args.score_forget_mcq_epoch_set = sorted(parse_epoch_set(args.score_forget_mcq_epochs))
    if int(args.steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if int(args.resume_from_epoch) < 0:
        raise ValueError("resume_from_epoch must be non-negative")
    if int(args.score_only_epoch) < 0:
        raise ValueError("score_only_epoch must be non-negative")
    if int(args.forget_batch_size) <= 0 or int(args.retain_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    if int(args.forget_agreement_microbatches) <= 0:
        raise ValueError("forget_agreement_microbatches must be positive")
    if int(args.dpo_reference_batch_size) < 0:
        raise ValueError("dpo_reference_batch_size must be non-negative")
    if str(args.forget_gradient_mode) == "standard" and int(args.forget_agreement_microbatches) != 1:
        log("forget_gradient_mode=standard ignores forget_agreement_microbatches; using one forget batch per step")
    if str(args.loss_mode) == "prefixdpo_retainkl" and str(args.forget_gradient_mode) != "standard":
        raise ValueError("prefixdpo_retainkl currently requires forget_gradient_mode=standard")
    if float(args.forget_agreement_eps) <= 0.0:
        raise ValueError("forget_agreement_eps must be positive")
    if int(args.retain_grad_accum_steps) <= 0:
        raise ValueError("retain_grad_accum_steps must be positive")
    if int(args.score_eval_every) < 0:
        raise ValueError("score_eval_every must be non-negative")
    if int(args.score_sample_n) < 0:
        raise ValueError("score_sample_n must be non-negative")
    if int(args.score_retain_sample_n) < 0:
        raise ValueError("score_retain_sample_n must be non-negative")
    if int(args.score_retain_limit) < 0:
        raise ValueError("score_retain_limit must be non-negative")
    if int(args.score_holdout_sample_n) < 0:
        raise ValueError("score_holdout_sample_n must be non-negative")
    if int(args.score_forget_mcq_every) < 0:
        raise ValueError("score_forget_mcq_every must be non-negative")
    if int(args.score_forget_mcq_sample_n) < 0:
        raise ValueError("score_forget_mcq_sample_n must be non-negative")
    if float(args.forget_train_fraction) <= 0.0 or float(args.forget_train_fraction) > 1.0:
        raise ValueError("forget_train_fraction must be in (0, 1]")
    if int(args.forget_train_n) < 0:
        raise ValueError("forget_train_n must be non-negative")
    if bool(args.score_forget_only) and int(args.score_sample_n) > 0:
        if not str(args.score_holdout_forget_path or "") and not bool(args.score_holdout_from_unused_forget):
            raise ValueError(
                "score_forget_only requires score_holdout_forget_path or score_holdout_from_unused_forget"
            )
    if bool(args.score_forget_only) and str(args.score_retain_mode) != "generative_judge":
        raise ValueError("score_retain_mode is only used when score_forget_only=false")
    judge_config = JudgeConfig(
        base_url=str(args.score_judge_base_url),
        model=str(args.score_judge_model),
        api_key_env=str(args.score_judge_api_key_env),
        timeout=float(args.score_judge_timeout),
        max_retries=int(args.score_judge_max_retries),
        retry_sleep=float(args.score_judge_retry_sleep),
        retry_window=float(args.score_judge_retry_window),
        max_field_chars=int(args.score_judge_max_field_chars),
        threshold=float(args.score_threshold),
        penalty=float(args.score_penalty),
        score_mode=str(args.score_mode),
        forget_weight=float(args.score_forget_weight),
        retain_weight=float(args.score_retain_weight),
    )
    score_eval_enabled = (int(args.score_eval_every) > 0 or bool(args.score_epoch_set)) and int(args.score_sample_n) > 0
    if score_eval_enabled:
        validate_judge_config(judge_config, require_api_key=not bool(args.score_judge_skip_without_api_key))
        if bool(args.score_judge_skip_without_api_key) and not os.getenv(judge_config.api_key_env):
            log(f"score eval disabled because {judge_config.api_key_env} is not set")
            score_eval_enabled = False
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
        int(args.forget_limit),
    )
    forget_examples = attach_raw_forget_fields(forget_examples, Path(args.forget_path))
    forget_records = [
        {
            **ex,
            "prompt": build_prefix_training_prompt(ex["question"], model_cfg, args.prefix_position),
        }
        for ex in forget_examples
    ]
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
    retain_records = apply_prefix_training_prompt(retain_records, model_cfg, args.prefix_position)
    if str(args.score_retain_path or ""):
        score_retain_pool_records = build_retain_records(
            Path(args.score_retain_path),
            model_cfg,
            limit=int(args.score_retain_limit),
            seed=int(args.score_sample_seed),
            sample=str(args.score_retain_sample),
        )
        score_retain_pool_records = apply_prefix_training_prompt(
            score_retain_pool_records,
            model_cfg,
            args.prefix_position,
        )
    else:
        score_retain_pool_records = retain_records
    all_forget_records = list(forget_records)
    forget_train_records, forget_holdout_records, forget_train_split_meta = split_fixed_forget_train_holdout(
        all_forget_records,
        train_fraction=float(args.forget_train_fraction),
        train_n=int(args.forget_train_n),
        seed=int(args.forget_train_seed if int(args.forget_train_seed) >= 0 else args.seed + 5000003),
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
    retain_eval_records = apply_prefix_training_prompt(retain_eval_records, model_cfg, args.prefix_position)
    forget_eval_records = forget_train_records[: int(args.eval_forget_limit or len(forget_train_records))]
    score_forget_in_records: list[dict[str, Any]] = []
    score_forget_out_records: list[dict[str, Any]] = []
    if score_eval_enabled and bool(args.score_forget_only):
        if str(args.score_holdout_forget_path or ""):
            score_holdout_records = build_forget_probe_records(
                Path(args.score_holdout_forget_path),
                model_cfg,
                task_id=str(args.task_id),
                limit=0,
            )
        else:
            score_holdout_records = [dict(row) for row in forget_holdout_records]
        score_forget_in_records, score_forget_out_records, score_sample_meta = sample_forget_in_out_records(
            forget_train_records,
            score_holdout_records,
            in_sample_n=int(args.score_sample_n),
            out_sample_n=int(args.score_holdout_sample_n or args.score_sample_n),
            seed=int(args.score_sample_seed),
        )
        score_forget_records, score_retain_records = [], []
    elif score_eval_enabled:
        score_forget_records, score_retain_records, score_sample_meta = sample_selection_records(
            forget_records,
            score_retain_pool_records,
            sample_n=int(args.score_sample_n),
            retain_sample_n=(
                int(args.score_retain_sample_n)
                if int(args.score_retain_sample_n) > 0
                else int(args.score_sample_n)
            ),
            seed=int(args.score_sample_seed),
        )
    else:
        score_forget_records, score_retain_records, score_sample_meta = [], [], {}
    write_jsonl(output_dir / "forget_examples.jsonl", all_forget_records)
    write_jsonl(output_dir / "forget_train_examples.jsonl", forget_train_records)
    write_jsonl(output_dir / "forget_holdout_examples.jsonl", forget_holdout_records)
    write_jsonl(output_dir / "retain_records.jsonl", retain_records)
    write_jsonl(output_dir / "retain_eval_records.jsonl", retain_eval_records)
    write_jsonl(results_dir / "retain_eval_records.jsonl", retain_eval_records)
    if score_eval_enabled and bool(args.score_forget_only):
        score_dir = results_dir / (
            f"score_forget_only_seed{int(args.score_sample_seed)}"
            f"_in{int(args.score_sample_n)}_out{int(args.score_holdout_sample_n or args.score_sample_n)}"
        )
        score_dir.mkdir(parents=True, exist_ok=True)
        write_json(score_dir / "sample_indices.json", score_sample_meta)
        write_jsonl(score_dir / "score_forget_in_records.jsonl", score_forget_in_records)
        write_jsonl(score_dir / "score_forget_out_records.jsonl", score_forget_out_records)
    elif score_eval_enabled:
        score_retain_n = int(args.score_retain_sample_n) if int(args.score_retain_sample_n) > 0 else int(args.score_sample_n)
        score_dir = results_dir / (
            f"score_eval_seed{int(args.score_sample_seed)}"
            f"_forget{int(args.score_sample_n)}_retain{score_retain_n}"
        )
        score_dir.mkdir(parents=True, exist_ok=True)
        write_json(score_dir / "sample_indices.json", score_sample_meta)
        write_jsonl(score_dir / "score_forget_records.jsonl", score_forget_records)
        write_jsonl(score_dir / "score_retain_records.jsonl", score_retain_records)
    else:
        score_dir = results_dir / "score_eval_disabled"
    split_stats = {
        "forget_total_n": len(all_forget_records),
        "forget_train_n": len(forget_train_records),
        "forget_holdout_n": len(forget_holdout_records),
        "forget_train_fraction": float(args.forget_train_fraction),
        "forget_train_n_requested": int(args.forget_train_n),
        "forget_train_seed": int(args.forget_train_seed if int(args.forget_train_seed) >= 0 else args.seed + 5000003),
        "forget_train_split_meta": forget_train_split_meta,
        "forget_epoch_samples": int(args.forget_epoch_samples),
        "forget_epoch_seed": int(args.forget_epoch_seed),
        "retain_train_n": len(retain_records),
        "score_retain_pool_n": len(score_retain_pool_records),
        "retain_eval_n": len(retain_eval_records),
        "score_eval_enabled": bool(score_eval_enabled),
        "score_forget_only": bool(args.score_forget_only),
        "score_forget_judge_mode": str(args.score_forget_judge_mode),
        "score_retain_mode": str(args.score_retain_mode),
        "score_retain_path": str(args.score_retain_path or ""),
        "score_retain_limit": int(args.score_retain_limit),
        "score_retain_sample": str(args.score_retain_sample),
        "score_holdout_forget_path": str(args.score_holdout_forget_path or ""),
        "score_holdout_sample_n": int(args.score_holdout_sample_n or args.score_sample_n),
        "score_sample_n": int(args.score_sample_n),
        "score_retain_sample_n": int(
            args.score_retain_sample_n if int(args.score_retain_sample_n) > 0 else args.score_sample_n
        ),
        "score_threshold": float(args.score_threshold),
        "score_eval_every": int(args.score_eval_every),
        "score_eval_epochs": list(args.score_epoch_set),
        "score_eval_include_epoch0": bool(args.score_eval_include_epoch0),
        "retain_sample": str(args.retain_sample),
        "retain_sampler": str(args.retain_sampler),
        "retain_extra_forget_path": str(args.retain_extra_forget_path or ""),
        "retain_extra_exclude_task_id": str(args.retain_extra_exclude_task_id or args.task_id),
        "eval_retain_sample": str(args.eval_retain_sample),
        "seed": int(args.seed),
        "train_dataset_counts": count_by_key(retain_records, "source_dataset"),
        "eval_dataset_counts": count_by_key(retain_eval_records, "source_dataset"),
    }
    write_json(output_dir / "retain_split_stats.json", split_stats)
    write_json(results_dir / "retain_split_stats.json", split_stats)
    log(
        f"loaded forget={len(all_forget_records)} forget_train={len(forget_train_records)} "
        f"forget_holdout={len(forget_holdout_records)} forget_epoch_samples={int(args.forget_epoch_samples)} "
        f"retain_pool={len(retain_records)} retain_eval={len(retain_eval_records)} "
        f"method={args.method} loss_mode={args.loss_mode} lambda_retain={args.lambda_retain} "
        f"beta={args.beta} lambda_cot_dpo={args.lambda_cot_dpo} lambda_answer_dpo={args.lambda_answer_dpo} "
        f"lambda_erase={args.lambda_erase} "
        f"kl_topk={args.kl_topk}"
    )

    model, tokenizer = load_model_and_tokenizer(args.model_path, get_torch_dtype(args.dtype), args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    retain_mcq_pool: list[dict[str, Any]] = []
    retain_mcq_pool_stats: dict[str, Any] = {"pool_n": 0, "samples": 0, "skipped": 0}
    if float(args.lambda_retain_mcq) > 0.0 or int(args.retain_mcq_samples_per_epoch) > 0:
        retain_mcq_pool, retain_mcq_pool_stats = build_retain_mcq_samples(retain_records, tokenizer)
        write_json(output_dir / "retain_mcq_pool_stats.json", retain_mcq_pool_stats)
        write_json(results_dir / "retain_mcq_pool_stats.json", retain_mcq_pool_stats)
        log("built retain_mcq_pool " + json.dumps(retain_mcq_pool_stats, ensure_ascii=False, sort_keys=True))
        if float(args.lambda_retain_mcq) > 0.0 and int(args.retain_mcq_samples_per_epoch) > 0 and not retain_mcq_pool:
            raise ValueError("lambda_retain_mcq > 0 but no retain MCQ samples could be built")

    hidden = int(model.get_input_embeddings().embedding_dim)
    prefix = SoftPrefix(int(args.prefix_len), hidden).to(args.device)
    prefix_anchor: torch.Tensor | None = None
    prefix_anchor_meta: dict[str, Any] = {
        "prefix_init": str(args.prefix_init),
        "prefix_len": int(args.prefix_len),
        "prefix_anchor_norm": 0.0,
    }
    if str(args.prefix_init) == "semantic_text":
        prefix_anchor, prefix_anchor_meta = build_semantic_prefix_anchor(
            model,
            tokenizer,
            prefix_text=str(args.prefix_text),
            prefix_len=int(args.prefix_len),
            device=str(args.device),
        )
        initialize_prefix_embedding(prefix, prefix_anchor, init_std=float(args.prefix_delta_init_std))
        prefix_anchor = prefix_anchor.detach().to(device=args.device, dtype=prefix.embedding.dtype)
        prefix_anchor_meta.update(
            {
                "lambda_prefix_anchor": float(args.lambda_prefix_anchor),
                "prefix_delta_init_std": float(args.prefix_delta_init_std),
                "prefix_max_delta_norm_ratio": float(args.prefix_max_delta_norm_ratio),
                **prefix_anchor_metrics(prefix, prefix_anchor),
            }
        )
        write_json(output_dir / "prefix_anchor_meta.json", prefix_anchor_meta)
        write_json(results_dir / "prefix_anchor_meta.json", prefix_anchor_meta)
        log("initialized semantic prefix " + json.dumps(prefix_anchor_meta, ensure_ascii=False, sort_keys=True))
    prefix_trainable_params = _trainable_params(prefix)

    if int(args.score_only_epoch) > 0:
        if not score_eval_enabled:
            raise ValueError("score_only_epoch requires score eval to be enabled")
        score_epoch = int(args.score_only_epoch)
        score_path = (
            Path(args.score_only_checkpoint)
            if str(args.score_only_checkpoint or "")
            else output_dir / f"prefix_epoch{score_epoch}.pt"
        )
        if not score_path.exists():
            raise FileNotFoundError(f"score-only checkpoint not found: {score_path}")
        ckpt = torch.load(score_path, map_location=args.device)
        prefix.load_state_dict(ckpt["prefix_state_dict"])
        prefix.eval()
        model.config.use_cache = False
        log(f"score-only loaded checkpoint epoch={score_epoch} path={score_path}")
        if bool(args.score_forget_only):
            run_forget_only_score_eval(
                epoch=score_epoch,
                model=model,
                prefix=prefix,
                tokenizer=tokenizer,
                in_records=score_forget_in_records,
                out_records=score_forget_out_records,
                model_cfg=model_cfg,
                args=args,
                score_dir=score_dir,
                judge_config=judge_config,
            )
        else:
            run_score_eval(
                epoch=score_epoch,
                model=model,
                prefix=prefix,
                tokenizer=tokenizer,
                forget_records=score_forget_records,
                retain_records=score_retain_records,
                model_cfg=model_cfg,
                args=args,
                score_dir=score_dir,
                judge_config=judge_config,
            )
        return

    length_stats = summarize_training_lengths(forget_train_records, tokenizer, model_cfg)
    write_json(output_dir / "forget_length_stats.json", length_stats)
    if int(length_stats["full_tokens"]["max"]) > int(args.max_length):
        raise ValueError(
            f"max_length={args.max_length} would truncate forget idkcot data; "
            f"required_at_least={length_stats['full_tokens']['max']}"
        )
    dpo_ref_scores: dict[str, dict[str, float]] = {}
    if str(args.loss_mode) == "prefixdpo_retainkl":
        dpo_all_rows, dpo_all_stats = build_dpo_block_pairs(
            encoder=BlockTargetEncoder(
                tokenizer,
                model_cfg,
                max_length=int(args.max_length),
                prefix_position=str(args.prefix_position),
            ),
            records=forget_train_records,
        )
        if not dpo_all_rows:
            raise ValueError("prefixdpo_retainkl requires non-empty DPO pairs")
        write_json(output_dir / "dpo_pair_stats.json", dpo_all_stats)
        write_json(results_dir / "dpo_pair_stats.json", dpo_all_stats)
        log("built DPO pairs " + json.dumps(dpo_all_stats, ensure_ascii=False, sort_keys=True))
        dpo_reference_cache = (
            Path(args.dpo_reference_cache)
            if str(args.dpo_reference_cache or "")
            else output_dir / "dpo_reference_block_logp.jsonl"
        )
        dpo_ref_scores = compute_reference_dpo_scores(
            model,
            dpo_all_rows,
            tokenizer=tokenizer,
            cache_path=dpo_reference_cache,
            device=str(args.device),
            batch_size=(
                int(args.dpo_reference_batch_size)
                if int(args.dpo_reference_batch_size) > 0
                else int(args.forget_batch_size)
            ),
            chunk_tokens=int(args.chunk_tokens),
            cot_weight=float(args.lambda_cot_dpo),
            answer_weight=float(args.lambda_answer_dpo),
            force=bool(args.force_rebuild_dpo_reference_cache),
        )
        write_json(output_dir / "dpo_reference_cache_meta.json", {
            "cache": str(dpo_reference_cache),
            "rows": len(dpo_ref_scores),
            "lambda_cot_dpo": float(args.lambda_cot_dpo),
            "lambda_answer_dpo": float(args.lambda_answer_dpo),
            "beta": float(args.beta),
        })
        write_json(results_dir / "dpo_reference_cache_meta.json", {
            "cache": str(dpo_reference_cache),
            "rows": len(dpo_ref_scores),
            "lambda_cot_dpo": float(args.lambda_cot_dpo),
            "lambda_answer_dpo": float(args.lambda_answer_dpo),
            "beta": float(args.beta),
        })

    trajectory_cache = Path(args.trajectory_cache) if args.trajectory_cache else output_dir / "retain_base_trajectories.jsonl"
    retain_trajectories = load_or_generate_retain_trajectories(
        model,
        tokenizer,
        retain_records,
        trajectory_cache,
        device=args.device,
        batch_size=int(args.trajectory_generation_batch_size),
        max_new_tokens=int(args.retain_max_new_tokens),
        force=bool(args.force_rebuild_trajectories),
    )
    write_jsonl(output_dir / "retain_base_trajectories.used.jsonl", retain_trajectories)
    retain_base_by_key = {retain_source_key(row): row for row in retain_trajectories}
    if bool(args.build_retain_cache_only):
        log(f"build_retain_cache_only done trajectories={len(retain_trajectories)} cache={trajectory_cache}")
        return

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

    opt = torch.optim.AdamW(prefix.parameters(), lr=float(args.lr), weight_decay=0.0)
    encoder = BlockTargetEncoder(
        tokenizer,
        model_cfg,
        max_length=int(args.max_length),
        prefix_position=str(args.prefix_position),
    )

    start_epoch = 1
    if int(args.resume_from_epoch) > 0:
        resume_epoch = int(args.resume_from_epoch)
        resume_path = Path(args.resume_checkpoint) if str(args.resume_checkpoint or "") else output_dir / f"prefix_epoch{resume_epoch}.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=args.device)
        prefix.load_state_dict(ckpt["prefix_state_dict"])
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state:
            opt.load_state_dict(opt_state)
            log(f"resumed optimizer state from {resume_path}")
        else:
            log(f"resumed prefix from {resume_path}; optimizer state unavailable, using fresh AdamW")
        start_epoch = resume_epoch + 1
        if prefix_anchor is not None and "prefix_anchor" in ckpt:
            prefix_anchor = ckpt["prefix_anchor"].detach().to(device=args.device, dtype=prefix.embedding.dtype)

    forget_rows, forget_stats = build_forget_targets(encoder=encoder, records=forget_train_records)
    forget_raw_rows: list[dict[str, Any]] = []
    forget_raw_stats: dict[str, Any] = {
        "sample_n": 0,
        "targets": 0,
        "skipped": 0,
        "raw_cot_tokens_mean": 0.0,
        "raw_answer_tokens_mean": 0.0,
    }
    if float(args.lambda_erase) > 0.0:
        forget_raw_rows, forget_raw_stats = build_forget_raw_answer_targets(
            encoder=encoder,
            records=forget_train_records,
        )
        if not forget_raw_rows:
            raise ValueError("lambda_erase > 0 but no raw-answer targets could be built")
    write_json(output_dir / "forget_target_stats.json", forget_stats)
    write_json(output_dir / "forget_raw_target_stats.json", forget_raw_stats)
    log(
        f"built forget_targets={len(forget_rows)} skipped={forget_stats['skipped']} "
        f"raw_answer_targets={len(forget_raw_rows)} raw_answer_skipped={forget_raw_stats['skipped']}"
    )

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

    if score_eval_enabled and bool(args.score_eval_include_epoch0) and int(args.resume_from_epoch) <= 0:
        if bool(args.score_forget_only):
            log(
                f"score forget-only epoch=0 in_n={len(score_forget_in_records)} "
                f"out_n={len(score_forget_out_records)} threshold={float(args.score_threshold):.3f}"
            )
            run_forget_only_score_eval(
                epoch=0,
                model=model,
                prefix=None,
                tokenizer=tokenizer,
                in_records=score_forget_in_records,
                out_records=score_forget_out_records,
                model_cfg=model_cfg,
                args=args,
                score_dir=score_dir,
                judge_config=judge_config,
            )
        else:
            log(f"score eval epoch=0 sample_n={len(score_forget_records)} threshold={float(args.score_threshold):.3f}")
            run_score_eval(
                epoch=0,
                model=model,
                prefix=None,
                tokenizer=tokenizer,
                forget_records=score_forget_records,
                retain_records=score_retain_records,
                model_cfg=model_cfg,
                args=args,
                score_dir=score_dir,
                judge_config=judge_config,
            )

    train_log_path = output_dir / "train_log.json"
    if int(args.resume_from_epoch) > 0 and train_log_path.exists():
        train_log = [row for row in json.loads(train_log_path.read_text(encoding="utf-8")) if int(row.get("epoch", 0)) < start_epoch]
    else:
        train_log: list[dict[str, Any]] = []
    epoch_sample_path = output_dir / "epoch_pair_samples.json"
    if int(args.resume_from_epoch) > 0 and epoch_sample_path.exists():
        epoch_sample_log = [
            row for row in json.loads(epoch_sample_path.read_text(encoding="utf-8")) if int(row.get("epoch", 0)) < start_epoch
        ]
    else:
        epoch_sample_log: list[dict[str, Any]] = []
    seen_retain_source_keys: set[str] = set()
    global_step = int(train_log[-1].get("step", 0)) if train_log else 0
    if train_log and "lambda_retain_dual" in train_log[-1]:
        lambda_retain_dual = float(train_log[-1]["lambda_retain_dual"])
    elif str(args.loss_mode) == "retain_budgeted_forgetce":
        lambda_retain_dual = float(args.lambda_retain_init)
    else:
        lambda_retain_dual = float(args.lambda_retain)
    retain_dataset_name = Path(args.retain_path).stem
    hard_dataset_name = Path(args.retain_extra_forget_path).stem if args.retain_extra_forget_path else ""

    if start_epoch > 1:
        for prev_epoch in range(1, start_epoch):
            prev_sample_n = int(args.retain_epoch_samples)
            if prev_sample_n <= 0:
                prev_sample_n = (
                    int(args.steps_per_epoch)
                    * int(args.retain_batch_size)
                    * int(args.retain_grad_accum_steps)
                )
            prev_records = sample_retain_records_for_epoch(
                retain_records,
                epoch=prev_epoch,
                seed=int(args.retain_epoch_seed if int(args.retain_epoch_seed) >= 0 else args.seed + 4000003),
                sample_n=prev_sample_n,
                sampler=str(args.retain_sampler),
                retain_dataset_name=retain_dataset_name,
                hard_dataset_name=hard_dataset_name,
            )
            seen_retain_source_keys.update(retain_source_key(record) for record in prev_records)

    for epoch in range(start_epoch, int(args.epochs) + 1):
        forget_epoch_records = forget_train_records
        forget_epoch_sample_meta = {
            "pool_n": len(forget_train_records),
            "sample_n": len(forget_train_records),
            "sampled_n": len(forget_train_records),
            "sample_seed": int(args.forget_epoch_seed if int(args.forget_epoch_seed) >= 0 else args.seed + 5000003),
            "epoch": epoch,
            "sample_indices": list(range(len(forget_train_records))),
        }
        if int(args.forget_epoch_samples) > 0:
            forget_epoch_records, forget_epoch_sample_meta = sample_forget_records_for_epoch(
                forget_train_records,
                epoch=epoch,
                seed=int(args.forget_epoch_seed if int(args.forget_epoch_seed) >= 0 else args.seed + 5000003),
                sample_n=int(args.forget_epoch_samples),
            )
        forget_rows, forget_stats = build_forget_targets(
            encoder=encoder,
            records=forget_epoch_records,
        )
        dpo_rows: list[dict[str, Any]] = []
        dpo_stats: dict[str, Any] = {"sample_n": 0, "pairs": 0, "skipped": 0}
        if str(args.loss_mode) == "prefixdpo_retainkl":
            dpo_rows, dpo_stats = build_dpo_block_pairs(
                encoder=encoder,
                records=forget_epoch_records,
            )
            if not dpo_rows:
                raise ValueError(f"prefixdpo_retainkl built no DPO pairs for epoch={epoch}")
            missing_ref = [str(row["pair_key"]) for row in dpo_rows if str(row["pair_key"]) not in dpo_ref_scores]
            if missing_ref:
                raise ValueError(f"DPO reference cache missing {len(missing_ref)} pair keys, first={missing_ref[0]}")
        forget_raw_rows = []
        forget_raw_stats = {
            "sample_n": 0,
            "targets": 0,
            "skipped": 0,
            "raw_cot_tokens_mean": 0.0,
            "raw_answer_tokens_mean": 0.0,
        }
        if float(args.lambda_erase) > 0.0:
            forget_raw_rows, forget_raw_stats = build_forget_raw_answer_targets(
                encoder=encoder,
                records=forget_epoch_records,
            )
            if not forget_raw_rows:
                raise ValueError("lambda_erase > 0 but no raw-answer targets could be built for this epoch")
        retain_epoch_sample_n = int(args.retain_epoch_samples)
        if retain_epoch_sample_n <= 0:
            retain_epoch_sample_n = (
                int(args.steps_per_epoch)
                * int(args.retain_batch_size)
                * int(args.retain_grad_accum_steps)
            )
        retain_epoch_records = sample_retain_records_for_epoch(
            retain_records,
            epoch=epoch,
            seed=int(args.retain_epoch_seed if int(args.retain_epoch_seed) >= 0 else args.seed + 4000003),
            sample_n=retain_epoch_sample_n,
            sampler=str(args.retain_sampler),
            retain_dataset_name=retain_dataset_name,
            hard_dataset_name=hard_dataset_name,
        )
        seen_retain_source_keys.update(retain_source_key(record) for record in retain_epoch_records)
        retain_rows, retain_stats = build_retain_targets(
            encoder=encoder,
            records=retain_epoch_records,
            trajectory_by_key=retain_base_by_key,
        )
        retain_mcq_epoch_samples: list[dict[str, Any]] = []
        retain_mcq_sample_meta: dict[str, Any] = {
            "pool_n": len(retain_mcq_pool),
            "sample_n": 0,
            "sampled_n": 0,
            "sample_seed": int(args.retain_epoch_seed if int(args.retain_epoch_seed) >= 0 else args.seed + 4000003),
            "epoch": epoch,
            "sample_indices": [],
            "with_replacement": False,
        }
        if float(args.lambda_retain_mcq) > 0.0 and int(args.retain_mcq_samples_per_epoch) > 0:
            retain_mcq_epoch_samples, retain_mcq_sample_meta = sample_retain_mcq_samples_for_epoch(
                retain_mcq_pool,
                epoch=epoch,
                seed=int(args.retain_epoch_seed if int(args.retain_epoch_seed) >= 0 else args.seed + 4000003),
                sample_n=int(args.retain_mcq_samples_per_epoch),
            )
        epoch_sample = {
            "epoch": epoch,
            "forget": forget_stats,
            "dpo": dpo_stats,
            "forget_raw": forget_raw_stats,
            "forget_sample": forget_epoch_sample_meta,
            "retain": retain_stats,
            "retain_mcq": retain_mcq_pool_stats,
            "retain_mcq_sample": retain_mcq_sample_meta,
            "retain_unique_seen": len(seen_retain_source_keys),
        }
        epoch_sample_log.append(epoch_sample)
        write_json(output_dir / "epoch_pair_samples.json", epoch_sample_log)
        write_json(results_dir / "epoch_pair_samples.json", epoch_sample_log)
        log(
            f"epoch {epoch} built forget_targets={len(forget_rows)} dpo_pairs={len(dpo_rows)} "
            f"raw_answer_targets={len(forget_raw_rows)} "
            f"retain_targets={len(retain_rows)} "
            f"retain_unique_seen={len(seen_retain_source_keys)}"
        )

        if str(args.loss_mode) == "prefixdpo_retainkl":
            forget_loader = DataLoader(
                DPOBlockPairDataset(dpo_rows),
                batch_size=int(args.forget_batch_size),
                shuffle=True,
                collate_fn=lambda b: dpo_block_pair_collate(b, tokenizer.pad_token_id),
            )
        else:
            forget_loader = DataLoader(
                BlockTargetDataset(forget_rows),
                batch_size=int(args.forget_batch_size),
                shuffle=True,
                collate_fn=lambda b: block_target_collate(b, tokenizer.pad_token_id),
            )
        forget_raw_loader = (
            DataLoader(
                BlockTargetDataset(forget_raw_rows),
                batch_size=int(args.forget_batch_size),
                shuffle=True,
                collate_fn=lambda b: block_target_collate(b, tokenizer.pad_token_id),
            )
            if float(args.lambda_erase) > 0.0
            else None
        )
        retain_loader = DataLoader(
            BlockTargetDataset(retain_rows),
            batch_size=int(args.retain_batch_size),
            shuffle=True,
            collate_fn=lambda b: block_target_collate(b, tokenizer.pad_token_id),
        )
        retain_mcq_loader = (
            DataLoader(
                MCQPromptDataset(retain_mcq_epoch_samples),
                batch_size=max(1, math.ceil(len(retain_mcq_epoch_samples) / int(args.steps_per_epoch) / int(args.retain_grad_accum_steps))),
                shuffle=True,
                collate_fn=lambda b: _collate_mcq_prompts(b, tokenizer.pad_token_id),
            )
            if retain_mcq_epoch_samples
            else None
        )
        forget_iter = cycle_loader(forget_loader)
        forget_raw_iter = cycle_loader(forget_raw_loader) if forget_raw_loader is not None else None
        retain_iter = cycle_loader(retain_loader)
        retain_mcq_iter = cycle_loader(retain_mcq_loader) if retain_mcq_loader is not None else None

        prefix.train()
        epoch_forget_losses: list[float] = []
        epoch_retain_losses: list[float] = []
        epoch_constraint_losses: list[float] = []
        epoch_budget_excesses: list[float] = []
        epoch_constraint_active: list[float] = []
        epoch_lambda_duals: list[float] = []
        forget_cot_ces: list[float] = []
        forget_answer_ces: list[float] = []
        forget_raw_cot_uls: list[float] = []
        forget_raw_uls: list[float] = []
        forget_raw_cot_nlls: list[float] = []
        forget_raw_nlls: list[float] = []
        forget_raw_cot_probs: list[float] = []
        forget_raw_probs: list[float] = []
        dpo_margins: list[float] = []
        dpo_policy_margins: list[float] = []
        dpo_ref_margins: list[float] = []
        dpo_pref_accs: list[float] = []
        dpo_chosen_scores: list[float] = []
        dpo_rejected_scores: list[float] = []
        dpo_chosen_cot_logps: list[float] = []
        dpo_chosen_answer_logps: list[float] = []
        dpo_rejected_cot_logps: list[float] = []
        dpo_rejected_answer_logps: list[float] = []
        retain_cot_kls: list[float] = []
        retain_answer_kls: list[float] = []
        retain_mcq_kls: list[float] = []
        retain_mcq_agrees: list[float] = []
        forget_agreement_cosines: list[float] = []
        forget_agreement_positive_fracs: list[float] = []
        forget_agreement_weight_maxes: list[float] = []
        prefix_anchor_losses: list[float] = []
        prefix_delta_norms: list[float] = []
        prefix_delta_norm_ratios: list[float] = []
        log(f"epoch {epoch} {args.loss_mode} steps={args.steps_per_epoch}")
        for _ in range(int(args.steps_per_epoch)):
            opt.zero_grad(set_to_none=True)
            forget_metrics_rows: list[dict[str, float]] = []
            forget_raw_metrics_rows: list[dict[str, float]] = []
            forget_loss_values: list[float] = []
            if str(args.loss_mode) == "prefixdpo_retainkl":
                forget_batch = next(forget_iter)
                forget_device_batch = {
                    side: {
                        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                        for key, value in payload.items()
                    }
                    if isinstance(payload, dict)
                    else payload
                    for side, payload in forget_batch.items()
                }
                forget_loss, forget_metrics = prefix_dpo_block_loss(
                    model,
                    prefix,
                    forget_device_batch,
                    dpo_ref_scores,
                    beta=float(args.beta),
                    chunk_tokens=int(args.chunk_tokens),
                    cot_weight=float(args.lambda_cot_dpo),
                    answer_weight=float(args.lambda_answer_dpo),
                )
                forget_raw_metrics = {
                    "loss": 0.0,
                    "raw_block_ul": 0.0,
                    "raw_cot_ul": 0.0,
                    "raw_answer_ul": 0.0,
                    "raw_cot_nll": 0.0,
                    "raw_answer_nll": 0.0,
                    "raw_cot_prob": 0.0,
                    "raw_answer_prob": 0.0,
                    "raw_cot_tokens": 0.0,
                    "raw_answer_tokens": 0.0,
                }
                forget_loss.backward()
                forget_loss_value_for_log = float(forget_loss.detach().cpu())
                forget_agreement_cosines.append(1.0)
                forget_agreement_positive_fracs.append(1.0)
                forget_agreement_weight_maxes.append(1.0)
            elif str(args.forget_gradient_mode) == "agreement":
                forget_grad_vectors: list[torch.Tensor] = []
                for _forget_micro_step in range(int(args.forget_agreement_microbatches)):
                    opt.zero_grad(set_to_none=True)
                    forget_batch = next(forget_iter)
                    forget_device_batch = {
                        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                        for key, value in forget_batch.items()
                    }
                    micro_forget_loss, micro_forget_metrics = normalized_block_ce_loss(
                        model,
                        prefix,
                        forget_device_batch,
                        chunk_tokens=int(args.chunk_tokens),
                        cot_weight=float(args.lambda_cot_ce),
                        answer_weight=float(args.lambda_answer_ce),
                        normalize_block_weights=True,
                    )
                    micro_forget_raw_metrics: dict[str, float] = {
                        "loss": 0.0,
                        "raw_block_ul": 0.0,
                        "raw_cot_ul": 0.0,
                        "raw_answer_ul": 0.0,
                        "raw_cot_nll": 0.0,
                        "raw_answer_nll": 0.0,
                        "raw_cot_prob": 0.0,
                        "raw_answer_prob": 0.0,
                        "raw_cot_tokens": 0.0,
                        "raw_answer_tokens": 0.0,
                    }
                    if float(args.lambda_erase) > 0.0:
                        if forget_raw_iter is None:
                            raise RuntimeError("missing raw-answer loader for lambda_erase > 0")
                        forget_raw_batch = next(forget_raw_iter)
                        forget_raw_device_batch = {
                            key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                            for key, value in forget_raw_batch.items()
                        }
                        micro_forget_raw_loss, micro_forget_raw_metrics = normalized_raw_answer_unlikelihood_loss(
                            model,
                            prefix,
                            forget_raw_device_batch,
                            chunk_tokens=int(args.chunk_tokens),
                        )
                        micro_forget_loss = micro_forget_loss + float(args.lambda_erase) * micro_forget_raw_loss
                    micro_forget_metrics["loss"] = float(micro_forget_loss.detach().cpu())
                    micro_forget_loss.backward()
                    forget_grad_vectors.append(_flatten_param_grads(prefix_trainable_params))
                    forget_metrics_rows.append(micro_forget_metrics)
                    forget_raw_metrics_rows.append(micro_forget_raw_metrics)
                    forget_loss_values.append(float(micro_forget_loss.detach().cpu()))
                agreed_forget_grad, agreement_metrics = _cosine_reweighted_gradient(
                    forget_grad_vectors,
                    eps=float(args.forget_agreement_eps),
                )
                opt.zero_grad(set_to_none=True)
                _assign_flat_param_grad(prefix_trainable_params, agreed_forget_grad)
                forget_agreement_cosines.append(float(agreement_metrics["forget_agreement_cosine"]))
                forget_agreement_positive_fracs.append(float(agreement_metrics["forget_agreement_positive_frac"]))
                forget_agreement_weight_maxes.append(float(agreement_metrics["forget_agreement_weight_max"]))
                forget_loss_value_for_log = float(sum(forget_loss_values) / max(1, len(forget_loss_values)))
                forget_loss = next(param for param in prefix_trainable_params).sum() * 0.0 + forget_loss_value_for_log
                forget_metrics = _mean_metric_dict(forget_metrics_rows)
                forget_raw_metrics = _mean_metric_dict(forget_raw_metrics_rows)
            else:
                forget_batch = next(forget_iter)
                forget_device_batch = {
                    key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                    for key, value in forget_batch.items()
                }
                forget_loss, forget_metrics = normalized_block_ce_loss(
                    model,
                    prefix,
                    forget_device_batch,
                    chunk_tokens=int(args.chunk_tokens),
                    cot_weight=float(args.lambda_cot_ce),
                    answer_weight=float(args.lambda_answer_ce),
                    normalize_block_weights=True,
                )
                forget_raw_loss = forget_loss.new_zeros(())
                forget_raw_metrics = {
                    "loss": 0.0,
                    "raw_block_ul": 0.0,
                    "raw_cot_ul": 0.0,
                    "raw_answer_ul": 0.0,
                    "raw_cot_nll": 0.0,
                    "raw_answer_nll": 0.0,
                    "raw_cot_prob": 0.0,
                    "raw_answer_prob": 0.0,
                    "raw_cot_tokens": 0.0,
                    "raw_answer_tokens": 0.0,
                }
                if float(args.lambda_erase) > 0.0:
                    if forget_raw_iter is None:
                        raise RuntimeError("missing raw-answer loader for lambda_erase > 0")
                    forget_raw_batch = next(forget_raw_iter)
                    forget_raw_device_batch = {
                        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                        for key, value in forget_raw_batch.items()
                    }
                    forget_raw_loss, forget_raw_metrics = normalized_raw_answer_unlikelihood_loss(
                        model,
                        prefix,
                        forget_raw_device_batch,
                        chunk_tokens=int(args.chunk_tokens),
                    )
                    forget_loss = forget_loss + float(args.lambda_erase) * forget_raw_loss
                forget_metrics["loss"] = float(forget_loss.detach().cpu())
                forget_loss.backward()
                forget_loss_value_for_log = float(forget_loss.detach().cpu())
                forget_agreement_cosines.append(1.0)
                forget_agreement_positive_fracs.append(1.0)
                forget_agreement_weight_maxes.append(1.0)

            retain_accum_steps = int(args.retain_grad_accum_steps)
            retain_loss_values: list[float] = []
            retain_constraint_values: list[float] = []
            retain_budget_values: list[float] = []
            retain_active_values: list[float] = []
            retain_lambda_values: list[float] = []
            retain_cot_values: list[float] = []
            retain_answer_values: list[float] = []
            for _retain_micro_step in range(retain_accum_steps):
                retain_batch = next(retain_iter)
                retain_device_batch = {
                    key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                    for key, value in retain_batch.items()
                }
                retain_loss, micro_retain_metrics = normalized_block_topk_kl_loss(
                    model,
                    prefix,
                    retain_device_batch,
                    chunk_tokens=int(args.chunk_tokens),
                    kl_topk=int(args.kl_topk),
                )
                micro_mcq_loss_value = 0.0
                micro_mcq_agree_value = 0.0
                if retain_mcq_iter is not None and float(args.lambda_retain_mcq) > 0.0:
                    mcq_batch = next(retain_mcq_iter)
                    mcq_device_batch = {
                        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                        for key, value in mcq_batch.items()
                    }
                    mcq_loss, mcq_metrics = retain_mcq_logit_kl_loss(
                        model,
                        prefix,
                        mcq_device_batch,
                        temperature=float(args.retain_mcq_temperature),
                    )
                    retain_loss = retain_loss + float(args.lambda_retain_mcq) * mcq_loss
                    micro_retain_metrics["mcq_kl"] = float(mcq_metrics["mcq_kl"])
                    micro_retain_metrics["mcq_teacher_student_agree"] = float(mcq_metrics["teacher_student_agree"])
                    micro_mcq_loss_value = float(mcq_metrics["mcq_kl"])
                    micro_mcq_agree_value = float(mcq_metrics["teacher_student_agree"])
                if str(args.loss_mode) == "retain_budgeted_forgetce":
                    constraint_loss = torch.clamp(
                        retain_loss - retain_loss.new_tensor(float(args.retain_budget_epsilon)),
                        min=0.0,
                    )
                    if float(lambda_retain_dual) > 0.0:
                        (float(lambda_retain_dual) * constraint_loss / retain_accum_steps).backward()
                    micro_retain_loss_value = float(retain_loss.detach().cpu())
                    micro_budget_excess_value = micro_retain_loss_value - float(args.retain_budget_epsilon)
                    lambda_retain_dual = max(
                        0.0,
                        min(
                            float(args.lambda_retain_max),
                            float(lambda_retain_dual) + float(args.lambda_retain_lr) * micro_budget_excess_value,
                        ),
                    )
                    micro_constraint_loss_value = float(constraint_loss.detach().cpu())
                    micro_constraint_active_value = 1.0 if micro_budget_excess_value > 0.0 else 0.0
                else:
                    constraint_loss = retain_loss
                    if float(args.lambda_retain) > 0.0:
                        (float(args.lambda_retain) * constraint_loss / retain_accum_steps).backward()
                    micro_retain_loss_value = float(retain_loss.detach().cpu())
                    micro_budget_excess_value = 0.0
                    micro_constraint_loss_value = micro_retain_loss_value
                    micro_constraint_active_value = 1.0
                    lambda_retain_dual = float(args.lambda_retain)

                retain_loss_values.append(micro_retain_loss_value)
                retain_constraint_values.append(micro_constraint_loss_value)
                retain_budget_values.append(micro_budget_excess_value)
                retain_active_values.append(micro_constraint_active_value)
                retain_lambda_values.append(float(lambda_retain_dual))
                retain_cot_values.append(float(micro_retain_metrics["cot_kl"]))
                retain_answer_values.append(float(micro_retain_metrics["answer_kl"]))
                retain_mcq_kls.append(float(micro_mcq_loss_value))
                retain_mcq_agrees.append(float(micro_mcq_agree_value))

            anchor_metric = prefix_anchor_metrics(prefix, prefix_anchor)
            if prefix_anchor is not None and float(args.lambda_prefix_anchor) > 0.0:
                anchor_target = prefix_anchor.to(device=prefix.embedding.device, dtype=prefix.embedding.dtype)
                anchor_loss = (prefix.embedding - anchor_target).float().pow(2).mean()
                (float(args.lambda_prefix_anchor) * anchor_loss).backward()
                anchor_metric["prefix_anchor_loss"] = float(anchor_loss.detach().cpu())
            prefix_anchor_losses.append(float(anchor_metric["prefix_anchor_loss"]))
            prefix_delta_norms.append(float(anchor_metric["prefix_delta_norm"]))
            prefix_delta_norm_ratios.append(float(anchor_metric["prefix_delta_norm_ratio"]))

            retain_loss_value = float(sum(retain_loss_values) / len(retain_loss_values))
            constraint_loss_value = float(sum(retain_constraint_values) / len(retain_constraint_values))
            budget_excess_value = float(sum(retain_budget_values) / len(retain_budget_values))
            constraint_active_value = float(sum(retain_active_values) / len(retain_active_values))
            retain_metrics = {
                "loss": retain_loss_value,
                "cot_kl": float(sum(retain_cot_values) / len(retain_cot_values)),
                "answer_kl": float(sum(retain_answer_values) / len(retain_answer_values)),
                "mcq_kl": float(sum(retain_mcq_kls) / max(1, len(retain_mcq_kls))),
                "mcq_teacher_student_agree": float(sum(retain_mcq_agrees) / max(1, len(retain_mcq_agrees))),
            }
            torch.nn.utils.clip_grad_norm_(prefix.parameters(), float(args.max_grad_norm))
            opt.step()
            project_prefix_delta(prefix, prefix_anchor, float(args.prefix_max_delta_norm_ratio))

            epoch_forget_losses.append(float(forget_loss_value_for_log))
            epoch_retain_losses.append(retain_loss_value)
            epoch_constraint_losses.append(constraint_loss_value)
            epoch_budget_excesses.append(budget_excess_value)
            epoch_constraint_active.append(constraint_active_value)
            epoch_lambda_duals.append(float(lambda_retain_dual))
            forget_cot_ces.append(float(forget_metrics.get("cot_ce", 0.0)))
            forget_answer_ces.append(float(forget_metrics.get("answer_ce", 0.0)))
            forget_raw_cot_uls.append(float(forget_raw_metrics.get("raw_cot_ul", 0.0)))
            forget_raw_uls.append(float(forget_raw_metrics["raw_answer_ul"]))
            forget_raw_cot_nlls.append(float(forget_raw_metrics.get("raw_cot_nll", 0.0)))
            forget_raw_nlls.append(float(forget_raw_metrics["raw_answer_nll"]))
            forget_raw_cot_probs.append(float(forget_raw_metrics.get("raw_cot_prob", 0.0)))
            forget_raw_probs.append(float(forget_raw_metrics["raw_answer_prob"]))
            dpo_margins.append(float(forget_metrics.get("dpo_margin", 0.0)))
            dpo_policy_margins.append(float(forget_metrics.get("dpo_policy_margin", 0.0)))
            dpo_ref_margins.append(float(forget_metrics.get("dpo_ref_margin", 0.0)))
            dpo_pref_accs.append(float(forget_metrics.get("preference_acc", 0.0)))
            dpo_chosen_scores.append(float(forget_metrics.get("chosen_score", 0.0)))
            dpo_rejected_scores.append(float(forget_metrics.get("rejected_score", 0.0)))
            dpo_chosen_cot_logps.append(float(forget_metrics.get("chosen_cot_logp", 0.0)))
            dpo_chosen_answer_logps.append(float(forget_metrics.get("chosen_answer_logp", 0.0)))
            dpo_rejected_cot_logps.append(float(forget_metrics.get("rejected_cot_logp", 0.0)))
            dpo_rejected_answer_logps.append(float(forget_metrics.get("rejected_answer_logp", 0.0)))
            retain_cot_kls.append(float(retain_metrics["cot_kl"]))
            retain_answer_kls.append(float(retain_metrics["answer_kl"]))
            global_step += 1
            if global_step % 50 == 0:
                if str(args.loss_mode) == "prefixdpo_retainkl":
                    log(
                        f"step={global_step} epoch={epoch} "
                        f"dpo_loss={forget_metrics['loss']:.6f} rel_margin={forget_metrics['dpo_margin']:.6f} "
                        f"policy_margin={forget_metrics['dpo_policy_margin']:.6f} "
                        f"ref_margin={forget_metrics['dpo_ref_margin']:.6f} "
                        f"pref_acc={forget_metrics['preference_acc']:.3f} "
                        f"chosen={forget_metrics['chosen_score']:.6f} rejected={forget_metrics['rejected_score']:.6f} "
                        f"retain_loss={retain_metrics['loss']:.6f} "
                        f"retain_cot_kl={retain_metrics['cot_kl']:.6f} retain_answer_kl={retain_metrics['answer_kl']:.6f}"
                    )
                else:
                    log(
                        f"step={global_step} epoch={epoch} "
                        f"forget_loss={forget_metrics['loss']:.6f} forget_cot_ce={forget_metrics['cot_ce']:.6f} "
                        f"forget_answer_ce={forget_metrics['answer_ce']:.6f} "
                        f"raw_cot_ul={forget_raw_metrics.get('raw_cot_ul', 0.0):.6f} "
                        f"raw_answer_ul={forget_raw_metrics['raw_answer_ul']:.6f} "
                        f"raw_cot_nll={forget_raw_metrics.get('raw_cot_nll', 0.0):.6f} "
                        f"raw_answer_nll={forget_raw_metrics['raw_answer_nll']:.6f} "
                        f"retain_loss={retain_metrics['loss']:.6f} "
                        f"retain_cot_kl={retain_metrics['cot_kl']:.6f} retain_answer_kl={retain_metrics['answer_kl']:.6f} "
                        f"retain_mcq_kl={retain_metrics['mcq_kl']:.6f} "
                        f"retain_budget_excess={budget_excess_value:.6f} constraint_loss={constraint_loss_value:.6f} "
                        f"lambda_retain_dual={float(lambda_retain_dual):.6f} "
                        f"prefix_anchor_loss={anchor_metric['prefix_anchor_loss']:.6f} "
                        f"prefix_delta_ratio={anchor_metric['prefix_delta_norm_ratio']:.6f} "
                        f"forget_grad_cos={forget_agreement_cosines[-1]:.6f}"
                    )

        forget_loss_mean = float(sum(epoch_forget_losses) / max(1, len(epoch_forget_losses)))
        retain_loss_mean = float(sum(epoch_retain_losses) / max(1, len(epoch_retain_losses)))
        constraint_loss_mean = float(sum(epoch_constraint_losses) / max(1, len(epoch_constraint_losses)))
        budget_excess_mean = float(sum(epoch_budget_excesses) / max(1, len(epoch_budget_excesses)))
        constraint_active_mean = float(sum(epoch_constraint_active) / max(1, len(epoch_constraint_active)))
        lambda_retain_dual_mean = float(sum(epoch_lambda_duals) / max(1, len(epoch_lambda_duals)))
        final_anchor_metric = prefix_anchor_metrics(prefix, prefix_anchor)
        prefix_anchor_loss_mean = float(sum(prefix_anchor_losses) / max(1, len(prefix_anchor_losses)))
        prefix_delta_norm_mean = float(sum(prefix_delta_norms) / max(1, len(prefix_delta_norms)))
        prefix_delta_norm_ratio_mean = float(sum(prefix_delta_norm_ratios) / max(1, len(prefix_delta_norm_ratios)))
        row = {
            "epoch": epoch,
            "step": global_step,
            "method": str(args.method),
            "loss_mode": str(args.loss_mode),
            "prefix_init": str(args.prefix_init),
            "prefix_text": str(args.prefix_text),
            "lambda_prefix_anchor": float(args.lambda_prefix_anchor),
            "prefix_delta_init_std": float(args.prefix_delta_init_std),
            "prefix_max_delta_norm_ratio": float(args.prefix_max_delta_norm_ratio),
            "prefix_anchor_loss": float(final_anchor_metric["prefix_anchor_loss"]),
            "prefix_anchor_loss_mean": prefix_anchor_loss_mean,
            "prefix_delta_norm": float(final_anchor_metric["prefix_delta_norm"]),
            "prefix_delta_norm_mean": prefix_delta_norm_mean,
            "prefix_anchor_norm": float(final_anchor_metric["prefix_anchor_norm"]),
            "prefix_delta_norm_ratio": float(final_anchor_metric["prefix_delta_norm_ratio"]),
            "prefix_delta_norm_ratio_mean": prefix_delta_norm_ratio_mean,
            "forget_gradient_mode": str(args.forget_gradient_mode),
            "forget_agreement_microbatches": int(args.forget_agreement_microbatches),
            "forget_agreement_cosine": float(sum(forget_agreement_cosines) / max(1, len(forget_agreement_cosines))),
            "forget_agreement_positive_frac": float(
                sum(forget_agreement_positive_fracs) / max(1, len(forget_agreement_positive_fracs))
            ),
            "forget_agreement_weight_max": float(
                sum(forget_agreement_weight_maxes) / max(1, len(forget_agreement_weight_maxes))
            ),
            "lambda_retain": float(args.lambda_retain),
            "lambda_retain_mcq": float(args.lambda_retain_mcq),
            "retain_mcq_temperature": float(args.retain_mcq_temperature),
            "retain_mcq_samples_per_epoch": int(args.retain_mcq_samples_per_epoch),
            "retain_grad_accum_steps": int(args.retain_grad_accum_steps),
            "retain_effective_batch_size": int(args.retain_batch_size) * int(args.retain_grad_accum_steps),
            "lambda_cot_ce": float(args.lambda_cot_ce),
            "lambda_answer_ce": float(args.lambda_answer_ce),
            "beta": float(args.beta),
            "lambda_cot_dpo": float(args.lambda_cot_dpo),
            "lambda_answer_dpo": float(args.lambda_answer_dpo),
            "lambda_erase": float(args.lambda_erase),
            "retain_budget_epsilon": float(args.retain_budget_epsilon),
            "lambda_retain_init": float(args.lambda_retain_init),
            "lambda_retain_max": float(args.lambda_retain_max),
            "lambda_retain_lr": float(args.lambda_retain_lr),
            "lambda_retain_dual": float(lambda_retain_dual),
            "lambda_retain_dual_mean": lambda_retain_dual_mean,
            "kl_topk": int(args.kl_topk),
            "save_epochs": list(args.save_epoch_set),
            "forget_loss": forget_loss_mean,
            "retain_loss": retain_loss_mean,
            "retain_constraint_loss": constraint_loss_mean,
            "retain_budget_excess": budget_excess_mean,
            "retain_constraint_active": constraint_active_mean,
            "total_loss": forget_loss_mean + lambda_retain_dual_mean * constraint_loss_mean,
            "forget_pool_n": len(forget_train_records),
            "forget_sample_n": int(forget_stats.get("sample_n", len(forget_epoch_records))),
            "forget_skipped": int(forget_stats.get("skipped", 0)),
            "dpo_pairs": int(dpo_stats.get("pairs", 0)),
            "dpo_skipped": int(dpo_stats.get("skipped", 0)),
            "dpo_margin": float(sum(dpo_margins) / max(1, len(dpo_margins))),
            "dpo_policy_margin": float(sum(dpo_policy_margins) / max(1, len(dpo_policy_margins))),
            "dpo_ref_margin": float(sum(dpo_ref_margins) / max(1, len(dpo_ref_margins))),
            "dpo_preference_acc": float(sum(dpo_pref_accs) / max(1, len(dpo_pref_accs))),
            "dpo_chosen_score": float(sum(dpo_chosen_scores) / max(1, len(dpo_chosen_scores))),
            "dpo_rejected_score": float(sum(dpo_rejected_scores) / max(1, len(dpo_rejected_scores))),
            "dpo_chosen_cot_logp": float(sum(dpo_chosen_cot_logps) / max(1, len(dpo_chosen_cot_logps))),
            "dpo_chosen_answer_logp": float(sum(dpo_chosen_answer_logps) / max(1, len(dpo_chosen_answer_logps))),
            "dpo_rejected_cot_logp": float(sum(dpo_rejected_cot_logps) / max(1, len(dpo_rejected_cot_logps))),
            "dpo_rejected_answer_logp": float(sum(dpo_rejected_answer_logps) / max(1, len(dpo_rejected_answer_logps))),
            "forget_cot_ce": float(sum(forget_cot_ces) / max(1, len(forget_cot_ces))),
            "forget_answer_ce": float(sum(forget_answer_ces) / max(1, len(forget_answer_ces))),
            "forget_raw_cot_ul": float(sum(forget_raw_cot_uls) / max(1, len(forget_raw_cot_uls))),
            "forget_raw_ul": float(sum(forget_raw_uls) / max(1, len(forget_raw_uls))),
            "forget_raw_cot_nll": float(sum(forget_raw_cot_nlls) / max(1, len(forget_raw_cot_nlls))),
            "forget_raw_nll": float(sum(forget_raw_nlls) / max(1, len(forget_raw_nlls))),
            "forget_raw_cot_prob": float(sum(forget_raw_cot_probs) / max(1, len(forget_raw_cot_probs))),
            "forget_raw_prob": float(sum(forget_raw_probs) / max(1, len(forget_raw_probs))),
            "retain_cot_kl": float(sum(retain_cot_kls) / max(1, len(retain_cot_kls))),
            "retain_answer_kl": float(sum(retain_answer_kls) / max(1, len(retain_answer_kls))),
            "retain_mcq_kl": float(sum(retain_mcq_kls) / max(1, len(retain_mcq_kls))),
            "retain_mcq_teacher_student_agree": float(sum(retain_mcq_agrees) / max(1, len(retain_mcq_agrees))),
            "forget_targets": len(forget_rows),
            "forget_raw_targets": len(forget_raw_rows),
            "retain_targets": len(retain_rows),
            "retain_mcq_targets": len(retain_mcq_epoch_samples),
            "retain_unique_seen": len(seen_retain_source_keys),
            "steps_per_epoch": int(args.steps_per_epoch),
            "training_schedule": (
                "prefix_dpo_retain_kl"
                if str(args.loss_mode) == "prefixdpo_retainkl"
                else (
                    "forget_ce_with_retain_kl_trust_region"
                    if str(args.loss_mode) == "retain_budgeted_forgetce"
                    else (
                        "agreement_forget_ce_retain_kl"
                        if str(args.forget_gradient_mode) == "agreement"
                        else "paired_forget_ce_retain_kl"
                    )
                )
            ),
        }
        row.update({f"retain_{key}": value for key, value in retain_stats.items() if key not in row})
        train_log.append(row)
        write_json(output_dir / "train_log.json", train_log)
        write_json(results_dir / "train_log.json", train_log)
        log("epoch " + json.dumps(row, ensure_ascii=False, sort_keys=True))

        if epoch in set(args.save_epoch_set):
            save_payload = {
                "prefix_state_dict": prefix.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "train_row": row,
                "prefix_anchor_meta": prefix_anchor_meta,
            }
            if prefix_anchor is not None:
                save_payload["prefix_anchor"] = prefix_anchor.detach().cpu()
            torch.save(save_payload, output_dir / f"prefix_epoch{epoch}.pt")

        should_eval = (
            int(args.eval_every) > 0
            and epoch >= int(args.eval_start_epoch)
            and (epoch % int(args.eval_every) == 0 or epoch == int(args.epochs))
        )
        if should_eval:
            if epoch not in set(args.save_epoch_set):
                save_payload = {
                    "prefix_state_dict": prefix.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "train_row": row,
                    "prefix_anchor_meta": prefix_anchor_meta,
                }
                if prefix_anchor is not None:
                    save_payload["prefix_anchor"] = prefix_anchor.detach().cpu()
                torch.save(save_payload, output_dir / f"prefix_epoch{epoch}.pt")
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

        should_score_eval = (
            score_eval_enabled
            and (
                epoch in set(args.score_epoch_set)
                or (
                    int(args.score_eval_every) > 0
                    and epoch >= int(args.score_eval_start_epoch)
                    and epoch % int(args.score_eval_every) == 0
                )
            )
        )
        if should_score_eval:
            if epoch not in set(args.save_epoch_set):
                save_payload = {
                    "prefix_state_dict": prefix.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "train_row": row,
                    "prefix_anchor_meta": prefix_anchor_meta,
                }
                if prefix_anchor is not None:
                    save_payload["prefix_anchor"] = prefix_anchor.detach().cpu()
                torch.save(save_payload, output_dir / f"prefix_epoch{epoch}.pt")
            prefix.eval()
            if bool(args.score_forget_only):
                run_forget_only_score_eval(
                    epoch=epoch,
                    model=model,
                    prefix=prefix,
                    tokenizer=tokenizer,
                    in_records=score_forget_in_records,
                    out_records=score_forget_out_records,
                    model_cfg=model_cfg,
                    args=args,
                    score_dir=score_dir,
                    judge_config=judge_config,
                )
            else:
                run_score_eval(
                    epoch=epoch,
                    model=model,
                    prefix=prefix,
                    tokenizer=tokenizer,
                    forget_records=score_forget_records,
                    retain_records=score_retain_records,
                    model_cfg=model_cfg,
                    args=args,
                    score_dir=score_dir,
                    judge_config=judge_config,
                )

        should_forget_mcq_eval = (
            int(args.score_forget_mcq_sample_n) > 0
            and (
                epoch in set(args.score_forget_mcq_epoch_set)
                or (
                    int(args.score_forget_mcq_every) > 0
                    and epoch >= int(args.score_forget_mcq_start_epoch)
                    and epoch % int(args.score_forget_mcq_every) == 0
                )
            )
        )
        if should_forget_mcq_eval:
            prefix.eval()
            score_forget_mcq_logprob(
                epoch=epoch,
                model=model,
                prefix=prefix,
                tokenizer=tokenizer,
                forget_records=all_forget_records,
                device=str(args.device),
                batch_size=int(args.generation_batch_size),
                sample_n=int(args.score_forget_mcq_sample_n),
                seed=int(args.score_forget_mcq_seed),
                output_dir=score_dir,
            )

    torch.save(
        {
            "prefix_state_dict": prefix.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "args": vars(args),
            "epoch": int(args.epochs),
            "prefix_anchor": prefix_anchor.detach().cpu() if prefix_anchor is not None else None,
            "prefix_anchor_meta": prefix_anchor_meta,
        },
        output_dir / "prefix_final.pt",
    )
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
