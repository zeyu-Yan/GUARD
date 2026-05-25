#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

GUARD_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("GUARD_PROJECT_ROOT", str(GUARD_ROOT)))
DEFAULT_MODEL = Path(os.getenv("GUARD_MODEL_PATH", str(PROJECT_ROOT / "models" / "base-lrm")))
DEFAULT_MODEL_CONFIG = Path(os.getenv("GUARD_MODEL_CONFIG", str(PROJECT_ROOT / "configs" / "model_config.yaml")))
DEFAULT_FORGET = PROJECT_ROOT / "data" / "forget.json"
DEFAULT_IDKCOT = PROJECT_ROOT / "data" / "forget_safe_cot.jsonl"
DEFAULT_IDONTKNOW = PROJECT_ROOT / "data" / "forget_safe_answer.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "prefix_runs"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "prefix_results"
DEFAULT_PREFIX_ROOT = PROJECT_ROOT / "outputs" / "prefix"
DEFAULT_RUN_NAME = "guard_prefix"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cleanup_generated_text(text: str) -> str:
    text = (text or "").replace("\r", "")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_completion(completion: str, model_cfg: dict[str, Any], _: str | None = None) -> tuple[str, str]:
    completion = cleanup_generated_text(completion)
    think_end = str(model_cfg.get("think_end_tag") or "</think>")
    answer_tag = str(model_cfg.get("answer_tag") or "")
    if think_end in completion:
        cot, answer = completion.split(think_end, 1)
        answer = answer.lstrip()
        if answer_tag and answer.startswith(answer_tag):
            answer = answer[len(answer_tag):]
        return cot.strip(), answer.strip()
    return completion.strip(), ""


def ngram_repetition_ratio(text: str, n: int = 3) -> float:
    tokens = re.findall(r"\S+", cleanup_generated_text(text))
    if n <= 0 or len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[idx:idx + n]) for idx in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return float(1.0 - unique / len(grams))


def answer_similarity_score(reference: str, candidate: str) -> float:
    ref = cleanup_generated_text(reference).lower()
    cand = cleanup_generated_text(candidate).lower()
    if not ref or not cand:
        return 0.0
    return float(SequenceMatcher(None, ref, cand).ratio())


def substring_present(needle: str, haystack: str) -> bool:
    needle = cleanup_generated_text(needle).lower()
    haystack = cleanup_generated_text(haystack).lower()
    return bool(needle and haystack and needle in haystack)


def extract_private_facts(answer: str, question: str) -> list[str]:
    source = cleanup_generated_text(f"{answer}\n{question}")
    candidates: list[str] = []
    clauses = [piece.strip(" \t\r\n,;:") for piece in re.split(r"[.\n;]+", answer or "") if piece.strip()]
    for clause in clauses:
        if len(clause.split()) >= 3:
            candidates.append(clause)
    for pattern in [
        r'"([^"]{3,120})"',
        r"'([^']{3,120})'",
        r"\b\d{1,4}(?:/\d{1,2}/\d{2,4})?\b",
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b",
    ]:
        for match in re.findall(pattern, source):
            if isinstance(match, tuple):
                match = next((item for item in match if item), "")
            match = cleanup_generated_text(str(match))
            if len(match) >= 3:
                candidates.append(match)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        norm = re.sub(r"\s+", " ", item).strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(item.strip())
    return deduped[:20]


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    name = (dtype_name or "").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def get_input_device(model) -> torch.device:
    return next(model.parameters()).device


def get_model_config(model_family: str, model_config_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"model config must be a mapping: {config_path}")
    if model_family not in payload:
        raise KeyError(f"unknown model family {model_family!r} in {config_path}")
    return payload[model_family]


def build_prompt(question: str, model_cfg: dict[str, Any]) -> str:
    question_start = str(model_cfg.get("question_start_tag") or "")
    question_end = str(model_cfg.get("question_end_tag") or "")
    think_start = str(model_cfg.get("think_start_tag") or "")
    return f"{question_start}{question}{question_end}{think_start}"


def normalize_prefix_position(prefix_position: str | None) -> str:
    value = (prefix_position or "prompt_end").strip().lower().replace("-", "_")
    aliases = {
        "promptend": "prompt_end",
        "prompt_end": "prompt_end",
        "prompt_suffix": "prompt_end",
        "suffix": "prompt_end",
        "end": "prompt_end",
        "question_end": "question_end",
        "after_question": "question_end",
        "after_user_question": "question_end",
        "before_assistant": "question_end",
        "sequence_start": "sequence_start",
        "prefix_start": "sequence_start",
        "start": "sequence_start",
        "front": "sequence_start",
    }
    if value not in aliases:
        raise ValueError(f"unsupported prefix_position={prefix_position!r}; use prompt_end, question_end, or sequence_start")
    return aliases[value]


def load_model_and_tokenizer(model_path: str | Path, torch_dtype: torch.dtype, device: str | torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    return model, tokenizer


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

IDK_PATTERNS = [re.compile(p, re.I) for p in [
    r"\bi don['’]?t know\b", r"\bi do not know\b", r"not enough", r"cannot determine", r"can't determine",
    r"unable to determine", r"would be guessing", r"without additional", r"insufficient", r"can't confidently", r"cannot confidently",
]]
BAD_PATTERNS = [re.compile(p, re.I) for p in [
    r"(?:</think>\s*){2,}", r"(?:\bi don't know\b\.?\s*){4,}", r"(?:\bi do not know\b\.?\s*){4,}",
]]


def normalize_match_text(text: str) -> str:
    text = cleanup_generated_text(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_refusal_answers(path: Path) -> list[str]:
    answers: list[str] = []
    seen: set[str] = set()
    for row in load_json_or_jsonl(path):
        text = cleanup_generated_text(str(row))
        norm = normalize_match_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        answers.append(text)
    return answers


def refusal_phrase_matches(needle: str, haystack: str) -> bool:
    needle_norm = normalize_match_text(needle)
    haystack_norm = normalize_match_text(haystack)
    return bool(needle_norm and haystack_norm and (needle_norm in haystack_norm or haystack_norm in needle_norm))


try:
    DEFAULT_REFUSAL_ANSWERS = load_refusal_answers(DEFAULT_IDONTKNOW)
except Exception:
    DEFAULT_REFUSAL_ANSWERS = []


def log(msg: str) -> None:
    print(f"[guard-prefix] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def load_json_or_jsonl(path: Path) -> list[Any]:
    text = path.read_text(encoding='utf-8').strip()
    try:
        val = json.loads(text)
        return val if isinstance(val, list) else [val]
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(line)
        return rows


def contains_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text or '') for p in patterns)


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text or '', add_special_tokens=False).get('input_ids', []))


def quantile_int(values: list[int], p: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    idx = int(round((len(sorted_values) - 1) * p))
    return int(sorted_values[idx])


def summarize_training_lengths(examples: list[dict[str, Any]], tokenizer, model_cfg: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ex in examples:
        prompt = build_prompt(ex['question'], model_cfg)
        closure = model_cfg['think_end_tag'] + model_cfg.get('answer_tag', '') + ex['target_answer']
        full = prompt + ex['target_cot'] + closure
        if tokenizer.eos_token:
            full += tokenizer.eos_token
        row = {
            'probe_id': ex['probe_id'],
            'source_idx': ex.get('source_idx'),
            'prompt_tokens': token_count(tokenizer, prompt),
            'target_cot_tokens': token_count(tokenizer, ex['target_cot']),
            'target_answer_tokens': token_count(tokenizer, ex['target_answer']),
            'full_tokens': len(tokenizer(full, add_special_tokens=True, truncation=False).get('input_ids', [])),
        }
        rows.append(row)

    def summary_for(key: str) -> dict[str, int]:
        values = [int(row[key]) for row in rows]
        return {
            'max': max(values) if values else 0,
            'p95': quantile_int(values, 0.95),
            'p99': quantile_int(values, 0.99),
        }

    return {
        'n': len(rows),
        'prompt_tokens': summary_for('prompt_tokens'),
        'target_cot_tokens': summary_for('target_cot_tokens'),
        'target_answer_tokens': summary_for('target_answer_tokens'),
        'full_tokens': summary_for('full_tokens'),
        'top_full_tokens': sorted(rows, key=lambda row: row['full_tokens'], reverse=True)[:10],
    }


def build_examples(
    forget_path: Path,
    idkcot_path: Path,
    idontknow_path: Path,
    task_id: str,
    limit: int = 0,
) -> list[dict[str, Any]]:
    forget_rows = load_json_or_jsonl(forget_path)
    idkcot_rows = load_json_or_jsonl(idkcot_path)
    idontknow_rows = [str(row).strip() for row in load_json_or_jsonl(idontknow_path)]
    filtered_forget = [(idx, row) for idx, row in enumerate(forget_rows) if str(row.get('task_id')) == str(task_id)]
    n = min(len(filtered_forget), len(idkcot_rows), limit or 10**9)
    if n == 0:
        raise ValueError(f"no examples found for task_id={task_id!r} from {forget_path}")
    dataset_name = forget_path.stem
    examples = []
    for idx in range(n):
        source_idx, forget_row = filtered_forget[idx]
        question = str(forget_row['question']).strip()
        source_answer = str(forget_row['answer']).strip()
        cot = str(idkcot_rows[idx]).strip().strip('"')
        target_answer = "I don't know."
        if idontknow_rows:
            target_answer = idontknow_rows[idx % len(idontknow_rows)]
        examples.append({
            'idx': idx,
            'source_idx': source_idx,
            'task_id': str(task_id),
            'probe_id': f'{dataset_name}_task{task_id}_{idx:03d}',
            'question': question,
            'source_answer': source_answer,
            'target_cot': cot,
            'target_answer': target_answer,
            'source_private_facts': extract_private_facts(source_answer, question),
        })
    return examples


class PrefixDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer,
        model_cfg: dict[str, Any],
        max_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.max_length = max_length
        self.rows = [self._encode(ex) for ex in examples]

    def _encode(self, ex: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(ex['question'], self.model_cfg)
        think_end_tag = str(self.model_cfg['think_end_tag'])
        answer_tag = str(self.model_cfg.get('answer_tag', '') or '')
        closure = think_end_tag + answer_tag + ex['target_answer']
        full_without_eos = prompt + ex['target_cot'] + closure
        full = full_without_eos
        if self.tokenizer.eos_token:
            full += self.tokenizer.eos_token
        enc_full = self.tokenizer(full, add_special_tokens=True, truncation=False)
        full_len = len(enc_full['input_ids'])
        if full_len > self.max_length:
            raise ValueError(
                f"training example would be truncated: probe_id={ex.get('probe_id')} "
                f"tokens={full_len} max_length={self.max_length}"
            )
        enc_prompt = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
        enc_prompt_cot = self.tokenizer(prompt + ex['target_cot'], add_special_tokens=True, truncation=False)
        enc_prompt_cot_end = self.tokenizer(prompt + ex['target_cot'] + think_end_tag, add_special_tokens=True, truncation=False)
        enc_prompt_cot_end_answer_tag = self.tokenizer(
            prompt + ex['target_cot'] + think_end_tag + answer_tag,
            add_special_tokens=True,
            truncation=False,
        )
        enc_full_without_eos = self.tokenizer(full_without_eos, add_special_tokens=True, truncation=False)
        input_ids = list(enc_full['input_ids'])
        attention_mask = list(enc_full['attention_mask'])
        labels = list(input_ids)
        loss_weights = [1.0] * len(input_ids)
        cot_mask = [0.0] * len(input_ids)
        answer_mask = [0.0] * len(input_ids)
        prompt_len = min(len(enc_prompt['input_ids']), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
            loss_weights[i] = 0.0
        cot_end = min(len(enc_prompt_cot['input_ids']), len(loss_weights))
        think_end_end = min(len(enc_prompt_cot_end['input_ids']), len(loss_weights))
        answer_start = min(len(enc_prompt_cot_end_answer_tag['input_ids']), len(loss_weights))
        no_eos_len = min(len(enc_full_without_eos['input_ids']), len(loss_weights))
        for i in range(prompt_len, cot_end):
            if labels[i] != -100:
                loss_weights[i] = 1.0
                cot_mask[i] = 1.0
        for i in range(cot_end, think_end_end):
            if labels[i] != -100:
                loss_weights[i] = 1.0
                answer_mask[i] = 1.0
        for i in range(think_end_end, answer_start):
            if labels[i] != -100:
                loss_weights[i] = 1.0
                answer_mask[i] = 1.0
        for i in range(answer_start, no_eos_len):
            if labels[i] != -100:
                loss_weights[i] = 1.0
                answer_mask[i] = 1.0
        for i in range(no_eos_len, len(loss_weights)):
            if labels[i] != -100:
                loss_weights[i] = 1.0
                answer_mask[i] = 1.0
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_weights': loss_weights,
            'cot_mask': cot_mask,
            'answer_mask': answer_mask,
            'prompt_len': prompt_len,
            'example': ex,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    loss_weights, cot_mask, answer_mask, prompt_lens, examples = [], [], [], [], []
    for row in batch:
        pad = max_len - len(row['input_ids'])
        input_ids.append(row['input_ids'] + [pad_id] * pad)
        attention_mask.append(row['attention_mask'] + [0] * pad)
        labels.append(row['labels'] + [-100] * pad)
        loss_weights.append(row['loss_weights'] + [0.0] * pad)
        cot_mask.append(row['cot_mask'] + [0.0] * pad)
        answer_mask.append(row['answer_mask'] + [0.0] * pad)
        prompt_lens.append(int(row['prompt_len']))
        examples.append(row['example'])
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'loss_weights': torch.tensor(loss_weights, dtype=torch.float),
        'cot_mask': torch.tensor(cot_mask, dtype=torch.float),
        'answer_mask': torch.tensor(answer_mask, dtype=torch.float),
        'prompt_lens': torch.tensor(prompt_lens, dtype=torch.long),
        'examples': examples,
    }


class SoftPrefix(nn.Module):
    def __init__(self, prefix_len: int, hidden_size: int, init_std: float = 0.02):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(prefix_len, hidden_size))
        nn.init.normal_(self.embedding, mean=0.0, std=init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)


def forward_loss(
    model,
    prefix: SoftPrefix,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    prompt_lens: torch.Tensor | None = None,
    prefix_position: str = "prompt_end",
    loss_mode: str = "block_ce",
    cot_mask: torch.Tensor | None = None,
    answer_mask: torch.Tensor | None = None,
    cot_loss_weight: float = 1.0,
    answer_loss_weight: float = 1.0,
) -> torch.Tensor:
    prefix_position = normalize_prefix_position(prefix_position)
    loss_mode = (loss_mode or "block_ce").strip().lower()
    if loss_mode not in {"block_ce", "global_ce"}:
        raise ValueError(f"unsupported loss_mode={loss_mode!r}; use block_ce or global_ce")
    embed_layer = model.get_input_embeddings()
    tok_embeds = embed_layer(input_ids)
    bsz = input_ids.shape[0]
    pref = prefix(bsz).to(tok_embeds.dtype)
    pref_mask = torch.ones((bsz, pref.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
    pref_labels = torch.full((bsz, pref.shape[1]), -100, dtype=labels.dtype, device=labels.device)
    pref_weights = torch.zeros((bsz, pref.shape[1]), dtype=loss_weights.dtype, device=loss_weights.device)
    pref_block_mask = torch.zeros((bsz, pref.shape[1]), dtype=loss_weights.dtype, device=loss_weights.device)
    if cot_mask is None:
        cot_mask = (loss_weights > 0).to(loss_weights.dtype)
    if answer_mask is None:
        answer_mask = torch.zeros_like(loss_weights)
    if prefix_position == "sequence_start":
        inputs_embeds = torch.cat([pref, tok_embeds], dim=1)
        attn = torch.cat([pref_mask, attention_mask], dim=1)
        full_labels = torch.cat([pref_labels, labels], dim=1)
        full_weights = torch.cat([pref_weights, loss_weights], dim=1)
        full_cot_mask = torch.cat([pref_block_mask, cot_mask], dim=1)
        full_answer_mask = torch.cat([pref_block_mask, answer_mask], dim=1)
    else:
        if prompt_lens is None:
            raise ValueError("prompt_lens is required when prefix_position='prompt_end'")
        prompt_lens = prompt_lens.to(device=input_ids.device)
        embed_rows = []
        attn_rows = []
        label_rows = []
        weight_rows = []
        cot_rows = []
        answer_rows = []
        seq_len = input_ids.shape[1]
        for row_idx in range(bsz):
            insert_at = int(prompt_lens[row_idx].item())
            if insert_at < 0 or insert_at > seq_len:
                raise ValueError(f"invalid prompt_len={insert_at}; seq_len={seq_len}")
            embed_rows.append(torch.cat([tok_embeds[row_idx, :insert_at], pref[row_idx], tok_embeds[row_idx, insert_at:]], dim=0))
            attn_rows.append(torch.cat([attention_mask[row_idx, :insert_at], pref_mask[row_idx], attention_mask[row_idx, insert_at:]], dim=0))
            label_rows.append(torch.cat([labels[row_idx, :insert_at], pref_labels[row_idx], labels[row_idx, insert_at:]], dim=0))
            weight_rows.append(torch.cat([loss_weights[row_idx, :insert_at], pref_weights[row_idx], loss_weights[row_idx, insert_at:]], dim=0))
            cot_rows.append(torch.cat([cot_mask[row_idx, :insert_at], pref_block_mask[row_idx], cot_mask[row_idx, insert_at:]], dim=0))
            answer_rows.append(torch.cat([answer_mask[row_idx, :insert_at], pref_block_mask[row_idx], answer_mask[row_idx, insert_at:]], dim=0))
        inputs_embeds = torch.stack(embed_rows, dim=0)
        attn = torch.stack(attn_rows, dim=0)
        full_labels = torch.stack(label_rows, dim=0)
        full_weights = torch.stack(weight_rows, dim=0)
        full_cot_mask = torch.stack(cot_rows, dim=0)
        full_answer_mask = torch.stack(answer_rows, dim=0)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
    logits = out.logits[:, :-1, :].contiguous()
    shift_labels = full_labels[:, 1:].contiguous()
    shift_cot_mask = full_cot_mask[:, 1:].contiguous().to(logits.dtype)
    shift_answer_mask = full_answer_mask[:, 1:].contiguous().to(logits.dtype)
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='none',
    ).view_as(shift_labels)
    valid = (shift_labels != -100).to(logits.dtype)
    if loss_mode == "global_ce":
        denom = valid.sum().clamp_min(1.0)
        return (token_loss * valid).sum() / denom
    if loss_mode == "block_ce":
        cot_valid = shift_cot_mask * valid
        answer_valid = shift_answer_mask * valid
        cot_weight = torch.as_tensor(float(cot_loss_weight), dtype=logits.dtype, device=logits.device)
        answer_weight = torch.as_tensor(float(answer_loss_weight), dtype=logits.dtype, device=logits.device)
        cot_present = (cot_valid.sum() > 0).to(logits.dtype) * cot_weight
        answer_present = (answer_valid.sum() > 0).to(logits.dtype) * answer_weight
        cot_loss = (token_loss * cot_valid).sum() / cot_valid.sum().clamp_min(1.0)
        answer_loss = (token_loss * answer_valid).sum() / answer_valid.sum().clamp_min(1.0)
        denom = (cot_present + answer_present).clamp_min(1e-8)
        return (cot_weight * cot_loss * (cot_valid.sum() > 0).to(logits.dtype) + answer_weight * answer_loss * (answer_valid.sum() > 0).to(logits.dtype)) / denom
    raise RuntimeError(f"unreachable loss_mode={loss_mode!r}")


def generate_batch(
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    prefix_position: str = "prompt_end",
) -> list[str]:
    tokenizer.padding_side = 'left'
    enc = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=True).to(device)
    if prefix is None:
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        suffix = out[:, enc['input_ids'].shape[1]:]
        return [cleanup_generated_text(x) for x in tokenizer.batch_decode(suffix, skip_special_tokens=True)]
    with torch.inference_mode():
        prefix_position = normalize_prefix_position(prefix_position)
        tok_embeds = model.get_input_embeddings()(enc['input_ids'])
        bsz = tok_embeds.shape[0]
        pref = prefix(bsz).to(device=device, dtype=tok_embeds.dtype)
        pref_mask = torch.ones((bsz, pref.shape[1]), dtype=enc['attention_mask'].dtype, device=device)
        if prefix_position == "sequence_start":
            inputs_embeds = torch.cat([pref, tok_embeds], dim=1)
            attn = torch.cat([pref_mask, enc['attention_mask']], dim=1)
        else:
            # prompt_end/question_end both append the soft prefix to the provided prompt.
            # question_end callers pass a prompt that stops after the user question.
            inputs_embeds = torch.cat([tok_embeds, pref], dim=1)
            attn = torch.cat([enc['attention_mask'], pref_mask], dim=1)
        dummy_ids = torch.full((bsz, inputs_embeds.shape[1]), tokenizer.pad_token_id, dtype=torch.long, device=device)
        out = model.generate(
            input_ids=dummy_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        suffix = out[:, dummy_ids.shape[1]:]
        return [cleanup_generated_text(x) for x in tokenizer.batch_decode(suffix, skip_special_tokens=True)]


def generate_one(
    model,
    prefix: SoftPrefix | None,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    prefix_position: str = "prompt_end",
) -> str:
    return generate_batch(model, prefix, tokenizer, [prompt], device, max_new_tokens, prefix_position)[0]


def evaluate_completion(
    row: dict[str, Any],
    completion: str,
    tokenizer,
    model_cfg: dict[str, Any],
    refusal_answers: list[str] | None = None,
) -> dict[str, Any]:
    cot, answer = parse_completion(completion, model_cfg, 'DefaultThink')
    response = cleanup_generated_text((cot + '\n' + answer).strip())
    facts = row.get('source_private_facts') or extract_private_facts(row['source_answer'], row['question'])
    fact_hits = sum(1 for fact in facts if fact and substring_present(fact, response))
    source_sim = answer_similarity_score(row['source_answer'], response)
    repetition = ngram_repetition_ratio(response, 3)
    empty_answer = not bool(answer.strip())
    has_end = '</think>' in completion
    refusal_pool = list(refusal_answers) if refusal_answers is not None else list(DEFAULT_REFUSAL_ANSWERS)
    target_answer = cleanup_generated_text(str(row.get('target_answer') or ''))
    if target_answer:
        refusal_pool.append(target_answer)
    gold_refusal_like = any(
        refusal_phrase_matches(candidate, answer or response)
        for candidate in refusal_pool
        if candidate
    )
    idk_like = contains_any(answer or response, IDK_PATTERNS) or gold_refusal_like
    bad = repetition > 0.28 or contains_any(completion, BAD_PATTERNS) or len(response) < 8
    concise_answer = 0 < token_count(tokenizer, answer) <= 32
    enough_cot = token_count(tokenizer, cot) >= 100
    clean = bool(idk_like and has_end and not empty_answer and not bad and fact_hits == 0 and source_sim < 0.25 and concise_answer and enough_cot)
    return {
        'cot': cot,
        'answer': answer,
        'response_text': response,
        'cot_tokens': token_count(tokenizer, cot),
        'answer_tokens': token_count(tokenizer, answer),
        'completion_tokens': token_count(tokenizer, completion),
        'has_think_end': int(has_end),
        'empty_answer': int(empty_answer),
        'idk_like': int(idk_like),
        'gold_refusal_like': int(gold_refusal_like),
        'source_private_fact_hits': int(fact_hits),
        'source_answer_similarity': float(source_sim),
        'ngram3_repetition': float(repetition),
        'bad_generation': int(bool(bad)),
        'concise_answer': int(bool(concise_answer)),
        'enough_cot': int(bool(enough_cot)),
        'clean_refusal': int(clean),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GUARD soft-prefix utility trainer.")
    p.add_argument('--config', default=None)
    p.add_argument('--stamp', default=None)
    p.add_argument('--model_path', default=str(DEFAULT_MODEL))
    p.add_argument('--model_config', default=str(DEFAULT_MODEL_CONFIG))
    p.add_argument('--model_family', default='deepseek-r1-style')
    p.add_argument('--forget_path', default=str(DEFAULT_FORGET))
    p.add_argument('--idkcot_path', default=str(DEFAULT_IDKCOT))
    p.add_argument('--idontknow_path', default=str(DEFAULT_IDONTKNOW))
    p.add_argument('--output_root', default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument('--results_root', default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument('--prefix_root', default=str(DEFAULT_PREFIX_ROOT))
    p.add_argument('--run_name', default=DEFAULT_RUN_NAME)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--results_dir', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', default='bf16', choices=['bf16','fp16','fp32'])
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--task_id', default='1')
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--prefix_len', type=int, default=5)
    p.add_argument('--prefix_position', default='prompt_end')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--max_new_tokens', type=int, default=384)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--generation_batch_size', type=int, default=2)
    p.add_argument('--loss_mode', default='block_ce', choices=['block_ce', 'global_ce'])
    p.add_argument('--cot_loss_weight', type=float, default=1.0)
    p.add_argument('--answer_loss_weight', type=float, default=1.0)
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config)
        config_payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(config_payload, dict):
            raise ValueError(f"config must be a mapping: {config_path}")
        p.set_defaults(**config_payload)
    return p.parse_args()


def resolve_run_dirs(args: argparse.Namespace) -> tuple[Path, Path, str]:
    stamp = str(args.stamp or time.strftime('%Y%m%d_%H%M%S'))
    run_name = str(getattr(args, 'run_name', None) or f'prefix_forget05_task{args.task_id}_v2_weighted')
    output_dir = Path(args.output_dir) if getattr(args, 'output_dir', None) else Path(args.output_root) / f'{stamp}_{run_name}'
    results_dir = Path(args.results_dir) if getattr(args, 'results_dir', None) else Path(args.results_root) / f'{stamp}_{run_name}'
    return output_dir, results_dir, stamp


def main() -> None:
    args = parse_args()
    args.prefix_position = normalize_prefix_position(args.prefix_position)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    artifact_dir, results_dir, stamp = resolve_run_dirs(args)
    args.output_dir = str(artifact_dir)
    args.results_dir = str(results_dir)
    args.run_name = str(getattr(args, 'run_name', None) or f'prefix_forget05_task{args.task_id}_v2_weighted')
    args.stamp = stamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / 'train_config.json', vars(args))
    model_cfg = get_model_config(args.model_family, args.model_config)
    examples = build_examples(Path(args.forget_path), Path(args.idkcot_path), Path(args.idontknow_path), args.task_id, args.limit)
    write_jsonl(artifact_dir / 'examples.jsonl', examples)
    log(
        f'loaded examples={len(examples)} prefix_len={args.prefix_len} '
        f'prefix_position={args.prefix_position} loss_mode={args.loss_mode} '
        f'artifact_dir={artifact_dir} results_dir={results_dir}'
    )

    model, tokenizer = load_model_and_tokenizer(args.model_path, get_torch_dtype(args.dtype), args.device)
    model.config.use_cache = False
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    length_stats = summarize_training_lengths(examples, tokenizer, model_cfg)
    write_json(artifact_dir / 'length_stats.json', length_stats)
    write_json(results_dir / 'length_stats.json', length_stats)
    max_full_tokens = int(length_stats['full_tokens']['max'])
    log(
        f"length_stats full_tokens_max={max_full_tokens} "
        f"full_tokens_p99={length_stats['full_tokens']['p99']} max_length={args.max_length}"
    )
    if max_full_tokens > args.max_length:
        top = length_stats['top_full_tokens'][0] if length_stats['top_full_tokens'] else {}
        raise ValueError(
            f"max_length={args.max_length} would truncate training data; "
            f"required_at_least={max_full_tokens}; top_example={top}"
        )

    hidden = model.get_input_embeddings().embedding_dim
    prefix = SoftPrefix(args.prefix_len, hidden).to(args.device)
    opt = torch.optim.AdamW(prefix.parameters(), lr=args.lr, weight_decay=0.0)
    ds = PrefixDataset(examples, tokenizer, model_cfg, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    train_log: list[dict[str, Any]] = []

    def run_validation(tag: str) -> dict[str, Any]:
        prefix.eval()
        rows = []
        prompts = [build_prompt(ex['question'], model_cfg) for ex in examples]
        total_batches = math.ceil(len(examples) / max(1, args.generation_batch_size))
        for start_idx in range(0, len(examples), args.generation_batch_size):
            batch_examples = examples[start_idx:start_idx + args.generation_batch_size]
            batch_prompts = prompts[start_idx:start_idx + args.generation_batch_size]
            comps = generate_batch(
                model,
                prefix,
                tokenizer,
                batch_prompts,
                args.device,
                args.max_new_tokens,
                args.prefix_position,
            )
            for ex, prompt, comp in zip(batch_examples, batch_prompts, comps):
                metrics = evaluate_completion(ex, comp, tokenizer, model_cfg)
                rows.append({**ex, 'tag': tag, 'prompt': prompt, 'completion': comp, **metrics})
            batch_no = start_idx // args.generation_batch_size + 1
            if batch_no == 1 or batch_no % 10 == 0 or batch_no == total_batches:
                log(f'validation {tag} generation_batch={batch_no}/{total_batches}')
        write_jsonl(results_dir / f'generations_{tag}.jsonl', rows)
        write_csv(results_dir / f'generations_{tag}.csv', rows)
        summary = {
            'tag': tag,
            'n': len(rows),
                'clean_refusal_mean': sum(r['clean_refusal'] for r in rows)/len(rows),
                'idk_like_mean': sum(r['idk_like'] for r in rows)/len(rows),
                'gold_refusal_like_mean': sum(r.get('gold_refusal_like', 0) for r in rows)/len(rows),
                'fact_hit_mean': sum(r['source_private_fact_hits'] for r in rows)/len(rows),
                'empty_answer_mean': sum(r['empty_answer'] for r in rows)/len(rows),
                'bad_generation_mean': sum(r['bad_generation'] for r in rows)/len(rows),
            'has_think_end_mean': sum(r['has_think_end'] for r in rows)/len(rows),
            'concise_answer_mean': sum(r['concise_answer'] for r in rows)/len(rows),
            'enough_cot_mean': sum(r['enough_cot'] for r in rows)/len(rows),
            'cot_tokens_mean': sum(r['cot_tokens'] for r in rows)/len(rows),
            'answer_tokens_mean': sum(r['answer_tokens'] for r in rows)/len(rows),
            'source_answer_similarity_mean': sum(r['source_answer_similarity'] for r in rows)/len(rows),
        }
        write_json(results_dir / f'summary_{tag}.json', summary)
        log('validation ' + json.dumps(summary, ensure_ascii=False, sort_keys=True))
        prefix.train()
        return summary

    # Baseline no-prefix for comparison.
    base_rows = []
    base_prompts = [build_prompt(ex['question'], model_cfg) for ex in examples]
    total_base_batches = math.ceil(len(examples) / max(1, args.generation_batch_size))
    for start_idx in range(0, len(examples), args.generation_batch_size):
        batch_examples = examples[start_idx:start_idx + args.generation_batch_size]
        batch_prompts = base_prompts[start_idx:start_idx + args.generation_batch_size]
        comps = generate_batch(model, None, tokenizer, batch_prompts, args.device, args.max_new_tokens)
        for ex, prompt, comp in zip(batch_examples, batch_prompts, comps):
            base_rows.append({**ex, 'tag': 'base', 'prompt': prompt, 'completion': comp, **evaluate_completion(ex, comp, tokenizer, model_cfg)})
        batch_no = start_idx // args.generation_batch_size + 1
        if batch_no == 1 or batch_no % 10 == 0 or batch_no == total_base_batches:
            log(f'baseline generation_batch={batch_no}/{total_base_batches}')
    write_jsonl(results_dir / 'generations_base.jsonl', base_rows)
    write_csv(results_dir / 'generations_base.csv', base_rows)
    base_summary = {
        'tag': 'base', 'n': len(base_rows),
        'clean_refusal_mean': sum(r['clean_refusal'] for r in base_rows)/len(base_rows),
        'idk_like_mean': sum(r['idk_like'] for r in base_rows)/len(base_rows),
        'gold_refusal_like_mean': sum(r.get('gold_refusal_like', 0) for r in base_rows)/len(base_rows),
        'fact_hit_mean': sum(r['source_private_fact_hits'] for r in base_rows)/len(base_rows),
        'empty_answer_mean': sum(r['empty_answer'] for r in base_rows)/len(base_rows),
        'bad_generation_mean': sum(r['bad_generation'] for r in base_rows)/len(base_rows),
        'has_think_end_mean': sum(r['has_think_end'] for r in base_rows)/len(base_rows),
        'concise_answer_mean': sum(r['concise_answer'] for r in base_rows)/len(base_rows),
        'enough_cot_mean': sum(r['enough_cot'] for r in base_rows)/len(base_rows),
        'cot_tokens_mean': sum(r['cot_tokens'] for r in base_rows)/len(base_rows),
        'answer_tokens_mean': sum(r['answer_tokens'] for r in base_rows)/len(base_rows),
        'source_answer_similarity_mean': sum(r['source_answer_similarity'] for r in base_rows)/len(base_rows),
    }
    write_json(results_dir / 'summary_base.json', base_summary)
    log('baseline ' + json.dumps(base_summary, ensure_ascii=False, sort_keys=True))

    global_step = 0
    prefix.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            loss_weights = batch['loss_weights'].to(args.device)
            cot_mask = batch['cot_mask'].to(args.device)
            answer_mask = batch['answer_mask'].to(args.device)
            prompt_lens = batch['prompt_lens'].to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = forward_loss(
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
                args.cot_loss_weight,
                args.answer_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prefix.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
        mean_loss = sum(losses) / max(1, len(losses))
        train_log.append({'epoch': epoch, 'step': global_step, 'loss': mean_loss})
        log(f'epoch={epoch} step={global_step} loss={mean_loss:.4f}')
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            torch.save({'prefix_state_dict': prefix.state_dict(), 'args': vars(args), 'epoch': epoch, 'loss': mean_loss}, artifact_dir / f'prefix_epoch{epoch}.pt')
            run_validation(f'epoch{epoch}')
    write_json(artifact_dir / 'train_log.json', train_log)
    write_json(results_dir / 'train_log.json', train_log)
    torch.save({'prefix_state_dict': prefix.state_dict(), 'args': vars(args), 'epoch': args.epochs}, artifact_dir / 'prefix_final.pt')
    log(f'done output={artifact_dir} results={results_dir}')


if __name__ == '__main__':
    main()
