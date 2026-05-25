# GUARD: Guided Unlearning via Answer-Reasoning Distillation

Code for GUARD: Guided Unlearning via Answer-Reasoning Distillation, including natural trajectory rewriting, Guided Trajectory Alignment (GTA), and Answer-Reasoning Distillation (ARD) for long-reasoning-model unlearning.

## 1. Environment

```bash
cd GUARD
conda create -n guard python=3.10 -y
conda activate guard
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Install a PyTorch build matching the local CUDA runtime. If `flash_attention_2` is enabled in `configs/model_config.yaml`, install the compatible flash-attention package in the same environment.

## 2. Models

```bash
mkdir -p models

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local-dir models/deepseek-r1-distill-llama-8b

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --local-dir models/deepseek-r1-distill-qwen-14b
```

Set `model_path` in the training config to one local model directory, for example:

```yaml
model_path: models/deepseek-r1-distill-llama-8b
model_family: deepseek-r1-style
```

## 3. Data Preparation

Download public source data into `raw_data/`:

```bash
python tools/download_public_data.py --dataset all --output_root raw_data
```

This prepares the public r-tofu, star, SQuAD, and MMLU sources used by the release scripts. No processed experiment data is included in the repository.

Prepare r-tofu. Source traces are generated from the base LRM so the forget split contains the model's original reasoning and answer trajectory:

```bash
python tools/generate_source_traces.py \
  --model_path models/deepseek-r1-distill-llama-8b \
  --input raw_data/rtofu/forget.json \
  --out data/rtofu/source_traces.jsonl \
  --batch_size 4 \
  --max_new_tokens 2048

python tools/prepare_rtofu.py \
  --forget data/rtofu/source_traces.jsonl \
  --retain raw_data/rtofu/retain.json \
  --out_dir data/rtofu \
  --task_id 1
```

Prepare star. SQuAD is used as the retain-training set; the script randomly samples 5,000 SQuAD train examples by default with seed `20260519`. MMLU is used for retain evaluation.

```bash
python tools/generate_source_traces.py \
  --model_path models/deepseek-r1-distill-llama-8b \
  --input raw_data/star/star1.jsonl \
  --out data/star/source_traces.jsonl \
  --batch_size 4 \
  --max_new_tokens 2048

python tools/prepare_star.py \
  --forget data/star/source_traces.jsonl \
  --out_dir data/star \
  --task_id 1

python tools/prepare_squad_wiki_retain.py \
  --cache raw_data/squad/train.json \
  --out data/star/retain_squad.json \
  --limit 5000
```

For star retain evaluation, use the MMLU file written by the downloader:

```yaml
score_retain_path: raw_data/mmlu/mmlu_retain.json
score_retain_mode: mcq_logprob
```

## 4. Trajectory Rewriting

Build rewrite requests from the base-model source traces:

```bash
python tools/build_rewrite_requests.py \
  --forget data/rtofu/forget.json \
  --task_id 1 \
  --out data/rtofu/rewrite_requests.jsonl

python tools/build_rewrite_requests.py \
  --forget data/star/forget.json \
  --task_id 1 \
  --out data/star/rewrite_requests.jsonl
```

Run the offline rewriter with local provider settings. The script calls a Responses-compatible API and writes `safe_cot` plus `safe_answer` for each request.

```bash
export GUARD_REWRITER_BASE_URL="<provider-base-url>"
export GUARD_REWRITER_API_KEY="<api-key>"
export GUARD_REWRITER_MODEL="<rewriter-model>"

python tools/run_gpt_rewriter.py \
  --requests data/rtofu/rewrite_requests.jsonl \
  --out data/rtofu/rewrite_outputs.jsonl

python tools/run_gpt_rewriter.py \
  --requests data/star/rewrite_requests.jsonl \
  --out data/star/rewrite_outputs.jsonl
```

Materialize GTA targets:

```bash
python tools/materialize_rewrite_targets.py \
  --rewrites data/rtofu/rewrite_outputs.jsonl \
  --cot_out data/rtofu/forget_safe_cot.jsonl \
  --answer_out data/rtofu/forget_safe_answer.jsonl

python tools/materialize_rewrite_targets.py \
  --rewrites data/star/rewrite_outputs.jsonl \
  --cot_out data/star/forget_safe_cot.jsonl \
  --answer_out data/star/forget_safe_answer.jsonl
```

## 5. Training

### GTA

```bash
cp configs/gta_template.yaml configs/gta_rtofu.yaml
```

For r-tofu, set:

```yaml
model_path: models/deepseek-r1-distill-llama-8b
forget_path: data/rtofu/forget.json
retain_path: data/rtofu/retain.json
idkcot_path: data/rtofu/forget_safe_cot.jsonl
idontknow_path: data/rtofu/forget_safe_answer.jsonl
run_name: guard_gta_rtofu
```

For star, set:

```yaml
model_path: models/deepseek-r1-distill-llama-8b
forget_path: data/star/forget.json
retain_path: data/star/retain_squad.json
idkcot_path: data/star/forget_safe_cot.jsonl
idontknow_path: data/star/forget_safe_answer.jsonl
score_forget_judge_mode: star_safety
score_retain_mode: mcq_logprob
score_retain_path: raw_data/mmlu/mmlu_retain.json
run_name: guard_gta_star
```

GTA uses equal block-normalized forget CE by default: `lambda_cot_ce=1.0`, `lambda_answer_ce=1.0`, and `enforce_equal_forget_block_weights=true`, which gives reasoning and boundary-answer blocks equal weight after normalization.

Run GTA:

```bash
bash scripts/run_gta.sh configs/gta_rtofu.yaml cuda:0
```

GTA scoring runs during training when `score_eval_every > 0`. To score a saved prefix without continuing training:

```bash
python -m guard.gta_train \
  --config configs/gta_rtofu.yaml \
  --score_only_epoch 20 \
  --score_only_checkpoint outputs/gta_runs/guard_gta_rtofu/prefix_epoch20.pt
```

For scoring with a generative judge, set local `score_judge_base_url`, `score_judge_model`, and `GUARD_JUDGE_API_KEY`. The template leaves these values blank.

### ARD

```bash
cp configs/ard_template.yaml configs/ard_rtofu.yaml
```

Set the selected GTA prefix:

```yaml
model_path: models/deepseek-r1-distill-llama-8b
prefix_path: outputs/gta_runs/guard_gta_rtofu/prefix_epochXX.pt
forget_path: data/rtofu/forget.json
retain_path: data/rtofu/retain.json
run_name: guard_ard_rtofu
```

ARD uses a 1:1 forget/retain schedule by default (`forget_updates_per_cycle=1`, `retain_updates_per_cycle=1`) and keeps `retain_teacher_uses_prefix=true`, so both streams are distilled from the optimized guidance-conditioned teacher.

Run ARD:

```bash
bash scripts/run_ard.sh configs/ard_rtofu.yaml cuda:0
```
