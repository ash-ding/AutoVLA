# CLAUDE.md

## Project Overview

AutoVLA is a Vision-Language-Action model for end-to-end autonomous driving (NeurIPS 2025). It wraps Qwen2.5-VL with action tokenization (2048-token codebook) and supports SFT and GRPO reinforcement fine-tuning, with dual "fast" (trajectory-only) and "slow" (chain-of-thought + trajectory) inference modes.

**Current research direction**: Embedding human-understandable symbolic rules into the VLM decision process — replacing free-text CoT with structured, parseable symbolic rule chains, optimized via GRPO with a rule complexity penalty. See `docs/symbolic_rules_plan.md` for the full roadmap.

## Environment

Environment is already set up. Every session:
```bash
conda activate autovla
source .envrc  # sets NUPLAN_MAPS_ROOT, NAVSIM paths, NCCL_P2P_DISABLE=1, API keys
```

- Conda env: `autovla` (Python 3.10) at `/export/scratch_large/ding/.conda/envs/autovla`
- Key packages: vllm==0.8.5.post1, torch==2.6.0, transformers==4.57.6, pytorch-lightning==2.6.1
- Machine: L40S GPUs — requires `NCCL_P2P_DISABLE=1` (already in `.envrc`)

### Known Issues
- pytorch-lightning < 2.6 monkey-patches `torch.compile`, breaking vLLM — keep PL >= 2.6.1
- vLLM needs `enforce_eager=True` and `disable_custom_all_reduce=True` on this machine
- Import order matters: vLLM must be imported/initialized BEFORE pytorch_lightning and nuplan/navsim

## Current Progress (nuPlan mini split prototype)

Working on the **navmini** split (396 scenes) as a fast iteration prototype before scaling up.

### What's Done
1. **Data preprocessing** — 4 variants of 396 scenes each at `/export/scratch_large/ding/navsim_workspace/preprocessed/`:
   - `mini_nocot/` — no-CoT (action tokens only)
   - `mini_cot_Qwen2.5-VL-3B-Instruct/` — CoT from Qwen 3B
   - `mini_cot_Qwen2.5-VL-72B-Instruct-AWQ/` — CoT from Qwen 72B
   - `mini_cot_gpt-4o/` — CoT from GPT-4o (**primary training data**)

2. **Metric cache** — 396 pkl files at `/export/scratch_large/ding/navsim_workspace/navmini_metric_cache/` (required for GRPO reward computation)

3. **SFT training** — checkpoint at `runs/sft/2026-02-25_17-08-08/epoch=0-loss=2.2746.ckpt` (CoT-enabled, trained on GPT-4o CoT data)

4. **GRPO config ready** — `config/training/qwen2.5-vl-3B-mini-grpo-cot.yaml` wired up with SFT checkpoint, metric cache, and LoRA (r=8)

### What's Next
- GRPO/RFT training on navmini
- NAVSIM evaluation of SFT checkpoint
- Symbolic rules integration (Phase 1: DSL + parser — see `docs/symbolic_rules_plan.md`)

## Key Data Paths

| What | Path |
|------|------|
| Sensor blobs (mini) | `/export/scratch_large/ding/navsim_workspace/dataset/sensor_blobs/mini/` |
| Preprocessed JSONs | `/export/scratch_large/ding/navsim_workspace/preprocessed/mini_cot_gpt-4o/` |
| Metric cache (mini) | `/export/scratch_large/ding/navsim_workspace/navmini_metric_cache/` |
| Maps | `/export/scratch_large/ding/navsim_workspace/dataset/maps/nuplan-maps-v1.0/` |
| Qwen 3B weights | `model_weights/Qwen2.5-VL-3B-Instruct/` |
| Qwen 72B-AWQ weights | `model_weights/Qwen2.5-VL-72B-Instruct-AWQ/` |
| Codebook | `codebook_cache/agent_vocab.pkl` |
| SFT checkpoint | `runs/sft/2026-02-25_17-08-08/epoch=0-loss=2.2746.ckpt` |

## Common Commands

```bash
# SFT training (mini)
python tools/run_sft.py --config training/qwen2.5-vl-3B-mini-sft

# GRPO/RFT training (mini)
python tools/run_rft.py --config training/qwen2.5-vl-3B-mini-grpo-cot

# CoT annotation (vLLM or OpenAI backend)
python tools/preprocessing/cot_sample_generation.py --config dataset/openai-nuplan-mini --backend openai
python tools/preprocessing/cot_sample_generation.py --config dataset/qwen2.5-vl-72B-nuplan-mini --backend vllm

# No-CoT preprocessing
python tools/preprocessing/cot_sample_generation.py --config dataset/nocot_nuplan-mini --backend vllm

# Metric cache generation
bash scripts/run_navmini_metric_caching.sh

# Inspect metric cache
python tools/inspect_metric_cache.py --token <scene_token>

# NAVSIM evaluation
bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh
```

## Architecture (condensed)

### Core Models (`models/`)
- **`autovla.py`**: `AutoVLA` (base VLM wrapper), `SFTAutoVLA` (SFT with FSDP), `GRPOAutoVLA` (GRPO with PDM reward + KL regularization)
- **`action_tokenizer.py`**: codebook shape `(2048, 6, 4, 2)` — maps trajectory poses to discrete tokens and back via corner-point rollout
- **`models/utils/score.py`**: `PDM_Reward` — loads metric cache, runs PDM simulation and scoring

### Data Pipeline
- **`dataset_utils/sft_dataset.py`**: `SFTDataset` + `DataCollator` — builds VLM conversations with `<think>` (CoT) + `<answer>` (action tokens)
- **`dataset_utils/rft_dataset.py`**: `RFTDataset` — minimal loader; prompt construction happens in `GRPOAutoVLA.generate_sample()`
- **`dataset_utils/preprocessing/`**: CoT annotation backends (`vllm_cot_annotation_model.py`, `openai_cot_annotation_model.py`), prompt templates (`cot_prompts.py`)

### GRPO Training Flow
1. `generate_sample()` → VLM generates completion with action tokens
2. `reward_function()` → PDM score [0,10] minus optional CoT penalty
3. `all_gather(reward)` across GPUs (same scene, different samples) → group advantage
4. Policy gradient loss with KL regularization against frozen reference model

### Config System
- Hydra/OmegaConf YAML configs via `--config` (relative to `config/`)
- `config/training/` — SFT and GRPO configs
- `config/dataset/` — preprocessing configs (annotation backend, model, data paths)

### NAVSIM Evaluation (`navsim/`)
- `navsim/navsim/agents/autovla_agent.py` — wraps model for open-loop evaluation
- `navsim/navsim/evaluate/pdm_score.py` — ego-frame → UTM transform, LQR simulation, multi-metric scoring
- PDM score = (no_collision * drivable_area) * weighted_avg(progress, TTC, comfort) / 12

## Documentation

- `docs/symbolic_rules_plan.md` — Neuro-symbolic integration roadmap (6 phases)
- `docs/metric_cache_fields.md` — MetricCache data structure reference
