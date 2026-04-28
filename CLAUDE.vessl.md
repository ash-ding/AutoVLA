# CLAUDE.md

## Project Overview

AutoVLA is a Vision-Language-Action model for end-to-end autonomous driving (NeurIPS 2025). It wraps Qwen2.5-VL with action tokenization (2048-token codebook) and supports SFT and GRPO reinforcement fine-tuning, with dual "fast" (trajectory-only) and "slow" (chain-of-thought + trajectory) inference modes.

## Environment

Every session:
```bash
conda activate autovla
source .envrc  # sets NUPLAN_MAPS_ROOT, NAVSIM paths, NCCL_P2P_DISABLE=1
```

- Conda env: `autovla` (Python 3.10) at `/root/install/anaconda3/envs/autovla`
- Key packages (per `requirements.txt`): torch==2.10.0 (+cu128), transformers==4.57.6, pytorch-lightning==2.6.1, vllm==0.17.1, flashinfer-python==0.6.4. `flash-attn` is **not** installed — see Known Issues.
- Machine: 1× NVIDIA L40S (46GB VRAM). Driver reports CUDA 13.0; torch 2.10.0 ships its own CUDA 12.8 runtime (`nvidia-cuda-runtime-cu12-12.8.x`) — driver 580.x is backward-compatible with that.
- `NCCL_P2P_DISABLE=1` is kept in `.envrc` for forward-compat with multi-GPU L40S setups. On this 1-GPU box it's a no-op (no inter-GPU P2P traffic).
- Disk: `/root` is ~91GB free (on a CephFS mount, persistent); the env itself lives at `/root/install/anaconda3/envs/autovla`

### Post-install numpy pin
Installing `navsim` in editable mode downgrades `numpy`/`scipy`/`scikit-learn` because `navsim/requirements.txt` pins `numpy==1.23.4`, `scikit-learn==1.2.2`, etc. The main repo pinned `numpy==2.2.6`. End state after the full install sequence is the **navsim-compatible** set (numpy 1.23, scipy 1.13, scikit-learn 1.2, setuptools 65.5). This is the upstream-intended order — don't try to "fix" it back to numpy 2.x or nuplan/navsim simulation code will break.

### Known Issues
- pytorch-lightning < 2.6 monkey-patches `torch.compile`, breaking vLLM — keep PL >= 2.6.1
- vLLM needs `enforce_eager=True` and `disable_custom_all_reduce=True` on this machine
- Import order matters: vLLM must be imported/initialized BEFORE pytorch_lightning and nuplan/navsim
- **Flash-attn: intentionally not installed.** The repo has zero `flash_attn` imports; `models/autovla.py` loads Qwen2.5-VL without `attn_implementation`, so transformers 4.57.6 defaults to SDPA (built into torch) and falls back to eager. vLLM bundles its own pre-compiled flash-attn (`site-packages/vllm/vllm_flash_attn/*.so`) for its internal use — nothing else needs it. `flash-attn==2.7.4.post1` has no prebuilt wheel for torch 2.10 + cu128, so pip falls back to source build; nvcc then OOMs and drops the SSH session. **Do not retry `pip install flash-attn` on this box** — see the commented-out line in `install.sh`. If flash-attn speedups are ever needed, install a newer flash-attn version (>= 2.8/2.9) that publishes matching wheels and explicitly pass `attn_implementation="flash_attention_2"` at `from_pretrained`.

### Single-GPU Caveats
The upstream repo was developed on a multi-GPU node. On this 1×L40S box:
- 72B-AWQ CoT annotation likely won't fit — prefer the 3B model or an API backend (OpenAI / Anthropic)
- Batch sizes and FSDP settings in existing configs may need reduction
- GRPO group sampling that relies on `all_gather` across ranks collapses to a single rank — reward variance drops, which can make advantage estimation noisy

## Current State

Python env is fully set up; data/weights/checkpoints are not:
- **Env**: conda `autovla` ready. torch 2.10.0+cu128, transformers 4.57.6, PL 2.6.1, vllm 0.17.1, flashinfer 0.6.4, autoawq 0.2.8, nuplan-devkit 1.2.0 all installed. numpy 1.23.4 / scipy 1.13.1 / scikit-learn 1.2.2 / setuptools 65.5.1 (navsim-compatible end-state). `AutoVLA` and `navsim` installed editable. Smoke imports pass (`from models.autovla import AutoVLA`).
- **flash-attn**: NOT installed, intentionally — see Known Issues.
- **Codebook**: `codebook_cache/agent_vocab.pkl` present (1.18 MB). ✅
- **No dataset downloaded** (nuPlan / mini / trainval / maps)
- **No Qwen weights downloaded** — `model_weights/` directory does not exist
- **No preprocessed data** — no CoT annotations, no metric cache
- **No SFT / GRPO checkpoints** — `runs/` directory does not exist

### Resume Checklist (remaining work)
1. Download nuPlan maps + data splits (see `navsim/download/*.sh`; placeholders written to `$NAVSIM_WORKSPACE`)
2. Download Qwen2.5-VL weights: `bash scripts/download_qwen.sh` (3B) and optionally `bash scripts/download_qwen2.5_72b_awq.sh` (72B-AWQ — may not fit on 1×L40S)
3. Preprocess a split (mini recommended for first run): `bash scripts/run_nuplan_preprocessing.sh`
4. Generate metric cache: `bash scripts/run_navmini_metric_caching.sh`
5. After preprocessing, update YAML configs in `config/` — they currently hold hard-coded `/export/scratch_large/ding/...` paths from the upstream author's machine and must be rewritten before training

### Config Files Needing Path Rewrites
These configs contain `/export/scratch_large/ding/...` paths that won't resolve locally:
- `config/training/qwen2.5-vl-3B-mini-sft.yaml`
- `config/training/qwen2.5-vl-3B-mini-grpo-cot.yaml`
- `config/training/qwen2.5-vl-3B-mix-sft.yaml`
- `config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml`

Rewrite `pretrained_model_path`, `json_dataset_path`, `sensor_data_path`, and any checkpoint resume paths before use.

## Key Data Paths (target layout — see `.envrc`)

`$NAVSIM_WORKSPACE = $HOME/data/navsim_workspace` by default.

| What | Path |
|------|------|
| Sensor blobs | `$NAVSIM_WORKSPACE/dataset/sensor_blobs/<split>/` |
| Preprocessed JSONs | `$NAVSIM_WORKSPACE/preprocessed/<variant>/` |
| Metric cache | `$NAVSIM_WORKSPACE/<split>_metric_cache/` |
| Maps | `$NAVSIM_WORKSPACE/dataset/maps/nuplan-maps-v1.0/` |
| Qwen weights | `model_weights/Qwen2.5-VL-3B-Instruct/` (create via `scripts/download_qwen.sh`) |
| Codebook | `codebook_cache/agent_vocab.pkl` ✅ already present |

## Common Commands

```bash
# SFT training
python tools/run_sft.py --config training/qwen2.5-vl-3B-mini-sft

# GRPO/RFT training
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
