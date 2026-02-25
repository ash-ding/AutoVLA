# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoVLA is a Vision-Language-Action model for end-to-end autonomous driving (NeurIPS 2025). It wraps Qwen2.5-VL with action tokenization and supports both supervised fine-tuning (SFT) and GRPO-based reinforcement fine-tuning (RFT), with dual "fast" (trajectory-only) and "slow" (chain-of-thought + trajectory) inference modes.

## Environment Setup

```bash
# Primary environment
conda env create -f environment.yml
conda activate autovla
pip install -e . --no-warn-conflicts
bash install.sh

# NAVSIM evaluation submodule
cd navsim && pip install -e . --no-warn-conflicts && cd ..

# NuScenes preprocessing (separate env)
conda env create -f environment_nusc_preprocess.yml
conda activate nusc_preprocess
```

Required environment variables for NAVSIM evaluation:
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

## Common Commands

```bash
# SFT training
python tools/run_sft.py --config training/qwen2.5-vl-3B-mix-sft

# GRPO/RFT training
bash scripts/run_rft.sh

# Data preprocessing
bash scripts/run_nuplan_preprocessing.sh
bash scripts/run_waymo_e2e_preprocessing.sh
bash scripts/run_nuscenes_preprocessing.sh

# Download base model
bash scripts/download_qwen.sh

# NAVSIM evaluation
bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh

# NuScenes evaluation
python tools/eval/nusc_eval.py --config <config> --checkpoint <path>
```

## Architecture

### Training Pipeline
1. **Dataset preprocessing** → JSON metadata files + sensor blob files (camera images)
2. **Action tokenization** → 2048-token codebook built via K-means clustering on trajectory data (`tools/action_token/`)
3. **CoT annotation** (optional) → uses Qwen2.5-VL-72B to generate chain-of-thought reasoning (`dataset_utils/preprocessing/cot_annotation_model.py`)
4. **SFT** → supervised fine-tuning on (image, text prompt, action token) triples
5. **GRPO/RFT** → reinforcement fine-tuning with PDM reward signal
6. **Evaluation** → NAVSIM PDM scoring (nuPlan/Waymo) or nuScenes metrics

### Core Models (`models/`)
- **`autovla.py`**: Three model classes:
  - `AutoVLA` — base VLM wrapper around Qwen2.5-VL; handles tokenization, image processing, CoT vs. fast-mode generation
  - `SFTAutoVLA` — PyTorch Lightning module for supervised fine-tuning with FSDP
  - `GRPOAutoVLA` — PyTorch Lightning module for GRPO reinforcement learning; computes PDM rewards
- **`action_tokenizer.py`**: Maps continuous (x, y, heading) trajectory poses to discrete integer tokens (0–2047) and back, using a precomputed codebook cache
- **`models/utils/score.py`**: PDM reward functions used during GRPO training

### Dataset Utils (`dataset_utils/`)
- **`sft_dataset.py`**: `SFTDataset` — loads multiple JSON/sensor-blob datasets, builds VLM conversation format, optionally includes CoT
- **`rft_dataset.py`**: `RFTDataset` — RL training data with on-policy sampling
- **`preprocessing/nuplan_dataset.py`**, **`waymo_e2e_dataset.py`**, **`preprocessing/cot_prompts.py`**: Dataset-specific preprocessing and CoT prompt templates

### Configuration System (`config/`)
- Uses **Hydra/OmegaConf** YAML configs loaded via `--config` argument (relative to `config/` directory)
- `config/training/` — SFT and GRPO training configs
- `config/dataset/` — dataset path and structure configs
- `config/eval/` — evaluation configs
- Key config fields: `model.use_cot`, `model.pretrained_model_path`, `model.codebook_cache_path`, `training.batch_size`, `data.train.json_dataset_path[]`, `data.train.sensor_data_path[]`

### NAVSIM Integration (`navsim/`)
- Self-contained evaluation framework (submodule-like structure)
- **`navsim/navsim/agents/autovla_agent.py`** — `AutoVLAAgent` wraps the trained model for open-loop evaluation
- **`navsim/navsim/planning/`** — PDM scoring and simulation utilities
- Non-reactive open-loop simulation; evaluates PDM score (composite driving quality metric)

### Training Infrastructure
- **FSDP** (Fully Sharded Data Parallel) for multi-GPU training
- **bfloat16** mixed precision
- **Gradient checkpointing** enabled
- **LoRA/PEFT** support for parameter-efficient fine-tuning
- **Ray** (`ray==2.44.1`) for distributed data processing

## Data Format

Training JSON files contain records with fields: camera image paths (front/front-left/front-right), ego trajectory poses, optional CoT reasoning text, and scene metadata. Sensor blobs store raw image bytes keyed by path. Action tokens are integer indices into the 2048-token codebook stored at `codebook_cache/`.
