# AutoVLA — Local Adaptation README

> **This README is the entry point for our locally adapted version of the project.**
> For the original paper README (project description, citation, contributors, training recipes), please read [`AutoVLA.README.md`](./AutoVLA.README.md).

---

## 1. Prerequisites

The commands below assume **you have already downloaded and unpacked every required dataset**. For dataset sources, download links, and the canonical directory structure please refer to the **"Devkit Setup → Dataset Downloading"** section of [`AutoVLA.README.md`](./AutoVLA.README.md).

If you are running on the **vessl platform**, dataset loading from `/backup` into `/data` is automated by the helpers under [`tools/vessl/`](./tools/vessl/) — see [`tools/vessl/README.md`](./tools/vessl/README.md) for the exact load commands.

The expected on-disk layout (paths in the scripts and YAML configs assume this):

```
/data/
├── nuPlan/
│   ├── navsim_logs/{trainval,test}/<log>.pkl
│   ├── sensor_blobs/{trainval,test}/<log>/<cam>/<jpg>
│   └── maps/                                      # required only for NAVSIM eval
├── nuScenes/{samples,maps,v1.0-trainval,...}
├── drivelm/
│   ├── v1_1_train_nus.json                        # train split: full Q+A
│   └── v1_1_val_nus_q_only.json                   # val split: Q only, no A
└── nuplan_nuscenes_train_mix_185k/
    └── scaling_185k_token_list.json               # token list + bucket_counts

<your model store>/
└── Qwen2.5-VL-{3B-Instruct, 72B-Instruct-AWQ}/
```

### Environment variables

The repo ships a template at [`.envrc.example`](./.envrc.example). Copy it to `.envrc` and adjust paths to match your machine:

```bash
cp .envrc.example .envrc
# then edit .envrc — typically only NUPLAN_MAPS_ROOT, OPENSCENE_DATA_ROOT,
# NAVSIM_EXP_ROOT, and NUSCENES_PATH need to change.
```

`.envrc` is `.gitignored`. The preprocessing scripts in §2 do **not** depend on these env vars (they hardcode absolute paths in the YAML configs), but NAVSIM eval and metric caching do.

### Recommended: use [`direnv`](https://direnv.net) to auto-load `.envrc`

Manually `source .envrc` every shell gets old fast. We recommend installing [direnv](https://direnv.net) so the right env loads automatically whenever you `cd` into the repo:

```bash
# Install (Linux example; see https://direnv.net/docs/installation.html for other OSes)
sudo apt install direnv     # or: brew install direnv  (macOS)

# Hook direnv into your shell — pick one matching your shell:
eval "$(direnv hook bash)"  >> ~/.bashrc       # bash
eval "$(direnv hook zsh)"   >> ~/.zshrc        # zsh

# In the repo, allow this .envrc once:
cd /path/to/AutoVLA
direnv allow
```

After that, every subsequent `cd /path/to/AutoVLA` automatically exports the variables, and leaving the directory unsets them.

### Other one-time setup

```bash
conda activate autovla
# direnv (or `source .envrc`) handles env vars from here on.
```

Symbolic CoT additionally requires `export OPENAI_API_KEY=...` (commented at the bottom of `.envrc.example` — un-comment and set if you plan to use it).

---

## 2. One-shot preprocessing: four shell scripts

These four scripts wrap the **current preprocessing pipeline**, covering nuPlan/nuScenes × NL CoT/Symbolic CoT. Each is directly `bash`-runnable; the env vars `DP_SIZE` (default `1`; raise on multi-GPU boxes) and `LOG_DIR` (default `$AUTOVLA_ROOT/logs`, set from `.envrc`) can be overridden inline.

```bash
# 1) nuPlan NL CoT + no-CoT + no-CoT test set
bash scripts/run_nuplan_preprocess_nl.sh

# 2) nuScenes NL CoT + no-CoT + no-CoT test set (DriveLM splits CoT vs. no-CoT)
bash scripts/run_nuscenes_preprocess_nl.sh

# 3) nuPlan Symbolic CoT (GPT-4o-mini + RLIB, on the nuplan_cot bucket)
bash scripts/run_nuplan_preprocess_symbolic.sh

# 4) nuScenes Symbolic CoT (same, on the nuscenes_cot bucket)
bash scripts/run_nuscenes_preprocess_symbolic.sh
```

What each script does, in one line:

| Script | Behavior | Outputs (under `/data/`) |
|---|---|---|
| `run_nuplan_preprocess_nl.sh` | Generate NL CoT with Qwen2.5-VL-72B-AWQ on the `nuplan_cot` bucket; generate no-CoT metadata on `nuplan_nocot`; generate no-CoT navtest set | `nuplan_nl_reasoning_samples_Qwen2.5_VL_72B_Instruct_AWQ_45600/`<br>`nuplan_action_only_samples_120682/`<br>`nuplan_test_samples_12146/` |
| `run_nuscenes_preprocess_nl.sh` | train split: DriveLM splits CoT vs. no-CoT into two flat sibling dirs; val split is treated as the test set (no DriveLM) | `nuscenes_nl_reasoning_samples_2895/`<br>`nuscenes_action_only_samples_16135/`<br>`nuscenes_test_samples_5569/` |
| `run_nuplan_preprocess_symbolic.sh` | Generate symbolic CoT via gpt-4o-mini + RLIB on the `nuplan_cot` bucket; **does not depend on NL CoT by default** | `nuplan_symbolic_reasoning_samples_gpt_4o_mini_45600/` |
| `run_nuscenes_preprocess_symbolic.sh` | Same, on the `nuscenes_cot` bucket | `nuscenes_symbolic_reasoning_samples_gpt_4o_mini_2895/` |

### These are example invocations — tune them for your setup

The four scripts hardcode the **scaling token list path** (185k), **model name/path**, **output directory names** (sample-count and model-slug suffixes), and **`dp_size`** that match our 185k mix + Qwen2.5-VL-72B-AWQ + gpt-4o-mini reference configuration. **Adjust them before running on your data**:

- To use 10k / 50k / 100k instead of 185k: edit `SCALING_TOKEN_LIST` at the top of each script
- To swap models: edit `pretrained_model_path` or `api_model` in the relevant `config/dataset/<name>.yaml`, then update the output directory suffix accordingly
- To change vLLM batch / TP / sequence length: edit `batch_size` / `max_num_seqs` / `max_model_len` / `tensor_parallel_size` in `config/dataset/qwen2.5-vl-72B-nuplan-trainval.yaml`
- To switch the symbolic CoT backend: edit `annotation_backend` / `api_model` in `symbolic-cot-gpt4o-mini-*.yaml`
- For DP fanout across multiple GPUs: `DP_SIZE=8 bash scripts/run_nuplan_preprocess_nl.sh`

---

## 3. Python entry points under the hood

The four shell scripts above ultimately call these four Python scripts in `tools/preprocessing/`. You can **bypass the shell scripts entirely and call the Python entry points directly** for finer control.

All four scripts share the same design: the dataset adapter class is dispatched by `dataset_name` (`nuplan` / `waymo` / `nuscenes`); token selection uses `--sample-ids-json` + optional `--sample-ids-key`; multi-GPU inference uses `--dp_size`; resume uses `--resume`; output is one `<token>.json` per token.

### 3.1 `tools/preprocessing/nl_cot_sample_generation.py` — NL CoT generation

**Purpose**: generate a free-form natural-language CoT (`cot_output` as a string) for each token, using a VLM either via local vLLM (Qwen2.5-VL) or via the OpenAI API.

| Flag | Type / default | Description |
|---|---|---|
| `--config <name>` | **required** | YAML config name, relative to `config/`. Example: `dataset/qwen2.5-vl-72B-nuplan-trainval` |
| `--output_dir <path>` | **required** | Output directory; one `<token>.json` per token |
| `--backend {vllm,openai}` | reads YAML | Override `annotation_backend` from the YAML |
| `--sample-ids-json <path>` | None | JSON file listing the tokens to process |
| `--sample-ids-key <bucket>` | None | When `--sample-ids-json` is a scaling file, pick `buckets[<bucket>]` |
| `--seed <int>` | `42` | Random seed |
| `--resume` | flag | Skip tokens already present in `--output_dir` |
| `--dp_size <int>` | `1` | Fork N data-parallel subprocesses |
| `--tp_size <int>` | None | Override `tensor_parallel_size` in the YAML (vllm only) |
| `--num_parts <int>` / `--sample_num <int>` | `1` / `1` | Manual sharding; mutually exclusive with `--dp_size > 1` |

**Examples**:

```bash
# (A) NL CoT on the 100k mix dataset's nuplan_cot bucket
python -m tools.preprocessing.nl_cot_sample_generation \
    --config dataset/qwen2.5-vl-72B-nuplan-trainval \
    --output_dir /data/demo_nl_cot \
    --sample-ids-json /data/nuplan_nuscenes_train_mix_100k/scaling_100k_token_list.json \
    --sample-ids-key nuplan_cot \
    --dp_size 1 \
    --resume

# (B) Full trainval (no --sample-ids-json: process every token in the dataset)
python -m tools.preprocessing.nl_cot_sample_generation \
    --config dataset/qwen2.5-vl-72B-nuplan-trainval \
    --output_dir /data/demo_full_nl_cot \
    --dp_size 1 --resume

# (C) Switch to OpenAI API backend (CLI overrides annotation_backend in the YAML)
python -m tools.preprocessing.nl_cot_sample_generation \
    --config dataset/openai-nuplan-mini \
    --backend openai \
    --output_dir /data/demo_openai_cot \
    --sample-ids-json /data/nuplan_nuscenes_train_mix_10k/scaling_10k_token_list.json \
    --sample-ids-key nuplan_cot
```

### 3.2 `tools/preprocessing/nocot_sample_generation.py` — model-agnostic no-CoT metadata

**Purpose**: serialize per-token scene metadata (camera paths, GT trajectory, ego state) to `<token>.json`. `cot_output` is always `[]`. **No VLM is involved**; backend selection is irrelevant.

| Flag | Type / default | Description |
|---|---|---|
| `--config <name>` | **required** | YAML config name, relative to `config/` |
| `--output_dir <path>` | **required** | Output directory |
| `--num_workers <int>` | `32` | DataLoader worker count |
| `--pre_generated_dir <path>` | None | Additional "already processed" directory; tokens there are also skipped |
| `--sample-ids-json <path>` | None | Same as in §3.1 |
| `--sample-ids-key <bucket>` | None | Same as in §3.1 |
| `--seed <int>` | `42` | Random seed |
| `--resume` | flag | Skip tokens already in `--output_dir` |

> There is no `--dp_size` / `--backend` — this is a pure I/O-bound single-process script.

**Examples**:

```bash
# (A) Filter by nuplan_nocot bucket + write to a specific dir
python -m tools.preprocessing.nocot_sample_generation \
    --config dataset/nocot_nuplan-trainval \
    --output_dir /data/demo_action_only \
    --sample-ids-json /data/nuplan_nuscenes_train_mix_185k/scaling_185k_token_list.json \
    --sample-ids-key nuplan_nocot \
    --resume

# (B) Full navtest split (136 logs, ~12k tokens)
python -m tools.preprocessing.nocot_sample_generation \
    --config dataset/nocot_nuplan-navtest \
    --output_dir /data/demo_navtest \
    --resume
```

### 3.3 `tools/preprocessing/nusc_sample_generation.py` — nuScenes extraction (CoT comes from DriveLM)

**Purpose**: extract per-token JSON from raw nuScenes + DriveLM annotations. Frames covered by DriveLM get a 5-element `cot_output` (fov / perception / move_intent / prediction / planning); all others get an empty `cot_output=[]`. **Does not use a Hydra YAML config**; everything is passed via CLI.

| Flag | Type / default | Description |
|---|---|---|
| `--nuscenes_path <path>` | **required** | nuScenes root (contains `samples/`, `v1.0-trainval/`, ...) |
| `--output_dir <path>` | see below | Primary output dir. Becomes optional when both `--cot_output_dir` and `--nocot_output_dir` are set |
| `--split {train,val}` | `train` | Which split (the script has no `test` option) |
| `--version <str>` | `v1.0-trainval` | nuScenes-devkit version label |
| `--drivelm_path <path>` | None | DriveLM annotations JSON. **Only `v1_1_train_nus.json` on the train split carries answers** — the val file is "questions only" |
| `--cot_output_dir <path>` | None | Override: DriveLM-derived CoT JSONs go here directly (bypassing `<output_dir>/nl_reasoning_samples/`) |
| `--nocot_output_dir <path>` | None | Override: no-CoT JSONs go here directly (bypassing `<output_dir>/action_only_samples/`) |

**Output-dir resolution rules** (important):

- `--drivelm_path` omitted → every sample is no-CoT, written flat into `--output_dir`
- `--drivelm_path` given, no overrides → `--output_dir/nl_reasoning_samples/` and `--output_dir/action_only_samples/` subdirs
- `--drivelm_path` + both overrides → the two streams go to `--cot_output_dir` and `--nocot_output_dir` respectively (this is the mode used by `run_nuscenes_preprocess_nl.sh` for the flat-at-`/data/` layout)

**Examples**:

```bash
# (A) train split + DriveLM + both override dirs (flat layout)
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path /data/nuScenes \
    --split train \
    --drivelm_path /data/drivelm/v1_1_train_nus.json \
    --cot_output_dir   /data/demo_nuscenes_cot \
    --nocot_output_dir /data/demo_nuscenes_nocot

# (B) val split used as test set (no DriveLM, everything is no-CoT)
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path /data/nuScenes \
    --split val \
    --output_dir /data/demo_nuscenes_test

# (C) train split + DriveLM but use the default subdir layout (no overrides)
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path /data/nuScenes \
    --split train \
    --drivelm_path /data/drivelm/v1_1_train_nus.json \
    --output_dir /data/demo_nuscenes_train
# Result: /data/demo_nuscenes_train/nl_reasoning_samples/  +  .../action_only_samples/
```

### 3.4 `tools/preprocessing/symbolic_cot_sample_generation.py` — Symbolic CoT generation

**Purpose**: re-walk the raw dataset with the prompt rewritten in RLIB symbolic format, asking the VLM/LLM to emit a five-section CoT (`PERCEPTION / OPERATIONS / FACTS / RULES / ACTION`). Optionally takes a previous NL CoT directory as warm-start reference.

| Flag | Type / default | Description |
|---|---|---|
| `--config <name>` | **required** | YAML config name. Example: `dataset/symbolic-cot-gpt4o-mini-nuplan-trainval` |
| `--output_dir <path>` | **required** | Output directory |
| `--backend {vllm,openai}` | reads YAML | Override YAML |
| `--rlib_dir <path>` | `./RLIB` | RLIB rule library directory |
| `--nl-cot-dir <path>` | None | **Optional**: directory of NL CoT JSONs. For each matched token, the previous `cot_output` is used as an action hint + an in-prompt translation reference. Missing files / empty `cot_output` print a warning to stderr and fall back |
| `--free-rules` / `--no-free-rules` | default `--free-rules` | Free-rule mode lets the LLM compose rules freely; `--no-free-rules` reverts to the predefined RLIB rule set |
| `--path-prefix-map FROM=TO` | None (repeatable) | Cross-host path rewriting, e.g. `--path-prefix-map /data=./data` |
| `--sample-ids-json` / `--sample-ids-key` / `--seed` / `--resume` / `--dp_size` / `--tp_size` / `--num_parts` / `--sample_num` | same as §3.1 | |

**Examples**:

```bash
# (A) Run without NL CoT warm-start (project default; the shell script does this)
python -m tools.preprocessing.symbolic_cot_sample_generation \
    --config dataset/symbolic-cot-gpt4o-mini-nuplan-trainval \
    --output_dir /data/demo_symbolic_cot \
    --sample-ids-json /data/nuplan_nuscenes_train_mix_185k/scaling_185k_token_list.json \
    --sample-ids-key nuplan_cot \
    --rlib_dir ./RLIB \
    --dp_size 1 \
    --resume

# (B) Use a previously-generated NL CoT directory as warm-start reference
python -m tools.preprocessing.symbolic_cot_sample_generation \
    --config dataset/symbolic-cot-gpt4o-mini-nuplan-trainval \
    --output_dir /data/demo_symbolic_warmstart \
    --nl-cot-dir /data/nuplan_nl_reasoning_samples_Qwen2.5_VL_72B_Instruct_AWQ_45600 \
    --sample-ids-json /data/nuplan_nuscenes_train_mix_185k/scaling_185k_token_list.json \
    --sample-ids-key nuplan_cot \
    --resume

# (C) Use the predefined RLIB rule set instead of free-rule mode
python -m tools.preprocessing.symbolic_cot_sample_generation \
    --config dataset/symbolic-cot-gpt4o-mini-nuscenes-trainval \
    --output_dir /data/demo_symbolic_predefined \
    --no-free-rules \
    --sample-ids-json /data/nuplan_nuscenes_train_mix_185k/scaling_185k_token_list.json \
    --sample-ids-key nuscenes_cot \
    --resume
```

---

## 4. Debugging tips

- **Smoke-test first**: try the 10k scaling list (`/data/nuplan_nuscenes_train_mix_10k/scaling_10k_token_list.json`) before scaling to 185k.
- **Resume**: every script supports `--resume`, so re-running after a crash skips completed tokens.
- **Worker warnings**: symbolic CoT prints `[nl-cot] WARN: ...` to stderr when `--nl-cot-dir` lookups miss or `cot_output` is empty. `bash scripts/run_*.sh 2> >(tee /tmp/warn.log >&2)`, then grep.
- **Per-phase logs**: the four shell scripts tee each phase to `$AUTOVLA_ROOT/logs/run_*.log` by default (i.e. `<repo>/logs/`, gitignored) — handy for OOM / API-error postmortems.
- **DP fanout**: `DP_SIZE=1` for single-GPU; `DP_SIZE=8` on an 8-GPU box. The parent's `--dp_size` automatically forks subprocesses and partitions `CUDA_VISIBLE_DEVICES`.

---

## 5. Related docs

- [`AutoVLA.README.md`](./AutoVLA.README.md) — original paper README (dataset downloads, training commands, citation)
- [`tools/vessl/README.md`](./tools/vessl/README.md) — vessl-platform data loading helpers
- [`scripts/`](./scripts/) — all shell entry points
- [`tools/preprocessing/`](./tools/preprocessing/) — four Python entry points + `sample_selection_utils.py`
- [`config/dataset/`](./config/dataset/) — preprocessing YAML configs
- [`config/training/`](./config/training/) — SFT / GRPO training configs
- [`.envrc.example`](./.envrc.example) — environment template (`cp .envrc.example .envrc`, then `direnv allow`)
