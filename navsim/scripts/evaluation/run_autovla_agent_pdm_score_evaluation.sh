#!/bin/bash
# PDM-score closed-loop evaluation for AutoVLA on nuPlan (navtest).
#
# Required env vars (per-run override; no sensible default for these):
#   CONFIG_PATH  : training YAML for the SFT'd model
#                  (e.g. config/training/qwen2.5-vl-3B-mix-sft-10k-rlib1.0-4v90.yaml)
#   CHECKPOINT   : SFT checkpoint .ckpt path
#                  (e.g. /backup/runs/sft/.../epoch=N-loss=*.ckpt)
#
# Optional env vars (defaults match this box's layout):
#   TRAIN_TEST_SPLIT       default: navtest
#   NAVSIM_DEVKIT_ROOT     default: $REPO_ROOT/navsim
#   CACHE_PATH             default: $NAVSIM_EXP_ROOT/navtest_metric_cache
#   JSON_DATA_PATH         default: /data/nuplan_test/test_samples_12146
#   SENSOR_DATA_PATH       default: /data/nuPlan/sensor_blobs/test
#   LORA                   default: false
#   EXPERIMENT_NAME        default: autovla_agent
#   NUM_EVAL_WORKERS       default: 8        (1 per GPU on 8x A100)
#   GPUS_PER_EVAL_WORKER   default: 1
#   WORKER_TYPE            default: single_machine_thread_pool
#                          fallback to: ray_distributed (if GPU pinning per worker fails)
#                          or: sequential (single-GPU debug)
#   CUDA_VISIBLE_DEVICES   default: 0,1,2,3,4,5,6,7
#   EXTRA_ARGS             default: ""       (passed to run_pdm_score_cot.py)
#
# Usage examples:
#   # Arm A on 8 GPU
#   CONFIG_PATH=config/training/qwen2.5-vl-3B-mix-sft-10k-rlib1.0-4v90.yaml \
#   CHECKPOINT=/backup/runs/sft/.../4v90/.../epoch=3-loss=0.8973.ckpt \
#   EXPERIMENT_NAME=eval_4v90_nuplan \
#   bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh
#
#   # Single-GPU fallback if multi-worker doesn't dispatch GPUs as expected
#   CUDA_VISIBLE_DEVICES=0 NUM_EVAL_WORKERS=1 GPUS_PER_EVAL_WORKER=1 \
#   WORKER_TYPE=sequential CONFIG_PATH=... CHECKPOINT=... \
#   bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Activate conda + source repo env vars (NAVSIM_EXP_ROOT, OPENSCENE_DATA_ROOT, etc.)
source /root/install/anaconda3/etc/profile.d/conda.sh
conda activate autovla
source .envrc

export PYTHONPATH="$REPO_ROOT/navsim:${PYTHONPATH:-}"

# === Required ===
: "${CONFIG_PATH:?must set CONFIG_PATH (e.g. config/training/qwen2.5-vl-3B-mix-sft-10k-rlib1.0-4v90.yaml)}"
: "${CHECKPOINT:?must set CHECKPOINT (e.g. /backup/runs/sft/.../epoch=N-loss=*.ckpt)}"

# === Defaults ===
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$REPO_ROOT/navsim}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/navtest_metric_cache}"
JSON_DATA_PATH="${JSON_DATA_PATH:-/data/nuplan_test/test_samples_12146}"
SENSOR_DATA_PATH="${SENSOR_DATA_PATH:-/data/nuPlan/sensor_blobs/test}"
LORA="${LORA:-false}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-autovla_agent}"
NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-8}"
GPUS_PER_EVAL_WORKER="${GPUS_PER_EVAL_WORKER:-1}"
WORKER_TYPE="${WORKER_TYPE:-single_machine_thread_pool}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "===== [run_autovla_agent_pdm_score_evaluation] ====="
echo "  CONFIG_PATH:           $CONFIG_PATH"
echo "  CHECKPOINT:            $CHECKPOINT"
echo "  EXPERIMENT_NAME:       $EXPERIMENT_NAME"
echo "  TRAIN_TEST_SPLIT:      $TRAIN_TEST_SPLIT"
echo "  CACHE_PATH:            $CACHE_PATH"
echo "  JSON_DATA_PATH:        $JSON_DATA_PATH"
echo "  SENSOR_DATA_PATH:      $SENSOR_DATA_PATH"
echo "  CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "  WORKER_TYPE:           $WORKER_TYPE"
echo "  NUM_EVAL_WORKERS:      $NUM_EVAL_WORKERS"
echo "  GPUS_PER_EVAL_WORKER:  $GPUS_PER_EVAL_WORKER"
echo "==================================================="

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
  train_test_split=$TRAIN_TEST_SPLIT \
  agent=autovla_agent \
  +agent.config_path="$CONFIG_PATH" \
  +agent.checkpoint_path="$CHECKPOINT" \
  +agent.sensor_data_path="$SENSOR_DATA_PATH" \
  +agent.lora_conf.use_lora=$LORA \
  metric_cache_path="$CACHE_PATH" \
  json_data_path="$JSON_DATA_PATH" \
  experiment_name="$EXPERIMENT_NAME" \
  worker="$WORKER_TYPE" \
  num_eval_workers=$NUM_EVAL_WORKERS \
  +gpus_per_eval_worker=$GPUS_PER_EVAL_WORKER \
  $EXTRA_ARGS
