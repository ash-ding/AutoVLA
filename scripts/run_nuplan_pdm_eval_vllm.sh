#!/bin/bash
# Env-var driven driver for vLLM-backed nuPlan PDM-score eval.
#
# Required env:
#   CONFIG          : SFT YAML (e.g. config/training/qwen2.5-vl-3B-mix-sft-10k-rlib1.0-4v90.yaml)
#   HF_DIR          : HF safetensors dir from tools/convert_sft_ckpt_to_hf.py
#   METRIC_CACHE_PATH : navtest metric cache dir (e.g. /data/navsim_exp/navtest_metric_cache)
#   OUTPUT_DIR      : where to write <timestamp>.csv + per_sample/
#
# Optional env:
#   JSON_DATA_PATH   (default /data/nuplan_test/test_samples_12146)
#   SENSOR_DATA_PATH (default /data/nuPlan/sensor_blobs/test)
#   TRAIN_TEST_SPLIT (default navtest)
#   NUM_TOKENS       (default: all)
#   BATCH_SIZE       (default 32)
#   PDM_WORKERS      (default 16)
#   GPU_MEM_UTIL     (default 0.85)
#   NUPLAN_SIDE_FIELD (default: from CONFIG.model.nuplan_side_field)
#   CUDA_VISIBLE_DEVICES (default 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /root/install/anaconda3/etc/profile.d/conda.sh
conda activate autovla
source .envrc

: "${CONFIG:?must set CONFIG (e.g. config/training/qwen2.5-vl-3B-mix-sft-10k-rlib1.0-4v90.yaml)}"
: "${HF_DIR:?must set HF_DIR (run tools/convert_sft_ckpt_to_hf.py first)}"
: "${METRIC_CACHE_PATH:?must set METRIC_CACHE_PATH (build via scripts/run_navtest_metric_caching.sh)}"
: "${OUTPUT_DIR:?must set OUTPUT_DIR}"

JSON_DATA_PATH="${JSON_DATA_PATH:-/data/nuplan_test/test_samples_12146}"
SENSOR_DATA_PATH="${SENSOR_DATA_PATH:-/data/nuPlan/sensor_blobs/test}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
NUM_TOKENS="${NUM_TOKENS:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PDM_WORKERS="${PDM_WORKERS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
NUPLAN_SIDE_FIELD="${NUPLAN_SIDE_FIELD:-}"
SAVE_PER_SAMPLE_RESULT="${SAVE_PER_SAMPLE_RESULT:-true}"
DP_SIZE="${DP_SIZE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUTPUT_DIR"

echo "===== [run_nuplan_pdm_eval_vllm] ====="
echo "  CONFIG:              $CONFIG"
echo "  HF_DIR:              $HF_DIR"
echo "  METRIC_CACHE_PATH:   $METRIC_CACHE_PATH"
echo "  JSON_DATA_PATH:      $JSON_DATA_PATH"
echo "  SENSOR_DATA_PATH:    $SENSOR_DATA_PATH"
echo "  OUTPUT_DIR:          $OUTPUT_DIR"
echo "  TRAIN_TEST_SPLIT:    $TRAIN_TEST_SPLIT"
echo "  BATCH_SIZE:          $BATCH_SIZE"
echo "  PDM_WORKERS:         $PDM_WORKERS"
echo "  GPU_MEM_UTIL:        $GPU_MEM_UTIL"
echo "  GPUs:                $CUDA_VISIBLE_DEVICES"
echo "======================================"

EXTRA=""
[ -n "$NUM_TOKENS" ] && EXTRA="$EXTRA --num_tokens $NUM_TOKENS"
[ -n "$NUPLAN_SIDE_FIELD" ] && EXTRA="$EXTRA --nuplan_side_field $NUPLAN_SIDE_FIELD"
case "$SAVE_PER_SAMPLE_RESULT" in
    false|0|no) EXTRA="$EXTRA --no-save_per_sample_result" ;;
esac
[ "$DP_SIZE" -gt 1 ] && EXTRA="$EXTRA --dp_size $DP_SIZE"

python tools/eval/nuplan_pdm_eval_vllm.py \
    --config "$CONFIG" \
    --hf_dir "$HF_DIR" \
    --metric_cache_path "$METRIC_CACHE_PATH" \
    --json_data_path "$JSON_DATA_PATH" \
    --sensor_data_path "$SENSOR_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_test_split "$TRAIN_TEST_SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --pdm_workers "$PDM_WORKERS" \
    --gpu_mem_util "$GPU_MEM_UTIL" \
    $EXTRA
