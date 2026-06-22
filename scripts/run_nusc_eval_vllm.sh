#!/bin/bash
# Env-var driven driver for vLLM-backed nuScenes eval.
#
# Required env:
#   CONFIG       : eval YAML (e.g. config/training/eval-4v90-nuscenes.yaml)
#   HF_DIR       : HF safetensors dir from tools/convert_sft_ckpt_to_hf.py
#   OUTPUT_DIR   : where to write results.txt + per_sample/
#
# Optional env:
#   SEG_DATA_PATH (default /data/nusc_eval_seg_6s)
#   NUM_SAMPLES   (default: all)
#   BATCH_SIZE    (default 64)
#   GPU_MEM_UTIL  (default 0.85)
#   CUDA_VISIBLE_DEVICES (default 0)
#
# Example:
#   CONFIG=config/training/eval-4v90-nuscenes.yaml \
#   HF_DIR=/backup/hf_ckpt/4v90 \
#   OUTPUT_DIR=/data/eval_results/4v90/nuscenes \
#   bash scripts/run_nusc_eval_vllm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /root/install/anaconda3/etc/profile.d/conda.sh
conda activate autovla
source .envrc

: "${CONFIG:?must set CONFIG (e.g. config/training/eval-4v90-nuscenes.yaml)}"
: "${HF_DIR:?must set HF_DIR (run tools/convert_sft_ckpt_to_hf.py first)}"
: "${OUTPUT_DIR:?must set OUTPUT_DIR}"

SEG_DATA_PATH="${SEG_DATA_PATH:-/data/nusc_eval_seg_6s}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SAVE_PER_SAMPLE_RESULT="${SAVE_PER_SAMPLE_RESULT:-true}"
DP_SIZE="${DP_SIZE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUTPUT_DIR"

echo "===== [run_nusc_eval_vllm] ====="
echo "  CONFIG:        $CONFIG"
echo "  HF_DIR:        $HF_DIR"
echo "  OUTPUT_DIR:    $OUTPUT_DIR"
echo "  SEG:           $SEG_DATA_PATH"
echo "  BATCH_SIZE:    $BATCH_SIZE"
echo "  GPU_MEM_UTIL:  $GPU_MEM_UTIL"
echo "  GPUs:          $CUDA_VISIBLE_DEVICES"
echo "================================"

EXTRA=""
[ -n "$NUM_SAMPLES" ] && EXTRA="$EXTRA --num_samples $NUM_SAMPLES"
case "$SAVE_PER_SAMPLE_RESULT" in
    false|0|no) EXTRA="$EXTRA --no-save_per_sample_result" ;;
esac
[ "$DP_SIZE" -gt 1 ] && EXTRA="$EXTRA --dp_size $DP_SIZE"

python tools/eval/nusc_eval_vllm.py \
    --config "$CONFIG" \
    --hf_dir "$HF_DIR" \
    --seg_data_path "$SEG_DATA_PATH" \
    --output "$OUTPUT_DIR/results.txt" \
    --per_sample_dir "$OUTPUT_DIR/per_sample" \
    --batch_size "$BATCH_SIZE" \
    --gpu_mem_util "$GPU_MEM_UTIL" \
    $EXTRA
