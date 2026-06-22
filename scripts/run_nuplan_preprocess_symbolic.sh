#!/usr/bin/env bash
# Generate symbolic CoT JSONs for a nuPlan bucket via symbolic_cot_sample_generation.py.
#
# Env overrides (all optional):
#   CONFIG_NAME         hydra-style dataset config name (without .yaml).
#                       Default: dataset/symbolic-cot-gpt4o-mini-nuplan-trainval
#                       (legacy gpt-4o-mini setup; the rlib1.0 ablation experiments
#                       use dataset/qwen2.5-vl-72B-nuplan-symbolic-{4v90,3v90,4v45})
#   OUTPUT_DIR          where per-token JSONs land.
#                       Default: /data/nuplan_symbolic_reasoning_samples_gpt_4o_mini_45600
#   SCALING_TOKEN_LIST  path to scaling_<N>_token_list.json.
#                       Default: /data/nuplan_nuscenes_train_mix_185k/scaling_185k_token_list.json
#   SAMPLE_IDS_KEY      bucket name within the scaling JSON. Default: nuplan_cot
#   RLIB_DIR            symbolic ontology dir. Default: $REPO_ROOT/symdrive/rlib1_0/rlib
#   DP_SIZE             vLLM data-parallel replicas. Default: 1 (set 8 on 8xA100)
#   TP_SIZE             vLLM tensor-parallel size. Default: omitted (use YAML's value)
#   EXTRA_ARGS          additional flags forwarded to the python entrypoint.
#   LOG_DIR             log directory. Default: $AUTOVLA_ROOT/logs
#
# OpenAI backend needs OPENAI_API_KEY in env; vLLM backend needs the model on disk.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /root/install/anaconda3/etc/profile.d/conda.sh
conda activate autovla
source .envrc

CONFIG_NAME="${CONFIG_NAME:-dataset/symbolic-cot-gpt4o-mini-nuplan-trainval}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/nuplan_symbolic_reasoning_samples_gpt_4o_mini_45600}"
SCALING_TOKEN_LIST="${SCALING_TOKEN_LIST:-/data/nuplan_nuscenes_train_mix_185k/scaling_185k_token_list.json}"
SAMPLE_IDS_KEY="${SAMPLE_IDS_KEY:-nuplan_cot}"
RLIB_DIR="${RLIB_DIR:-$REPO_ROOT/symdrive/rlib1_0/rlib}"
DP_SIZE="${DP_SIZE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${LOG_DIR:-$AUTOVLA_ROOT/logs}"
mkdir -p "$LOG_DIR"

# Log file: derive from output dir name when caller didn't override.
LOG_NAME="$(basename "$OUTPUT_DIR").log"
LOG="$LOG_DIR/run_nuplan_preprocess_symbolic_${LOG_NAME}"

echo "===== [run_nuplan_preprocess_symbolic] config=${CONFIG_NAME} out=${OUTPUT_DIR} =====" | tee -a "$LOG"
echo "  bucket=${SAMPLE_IDS_KEY}  scaling=${SCALING_TOKEN_LIST}" | tee -a "$LOG"
echo "  rlib_dir=${RLIB_DIR}  dp=${DP_SIZE}  extra='${EXTRA_ARGS}'  log=${LOG}" | tee -a "$LOG"

# Build optional TP_SIZE flag
TP_FLAG=()
if [[ -n "${TP_SIZE:-}" ]]; then
    TP_FLAG=(--tp_size "$TP_SIZE")
fi

python -m tools.preprocessing.symbolic_cot_sample_generation \
    --config "$CONFIG_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --sample-ids-json "$SCALING_TOKEN_LIST" \
    --sample-ids-key "$SAMPLE_IDS_KEY" \
    --rlib_dir "$RLIB_DIR" \
    --dp_size "$DP_SIZE" \
    "${TP_FLAG[@]}" \
    --resume $EXTRA_ARGS 2>&1 | tee -a "$LOG"

echo "===== [run_nuplan_preprocess_symbolic] complete =====" | tee -a "$LOG"
