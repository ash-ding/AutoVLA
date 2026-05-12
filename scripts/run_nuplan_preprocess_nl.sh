#!/usr/bin/env bash
# Generate NL CoT + no-CoT JSONs for the nuPlan portion of the 185k mix scaling list,
# plus full no-CoT JSONs for the nuPlan navtest split. Outputs land flat under /data/.
#
#   Phase A: NL CoT via Qwen2.5-VL-72B-AWQ on nuplan_cot bucket    (45600 samples)
#   Phase B: no-CoT (model-agnostic)         on nuplan_nocot bucket (120682 samples)
#   Phase C: no-CoT navtest (no token-list filtering)              (12146 samples)
#
# Env overrides: DP_SIZE (default 1), LOG_DIR (default "$AUTOVLA_ROOT/logs")
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /root/install/anaconda3/etc/profile.d/conda.sh
conda activate autovla
source .envrc

SCALING_TOKEN_LIST=/data/nuplan_nuscenes_train_mix_185k/scaling_185k_token_list.json
DP_SIZE="${DP_SIZE:-1}"
LOG_DIR="${LOG_DIR:-$AUTOVLA_ROOT/logs}"
mkdir -p "$LOG_DIR"

run_phase() {
    local label="$1"; shift
    local log="$LOG_DIR/run_nuplan_preprocess_nl_${label}.log"
    echo "===== [run_nuplan_preprocess_nl] phase=${label} | log=${log} =====" | tee -a "$log"
    "$@" 2>&1 | tee -a "$log"
}

# --- Phase A: NL CoT (Qwen2.5-VL-72B-AWQ) on nuplan_cot bucket ---
run_phase nl_cot \
    python -m tools.preprocessing.nl_cot_sample_generation \
        --config dataset/qwen2.5-vl-72B-nuplan-trainval \
        --output_dir /data/nuplan_nl_reasoning_samples_Qwen2.5_VL_72B_Instruct_AWQ_45600 \
        --sample-ids-json "$SCALING_TOKEN_LIST" \
        --sample-ids-key nuplan_cot \
        --dp_size "$DP_SIZE" \
        --resume

# --- Phase B: no-CoT trainval on nuplan_nocot bucket ---
run_phase no_cot_trainval \
    python -m tools.preprocessing.nocot_sample_generation \
        --config dataset/nocot_nuplan-trainval \
        --output_dir /data/nuplan_action_only_samples_120682 \
        --sample-ids-json "$SCALING_TOKEN_LIST" \
        --sample-ids-key nuplan_nocot \
        --resume

# --- Phase C: no-CoT navtest (full split, no token-list filter) ---
run_phase no_cot_test \
    python -m tools.preprocessing.nocot_sample_generation \
        --config dataset/nocot_nuplan-navtest \
        --output_dir /data/nuplan_test_samples_12146 \
        --resume

echo "===== [run_nuplan_preprocess_nl] all phases complete ====="
