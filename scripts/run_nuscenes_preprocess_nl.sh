#!/usr/bin/env bash
# Generate NL CoT + no-CoT JSONs for nuScenes via nusc_sample_generation.py.
# Outputs land flat under /data/, using the new --cot_output_dir / --nocot_output_dir
# overrides to bypass the default nl_reasoning_samples/ + action_only_samples/ subdir layout.
#
#   Phase A: train split with DriveLM (v1_1_train_nus.json) -> two flat sibling dirs
#            nuscenes_nl_reasoning_samples_2895/   (DriveLM-derived 5-element CoT)
#            nuscenes_action_only_samples_16135/   (train frames without DriveLM coverage)
#   Phase B: val split, no DriveLM (-> nuscenes_test_samples_5569/, all no-CoT)
#
# Env overrides: LOG_DIR (default "$AUTOVLA_ROOT/logs")
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /root/install/anaconda3/etc/profile.d/conda.sh
conda activate autovla
source .envrc

NUSCENES_PATH=/data/nuScenes
NUSCENES_VERSION=v1.0-trainval
DRIVELM_TRAIN=/data/drivelm/v1_1_train_nus.json
LOG_DIR="${LOG_DIR:-$AUTOVLA_ROOT/logs}"
mkdir -p "$LOG_DIR"

run_phase() {
    local label="$1"; shift
    local log="$LOG_DIR/run_nuscenes_preprocess_nl_${label}.log"
    echo "===== [run_nuscenes_preprocess_nl] phase=${label} | log=${log} =====" | tee -a "$log"
    "$@" 2>&1 | tee -a "$log"
}

# --- Phase A: train split, DriveLM splits CoT vs. no-CoT into two flat sibling dirs ---
run_phase train \
    python tools/preprocessing/nusc_sample_generation.py \
        --nuscenes_path "$NUSCENES_PATH" \
        --version "$NUSCENES_VERSION" \
        --split train \
        --drivelm_path "$DRIVELM_TRAIN" \
        --cot_output_dir   /data/nuscenes_nl_reasoning_samples_2895 \
        --nocot_output_dir /data/nuscenes_action_only_samples_16135

# --- Phase B: val split as test set (no DriveLM, all no-CoT) ---
run_phase test \
    python tools/preprocessing/nusc_sample_generation.py \
        --nuscenes_path "$NUSCENES_PATH" \
        --version "$NUSCENES_VERSION" \
        --split val \
        --output_dir /data/nuscenes_test_samples_5569

echo "===== [run_nuscenes_preprocess_nl] all phases complete ====="
