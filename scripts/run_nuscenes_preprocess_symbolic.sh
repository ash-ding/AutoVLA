#!/usr/bin/env bash
# Generate symbolic CoT JSONs for the nuscenes_cot bucket of the 185k mix scaling list
# via symbolic_cot_sample_generation.py + GPT-4o-mini + RLIB. No --nl-cot-dir.
#
# Env overrides: DP_SIZE (default 1), LOG_DIR (default "$AUTOVLA_ROOT/logs").
# Requires: OPENAI_API_KEY in env.
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

LOG="$LOG_DIR/run_nuscenes_preprocess_symbolic.log"
echo "===== [run_nuscenes_preprocess_symbolic] log=${LOG} =====" | tee -a "$LOG"

python -m tools.preprocessing.symbolic_cot_sample_generation \
    --config dataset/symbolic-cot-gpt4o-mini-nuscenes-trainval \
    --output_dir /data/nuscenes_symbolic_reasoning_samples_gpt_4o_mini_2895 \
    --sample-ids-json "$SCALING_TOKEN_LIST" \
    --sample-ids-key nuscenes_cot \
    --rlib_dir "$REPO_ROOT/symdrive/rlib1_0/rlib" \
    --dp_size "$DP_SIZE" \
    --resume 2>&1 | tee -a "$LOG"

echo "===== [run_nuscenes_preprocess_symbolic] complete ====="
