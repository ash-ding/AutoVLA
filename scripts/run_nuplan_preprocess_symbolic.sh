#!/usr/bin/env bash
# Generate symbolic CoT JSONs for the nuplan_cot bucket of the 185k mix scaling list
# via symbolic_cot_sample_generation.py + GPT-4o-mini + RLIB. No --nl-cot-dir
# (per project decision; symbolic runs independently of the NL CoT outputs).
#
# Env overrides: DP_SIZE (default 1), LOG_DIR (default "$AUTOVLA_ROOT/logs").
# Requires: OPENAI_API_KEY in env (annotation_backend=openai in the YAML).
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

LOG="$LOG_DIR/run_nuplan_preprocess_symbolic.log"
echo "===== [run_nuplan_preprocess_symbolic] log=${LOG} =====" | tee -a "$LOG"

python -m tools.preprocessing.symbolic_cot_sample_generation \
    --config dataset/symbolic-cot-gpt4o-mini-nuplan-trainval \
    --output_dir /data/nuplan_symbolic_reasoning_samples_gpt_4o_mini_45600 \
    --sample-ids-json "$SCALING_TOKEN_LIST" \
    --sample-ids-key nuplan_cot \
    --rlib_dir "$REPO_ROOT/RLIB" \
    --dp_size "$DP_SIZE" \
    --resume 2>&1 | tee -a "$LOG"

echo "===== [run_nuplan_preprocess_symbolic] complete ====="
