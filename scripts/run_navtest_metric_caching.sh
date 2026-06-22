#!/bin/bash
# Generate metric cache for navtest split (required for PDM-score eval).
# This is CPU-bound (no GPU inference); set NUM_WORKERS to roughly nproc-8.
#
# Optional env vars:
#   NUM_WORKERS  : thread-pool size (default: nproc - 8, clamped to >= 1)
#   WORKER_TYPE  : single_machine_thread_pool | sequential | ray_distributed
#                  (default: single_machine_thread_pool)
#
# Output: $NAVSIM_EXP_ROOT/navtest_metric_cache/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source /root/install/anaconda3/etc/profile.d/conda.sh
conda activate autovla
source "$PROJECT_ROOT/.envrc"
export PYTHONPATH="$PROJECT_ROOT/navsim:${PYTHONPATH:-}"

TRAIN_TEST_SPLIT=navtest
CACHE_PATH="$NAVSIM_EXP_ROOT/${TRAIN_TEST_SPLIT}_metric_cache"
WORKER_TYPE="${WORKER_TYPE:-single_machine_thread_pool}"

# Default to nproc - 8 to leave headroom for OS + concurrent SSH session
DEFAULT_WORKERS=$(( $(nproc) - 8 ))
[ "$DEFAULT_WORKERS" -lt 1 ] && DEFAULT_WORKERS=1
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_WORKERS}"

mkdir -p "$NAVSIM_EXP_ROOT"

echo "=== Generating metric cache for $TRAIN_TEST_SPLIT ==="
echo "Output:       $CACHE_PATH"
echo "Logs:         $OPENSCENE_DATA_ROOT/navsim_logs/test/"
echo "Maps:         $NUPLAN_MAPS_ROOT"
echo "Worker type:  $WORKER_TYPE"
echo "Workers:      $NUM_WORKERS"
echo "Free /data:   $(df -h /data | tail -1 | awk '{print $4}')"
echo "================================================="

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path="$CACHE_PATH" \
    worker="$WORKER_TYPE" \
    worker.max_workers="$NUM_WORKERS" \
    worker.use_process_pool=true
# IMPORTANT: use_process_pool=true forces ProcessPoolExecutor instead of the
# YAML default ThreadPoolExecutor. The work is CPU-bound (Python),
# so threads hit the GIL and pin throughput to ~1 core (observed: 20 token-
# caches/min => ~10 hours). Process pool gets near-linear scaling
# (~40× speedup → ~15-30 min on navtest 12146 tokens).
