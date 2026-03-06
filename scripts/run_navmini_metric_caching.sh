#!/bin/bash
# Generate metric cache for navmini split (required for GRPO training)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source environment variables
source "$PROJECT_ROOT/.envrc"
export PYTHONPATH="$PROJECT_ROOT/navsim:$PYTHONPATH"

TRAIN_TEST_SPLIT=navmini
CACHE_PATH=/export/scratch_large/ding/navsim_workspace/navmini_metric_cache

echo "=== Generating metric cache for $TRAIN_TEST_SPLIT ==="
echo "Output: $CACHE_PATH"
echo "Logs:   $OPENSCENE_DATA_ROOT/navsim_logs/mini/"
echo "Maps:   $NUPLAN_MAPS_ROOT"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH
