#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/maps"
export NAVSIM_EXP_ROOT="/data/exp"

PYTHON="${PYTHON:-/root/AutoVLA/env/bin/python}"
INCLUDE_COT="${INCLUDE_COT:-true}"
CONFIG="${CONFIG:-dataset/qwen2.5-vl-72B-nuplan-8gpu-tp1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SEED="${SEED:-42}"
SAMPLE_IDS_JSON="${SAMPLE_IDS_JSON:-}"
RESUME="${RESUME:-false}"
LOG_DIR="${LOG_DIR:-}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nuplan_preprocessing.sh [options]

Options:
  --sample-ids-json PATH   JSON file containing the nuPlan tokens to process
  --resume                 Skip tokens whose JSON outputs already exist
  --config NAME            Config name under config/ (default: dataset/qwen2.5-vl-72B-nuplan-8gpu-tp1)
  --output-root PATH       Root output directory (default: /data)
  --num-shards N           Number of CoT shards to launch (default: 8)
  --seed N                 Random seed for shuffling when no token list is provided (default: 42)
  --include-cot true|false Whether to run CoT or No-CoT preprocessing (default: true)
  --log-dir PATH           Directory for per-shard log files
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample-ids-json)
            SAMPLE_IDS_JSON="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --include-cot)
            INCLUDE_COT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -n "$SAMPLE_IDS_JSON" && ! -f "$SAMPLE_IDS_JSON" ]]; then
    echo "Sample ID file not found: $SAMPLE_IDS_JSON" >&2
    exit 1
fi

if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
fi

COMMON_ARGS=(
    --config "$CONFIG"
    --seed "$SEED"
)

if [[ -n "$SAMPLE_IDS_JSON" ]]; then
    COMMON_ARGS+=(--sample-ids-json "$SAMPLE_IDS_JSON")
fi

if [[ "$RESUME" == true ]]; then
    COMMON_ARGS+=(--resume)
fi

if [[ "$INCLUDE_COT" == true ]]; then
    echo "Preprocessing with CoT using $NUM_SHARDS shards."
    if [[ -n "$SAMPLE_IDS_JSON" ]]; then
        echo "Using sample IDs from $SAMPLE_IDS_JSON"
    fi
    if [[ "$RESUME" == true ]]; then
        echo "Resume mode is enabled."
    fi
    if [[ -n "$LOG_DIR" ]]; then
        echo "Writing shard logs to $LOG_DIR"
    fi

    PIDS=()
    for i in $(seq 1 "$NUM_SHARDS"); do
        GPU_IDX=$((i - 1))
        OUT_DIR="${OUTPUT_ROOT}/CoT_part${i}"
        SHARD_ARGS=(
            "${COMMON_ARGS[@]}"
            --output_dir "$OUT_DIR"
            --num_parts "$NUM_SHARDS"
            --sample_num "$i"
        )

        if [[ -n "$LOG_DIR" ]]; then
            LOG_FILE="${LOG_DIR}/nuplan_preprocessing_shard${i}.log"
            CUDA_VISIBLE_DEVICES=$GPU_IDX "$PYTHON" tools/preprocessing/cot_sample_generation.py \
                "${SHARD_ARGS[@]}" >"$LOG_FILE" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$GPU_IDX "$PYTHON" tools/preprocessing/cot_sample_generation.py \
                "${SHARD_ARGS[@]}" &
        fi
        PIDS+=($!)
    done

    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=1
    done

    if [[ "$FAIL" -ne 0 ]]; then
        echo "One or more CoT preprocessing shards failed."
        exit 1
    fi
else
    OUTPUT_DIR="${OUTPUT_ROOT}/CoT"
    echo "Preprocessing without Chain-of-Thought (No-CoT)..."
    if [[ -n "$SAMPLE_IDS_JSON" ]]; then
        echo "Using sample IDs from $SAMPLE_IDS_JSON"
    fi
    if [[ "$RESUME" == true ]]; then
        echo "Resume mode is enabled."
    fi
    if [[ -n "$LOG_DIR" ]]; then
        echo "Writing logs to $LOG_DIR"
    fi

    NO_COT_ARGS=(
        "${COMMON_ARGS[@]}"
        --output_dir "$OUTPUT_DIR"
    )

    if [[ -n "$LOG_DIR" ]]; then
        "$PYTHON" tools/preprocessing/nocot_sample_generation.py \
            "${NO_COT_ARGS[@]}" >"${LOG_DIR}/nuplan_preprocessing_nocot.log" 2>&1
    else
        "$PYTHON" tools/preprocessing/nocot_sample_generation.py "${NO_COT_ARGS[@]}"
    fi
fi
