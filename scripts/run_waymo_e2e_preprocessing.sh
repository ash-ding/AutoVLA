#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

PYTHON="${PYTHON:-python}"
INCLUDE_COT="${INCLUDE_COT:-false}"
CONFIG="${CONFIG:-dataset/qwen2.5-vl-72B-waymo}"
OUTPUT_DIR="${OUTPUT_DIR:-temp}"
SAMPLE_IDS_JSON="${SAMPLE_IDS_JSON:-}"
RESUME="${RESUME:-false}"
NUM_WORKERS="${NUM_WORKERS:-32}"
SEED="${SEED:-42}"
CUDA_DEVICES="${CUDA_DEVICES:-}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_waymo_e2e_preprocessing.sh [options]

Options:
  --config NAME            Config name under config/ (default: dataset/qwen2.5-vl-72B-waymo)
  --output_dir PATH        Output directory for generated JSONs (default: temp)
  --include-cot true|false Run CoT or No-CoT preprocessing (default: false)
  --sample-ids-json PATH   JSON file containing tokens to preprocess
  --resume                 Skip tokens that already have JSON outputs in --output_dir
  --num-workers N          DataLoader workers (default: 32)
  --seed N                 Random seed (default: 42)
  --cuda-devices LIST      Value for CUDA_VISIBLE_DEVICES (default: leave unset; CoT defaults to "0,1")
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output_dir|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --include-cot)
            INCLUDE_COT="$2"
            shift 2
            ;;
        --sample-ids-json)
            SAMPLE_IDS_JSON="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
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

COMMON_ARGS=(
    --config "$CONFIG"
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
)

if [[ -n "$SAMPLE_IDS_JSON" ]]; then
    COMMON_ARGS+=(--sample-ids-json "$SAMPLE_IDS_JSON")
fi

if [[ "$RESUME" == true ]]; then
    COMMON_ARGS+=(--resume)
fi

if [[ "$INCLUDE_COT" == true ]]; then
    echo "Preprocessing with Chain-of-Thought (CoT)..."
    DEVICES="${CUDA_DEVICES:-0,1}"
    CUDA_VISIBLE_DEVICES="$DEVICES" "$PYTHON" tools/preprocessing/cot_sample_generation.py \
        "${COMMON_ARGS[@]}"
else
    echo "Preprocessing without Chain-of-Thought (No-CoT)..."
    NOCOT_ARGS=("${COMMON_ARGS[@]}" --num_workers "$NUM_WORKERS")
    if [[ -n "$CUDA_DEVICES" ]]; then
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" tools/preprocessing/nocot_sample_generation.py \
            "${NOCOT_ARGS[@]}"
    else
        "$PYTHON" tools/preprocessing/nocot_sample_generation.py "${NOCOT_ARGS[@]}"
    fi
fi
