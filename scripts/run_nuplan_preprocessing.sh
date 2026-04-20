#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/maps"
export NAVSIM_EXP_ROOT="/data/exp"

PYTHON="/root/AutoVLA/env/bin/python"
INCLUDE_COT=true
CONFIG="dataset/qwen2.5-vl-72B-nuplan-8gpu-tp1"
OUTPUT_ROOT="/data"
NUM_SHARDS=8

if [ "$INCLUDE_COT" = true ]; then
    echo "Preprocessing with CoT using $NUM_SHARDS shards (TP=1, one GPU per shard)..."
    PIDS=()
    for i in $(seq 1 "$NUM_SHARDS"); do
        GPU_IDX=$((i - 1))
        OUT_DIR="${OUTPUT_ROOT}/CoT_part${i}"
        CUDA_VISIBLE_DEVICES=$GPU_IDX $PYTHON tools/preprocessing/cot_sample_generation.py \
            --config "$CONFIG" \
            --output_dir "$OUT_DIR" \
            --num_parts "$NUM_SHARDS" \
            --sample_num "$i" \
            --seed 42 &
        PIDS+=($!)
    done

    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=1
    done

    if [ "$FAIL" -ne 0 ]; then
        echo "One or more CoT preprocessing shards failed."
        exit 1
    fi
else
    echo "Preprocessing without Chain-of-Thought (No-CoT)..."
    $PYTHON tools/preprocessing/nocot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "/data/CoT"
fi
