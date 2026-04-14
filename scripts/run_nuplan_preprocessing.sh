#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/export/scratch_large/pouya/autovla_dataset/maps"
export NAVSIM_EXP_ROOT="/export/scratch_large/pouya/autovla_dataset/exp"

PYTHON="/export/scratch_large/pouya/AutoVLA/env/bin/python"
INCLUDE_COT=true
CONFIG="dataset/qwen2.5-vl-72B-nuplan"
OUTPUT_DIR="/export/scratch_large/pouya/autovla_dataset/CoT"

if [ "$INCLUDE_COT" = true ]; then
    echo "Preprocessing with Chain-of-Thought (CoT)..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON tools/preprocessing/cot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Preprocessing without Chain-of-Thought (No-CoT)..."
    $PYTHON tools/preprocessing/nocot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
fi