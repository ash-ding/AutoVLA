#!/bin/bash

REPO_ID="Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_DIR="./model_weights/Qwen2.5-VL-3B-Instruct"

python tools/download/download_qwen.py \
    --repo_id "$REPO_ID" \
    --local_dir "$LOCAL_DIR" \