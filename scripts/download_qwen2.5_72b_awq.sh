REPO_ID="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
LOCAL_DIR="/root/AutoVLA/autovla_models/Qwen2.5-VL-72B-Instruct-AWQ"

python tools/download/download_qwen.py \
    --repo_id "$REPO_ID" \
    --local_dir "$LOCAL_DIR"