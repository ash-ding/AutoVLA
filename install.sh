#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$PWD/env/bin/python" ]]; then
        PYTHON="$PWD/env/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        echo "No Python interpreter found. Set PYTHON=/path/to/python and rerun." >&2
        exit 1
    fi
fi

if command -v uv >/dev/null 2>&1; then
    PIP_INSTALL=(uv pip install --python "$PYTHON")
else
    if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
        "$PYTHON" -m ensurepip --upgrade
    fi
    PIP_INSTALL=("$PYTHON" -m pip install)
fi

echo "Installing into: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# Keep local packages importable even when the environment was created by uv and
# does not include a pip executable.
"${PIP_INSTALL[@]}" -e . --no-deps
if [[ -d navsim ]]; then
    "${PIP_INSTALL[@]}" -e navsim --no-deps
fi

# NOTE: flash-attn is intentionally NOT installed.
#   - The AutoVLA repo has zero `flash_attn` imports; Qwen2.5-VL is loaded
#     without an attn_implementation kwarg, so transformers defaults to SDPA
#     (built into torch 2.10 — no compile needed) and falls back to eager.
#   - vLLM ships its own pre-compiled flash-attn internally (vllm_flash_attn);
#     the repo doesn't import it either.
#   - Building flash-attn==2.7.4.post1 from source reliably OOMs nvcc on this
#     box (no matching prebuilt wheel for torch 2.10 + cu128) and drops SSH.
# pip install flash-attn==2.7.4.post1

# Required by dataset_utils/preprocessing/waymo_e2e_dataset.py, which is used
# by tools/streaming/waymo_run_prepared_chunks.py. The Waymo 1.6.7 wheel's
# metadata pins an unavailable jaxlib, but this workflow only needs its TF
# protos and camera ops, so install the runtime pieces explicitly.
"${PIP_INSTALL[@]}" tensorflow==2.13.0
"${PIP_INSTALL[@]}" waymo-open-dataset-tf-2-12-0==1.6.7 --no-deps

# TensorFlow's resolver pulls older protobuf/grpc/tensorboard pins. The CoT
# path also imports vLLM, so keep the shared stack on versions that both import
# cleanly in this environment.
"${PIP_INSTALL[@]}" protobuf==5.29.6 grpcio==1.80.0 tensorboard==2.20.0 cryptography==47.0.0
"${PIP_INSTALL[@]}" --upgrade typing_extensions
"${PIP_INSTALL[@]}" autoawq==0.2.8 --no-deps --no-build-isolation

# if [[ "${SKIP_IMPORT_CHECK:-0}" != "1" ]]; then
#     TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
#     TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}" \
#     "$PYTHON" - <<'PY'
# import yaml
# import tensorflow
# from waymo_open_dataset.protos import end_to_end_driving_data_pb2
# import dataset_utils.preprocessing.waymo_e2e_dataset

# print("Waymo prepared-chunk imports OK.")
# PY
# fi
