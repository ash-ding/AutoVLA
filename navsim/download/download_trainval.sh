#!/usr/bin/env bash

set -euo pipefail

STATE_DIR=".navsim_download_state/trainval"
ARCHIVE_DIR="${STATE_DIR}/archives"

if ! command -v aria2c >/dev/null 2>&1; then
    echo "aria2c is required but not installed. See https://aria2.github.io/" >&2
    exit 1
fi

mkdir -p "${ARCHIVE_DIR}"

download_and_extract() {
    local url="$1"
    local archive_name="${url##*/}"
    local archive_path="${ARCHIVE_DIR}/${archive_name}"
    local marker_path="${STATE_DIR}/${archive_name}.done"

    if [ -f "${marker_path}" ]; then
        echo "Skipping ${archive_name}; already downloaded and extracted."
        return
    fi

    if [ ! -f "${archive_path}" ]; then
        aria2c \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --continue=true \
            --summary-interval=0 \
            --console-log-level=warn \
            --dir="${ARCHIVE_DIR}" \
            --out="${archive_name}" \
            "${url}"
    else
        echo "Reusing downloaded archive ${archive_name}..."
    fi

    tar -xzf "${archive_path}"
    rm -f "${archive_path}"
    touch "${marker_path}"
}

export -f download_and_extract
export STATE_DIR ARCHIVE_DIR

move_if_needed() {
    local src_dir="$1"
    local dest_dir="$2"

    if [ -d "${dest_dir}" ]; then
        echo "Keeping existing ${dest_dir}; skipping move from ${src_dir}."
        return
    fi

    if [ -d "${src_dir}" ]; then
        mv "${src_dir}" "${dest_dir}"
    fi
}

if [ -d trainval_navsim_logs ] && [ -d trainval_sensor_blobs ]; then
    echo "trainval_navsim_logs and trainval_sensor_blobs already exist; nothing to do."
    exit 0
fi

if [ -d trainval_navsim_logs ] || [ -d openscene-v1.1/meta_datas ]; then
    touch "${STATE_DIR}/openscene_metadata_trainval.tgz.done"
    echo "Skipping openscene_metadata_trainval.tgz; metadata is already available."
else
    download_and_extract "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz"
fi

if [ -d trainval_sensor_blobs ]; then
    echo "trainval_sensor_blobs already exists; skipping camera archives."
else
    printf "%s\n" {0..199} | xargs -I {} -P 1 bash -c '
        split="$1"
        echo "Processing camera split ${split}..."
        download_and_extract "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_${split}.tgz"
    ' _ {}
fi

# printf "%s\n" {0..199} | xargs -I {} -P 8 bash -c '
#     split="$1"
#     echo "Processing lidar trainval split ${split}..."
#     download_and_extract "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_lidar/openscene_sensor_trainval_lidar_${split}.tgz"
# ' _ {}

move_if_needed "openscene-v1.1/meta_datas" "trainval_navsim_logs"
move_if_needed "openscene-v1.1/sensor_blobs" "trainval_sensor_blobs"
rmdir openscene-v1.1 2>/dev/null || true
