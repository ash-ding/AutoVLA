#!/usr/bin/env bash

set -euo pipefail

STATE_DIR=".navsim_download_state/trainval"
ARCHIVE_DIR="${STATE_DIR}/archives"
DOWNLOADED_DIR="${STATE_DIR}/downloaded"
EXTRACTED_DIR="${STATE_DIR}/extracted"

if ! command -v aria2c >/dev/null 2>&1; then
    echo "aria2c is required but not installed. See https://aria2.github.io/" >&2
    exit 1
fi

mkdir -p "${ARCHIVE_DIR}" "${DOWNLOADED_DIR}" "${EXTRACTED_DIR}"

download_and_extract() {
    local url="$1"
    local archive_name="${url##*/}"
    local archive_path="${ARCHIVE_DIR}/${archive_name}"
    local control_path="${archive_path}.aria2"
    local downloaded_marker="${DOWNLOADED_DIR}/${archive_name}.done"
    local extracted_marker="${EXTRACTED_DIR}/${archive_name}.done"

    if [ -f "${extracted_marker}" ]; then
        echo "Skipping ${archive_name}; already downloaded and extracted."
        return
    fi

    if [ -f "${downloaded_marker}" ] && [ ! -f "${archive_path}" ]; then
        echo "Completed download marker found for ${archive_name}, but archive is missing. Downloading again..."
        rm -f "${downloaded_marker}"
    fi

    if [ ! -f "${downloaded_marker}" ]; then
        # Any archive without a completed-download marker is treated as partial and re-downloaded.
        rm -f "${archive_path}" "${control_path}"

        if ! aria2c \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --continue=false \
            --summary-interval=0 \
            --console-log-level=warn \
            --dir="${ARCHIVE_DIR}" \
            --out="${archive_name}" \
            "${url}"; then
            echo "Download of ${archive_name} failed; removing partial archive so it will be retried." >&2
            rm -f "${archive_path}" "${control_path}" "${downloaded_marker}"
            return 1
        fi

        # Verify the archive is a complete, readable gzip tarball before marking it as downloaded.
        if ! tar -tzf "${archive_path}" >/dev/null 2>&1; then
            echo "Archive ${archive_name} is corrupt or truncated; removing so it will be retried." >&2
            rm -f "${archive_path}" "${control_path}" "${downloaded_marker}"
            return 1
        fi

        touch "${downloaded_marker}"
    else
        echo "Using previously completed archive ${archive_name} for extraction..."
    fi

    if ! tar -xzf "${archive_path}"; then
        echo "Extraction of ${archive_name} failed; discarding archive so it will be re-downloaded." >&2
        rm -f "${archive_path}" "${control_path}" "${downloaded_marker}"
        return 1
    fi
    rm -f "${archive_path}" "${control_path}"
    touch "${extracted_marker}"
}

export -f download_and_extract
export STATE_DIR ARCHIVE_DIR DOWNLOADED_DIR EXTRACTED_DIR

merge_tree() {
    local src_dir="$1"
    local dest_dir="$2"
    local src_path
    local name
    local dest_path
    local had_dotglob="off"
    local had_nullglob="off"

    if [ ! -d "${src_dir}" ]; then
        return
    fi

    mkdir -p "${dest_dir}"

    if shopt -q dotglob; then
        had_dotglob="on"
    fi
    if shopt -q nullglob; then
        had_nullglob="on"
    fi
    shopt -s dotglob nullglob

    for src_path in "${src_dir}"/*; do
        name="$(basename "${src_path}")"
        dest_path="${dest_dir}/${name}"

        if [ -d "${src_path}" ] && [ -d "${dest_path}" ]; then
            merge_tree "${src_path}" "${dest_path}"
            rmdir "${src_path}" 2>/dev/null || true
        else
            mv -f "${src_path}" "${dest_path}"
        fi
    done

    if [ "${had_dotglob}" = "off" ]; then
        shopt -u dotglob
    fi
    if [ "${had_nullglob}" = "off" ]; then
        shopt -u nullglob
    fi

    rmdir "${src_dir}" 2>/dev/null || true
}

if [ -d trainval_navsim_logs ] || [ -d openscene-v1.1/meta_datas ]; then
    touch "${EXTRACTED_DIR}/openscene_metadata_trainval.tgz.done"
fi

if [ -d trainval_navsim_logs ] && [ -d trainval_sensor_blobs ] && \
    [ "$(find "${EXTRACTED_DIR}" -maxdepth 1 -name '*.done' | wc -l | tr -d '[:space:]')" = "201" ]; then
    echo "All tracked trainval archives are already extracted; nothing to do."
    exit 0
fi

if [ -d trainval_navsim_logs ] || [ -d openscene-v1.1/meta_datas ]; then
    echo "Skipping openscene_metadata_trainval.tgz; metadata is already available."
else
    download_and_extract "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz"
fi

printf "%s\n" {0..199} | xargs -I {} -P 1 bash -c '
    set -euo pipefail
    split="$1"
    echo "Processing camera split ${split}..."
    download_and_extract "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_${split}.tgz"
' _ {}

# printf "%s\n" {0..199} | xargs -I {} -P 8 bash -c '
#     split="$1"
#     echo "Processing lidar trainval split ${split}..."
#     download_and_extract "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_lidar/openscene_sensor_trainval_lidar_${split}.tgz"
# ' _ {}

merge_tree "openscene-v1.1/meta_datas" "trainval_navsim_logs"
merge_tree "openscene-v1.1/sensor_blobs" "trainval_sensor_blobs"
rmdir openscene-v1.1 2>/dev/null || true
