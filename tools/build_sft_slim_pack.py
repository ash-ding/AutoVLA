#!/usr/bin/env python3
"""Build a slim AutoVLA SFT pack from preprocessed JSONs and OpenScene tgz shards.

The sampler mirrors tools/run_sft.py for a single JSON directory:

    sorted(json_dir.glob("*.json"))
    torch.randperm(len(dataset), seed=seed)[:sample_size]

By default, copied JSONs are rewritten so camera paths become relative to the
pack's sensor root. This avoids the current SFT loader issue where absolute
JSON image paths ignore `sensor_data_path` in os.path.join().
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Iterable

import torch


NUPLAN_CAMERA_KEYS = (
    "front_camera_paths",
    "left_camera_paths",
    "right_camera_paths",
)

OTHER_CAMERA_KEYS = (
    "front_camera_paths",
    "front_left_camera_paths",
    "front_right_camera_paths",
)

ALL_CAMERA_KEYS = (
    "front_camera_paths",
    "front_left_camera_paths",
    "front_right_camera_paths",
    "left_camera_paths",
    "right_camera_paths",
    "back_camera_paths",
    "back_left_camera_paths",
    "back_right_camera_paths",
)

DEFAULT_STRIP_PREFIXES = (
    "/data/trainval_sensor_blobs/trainval/",
    "trainval_sensor_blobs/trainval/",
)

DEFAULT_TAR_MEMBER_PREFIX = "openscene-v1.1/sensor_blobs/trainval/"
SENSOR_MARKER = "sensor_blobs/trainval/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample SFT JSONs, collect referenced image paths, and build a slim data pack."
    )
    parser.add_argument("--json-dir", required=True, type=Path, help="Directory containing preprocessed JSON files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output slim-pack directory.")
    parser.add_argument("--sample-size", type=int, default=10_000, help="Number of JSON samples to select.")
    parser.add_argument("--seed", type=int, default=42, help="Seed matching tools/run_sft.py default.")
    parser.add_argument(
        "--camera-profile",
        choices=("auto", "nuplan", "other"),
        default="auto",
        help="Which camera keys to include for the training manifest.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=list(DEFAULT_STRIP_PREFIXES),
        help="Prefix to strip from JSON camera paths. Can be passed multiple times.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Directory containing OpenScene .tgz shards. Required for --extract-archives.",
    )
    parser.add_argument("--archive-glob", default="openscene_sensor_trainval_camera_*.tgz")
    parser.add_argument("--tar-member-prefix", default=DEFAULT_TAR_MEMBER_PREFIX)
    parser.add_argument(
        "--extract-archives",
        action="store_true",
        help="Scan tgz shards and extract only referenced image files into the slim pack.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Create a .tar.zst archive next to output-dir after building the pack.",
    )
    parser.add_argument(
        "--pack-name",
        default=None,
        help="Archive filename for --compress. Defaults to '<output-dir-name>.tar.zst'.",
    )
    parser.add_argument(
        "--no-rewrite-json-paths",
        action="store_true",
        help="Copy sampled JSONs without normalizing camera paths.",
    )
    return parser.parse_args()


def normalize_sensor_path(raw_path: str, strip_prefixes: Iterable[str]) -> str:
    """Return a sensor path relative to sensor_blobs/trainval."""
    path = raw_path.replace("\\", "/")

    for prefix in strip_prefixes:
        prefix = prefix.replace("\\", "/")
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    else:
        if SENSOR_MARKER in path:
            path = path.split(SENSOR_MARKER, 1)[1]
        elif path.startswith("/"):
            # Last-resort fallback: keep it relative so os.path.join uses sensor_data_path.
            path = path.lstrip("/")

    path = path.lstrip("/")
    parts = Path(path).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"Refusing path traversal in camera path: {raw_path!r}")
    return path


def camera_keys_for_sample(sample: dict, profile: str) -> tuple[str, ...]:
    if profile == "nuplan":
        return NUPLAN_CAMERA_KEYS
    if profile == "other":
        return OTHER_CAMERA_KEYS

    dataset_name = sample.get("dataset_name")
    if dataset_name == "nuplan":
        keys = list(NUPLAN_CAMERA_KEYS)
        # Some preprocessed nuPlan files also carry front_left/front_right names.
        # Use them only if the current loader's left/right keys are absent.
        if not sample.get("left_camera_paths"):
            keys[1] = "front_left_camera_paths"
        if not sample.get("right_camera_paths"):
            keys[2] = "front_right_camera_paths"
        return tuple(keys)
    return OTHER_CAMERA_KEYS


def select_jsons(json_dir: Path, sample_size: int, seed: int) -> list[Path]:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No top-level JSON files found in {json_dir}")

    if sample_size >= len(json_files):
        return json_files

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(json_files), generator=generator)[:sample_size].tolist()
    return [json_files[i] for i in indices]


def rewrite_camera_paths(sample: dict, strip_prefixes: Iterable[str]) -> dict:
    for key in ALL_CAMERA_KEYS:
        values = sample.get(key)
        if isinstance(values, list):
            sample[key] = [
                normalize_sensor_path(value, strip_prefixes) if isinstance(value, str) else value
                for value in values
            ]
    return sample


def copy_jsons_and_build_manifest(
    selected_jsons: list[Path],
    output_json_dir: Path,
    strip_prefixes: Iterable[str],
    camera_profile: str,
    rewrite_json_paths: bool,
) -> set[str]:
    image_paths: set[str] = set()
    output_json_dir.mkdir(parents=True, exist_ok=True)

    for src in selected_jsons:
        with src.open("r") as f:
            sample = json.load(f)

        for key in camera_keys_for_sample(sample, camera_profile):
            values = sample.get(key)
            if not values:
                raise KeyError(f"{src} is missing required camera key {key!r}")
            for value in values[:4]:
                image_paths.add(normalize_sensor_path(value, strip_prefixes))

        if rewrite_json_paths:
            sample = rewrite_camera_paths(sample, strip_prefixes)
            with (output_json_dir / src.name).open("w") as f:
                json.dump(sample, f, indent=2)
                f.write("\n")
        else:
            shutil.copy2(src, output_json_dir / src.name)

    return image_paths


def write_manifests(
    manifest_dir: Path,
    selected_jsons: list[Path],
    image_paths: set[str],
    tar_member_prefix: str,
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "selected_jsons.txt").write_text(
        "".join(f"{path}\n" for path in selected_jsons)
    )
    (manifest_dir / "image_manifest_relative.txt").write_text(
        "".join(f"{path}\n" for path in sorted(image_paths))
    )
    (manifest_dir / "image_manifest_tar_members.txt").write_text(
        "".join(f"{tar_member_prefix}{path}\n" for path in sorted(image_paths))
    )


def normalize_tar_member(member_name: str) -> str | None:
    name = member_name.replace("\\", "/")
    if SENSOR_MARKER not in name:
        return None
    rel = name.split(SENSOR_MARKER, 1)[1].lstrip("/")
    if not rel or rel.endswith("/"):
        return None
    return rel


def extract_referenced_images(
    archive_dir: Path,
    archive_glob: str,
    sensor_out_dir: Path,
    image_paths: set[str],
) -> set[str]:
    if not archive_dir:
        raise ValueError("--archive-dir is required with --extract-archives")

    archives = sorted(archive_dir.glob(archive_glob))
    if not archives:
        raise FileNotFoundError(f"No archives matching {archive_glob!r} under {archive_dir}")

    sensor_out_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    remaining = set(image_paths)

    for idx, archive in enumerate(archives, start=1):
        start = time.time()
        archive_found = 0
        print(f"[{idx:03d}/{len(archives):03d}] scanning {archive.name} ({len(remaining)} remaining)")

        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                rel = normalize_tar_member(member.name)
                if rel not in remaining:
                    continue

                out_path = sensor_out_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                tmp_path = out_path.with_name(f".{out_path.name}.tmp")
                with src, tmp_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                os.replace(tmp_path, out_path)

                remaining.remove(rel)
                found.add(rel)
                archive_found += 1

        elapsed = time.time() - start
        print(f"    extracted {archive_found} image(s) in {elapsed:.1f}s")
        if not remaining:
            break

    if remaining:
        print(f"WARNING: {len(remaining)} referenced image(s) were not found in archives.")
    return found


def compress_pack(output_dir: Path, pack_name: str | None) -> Path:
    archive_path = output_dir.parent / (pack_name or f"{output_dir.name}.tar.zst")
    cmd = [
        "tar",
        "-I",
        "zstd -T0 -3",
        "-cf",
        str(archive_path),
        "-C",
        str(output_dir.parent),
        output_dir.name,
    ]
    print("compressing:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return archive_path


def main() -> None:
    args = parse_args()

    selected = select_jsons(args.json_dir, args.sample_size, args.seed)
    rewrite_json_paths = not args.no_rewrite_json_paths

    output_json_dir = args.output_dir / "json" / args.json_dir.name
    manifest_dir = args.output_dir / "manifests"
    sensor_out_dir = args.output_dir / "sensor_blobs" / "trainval"

    image_paths = copy_jsons_and_build_manifest(
        selected_jsons=selected,
        output_json_dir=output_json_dir,
        strip_prefixes=args.strip_prefix,
        camera_profile=args.camera_profile,
        rewrite_json_paths=rewrite_json_paths,
    )
    write_manifests(manifest_dir, selected, image_paths, args.tar_member_prefix)

    print(f"selected_jsons: {len(selected)}")
    print(f"referenced_images: {len(image_paths)}")
    print(f"json_output: {output_json_dir}")
    print(f"manifest_output: {manifest_dir}")

    if args.extract_archives:
        found = extract_referenced_images(
            archive_dir=args.archive_dir,
            archive_glob=args.archive_glob,
            sensor_out_dir=sensor_out_dir,
            image_paths=image_paths,
        )
        (manifest_dir / "image_manifest_found.txt").write_text(
            "".join(f"{path}\n" for path in sorted(found))
        )
        print(f"sensor_output: {sensor_out_dir}")
        print(f"found_images: {len(found)} / {len(image_paths)}")

    if args.compress:
        archive_path = compress_pack(args.output_dir, args.pack_name)
        print(f"archive_output: {archive_path}")


if __name__ == "__main__":
    main()
