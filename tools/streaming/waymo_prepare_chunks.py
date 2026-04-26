#!/usr/bin/env python3
"""Prepare Waymo E2E preprocessing chunks locally and optionally upload them.

This is the storage-box side of the workflow:

  1. Read the shared LMDB to discover processable sample tokens.
  2. Group samples into chunks capped by --target-extracted-gb.
  3. Scan local tfrecords once and route each required frame into its chunk.
  4. Archive each chunk as a tarball containing:
       chunk_00000/
         training_images/
         sample_ids.json
         frame_tokens.json
         metadata.json
  5. Optionally archive the shared LMDB once and upload all artifacts to S3.

The cloud GPU can then download prepared chunk tarballs instead of repeatedly
scanning Waymo's striped tfrecords.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
import yaml
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


CAMERA_COUNT = 8
CAMERA_MAPPING = {
    1: "front",
    2: "front_left",
    3: "front_right",
    4: "left",
    5: "right",
    6: "back_left",
    7: "back",
    8: "back_right",
}


@dataclass
class SequencePlan:
    sequence: str
    sample_tokens: List[str]
    frame_tokens: Set[str]

    @property
    def sample_count(self) -> int:
        return len(self.sample_tokens)


@dataclass
class ChunkPlan:
    index: int
    sequences: List[str]
    sample_tokens: List[str]
    frame_tokens: Set[str]
    estimated_bytes: int

    @property
    def sample_count(self) -> int:
        return len(self.sample_tokens)

    @property
    def frame_count(self) -> int:
        return len(self.frame_tokens)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="Config name under config/ or a direct YAML path.")
    p.add_argument("--dataset-path", type=Path, required=True,
                   help="Dataset root containing {split}_lmdb.")
    p.add_argument("--local-tfrecord-dir", type=Path, required=True,
                   help="Directory containing local Waymo tfrecords.")
    p.add_argument("--keys-file", type=Path, default=None,
                   help="Optional S3-key list; local files are matched by basename.")
    p.add_argument("--prepared-dir", type=Path, required=True,
                   help="Directory where chunk folders are built.")
    p.add_argument("--archive-dir", type=Path, required=True,
                   help="Directory where chunk tarballs and manifest are written.")
    p.add_argument("--target-extracted-gb", type=float, default=30.0,
                   help="Approximate extracted JPEG budget per chunk.")
    p.add_argument("--bytes-per-image", type=int, default=110_000,
                   help="Estimated average JPEG size for planning.")
    p.add_argument("--max-samples-per-chunk", type=int, default=None)
    p.add_argument("--max-frame-tokens-per-chunk", type=int, default=None)
    p.add_argument("--archive-format", choices=["tar", "tar.gz"], default="tar")
    p.add_argument("--upload-prefix", default=None,
                   help="Optional S3 prefix like s3://bucket/path/to/chunks.")
    p.add_argument("--archive-lmdb", action="store_true",
                   help="Also archive/upload {split}_lmdb once for the cloud runner.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the chunk plan and exit before extracting.")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing prepared/archive contents before writing.")
    p.add_argument("--keep-prepared-dirs", action="store_true",
                   help="Keep unarchived chunk directories after tarballs are created.")
    p.add_argument("--allow-missing-frames", action="store_true",
                   help="Create archives even if requested frames are missing.")
    return p.parse_args()


def resolve_config_path(config_name: str) -> str:
    candidate = os.path.expanduser(config_name)
    if os.path.isfile(candidate):
        return candidate
    return f"./config/{config_name}.yaml"


def load_config(name: str) -> dict:
    with open(resolve_config_path(name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI must include bucket and key prefix: {uri}")
    return bucket, key.rstrip("/")


def upload_file(local_path: Path, s3_uri: str):
    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    print(f"upload {local_path} -> s3://{bucket}/{key}", flush=True)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def load_key_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def list_tfrecord_keys(args, split: str) -> List[str]:
    if args.keys_file is not None:
        return load_key_lines(args.keys_file)
    pattern = f"{split}_*.tfrecord-*"
    exact = re.compile(rf"^{re.escape(split)}_.*\.tfrecord-\d{{5}}-of-\d{{5}}$")
    return [
        p.name for p in sorted(args.local_tfrecord_dir.glob(pattern))
        if exact.match(p.name)
    ]


def local_source_for_key(local_dir: Path, key: str) -> Path:
    direct = local_dir / Path(key).name
    if direct.exists():
        return direct
    nested = local_dir / key
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Local tfrecord not found for {key} under {local_dir}")


def parse_token(token: str):
    try:
        sequence, frame = token.rsplit("-", 1)
        return sequence, int(frame)
    except ValueError:
        return None, None


def read_lmdb_tokens(lmdb_dir: Path) -> List[str]:
    import lmdb

    if not lmdb_dir.exists():
        raise SystemExit(f"LMDB not found at {lmdb_dir}")

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            return [
                key.decode("utf-8")
                for key in cursor.iternext(keys=True, values=False)
            ]
    finally:
        env.close()


def build_sequence_frames(tokens: Sequence[str]) -> Dict[str, List[int]]:
    sequences: Dict[str, Set[int]] = {}
    for token in tokens:
        sequence, frame = parse_token(token)
        if sequence is None:
            continue
        sequences.setdefault(sequence, set()).add(frame)
    return {seq: sorted(frames) for seq, frames in sequences.items()}


def stop_frame_for_split(config: dict, max_frame: int, num_fut_frames: int) -> int:
    split = config["dataset_split"]
    if split == "training":
        return max_frame + 1
    if split in ("val", "test"):
        return max_frame + 1 - num_fut_frames
    raise ValueError(f"Invalid dataset_split '{split}'.")


def build_sequence_plans(config: dict, lmdb_dir: Path) -> List[SequencePlan]:
    tokens = read_lmdb_tokens(lmdb_dir)
    sequence_frames = build_sequence_frames(tokens)

    frequency_ratio = int(config["raw_images_freq"] / config["model_freq"])
    num_history_frames = frequency_ratio * (config["model_his_frames"] - 1) + 1
    num_fut_frames = frequency_ratio * config["model_fut_frames"]
    frame_offsets = list(range(num_history_frames - 1, -1, -frequency_ratio))

    plans = []
    for sequence in sorted(sequence_frames):
        frames = sequence_frames[sequence]
        if not frames:
            continue
        frame_set = set(frames)
        start = frames[0] + num_history_frames + config["frame_shift"] - 1
        stop = stop_frame_for_split(config, frames[-1], num_fut_frames)
        sample_tokens = []
        frame_tokens = set()
        for current_frame in range(start, stop, config["scene_frame_interval"]):
            history = [current_frame - offset for offset in frame_offsets]
            if any(frame not in frame_set for frame in history):
                continue
            sample_tokens.append(f"{sequence}-{current_frame:03d}")
            frame_tokens.update(f"{sequence}-{frame:03d}" for frame in history)
        if sample_tokens:
            plans.append(SequencePlan(sequence, sample_tokens, frame_tokens))
    return plans


def estimate_bytes(frame_count: int, bytes_per_image: int) -> int:
    return frame_count * CAMERA_COUNT * bytes_per_image


def build_chunks(args, plans: Sequence[SequencePlan]) -> List[ChunkPlan]:
    target_bytes = int(args.target_extracted_gb * 1024 ** 3)
    chunks = []
    current_sequences = []
    current_samples = []
    current_frames = set()

    def flush():
        if not current_sequences:
            return
        chunks.append(
            ChunkPlan(
                index=len(chunks),
                sequences=list(current_sequences),
                sample_tokens=list(current_samples),
                frame_tokens=set(current_frames),
                estimated_bytes=estimate_bytes(len(current_frames), args.bytes_per_image),
            )
        )
        current_sequences.clear()
        current_samples.clear()
        current_frames.clear()

    for plan in plans:
        next_frame_count = len(current_frames | plan.frame_tokens)
        next_sample_count = len(current_samples) + plan.sample_count
        next_estimate = estimate_bytes(next_frame_count, args.bytes_per_image)
        over_bytes = current_sequences and next_estimate > target_bytes
        over_samples = (
            args.max_samples_per_chunk is not None
            and current_sequences
            and next_sample_count > args.max_samples_per_chunk
        )
        over_frames = (
            args.max_frame_tokens_per_chunk is not None
            and current_sequences
            and next_frame_count > args.max_frame_tokens_per_chunk
        )
        if over_bytes or over_samples or over_frames:
            flush()

        current_sequences.append(plan.sequence)
        current_samples.extend(plan.sample_tokens)
        current_frames.update(plan.frame_tokens)

    flush()
    return chunks


def print_plan(chunks: Sequence[ChunkPlan]):
    total_samples = sum(c.sample_count for c in chunks)
    total_frames = sum(c.frame_count for c in chunks)
    total_est_gib = sum(c.estimated_bytes for c in chunks) / 1024 ** 3
    print(
        f"Plan: {len(chunks)} chunks, {total_samples} samples, "
        f"{total_frames} required frame tokens, ~{total_est_gib:.1f} GiB estimated JPEGs."
    )
    for chunk in chunks:
        print(
            f"chunk {chunk.index}: {len(chunk.sequences)} sequences, "
            f"{chunk.sample_count} samples, {chunk.frame_count} frames, "
            f"~{chunk.estimated_bytes / 1024 ** 3:.1f} GiB"
        )


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def prepare_chunk_dirs(args, chunks: Sequence[ChunkPlan], split: str):
    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        root = args.prepared_dir / f"chunk_{chunk.index:05d}"
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{split}_images").mkdir(parents=True, exist_ok=True)
        write_json(root / "sample_ids.json", {"tokens": sorted(chunk.sample_tokens)})
        write_json(root / "frame_tokens.json", {"tokens": sorted(chunk.frame_tokens)})
        write_json(
            root / "metadata.json",
            {
                "index": chunk.index,
                "split": split,
                "sequence_count": len(chunk.sequences),
                "sample_count": chunk.sample_count,
                "frame_token_count": chunk.frame_count,
                "estimated_bytes": chunk.estimated_bytes,
            },
        )


def write_frame_images(frame, token: str, images_dir: Path) -> int:
    sequence = token.rsplit("-", 1)[0]
    written_bytes = 0
    for image_content in frame.frame.images:
        camera = CAMERA_MAPPING.get(image_content.name)
        if camera is None:
            continue
        out_path = images_dir / sequence / camera / f"{token}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        image_bytes = image_content.image
        tmp.write_bytes(image_bytes)
        tmp.replace(out_path)
        written_bytes += len(image_bytes)
    return written_bytes


def extract_local_tfrecords(args, chunks: Sequence[ChunkPlan], keys: Sequence[str], split: str):
    frame_to_chunk = {}
    for chunk in chunks:
        for token in chunk.frame_tokens:
            frame_to_chunk[token] = chunk.index

    found = set()
    actual_image_bytes = {chunk.index: 0 for chunk in chunks}
    total_targets = len(frame_to_chunk)
    print(f"Extracting {total_targets} required frame tokens from {len(keys)} tfrecords.", flush=True)

    for i, key in enumerate(keys, 1):
        local_path = local_source_for_key(args.local_tfrecord_dir, key)
        matched = 0
        for raw in tf.data.TFRecordDataset(str(local_path), compression_type=""):
            frame = wod_e2ed_pb2.E2EDFrame()
            frame.ParseFromString(raw.numpy())
            token = frame.frame.context.name
            chunk_index = frame_to_chunk.get(token)
            if chunk_index is None or token in found:
                continue
            images_dir = args.prepared_dir / f"chunk_{chunk_index:05d}" / f"{split}_images"
            actual_image_bytes[chunk_index] += write_frame_images(frame, token, images_dir)
            found.add(token)
            matched += 1
        print(
            f"[{i}/{len(keys)}] {Path(key).name}: +{matched} frames "
            f"({len(found)}/{total_targets})",
            flush=True,
        )

    missing = set(frame_to_chunk) - found
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        msg = f"Missing {len(missing)} requested frame tokens. Examples: {preview}"
        if not args.allow_missing_frames:
            raise SystemExit(msg)
        print("WARNING:", msg, flush=True)

    for chunk in chunks:
        metadata_path = args.prepared_dir / f"chunk_{chunk.index:05d}" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["actual_image_bytes"] = actual_image_bytes[chunk.index]
        metadata["missing_frame_count"] = sum(1 for token in missing if frame_to_chunk[token] == chunk.index)
        write_json(metadata_path, metadata)

    return actual_image_bytes, missing


def archive_path_for(args, name: str) -> Path:
    suffix = ".tar.gz" if args.archive_format == "tar.gz" else ".tar"
    return args.archive_dir / f"{name}{suffix}"


def add_tree_to_tar(tar, path: Path, arcname: str):
    tar.add(path, arcname=arcname, recursive=True)


def archive_dir(path: Path, archive_path: Path, arcname: str, fmt: str):
    mode = "w:gz" if fmt == "tar.gz" else "w"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode) as tar:
        add_tree_to_tar(tar, path, arcname)
    return archive_path


def archive_chunks(args, chunks: Sequence[ChunkPlan]):
    archives = []
    for chunk in chunks:
        root = args.prepared_dir / f"chunk_{chunk.index:05d}"
        archive_path = archive_path_for(args, f"waymo_chunk_{chunk.index:05d}")
        print(f"archive {root} -> {archive_path}", flush=True)
        archive_dir(root, archive_path, root.name, args.archive_format)
        archives.append(archive_path)
        if not args.keep_prepared_dirs:
            shutil.rmtree(root)
    return archives


def maybe_archive_lmdb(args, split: str, lmdb_dir: Path):
    if not args.archive_lmdb:
        return None
    archive_path = archive_path_for(args, f"{split}_lmdb")
    print(f"archive {lmdb_dir} -> {archive_path}", flush=True)
    archive_dir(lmdb_dir, archive_path, f"{split}_lmdb", args.archive_format)
    return archive_path


def make_uri(local_path: Path, upload_prefix: str | None) -> str:
    if upload_prefix is None:
        return str(local_path)
    return upload_prefix.rstrip("/") + "/" + local_path.name


def main():
    args = parse_args()
    config = load_config(args.config)
    config["dataset_path"] = str(args.dataset_path)
    split = config["dataset_split"]
    lmdb_dir = args.dataset_path / f"{split}_lmdb"

    if args.overwrite:
        shutil.rmtree(args.prepared_dir, ignore_errors=True)
        shutil.rmtree(args.archive_dir, ignore_errors=True)
    elif (
        (args.prepared_dir.exists() and any(args.prepared_dir.iterdir()))
        or (args.archive_dir.exists() and any(args.archive_dir.iterdir()))
    ):
        raise SystemExit(
            "Prepared/archive directory is not empty. Pass --overwrite or choose new dirs."
        )

    keys = list_tfrecord_keys(args, split)
    if not keys:
        raise SystemExit(f"No local tfrecords found in {args.local_tfrecord_dir}")

    print(f"Loading sample plan from LMDB: {lmdb_dir}", flush=True)
    sequence_plans = build_sequence_plans(config, lmdb_dir)
    chunks = build_chunks(args, sequence_plans)
    print_plan(chunks)
    print(f"Local tfrecords to scan once: {len(keys)}")

    if args.dry_run:
        return

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    prepare_chunk_dirs(args, chunks, split)
    actual_image_bytes, missing = extract_local_tfrecords(args, chunks, keys, split)
    archives = archive_chunks(args, chunks)
    lmdb_archive = maybe_archive_lmdb(args, split, lmdb_dir)

    manifest = {
        "format": "autovla_waymo_prepared_chunks_v1",
        "split": split,
        "config": args.config,
        "dataset_name": config.get("dataset_name", "waymo"),
        "target_extracted_gb": args.target_extracted_gb,
        "bytes_per_image": args.bytes_per_image,
        "missing_frame_count": len(missing),
        "lmdb_archive": make_uri(lmdb_archive, args.upload_prefix) if lmdb_archive else None,
        "chunks": [],
    }
    for chunk, archive_path in zip(chunks, archives):
        manifest["chunks"].append(
            {
                "index": chunk.index,
                "archive": make_uri(archive_path, args.upload_prefix),
                "archive_name": archive_path.name,
                "sequence_count": len(chunk.sequences),
                "sample_count": chunk.sample_count,
                "frame_token_count": chunk.frame_count,
                "estimated_bytes": chunk.estimated_bytes,
                "actual_image_bytes": actual_image_bytes[chunk.index],
            }
        )

    manifest_path = args.archive_dir / "prepared_chunks_manifest.json"
    write_json(manifest_path, manifest)

    if args.upload_prefix is not None:
        for archive_path in archives:
            upload_file(archive_path, make_uri(archive_path, args.upload_prefix))
        if lmdb_archive is not None:
            upload_file(lmdb_archive, make_uri(lmdb_archive, args.upload_prefix))
        upload_file(manifest_path, make_uri(manifest_path, args.upload_prefix))

    print(f"Prepared manifest: {manifest_path}")
    if args.upload_prefix is not None:
        print(f"Uploaded manifest: {make_uri(manifest_path, args.upload_prefix)}")


if __name__ == "__main__":
    main()
