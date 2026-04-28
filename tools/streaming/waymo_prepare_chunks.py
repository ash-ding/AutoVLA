#!/usr/bin/env python3
"""Prepare Waymo E2E preprocessing chunks locally and optionally upload them.

This is the storage-box side of the workflow:

  1. Read the shared LMDB, or scan local tfrecords, to discover processable
     sample tokens.
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
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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

WORKER_LOCAL_TFRECORD_DIR = None
WORKER_PREPARED_DIR = None
WORKER_SPLIT = None
WORKER_FRAME_TOKENS = None
WORKER_FRAME_TO_CHUNK = None
WORKER_LMDB_TOKENS = None


@dataclass
class SequencePlan:
    sequence: str
    sample_tokens: List[str]
    frame_tokens: Set[str]
    planned_bytes: int = 0

    @property
    def sample_count(self) -> int:
        return len(self.sample_tokens)

    @property
    def frame_count(self) -> int:
        return len(self.frame_tokens)


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
    p.add_argument("--plan-from-tfrecords", action="store_true",
                   help="Scan local tfrecords for frame tokens instead of requiring {split}_lmdb.")
    p.add_argument("--keys-file", type=Path, default=None,
                   help="Optional S3-key list; local files are matched by basename.")
    p.add_argument("--token-to-tfrecord-json", type=Path, default=None,
                   help=(
                       "Where to write a persistent frame-token to tfrecord-key map. "
                       "Defaults to {dataset_path}/{split}_token_to_tfrecord.json."
                   ))
    p.add_argument("--prepared-dir", type=Path, required=True,
                   help="Directory where chunk folders are built.")
    p.add_argument("--archive-dir", type=Path, required=True,
                   help="Directory where chunk tarballs and manifest are written.")
    p.add_argument("--target-extracted-gb", type=float, default=30.0,
                   help="Extracted JPEG budget per chunk. Estimated unless --hard-limit is set.")
    p.add_argument("--bytes-per-image", type=int, default=110_000,
                   help="Estimated average JPEG size for planning.")
    p.add_argument("--hard-limit", action="store_true",
                   help="Pre-scan local tfrecords to make --target-extracted-gb a hard JPEG-byte limit.")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Parallel local tfrecord workers for measurement/extraction. Start with 4-8.")
    p.add_argument("--max-samples-per-chunk", type=int, default=None)
    p.add_argument("--max-frame-tokens-per-chunk", type=int, default=None)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Build only the first N planned chunks. Useful for pilot runs.")
    p.add_argument("--early-stop-after-chunks", type=int, default=None,
                   help=(
                       "With --plan-from-tfrecords, stop scanning for the sample plan "
                       "as soon as this many estimated chunks can be built. Pilot mode; "
                       "does not produce the globally planned first chunks."
                   ))
    p.add_argument("--archive-format", choices=["tar", "tar.gz"], default="tar")
    p.add_argument("--upload-prefix", default=None,
                   help="Optional S3 prefix like s3://bucket/path/to/chunks.")
    p.add_argument("--archive-lmdb", action="store_true",
                   help="Also archive/upload {split}_lmdb once for the cloud runner.")
    p.add_argument("--build-scoped-lmdb", action="store_true",
                   help=(
                       "Build a compact {split}_lmdb containing only metadata records needed "
                       "by the selected chunks."
                   ))
    p.add_argument("--output-lmdb-dir", type=Path, default=None,
                   help=(
                       "Where --build-scoped-lmdb writes the compact LMDB. "
                       "Defaults to {archive_dir}/{split}_lmdb."
                   ))
    p.add_argument("--lmdb-map-size-gb", type=int, default=1500,
                   help="Virtual map size for --build-scoped-lmdb; not actual disk usage.")
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


def init_measure_worker(local_tfrecord_dir: str, frame_tokens: Set[str]):
    global WORKER_LOCAL_TFRECORD_DIR, WORKER_FRAME_TOKENS
    WORKER_LOCAL_TFRECORD_DIR = Path(local_tfrecord_dir)
    WORKER_FRAME_TOKENS = frame_tokens


def init_scan_worker(local_tfrecord_dir: str):
    global WORKER_LOCAL_TFRECORD_DIR
    WORKER_LOCAL_TFRECORD_DIR = Path(local_tfrecord_dir)


def init_extract_worker(
    local_tfrecord_dir: str,
    prepared_dir: str,
    split: str,
    frame_to_chunk: Dict[str, int],
    lmdb_tokens: Set[str] | None,
):
    global WORKER_LOCAL_TFRECORD_DIR, WORKER_PREPARED_DIR, WORKER_SPLIT, WORKER_FRAME_TO_CHUNK, WORKER_LMDB_TOKENS
    WORKER_LOCAL_TFRECORD_DIR = Path(local_tfrecord_dir)
    WORKER_PREPARED_DIR = Path(prepared_dir)
    WORKER_SPLIT = split
    WORKER_FRAME_TO_CHUNK = frame_to_chunk
    WORKER_LMDB_TOKENS = lmdb_tokens


def scan_tfrecord_worker(key: str):
    local_path = local_source_for_key(WORKER_LOCAL_TFRECORD_DIR, key)
    tokens = []
    token_to_tfrecord = {}
    for raw in tf.data.TFRecordDataset(str(local_path), compression_type=""):
        frame = wod_e2ed_pb2.E2EDFrame()
        frame.ParseFromString(raw.numpy())
        token = frame.frame.context.name
        tokens.append(token)
        token_to_tfrecord[token] = key
    return key, tokens, token_to_tfrecord


def measure_tfrecord_worker(key: str):
    local_path = local_source_for_key(WORKER_LOCAL_TFRECORD_DIR, key)
    frame_sizes = {}
    token_to_tfrecord = {}
    for raw in tf.data.TFRecordDataset(str(local_path), compression_type=""):
        frame = wod_e2ed_pb2.E2EDFrame()
        frame.ParseFromString(raw.numpy())
        token = frame.frame.context.name
        token_to_tfrecord[token] = key
        if token not in WORKER_FRAME_TOKENS:
            continue

        frame_sizes[token] = sum(
            len(image_content.image)
            for image_content in frame.frame.images
            if image_content.name in CAMERA_MAPPING
        )
    return key, frame_sizes, token_to_tfrecord


def extract_tfrecord_worker(key: str):
    local_path = local_source_for_key(WORKER_LOCAL_TFRECORD_DIR, key)
    found = set()
    actual_image_bytes = {}
    token_to_tfrecord = {}
    lmdb_records = {}
    matched = 0
    for raw in tf.data.TFRecordDataset(str(local_path), compression_type=""):
        frame = wod_e2ed_pb2.E2EDFrame()
        frame.ParseFromString(raw.numpy())
        token = frame.frame.context.name
        token_to_tfrecord[token] = key
        chunk_index = WORKER_FRAME_TO_CHUNK.get(token)
        needs_lmdb = WORKER_LMDB_TOKENS is not None and token in WORKER_LMDB_TOKENS
        if chunk_index is None and not needs_lmdb:
            continue

        if chunk_index is not None:
            images_dir = (
                WORKER_PREPARED_DIR
                / f"chunk_{chunk_index:05d}"
                / f"{WORKER_SPLIT}_images"
            )
            actual_image_bytes[chunk_index] = (
                actual_image_bytes.get(chunk_index, 0)
                + write_frame_images(frame, token, images_dir)
            )
            found.add(token)
            matched += 1

        if needs_lmdb:
            del frame.frame.images[:]
            lmdb_records[token] = frame.SerializeToString()
    return key, matched, found, actual_image_bytes, token_to_tfrecord, lmdb_records


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


def build_sequence_plans_from_tokens(config: dict, tokens: Sequence[str]) -> List[SequencePlan]:
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


def build_sequence_plans(config: dict, lmdb_dir: Path) -> List[SequencePlan]:
    tokens = read_lmdb_tokens(lmdb_dir)
    return build_sequence_plans_from_tokens(config, tokens)


def scan_tfrecord_tokens(
    args,
    keys: Sequence[str],
    config: dict | None = None,
) -> tuple[List[str], Dict[str, str], List[str]]:
    tokens = []
    token_to_tfrecord: Dict[str, str] = {}
    scanned_keys = []
    effective_workers = 1 if args.early_stop_after_chunks is not None else args.num_workers
    print(
        f"Scanning {len(keys)} local tfrecords for frame tokens with "
        f"{effective_workers} worker(s).",
        flush=True,
    )

    if args.early_stop_after_chunks is not None:
        if config is None:
            raise ValueError("config is required for early-stop planning")
        print(
            f"Early-stop planning enabled: scanning until "
            f"{args.early_stop_after_chunks} chunk(s) can be estimated.",
            flush=True,
        )
        init_scan_worker(str(args.local_tfrecord_dir))
        for i, key in enumerate(keys, 1):
            _, partial_tokens, partial_token_map = scan_tfrecord_worker(key)
            tokens.extend(partial_tokens)
            token_to_tfrecord.update(partial_token_map)
            scanned_keys.append(key)

            plans = build_sequence_plans_from_tokens(config, tokens)
            chunks = build_chunks(args, plans)
            print(
                f"[plan {i}/{len(keys)}] {Path(key).name}: "
                f"+{len(partial_tokens)} tokens ({len(tokens)} total), "
                f"{len(chunks)} candidate chunk(s)",
                flush=True,
            )
            if len(chunks) >= args.early_stop_after_chunks:
                print(
                    f"Stopping planning after {len(scanned_keys)} tfrecord(s): "
                    f"{len(chunks)} candidate chunk(s) available.",
                    flush=True,
                )
                break
    elif args.num_workers <= 1:
        init_scan_worker(str(args.local_tfrecord_dir))
        for i, key in enumerate(keys, 1):
            _, partial_tokens, partial_token_map = scan_tfrecord_worker(key)
            tokens.extend(partial_tokens)
            token_to_tfrecord.update(partial_token_map)
            scanned_keys.append(key)
            print(
                f"[plan {i}/{len(keys)}] {Path(key).name}: "
                f"+{len(partial_tokens)} tokens ({len(tokens)} total)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_scan_worker,
            initargs=(str(args.local_tfrecord_dir),),
        ) as pool:
            futures = {pool.submit(scan_tfrecord_worker, key): key for key in keys}
            completed = 0
            for fut in as_completed(futures):
                key, partial_tokens, partial_token_map = fut.result()
                completed += 1
                tokens.extend(partial_tokens)
                token_to_tfrecord.update(partial_token_map)
                scanned_keys.append(key)
                print(
                    f"[plan {completed}/{len(keys)}] {Path(key).name}: "
                    f"+{len(partial_tokens)} tokens ({len(tokens)} total)",
                    flush=True,
                )

    return tokens, token_to_tfrecord, scanned_keys


def estimate_bytes(frame_count: int, bytes_per_image: int) -> int:
    return frame_count * CAMERA_COUNT * bytes_per_image


def target_bytes(args) -> int:
    return int(args.target_extracted_gb * 1024 ** 3)


def measure_required_frame_bytes(
    args,
    keys: Sequence[str],
    frame_tokens: Set[str],
) -> tuple[Dict[str, int], Set[str], Dict[str, str]]:
    frame_sizes: Dict[str, int] = {}
    token_to_tfrecord: Dict[str, str] = {}
    print(
        f"Measuring actual JPEG bytes for {len(frame_tokens)} required frame tokens "
        f"from {len(keys)} tfrecords with {args.num_workers} worker(s).",
        flush=True,
    )

    if args.num_workers <= 1:
        init_measure_worker(str(args.local_tfrecord_dir), frame_tokens)
        for i, key in enumerate(keys, 1):
            _, partial_sizes, partial_token_map = measure_tfrecord_worker(key)
            frame_sizes.update(partial_sizes)
            token_to_tfrecord.update(partial_token_map)
            print(
                f"[measure {i}/{len(keys)}] {Path(key).name}: +{len(partial_sizes)} frames "
                f"({len(frame_sizes)}/{len(frame_tokens)})",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_measure_worker,
            initargs=(str(args.local_tfrecord_dir), frame_tokens),
        ) as pool:
            futures = {pool.submit(measure_tfrecord_worker, key): key for key in keys}
            completed = 0
            for fut in as_completed(futures):
                key, partial_sizes, partial_token_map = fut.result()
                completed += 1
                frame_sizes.update(partial_sizes)
                token_to_tfrecord.update(partial_token_map)
                print(
                    f"[measure {completed}/{len(keys)}] {Path(key).name}: "
                    f"+{len(partial_sizes)} frames ({len(frame_sizes)}/{len(frame_tokens)})",
                    flush=True,
                )

    missing = frame_tokens - set(frame_sizes)
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        msg = f"Missing {len(missing)} requested frame tokens during size measurement. Examples: {preview}"
        if not args.allow_missing_frames:
            raise SystemExit(msg)
        print("WARNING:", msg, flush=True)
    return frame_sizes, missing, token_to_tfrecord


def apply_actual_frame_sizes(plans: Sequence[SequencePlan], frame_sizes: Dict[str, int]):
    for plan in plans:
        plan.planned_bytes = sum(frame_sizes.get(token, 0) for token in plan.frame_tokens)


def build_chunks(args, plans: Sequence[SequencePlan]) -> List[ChunkPlan]:
    limit_bytes = target_bytes(args)
    chunks = []
    current_sequences = []
    current_samples = []
    current_frames = set()
    current_bytes = 0

    def flush():
        nonlocal current_bytes
        if not current_sequences:
            return
        chunks.append(
            ChunkPlan(
                index=len(chunks),
                sequences=list(current_sequences),
                sample_tokens=list(current_samples),
                frame_tokens=set(current_frames),
                estimated_bytes=current_bytes,
            )
        )
        current_sequences.clear()
        current_samples.clear()
        current_frames.clear()
        current_bytes = 0

    for plan in plans:
        plan_bytes = (
            plan.planned_bytes
            if args.hard_limit
            else estimate_bytes(plan.frame_count, args.bytes_per_image)
        )
        if plan_bytes > limit_bytes:
            raise SystemExit(
                f"Sequence {plan.sequence} needs {plan_bytes / 1024 ** 3:.2f} GiB, "
                f"which exceeds --target-extracted-gb={args.target_extracted_gb}. "
                "Use a larger target or add sequence-splitting support."
            )

        next_frame_count = len(current_frames | plan.frame_tokens)
        next_sample_count = len(current_samples) + plan.sample_count
        next_bytes = current_bytes + plan_bytes
        over_bytes = current_sequences and next_bytes > limit_bytes
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
        current_bytes += plan_bytes

    flush()
    return chunks


def print_plan(chunks: Sequence[ChunkPlan], hard_limit: bool):
    total_samples = sum(c.sample_count for c in chunks)
    total_frames = sum(c.frame_count for c in chunks)
    total_plan_gib = sum(c.estimated_bytes for c in chunks) / 1024 ** 3
    label = "actual JPEG bytes" if hard_limit else "estimated JPEGs"
    print(
        f"Plan: {len(chunks)} chunks, {total_samples} samples, "
        f"{total_frames} required frame tokens, ~{total_plan_gib:.1f} GiB {label}."
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


def write_token_to_tfrecord_json(
    path: Path,
    split: str,
    keys: Sequence[str],
    token_to_tfrecord: Dict[str, str],
):
    write_json(
        path,
        {
            "format": "autovla_waymo_token_to_tfrecord_v1",
            "split": split,
            "token_count": len(token_to_tfrecord),
            "tfrecord_count": len(keys),
            "token_to_tfrecord": dict(sorted(token_to_tfrecord.items())),
        },
    )
    print(
        f"Wrote token-to-tfrecord map: {path} "
        f"({len(token_to_tfrecord)} frame tokens across {len(keys)} tfrecords)",
        flush=True,
    )


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
                "planned_image_bytes": chunk.estimated_bytes,
                "size_plan": "actual" if args.hard_limit else "estimated",
                "target_extracted_bytes": target_bytes(args),
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


def write_lmdb_batch(env, records: Dict[str, bytes]):
    if not records:
        return
    with env.begin(write=True) as txn:
        for token, raw_bytes in records.items():
            txn.put(token.encode("utf-8"), raw_bytes)


def extract_local_tfrecords(
    args,
    chunks: Sequence[ChunkPlan],
    keys: Sequence[str],
    split: str,
    lmdb_tokens: Set[str] | None = None,
    output_lmdb_dir: Path | None = None,
):
    frame_to_chunk = {}
    for chunk in chunks:
        for token in chunk.frame_tokens:
            frame_to_chunk[token] = chunk.index

    found = set()
    lmdb_found = set()
    actual_image_bytes = {chunk.index: 0 for chunk in chunks}
    token_to_tfrecord: Dict[str, str] = {}
    total_targets = len(frame_to_chunk)
    lmdb_targets = lmdb_tokens or set()
    effective_workers = 1 if args.early_stop_after_chunks is not None else args.num_workers
    print(
        f"Extracting {total_targets} required frame tokens from {len(keys)} tfrecords "
        f"with {effective_workers} worker(s).",
        flush=True,
    )
    if args.early_stop_after_chunks is not None and args.num_workers > 1:
        print(
            "Early-stop extraction enabled: scanning tfrecords sequentially so the "
            "run can stop as soon as the selected chunks are complete.",
            flush=True,
        )
    if lmdb_tokens is not None:
        print(
            f"Writing scoped LMDB metadata for {len(lmdb_tokens)} frame tokens "
            f"to {output_lmdb_dir}.",
            flush=True,
        )

    lmdb_env = None
    if output_lmdb_dir is not None:
        import lmdb

        if output_lmdb_dir.exists() and any(output_lmdb_dir.iterdir()):
            if args.overwrite:
                shutil.rmtree(output_lmdb_dir)
            else:
                raise SystemExit(
                    f"Output LMDB directory is not empty. Pass --overwrite or choose a new path: "
                    f"{output_lmdb_dir}"
                )
        output_lmdb_dir.mkdir(parents=True, exist_ok=True)
        lmdb_env = lmdb.open(
            str(output_lmdb_dir),
            map_size=args.lmdb_map_size_gb * 1024 ** 3,
        )

    try:
        if args.num_workers <= 1 or args.early_stop_after_chunks is not None:
            init_extract_worker(
                str(args.local_tfrecord_dir),
                str(args.prepared_dir),
                split,
                frame_to_chunk,
                lmdb_tokens,
            )
            for i, key in enumerate(keys, 1):
                (
                    _,
                    matched,
                    partial_found,
                    partial_bytes,
                    partial_token_map,
                    partial_lmdb_records,
                ) = extract_tfrecord_worker(key)
                found.update(partial_found)
                token_to_tfrecord.update(partial_token_map)
                for chunk_index, byte_count in partial_bytes.items():
                    actual_image_bytes[chunk_index] += byte_count
                if lmdb_env is not None:
                    write_lmdb_batch(lmdb_env, partial_lmdb_records)
                lmdb_found.update(partial_lmdb_records)
                print(
                    f"[{i}/{len(keys)}] {Path(key).name}: +{matched} frames "
                    f"({len(found)}/{total_targets})",
                    flush=True,
                )
                if set(frame_to_chunk).issubset(found) and lmdb_targets.issubset(lmdb_found):
                    print(
                        f"Stopping extraction after {i} tfrecord(s): selected chunks are complete.",
                        flush=True,
                    )
                    break
        else:
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=init_extract_worker,
                initargs=(
                    str(args.local_tfrecord_dir),
                    str(args.prepared_dir),
                    split,
                    frame_to_chunk,
                    lmdb_tokens,
                ),
            ) as pool:
                futures = {pool.submit(extract_tfrecord_worker, key): key for key in keys}
                completed = 0
                for fut in as_completed(futures):
                    (
                        key,
                        matched,
                        partial_found,
                        partial_bytes,
                        partial_token_map,
                        partial_lmdb_records,
                    ) = fut.result()
                    completed += 1
                    found.update(partial_found)
                    token_to_tfrecord.update(partial_token_map)
                    for chunk_index, byte_count in partial_bytes.items():
                        actual_image_bytes[chunk_index] += byte_count
                    if lmdb_env is not None:
                        write_lmdb_batch(lmdb_env, partial_lmdb_records)
                    lmdb_found.update(partial_lmdb_records)
                    print(
                        f"[{completed}/{len(keys)}] {Path(key).name}: +{matched} frames "
                        f"({len(found)}/{total_targets})",
                        flush=True,
                    )
    finally:
        if lmdb_env is not None:
            lmdb_env.sync()
            lmdb_env.close()

    missing = set(frame_to_chunk) - found
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        msg = f"Missing {len(missing)} requested frame tokens. Examples: {preview}"
        if not args.allow_missing_frames:
            raise SystemExit(msg)
        print("WARNING:", msg, flush=True)

    missing_lmdb = lmdb_targets - lmdb_found
    if missing_lmdb:
        preview = ", ".join(sorted(missing_lmdb)[:10])
        msg = f"Missing {len(missing_lmdb)} requested scoped LMDB tokens. Examples: {preview}"
        if not args.allow_missing_frames:
            raise SystemExit(msg)
        print("WARNING:", msg, flush=True)

    for chunk in chunks:
        metadata_path = args.prepared_dir / f"chunk_{chunk.index:05d}" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["actual_image_bytes"] = actual_image_bytes[chunk.index]
        metadata["missing_frame_count"] = sum(1 for token in missing if frame_to_chunk[token] == chunk.index)
        write_json(metadata_path, metadata)

        if args.hard_limit and actual_image_bytes[chunk.index] > target_bytes(args):
            raise SystemExit(
                f"Chunk {chunk.index} exceeded hard limit after extraction: "
                f"{actual_image_bytes[chunk.index] / 1024 ** 3:.2f} GiB > "
                f"{args.target_extracted_gb:.2f} GiB"
            )

    return actual_image_bytes, missing, token_to_tfrecord


def archive_path_for(args, name: str) -> Path:
    suffix = ".tar.gz" if args.archive_format == "tar.gz" else ".tar"
    return args.archive_dir / f"{name}{suffix}"


def add_tree_to_tar(tar, path: Path, arcname: str):
    tar.add(path, arcname=arcname, recursive=True)


def archive_dir(path: Path, archive_path: Path, arcname: str, fmt: str):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tar_bin = shutil.which("tar")
    if tar_bin is not None:
        cmd = [tar_bin, "-C", str(path.parent)]
        cmd.append("-czf" if fmt == "tar.gz" else "-cf")
        cmd.extend([str(archive_path), arcname])
        subprocess.run(cmd, check=True)
        return archive_path

    mode = "w:gz" if fmt == "tar.gz" else "w"
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
    if lmdb_dir is None:
        raise SystemExit(
            "No LMDB directory is available to archive. Use an existing {split}_lmdb "
            "or pass --build-scoped-lmdb."
        )
    archive_path = archive_path_for(args, f"{split}_lmdb")
    print(f"archive {lmdb_dir} -> {archive_path}", flush=True)
    archive_dir(lmdb_dir, archive_path, f"{split}_lmdb", args.archive_format)
    return archive_path


def lmdb_tokens_for_chunks(config: dict, chunks: Sequence[ChunkPlan]) -> Set[str]:
    tokens = set().union(*(chunk.frame_tokens for chunk in chunks)) if chunks else set()
    if config["dataset_split"] == "test":
        for chunk in chunks:
            for sample_token in chunk.sample_tokens:
                base, frame = parse_token(sample_token)
                if base is None:
                    continue
                tokens.add(f"{base}-{frame + 35:03d}")
                tokens.add(f"{base}-{frame + 50:03d}")
    return tokens


def make_manifest_ref(local_path: Path, archive_dir: Path) -> str:
    try:
        return str(local_path.relative_to(archive_dir))
    except ValueError:
        return local_path.name


def make_dataset_ref(local_path: Path, dataset_path: Path) -> str:
    try:
        return str(local_path.relative_to(dataset_path))
    except ValueError:
        return local_path.name


def make_upload_uri(local_path: Path, upload_prefix: str | None) -> str:
    if upload_prefix is None:
        return str(local_path)
    return upload_prefix.rstrip("/") + "/" + local_path.name


def main():
    args = parse_args()
    config = load_config(args.config)
    config["dataset_path"] = str(args.dataset_path)
    split = config["dataset_split"]
    lmdb_dir = args.dataset_path / f"{split}_lmdb"
    if args.token_to_tfrecord_json is None:
        args.token_to_tfrecord_json = args.dataset_path / f"{split}_token_to_tfrecord.json"

    if args.early_stop_after_chunks is not None:
        if args.early_stop_after_chunks < 1:
            raise SystemExit("--early-stop-after-chunks must be >= 1")
        if not args.plan_from_tfrecords:
            raise SystemExit("--early-stop-after-chunks requires --plan-from-tfrecords")
        if args.hard_limit:
            raise SystemExit("--early-stop-after-chunks cannot be combined with --hard-limit")
        if args.max_chunks is None:
            args.max_chunks = args.early_stop_after_chunks
        elif args.max_chunks > args.early_stop_after_chunks:
            raise SystemExit(
                "--max-chunks cannot be greater than --early-stop-after-chunks"
            )

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

    planning_token_to_tfrecord: Dict[str, str] = {}
    planning_keys = keys
    if args.plan_from_tfrecords:
        print(
            f"Loading sample plan by scanning local tfrecords: {args.local_tfrecord_dir}",
            flush=True,
        )
        frame_tokens, planning_token_to_tfrecord, planning_keys = scan_tfrecord_tokens(
            args, keys, config
        )
        sequence_plans = build_sequence_plans_from_tokens(config, frame_tokens)
    else:
        print(f"Loading sample plan from LMDB: {lmdb_dir}", flush=True)
        sequence_plans = build_sequence_plans(config, lmdb_dir)
    size_measurement_missing = set()
    token_to_tfrecord: Dict[str, str] = dict(planning_token_to_tfrecord)
    if args.hard_limit:
        all_required_frames = set().union(*(plan.frame_tokens for plan in sequence_plans))
        frame_sizes, size_measurement_missing, token_to_tfrecord = measure_required_frame_bytes(
            args, keys, all_required_frames
        )
        write_token_to_tfrecord_json(args.token_to_tfrecord_json, split, keys, token_to_tfrecord)
        apply_actual_frame_sizes(sequence_plans, frame_sizes)
    chunks = build_chunks(args, sequence_plans)
    if args.max_chunks is not None:
        if args.max_chunks < 1:
            raise SystemExit("--max-chunks must be >= 1")
        total_chunks = len(chunks)
        chunks = chunks[:args.max_chunks]
        print(
            f"Limiting build to first {len(chunks)} of {total_chunks} planned chunks "
            f"(--max-chunks={args.max_chunks}).",
            flush=True,
        )
    print_plan(chunks, args.hard_limit)
    if args.hard_limit:
        print(f"Local tfrecords to scan twice: {len(keys)}")
        print(
            "Hard limit mode: the tfrecords are scanned once for exact JPEG sizes "
            "and once again for extraction.",
            flush=True,
        )
    else:
        if args.early_stop_after_chunks is not None:
            print(
                f"Local tfrecords scanned for planning: {len(planning_keys)}/{len(keys)}. "
                "Extraction will scan only this planned prefix and stop once complete.",
                flush=True,
            )
        else:
            print(f"Local tfrecords to scan once: {len(keys)}")

    if args.dry_run:
        return

    scoped_lmdb_dir = None
    scoped_lmdb_tokens = None
    if args.build_scoped_lmdb:
        scoped_lmdb_dir = args.output_lmdb_dir or (args.archive_dir / f"{split}_lmdb")
        scoped_lmdb_tokens = lmdb_tokens_for_chunks(config, chunks)

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    prepare_chunk_dirs(args, chunks, split)
    extraction_keys = planning_keys if args.early_stop_after_chunks is not None else keys
    actual_image_bytes, missing, extraction_token_to_tfrecord = extract_local_tfrecords(
        args,
        chunks,
        extraction_keys,
        split,
        lmdb_tokens=scoped_lmdb_tokens,
        output_lmdb_dir=scoped_lmdb_dir,
    )
    token_to_tfrecord.update(extraction_token_to_tfrecord)
    token_map_keys = extraction_keys if args.early_stop_after_chunks is not None else keys
    write_token_to_tfrecord_json(
        args.token_to_tfrecord_json,
        split,
        token_map_keys,
        token_to_tfrecord,
    )
    archives = archive_chunks(args, chunks)
    archive_lmdb_dir = scoped_lmdb_dir if scoped_lmdb_dir is not None else lmdb_dir
    lmdb_archive = maybe_archive_lmdb(args, split, archive_lmdb_dir)

    manifest = {
        "format": "autovla_waymo_prepared_chunks_v1",
        "split": split,
        "config": args.config,
        "dataset_name": config.get("dataset_name", "waymo"),
        "target_extracted_gb": args.target_extracted_gb,
        "target_extracted_bytes": target_bytes(args),
        "bytes_per_image": args.bytes_per_image,
        "size_plan": "actual" if args.hard_limit else "estimated",
        "missing_frame_count": len(missing),
        "size_measurement_missing_frame_count": len(size_measurement_missing),
        "token_to_tfrecord_json": make_dataset_ref(args.token_to_tfrecord_json, args.dataset_path),
        "token_to_tfrecord_count": len(token_to_tfrecord),
        "lmdb_archive": make_manifest_ref(lmdb_archive, args.archive_dir) if lmdb_archive else None,
        "chunks": [],
    }
    for chunk, archive_path in zip(chunks, archives):
        manifest["chunks"].append(
            {
                "index": chunk.index,
                "archive": make_manifest_ref(archive_path, args.archive_dir),
                "archive_name": archive_path.name,
                "sequence_count": len(chunk.sequences),
                "sample_count": chunk.sample_count,
                "frame_token_count": chunk.frame_count,
                "estimated_bytes": chunk.estimated_bytes,
                "planned_image_bytes": chunk.estimated_bytes,
                "actual_image_bytes": actual_image_bytes[chunk.index],
            }
        )

    manifest_path = args.archive_dir / "prepared_chunks_manifest.json"
    write_json(manifest_path, manifest)

    if args.upload_prefix is not None:
        for archive_path in archives:
            upload_file(archive_path, make_upload_uri(archive_path, args.upload_prefix))
        if lmdb_archive is not None:
            upload_file(lmdb_archive, make_upload_uri(lmdb_archive, args.upload_prefix))
        upload_file(
            args.token_to_tfrecord_json,
            make_upload_uri(args.token_to_tfrecord_json, args.upload_prefix),
        )
        upload_file(manifest_path, make_upload_uri(manifest_path, args.upload_prefix))

    print(f"Prepared manifest: {manifest_path}")
    if args.upload_prefix is not None:
        print(f"Uploaded manifest: {make_upload_uri(manifest_path, args.upload_prefix)}")


if __name__ == "__main__":
    main()
