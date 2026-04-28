#!/usr/bin/env python3
"""Build a Waymo end-to-end LMDB and sequence index from local or S3 tfrecords.

This is a one-time pre-step for the streaming preprocessing pipeline. It
parses each tfrecord into LMDB and records which sequences live in which
tfrecords. In S3 mode, disk usage stays bounded to roughly
num_download_workers * tfrecord_size because downloaded tfrecords are deleted
after parsing.

Outputs:
  - {output_lmdb}/                LMDB matching what waymo_e2e_dataset.py expects
  - {output_index}                JSON sequence index used by the chunked orchestrator
  - {state_file}                  Sidecar of completed tfrecord keys (resume marker)
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import BoundedSemaphore

import boto3
import lmdb
import tensorflow as tf
from botocore.config import Config
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default=None, help="S3 bucket name.")
    p.add_argument("--s3-prefix", default=None,
                   help="S3 prefix under which tfrecords live, e.g. "
                        "'object_storage_xxx/objvol_yyy/waymo_end_to_end/train/'.")
    p.add_argument("--local-tfrecord-dir", default=None, type=Path,
                   help="Local directory containing Waymo tfrecords. Use instead of --bucket/--s3-prefix.")
    p.add_argument("--split", required=True, choices=["training", "val", "test"],
                   help="Split name as the dataset code expects (used for filenames and verification).")
    p.add_argument("--output-lmdb", required=True, type=Path,
                   help="LMDB directory; should be {dataset_path}/{split}_lmdb to match dataset code.")
    p.add_argument("--output-index", required=True, type=Path,
                   help="Sequence index JSON path.")
    p.add_argument("--staging-dir", default=None, type=Path,
                   help="Scratch directory for downloaded tfrecords (cleared continuously).")
    p.add_argument("--state-file", default=None, type=Path,
                   help="Resume marker; default: {output_index}.processed.txt")
    p.add_argument("--num-download-workers", type=int, default=4)
    p.add_argument("--lmdb-map-size-gb", type=int, default=1500,
                   help="Virtual address space, not actual disk usage.")
    p.add_argument("--save-index-every", type=int, default=20)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N (smallest, by size) tfrecords. For testing.")
    args = p.parse_args()
    using_local = args.local_tfrecord_dir is not None
    using_s3 = args.bucket is not None or args.s3_prefix is not None
    if using_local and using_s3:
        p.error("Use either --local-tfrecord-dir or --bucket/--s3-prefix, not both.")
    if not using_local and not (args.bucket and args.s3_prefix):
        p.error("Provide --local-tfrecord-dir, or provide both --bucket and --s3-prefix.")
    if not using_local and args.staging_dir is None:
        p.error("--staging-dir is required when reading from S3.")
    if using_local and not args.local_tfrecord_dir.exists():
        p.error(f"--local-tfrecord-dir does not exist: {args.local_tfrecord_dir}")
    return args


def load_state(path: Path) -> set:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_state(path: Path, key: str):
    with path.open("a") as f:
        f.write(key + "\n")


def list_tfrecords(s3, bucket: str, prefix: str) -> list:
    paginator = s3.get_paginator("list_objects_v2")
    objs = []
    full_prefix = prefix.rstrip("/") + "/"
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if ".tfrecord-" in key.rsplit("/", 1)[-1]:
                objs.append((key, obj["Size"]))
    return objs  # list of (key, size); caller decides ordering


def list_local_tfrecords(local_dir: Path, split: str) -> list:
    pattern = f"{split}_*.tfrecord-*"
    exact = re.compile(rf"^{re.escape(split)}_.*\.tfrecord-\d{{5}}-of-\d{{5}}$")
    objs = []
    for path in sorted(local_dir.glob(pattern)):
        if exact.match(path.name):
            objs.append((path.name, path.stat().st_size))
    return objs


def local_source_for_key(local_dir: Path, key: str) -> Path:
    direct = local_dir / Path(key).name
    if direct.exists():
        return direct
    nested = local_dir / key
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Local tfrecord not found for {key} under {local_dir}")


def keys_sorted(objs):
    return [k for k, _ in sorted(objs)]


def keys_smallest_first(objs, n: int):
    return [k for k, _ in sorted(objs, key=lambda kv: kv[1])[:n]]


def index_to_serializable(index: dict) -> dict:
    return {
        "sequences": {
            seq: {"tfrecords": sorted(v["tfrecords"]), "tokens": sorted(v["tokens"])}
            for seq, v in index["sequences"].items()
        },
        "tfrecord_to_sequences": {
            k: sorted(v) for k, v in index["tfrecord_to_sequences"].items()
        },
    }


def save_index(index: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(index_to_serializable(index), indent=2, sort_keys=True))
    tmp.replace(path)


def load_index(path: Path) -> dict:
    if not path.exists():
        return {"sequences": {}, "tfrecord_to_sequences": {}}
    raw = json.loads(path.read_text())
    return {
        "sequences": {
            seq: {"tfrecords": set(v["tfrecords"]), "tokens": set(v["tokens"])}
            for seq, v in raw.get("sequences", {}).items()
        },
        "tfrecord_to_sequences": {
            k: set(v) for k, v in raw.get("tfrecord_to_sequences", {}).items()
        },
    }


def process_tfrecord(env, index: dict, key: str, local_path: Path):
    seq_set = set()
    tokens_added = 0
    with env.begin(write=True) as txn:
        for raw in tf.data.TFRecordDataset(str(local_path), compression_type=""):
            raw_bytes = raw.numpy()
            frame = wod_e2ed_pb2.E2EDFrame()
            frame.ParseFromString(raw_bytes)
            token = frame.frame.context.name
            seq_name = token.split("-")[0]
            # Strip embedded camera image bytes; the dataset reads JPEGs from disk
            # and only uses metadata fields from LMDB.
            del frame.frame.images[:]
            stripped_bytes = frame.SerializeToString()
            txn.put(token.encode("utf-8"), stripped_bytes)
            seq_entry = index["sequences"].setdefault(
                seq_name, {"tfrecords": set(), "tokens": set()}
            )
            seq_entry["tokens"].add(token)
            seq_entry["tfrecords"].add(key)
            seq_set.add(seq_name)
            tokens_added += 1
    index["tfrecord_to_sequences"].setdefault(key, set()).update(seq_set)
    return tokens_added, seq_set


def print_progress(done_idx: int, total: int, key: str, tokens_added: int, seq_set: set, started: float):
    elapsed = time.time() - started
    rate = done_idx / max(elapsed, 1e-3)
    eta_min = (total - done_idx) / max(rate, 1e-3) / 60.0
    print(
        f"[{done_idx}/{total}] {Path(key).name}: "
        f"+{tokens_added} tokens, +{len(seq_set)} sequences  "
        f"(rate {rate:.2f}/s, eta {eta_min:.1f}min)",
        flush=True,
    )


def main():
    args = parse_args()
    args.output_lmdb.mkdir(parents=True, exist_ok=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    if args.staging_dir is not None:
        args.staging_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file or args.output_index.with_suffix(
        args.output_index.suffix + ".processed.txt"
    )
    state_file.parent.mkdir(parents=True, exist_ok=True)

    s3 = None
    if args.local_tfrecord_dir is not None:
        print(f"Listing tfrecords under {args.local_tfrecord_dir}")
        objs = list_local_tfrecords(args.local_tfrecord_dir, args.split)
        source_desc = str(args.local_tfrecord_dir)
    else:
        s3 = boto3.client(
            "s3", config=Config(retries={"max_attempts": 5, "mode": "adaptive"})
        )
        print(f"Listing tfrecords under s3://{args.bucket}/{args.s3_prefix}")
        objs = list_tfrecords(s3, args.bucket, args.s3_prefix)
        source_desc = f"s3://{args.bucket}/{args.s3_prefix}"
    if not objs:
        sys.exit(f"No tfrecords found under {source_desc}")
    if args.limit:
        all_keys = keys_smallest_first(objs, args.limit)
        print(f"Found {len(objs)} tfrecord keys; testing on smallest {len(all_keys)}.")
    else:
        all_keys = keys_sorted(objs)
        print(f"Found {len(all_keys)} tfrecord keys.")

    processed = load_state(state_file)
    remaining = [k for k in all_keys if k not in processed]
    print(f"Already processed: {len(processed)}. Remaining: {len(remaining)}.")
    if not remaining:
        print("Nothing to do.")
        return

    index = load_index(args.output_index)
    env = lmdb.open(str(args.output_lmdb), map_size=args.lmdb_map_size_gb * 1024 ** 3)

    if args.local_tfrecord_dir is not None:
        saved_after = 0
        started = time.time()
        try:
            for done_idx, key in enumerate(remaining, 1):
                local_path = local_source_for_key(args.local_tfrecord_dir, key)
                tokens_added, seq_set = process_tfrecord(env, index, key, local_path)
                append_state(state_file, key)
                saved_after += 1
                print_progress(done_idx, len(remaining), key, tokens_added, seq_set, started)
                if saved_after >= args.save_index_every:
                    save_index(index, args.output_index)
                    saved_after = 0
        finally:
            save_index(index, args.output_index)
            env.sync()
            env.close()
        print_done(index, args.output_lmdb, args.output_index)
        return

    download_slots = BoundedSemaphore(args.num_download_workers)

    def download(key: str):
        download_slots.acquire()
        local_path = args.staging_dir / key.replace("/", "__")
        try:
            s3.download_file(args.bucket, key, str(local_path))
            return key, local_path
        except Exception:
            local_path.unlink(missing_ok=True)
            download_slots.release()
            raise

    saved_after = 0
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.num_download_workers) as pool:
            futures = {pool.submit(download, k): k for k in remaining}
            for done_idx, fut in enumerate(as_completed(futures), 1):
                key = futures[fut]
                try:
                    _, local_path = fut.result()
                except Exception as e:
                    print(f"[{done_idx}/{len(remaining)}] download FAILED for {key}: {e}")
                    continue
                try:
                    tokens_added, seq_set = process_tfrecord(env, index, key, local_path)
                    append_state(state_file, key)
                    saved_after += 1
                    print_progress(done_idx, len(remaining), key, tokens_added, seq_set, started)
                    if saved_after >= args.save_index_every:
                        save_index(index, args.output_index)
                        saved_after = 0
                finally:
                    local_path.unlink(missing_ok=True)
                    download_slots.release()
    finally:
        save_index(index, args.output_index)
        env.sync()
        env.close()

    print_done(index, args.output_lmdb, args.output_index)


def print_done(index: dict, output_lmdb: Path, output_index: Path):
    multi_tfrecord = [
        seq for seq, v in index["sequences"].items() if len(v["tfrecords"]) > 1
    ]
    if multi_tfrecord:
        print(
            f"Note: {len(multi_tfrecord)} sequences span multiple tfrecords. "
            f"Chunks must include the full tfrecord set for these sequences. "
            f"Examples: {multi_tfrecord[:5]}"
        )
    print(f"Done. LMDB: {output_lmdb}  Index: {output_index}")


if __name__ == "__main__":
    main()
