#!/usr/bin/env python3
"""Run preprocessing from prepared Waymo chunk archives.

This is the cloud-GPU side of the prepared-chunk workflow.  It consumes the
manifest produced by tools/streaming/waymo_prepare_chunks.py, downloads and
extracts one prepared chunk at a time, points a generated config at that chunk
root, runs the existing preprocessing script with --sample-ids-json, and cleans
the chunk while optionally prefetching the next archive.
"""

import argparse
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True,
                   help="Local path or s3:// URI to prepared_chunks_manifest.json.")
    p.add_argument("--config", required=True,
                   help="Config name under config/ or a direct YAML path.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path,
                   help="Local scratch area for downloaded/extracted chunks.")
    p.add_argument("--include-cot", default="false", choices=["true", "false"])
    p.add_argument("--cuda-devices", default=None)
    p.add_argument("--start-chunk", type=int, default=0)
    p.add_argument("--end-chunk", type=int, default=None,
                   help="Stop before this chunk index.")
    p.add_argument("--no-pipeline", action="store_true")
    p.add_argument("--keep-chunks", action="store_true")
    p.add_argument("--skip-lmdb-download", action="store_true",
                   help="Assume {work-dir}/lmdb/{split}_lmdb already exists.")
    return p.parse_args()


def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def parse_s3_uri(uri: str):
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def s3_client():
    import boto3

    return boto3.client("s3")


def download_uri(uri: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        print(f"download s3://{bucket}/{key} -> {dest}", flush=True)
        s3_client().download_file(bucket, key, str(dest))
    else:
        src = Path(uri)
        if src.resolve() != dest.resolve():
            print(f"copy {src} -> {dest}", flush=True)
            shutil.copy2(src, dest)
    return dest


def resolve_manifest(manifest_uri: str, work_dir: Path) -> Path:
    if is_s3_uri(manifest_uri):
        return download_uri(manifest_uri, work_dir / "manifest" / "prepared_chunks_manifest.json")
    return Path(manifest_uri)


def resolve_relative_uri(base_manifest: str, artifact: str) -> str:
    if is_s3_uri(artifact) or artifact.startswith("/"):
        return artifact
    if is_s3_uri(base_manifest):
        bucket, key = parse_s3_uri(base_manifest)
        prefix = key.rsplit("/", 1)[0] if "/" in key else ""
        full_key = f"{prefix}/{artifact}" if prefix else artifact
        return f"s3://{bucket}/{full_key}"
    return str((Path(base_manifest).parent / artifact).resolve())


def extract_tar(archive_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if archive_path.name.endswith(".tar.gz") else "r"
    with tarfile.open(archive_path, mode) as tar:
        tar.extractall(dest_dir)


def resolve_config_path(config_name: str) -> str:
    candidate = os.path.expanduser(config_name)
    if os.path.isfile(candidate):
        return candidate
    return f"./config/{config_name}.yaml"


def load_config(name: str) -> dict:
    with open(resolve_config_path(name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_lmdb(manifest: dict, manifest_uri: str, work_dir: Path, skip_download: bool) -> Path:
    split = manifest["split"]
    lmdb_dir = work_dir / "lmdb" / f"{split}_lmdb"
    if lmdb_dir.exists():
        return lmdb_dir
    if skip_download:
        raise SystemExit(f"Expected existing LMDB at {lmdb_dir}")

    archive_uri = manifest.get("lmdb_archive")
    if not archive_uri:
        raise SystemExit("Manifest has no lmdb_archive. Provide one or use --skip-lmdb-download.")
    archive_uri = resolve_relative_uri(manifest_uri, archive_uri)
    archive_path = work_dir / "downloads" / Path(archive_uri).name
    download_uri(archive_uri, archive_path)
    extract_tar(archive_path, work_dir / "lmdb")
    archive_path.unlink(missing_ok=True)
    if not lmdb_dir.exists():
        raise SystemExit(f"LMDB archive did not produce {lmdb_dir}")
    return lmdb_dir


def symlink_lmdb(chunk_root: Path, split: str, lmdb_dir: Path):
    link = chunk_root / f"{split}_lmdb"
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == lmdb_dir.resolve():
            return
        raise SystemExit(f"Unexpected LMDB path already exists: {link}")
    link.symlink_to(lmdb_dir, target_is_directory=True)


def prepare_chunk(manifest: dict, manifest_uri: str, work_dir: Path, chunk: dict, lmdb_dir: Path) -> Path:
    archive_uri = resolve_relative_uri(manifest_uri, chunk["archive"])
    archive_path = work_dir / "downloads" / Path(chunk.get("archive_name") or archive_uri).name
    download_uri(archive_uri, archive_path)

    extract_root = work_dir / "chunks"
    expected_root = extract_root / f"chunk_{chunk['index']:05d}"
    if expected_root.exists():
        shutil.rmtree(expected_root)
    extract_tar(archive_path, extract_root)
    archive_path.unlink(missing_ok=True)
    if not expected_root.exists():
        raise SystemExit(f"Chunk archive did not produce {expected_root}")

    symlink_lmdb(expected_root, manifest["split"], lmdb_dir)
    return expected_root


def make_chunk_config(base_config: dict, chunk_root: Path, config_dir: Path, chunk_index: int) -> Path:
    cfg = dict(base_config)
    cfg["dataset_path"] = str(chunk_root)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"chunk_{chunk_index:05d}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_preprocessing(args, config_path: Path, sample_ids_json: Path) -> int:
    cmd = [
        "bash", "scripts/run_waymo_e2e_preprocessing.sh",
        "--config", str(config_path),
        "--output_dir", str(args.output_dir),
        "--include-cot", args.include_cot,
        "--sample-ids-json", str(sample_ids_json),
        "--resume",
    ]
    if args.cuda_devices:
        cmd += ["--cuda-devices", args.cuda_devices]
    return subprocess.run(cmd).returncode


def cleanup_chunk(args, chunk_root: Optional[Path]):
    if args.keep_chunks or chunk_root is None:
        return
    shutil.rmtree(chunk_root, ignore_errors=True)


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_manifest(args.manifest, args.work_dir)
    manifest = json.loads(manifest_path.read_text())
    chunks = manifest["chunks"]
    selected = [
        chunk for chunk in chunks
        if chunk["index"] >= args.start_chunk
        and (args.end_chunk is None or chunk["index"] < args.end_chunk)
    ]
    if not selected:
        print("No chunks to process.")
        return

    base_config = load_config(args.config)
    lmdb_dir = ensure_lmdb(manifest, args.manifest, args.work_dir, args.skip_lmdb_download)
    config_dir = Path(tempfile.mkdtemp(prefix="waymo_prepared_configs_"))
    atexit.register(lambda: shutil.rmtree(config_dir, ignore_errors=True))

    print(
        f"Prepared run: {len(selected)} chunks, "
        f"{sum(c['sample_count'] for c in selected)} samples.",
        flush=True,
    )

    executor = ThreadPoolExecutor(max_workers=1)
    next_future = None
    current_root = None

    def kill_pending(*_):
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, kill_pending)
    signal.signal(signal.SIGTERM, kill_pending)

    try:
        current_root = prepare_chunk(manifest, args.manifest, args.work_dir, selected[0], lmdb_dir)
        for pos, chunk in enumerate(selected):
            print(
                f"=== chunk {chunk['index']} ({pos + 1}/{len(selected)}): "
                f"{chunk['sample_count']} samples ===",
                flush=True,
            )

            if not args.no_pipeline and pos + 1 < len(selected):
                next_chunk = selected[pos + 1]
                next_future = executor.submit(
                    prepare_chunk, manifest, args.manifest, args.work_dir, next_chunk, lmdb_dir
                )
                print(f"  background download/extract started for chunk {next_chunk['index']}", flush=True)

            config_path = make_chunk_config(base_config, current_root, config_dir, chunk["index"])
            rc = run_preprocessing(args, config_path, current_root / "sample_ids.json")
            if rc != 0:
                raise SystemExit(f"Preprocessing failed for chunk {chunk['index']} (exit {rc}).")

            if next_future is not None:
                next_root = next_future.result()
                next_future = None
            else:
                next_root = None

            cleanup_chunk(args, current_root)
            print(f"  cleaned up chunk {chunk['index']}", flush=True)

            if args.no_pipeline and pos + 1 < len(selected):
                next_root = prepare_chunk(manifest, args.manifest, args.work_dir, selected[pos + 1], lmdb_dir)

            if next_root is not None:
                current_root = next_root

    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    print("All prepared chunks processed.")


if __name__ == "__main__":
    main()
