"""Prepare a per-scale training workspace.

For a chosen scale (10k/50k/100k/185k) and CoT model (e.g. Qwen2.4_VL_72B_Instruct_AWQ):

  1. Stream-extract 4 sample-JSON tarballs from /backup/nuplan_nuscenes_train_mix/
     into /data/nuplan_nuscenes_train_mix_<scale>/{nuPlan,nuScenes}/
  2. Walk those JSONs to build the set of raw image paths actually referenced.
  3. Stream-extract raw nuPlan from /backup/raw_dataset_tarball/nuplan/ into the
     shared /data/nuPlan/{trainval_navsim_logs,trainval_sensor_blobs}/, keeping
     only files in the needed set that don't already exist on disk.
  4. Same for nuScenes (only samples/CAM_<6>, plus v1.0-trainval/ + maps/).

Camera pruning is "level 3": only the 3 cameras SFTDataset actually reads (front,
front-left, front-right). Sweeps + LIDAR/RADAR samples are excluded for nuScenes.

Resumable: re-running with the same args is safe; existing files are skipped, so
running --scale 10k after --scale 50k does almost no work for raw data.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter


BACKUP_SCALING_DIR = Path("/backup/nuplan_nuscenes_train_mix")
BACKUP_RAW_NUPLAN  = Path("/backup/raw_dataset_tarball/nuplan")
BACKUP_RAW_NUSC    = Path("/backup/raw_dataset_tarball/nuscenes")
DATA_ROOT          = Path("/data")
NUPLAN_RAW         = DATA_ROOT / "nuPlan"
NUSC_RAW           = DATA_ROOT / "nuScenes"

SCALES = ("10k", "50k", "100k", "185k")

# nuScenes paths starting with any of these top-level prefixes are skipped
# (we keep only the 3 needed cameras under samples/).
NUSC_SKIP_PREFIXES = (
    "sweeps/",
    "samples/LIDAR_TOP/",
    "samples/RADAR_FRONT/",
    "samples/RADAR_FRONT_LEFT/",
    "samples/RADAR_FRONT_RIGHT/",
    "samples/RADAR_BACK_LEFT/",
    "samples/RADAR_BACK_RIGHT/",
    "samples/CAM_BACK/",
    "samples/CAM_BACK_LEFT/",
    "samples/CAM_BACK_RIGHT/",
)

# SFTDataset only reads `front_camera` / `front_left_camera` / `front_right_camera`.
# autovla_agent.py:256-261 maps these to *different* JSON fields per dataset:
#   nuplan   → front_camera_paths,        left_camera_paths,        right_camera_paths
#   nuscenes → front_camera_paths,        front_left_camera_paths,  front_right_camera_paths
NUPLAN_USED_FIELDS = ("front_camera_paths", "left_camera_paths", "right_camera_paths")
NUSC_USED_FIELDS   = ("front_camera_paths", "front_left_camera_paths", "front_right_camera_paths")


def open_streaming_tar(tarball: Path):
    """Open a tarball for streaming reads (mode 'r|'). Returns (TarFile, decoder_proc).

    Decompression always runs in a subprocess (zstdcat / pigz / gzip) so that an
    8-way ThreadPool actually parallelizes — the alternative ('r|gz' built-in)
    holds the GIL across enough of the inflate loop to cap aggregate read at
    ~3× single-stream instead of the benchmark's 6-7×.
    """
    name = tarball.name
    if name.endswith(".tar.zst") or name.endswith(".tzst"):
        decompressor = ["zstdcat", str(tarball)]
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        gz = "pigz" if shutil.which("pigz") else "gzip"
        decompressor = [gz, "-dc", str(tarball)]
    else:
        return tarfile.open(str(tarball), mode="r|"), None
    proc = subprocess.Popen(decompressor, stdout=subprocess.PIPE, bufsize=4 << 20)
    return tarfile.open(fileobj=proc.stdout, mode="r|"), proc


def stream_extract_filtered(tarball: Path, predicate, output_path_fn):
    """Stream tarball, extract members where predicate(name)==True to output_path_fn(name).
    Returns (written, skipped_already_exist, skipped_by_predicate)."""
    tar, proc = open_streaming_tar(tarball)
    written = 0
    skipped_exist = 0
    skipped_pred = 0
    try:
        for member in tar:
            if not member.isfile():
                continue
            if not predicate(member.name):
                skipped_pred += 1
                continue
            out = output_path_fn(member.name)
            if out is None:
                skipped_pred += 1
                continue
            if out.exists():
                skipped_exist += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            f = tar.extractfile(member)
            if f is None:
                continue
            tmp = out.with_suffix(out.suffix + ".tmp")
            try:
                with open(tmp, "wb") as fout:
                    shutil.copyfileobj(f, fout, length=1 << 18)
                os.replace(tmp, out)
                written += 1
            finally:
                if tmp.exists():
                    tmp.unlink()
    finally:
        tar.close()
        if proc is not None:
            proc.wait()
    return written, skipped_exist, skipped_pred


def find_sample_tarball(scale: str, bucket: str, model: str) -> Path:
    scale_dir = BACKUP_SCALING_DIR / f"nuplan_nuscenes_train_mix_{scale}"
    patterns = {
        "nuplan_nocot":   "nuplan_action_only_samples_*.tar.zst",
        "nuplan_cot":     f"nuplan_nl_reasoning_samples_{model}_*.tar.zst",
        "nuscenes_nocot": "nuscenes_action_only_samples_*.tar.zst",
        "nuscenes_cot":   "nuscenes_nl_reasoning_samples_*.tar.zst",
    }
    pat = patterns[bucket]
    matches = list(scale_dir.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No tarball matching {pat} in {scale_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {pat}: {matches}")
    return matches[0]


def extract_sample_tarball(tarball: Path, target_subdir: Path, dataset_prefix: str) -> int:
    """Extract a sample-JSON tarball into target_subdir, stripping `<dataset_prefix>_`
    from the leading top-level directory name inside the archive."""
    strip = f"{dataset_prefix}_"
    n = 0
    tar, proc = open_streaming_tar(tarball)
    try:
        for member in tar:
            if not member.isfile():
                continue
            inner = member.name
            if inner.startswith(strip):
                inner = inner[len(strip):]
            out = target_subdir / inner
            out.parent.mkdir(parents=True, exist_ok=True)
            f = tar.extractfile(member)
            if f is None:
                continue
            with open(out, "wb") as fout:
                shutil.copyfileobj(f, fout, length=1 << 18)
            n += 1
    finally:
        tar.close()
        if proc is not None:
            proc.wait()
    return n


def collect_needed_raw_paths(workspace: Path):
    """Walk the per-scale workspace JSONs, return:
      - nuplan_jpgs: set of absolute jpg paths under /data/nuPlan/trainval_sensor_blobs/
      - nuplan_pkls: set of absolute pkl paths under /data/nuPlan/trainval_navsim_logs/
      - nusc_jpgs:   set of absolute jpg paths under /data/nuScenes/samples/
    Each set is restricted to the 3 cameras SFTDataset actually consumes."""
    nuplan_jpgs = set()
    nuplan_scenes = set()
    nusc_jpgs = set()

    for jpath in (workspace / "nuPlan").rglob("*.json"):
        with open(jpath) as f:
            d = json.load(f)
        for field in NUPLAN_USED_FIELDS:
            for p in d.get(field) or []:
                if not isinstance(p, str):
                    continue
                nuplan_jpgs.add(p)
                # /data/nuPlan/trainval_sensor_blobs/trainval/<scene>/CAM_*/<jpg>.jpg
                parts = p.split("/")
                try:
                    i = parts.index("trainval_sensor_blobs")
                    nuplan_scenes.add(parts[i + 2])  # skip "trainval/"
                except (ValueError, IndexError):
                    pass

    for jpath in (workspace / "nuScenes").rglob("*.json"):
        with open(jpath) as f:
            d = json.load(f)
        for field in NUSC_USED_FIELDS:
            for p in d.get(field) or []:
                if not isinstance(p, str):
                    continue
                nusc_jpgs.add(p)

    nuplan_pkls = {
        str(NUPLAN_RAW / "trainval_navsim_logs" / "trainval" / f"{s}.pkl")
        for s in nuplan_scenes
    }
    return nuplan_jpgs, nuplan_pkls, nusc_jpgs, nuplan_scenes


def extract_raw_nuplan(jpgs: set, pkls: set, parallelism: int):
    sensor_strip = "openscene-v1.1/sensor_blobs/"
    sensor_dest = NUPLAN_RAW / "trainval_sensor_blobs"
    meta_strip = "openscene-v1.1/meta_datas/"
    meta_dest = NUPLAN_RAW / "trainval_navsim_logs"

    def jpg_predicate(name):
        return name.startswith(sensor_strip) and name.endswith(".jpg")

    def jpg_output_fn(name):
        rel = name[len(sensor_strip):]
        full = sensor_dest / rel
        return full if str(full) in jpgs else None

    def pkl_predicate(name):
        return name.startswith(meta_strip) and name.endswith(".pkl")

    def pkl_output_fn(name):
        rel = name[len(meta_strip):]
        full = meta_dest / rel
        return full if str(full) in pkls else None

    missing_jpgs = sum(1 for p in jpgs if not Path(p).exists())
    missing_pkls = sum(1 for p in pkls if not Path(p).exists())
    print(f"  needed: {len(jpgs)} jpgs ({missing_jpgs} missing), "
          f"{len(pkls)} pkls ({missing_pkls} missing)")
    if missing_jpgs == 0 and missing_pkls == 0:
        print("  all needed nuplan raw files already on /data, skipping extraction.")
        return

    if missing_pkls > 0:
        meta_tar = BACKUP_RAW_NUPLAN / "openscene_metadata_trainval.tgz"
        print(f"  extracting nuplan metadata ({meta_tar.name}) ...")
        t0 = perf_counter()
        w, se, sp = stream_extract_filtered(meta_tar, pkl_predicate, pkl_output_fn)
        print(f"    wrote={w}, skip-exist={se}, skip-pred={sp}, "
              f"elapsed={perf_counter()-t0:.1f}s")

    if missing_jpgs > 0:
        # Sort by size desc so the biggest tarballs start first and overlap with
        # the long tail; otherwise the largest two finish strictly after the
        # smaller eight, adding ~5 min of single-stream tail latency.
        cams = sorted(
            BACKUP_RAW_NUPLAN.glob("openscene_sensor_trainval_camera_*.tgz"),
            key=lambda p: -p.stat().st_size,
        )
        print(f"  streaming {len(cams)} nuplan camera tarballs (parallel={parallelism}) ...")
        t0 = perf_counter()
        total = [0, 0, 0]
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futs = {pool.submit(stream_extract_filtered, tb, jpg_predicate, jpg_output_fn): tb
                    for tb in cams}
            done = 0
            for fut in as_completed(futs):
                w, se, sp = fut.result()
                total[0] += w; total[1] += se; total[2] += sp
                done += 1
                if done % 20 == 0 or done == len(cams):
                    print(f"    progress: {done}/{len(cams)} tarballs, "
                          f"wrote={total[0]} skip-exist={total[1]}, "
                          f"elapsed={perf_counter()-t0:.1f}s")
        print(f"    final: wrote={total[0]}, skip-exist={total[1]}, skip-pred={total[2]}, "
              f"elapsed={perf_counter()-t0:.1f}s")


def extract_raw_nuscenes(jpgs: set, parallelism: int):
    def jpg_predicate(name):
        if any(name.startswith(p) for p in NUSC_SKIP_PREFIXES):
            return False
        return name.startswith("samples/") and name.endswith(".jpg")

    def jpg_output_fn(name):
        full = NUSC_RAW / name
        return full if str(full) in jpgs else None

    def meta_predicate(name):
        if any(name.startswith(p) for p in NUSC_SKIP_PREFIXES):
            return False
        return True

    def meta_output_fn(name):
        return NUSC_RAW / name

    # Always stream meta tarball — it's small and idempotent.
    meta_tar = BACKUP_RAW_NUSC / "v1.0-trainval_meta.tgz"
    print(f"  extracting {meta_tar.name} ...")
    t0 = perf_counter()
    w, se, sp = stream_extract_filtered(meta_tar, meta_predicate, meta_output_fn)
    print(f"    wrote={w}, skip-exist={se}, skip-pred={sp}, "
          f"elapsed={perf_counter()-t0:.1f}s")

    missing = sum(1 for p in jpgs if not Path(p).exists())
    print(f"  needed: {len(jpgs)} jpgs ({missing} missing)")
    if missing == 0:
        print("  all needed nuscenes jpgs already on /data, skipping blob extraction.")
        return

    # Largest first — trainval10 (39G) and trainval09 (32G) dominate tail latency.
    blobs = sorted(
        BACKUP_RAW_NUSC.glob("v1.0-trainval[0-9]*_blobs.tgz"),
        key=lambda p: -p.stat().st_size,
    )
    print(f"  streaming {len(blobs)} nuscenes trainval blobs (parallel={parallelism}) ...")
    t0 = perf_counter()
    total = [0, 0, 0]
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futs = {pool.submit(stream_extract_filtered, tb, jpg_predicate, jpg_output_fn): tb
                for tb in blobs}
        done = 0
        for fut in as_completed(futs):
            w, se, sp = fut.result()
            total[0] += w; total[1] += se; total[2] += sp
            done += 1
            print(f"    progress: {done}/{len(blobs)} blobs, wrote={total[0]} "
                  f"skip-exist={total[1]}, elapsed={perf_counter()-t0:.1f}s")
    print(f"    final: wrote={total[0]}, skip-exist={total[1]}, skip-pred={total[2]}, "
          f"elapsed={perf_counter()-t0:.1f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", required=True, choices=SCALES)
    p.add_argument("--model", required=True,
                   help="CoT model substring matched against the nuplan_cot tarball name "
                        "(e.g. Qwen2.4_VL_72B_Instruct_AWQ).")
    p.add_argument("--parallelism", type=int, default=8,
                   help="Number of tarballs streamed concurrently (per benchmark, 8 is optimal).")
    p.add_argument("--force", action="store_true",
                   help="rm -rf the per-scale workspace before starting.")
    p.add_argument("--skip-raw", action="store_true",
                   help="Skip raw-data extraction (only build the per-scale JSON workspace).")
    args = p.parse_args()

    workspace = DATA_ROOT / f"nuplan_nuscenes_train_mix_{args.scale}"
    print(f"workspace: {workspace}")
    if args.force and workspace.exists():
        print(f"--force: removing {workspace}")
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "nuPlan").mkdir(exist_ok=True)
    (workspace / "nuScenes").mkdir(exist_ok=True)

    # Phase 1
    print(f"\n=== Phase 1: sample-JSON tarballs → {workspace} ===")
    t1 = perf_counter()
    bucket_specs = [
        ("nuplan_nocot",   "nuplan",   workspace / "nuPlan"),
        ("nuplan_cot",     "nuplan",   workspace / "nuPlan"),
        ("nuscenes_nocot", "nuscenes", workspace / "nuScenes"),
        ("nuscenes_cot",   "nuscenes", workspace / "nuScenes"),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {}
        for bucket, ds, tgt in bucket_specs:
            tb = find_sample_tarball(args.scale, bucket, args.model)
            futs[pool.submit(extract_sample_tarball, tb, tgt, ds)] = (bucket, tb)
        for fut in as_completed(futs):
            bucket, tb = futs[fut]
            n = fut.result()
            print(f"  [{bucket:>14s}] {tb.name}: extracted {n} files")

    splits_src = (BACKUP_SCALING_DIR / f"nuplan_nuscenes_train_mix_{args.scale}"
                  / f"scaling_{args.scale}_token_list.json")
    shutil.copy2(splits_src, workspace / splits_src.name)
    print(f"  copied {splits_src.name}")
    print(f"  Phase 1 elapsed: {perf_counter()-t1:.1f}s")

    if args.skip_raw:
        print("\n--skip-raw set; done.")
        return 0

    # Phase 2
    print(f"\n=== Phase 2: compute needed raw paths ===")
    t2 = perf_counter()
    nuplan_jpgs, nuplan_pkls, nusc_jpgs, nuplan_scenes = collect_needed_raw_paths(workspace)
    print(f"  nuplan: {len(nuplan_jpgs)} jpgs, {len(nuplan_scenes)} scenes ({len(nuplan_pkls)} pkls)")
    print(f"  nuscenes: {len(nusc_jpgs)} jpgs")
    print(f"  Phase 2 elapsed: {perf_counter()-t2:.1f}s")

    # Phase 3
    print(f"\n=== Phase 3: raw nuPlan → /data/nuPlan/ ===")
    t3 = perf_counter()
    extract_raw_nuplan(nuplan_jpgs, nuplan_pkls, args.parallelism)
    print(f"  Phase 3 elapsed: {perf_counter()-t3:.1f}s")

    # Phase 4
    print(f"\n=== Phase 4: raw nuScenes → /data/nuScenes/ ===")
    t4 = perf_counter()
    extract_raw_nuscenes(nusc_jpgs, args.parallelism)
    print(f"  Phase 4 elapsed: {perf_counter()-t4:.1f}s")

    # Verify
    print(f"\n=== Verification ===")
    miss_np = sum(1 for p in nuplan_jpgs if not Path(p).exists())
    miss_pk = sum(1 for p in nuplan_pkls if not Path(p).exists())
    miss_ns = sum(1 for p in nusc_jpgs if not Path(p).exists())
    print(f"  missing: nuplan jpgs={miss_np}, nuplan pkls={miss_pk}, nuscenes jpgs={miss_ns}")
    if miss_np or miss_pk or miss_ns:
        print("  WARNING: some needed files are missing")
        return 1
    print("  all needed files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
