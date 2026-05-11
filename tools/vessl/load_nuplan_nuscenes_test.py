"""Prepare a per-dataset test workspace from /backup tarballs.

For --dataset nuplan / nuscenes / both:

  1. Stream-extract the test sample-JSON tarball from /backup/<ds>_test/
     into /data/<ds>_test/test_samples_<N>/<token>.json.
  2. Walk those JSONs to compute the set of raw image paths actually
     referenced (and pkls for nuPlan), restricted to the 3 cameras
     SFTDataset / AutoVLAAgent consume.
  3. Stream-extract those raw files from /backup/raw_dataset_tarball/
     into the canonical /data layout, skipping files already on disk.

This is purely a deploy-from-backup script; the sample JSONs and raw
tarballs in /backup are produced by a separate preprocessing flow.

Camera pruning is configurable via --num-cameras (3 or 4):
  num_cameras=3 (default, SFT/RL training set):
    nuPlan   → front_camera_paths, left_camera_paths, right_camera_paths
    nuScenes → front_camera_paths, front_left_camera_paths, front_right_camera_paths
  num_cameras=4 (CoT-annotation set, adds back):
    nuPlan   → above + back_camera_paths
    nuScenes → above + back_camera_paths

Resumable: existing files on disk are skipped; rerunning is safe.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

# Reuse helpers + constants from sibling script (same dir).
sys.path.insert(0, str(Path(__file__).parent))
from load_nuplan_nuscenes_mix_train import (  # noqa: E402
    open_streaming_tar,
    stream_extract_filtered,
    NUSC_SKIP_PREFIXES,
    used_fields,
    sync_drivelm,
)


BACKUP_NUPLAN_JSON = Path("/backup/nuplan_test")
BACKUP_NUSC_JSON   = Path("/backup/nuscenes_test")
BACKUP_RAW_NUPLAN  = Path("/backup/raw_dataset_tarball/nuplan")
BACKUP_RAW_NUSC    = Path("/backup/raw_dataset_tarball/nuscenes")

DATA_ROOT             = Path("/data")
NUPLAN_TEST_WORKSPACE = DATA_ROOT / "nuplan_test"
NUSC_TEST_WORKSPACE   = DATA_ROOT / "nuscenes_test"

# nuPlan test raw lives under the unified /data/nuPlan/{navsim_logs,sensor_blobs}/
# trees, distinguished by the "test/" subdir. Config 'placeholder' substitution
# of the form .../nuPlan/placeholder/test then yields the right paths.
NUPLAN_TEST_LOGS    = DATA_ROOT / "nuPlan" / "navsim_logs"
NUPLAN_TEST_SENSORS = DATA_ROOT / "nuPlan" / "sensor_blobs"

# nuScenes raw is shared with trainval (val frames live in samples/CAM_*/).
NUSC_RAW = DATA_ROOT / "nuScenes"


def find_test_json_tarball(backup_dir: Path) -> Path:
    matches = list(backup_dir.glob("test_samples_*.tar.zst"))
    if not matches:
        raise FileNotFoundError(f"No test_samples_*.tar.zst in {backup_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches in {backup_dir}: {matches}")
    return matches[0]


def extract_test_json_tarball(tarball: Path, target_root: Path):
    """Extract test JSON tarball preserving its top-level directory.
    Returns (extracted_dir, file_count)."""
    target_root.mkdir(parents=True, exist_ok=True)
    n = 0
    top_levels = set()
    tar, proc = open_streaming_tar(tarball)
    try:
        for member in tar:
            if not member.isfile():
                continue
            out = target_root / member.name
            out.parent.mkdir(parents=True, exist_ok=True)
            top_levels.add(member.name.split("/")[0])
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
    if len(top_levels) != 1:
        raise RuntimeError(f"Expected single top-level dir in {tarball}, got {top_levels}")
    return target_root / next(iter(top_levels)), n


def collect_nuplan_paths(samples_dir: Path, num_cameras: int = 3):
    """Walk extracted test JSONs and build the set of needed jpg/pkl paths.

    JSONs from the upstream pipeline use the legacy
    `/data/nuPlan/test_sensor_blobs/test/...` form; we translate them in-memory
    to the unified `/data/nuPlan/sensor_blobs/test/...` form so the in-memory
    set matches the destinations used by extract_nuplan_raw() below. Idempotent
    for JSONs that already use the new layout."""
    jpgs, scenes = set(), set()
    fields = used_fields("nuplan", num_cameras)
    for fp in samples_dir.rglob("*.json"):
        d = json.loads(fp.read_text())
        for field in fields:
            for p in d.get(field) or []:
                if not isinstance(p, str):
                    continue
                p_new = p.replace("/test_sensor_blobs/", "/sensor_blobs/")
                jpgs.add(p_new)
                # /data/nuPlan/sensor_blobs/test/<log>/CAM_X/<jpg>.jpg
                parts = p_new.split("/")
                try:
                    i = parts.index("sensor_blobs")
                    scenes.add(parts[i + 2])  # skip "test/"
                except (ValueError, IndexError):
                    pass
    pkls = {str(NUPLAN_TEST_LOGS / "test" / f"{s}.pkl") for s in scenes}
    return jpgs, pkls, scenes


def collect_nuscenes_paths(samples_dir: Path, num_cameras: int = 3):
    jpgs = set()
    fields = used_fields("nuscenes", num_cameras)
    for fp in samples_dir.rglob("*.json"):
        d = json.loads(fp.read_text())
        for field in fields:
            for p in d.get(field) or []:
                if isinstance(p, str):
                    jpgs.add(p)
    return jpgs


def extract_nuplan_raw(jpgs: set, pkls: set, parallelism: int):
    sensor_strip = "openscene-v1.1/sensor_blobs/"
    meta_strip   = "openscene-v1.1/meta_datas/"

    def jpg_predicate(name):
        return name.startswith(sensor_strip) and name.endswith(".jpg")
    def jpg_output_fn(name):
        full = NUPLAN_TEST_SENSORS / name[len(sensor_strip):]
        return full if str(full) in jpgs else None
    def pkl_predicate(name):
        return name.startswith(meta_strip) and name.endswith(".pkl")
    def pkl_output_fn(name):
        full = NUPLAN_TEST_LOGS / name[len(meta_strip):]
        return full if str(full) in pkls else None

    miss_j = sum(1 for p in jpgs if not Path(p).exists())
    miss_p = sum(1 for p in pkls if not Path(p).exists())
    print(f"  needed: {len(jpgs)} jpgs ({miss_j} missing), "
          f"{len(pkls)} pkls ({miss_p} missing)")
    if miss_j == 0 and miss_p == 0:
        print("  all needed nuplan raw files already on /data, skipping.")
        return

    if miss_p > 0:
        meta_tar = BACKUP_RAW_NUPLAN / "openscene_metadata_test.tgz"
        print(f"  extracting nuplan metadata ({meta_tar.name}) ...")
        t0 = perf_counter()
        w, se, sp = stream_extract_filtered(meta_tar, pkl_predicate, pkl_output_fn)
        print(f"    wrote={w}, skip-exist={se}, skip-pred={sp}, "
              f"elapsed={perf_counter()-t0:.1f}s")

    if miss_j > 0:
        # Largest tarballs first to avoid single-stream tail latency.
        cams = sorted(
            BACKUP_RAW_NUPLAN.glob("openscene_sensor_test_camera_*.tgz"),
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
                if done % 4 == 0 or done == len(cams):
                    print(f"    progress: {done}/{len(cams)} tarballs, "
                          f"wrote={total[0]} skip-exist={total[1]}, "
                          f"elapsed={perf_counter()-t0:.1f}s")
        print(f"    final: wrote={total[0]}, skip-exist={total[1]}, skip-pred={total[2]}, "
              f"elapsed={perf_counter()-t0:.1f}s")


def extract_nuscenes_raw(jpgs: set, parallelism: int):
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

    meta_tar = BACKUP_RAW_NUSC / "v1.0-trainval_meta.tgz"
    print(f"  extracting {meta_tar.name} ...")
    t0 = perf_counter()
    w, se, sp = stream_extract_filtered(meta_tar, meta_predicate, meta_output_fn)
    print(f"    wrote={w}, skip-exist={se}, skip-pred={sp}, "
          f"elapsed={perf_counter()-t0:.1f}s")

    miss = sum(1 for p in jpgs if not Path(p).exists())
    print(f"  needed: {len(jpgs)} jpgs ({miss} missing)")
    if miss == 0:
        print("  all needed nuscenes jpgs already on /data, skipping blob extraction.")
        return

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
            print(f"    progress: {done}/{len(blobs)} blobs, "
                  f"wrote={total[0]} skip-exist={total[1]}, "
                  f"elapsed={perf_counter()-t0:.1f}s")
    print(f"    final: wrote={total[0]}, skip-exist={total[1]}, skip-pred={total[2]}, "
          f"elapsed={perf_counter()-t0:.1f}s")


def process_nuplan(parallelism: int, force: bool, skip_raw: bool, num_cameras: int = 3) -> int:
    print(f"\n========== nuPlan test ==========")
    if force and NUPLAN_TEST_WORKSPACE.exists():
        print(f"--force: removing {NUPLAN_TEST_WORKSPACE}")
        shutil.rmtree(NUPLAN_TEST_WORKSPACE)

    print(f"\n=== Phase 1: sample-JSON tarball → {NUPLAN_TEST_WORKSPACE} ===")
    t1 = perf_counter()
    tarball = find_test_json_tarball(BACKUP_NUPLAN_JSON)
    samples_dir, n = extract_test_json_tarball(tarball, NUPLAN_TEST_WORKSPACE)
    print(f"  {tarball.name}: extracted {n} files into {samples_dir}/")
    print(f"  Phase 1 elapsed: {perf_counter()-t1:.1f}s")

    if skip_raw:
        print("\n--skip-raw set; nuPlan done.")
        return 0

    print(f"\n=== Phase 2: compute needed raw paths (num_cameras={num_cameras}) ===")
    t2 = perf_counter()
    jpgs, pkls, scenes = collect_nuplan_paths(samples_dir, num_cameras)
    print(f"  {len(jpgs)} jpgs, {len(scenes)} scenes ({len(pkls)} pkls)")
    print(f"  Phase 2 elapsed: {perf_counter()-t2:.1f}s")

    print(f"\n=== Phase 3: raw nuPlan → /data/nuPlan/test_*/ ===")
    t3 = perf_counter()
    extract_nuplan_raw(jpgs, pkls, parallelism)
    print(f"  Phase 3 elapsed: {perf_counter()-t3:.1f}s")

    print(f"\n=== Verification (nuPlan) ===")
    miss_j = sum(1 for p in jpgs if not Path(p).exists())
    miss_p = sum(1 for p in pkls if not Path(p).exists())
    print(f"  missing: jpgs={miss_j}, pkls={miss_p}")
    if miss_j or miss_p:
        print("  WARNING: some needed files are missing")
        return 1
    print("  all needed nuPlan files present.")
    return 0


def process_nuscenes(parallelism: int, force: bool, skip_raw: bool, num_cameras: int = 3) -> int:
    print(f"\n========== nuScenes test ==========")
    if force and NUSC_TEST_WORKSPACE.exists():
        print(f"--force: removing {NUSC_TEST_WORKSPACE}")
        shutil.rmtree(NUSC_TEST_WORKSPACE)

    print(f"\n=== Phase 1: sample-JSON tarball → {NUSC_TEST_WORKSPACE} ===")
    t1 = perf_counter()
    tarball = find_test_json_tarball(BACKUP_NUSC_JSON)
    samples_dir, n = extract_test_json_tarball(tarball, NUSC_TEST_WORKSPACE)
    print(f"  {tarball.name}: extracted {n} files into {samples_dir}/")
    print(f"  Phase 1 elapsed: {perf_counter()-t1:.1f}s")

    if skip_raw:
        print("\n--skip-raw set; nuScenes done.")
        return 0

    print(f"\n=== Phase 2: compute needed raw paths (num_cameras={num_cameras}) ===")
    t2 = perf_counter()
    jpgs = collect_nuscenes_paths(samples_dir, num_cameras)
    print(f"  {len(jpgs)} jpgs")
    print(f"  Phase 2 elapsed: {perf_counter()-t2:.1f}s")

    print(f"\n=== Phase 3: raw nuScenes → /data/nuScenes/ ===")
    t3 = perf_counter()
    extract_nuscenes_raw(jpgs, parallelism)
    print(f"  Phase 3 elapsed: {perf_counter()-t3:.1f}s")

    print(f"\n=== Verification (nuScenes) ===")
    miss = sum(1 for p in jpgs if not Path(p).exists())
    print(f"  missing: jpgs={miss}")
    if miss:
        print("  WARNING: some needed files are missing")
        return 1
    print("  all needed nuScenes files present.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("nuplan", "nuscenes", "both"))
    p.add_argument("--parallelism", type=int, default=8,
                   help="Concurrent tarball streams (default 8).")
    p.add_argument("--force", action="store_true",
                   help="rm -rf the test workspace before starting.")
    p.add_argument("--skip-raw", action="store_true",
                   help="Skip raw-data extraction (only build the JSON workspace).")
    p.add_argument("--num-cameras", type=int, choices=(3, 4), default=3,
                   help="Per-dataset camera count. 3 = SFT/RL training set "
                        "(nuPlan F0+L1+R1, nuScenes front+front_left+front_right). "
                        "4 = adds back camera, matches CoT-annotation prompt set.")
    args = p.parse_args()

    rc = 0
    if args.dataset in ("nuplan", "both"):
        rc |= process_nuplan(args.parallelism, args.force, args.skip_raw, args.num_cameras)
    if args.dataset in ("nuscenes", "both"):
        rc |= process_nuscenes(args.parallelism, args.force, args.skip_raw, args.num_cameras)
        # drivelm travels with nuScenes (JSON-only; sync even under --skip-raw)
        print(f"\n========== drivelm sync ==========")
        sync_drivelm()
    return rc


if __name__ == "__main__":
    sys.exit(main())
