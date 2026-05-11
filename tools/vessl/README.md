# tools/vessl/

Scripts that hydrate a `/data` workspace from `/backup`-side tarballs. They
do **only data movement** (extract sample-JSON tarballs, then selectively
extract the raw images/pkls those JSONs reference). All preprocessing —
generating sample JSONs, building scaling splits, downloading raw nuPlan /
nuScenes — happens elsewhere; these scripts assume the backup side is
already populated.

Two entry points:

| Script | Use for |
|---|---|
| `load_nuplan_nuscenes_mix_train.py` | Training:hydrate one of the 4-bucket train-mix scales (10k / 50k / 100k / 185k) |
| `load_nuplan_nuscenes_test.py`      | Eval / test:hydrate the nuPlan and/or nuScenes test set |

Both are resumable — re-running with the same args is a no-op for files
already on `/data`.

Both support `--datasets {nuplan,nuscenes,both}` (mix-train) /
`--dataset {nuplan,nuscenes,both}` (test) to hydrate a single side of the
mix. Whenever nuScenes is selected, `/backup/drivelm/` is also synced to
`/data/drivelm/` (DriveLM Q&A JSONs travel with the nuScenes side).

---

## Backup-side layout (assumed)

```
/backup/
├── nuplan_nuscenes_train_mix/
│   └── nuplan_nuscenes_train_mix_<scale>/
│       ├── nuplan_action_only_samples_<N>.tar.zst
│       ├── nuplan_nl_reasoning_samples_<MODEL>_<N>.tar.zst
│       ├── nuscenes_action_only_samples_<N>.tar.zst
│       ├── nuscenes_nl_reasoning_samples_<N>.tar.zst
│       └── scaling_<scale>_token_list.json
├── nuplan_test/
│   └── test_samples_<N>.tar.zst
├── nuscenes_test/
│   └── test_samples_<N>.tar.zst
├── drivelm/
│   ├── v1_1_train_nus.json                                  # ~185 MB
│   └── v1_1_val_nus_q_only.json                             # ~10 MB
└── raw_dataset_tarball/
    ├── nuplan/
    │   ├── openscene_metadata_trainval.tgz                      # train pkls
    │   ├── openscene_sensor_trainval_camera_{0..199}.tgz        # train jpgs (200)
    │   ├── openscene_metadata_test.tgz                          # test pkls
    │   └── openscene_sensor_test_camera_{0..31}.tgz             # test jpgs (32)
    └── nuscenes/
        ├── v1.0-trainval_meta.tgz
        ├── v1.0-trainval{01..10}_blobs.tgz                       # 10 blobs
        ├── v1.0-test_meta.tgz                                    # (test rare; trainval covers val)
        └── v1.0-test_blobs.tgz
```

## /data-side layout (produced)

```
/data/
├── nuplan_nuscenes_train_mix_<scale>/                             # produced by load_nuplan_nuscenes_mix_train
│   ├── nuPlan/{action_only_samples_*, nl_reasoning_samples_*}/    # present iff nuplan selected
│   ├── nuScenes/{action_only_samples_*, nl_reasoning_samples_*}/  # present iff nuscenes selected
│   └── scaling_<scale>_token_list.json
├── nuplan_test/test_samples_<N>/                                  # produced by load_nuplan_nuscenes_test
├── nuscenes_test/test_samples_<N>/                                # produced by load_nuplan_nuscenes_test
├── drivelm/                                                       # synced whenever nuScenes is selected
│   ├── v1_1_train_nus.json
│   └── v1_1_val_nus_q_only.json
├── nuPlan/
│   ├── navsim_logs/
│   │   ├── trainval/<log>.pkl                                     # train pkls
│   │   └── test/<log>.pkl                                         # test pkls
│   ├── sensor_blobs/
│   │   ├── trainval/<log>/CAM_*/<jpg>.jpg                         # train jpgs
│   │   └── test/<log>/CAM_*/<jpg>.jpg                             # test jpgs
│   └── maps/                                                      # static, not touched by these scripts
└── nuScenes/
    ├── samples/CAM_*/<jpg>.jpg                                    # train + val jpgs share this dir
    └── v1.0-trainval/
```

---

## Camera pruning ("level 3")

Only the 3 cameras `SFTDataset` / `AutoVLAAgent` actually consume are extracted:

| Dataset | JSON fields | nuPlan dir / nuScenes dir |
|---|---|---|
| nuPlan   | `front_camera_paths`, `left_camera_paths`, `right_camera_paths` | `CAM_F0`, `CAM_L1`, `CAM_R1` |
| nuScenes | `front_camera_paths`, `front_left_camera_paths`, `front_right_camera_paths` | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT` |

`--num-cameras 4` adds the back camera (`CAM_B0` / `CAM_BACK`), matching
the CoT-annotation prompt set. Other cameras (lidar, radar, sweeps) are
always skipped to save disk and bandwidth.

---

## `load_nuplan_nuscenes_mix_train.py`

Extract one of the 4 scaling bundles into `/data/nuplan_nuscenes_train_mix_<scale>/`.

### Arguments

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--scale {10k,50k,100k,185k}` | yes | — | Which scaling bundle to hydrate. |
| `--model <name>` | yes | — | Substring matching the nuPlan CoT tarball name (e.g. `Qwen2.4_VL_72B_Instruct_AWQ`). |
| `--datasets {nuplan,nuscenes,both}` | no | `both` | Restrict hydration to a single side of the mix. `nuscenes` (or `both`) also syncs `/backup/drivelm/` → `/data/drivelm/`. |
| `--parallelism N` | no | 8 | Concurrent tarball streams. 8 is optimal — diminishing returns past that. |
| `--num-cameras {3,4}` | no | 3 | 3 = training set; 4 = adds back camera, matches CoT-annotation prompt set. |
| `--force` | no | false | `rm -rf` the per-scale `/data/nuplan_nuscenes_train_mix_<scale>/` before starting. |
| `--skip-raw` | no | false | Only extract the sample-JSON tarballs (Phase 1) and sync drivelm if nuScenes is selected. Skip raw image / pkl extraction (Phase 2-4). Useful when `/data/nuPlan` and `/data/nuScenes` are already hydrated by an earlier scale. |

### Phases

1. **Phase 1** — extract the selected sample-JSON tarballs (parallel) into the per-scale workspace. (`drivelm` sync runs here too if nuScenes is selected.)
2. **Phase 2** — walk extracted JSONs to collect needed `<jpg>` and `<pkl>` paths.
3. **Phase 3** — selectively extract nuPlan jpgs (200 cam tarballs) + pkls (1 metadata tarball) into `/data/nuPlan/{sensor_blobs,navsim_logs}/trainval/`. Skipped when `--datasets nuscenes`.
4. **Phase 4** — selectively extract nuScenes jpgs (10 trainval blobs + 1 meta tarball) into `/data/nuScenes/`. Skipped when `--datasets nuplan`.

### Typical commands

```bash
# Hydrate the 10k workspace (smallest, fastest — for smoke test)
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 10k --model Qwen2.4_VL_72B_Instruct_AWQ

# Largest workspace (full 185k mix) — takes ~30-40 min on cold /data
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 185k --model Qwen2.4_VL_72B_Instruct_AWQ

# Only nuPlan side (no nuScenes JSONs, no drivelm sync, skip Phase 4)
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 50k --model Qwen2.4_VL_72B_Instruct_AWQ --datasets nuplan

# Only nuScenes side (also syncs drivelm)
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 50k --model Qwen2.4_VL_72B_Instruct_AWQ --datasets nuscenes

# Just the JSONs (skip raw); raw assumed already hydrated from a previous run
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 50k --model Qwen2.4_VL_72B_Instruct_AWQ --skip-raw

# Reset and re-extract the 100k workspace
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 100k --model Qwen2.4_VL_72B_Instruct_AWQ --force
```

### Time estimates (cold cache, parallelism=8)

| Scale | Phase 1 | Phase 3 (nuPlan) | Phase 4 (nuScenes) | Total |
|---|---|---|---|---|
| 10k   | ~5 s  | ~7 min if cold (instant if cached) | ~18 min | ~25 min cold / ~5s warm |
| 50k   | ~25 s | usually warm  | ~22 min | ~22 min |
| 100k  | ~45 s | usually warm  | ~15 min | ~15 min |
| 185k  | ~85 s | usually warm  | ~12 min | ~13 min |

(Phase 3-4 mostly skip-exist after the first scale runs, since
`/data/nuPlan` and `/data/nuScenes` are shared across scales.)
DriveLM sync (~194 MB, 2 JSONs) takes a few seconds the first time and is a no-op thereafter.

### Disk footprint (per scale, isolated)

What `/data` consumes if you run **only** that scale on a fresh disk
(no other scales hydrated, no test sets). Numbers from actual extracted
data: nuPlan jpg avg ≈ 217 KB, nuPlan pkl avg ≈ 10 MB, nuScenes jpg avg
≈ 155 KB. Camera pruning (3 cameras / 8) is already applied.

| Scale | nuPlan jpgs | nuPlan pkls | nuScenes jpgs | sample JSONs | **subtotal** |
|---|---:|---:|---:|---:|---:|
| 10k  | 107,676 (~22 GB)   | 1,189 (~12 GB) | 11,424 (~1.7 GB) | 61 MB  | **~36 GB** |
| 50k  | 538,392 (~111 GB)  | 1,250 (~12 GB) | 41,949 (~6.4 GB) | 302 MB | **~130 GB** |
| 100k | 1,076,772 (~223 GB)| 1,250 (~12 GB) | 57,954 (~8.8 GB) | 604 MB | **~245 GB** |
| 185k | 1,995,384 (~412 GB)| 1,250 (~12 GB) | 63,390 (~9.6 GB) | 1.1 GB | **~435 GB** |

Plus a shared one-off cost of **~4.2 GB** for nuPlan maps (1.4 GB),
nuScenes `v1.0-trainval/` metadata (2.5 GB), nuScenes maps (6 MB), and
DriveLM JSONs (~194 MB) — paid on the first run that hydrates each side,
free thereafter.

⚠️ Cumulative cost when stacking scales is **non-additive** — bigger
scales are strict supersets of smaller ones for sample JSONs (`10k ⊂ 50k
⊂ 100k ⊂ 185k`), and they share most raw images at the log level. Going
from 10k → 185k adds **~400 GB**, not 565 GB.

---

## `load_nuplan_nuscenes_test.py`

Hydrate the nuPlan navtest set, the nuScenes val ("test") set, or both
into `/data/{nuplan,nuscenes}_test/`.

### Arguments

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--dataset {nuplan,nuscenes,both}` | yes | — | Which test set to hydrate. `both` does nuPlan first, then nuScenes. `nuscenes` (or `both`) also syncs `/backup/drivelm/` → `/data/drivelm/`. |
| `--parallelism N` | no | 8 | Concurrent tarball streams. |
| `--num-cameras {3,4}` | no | 3 | Per-dataset camera count. |
| `--force` | no | false | `rm -rf` the per-dataset test workspace before starting. |
| `--skip-raw` | no | false | Only extract the JSON tarball (Phase 1). Skip pkl/jpg extraction. DriveLM sync still runs if nuScenes is selected. |

There is **no `--scale`, no `--model`, no `--run-preprocess`** — sample
JSONs are pre-baked in `/backup/{nuplan,nuscenes}_test/test_samples_*.tar.zst`,
and the script's only job is data movement.

### Phases (per dataset)

1. **Phase 1** — extract the sample-JSON tarball into `/data/<ds>_test/test_samples_<N>/`.
2. **Phase 2** — walk JSONs to collect needed jpg paths (and per-log pkl paths for nuPlan).
3. **Phase 3** — selectively extract raw files into the canonical `/data` layout:
   - **nuPlan**: 32 `openscene_sensor_test_camera_*.tgz` → `/data/nuPlan/sensor_blobs/test/`. Plus `openscene_metadata_test.tgz` → `/data/nuPlan/navsim_logs/test/`.
   - **nuScenes**: 10 `v1.0-trainval[0-9]*_blobs.tgz` → `/data/nuScenes/samples/CAM_*/` (val frames live in trainval blobs). Plus `v1.0-trainval_meta.tgz` → `/data/nuScenes/v1.0-trainval/`.

After the nuScenes phases, `/backup/drivelm/` is synced to `/data/drivelm/`.

### Typical commands

```bash
# Hydrate both test sets in one go (~22 min cold) + drivelm
python tools/vessl/load_nuplan_nuscenes_test.py --dataset both

# nuPlan navtest only (no drivelm)
python tools/vessl/load_nuplan_nuscenes_test.py --dataset nuplan

# nuScenes val ("test") only — raw lives in trainval blobs; also syncs drivelm
python tools/vessl/load_nuplan_nuscenes_test.py --dataset nuscenes

# Just the JSON workspaces, no raw extraction
python tools/vessl/load_nuplan_nuscenes_test.py --dataset both --skip-raw

# Force re-extraction
python tools/vessl/load_nuplan_nuscenes_test.py --dataset nuplan --force
```

### Time estimates (cold cache, parallelism=8)

| Dataset | Phase 1 | Phase 3 | Total |
|---|---|---|---|
| nuPlan   | ~60 s | ~4 min (32 cam tarballs, ~120 GB scanned) | ~6 min |
| nuScenes | ~30 s | ~13 min (10 trainval blobs, ~300 GB scanned) | ~14 min |

### Disk footprint (per test set, isolated)

What `/data` consumes if you run **only** that test set on a fresh
disk. Same average sizes as above.

| Test set | jpgs | pkls | sample JSONs | **subtotal** |
|---|---:|---:|---:|---:|
| nuPlan test (navtest, 12,146 samples) | 53,616 (~11 GB) | 136 (~1 GB) | 66 MB | **~13 GB** |
| nuScenes test (val, 5,569 samples)    | 18,057 (~2.8 GB) | — | 23 MB | **~3 GB** |

The nuPlan test and train sets share the same `sensor_blobs/` and
`navsim_logs/` roots but live in disjoint `test/` vs `trainval/` subdirs —
no file sharing between splits. The
nuScenes val frames live in `samples/CAM_*/` alongside train frames; if
the train workspace is already hydrated, **most val jpgs are free**
(verified on our run: 0 missing → ~3 GB still extracted because val
frames are temporally adjacent but distinct from train frames).

If you only need test sets (no training), **you do not need to run
`load_nuplan_nuscenes_mix_train.py`** — `load_nuplan_nuscenes_test.py` is
fully self-contained.

---

## Combined disk-budget cheat sheet

Common scenarios, fresh `/data`:

| Goal | What to run | `/data` needed |
|---|---|---|
| 10k SFT only | `load_nuplan_nuscenes_mix_train --scale 10k` | **~40 GB** |
| 50k SFT only | `load_nuplan_nuscenes_mix_train --scale 50k` | **~134 GB** |
| 100k SFT only | `load_nuplan_nuscenes_mix_train --scale 100k` | **~250 GB** |
| 185k SFT only | `load_nuplan_nuscenes_mix_train --scale 185k` | **~440 GB** |
| Full scaling sweep (10k+50k+100k+185k) | 4 × `load_nuplan_nuscenes_mix_train` | **~440 GB** (185k subsumes the rest) |
| nuPlan navtest eval only | `load_nuplan_nuscenes_test --dataset nuplan` | **~17 GB** (incl. nuPlan maps + nusc meta) |
| nuScenes test eval only | `load_nuplan_nuscenes_test --dataset nuscenes` | **~6 GB** (incl. nusc meta + maps + drivelm) |
| 10k SFT + both test sets | both scripts | **~55 GB** |
| 185k SFT + both test sets | both scripts | **~455 GB** |

(`/data` on this box is 1.5 TB → 185k + everything fits with ~1 TB to spare.)

---

## What each script *does not* do

- **No preprocessing**. They never invoke `nocot_sample_generation.py` or
  `cot_sample_generation.py`. Sample JSONs must already be in
  `/backup/<bucket>/test_samples_*.tar.zst` before running.
- **No path rewriting**. JSONs in the backup tarballs already reference
  the canonical `/data/...` paths. If you add a new bucket, run
  `tools/preprocessing/rewrite_sample_paths.py` first, then rebuild the
  tarball.
- **No metric-cache generation**. If you need NAVSIM PDMS, run
  `navsim/.../run_metric_caching.py` separately.
- **No download**. They assume `/backup/raw_dataset_tarball/` is fully
  populated. To add the test sensor tarballs (32 × 3.5 GB):
  `navsim/download/download_test.sh` (or wget the URLs into
  `/backup/raw_dataset_tarball/nuplan/`).

---

## Daily workflow cheat-sheet

```bash
# Activate env once per shell session
conda activate autovla
source .envrc

# === On a fresh /data (or after eviction) ===

# 1. Pick a training scale and hydrate it (both nuPlan + nuScenes + drivelm)
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 10k --model Qwen2.4_VL_72B_Instruct_AWQ

# 2. Hydrate test sets (nuPlan + nuScenes + drivelm)
python tools/vessl/load_nuplan_nuscenes_test.py --dataset both

# 3. Train
python tools/run_sft.py --config training/qwen2.5-vl-3B-mix-sft-10k-action-only

# === Switching scales ===
# Phase 3-4 will mostly skip (raw shared across scales). Only the new
# scale's sample JSONs get extracted (~30-90 s).
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 50k --model Qwen2.4_VL_72B_Instruct_AWQ

# === Single-side hydration ===
# nuPlan-only train mix at a fresh /data (no nuScenes work, no drivelm)
python tools/vessl/load_nuplan_nuscenes_mix_train.py \
    --scale 100k --model Qwen2.4_VL_72B_Instruct_AWQ --datasets nuplan
```

## Troubleshooting

- **`AssertionError: /cluster_dataset/nuPlan/maps does not exist`** — the
  upstream `.envrc` had `NUPLAN_MAPS_ROOT=/cluster_dataset/nuPlan/maps`.
  Already fixed in this checkout to `/data/nuPlan/maps`. If you see this,
  re-source `.envrc`.
- **`No tarball matching ...`** — The backup-side bundle is missing.
  Check `/backup/nuplan_nuscenes_train_mix/<scale>/` for the 4 expected
  tarballs, or `/backup/{nuplan,nuscenes}_test/` for the test bundles.
- **`needed: N jpgs (M missing)` after run** — Some jpgs referenced in
  the JSON are not in any tarball. Inspect a missing path; usually means
  the JSON references a camera/log that wasn't included in the
  `--scale` bundle's selection. Should not happen for the canonical
  bundles.
- **Slow extraction (~130 MB/s instead of ~500 MB/s)** — `open_streaming_tar`
  uses subprocess `zstdcat`/`pigz` to escape the GIL; if you see slow
  rates, check that those binaries are on PATH (`pigz` falls back to
  `gzip` automatically, which is single-threaded).
