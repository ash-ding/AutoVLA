"""
vLLM-backed nuPlan PDM-score eval for AutoVLA.

Replaces navsim's run_pdm_score_cot.py + autovla_agent.py for the inference
side. Architecture:
  1. Build scene_loader + metric_cache_loader (navsim, single-process)
  2. Build input_features for every token (CPU)
  3. Submit all prompts to vLLM in batches -> get trajectories (continuous batching)
  4. Run PDM simulator + scorer in CPU thread pool over (token, trajectory) pairs
  5. Aggregate scores -> CSV; dump per-sample JSON with cot + scores

Prereq:
  - HF safetensors dir from tools/convert_sft_ckpt_to_hf.py
  - navtest metric cache (tools/run_navtest_metric_caching.sh or
    scripts/run_navtest_metric_caching.sh)

Usage:
  python tools/eval/nuplan_pdm_eval_vllm.py \
      --config config/training/qwen2.5-vl-3B-mix-sft-10k-rlib1.0-4v90.yaml \
      --hf_dir /backup/hf_ckpt/4v90 \
      --metric_cache_path /data/navsim_exp/navtest_metric_cache \
      --json_data_path /data/nuplan_test/test_samples_12146 \
      --sensor_data_path /data/nuPlan/sensor_blobs/test \
      --output_dir /data/eval_results/4v90/nuplan \
      [--num_tokens 12146] [--batch_size 32] [--pdm_workers 16]
"""
import argparse
import json
import os
import pickle
import lzma
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# navsim is editable-installed; treat as regular import below
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "navsim"))

from navsim.common.dataclasses import SensorConfig, Trajectory as NSTrajectory
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tools.eval._dp_utils import spawn_dp_children, split_indices
from tools.eval.predict_vllm import VLLMAutoVLAPredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="SFT YAML (e.g. config/training/qwen2.5-vl-3B-mix-sft-10k-rlib1.0-4v90.yaml)")
    p.add_argument("--hf_dir", required=True,
                   help="HF safetensors dir from tools/convert_sft_ckpt_to_hf.py")
    p.add_argument("--metric_cache_path", required=True,
                   help="navtest metric cache dir (e.g. /data/navsim_exp/navtest_metric_cache)")
    p.add_argument("--json_data_path", default="/data/nuplan_test/test_samples_12146")
    p.add_argument("--sensor_data_path", default="/data/nuPlan/sensor_blobs/test")
    p.add_argument("--output_dir", required=True,
                   help="Output dir (CSV + per_sample JSONs land here under <timestamp>/)")
    p.add_argument("--train_test_split", default="navtest",
                   help="One of {navtest, navmini, ...} — picks navsim/.../train_test_split/<name>.yaml")
    p.add_argument("--num_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32,
                   help="vLLM submission chunk size")
    p.add_argument("--pdm_workers", type=int, default=16,
                   help="ThreadPoolExecutor workers for PDM sim (CPU-bound)")
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--dataset_name", default="nuplan", choices=["nuplan", "nuscenes", "waymo"])
    p.add_argument("--nuplan_side_field", default=None,
                   help="'left' or 'front_left' (default: from config.model.nuplan_side_field)")
    p.add_argument("--save_per_sample_result",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Save per-sample JSON with cot_trace + trajectories + scores "
                        "(default: True). Pass --no-save_per_sample_result to skip.")
    p.add_argument("--dp_size", type=int, default=1,
                   help="Data parallelism: fork N child processes (1 GPU each). "
                        "Default 1. Set >1 to scale across visible GPUs.")
    p.add_argument("--_dp_shard_id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_dp_total", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_dp_timestamp", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def build_pdm_score_for_token(metric_cache_path, model_trajectory, simulator, scorer):
    """Single-token PDM sim. CPU-bound. Safe to run in a thread (releases GIL on I/O)."""
    with lzma.open(metric_cache_path, "rb") as f:
        metric_cache = pickle.load(f)
    return pdm_score(
        metric_cache=metric_cache,
        model_trajectory=model_trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )


def resolve_cache_path(metric_cache_paths, token, cache_root):
    """Mirror the resolve_metric_cache_path logic in run_pdm_score_cot.py."""
    p = Path(metric_cache_paths[token])
    if p.exists():
        return p
    candidate = Path(cache_root).joinpath(*p.parts[-4:])
    return candidate if candidate.exists() else p


def _run_dp_parent(args):
    """Fork N DP children + concatenate their per-shard PDM row JSONs into one CSV."""
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[nuplan_pdm_eval_vllm] DP parent: forking {args.dp_size} children "
          f"(shared output dir = {out_dir})", flush=True)

    # Strip --dp_size, inject shared timestamp
    inner = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False; continue
        if a == "--dp_size":
            skip_next = True; continue
        if a.startswith("--dp_size="):
            continue
        inner.append(a)
    inner += ["--_dp_timestamp", timestamp]

    log_dir = out_dir / "dp_logs"
    failed = spawn_dp_children(args.dp_size, inner, log_dir=str(log_dir))
    if failed:
        raise RuntimeError(f"{failed}/{args.dp_size} DP children failed (see {log_dir})")

    print(f"[nuplan_pdm_eval_vllm] all {args.dp_size} children done, aggregating rows...",
          flush=True)
    all_rows = []
    for shard_id in range(args.dp_size):
        rows_path = out_dir / f"dp_rows.{shard_id}.json"
        if not rows_path.exists():
            raise RuntimeError(f"shard {shard_id} did not write its rows at {rows_path}")
        with open(rows_path) as f:
            all_rows.extend(json.load(f))
    df = pd.DataFrame(all_rows)
    n_ok = df["valid"].sum() if "valid" in df else 0
    n_fail = len(df) - n_ok
    avg = df.drop(columns=["token", "valid"]).mean(skipna=True)
    avg["token"] = "average"
    avg["valid"] = df["valid"].all()
    df.loc[len(df)] = avg
    csv_path = out_dir / f"{timestamp}.csv"
    df.to_csv(csv_path)
    print(f"[nuplan_pdm_eval_vllm] CSV: {csv_path}", flush=True)
    print(f"  Successful: {n_ok}, Failed: {n_fail}", flush=True)
    if "score" in df.columns:
        print(f"  Mean PDM score: {df.iloc[:-1]['score'].mean():.4f}", flush=True)
    # Clean up partial row files
    for shard_id in range(args.dp_size):
        (out_dir / f"dp_rows.{shard_id}.json").unlink()


def main():
    args = parse_args()

    # DP parent: fork children + aggregate, never falls through.
    if args.dp_size > 1 and args._dp_shard_id is None:
        return _run_dp_parent(args)

    is_dp_child = args._dp_shard_id is not None
    shard_id = args._dp_shard_id if is_dp_child else 0
    total_shards = args._dp_total if is_dp_child else 1

    # Children share the SAME timestamp dir so per_sample/ + dp_rows/ collide cleanly
    timestamp = args._dp_timestamp or datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_per_sample_result:
        per_sample_dir = out_dir / "per_sample"
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        _probe = per_sample_dir / ".write_probe"
        _probe.write_text("ok"); _probe.unlink()
    else:
        per_sample_dir = None
        print(f"[nuplan_pdm_eval_vllm] --no-save_per_sample_result: "
              f"skipping per-sample JSON dump", flush=True)

    print(f"[nuplan_pdm_eval_vllm] output: {out_dir}", flush=True)

    # Load config for predictor
    config = yaml.safe_load(open(args.config))
    nuplan_side_field = (args.nuplan_side_field or
                        config["model"].get("nuplan_side_field", "left"))
    print(f"[nuplan_pdm_eval_vllm] dataset={args.dataset_name}, "
          f"nuplan_side_field={nuplan_side_field}", flush=True)

    # Build scene loader (gets token list)
    # Use navsim Hydra config for scene_filter
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    cfg_path = str(Path(__file__).resolve().parents[2] /
                   "navsim/navsim/planning/script/config/common/train_test_split")
    with initialize_config_dir(version_base=None, config_dir=cfg_path):
        split_cfg = compose(config_name=args.train_test_split)
    scene_filter: SceneFilter = instantiate(split_cfg.scene_filter)

    navsim_log_path = Path(os.environ.get("OPENSCENE_DATA_ROOT", "/data/nuPlan")) / "navsim_logs" / split_cfg.data_split
    print(f"[nuplan_pdm_eval_vllm] navsim_log_path: {navsim_log_path}", flush=True)

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))

    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    if args.num_tokens is not None:
        tokens = tokens[:args.num_tokens]
    total_tokens = len(tokens)
    # DP shard slice (round-robin)
    tokens = tokens[split_indices(total_tokens, shard_id, total_shards)]
    if is_dp_child:
        print(f"[nuplan_pdm_eval_vllm shard {shard_id}/{total_shards}] "
              f"got {len(tokens)}/{total_tokens} tokens", flush=True)
    else:
        print(f"[nuplan_pdm_eval_vllm] tokens to evaluate: {len(tokens)}", flush=True)

    # Build PDM simulator + scorer
    cfg_default_path = str(Path(__file__).resolve().parents[2] /
                            "navsim/navsim/planning/script/config/pdm_scoring")
    with initialize_config_dir(version_base=None, config_dir=cfg_default_path):
        pdm_cfg = compose(config_name="default_run_pdm_score")
    simulator: PDMSimulator = instantiate(pdm_cfg.simulator)
    scorer: PDMScorer = instantiate(pdm_cfg.scorer)
    print(f"[nuplan_pdm_eval_vllm] PDM simulator/scorer ready", flush=True)

    # Build input_features for every token (load corresponding test JSON)
    print(f"[nuplan_pdm_eval_vllm] building input_features...", flush=True)
    samples = []
    for token in tokens:
        json_path = Path(args.json_data_path) / f"{token}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            scene_data = json.load(f)
        # For nuplan, predict_vllm._build_messages picks side cameras
        # based on nuplan_side_field; we pass all 5 channels here.
        images = {
            "front_camera": scene_data["front_camera_paths"][:4],
            "left_camera": scene_data.get("left_camera_paths", [])[:4],
            "right_camera": scene_data.get("right_camera_paths", [])[:4],
            "front_left_camera": scene_data.get("front_left_camera_paths", [])[:4],
            "front_right_camera": scene_data.get("front_right_camera_paths", [])[:4],
        }
        inp = {
            "token": token,
            "dataset_name": "nuplan",
            "sensor_data_path": args.sensor_data_path,
            "images": images,
            "vehicle_velocity": scene_data.get("velocity", [0.0, 0.0]),
            "vehicle_acceleration": scene_data.get("acceleration", [0.0, 0.0]),
            "driving_command": scene_data.get("instruction", "drive safely"),
        }
        samples.append({"token": token, "input_features": inp, "scene_data": scene_data})
    print(f"[nuplan_pdm_eval_vllm] built {len(samples)} input_features", flush=True)

    # Initialize vLLM
    predictor = VLLMAutoVLAPredictor(
        config, args.hf_dir,
        dataset_name="nuplan",
        nuplan_side_field=nuplan_side_field,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    predictor.initialize()

    # Phase 1: vLLM inference in batches
    print(f"\n[nuplan_pdm_eval_vllm] Phase 1: vLLM inference ({args.batch_size} per batch)", flush=True)
    trajectories = {}   # token -> Trajectory object (navsim Trajectory wrapping poses)
    cot_traces = {}
    t_infer = time.time()
    for batch_start in range(0, len(samples), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(samples))
        batch = samples[batch_start:batch_end]
        t_b = time.time()
        results = predictor.predict_batch(
            [s["input_features"] for s in batch], greedy=False,
        )
        for s, (traj_tensor, cot) in zip(batch, results):
            # navsim PDM expects a Trajectory dataclass with .poses (np array)
            # of shape (num_poses, 3). traj_tensor is torch.Tensor [num_poses, 3].
            sampling = TrajectorySampling(
                num_poses=traj_tensor.shape[0],
                interval_length=config["model"]["trajectory"]["interval_length"],
            )
            traj = NSTrajectory(
                poses=traj_tensor.detach().cpu().numpy().astype(np.float64),
                trajectory_sampling=sampling,
            )
            trajectories[s["token"]] = traj
            cot_traces[s["token"]] = cot
        print(f"  batch {batch_start}-{batch_end}: {time.time()-t_b:.1f}s "
              f"({(batch_end-batch_start)/(time.time()-t_b):.2f} sample/s)", flush=True)
    t_infer_total = time.time() - t_infer
    print(f"[nuplan_pdm_eval_vllm] inference done in {t_infer_total:.1f}s "
          f"({len(samples)/t_infer_total:.2f} sample/s)", flush=True)

    # Phase 2: PDM sim in thread pool
    print(f"\n[nuplan_pdm_eval_vllm] Phase 2: PDM sim with {args.pdm_workers} threads", flush=True)
    t_pdm = time.time()
    rows = []

    def _score_one(s):
        token = s["token"]
        try:
            mc_path = resolve_cache_path(metric_cache_loader.metric_cache_paths,
                                         token, args.metric_cache_path)
            pdm_result = build_pdm_score_for_token(
                mc_path, trajectories[token], simulator, scorer,
            )
            score_row = {"token": token, "valid": True}
            score_row.update(asdict(pdm_result))
            # Per-sample JSON dump (opt-out via --no-save_per_sample_result)
            if args.save_per_sample_result:
                rec = {
                    "token": token,
                    "cot_trace": cot_traces[token],
                    "pred_trajectory": trajectories[token].poses.tolist(),
                    "gt_trajectory": s["scene_data"].get("gt_trajectory"),
                    "his_trajectory": s["scene_data"].get("his_trajectory"),
                    "scores": asdict(pdm_result),
                    "backend": "vllm",
                }
                with open(per_sample_dir / f"{token}.json", "w") as f:
                    json.dump(rec, f)
            return score_row
        except Exception as e:
            print(f"  [PDM fail] {token}: {e}", flush=True)
            traceback.print_exc()
            return {"token": token, "valid": False}

    with ThreadPoolExecutor(max_workers=args.pdm_workers) as ex:
        futures = {ex.submit(_score_one, s): s["token"] for s in samples}
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 200 == 0:
                print(f"  PDM {i+1}/{len(samples)} ({(i+1)/(time.time()-t_pdm):.1f} token/s)", flush=True)
    t_pdm_total = time.time() - t_pdm
    print(f"[nuplan_pdm_eval_vllm] PDM done in {t_pdm_total:.1f}s "
          f"({len(samples)/t_pdm_total:.2f} token/s)", flush=True)

    # DP child: write raw rows for parent to concatenate, then exit.
    if is_dp_child:
        rows_path = out_dir / f"dp_rows.{shard_id}.json"
        with open(rows_path, "w") as f:
            json.dump(rows, f)
        print(f"[nuplan_pdm_eval_vllm shard {shard_id}] saved {len(rows)} rows to {rows_path}",
              flush=True)
        return

    # Aggregate (single-process mode)
    df = pd.DataFrame(rows)
    n_ok = df["valid"].sum() if "valid" in df else 0
    n_fail = len(df) - n_ok
    avg = df.drop(columns=["token", "valid"]).mean(skipna=True)
    avg["token"] = "average"
    avg["valid"] = df["valid"].all()
    df.loc[len(df)] = avg
    csv_path = out_dir / f"{timestamp}.csv"
    df.to_csv(csv_path)
    print(f"\n[nuplan_pdm_eval_vllm] CSV: {csv_path}", flush=True)
    print(f"  Successful: {n_ok}, Failed: {n_fail}", flush=True)
    if "score" in df.columns:
        print(f"  Mean PDM score: {df.iloc[:-1]['score'].mean():.4f}", flush=True)
    print(f"\nTotal wall: inference={t_infer_total:.1f}s + pdm={t_pdm_total:.1f}s "
          f"= {t_infer_total+t_pdm_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
