"""
vLLM-backed nuScenes open-loop eval for AutoVLA. ~6x faster than the HF version
(tools/eval/nusc_eval.py) on a single GPU; scales further on 8 GPU.

Prereq: SFT .ckpt must be converted to HF safetensors first:
  python tools/convert_sft_ckpt_to_hf.py \
      --sft_ckpt /backup/runs/sft/<arm>/<ts>/epoch=N-loss=*.ckpt \
      --base_model_path /backup/autovla_models/Qwen2.5-VL-3B-Instruct \
      --codebook_path codebook_cache/agent_vocab.pkl \
      --out_dir /backup/hf_ckpt/<arm>

Usage:
  python tools/eval/nusc_eval_vllm.py \
      --config config/training/eval-4v90-nuscenes.yaml \
      --hf_dir /backup/hf_ckpt/4v90 \
      --seg_data_path /data/nusc_eval_seg_6s \
      --output /data/eval_results/4v90/nuscenes/results.txt \
      --per_sample_dir /data/eval_results/4v90/nuscenes/per_sample \
      [--num_samples 5569] [--batch_size 64]

Notes:
  - Single process; vLLM handles parallelism via continuous batching.
  - Per-sample JSONs include cot_trace + pred_trajectory + gt_trajectory_raw,
    same schema as nusc_eval.py.
  - Aggregate metric matches nusc_eval.py (PlanningMetric from tools/eval/planning_metrics.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.eval._dp_utils import spawn_dp_children, split_indices
from tools.eval.planning_metrics import PlanningMetric
from tools.eval.predict_vllm import VLLMAutoVLAPredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Eval config YAML (e.g. config/training/eval-4v90-nuscenes.yaml)")
    p.add_argument("--hf_dir", required=True,
                   help="HF safetensors dir produced by tools/convert_sft_ckpt_to_hf.py")
    p.add_argument("--seg_data_path", required=True,
                   help="UniAD segmentation .pt dir (/data/nusc_eval_seg_6s)")
    p.add_argument("--output", required=True,
                   help="Aggregate metric table output (results.txt)")
    p.add_argument("--per_sample_dir", default=None,
                   help="Per-sample JSON dump dir; default: <output>.samples/. "
                        "Ignored when --no-save_per_sample_result is set.")
    p.add_argument("--save_per_sample_result",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Save per-sample JSON with cot_trace + trajectories "
                        "(default: True). Pass --no-save_per_sample_result to skip.")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Cap on samples (default: all)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="vLLM submission batch size (continuous batching handles the rest)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--dp_size", type=int, default=1,
                   help="Data parallelism: fork N child processes (1 GPU each). "
                        "Default 1 = single-process. Set >1 to scale across visible GPUs "
                        "(near-linear speedup; see README §5 for rationale).")
    # Internal flags (children only — never set by users directly)
    p.add_argument("--_dp_shard_id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_dp_total", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _run_dp_parent(args):
    """Fork N DP children + aggregate their partial PlanningMetric states."""
    print(f"[nusc_eval_vllm] DP parent: forking {args.dp_size} children", flush=True)
    # Strip --dp_size from forwarded args (children get --_dp_shard_id/--_dp_total instead)
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
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / f"{out_path.stem}.dp_logs"
    failed = spawn_dp_children(args.dp_size, inner, log_dir=str(log_dir))
    if failed:
        raise RuntimeError(f"{failed}/{args.dp_size} DP children failed (see {log_dir})")

    print(f"[nusc_eval_vllm] all {args.dp_size} children done, aggregating...", flush=True)
    sum_obj_col = None; sum_obj_box_col = None; sum_L2 = None; sum_total = None
    sum_processed = 0; sum_no_seg = 0; sum_mask_mm = 0; max_wall = 0.0
    for shard_id in range(args.dp_size):
        partial_path = out_path.with_suffix(f".dp_state.{shard_id}.pt")
        if not partial_path.exists():
            raise RuntimeError(f"shard {shard_id} did not write its partial state at {partial_path}")
        p = torch.load(partial_path, weights_only=False)
        sum_obj_col = p["obj_col"] if sum_obj_col is None else sum_obj_col + p["obj_col"]
        sum_obj_box_col = p["obj_box_col"] if sum_obj_box_col is None else sum_obj_box_col + p["obj_box_col"]
        sum_L2 = p["L2"] if sum_L2 is None else sum_L2 + p["L2"]
        sum_total = p["total"] if sum_total is None else sum_total + p["total"]
        sum_processed += p["processed"]
        sum_no_seg += p["skipped_no_seg"]
        sum_mask_mm += p["skipped_mask_mismatch"]
        max_wall = max(max_wall, p["wall"])

    total_count = int(sum_total)
    eval_result = {
        "obj_col": sum_obj_col / sum_total,
        "obj_box_col": sum_obj_box_col / sum_total,
        "L2": sum_L2 / sum_total,
    }
    print(f"[nusc_eval_vllm] aggregate: {sum_processed} processed, "
          f"{sum_no_seg} no_seg, {sum_mask_mm} mask_mismatch, "
          f"total samples in metric={total_count}, wall={max_wall:.1f}s", flush=True)

    stp3 = PrettyTable()
    stp3.title = "STP3's Definition Planning Metrics (Cumulative Average)"
    stp3.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    for k, v in eval_result.items():
        stp3.add_row([k] + ["%.4f" % float(v[: i + 1].mean()) for i in range(min(len(v), 6))])
    print(stp3)
    uniad_tab = PrettyTable()
    uniad_tab.title = "UniAD's Definition Planning Metrics (Per-Timestep)"
    uniad_tab.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    for k, v in eval_result.items():
        uniad_tab.add_row([k] + ["%.4f" % float(v[i]) for i in range(min(len(v), 6))])
    print(uniad_tab)
    with open(out_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluation Results (vLLM backend, DP={args.dp_size}) - {sum_processed} samples "
                f"(skipped no_seg={sum_no_seg}, mask_mismatch={sum_mask_mm})\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"HF dir: {args.hf_dir}\n")
        f.write(f"Wall time (max across shards): {max_wall:.1f}s\n")
        f.write(f"{'='*60}\n\n")
        f.write(str(stp3) + "\n\n")
        f.write(str(uniad_tab) + "\n")
    print(f"\nResults saved to {out_path}", flush=True)
    # Clean up partial state files
    for shard_id in range(args.dp_size):
        out_path.with_suffix(f".dp_state.{shard_id}.pt").unlink()


def main():
    args = parse_args()

    # DP parent: fork children + aggregate. Never falls through to single-shard logic.
    if args.dp_size > 1 and args._dp_shard_id is None:
        return _run_dp_parent(args)

    device = args.device
    is_dp_child = args._dp_shard_id is not None
    shard_id = args._dp_shard_id if is_dp_child else 0
    total_shards = args._dp_total if is_dp_child else 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_per_sample_result:
        per_sample_dir = Path(args.per_sample_dir) if args.per_sample_dir \
            else out_path.with_suffix("").parent / (out_path.stem + ".samples")
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        # Sanity probe: confirm writable before kicking off expensive inference
        _probe = per_sample_dir / ".write_probe"
        _probe.write_text("ok"); _probe.unlink()
    else:
        per_sample_dir = None
        print("[nusc_eval_vllm] --no-save_per_sample_result: skipping per-sample dump", flush=True)

    config = yaml.safe_load(open(args.config))

    if is_dp_child:
        print(f"[nusc_eval_vllm shard {shard_id}/{total_shards}] config={args.config}, hf_dir={args.hf_dir}", flush=True)
    else:
        print(f"[nusc_eval_vllm] config={args.config}, hf_dir={args.hf_dir}", flush=True)
    print(f"[nusc_eval_vllm] output={out_path}", flush=True)
    print(f"[nusc_eval_vllm] per_sample_dir={per_sample_dir}", flush=True)

    # Load dataset (same path as HF nusc_eval): SFTDataset val split provides scene list
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoProcessor
    from dataset_utils.sft_dataset import SFTDataset

    processor = AutoProcessor.from_pretrained(args.hf_dir, use_fast=True)
    train_dataset = SFTDataset(config["data"]["val"], config["model"], processor)
    total_sample_num = len(train_dataset.scenes)
    if args.num_samples is not None:
        total_sample_num = min(args.num_samples, total_sample_num)
    # DP shard slice (round-robin across shards)
    sample_indices = list(range(total_sample_num))[split_indices(total_sample_num, shard_id, total_shards)]
    sample_num = len(sample_indices)
    if is_dp_child:
        print(f"[nusc_eval_vllm shard {shard_id}/{total_shards}] "
              f"got {sample_num}/{total_sample_num} samples", flush=True)
    else:
        print(f"[nusc_eval_vllm] evaluating {sample_num} samples", flush=True)

    # Initialize vLLM predictor (heavy; once per process)
    predictor = VLLMAutoVLAPredictor(
        config, args.hf_dir,
        dataset_name="nuscenes",
        gpu_memory_utilization=args.gpu_mem_util,
    )
    predictor.initialize()

    # Planning metric (DDP-aware accumulators; we're single-process so .to(device) is enough)
    planning_metrics = PlanningMetric(n_future=6).to(device)

    # Pre-build input_features for all samples (CPU work)
    print(f"[nusc_eval_vllm] building input_features for {sample_num} samples...", flush=True)
    samples = []
    for idx in sample_indices:
        scene_path, _ = train_dataset.scenes[idx]
        with open(scene_path) as f:
            scene_data = json.load(f)
        # Feature builders produce the input_features dict; reuse them so any
        # downstream tweak (camera selection, instruction renaming) stays consistent.
        input_features = {}
        target_trajectory = {}
        for builder in train_dataset._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in train_dataset._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))
        samples.append({
            "scene_data": scene_data,
            "input_features": input_features,
            "target_trajectory": target_trajectory,
        })
    print(f"[nusc_eval_vllm] built {len(samples)} input_features", flush=True)

    # Run vLLM in batches (continuous batching inside vLLM handles within-batch overlap)
    t_start = time.time()
    skipped_no_seg = 0
    skipped_mask_mismatch = 0
    processed = 0
    for batch_start in range(0, len(samples), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(samples))
        batch = samples[batch_start:batch_end]
        t_batch = time.time()
        predictions = predictor.predict_batch(
            [s["input_features"] for s in batch], greedy=False,
        )
        print(f"[nusc_eval_vllm] batch {batch_start}-{batch_end}: "
              f"{time.time()-t_batch:.1f}s ({(batch_end-batch_start)/(time.time()-t_batch):.1f} samples/s)",
              flush=True)

        # Post-process + metric update + dump
        for s, (pred_trajectory, output_text) in zip(batch, predictions):
            scene_data = s["scene_data"]
            target_trajectory = s["target_trajectory"]
            token = scene_data["token"]

            seg_path = Path(args.seg_data_path) / f"{token}.pt"
            if not seg_path.exists():
                skipped_no_seg += 1
                continue

            gt_raw = target_trajectory["gt_pos_raw"].to(device)
            pred_xy = pred_trajectory[:, :2].to(device)
            uniad = torch.load(seg_path, map_location="cpu")
            sdc_planning_mask = uniad["sdc_planning_mask"].to(device=device, dtype=gt_raw.dtype)
            segmentation = uniad["segmentation"].to(device=device, dtype=gt_raw.dtype)

            gt_uni = gt_raw.unsqueeze(0).clone()
            gt_uni[:, :, [0, 1]] = gt_uni[:, :, [1, 0]]
            gt_uni[:, :, 0] = -gt_uni[:, :, 0]
            gt_uni = gt_uni.unsqueeze(0)
            pred_uni = pred_xy.unsqueeze(0).clone()
            pred_uni[:, :, [0, 1]] = pred_uni[:, :, [1, 0]]
            pred_uni[:, :, 0] = -pred_uni[:, :, 0]
            pred_uni = pred_uni.unsqueeze(0)

            cache_future_mask = torch.tensor(scene_data["future_mask"][:6], device=device)
            sdc_mask = sdc_planning_mask[0, 0, :, 0]
            if not torch.allclose(cache_future_mask, sdc_mask):
                skipped_mask_mismatch += 1
                continue

            planning_metrics(
                pred_uni[0, :, :6, :],
                gt_uni[0, :, :6, :],
                sdc_planning_mask[0, :, :6, :2],
                segmentation[:, [1, 2, 3, 4, 5, 6]],
            )

            # Per-sample dump (opt-out via --no-save_per_sample_result)
            if args.save_per_sample_result:
                rec = {
                    "token": token,
                    "cot_trace": output_text,
                    "pred_trajectory": pred_trajectory.detach().cpu().tolist(),
                    "gt_trajectory_raw": gt_raw.detach().cpu().tolist(),
                    "future_mask": scene_data.get("future_mask"),
                    "config": args.config,
                    "hf_dir": args.hf_dir,
                    "backend": "vllm",
                }
                with open(per_sample_dir / f"{token}.json", "w") as f:
                    json.dump(rec, f)
            processed += 1

    total_wall = time.time() - t_start
    print(f"\n[nusc_eval_vllm{'shard ' + str(shard_id) if is_dp_child else ''}] "
          f"inference + metric done in {total_wall:.1f}s "
          f"({sample_num/total_wall:.2f} samples/s)", flush=True)
    print(f"  processed: {processed}, skipped_no_seg: {skipped_no_seg}, "
          f"skipped_mask_mismatch: {skipped_mask_mismatch}", flush=True)

    # DP child: save partial metric state for parent to aggregate, then exit.
    # PlanningMetric uses `dist_reduce_fx='sum'` so summing is the correct merge.
    if is_dp_child:
        partial = {
            "obj_col": planning_metrics.obj_col.detach().cpu(),
            "obj_box_col": planning_metrics.obj_box_col.detach().cpu(),
            "L2": planning_metrics.L2.detach().cpu(),
            "total": planning_metrics.total.detach().cpu(),
            "processed": processed,
            "skipped_no_seg": skipped_no_seg,
            "skipped_mask_mismatch": skipped_mask_mismatch,
            "wall": total_wall,
        }
        partial_path = out_path.with_suffix(f".dp_state.{shard_id}.pt")
        torch.save(partial, partial_path)
        print(f"[nusc_eval_vllm shard {shard_id}] saved partial state to {partial_path}", flush=True)
        return  # don't write results.txt; parent will

    eval_result = planning_metrics.compute()

    stp3 = PrettyTable()
    stp3.title = "STP3's Definition Planning Metrics (Cumulative Average)"
    stp3.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    for k, v in eval_result.items():
        row = [k] + ["%.4f" % float(v[: i + 1].mean()) for i in range(min(len(v), 6))]
        stp3.add_row(row)
    print(stp3)

    uniad_tab = PrettyTable()
    uniad_tab.title = "UniAD's Definition Planning Metrics (Per-Timestep)"
    uniad_tab.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    for k, v in eval_result.items():
        row = [k] + ["%.4f" % float(v[i]) for i in range(min(len(v), 6))]
        uniad_tab.add_row(row)
    print(uniad_tab)

    with open(out_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluation Results (vLLM backend) - {processed} samples "
                f"(skipped no_seg={skipped_no_seg}, mask_mismatch={skipped_mask_mismatch})\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"HF dir: {args.hf_dir}\n")
        f.write(f"Wall time: {total_wall:.1f}s ({sample_num/total_wall:.2f} samples/s)\n")
        f.write(f"{'='*60}\n\n")
        f.write(str(stp3) + "\n\n")
        f.write(str(uniad_tab) + "\n")
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
