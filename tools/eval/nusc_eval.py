"""
NuScenes Evaluation Script for AutoVLA.

This script evaluates the AutoVLA model on NuScenes validation data,
computing planning metrics such as L2 distance and collision rate.

Supports both single-GPU and multi-GPU evaluation:

  Single-GPU (README default):
      python tools/eval/nusc_eval.py --config ... --checkpoint ... --seg_data_path ...

  Multi-GPU (8x A100, ~8x speedup):
      torchrun --nproc_per_node=8 tools/eval/nusc_eval.py --config ... --checkpoint ... --seg_data_path ...

In multi-GPU mode each rank evaluates a disjoint shard of test samples; the
PlanningMetric (torchmetrics) automatically all-reduces the accumulator state
on compute(), so the final table on rank 0 is identical to a single-GPU run.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root and navsim to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from transformers import AutoProcessor

from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA
from tools.eval.planning_metrics import PlanningMetric



def load_config(file_path):
    """Load configuration from YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AutoVLA on NuScenes validation data")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--seg_data_path", type=str, required=True,
                        help="Path to the segmentation data directory for evaluation")
    parser.add_argument("--output", type=str, default="planning_table.txt",
                        help="Output file for results (default: planning_table.txt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output for each sample")
    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed setup (when launched via torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend='nccl')
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
    else:
        device = args.device

    is_main = (rank == 0)

    # Load configuration
    config = load_config(args.config)

    # Initialize processor
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)

    # Build data config for SFTDataset from config
    train_dataset = SFTDataset(config['data']['val'], config['model'], processor)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = SFTAutoVLA(config)
    model.autovla.vlm.resize_token_embeddings(len(processor.tokenizer))

    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    model.autovla.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.autovla.device = device  # Update the device attribute for predict()
    model.eval()

    # Initialize planning metrics. PlanningMetric inherits torchmetrics.Metric;
    # its add_state(..., dist_reduce_fx="sum") declares DDP-aware accumulators,
    # so compute() will all-reduce across ranks transparently.
    planning_metrics = PlanningMetric(n_future=6).to(device)

    # Determine number of samples + shard across ranks (round-robin)
    sample_num = len(train_dataset.scenes)
    if args.num_samples is not None:
        sample_num = min(args.num_samples, sample_num)

    rank_indices = list(range(rank, sample_num, world_size))
    if is_main:
        print(f"Evaluating {sample_num} samples on {world_size} GPU(s) "
              f"(~{len(rank_indices)} per rank)...")

    # Evaluate this rank's shard
    iterator = tqdm(rank_indices, desc=f"rank{rank}", disable=not is_main)
    for idx in iterator:
        # scenes is a list of tuples: (scene_path, sensor_data_path)
        scene_path, _ = train_dataset.scenes[idx]
        
        # Load scene data
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
        
        # Get features and targets
        input_features = {}
        target_trajectory = {}
        
        start_time = time.time()

        for builder in train_dataset._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in train_dataset._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))

        # Model prediction
        pred_trajectory, output_text = model.autovla.predict(input_features)
        if pred_trajectory == [] or len(pred_trajectory) == 0:
            continue
        
        if args.verbose:
            print(f"Output: {output_text}")
            print(f"Predicted trajectory: {pred_trajectory}")
            print(f"Time taken: {time.time() - start_time:.3f} seconds")
       
        # Get ground truth trajectory
        gt_raw_trajectory = target_trajectory["gt_pos_raw"]
        pred_xy = pred_trajectory[:, :2].to(gt_raw_trajectory.device)

        # Load segmentation data for collision evaluation
        seg_path = Path(args.seg_data_path) / f"{scene_data['token']}.pt"
        if not seg_path.exists():
            print(f"Warning: Segmentation file not found: {seg_path}")
            continue
            
        uniad_data = torch.load(seg_path, map_location="cpu")
        sdc_planning_mask = uniad_data['sdc_planning_mask'].to(gt_raw_trajectory.dtype)
        segmentation = uniad_data['segmentation'].to(gt_raw_trajectory.dtype)

        # Transform GT trajectory to UniAD coordinate system
        gt_traj_uniadcoord = gt_raw_trajectory.unsqueeze(0).clone()
        gt_traj_uniadcoord[:, :, [0, 1]] = gt_traj_uniadcoord[:, :, [1, 0]]
        gt_traj_uniadcoord[:, :, 0] = -gt_traj_uniadcoord[:, :, 0]
        gt_traj_uniadcoord = gt_traj_uniadcoord.unsqueeze(0)

        # Transform predicted trajectory to UniAD coordinate system
        pred_traj_uniadcoord = pred_xy.unsqueeze(0).clone()
        pred_traj_uniadcoord[:, :, [0, 1]] = pred_traj_uniadcoord[:, :, [1, 0]]
        pred_traj_uniadcoord[:, :, 0] = -pred_traj_uniadcoord[:, :, 0]
        pred_traj_uniadcoord = pred_traj_uniadcoord.unsqueeze(0)

        # Validate future mask consistency
        cache_future_mask = torch.tensor(scene_data['future_mask'][:6])
        sdc_mask = sdc_planning_mask[0, 0, :, 0]
        if not torch.allclose(cache_future_mask, sdc_mask):
            print(f"Warning: Mismatch between mask values for sample {idx}")
            continue

        # Compute planning metrics
        planning_metrics(
            pred_traj_uniadcoord[0, :, :6, :], 
            gt_traj_uniadcoord[0, :, :6, :], 
            sdc_planning_mask[0, :, :6, :2], 
            segmentation[:, [1, 2, 3, 4, 5, 6]]
        )
    
    # Calculate and print overall statistics
    # compute() triggers torchmetrics' all-reduce across ranks for any state
    # declared with dist_reduce_fx — must be called on every rank.
    eval_result = planning_metrics.compute()

    if not is_main:
        if is_distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Create table with STP3's definition (cumulative average)
    planning_tab_stp3 = PrettyTable()
    planning_tab_stp3.title = "STP3's Definition Planning Metrics (Cumulative Average)"
    planning_tab_stp3.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    
    for key, value in eval_result.items():
        row_value = [key]
        for i in range(min(len(value), 6)):
            row_value.append("%.4f" % float(value[:i + 1].mean()))
        planning_tab_stp3.add_row(row_value)
    print(planning_tab_stp3)

    # Create table with UniAD's definition (per-timestep)
    planning_tab_uniad = PrettyTable()
    planning_tab_uniad.title = "UniAD's Definition Planning Metrics (Per-Timestep)"
    planning_tab_uniad.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    
    for key, value in eval_result.items():
        row_value = [key]
        for i in range(min(len(value), 6)):
            row_value.append("%.4f" % float(value[i]))
        planning_tab_uniad.add_row(row_value)
    print(planning_tab_uniad)

    # Save results to file
    with open(args.output, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluation Results - {sample_num} samples ({world_size} GPU)\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"{'='*60}\n\n")
        f.write(str(planning_tab_stp3) + "\n\n")
        f.write(str(planning_tab_uniad) + "\n")

    print(f"\nResults saved to {args.output}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
