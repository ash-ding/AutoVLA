import os
import json
import argparse
import yaml
import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Subset
from dataset_utils.preprocessing.nuplan_dataset import NuplanCoTAnnotationDataset, DataCollator as NuplanDataCollator
from tools.preprocessing.sample_selection_utils import (
    collect_processed_tokens,
    get_dataset_tokens,
    load_token_list,
    resolve_indices_from_tokens,
)


CAM_LIST = ['front', 'front_left', 'front_right', 
            'back', 'back_left', 'back_right', 'left', 'right']

def load_config(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def resolve_config_path(config_name):
    candidate = os.path.expanduser(config_name)
    if os.path.isfile(candidate):
        return candidate
    return f"./config/{config_name}.yaml"

def process_sample(sample, dataset_name):
    """
    Process a single sample from the dataset.
    """
    token = sample.get("token", "scene_unknown")
    
    gt_trajectory = sample.get("gt_trajectory", "")
    if torch.is_tensor(gt_trajectory):
        gt_trajectory = gt_trajectory.detach().cpu().tolist()
    his_trajectory = sample.get("his_trajectory", "")
    if torch.is_tensor(his_trajectory):
        his_trajectory = his_trajectory.detach().cpu().tolist()

    # Convert velocity and acceleration to lists if they are tensors.
    velocity = sample.get("velocity", "")
    if torch.is_tensor(velocity):
        velocity = velocity.detach().cpu().tolist()
    acceleration = sample.get("acceleration", "")
    if torch.is_tensor(acceleration):
        acceleration = acceleration.detach().cpu().tolist()
    preference_scores = sample.get("preference_scores", "")
    if torch.is_tensor(preference_scores):
        preference_scores = preference_scores.detach().cpu().tolist()
    preference_trajectories = sample.get("preference_trajectories", "") 
    if torch.is_tensor(preference_trajectories):
        preference_trajectories = preference_trajectories.detach().cpu().tolist()
    
    # Build the final output dictionary.
    result = {
        "token": token,
        "dataset_name": dataset_name,
        "cot_output": [],  # No CoT inference needed.
        "velocity": velocity,
        "acceleration": acceleration,
        "instruction": sample.get("instruction", ""),
        "gt_trajectory": gt_trajectory,
        "his_trajectory": his_trajectory
    }
    if dataset_name == "waymo":
            result["preference_scores"] = sample.get("preference_scores", "")
            result["preference_trajectories"] = sample.get("preference_trajectories", "")
    # Include camera paths.
    for side in CAM_LIST:
        key = f"{side}_camera_paths"
        result[key] = sample.get(key, [])
    
    return token, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dataset samples to JSON format using DataLoader for faster processing. "
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Name of the configuration file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the generated JSON files")
    parser.add_argument("--num_workers", type=int, default=32, 
                        help="Number of worker processes for the DataLoader")
    parser.add_argument("--pre_generated_dir", type=str, default=None,
                        help="Directory containing pre-generated scene JSON files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-ids-json",
        type=str,
        default=None,
        help="JSON file containing the tokens to preprocess",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tokens that already have JSON outputs",
    )
    parser.add_argument(
        "--resume-dir",
        action="append",
        default=None,
        help="Directory to scan for existing token JSONs. Can be passed multiple times.",
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    # load pre generated tokens
    print("Collecting pre-generated tokens from JSON sample files...")
    pre_generated_tokens = set()
    if args.pre_generated_dir is not None:
        if os.path.exists(args.pre_generated_dir):
            pre_generated_tokens = collect_processed_tokens([args.pre_generated_dir])
        else:
            print(f"Pre_generated_dir {args.pre_generated_dir} does not exist.")

    # Load configuration.
    config = load_config(resolve_config_path(args.config))
    
    # Initialize the processor and dataset.
    processor = AutoProcessor.from_pretrained(config['pretrained_model_path'], use_fast=True)
    dataset_name = config.get("dataset_name", "")

    if dataset_name == "nuplan":
        dataset = NuplanCoTAnnotationDataset(config, processor)
        collator = NuplanDataCollator(processor)
    elif dataset_name == "waymo":
        from dataset_utils.preprocessing.waymo_e2e_dataset import WaymoE2ECoTAnnotationDataset, DataCollator as WaymoDataCollator
        dataset = WaymoE2ECoTAnnotationDataset(config, processor)
        collator = WaymoDataCollator(processor)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Create output directory if it does not exist.
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_tokens = get_dataset_tokens(dataset)
    requested_tokens = None
    missing_tokens = []

    if args.sample_ids_json is not None:
        requested_tokens = load_token_list(args.sample_ids_json)
        selected_indices, missing_tokens = resolve_indices_from_tokens(
            dataset_tokens,
            requested_tokens,
        )
    else:
        selected_indices = list(range(len(dataset)))

    resume_dirs = list(args.resume_dir or [])
    if args.resume and not resume_dirs:
        resume_dirs.append(args.output_dir)

    processed_tokens = set(pre_generated_tokens)
    if args.resume:
        processed_tokens.update(collect_processed_tokens(resume_dirs))

    if processed_tokens:
        selected_indices = [
            idx for idx in selected_indices if dataset_tokens[idx] not in processed_tokens
        ]

    print(
        f"Selected {len(selected_indices)} samples from dataset of size {len(dataset)}."
    )
    if requested_tokens is not None:
        print(
            f"Loaded {len(requested_tokens)} requested tokens from {args.sample_ids_json}."
        )
        if missing_tokens:
            preview = ", ".join(missing_tokens[:10])
            print(
                f"Skipped {len(missing_tokens)} requested tokens not found in the dataset split. "
                f"Examples: {preview}"
            )
    if processed_tokens:
        print(f"Skipping {len(processed_tokens)} tokens already present in output directories.")
    if not selected_indices:
        print("No samples left to process.")
        raise SystemExit(0)

    # Use DataLoader to load samples concurrently.
    data_loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=False,
    )
    
    for batch in tqdm(data_loader, total=len(selected_indices), desc="Processing samples"):
        # Since batch_size=1, extract the single sample.
        sample = {key: batch[key][0] for key in batch}
        token, result = process_sample(sample, dataset_name)
        
        if token in processed_tokens:
            continue

        output_path = os.path.join(args.output_dir, f"{token}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"All preprocessing data without CoT results have been saved in directory: {args.output_dir}")
