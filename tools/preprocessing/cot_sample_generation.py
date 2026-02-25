import os
import random
import yaml
import argparse
from tqdm import tqdm
import json

CAM_LIST = ['front', 'front_left', 'front_right',
            'back', 'back_left', 'back_right', 'left', 'right']


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_annotation_model(config, backend):
    """Factory function to create the appropriate annotation model backend."""
    if backend == 'vllm':
        from dataset_utils.preprocessing.cot_annotation_model import CoTAnnotationModel
        return CoTAnnotationModel(config)
    elif backend == 'openai':
        from dataset_utils.preprocessing.openai_annotation_model import OpenAIAnnotationModel
        return OpenAIAnnotationModel(config)
    else:
        raise ValueError(f"Unknown annotation backend: {backend}. Supported: vllm, openai")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None,
                        help='Annotation backend: vllm or openai (overrides config)')
    parser.add_argument("--seed", type=int, default=42, help='Random seed which identifies the sample generation')
    parser.add_argument("--sample_num", type=int, default=1, help='Sample number to process')
    parser.add_argument("--num_parts", type=int, default=1, help='Number of parts to split the dataset into')
    args = parser.parse_args()

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Determine backend: CLI arg > config > default (vllm)
    backend = args.backend or config.get('annotation_backend', 'vllm')

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if backend == 'vllm':
        # IMPORTANT: CoTAnnotationModel (vLLM) must be imported and initialized BEFORE
        # pytorch_lightning and nuplan/navsim imports, because PL monkey-patches torch.compile
        # which breaks vLLM's CUDA graph compilation.
        model = create_annotation_model(config, backend)

        # Now safe to import PL and dataset classes
        import torch
        from pytorch_lightning import seed_everything
        from transformers import AutoProcessor
        seed_everything(args.seed)

        processor = AutoProcessor.from_pretrained(config['pretrained_model_path'], use_fast=True)
    else:
        # API backends have no import order constraints
        import torch
        from pytorch_lightning import seed_everything
        seed_everything(args.seed)

        model = create_annotation_model(config, backend)
        processor = None  # API backends don't need Qwen processor

    # Model, dataset, and dataloader
    dataset_name = config.get("dataset_name", "")

    if dataset_name == "nuplan":
        from dataset_utils.preprocessing.nuplan_dataset import NuplanCoTAnnotationDataset
        val_dataset = NuplanCoTAnnotationDataset(config, processor)
    elif dataset_name == "waymo":
        from dataset_utils.preprocessing.waymo_e2e_dataset import WaymoE2ECoTAnnotationDataset
        val_dataset = WaymoE2ECoTAnnotationDataset(config, processor)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    indices = list(range(len(val_dataset)))
    random.shuffle(indices)

    # seperate the dataset to num_parts samples
    if args.sample_num != 0:
        total_len = len(indices)
        part_len = total_len // args.num_parts
        start_idx = (args.sample_num - 1) * part_len
        end_idx = args.sample_num * part_len if args.sample_num < args.num_parts else total_len
        selected_indices = indices[start_idx:end_idx]
    else:
        selected_indices = indices

    for idx in tqdm(selected_indices, desc=f"Processing scenes (Sample {args.sample_num} of {args.num_parts} parts)"):
        sample = val_dataset[idx]

        cot_outputs = model.vlm_inference(sample)
        cot_text = cot_outputs[0] if cot_outputs and len(cot_outputs) > 0 else ""

        token = sample.pop("token") if "token" in sample else f"scene_{idx}"
        gt_trajectory = sample.get("gt_trajectory", "")
        if torch.is_tensor(gt_trajectory):
            gt_trajectory = gt_trajectory.detach().cpu().tolist()
        his_trajectory = sample.get("his_trajectory", "")
        if torch.is_tensor(his_trajectory):
            his_trajectory = his_trajectory.detach().cpu().tolist()

        result = {
            "token": token,
            "dataset_name": dataset_name,
            "cot_output": cot_text,
            "velocity": sample.get("velocity", ""),
            "acceleration": sample.get("acceleration", ""),
            "instruction": sample.get("instruction", ""),
            "gt_trajectory": gt_trajectory,
            "his_trajectory": his_trajectory,
            **{f"{side}_camera_paths": sample.get(f"{side}_camera_paths", []) for side in CAM_LIST}
        }

        if dataset_name == "waymo":
            result["preference_scores"] = sample.get("preference_scores", "")
            result["preference_trajectories"] = sample.get("preference_trajectories", "")

        output_path = os.path.join(args.output_dir, f"{token}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"All preprocessing data with CoT results have been saved in directory: {args.output_dir}")
