import os
import random
import yaml
import argparse
from tqdm import tqdm
import json
from tools.preprocessing.sample_selection_utils import (
    collect_processed_tokens,
    get_dataset_tokens,
    load_token_list,
    partition_indices,
    resolve_indices_from_tokens,
)

CAM_LIST = ['front', 'front_left', 'front_right',
            'back', 'back_left', 'back_right', 'left', 'right']


def resolve_config_path(config_name):
    candidate = os.path.expanduser(config_name)
    if os.path.isfile(candidate):
        return candidate
    return f"./config/{config_name}.yaml"


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_annotation_model(config, backend):
    """Factory function to create the appropriate annotation model backend."""
    if backend == 'vllm':
        from dataset_utils.preprocessing.vllm_cot_annotation_model import CoTAnnotationModel
        return CoTAnnotationModel(config)
    elif backend == 'openai':
        from dataset_utils.preprocessing.openai_cot_annotation_model import OpenAIAnnotationModel
        return OpenAIAnnotationModel(config)
    else:
        raise ValueError(f"Unknown annotation backend: {backend}. Supported: vllm, openai")


def sample_from_batch(batch, index):
    sample = {}
    for key, value in batch.items():
        if isinstance(value, list):
            sample[key] = value[index]
        else:
            sample[key] = value
    return sample


def serialize_trajectory(trajectory):
    try:
        import torch
        if torch.is_tensor(trajectory):
            return trajectory.detach().cpu().tolist()
    except Exception:
        pass
    return trajectory


def build_result(sample, cot_text, dataset_name, fallback_idx):
    token = sample.get("token", f"scene_{fallback_idx}")
    gt_trajectory = serialize_trajectory(sample.get("gt_trajectory", ""))
    his_trajectory = serialize_trajectory(sample.get("his_trajectory", ""))

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

    return token, result


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

    # Load configuration
    config = load_config(resolve_config_path(args.config))

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
        from dataset_utils.preprocessing.nuplan_dataset import (
            NuplanCoTAnnotationDataset,
            DataCollator as DatasetDataCollator,
        )
        val_dataset = NuplanCoTAnnotationDataset(config, processor)
    elif dataset_name == "waymo":
        from dataset_utils.preprocessing.waymo_e2e_dataset import (
            WaymoE2ECoTAnnotationDataset,
            DataCollator as DatasetDataCollator,
        )
        val_dataset = WaymoE2ECoTAnnotationDataset(config, processor)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    dataset_tokens = get_dataset_tokens(val_dataset)
    requested_tokens = None
    missing_tokens = []

    if args.sample_ids_json is not None:
        requested_tokens = load_token_list(args.sample_ids_json)
        indices, missing_tokens = resolve_indices_from_tokens(
            dataset_tokens,
            requested_tokens,
        )
    else:
        indices = list(range(len(val_dataset)))
        random.shuffle(indices)

    selected_indices = partition_indices(indices, args.sample_num, args.num_parts)

    resume_dirs = args.resume_dir or [args.output_dir]
    processed_tokens = collect_processed_tokens(resume_dirs) if args.resume else set()
    if processed_tokens:
        selected_indices = [
            idx for idx in selected_indices if dataset_tokens[idx] not in processed_tokens
        ]

    total_candidates = len(indices)
    shard_total = len(partition_indices(indices, args.sample_num, args.num_parts))
    print(
        f"Selected {total_candidates} samples from dataset of size {len(val_dataset)}."
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
    if args.sample_num != 0:
        print(
            f"Shard {args.sample_num}/{args.num_parts} received {shard_total} samples before resume filtering."
        )
    if args.resume:
        print(
            f"Resume mode found {len(processed_tokens)} processed tokens across {len(resume_dirs)} directories."
        )
        print(
            f"Shard {args.sample_num}/{args.num_parts} will process {len(selected_indices)} remaining samples."
        )

    if not selected_indices:
        print("No samples left to process for this shard.")
        raise SystemExit(0)

    saved_count = 0
    progress = tqdm(
        total=len(selected_indices),
        desc=f"CoT generation ({args.sample_num}/{args.num_parts})",
        unit="scene",
        dynamic_ncols=True,
    )

    if backend == 'vllm':
        from torch.utils.data import DataLoader, Subset

        batch_size = max(int(config.get("batch_size", 1)), 1)
        num_workers = max(int(config.get("num_workers", 0)), 0)
        collator = DatasetDataCollator(processor)

        data_loader_kwargs = {}
        if num_workers > 0:
            data_loader_kwargs["persistent_workers"] = True
            data_loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))

        data_loader = DataLoader(
            Subset(val_dataset, selected_indices),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            shuffle=False,
            **data_loader_kwargs,
        )

        for batch_idx, batch in enumerate(data_loader):
            cot_outputs = model.vlm_inference(batch)
            batch_size_actual = len(batch["token"])

            for sample_idx in range(batch_size_actual):
                sample = sample_from_batch(batch, sample_idx)
                cot_text = cot_outputs[sample_idx] if sample_idx < len(cot_outputs) else ""
                token, result = build_result(
                    sample,
                    cot_text,
                    dataset_name,
                    fallback_idx=batch_idx * batch_size + sample_idx,
                )

                output_path = os.path.join(args.output_dir, f"{token}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                saved_count += 1

            progress.update(batch_size_actual)
            progress.set_postfix_str(
                f"saved={saved_count} batch={batch_size_actual} token={batch['token'][-1]}"
            )
    else:
        for idx in selected_indices:
            sample = val_dataset[idx]
            cot_outputs = model.vlm_inference(sample)
            cot_text = cot_outputs[0] if cot_outputs and len(cot_outputs) > 0 else ""

            token, result = build_result(sample, cot_text, dataset_name, fallback_idx=idx)
            output_path = os.path.join(args.output_dir, f"{token}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            saved_count += 1

            progress.update(1)
            progress.set_postfix_str(f"saved={saved_count} token={token}")

    print(f"All preprocessing data with CoT results have been saved in directory: {args.output_dir}")
