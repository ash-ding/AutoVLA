"""
Symbolic CoT Sample Generation — RLIB 4-Stage Structured Reasoning.

Generates Chain-of-Thought annotations in the symbolic format
(PERCEPTION → OPERATIONS → FACTS → RULES → ACTION) defined by RLIB,
instead of the free-form CoT used by cot_sample_generation.py.

Usage:
    # OpenAI (GPT-4o-mini quick test)
    python tools/preprocessing/symbolic_cot_sample_generation.py \
        --config dataset/symbolic-cot-gpt4o-mini \
        --output_dir ./test_symbolic_cot \
        --backend openai

    # vLLM (local model)
    python tools/preprocessing/symbolic_cot_sample_generation.py \
        --config dataset/symbolic-cot-nuplan-mini \
        --output_dir ./symbolic_cot_output \
        --backend vllm
"""

import os
import random
import yaml
import argparse
from tqdm import tqdm
import json

from dataset_utils.preprocessing.symbolic_cot_prompts import (
    get_symbolic_cot_prompt,
    ego_state_to_qualitative,
    action_string_to_symbolic,
)

CAM_LIST = ['front', 'front_left', 'front_right',
            'back', 'back_left', 'back_right', 'left', 'right']


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


class SymbolicPromptWrapper:
    """Wraps an existing CoT annotation dataset, replacing the free-form
    prompt with the RLIB symbolic prompt. Supports both vLLM and OpenAI backends."""

    def __init__(self, base_dataset, rlib_dir, processor=None, nl_cot_dir=None, free_rules=False):
        self.base = base_dataset
        self.rlib_dir = rlib_dir
        self.processor = processor  # vLLM needs this; OpenAI passes None
        self.nl_cot_dir = nl_cot_dir
        self.free_rules = free_rules

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        # Get fut_ego_action (added to nuplan_dataset.py; waymo already has it)
        fut_ego_action = sample.get("fut_ego_action", "")

        # Quantize ego state
        ego_qual = ego_state_to_qualitative(
            sample["velocity"], sample["acceleration"], sample["instruction"]
        )

        # Load NL CoT reference if available
        nl_cot_ref = None
        if self.nl_cot_dir:
            token = sample.get("token", "")
            nl_cot_path = os.path.join(self.nl_cot_dir, f"{token}.json")
            if os.path.exists(nl_cot_path):
                with open(nl_cot_path) as f:
                    nl_data = json.load(f)
                nl_cot_ref = nl_data.get("cot_output", "")

        # Build symbolic prompt
        sym_prompt = get_symbolic_cot_prompt(
            self.rlib_dir,
            fut_ego_action,
            ego_qual["speed"],
            ego_qual["acceleration"],
            ego_qual["instruction"],
            nl_cot_reference=nl_cot_ref,
            use_predefined_rules=not self.free_rules,
        )

        # Replace the last content item in user message (was get_cot_reasoning_prompt)
        sample["messages"][-1]["content"][-1] = sym_prompt

        # vLLM backend: re-encode after prompt replacement
        if self.processor is not None:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(sample["messages"])
            text = self.processor.apply_chat_template(
                sample["messages"], tokenize=False,
                add_generation_prompt=True, add_vision_id=True,
            )
            sample["text"] = text
            sample["image_inputs"] = image_inputs
            sample["video_inputs"] = video_inputs

        return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symbolic CoT sample generation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None,
                        help='Annotation backend: vllm or openai (overrides config)')
    parser.add_argument("--rlib_dir", type=str, default="./RLIB",
                        help='Path to RLIB directory')
    parser.add_argument("--nl-cot-dir", type=str, default=None,
                        help='Directory of NL CoT JSONs (token.json with cot_output field) to use as reference')
    parser.add_argument("--free-rules", action="store_true", default=False,
                        help='Disable predefined RLIB rules; LLM composes rules freely from facts')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--num_parts", type=int, default=1)
    args = parser.parse_args()

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Determine backend
    backend = args.backend or config.get('annotation_backend', 'vllm')
    rlib_dir = args.rlib_dir or config.get('rlib_dir', './RLIB')

    # Override max_tokens if specified in config (symbolic output may be longer)
    if 'max_tokens' in config:
        # Will be picked up by the annotation model
        pass

    os.makedirs(args.output_dir, exist_ok=True)

    if backend == 'vllm':
        # IMPORTANT: vLLM must be initialized BEFORE pytorch_lightning imports
        model = create_annotation_model(config, backend)

        import torch
        from pytorch_lightning import seed_everything
        from transformers import AutoProcessor
        seed_everything(args.seed)

        processor = AutoProcessor.from_pretrained(config['pretrained_model_path'], use_fast=True)
    else:
        import torch
        from pytorch_lightning import seed_everything
        seed_everything(args.seed)

        model = create_annotation_model(config, backend)
        processor = None

    # Load dataset
    dataset_name = config.get("dataset_name", "")

    if dataset_name == "nuplan":
        from dataset_utils.preprocessing.nuplan_dataset import NuplanCoTAnnotationDataset
        base_dataset = NuplanCoTAnnotationDataset(config, processor)
    elif dataset_name == "waymo":
        from dataset_utils.preprocessing.waymo_e2e_dataset import WaymoE2ECoTAnnotationDataset
        base_dataset = WaymoE2ECoTAnnotationDataset(config, processor)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Wrap with symbolic prompt
    nl_cot_dir = getattr(args, 'nl_cot_dir', None)
    free_rules = getattr(args, 'free_rules', False)
    val_dataset = SymbolicPromptWrapper(base_dataset, rlib_dir, processor, nl_cot_dir=nl_cot_dir, free_rules=free_rules)

    # Partition
    indices = list(range(len(val_dataset)))
    random.shuffle(indices)

    if args.sample_num != 0:
        total_len = len(indices)
        part_len = total_len // args.num_parts
        start_idx = (args.sample_num - 1) * part_len
        end_idx = args.sample_num * part_len if args.sample_num < args.num_parts else total_len
        selected_indices = indices[start_idx:end_idx]
    else:
        selected_indices = indices

    # Initialize symbolic validator
    from models.symbolic_rules import (
        SymbolicSchema, SymbolicParser, SymbolicValidator, ParseError,
    )
    schema = SymbolicSchema(rlib_dir)
    sym_parser = SymbolicParser(schema)
    sym_validator = SymbolicValidator(schema, grounding_strictness="warn")

    # Statistics
    stats = {
        "total": 0,
        "parse_success": 0,
        "valid": 0,
        "grounding_scores": [],
    }

    for idx in tqdm(selected_indices, desc=f"Symbolic CoT (Part {args.sample_num}/{args.num_parts})"):
        sample = val_dataset[idx]
        cot_outputs = model.vlm_inference(sample)
        cot_text = cot_outputs[0] if cot_outputs and len(cot_outputs) > 0 else ""

        # Validate symbolic output
        symbolic_valid = False
        symbolic_violations = []
        grounding_warnings = []
        grounding_score = 0.0

        if cot_text:
            try:
                parsed = sym_parser.parse(cot_text)
                is_valid, violations, g_warnings = sym_validator.validate(parsed)
                symbolic_valid = is_valid
                symbolic_violations = violations
                grounding_warnings = g_warnings
                stats["parse_success"] += 1

                if is_valid:
                    stats["valid"] += 1

                # Compute grounding score for True facts
                checkable = 0
                grounded = 0
                entity_index = sym_validator._build_entity_index(parsed.entities)
                ego_ops = sym_validator._extract_ego_ops(parsed.operations)
                for fact in parsed.facts:
                    if not fact.value:
                        continue
                    fg = schema.get_fact_grounding(fact.name)
                    if fg is None:
                        continue
                    checkable += 1
                    if any(c.kind == "judgment" for c in fg.conditions):
                        grounded += 1
                    elif sym_validator._evaluate_grounding(fg.conditions, entity_index, ego_ops):
                        grounded += 1
                grounding_score = grounded / checkable if checkable > 0 else 1.0
                stats["grounding_scores"].append(grounding_score)

            except ParseError as e:
                symbolic_violations = [f"Parse error: {e}"]

        stats["total"] += 1

        # Build result
        token = sample.get("token", f"scene_{idx}")
        gt_trajectory = sample.get("gt_trajectory", "")
        if torch.is_tensor(gt_trajectory):
            gt_trajectory = gt_trajectory.detach().cpu().tolist()
        his_trajectory = sample.get("his_trajectory", "")
        if torch.is_tensor(his_trajectory):
            his_trajectory = his_trajectory.detach().cpu().tolist()

        result = {
            "token": token,
            "dataset_name": dataset_name,
            "cot_format": "symbolic",
            "cot_output": cot_text,
            "symbolic_valid": symbolic_valid,
            "symbolic_violations": symbolic_violations,
            "grounding_warnings": grounding_warnings,
            "grounding_score": grounding_score,
            "velocity": sample.get("velocity", ""),
            "acceleration": sample.get("acceleration", ""),
            "instruction": sample.get("instruction", ""),
            "gt_trajectory": gt_trajectory,
            "his_trajectory": his_trajectory,
            **{f"{side}_camera_paths": sample.get(f"{side}_camera_paths", []) for side in CAM_LIST},
        }

        if dataset_name == "waymo":
            result["preference_scores"] = sample.get("preference_scores", "")
            result["preference_trajectories"] = sample.get("preference_trajectories", "")

        output_path = os.path.join(args.output_dir, f"{token}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Symbolic CoT Generation Summary")
    print("=" * 60)
    print(f"Total processed:       {stats['total']}")
    print(f"Parse success:         {stats['parse_success']} ({100 * stats['parse_success'] / max(stats['total'], 1):.1f}%)")
    print(f"Validation success:    {stats['valid']} ({100 * stats['valid'] / max(stats['total'], 1):.1f}%)")
    if stats["grounding_scores"]:
        avg_gs = sum(stats["grounding_scores"]) / len(stats["grounding_scores"])
        print(f"Avg grounding score:   {avg_gs:.3f}")
    print(f"Output directory:      {args.output_dir}")
    print("=" * 60)
