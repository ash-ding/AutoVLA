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

import argparse
import base64
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml
from tqdm import tqdm

from dataset_utils.preprocessing.symbolic_cot_prompts import (
    get_symbolic_cot_prompt,
    ego_state_to_qualitative,
)
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
    candidate = Path(os.path.expanduser(config_name))
    if candidate.is_file():
        return str(candidate)
    return f"./config/{config_name}.yaml"


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file) or {}
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


def normalize_vector2(value):
    if isinstance(value, (int, float)):
        return [float(value), 0.0]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return [0.0, 0.0]
        if len(value) == 1:
            return [float(value[0]), 0.0]
        return [float(value[0]), float(value[1])]
    return [0.0, 0.0]


def serialize_value(value):
    try:
        import torch
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def cot_reference_to_text(cot_output):
    if cot_output is None:
        return None
    if isinstance(cot_output, list):
        return "\n".join(str(item) for item in cot_output if item is not None)
    return str(cot_output)


def extract_action_from_cot(cot_output):
    if isinstance(cot_output, list) and cot_output:
        return str(cot_output[-1]).strip()
    if not isinstance(cot_output, str):
        return ""

    match = re.search(
        r"Best Driving Action:\s*(?:\*\*)?([^\n*]+)",
        cot_output,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def get_action_instruction_from_trajectory(gt_trajectory):
    try:
        import numpy as np

        points = np.asarray([[p[0], p[1]] for p in gt_trajectory], dtype=float)
        if len(points) < 2:
            return "move forward with a constant speed"

        velocity = np.diff(points, axis=0) / 0.5
        velocity = np.concatenate([velocity, velocity[-1:]], axis=0)

        constant_eps = 0.8
        stop_eps = 0.3
        velos = np.linalg.norm(velocity, axis=1)
        cur_velo = velos[0]
        end_velo = velos[-1]

        if cur_velo < stop_eps and end_velo < stop_eps:
            speed_meta = "stop"
        elif end_velo < stop_eps:
            speed_meta = "a deceleration to zero"
        elif abs(end_velo - cur_velo) < constant_eps:
            speed_meta = "a constant speed"
        elif end_velo > cur_velo:
            speed_meta = "a quick acceleration" if end_velo > 2 * cur_velo else "an acceleration"
        else:
            speed_meta = "a quick deceleration" if cur_velo > 2 * end_velo else "a deceleration"

        if speed_meta == "stop":
            return "STOP"

        forward_th = 2.0
        lane_changing_th = 4.0
        final_lat = points[-1, 1]

        if np.all(np.abs(points[:, 1]) < forward_th):
            behavior_meta = "move forward"
        elif final_lat > 0:
            behavior_meta = "turn left" if abs(final_lat) > lane_changing_th else "change lane to left"
        elif final_lat < 0:
            behavior_meta = "turn right" if abs(final_lat) > lane_changing_th else "change lane to right"
        else:
            behavior_meta = "move forward"

        return f"{behavior_meta} with {speed_meta}"
    except Exception:
        return "move forward with a constant speed"


def infer_future_action(sample, use_nl_cot_action=False):
    if sample.get("fut_ego_action"):
        return sample["fut_ego_action"]

    if use_nl_cot_action:
        action = extract_action_from_cot(sample.get("cot_output"))
        if action:
            return action

    return get_action_instruction_from_trajectory(sample.get("gt_trajectory", []))


def parse_path_prefix_maps(raw_maps):
    mappings = []
    for raw in raw_maps or []:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --path-prefix-map value {raw!r}; expected FROM=TO."
            )
        src, dst = raw.split("=", 1)
        mappings.append((src.rstrip("/"), dst.rstrip("/")))
    return mappings


def resolve_data_path(path, prefix_maps):
    expanded = Path(os.path.expanduser(path))
    if expanded.exists():
        return expanded

    for src, dst in prefix_maps:
        if path == src or path.startswith(f"{src}/"):
            suffix = path[len(src):].lstrip("/")
            candidate = Path(os.path.expanduser(dst)) / suffix
            if candidate.exists():
                return candidate

    if path.startswith("/data/") and Path("data").exists():
        candidate = Path("data") / path[len("/data/"):]
        if candidate.exists():
            return candidate

    return expanded


def image_to_data_uri(path, prefix_maps):
    image_path = resolve_data_path(path, prefix_maps)
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def camera_order_for_sample(sample):
    dataset_name = str(sample.get("dataset_name", "")).lower()
    if dataset_name == "nuplan":
        return ["front", "left", "right"]
    if dataset_name == "nuscenes":
        return ["front", "front_left", "front_right"]
    return [side for side in CAM_LIST if sample.get(f"{side}_camera_paths")]


def encode_sample_for_vllm(sample, processor):
    from qwen_vl_utils import process_vision_info

    process_vision_kwargs = {
        "return_video_kwargs": True,
    }
    patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
    if patch_size is not None:
        process_vision_kwargs["image_patch_size"] = patch_size

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            sample["messages"],
            return_video_metadata=True,
            **process_vision_kwargs,
        )
    except TypeError:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            sample["messages"],
            **process_vision_kwargs,
        )
    text = processor.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    sample["text"] = text
    sample["image_inputs"] = image_inputs
    sample["video_inputs"] = video_inputs
    sample["mm_processor_kwargs"] = video_kwargs or {}
    return sample


def build_preprocessed_messages(
    sample,
    rlib_dir,
    prefix_maps,
    free_rules,
    use_nl_cot_reference=False,
):
    fut_ego_action = infer_future_action(
        sample,
        use_nl_cot_action=use_nl_cot_reference,
    )
    velocity = normalize_vector2(sample.get("velocity", [0.0, 0.0]))
    acceleration = normalize_vector2(sample.get("acceleration", [0.0, 0.0]))
    instruction = str(sample.get("instruction", "keep forward"))

    ego_qual = ego_state_to_qualitative(
        velocity,
        acceleration,
        instruction,
        rlib_dir,
    )

    content = [
        {
            "type": "text",
            "text": (
                "Multi-view, multi-frame camera images are provided as short videos. "
                "Use them with the ego vehicle state and route instruction to reason about driving."
            ),
        }
    ]

    for side in camera_order_for_sample(sample):
        paths = sample.get(f"{side}_camera_paths") or []
        if not paths:
            continue
        content.extend([
            {
                "type": "text",
                "text": (
                    f"The video is from the {side.replace('_', ' ')} camera, "
                    "capturing the recent history of that view."
                ),
            },
            {
                "type": "video",
                "min_pixels": 400 * 400,
                "max_pixels": 400 * 400,
                "video": [image_to_data_uri(path, prefix_maps) for path in paths],
            },
        ])

    content.append({
        "type": "text",
        "text": (
            f"The ego vehicle's current velocity is {velocity[0]:.3f} m/s at x-direction "
            f"and {velocity[1]:.3f} m/s at y-direction. "
            f"The ego vehicle's current acceleration is {acceleration[0]:.3f} m/s^2 "
            f"at x-direction and {acceleration[1]:.3f} m/s^2 at y-direction. "
            f"The current driving command instruction of ego vehicle is: {instruction}, "
            "indicating the intended route direction."
        ),
    })

    content.append(
        get_symbolic_cot_prompt(
            rlib_dir,
            fut_ego_action,
            ego_qual["speed"],
            ego_qual["acceleration"],
            ego_qual["instruction"],
            nl_cot_reference=(
                cot_reference_to_text(sample.get("cot_output"))
                if use_nl_cot_reference
                else None
            ),
            use_predefined_rules=not free_rules,
        )
    )

    return [
        {
            "role": "system",
            "content": "As a professional driver, how do you drive in the following scenario.",
        },
        {
            "role": "user",
            "content": content,
        },
    ], fut_ego_action


class PreprocessedCoTDataset:
    """Dataset backed by existing CoT JSON files from prepare_scaling_workspace.py."""

    def __init__(
        self,
        input_dirs,
        rlib_dir,
        processor=None,
        path_prefix_maps=None,
        free_rules=False,
        use_nl_cot_reference=False,
    ):
        self.input_dirs = [Path(path) for path in input_dirs]
        self.rlib_dir = rlib_dir
        self.processor = processor
        self.path_prefix_maps = path_prefix_maps or []
        self.free_rules = free_rules
        self.use_nl_cot_reference = use_nl_cot_reference
        self.sample_paths = []
        self.tokens = []
        seen_tokens = set()

        for input_dir in self.input_dirs:
            for sample_path in sorted(input_dir.glob("*.json")):
                with open(sample_path, "r", encoding="utf-8") as f:
                    sample = json.load(f)
                token = str(sample.get("token") or sample_path.stem)
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                self.sample_paths.append(sample_path)
                self.tokens.append(token)

        self.scenes = [(token,) for token in self.tokens]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        with open(self.sample_paths[idx], "r", encoding="utf-8") as f:
            sample = json.load(f)

        messages, fut_ego_action = build_preprocessed_messages(
            sample,
            self.rlib_dir,
            self.path_prefix_maps,
            self.free_rules,
            use_nl_cot_reference=self.use_nl_cot_reference,
        )
        sample["messages"] = messages
        sample["fut_ego_action"] = fut_ego_action
        sample["source_json_path"] = str(self.sample_paths[idx])

        if self.processor is not None:
            sample = encode_sample_for_vllm(sample, self.processor)

        return sample


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
            sample["velocity"], sample["acceleration"], sample["instruction"],
            self.rlib_dir,
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
            sample = encode_sample_for_vllm(sample, self.processor)

        return sample


def sample_from_batch(batch, index):
    sample = {}
    for key, value in batch.items():
        if isinstance(value, list):
            sample[key] = value[index]
        else:
            sample[key] = value
    return sample


def collate_symbolic_samples(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = set()
    for feature in features:
        keys.update(feature.keys())
    return {key: [feature.get(key) for feature in features] for key in keys}


def validate_symbolic_output(cot_text, schema, sym_parser, sym_validator, parse_error_cls):
    symbolic_valid = False
    symbolic_violations = []
    grounding_warnings = []
    grounding_score = 0.0
    parse_success = False

    if not cot_text:
        return (
            symbolic_valid,
            symbolic_violations,
            grounding_warnings,
            grounding_score,
            parse_success,
        )

    try:
        parsed = sym_parser.parse(cot_text)
        is_valid, violations, g_warnings = sym_validator.validate(parsed)
        symbolic_valid = is_valid
        symbolic_violations = violations
        grounding_warnings = g_warnings
        parse_success = True

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
            if any(c.kind == "judgment" for c in fg.conditions):
                continue
            checkable += 1
            if sym_validator._evaluate_grounding(fg.conditions, entity_index, ego_ops):
                grounded += 1
        grounding_score = grounded / checkable if checkable > 0 else 1.0
    except parse_error_cls as e:
        symbolic_violations = [f"Parse error: {e}"]

    return (
        symbolic_valid,
        symbolic_violations,
        grounding_warnings,
        grounding_score,
        parse_success,
    )


def build_result(sample, cot_text, fallback_dataset_name, validation):
    symbolic_valid, symbolic_violations, grounding_warnings, grounding_score = validation
    dataset_name = sample.get("dataset_name") or fallback_dataset_name
    token = sample.get("token", "")
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
        "gt_trajectory": serialize_value(sample.get("gt_trajectory", "")),
        "his_trajectory": serialize_value(sample.get("his_trajectory", "")),
        **{f"{side}_camera_paths": sample.get(f"{side}_camera_paths", []) for side in CAM_LIST},
    }

    if dataset_name == "waymo":
        result["preference_scores"] = sample.get("preference_scores", "")
        result["preference_trajectories"] = sample.get("preference_trajectories", "")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symbolic CoT sample generation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None,
                        help='Annotation backend: vllm or openai (overrides config)')
    parser.add_argument("--rlib_dir", type=str, default="./RLIB",
                        help='Path to RLIB directory')
    parser.add_argument(
        "--input-json-dir",
        action="append",
        default=None,
        help=(
            "Existing CoT JSON directory to re-annotate as symbolic CoT. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=None,
        help="Map absolute paths in sample JSONs to local paths, e.g. /data=./data.",
    )
    parser.add_argument("--nl-cot-dir", type=str, default=None,
                        help='Directory of NL CoT JSONs (token.json with cot_output field) to use as reference')
    parser.add_argument(
        "--use-nl-cot-reference",
        action="store_true",
        default=False,
        help=(
            "When using --input-json-dir, include each sample's cot_output as a "
            "natural-language reference and allow its final action as the hint. "
            "By default direct JSON mode ignores cot_output and derives the action "
            "hint from gt_trajectory."
        ),
    )
    parser.add_argument("--free-rules", action="store_true", default=False,
                        help='Disable predefined RLIB rules; LLM composes rules freely from facts')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument(
        "--sample-ids-json",
        type=str,
        default=None,
        help="JSON file containing tokens to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tokens that already have JSON outputs.",
    )
    parser.add_argument(
        "--resume-dir",
        action="append",
        default=None,
        help="Directory to scan for existing token JSONs. Can be passed multiple times.",
    )
    args = parser.parse_args()

    config = load_config(resolve_config_path(args.config))
    if args.sample_ids_json is not None:
        config["sample_ids_json"] = os.path.abspath(args.sample_ids_json)

    backend = args.backend or config.get('annotation_backend', 'vllm')
    rlib_dir = args.rlib_dir or config.get('rlib_dir', './RLIB')
    path_prefix_maps = parse_path_prefix_maps(args.path_prefix_map)

    os.makedirs(args.output_dir, exist_ok=True)

    if backend == 'vllm':
        # IMPORTANT: vLLM must be initialized BEFORE pytorch_lightning imports.
        model = create_annotation_model(config, backend)

        from pytorch_lightning import seed_everything
        from transformers import AutoProcessor
        seed_everything(args.seed)
        random.seed(args.seed)

        processor = AutoProcessor.from_pretrained(config['pretrained_model_path'], use_fast=True)
    else:
        from pytorch_lightning import seed_everything
        seed_everything(args.seed)
        random.seed(args.seed)

        model = create_annotation_model(config, backend)
        processor = None

    input_json_dirs = args.input_json_dir or []
    if input_json_dirs:
        dataset_name = "preprocessed"
        val_dataset = PreprocessedCoTDataset(
            input_json_dirs,
            rlib_dir,
            processor=processor,
            path_prefix_maps=path_prefix_maps,
            free_rules=args.free_rules,
            use_nl_cot_reference=args.use_nl_cot_reference,
        )
    else:
        dataset_name = config.get("dataset_name", "")

        if dataset_name == "nuplan":
            from dataset_utils.preprocessing.nuplan_dataset import NuplanCoTAnnotationDataset
            base_dataset = NuplanCoTAnnotationDataset(config, processor)
        elif dataset_name == "waymo":
            from dataset_utils.preprocessing.waymo_e2e_dataset import WaymoE2ECoTAnnotationDataset
            base_dataset = WaymoE2ECoTAnnotationDataset(config, processor)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        val_dataset = SymbolicPromptWrapper(
            base_dataset,
            rlib_dir,
            processor,
            nl_cot_dir=args.nl_cot_dir,
            free_rules=args.free_rules,
        )

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

    print(f"Selected {len(indices)} samples from dataset of size {len(val_dataset)}.")
    if input_json_dirs:
        print(f"Input JSON dirs: {', '.join(input_json_dirs)}")
    if requested_tokens is not None:
        print(f"Loaded {len(requested_tokens)} requested tokens from {args.sample_ids_json}.")
        if missing_tokens:
            preview = ", ".join(missing_tokens[:10])
            print(
                f"Skipped {len(missing_tokens)} requested tokens not found in the dataset. "
                f"Examples: {preview}"
            )
    if args.sample_num != 0:
        shard_total = len(partition_indices(indices, args.sample_num, args.num_parts))
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

    from models.symbolic_rules import (
        SymbolicSchema, SymbolicParser, SymbolicValidator, ParseError,
    )
    schema = SymbolicSchema(rlib_dir)
    sym_parser = SymbolicParser(schema)
    sym_validator = SymbolicValidator(
        schema,
        grounding_strictness="warn",
        strict_action_match=args.free_rules,
    )

    stats = {
        "total": 0,
        "parse_success": 0,
        "valid": 0,
        "grounding_scores": [],
    }

    saved_count = 0

    def save_symbolic_sample(sample, cot_text):
        fallback_token = sample.get("token", f"scene_{stats['total']}")
        validation_with_parse = validate_symbolic_output(
            cot_text,
            schema,
            sym_parser,
            sym_validator,
            ParseError,
        )
        validation = validation_with_parse[:4]
        parse_success = validation_with_parse[4]

        stats["total"] += 1
        if parse_success:
            stats["parse_success"] += 1
            stats["grounding_scores"].append(validation[3])
        if validation[0]:
            stats["valid"] += 1

        result = build_result(sample, cot_text, dataset_name, validation)
        if not result["token"]:
            result["token"] = fallback_token

        output_path = os.path.join(args.output_dir, f"{result['token']}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result["token"]

    progress = tqdm(
        total=len(selected_indices),
        desc=f"Symbolic CoT ({args.sample_num}/{args.num_parts})",
        unit="scene",
        dynamic_ncols=True,
    )

    if backend == "vllm":
        from torch.utils.data import DataLoader, Subset

        batch_size = max(int(config.get("batch_size", 1)), 1)
        num_workers = max(int(config.get("num_workers", 0)), 0)
        data_loader_kwargs = {}
        if num_workers > 0:
            data_loader_kwargs["persistent_workers"] = True
            data_loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))

        data_loader = DataLoader(
            Subset(val_dataset, selected_indices),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_symbolic_samples,
            shuffle=False,
            **data_loader_kwargs,
        )

        for batch in data_loader:
            cot_outputs = model.vlm_inference(batch)
            batch_size_actual = len(batch.get("token", []))
            last_token = ""
            for sample_idx in range(batch_size_actual):
                sample = sample_from_batch(batch, sample_idx)
                cot_text = cot_outputs[sample_idx] if sample_idx < len(cot_outputs) else ""
                last_token = save_symbolic_sample(sample, cot_text)
                saved_count += 1
            progress.update(batch_size_actual)
            progress.set_postfix_str(
                f"saved={saved_count} valid={stats['valid']} parse_ok={stats['parse_success']} token={last_token}"
            )
    else:
        for idx in selected_indices:
            sample = val_dataset[idx]
            cot_outputs = model.vlm_inference(sample)
            cot_text = cot_outputs[0] if cot_outputs and len(cot_outputs) > 0 else ""
            token = save_symbolic_sample(sample, cot_text)
            saved_count += 1
            progress.update(1)
            progress.set_postfix_str(
                f"saved={saved_count} valid={stats['valid']} parse_ok={stats['parse_success']} token={token}"
            )

    progress.close()

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
