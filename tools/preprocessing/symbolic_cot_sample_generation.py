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
import signal
import subprocess
import sys
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


def infer_future_action(sample, nl_cot_text=None):
    """2-level action-hint resolver.

    - With NL CoT reference (--nl-cot-dir loaded a token JSON), extract the
      final action from the NL CoT text via regex so the prompt's Hint and
      the NL CoT reference agree.
    - Otherwise use sample['fut_ego_action'], which the raw-dataset class
      (NuplanCoTAnnotationDataset / WaymoE2ECoTAnnotationDataset /
      NuscenesCoTAnnotationDataset) populates from the ground-truth future
      trajectory.
    """
    if nl_cot_text:
        action = extract_action_from_cot(nl_cot_text)
        if action:
            return action
    return sample.get("fut_ego_action", "")


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

        # Load NL CoT reference if --nl-cot-dir was provided AND the token has a JSON there.
        nl_cot_ref = None
        if self.nl_cot_dir:
            token = sample.get("token", "")
            nl_cot_path = os.path.join(self.nl_cot_dir, f"{token}.json")
            if os.path.exists(nl_cot_path):
                with open(nl_cot_path) as f:
                    nl_data = json.load(f)
                nl_cot_ref_raw = nl_data.get("cot_output", "")
                # nl CoT may be a list (nuScenes/DriveLM 5-field) or a string (nuPlan/VLM).
                if isinstance(nl_cot_ref_raw, list):
                    nl_cot_ref = "\n".join(str(x) for x in nl_cot_ref_raw if x is not None)
                else:
                    nl_cot_ref = str(nl_cot_ref_raw) if nl_cot_ref_raw else None

        # 2-level action hint: NL CoT regex (if available) else sample's fut_ego_action.
        fut_ego_action = infer_future_action(sample, nl_cot_text=nl_cot_ref)

        # Quantize ego state
        ego_qual = ego_state_to_qualitative(
            sample["velocity"], sample["acceleration"], sample["instruction"],
            self.rlib_dir,
        )

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


# ============================================================
# DP fan-out helpers (mirror tools/preprocessing/cot_sample_generation.py).
# ============================================================

def _resolve_visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def _build_child_argv(args, dp_index, dp_size):
    """Re-emit CLI args for a DP child, swapping in per-shard sharding values."""
    cmd = [sys.executable, sys.argv[0],
           "--config", args.config,
           "--output_dir", args.output_dir,
           "--seed", str(args.seed),
           "--rlib_dir", args.rlib_dir,
           "--dp_size", "1",
           "--num_parts", str(dp_size),
           "--sample_num", str(dp_index + 1)]
    if args.backend is not None:
        cmd += ["--backend", args.backend]
    if args.tp_size is not None:
        cmd += ["--tp_size", str(args.tp_size)]
    if args.nl_cot_dir is not None:
        cmd += ["--nl-cot-dir", args.nl_cot_dir]
    if args.free_rules:
        cmd += ["--free-rules"]
    if args.sample_ids_json is not None:
        cmd += ["--sample-ids-json", args.sample_ids_json]
    if args.resume:
        cmd += ["--resume"]
    if args.path_prefix_map:
        for m in args.path_prefix_map:
            cmd += ["--path-prefix-map", m]
    return cmd


def _spawn_dp_children(args, backend):
    """Fan out N DP children via subprocess; never returns."""
    dp = args.dp_size

    if backend == "vllm":
        visible = _resolve_visible_gpus()
        tp = args.tp_size if args.tp_size is not None else 1
    else:
        visible = []
        tp = 1

    procs = []
    log_files = []
    log_paths = []

    def _terminate_all(signum=None, frame=None):
        print(f"\n[DP parent] received signal {signum}, terminating {len(procs)} children...", flush=True)
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
        for f in log_files:
            try:
                f.close()
            except Exception:
                pass
        sys.exit(130)

    signal.signal(signal.SIGINT, _terminate_all)
    signal.signal(signal.SIGTERM, _terminate_all)

    for i in range(dp):
        child_env = os.environ.copy()
        child_env["AUTOVLA_DP_CHILD"] = "1"
        if backend == "vllm":
            child_env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[i * tp:(i + 1) * tp])

        cmd = _build_child_argv(args, i, dp)
        log_path = os.path.join(args.output_dir, f"_dp_shard_{i}.log")
        log_paths.append(log_path)
        f = open(log_path, "w")
        log_files.append(f)
        gpu_info = f" CUDA_VISIBLE_DEVICES={child_env['CUDA_VISIBLE_DEVICES']}" if backend == "vllm" else ""
        print(f"[DP parent] spawn shard {i + 1}/{dp}{gpu_info} -> {log_path}", flush=True)
        procs.append(subprocess.Popen(cmd, env=child_env, stdout=f, stderr=subprocess.STDOUT))

    rc_total = 0
    for i, p in enumerate(procs):
        rc = p.wait()
        log_files[i].close()
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[DP parent] shard {i + 1}/{dp} {status} (log: {log_paths[i]})", flush=True)
        if rc != 0:
            rc_total = 1

    if rc_total != 0:
        print(f"[DP parent] one or more shards failed; see logs above.", flush=True)
        sys.exit(1)
    print(f"[DP parent] DP run complete: {dp} shards.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symbolic CoT sample generation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None,
                        help='Annotation backend: vllm or openai (overrides config)')
    parser.add_argument("--rlib_dir", type=str, default="./RLIB",
                        help='Path to RLIB directory')
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=None,
        help="Map absolute paths to local paths when crossing hosts, e.g. /data=./data. "
             "Propagated into config['path_prefix_maps'] for the dataset class to consume.",
    )
    parser.add_argument("--nl-cot-dir", type=str, default=None,
                        help="Directory of NL CoT JSONs ({token}.json with cot_output field). "
                             "When set, NL CoT is loaded as prompt reference AND used as action-hint source.")
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
        help="Skip tokens that already have JSON outputs in --output_dir.",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Number of DP replicas to fan out via subprocess. 1 = single process (default).",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Override config's tensor_parallel_size. Requires --backend vllm.",
    )
    args = parser.parse_args()

    config = load_config(resolve_config_path(args.config))
    if args.sample_ids_json is not None:
        config["sample_ids_json"] = os.path.abspath(args.sample_ids_json)

    backend = args.backend or config.get('annotation_backend', 'vllm')
    rlib_dir = args.rlib_dir or config.get('rlib_dir', './RLIB')
    path_prefix_maps = parse_path_prefix_maps(args.path_prefix_map)

    # --- Validation: --tp_size, --dp_size, manual sharding interactions ---
    if args.tp_size is not None and backend != 'vllm':
        sys.exit(f"--tp_size requires --backend vllm; current backend is '{backend}'.")
    if args.dp_size < 1:
        sys.exit(f"--dp_size must be >= 1, got {args.dp_size}.")
    if args.dp_size > 1 and (args.num_parts > 1 or args.sample_num != 1):
        sys.exit(
            "--dp_size is mutually exclusive with manual sharding via --num_parts/--sample_num. "
            f"Got dp_size={args.dp_size}, num_parts={args.num_parts}, sample_num={args.sample_num}."
        )

    # Output directory (created early so DP shard logs can be written into it).
    os.makedirs(args.output_dir, exist_ok=True)

    # Parent path: fan out DP children, then exit. Skipped when DP=1 or already a child.
    is_dp_child = os.environ.get("AUTOVLA_DP_CHILD") == "1"
    if args.dp_size > 1 and not is_dp_child:
        if backend == 'vllm':
            tp_eff = args.tp_size if args.tp_size is not None else int(config.get('tensor_parallel_size', 1))
            visible = _resolve_visible_gpus()
            if args.dp_size * tp_eff > len(visible):
                sys.exit(
                    f"DP*TP = {args.dp_size}*{tp_eff} = {args.dp_size * tp_eff} > "
                    f"{len(visible)} visible GPUs ({visible})."
                )
        _spawn_dp_children(args, backend)
        # _spawn_dp_children calls sys.exit; unreachable below.

    # Inject CLI TP override into config so the vLLM backend picks it up.
    if args.tp_size is not None:
        config['tensor_parallel_size'] = args.tp_size

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

    dataset_name = config.get("dataset_name", "")

    # Inject path-prefix maps into config so per-dataset classes that resolve
    # camera paths cross-host (currently only NuscenesCoTAnnotationDataset)
    # can apply them.
    if path_prefix_maps:
        config["path_prefix_maps"] = path_prefix_maps

    if dataset_name == "nuplan":
        from dataset_utils.preprocessing.nuplan_dataset import NuplanCoTAnnotationDataset
        base_dataset = NuplanCoTAnnotationDataset(config, processor)
    elif dataset_name == "waymo":
        from dataset_utils.preprocessing.waymo_e2e_dataset import WaymoE2ECoTAnnotationDataset
        base_dataset = WaymoE2ECoTAnnotationDataset(config, processor)
    elif dataset_name == "nuscenes":
        from dataset_utils.preprocessing.nuscenes_dataset import NuscenesCoTAnnotationDataset
        base_dataset = NuscenesCoTAnnotationDataset(config, processor)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name!r} (expected nuplan/waymo/nuscenes)")

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

    processed_tokens = collect_processed_tokens([args.output_dir]) if args.resume else set()
    if processed_tokens:
        selected_indices = [
            idx for idx in selected_indices if dataset_tokens[idx] not in processed_tokens
        ]

    print(f"Selected {len(indices)} samples from dataset of size {len(val_dataset)}.")
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
            f"Resume mode found {len(processed_tokens)} processed tokens in {args.output_dir}."
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
