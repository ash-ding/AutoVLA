import os
import random
import signal
import subprocess
import sys
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
           "--dp_size", "1",
           "--num_parts", str(dp_size),
           "--sample_num", str(dp_index + 1)]
    if args.backend is not None:
        cmd += ["--backend", args.backend]
    if args.tp_size is not None:
        cmd += ["--tp_size", str(args.tp_size)]
    if args.sample_ids_json is not None:
        cmd += ["--sample-ids-json", args.sample_ids_json]
    if args.sample_ids_key is not None:
        cmd += ["--sample-ids-key", args.sample_ids_key]
    if args.resume:
        cmd += ["--resume"]
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
        "--sample-ids-key",
        type=str,
        default=None,
        help="When --sample-ids-json points at a scaling-style file with a top-level "
             "'buckets' dict, pick the bucket name to use (e.g. 'nuplan_cot').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tokens that already have JSON outputs in --output_dir",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Number of DP replicas to fan out via subprocess. vLLM only — "
             "openai backend uses --concurrency (in-process asyncio) instead.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Override config's tensor_parallel_size. Requires --backend vllm.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="OpenAI backend: number of concurrent in-flight API requests via "
             "asyncio.gather. Default from config['concurrency'] or 10. Ignored "
             "for vllm backend (use --dp_size there).",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(resolve_config_path(args.config))
    if args.sample_ids_json is not None:
        config["sample_ids_json"] = os.path.abspath(args.sample_ids_json)

    # Determine backend: CLI arg > config > default (vllm)
    backend = args.backend or config.get('annotation_backend', 'vllm')

    # --- Validation: --tp_size, --dp_size, --concurrency, manual sharding interactions ---
    if args.tp_size is not None and backend != 'vllm':
        sys.exit(f"--tp_size requires --backend vllm; current backend is '{backend}'.")
    if args.dp_size < 1:
        sys.exit(f"--dp_size must be >= 1, got {args.dp_size}.")
    if args.dp_size > 1 and backend == 'openai':
        sys.exit(
            f"--dp_size {args.dp_size} requested with openai backend, but the openai "
            "backend uses in-process asyncio concurrency instead of subprocess DP. "
            f"Use --concurrency {args.dp_size} (or higher) for I/O-parallel API calls."
        )
    if args.concurrency is not None and backend != 'openai':
        sys.exit(f"--concurrency requires --backend openai; current backend is '{backend}'.")
    if args.concurrency is not None and args.concurrency < 1:
        sys.exit(f"--concurrency must be >= 1, got {args.concurrency}.")
    if args.dp_size > 1 and (args.num_parts > 1 or args.sample_num != 1):
        sys.exit(
            "--dp_size is mutually exclusive with manual sharding via --num_parts/--sample_num. "
            f"Got dp_size={args.dp_size}, num_parts={args.num_parts}, sample_num={args.sample_num}."
        )

    # Output directory (created early so DP shard logs can be written into it)
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
        requested_tokens = load_token_list(args.sample_ids_json, args.sample_ids_key)
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
            f"Resume mode found {len(processed_tokens)} processed tokens in {args.output_dir}."
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
        # OpenAI backend: in-process asyncio concurrency. Network I/O bound
        # so subprocess-level DP is wasteful — one process with N in-flight
        # requests via asyncio.gather is the right shape.
        import asyncio

        concurrency = args.concurrency if args.concurrency is not None else int(config.get('concurrency', 10))
        print(f"OpenAI backend: running asyncio.gather with concurrency={concurrency}.")
        sem = asyncio.Semaphore(concurrency)

        async def process_one(idx):
            async with sem:
                sample = val_dataset[idx]
                cot_outputs = await model.vlm_inference_async(sample)
                cot_text = cot_outputs[0] if cot_outputs and len(cot_outputs) > 0 else ""
                return idx, sample, cot_text

        async def run_all():
            global saved_count
            tasks = [asyncio.create_task(process_one(idx)) for idx in selected_indices]
            for coro in asyncio.as_completed(tasks):
                idx, sample, cot_text = await coro
                token, result = build_result(sample, cot_text, dataset_name, fallback_idx=idx)
                output_path = os.path.join(args.output_dir, f"{token}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                saved_count += 1
                progress.update(1)
                progress.set_postfix_str(f"saved={saved_count} token={token}")

        asyncio.run(run_all())

    print(f"All preprocessing data with CoT results have been saved in directory: {args.output_dir}")
