"""
Shared utilities for data-parallel (DP) execution of vLLM-backed eval scripts.

Design:
- Parent process partitions the input list, then forks N children via
  subprocess.Popen. Each child gets `--_dp_shard_id i --_dp_total N` and a
  single GPU via `CUDA_VISIBLE_DEVICES=<i>`.
- Children write their per-sample artifacts to disk (per_sample JSONs and
  a partial metric/score state file).
- Parent waits for all children to exit, then aggregates the partial state
  files into the final result.

We use DP rather than vLLM tensor parallelism because:
  - Our model is 3B (~7 GB bf16), single A100/L40S has plenty of headroom.
  - TP introduces NCCL all-reduce per forward step → slowdown for small models.
  - DP scales near-linearly with GPU count (independent samples, no comm).
"""
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional


def _resolve_visible_gpus() -> List[str]:
    """Return current CUDA_VISIBLE_DEVICES as list of GPU IDs (default: 0..7)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",") if g.strip()]
    # If unset, query nvidia-smi as fallback
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return [line.strip() for line in out.strip().splitlines() if line.strip()]
    except Exception:
        return ["0"]


def spawn_dp_children(
    dp_size: int,
    inner_args: List[str],
    log_dir: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> int:
    """
    Spawn `dp_size` copies of the current Python script with sharded args.

    Each child gets:
      - one GPU via CUDA_VISIBLE_DEVICES
      - --_dp_shard_id <i> --_dp_total <N> appended to argv
    Caller is responsible for slicing the input by (shard_id, total) and for
    aggregating per-shard artifacts after this function returns.

    :param dp_size: number of children to fork
    :param inner_args: argv to pass to each child (typically `sys.argv[1:]`
        with --dp_size removed). The function appends the shard flags.
    :param log_dir: if given, each child's stdout/stderr go to
        <log_dir>/dp_child_<i>.log; otherwise inherit parent stdout/stderr.
    :param extra_env: extra env vars for all children (merged with os.environ)
    :return: 0 if all children exited 0; non-zero count of failed children otherwise.
    """
    visible = _resolve_visible_gpus()
    if len(visible) < dp_size:
        raise RuntimeError(
            f"--dp_size={dp_size} but only {len(visible)} GPU(s) visible: {visible}. "
            f"Either lower --dp_size or expand CUDA_VISIBLE_DEVICES."
        )

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    procs = []
    log_files = []

    def _terminate_all(signum=None, frame=None):
        print(f"\n[DP parent] received signal {signum}, terminating {len(procs)} children...",
              flush=True)
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
        for lf in log_files:
            try:
                lf.close()
            except Exception:
                pass
        sys.exit(130)

    signal.signal(signal.SIGINT, _terminate_all)
    signal.signal(signal.SIGTERM, _terminate_all)

    for shard_id in range(dp_size):
        gpu_id = visible[shard_id]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        if extra_env:
            env.update(extra_env)
        cmd = [sys.executable, sys.argv[0]] + list(inner_args) + [
            "--_dp_shard_id", str(shard_id),
            "--_dp_total", str(dp_size),
        ]
        if log_dir:
            lf = open(os.path.join(log_dir, f"dp_child_{shard_id}.log"), "w")
            log_files.append(lf)
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        else:
            # Prefix each line with [shard_i] for readability when stdout is shared
            proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)
        print(f"[DP parent] spawned shard {shard_id} on GPU {gpu_id} (pid={proc.pid})",
              flush=True)

    print(f"[DP parent] waiting for {dp_size} children to finish...", flush=True)
    failed = 0
    t0 = time.time()
    for shard_id, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            print(f"[DP parent] ⚠ shard {shard_id} (pid={p.pid}) exited with code {rc}",
                  flush=True)
            failed += 1
        else:
            print(f"[DP parent] ✓ shard {shard_id} done ({time.time()-t0:.1f}s)", flush=True)
    for lf in log_files:
        lf.close()

    return failed


def split_indices(total: int, shard_id: int, total_shards: int) -> slice:
    """Round-robin partition: shard i gets [i, i+N, i+2N, ...]. Even balance.

    Returns a Python `slice` you can apply to any list-like: `samples[s]`.
    """
    if total_shards < 1 or not (0 <= shard_id < total_shards):
        raise ValueError(f"bad shard_id={shard_id} / total_shards={total_shards}")
    return slice(shard_id, total, total_shards)
