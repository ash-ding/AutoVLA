from typing import Any, Dict, List, Union
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_pool import Task
from nuplan.planning.utils.multithreading.worker_utils import chunk_list

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.visualization.plots import plot_cameras_frame_with_bev_agent_cot

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def resolve_metric_cache_path(metric_cache_path: Union[str, Path], cache_root: Union[str, Path]) -> Path:
    """Resolve stale absolute metric-cache paths against the configured cache root."""
    metric_cache_path = Path(metric_cache_path)
    if metric_cache_path.exists():
        return metric_cache_path

    # Metric cache layout is <cache_root>/<log_name>/unknown/<token>/metric_cache.pkl.
    candidate = Path(cache_root).joinpath(*metric_cache_path.parts[-4:])
    if candidate.exists():
        return candidate

    return metric_cache_path


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    # Per-token dump dir: resolved + pre-created in main(), passed via args
    # (cfg.output_dir contains a Hydra interpolation that ray serialization
    # would break — see issue with UnsupportedInterpolationType).
    per_sample_dir = Path(args[0]["per_sample_dir"])

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = resolve_metric_cache_path(
                metric_cache_loader.metric_cache_paths[token],
                cfg.metric_cache_path,
            )
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            # load input
            json_path = os.path.join(cfg.json_data_path, f"{token}.json")
            with open(json_path, 'r') as f:
                agent_input = json.load(f)

            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory, cot_results = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory, cot_results = agent.compute_trajectory(agent_input)

            # scene = scene_loader.get_scene_from_token(token)
            # frame_idx = scene.scene_metadata.num_history_frames - 1
            # fig, _ = plot_cameras_frame_with_bev_agent_cot(scene, frame_idx, agent_trajectory=trajectory, cot=cot_results)
            # vis_dir = Path(cfg.output_dir) / "Visualization"
            # vis_dir.mkdir(parents=True, exist_ok=True)
            # vis_path = vis_dir / f"{token}_bevagent.png"
            # fig.savefig(vis_path, bbox_inches="tight")
            # plt.close(fig)
            # if cot_results:
            #     cot_md_path = vis_dir / f"{token}_cot.md"
            #     with open(cot_md_path, "w", encoding="utf-8") as f:
            #         f.write(cot_results.strip() + "\n")

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))

            # Per-token dump: cot trace + predicted/gt trajectory + scores.
            # Needed for cross-arm qualitative ablation; PDM-score CSV alone
            # carries no reasoning text and no per-token trajectory.
            try:
                token_record = {
                    "token": token,
                    "cot_trace": cot_results if cot_results else "",
                    "pred_trajectory": trajectory.poses.tolist(),
                    "gt_trajectory": agent_input.get("gt_trajectory", None),
                    "his_trajectory": agent_input.get("his_trajectory", None),
                    "scores": asdict(pdm_result),
                }
                with open(per_sample_dir / f"{token}.json", "w", encoding="utf-8") as f:
                    json.dump(token_record, f)
            except Exception:
                logger.warning(f"Failed to dump per-token record for {token}")
                traceback.print_exc()
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    # Resolve cfg.output_dir here (main has the Hydra resolver). Pre-create the
    # per_sample dump dir and ship the resolved string path through to workers.
    resolved_output_dir = str(cfg.output_dir)
    per_sample_dir = Path(resolved_output_dir) / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "per_sample_dir": str(per_sample_dir),
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    num_eval_workers = cfg.get("num_eval_workers", None)
    num_eval_workers = int(num_eval_workers) if num_eval_workers is not None else worker.number_of_threads
    gpus_per_eval_worker = cfg.get("gpus_per_eval_worker", None)

    data_chunks = chunk_list(data_points, num_eval_workers)
    score_rows_nested: List[List[Dict[str, Any]]] = worker.map(
        Task(fn=run_pdm_score, num_gpus=gpus_per_eval_worker),
        data_chunks,
    )
    score_rows: List[Dict[str, Any]] = [row for rows in score_rows_nested for row in rows]

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
