import numpy as np
import numpy.typing as npt

from typing import List

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.planning.metric_caching.metric_cache import MetricCache


def transform_trajectory(pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states) # transform list of EgoState to array of shape (num_poses, state_dim), shape (41, 11): [x, y, heading, vx, vy, ax, ay, steering_angle, steering_rate, angular_vel, angular_acc]


def pdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> PDMResults:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """

    # Current ego pose, velocity, etc. in global frame — simulation starting point
    initial_ego_state = metric_cache.ego_state

    # Reference (expert/GT) trajectory from cache; convert model prediction from ego frame to global frame
    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    from models.autovla import ForkedPdb; ForkedPdb().set_trace()

    # Interpolate both trajectories onto a uniform time grid (40 poses @ 10Hz = 4s)
    # Output shape: (num_poses, state_dim) for each
    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )

    # Stack into (2, num_poses, state_dim): index 0 = reference, index 1 = prediction
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    # ── Step 4: Closed-loop simulation ──────────────────────────────────────
    # 输入轨迹不直接用于评分, 而是作为 LQR 控制器的**跟踪目标**,
    # 通过自行车运动学模型做闭环仿真, 确保轨迹满足物理约束.
    #
    # 仿真过程 (每个 0.1s 时间步):
    #   1) LQR 控制器根据当前状态和目标轨迹算出控制指令:
    #      - 纵向 LQR: 跟踪目标速度 → 输出期望加速度
    #      - 横向 LQR: 跟踪横向偏差+航向偏差 → 输出期望转向速率
    #      - 低速 (<0.2m/s) 时切换为比例控制 (避免 LQR 不稳定)
    #   2) 控制指令经低通滤波 (模拟执行器延迟: acc τ=0.2s, steering τ=0.05s)
    #   3) 自行车运动学积分:
    #        dheading/dt = v * tan(steering_angle) / wheelbase
    #        dx/dt = v * cos(heading),  dy/dt = v * sin(heading)
    #        dv/dt = a,  转向角限幅 ±60°
    #
    # 输入: trajectory_states (2, 41, 11) — 目标轨迹
    # 输出: simulated_states (2, 41, 11) — 物理可行的仿真轨迹
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    # ── Step 5: Multi-metric scoring ─────────────────────────────────────
    # 对仿真后的轨迹, 逐帧与场景上下文做碰撞/越界/方向等检测.
    #
    # 评分分 4 个阶段:
    #
    # 阶段 1 — 自车区域分类 (每个时间步):
    #   MULTIPLE_LANES: 车辆跨越多条车道 (如正在变道)
    #   NON_DRIVABLE_AREA: 至少一个角点在可行驶区域外 (冲出路面)
    #   ONCOMING_TRAFFIC: 车辆中心不在路线车道内 (可能在逆行车道)
    #
    # 阶段 2 — 二值门控指标 (一票否决, 作为乘法基底):
    #   no_at_fault_collisions:   逐帧检测自车 polygon 与 agent polygon 是否相交
    #                             无碰撞=1.0, 撞静态物=0.5, 撞动态 agent=0.0
    #   drivable_area_compliance: 任何一帧有角点越界 → 0.0, 否则 1.0
    #
    # 阶段 3 — 连续加权指标:
    #   ego_progress:       轨迹沿 centerline 的前进距离, 归一化到 [0,1]
    #   TTC:                向前投影 1.5s, 是否有即将碰撞的近距离风险, {0,1}
    #   comfort:            6 项运动学约束 (纵/横向加速度、jerk、yaw rate 等), {0,1}
    #   driving_direction:  逆行车道累积距离: <2m→1.0, 2~6m→0.5, >6m→0.0
    #
    # 阶段 4 — 综合评分:
    #   score = (no_collision × drivable_area)
    #         × weighted_avg(5×progress + 5×TTC + 2×comfort + 0×direction) / 12
    #   注: driving_direction_weight=0, 实际不参与计算
    #
    # 输入: simulated_states (2, 41, 11) + metric_cache 的场景上下文
    # 输出: scores (2,) — 每条轨迹的综合得分
    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,       # 其他交通参与者 (碰撞检测)
        metric_cache.centerline,        # 车道中心线 (方向/进度检查)
        metric_cache.route_lane_ids,    # 路线车道 ID (路线偏离检查)
        metric_cache.drivable_area_map, # 可行驶区域边界 (越界检查)
    )

    # Extract sub-metrics for the predicted trajectory (index 1, not reference)
    pred_idx = 1

    # Binary (pass/fail) metrics — act as multiplicative gates on final score
    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]

    # Continuous weighted metrics (0~1) — contribute additively to final score
    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]
    driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]

    # Composite score (0~10): binary gates * weighted sum of continuous metrics
    score = scores[pred_idx]

    return PDMResults(
        no_at_fault_collisions,
        drivable_area_compliance,
        ego_progress,
        time_to_collision_within_bound,
        comfort,
        driving_direction_compliance,
        score,
    )
