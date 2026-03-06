import os
import torch
import yaml
import lzma
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Scene, Trajectory


class PDM_Reward:
    """
    RL reward calculator using PDM (Planning-Decision-Making) scoring.

    Evaluates model-generated trajectories against pre-cached scene data by running
    closed-loop simulation and computing a composite driving quality score (0~10).

    Sub-metrics: no_collision, drivable_area, ego_progress, time_to_collision,
                 comfort, driving_direction.
    """
    def __init__(self, metric_cache_path):
        """
        :param metric_cache_path: Root dir of metric cache (contains metadata/*.csv and pkl files).
        """
        # Build {token -> pkl_path} index from metadata CSV (no data loaded yet)
        self.metric_cache_loader = MetricCacheLoader(metric_cache_path)

        # Simulation time grid: 40 poses @ 10Hz = 4s horizon
        # (model outputs 10 poses @ 2Hz / 5s, will be interpolated to match)
        self.future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)

        # Simulator: replays ego trajectory against cached observations (other agents, map)
        self.simulator = PDMSimulator(self.future_sampling)

        # Scorer: computes weighted sub-metrics from simulation results
        self.scorer = PDMScorer(self.future_sampling)

    def rl_pdm_score(self, trajectory, token):
        """
        Compute PDM reward for a single scene.

        :param trajectory: Trajectory object with poses [10, 3] (x, y, heading) in ego frame.
        :param token: Scene token string to look up cached environment data.
        :return: float score in [0, 10], or 0.0 on failure.
        """
        # Load pre-cached scene data: reference trajectory, ego_state,
        # observations (other vehicles), centerline, drivable_area_map
        metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)

        # ============================================================================
        # metric_cache: MetricCache dataclass (定义在 navsim/planning/metric_caching/metric_cache.py)
        # 由 MetricCacheProcessor.compute_metric_cache() 预计算并序列化为 lzma-compressed pickle
        #
        # 【关键坐标系区别】
        #   - metric_cache 中所有空间数据均使用 **全局 UTM 坐标系** (绝对地图坐标, 单位: 米)
        #   - 训练数据 JSON 中的 gt_trajectory 使用 **自车相对坐标系** (原点=t0后轴, x=车头方向, y=左侧)
        #   - 评分时 pdm_score() 会调用 relative_to_absolute_poses() 将模型输出从自车系转换到全局系
        #   这样设计的原因: 训练时用自车系让模型学习与朝向无关的驾驶策略(平移/旋转不变性),
        #   而评分需要全局系才能与场景中的其他车辆、车道线、可行驶区域进行空间比较
        #
        # ============================================================================
        #
        # 1. file_path: PosixPath — 缓存文件路径
        #    e.g. '.../navmini_metric_cache/2021.06.08.14.35.24_veh-26_02555_03004/unknown/b549b6c92312537d/metric_cache.pkl'
        #    层级结构: {cache_root}/{log_name}/{scenario_type}/{scene_token}/metric_cache.pkl
        #    scene_token 与训练 JSON 文件名一一对应 (如 b549b6c92312537d.json)
        #
        # 2. trajectory: InterpolatedTrajectory — PDM-Closed 规划器生成的参考轨迹 (全局 UTM 系)
        #    【数据结构】51 个 EgoState 对象 (50 poses @ 10Hz = 5s), 支持时间插值
        #      每个 EgoState 转为 state array 后为 (11,):
        #      [x, y, heading, vx, vy, ax, ay, steering_angle, steering_rate, angular_vel, angular_acc]
        #      
        # 索引	名称	含义	示例值
        # 0	X	全局 UTM 东向位置 (米)	637045.3
        # 1	Y	全局 UTM 北向位置 (米)	3999490.1
        # 2	HEADING	航向角/偏航角 (弧度, 相对正东逆时针)	1.25
        # 3	VELOCITY_X	后轴纵向速度 (米/秒, 沿车头方向)	6.5
        # 4	VELOCITY_Y	后轴横向速度 (米/秒, 沿车左方向)	-0.2
        # 5	ACCELERATION_X	后轴纵向加速度 (米/秒²)	0.8
        # 6	ACCELERATION_Y	后轴横向加速度 (米/秒²)	0.1
        # 7	STEERING_ANGLE	前轮转向角 (弧度, 左转为正)	0.02
        # 8	STEERING_RATE	转向角变化率 (弧度/秒)	0.001
        # 9	ANGULAR_VELOCITY	偏航角速度 (弧度/秒)	0.01
        # 10	ANGULAR_ACCELERATION	偏航角加速度 (弧度/秒²)	0.0
        # 其中 0-2 是位姿（SE2），全局坐标系，3-6 是线速度/加速度，7-8 是转向，9-10 是旋转动力学， 3-10都是自坐标系。

        # 注意 velocity/acceleration 的 x/y 是车体系（沿车头/车左），而 position 的 X/Y 是全局系（UTM 东/北），两者坐标系不同。
        #      e.g. [637045.3, 3999490.1, 1.25, 6.5, -0.2, 0.8, 0.1, 0.02, 0.001, 0.01, 0.0]
        #    【与训练数据的区别】
        #      - 训练 JSON 的 gt_trajectory: 10 poses @ 2Hz, 自车系, shape [10, 3] (x, y, heading)
        #        e.g. [[3.31, 0.19, 0.12], ..., [39.52, 6.12, 0.16]]
        #      - 此处 trajectory: 51 waypoints @ 10Hz, 全局 UTM 系, 包含完整动力学状态 (11维)
        #    【为什么不直接用 GT 轨迹？】
        #      这条轨迹并非人类驾驶员的 GT, 而是 PDM-Closed planner 用 IDM 策略规划的结果
        #      (5种速度档位 × 3种横向偏移 = 15条候选轨迹, 取最优)
        #      用它作为 reference baseline: 模型需要超越或持平这个 rule-based planner 才算好
        #    【评分时如何使用】
        #      pdm_score() 中 get_trajectory_as_array() 将其重采样到 40 poses @ 10Hz (4s),
        #      与模型轨迹一起 stack 成 (2, 40, 11), 过 simulator 后对比评分
        #
        # 3. ego_state: EgoState — t=0 时刻的自车状态 (全局 UTM 系)
        #    【数据结构】
        #      rear_axle: StateSE2 — (x, y, heading)
        #        e.g. (637029.12, 3999487.56, 1.23)  // UTM 东向, 北向, 航向角(弧度)
        #      dynamic_car_state:
        #        velocity: (vx, vy)        e.g. (6.64, -0.18) m/s
        #        acceleration: (ax, ay)    e.g. (0.81, 2.36) m/s²
        #      tire_steering_angle: float  e.g. 0.02 rad 前轮转角
        #      car_footprint: 车辆包络框 (length ~4.6m, width ~1.9m, wheelbase 前轴中心到后轴中心的距离~2.7m)
        #      time_point: TimePoint       时间戳 (微秒)
        #    【与训练数据的关系】
        #      训练 JSON 中的 velocity/acceleration 就是从同一个 EgoState 提取的:
        #        JSON.velocity = [ego_state.vx, ego_state.vy] = [6.64, -0.18]
        #      但训练时只用标量速度 sqrt(vx²+vy²) 作为 prompt 的一部分
        #    【为什么需要它】
        #      1) 坐标系转换的锚点: 模型输出的自车系轨迹 → 全局系, 需要 ego_state.rear_axle 作为旋转平移基准
        #      2) 仿真起点: PDMSimulator.simulate_proposals() 从此状态开始做运动学约束仿真
        #      3) 时间对齐: ego_state.time_point 用于将轨迹插值到统一时间网格
        #
        # 4. observation: PDMObservation — 场景中其他交通参与者的时序轨迹 (全局 UTM 系)
        #    【数据结构】
        #      _occupancy_maps: List[PDMOccupancyMap], 51 个时间步 @ 10Hz
        #        每个 OccupancyMap 包含该时刻所有 agent 的 Shapely Polygon (bbox)
        #        用 STRtree 空间索引加速碰撞检测查询
        #      _unique_objects: Dict[str, TrackedObject], 5~50+ 个独立 agent
        #        key = track_token (字符串 ID), value = 初始检测对象
        #      每个 agent 的状态: (time, x, y, heading, vx, vy), shape (6,)
        #    【数据来源与处理】
        #      原始数据: nuPlan DB 中每帧的 tracked_objects (激光雷达检测+跟踪), 原始 2Hz
        #      预处理: MetricCacheProcessor._interpolate_gt_observation() 做了:
        #        1) 每 0.5s 采样一次原始检测 (2Hz) → 得到 11 个关键帧
        #        2) 对每个 agent 的 (x,y,heading,vx,vy) 用三次样条插值到 10Hz → 51 个时间步
        #        3) 为每个时刻重建 OrientedBox (位姿+尺寸) 和 Agent/StaticObject 对象
        #        4) 封装为 PDMOccupancyMap 序列, 构建空间索引
        #      最大 agent 数: 50 vehicles + 25 pedestrians + 10 bicycles + 50 static objects
        #    【为什么需要插值到 10Hz？】
        #      PDM 评分的仿真步长是 0.1s, 需要逐帧检测碰撞; 原始 2Hz 太稀疏会漏检
        #    【与训练数据的对比】
        #      训练数据完全不包含其他车辆信息 — 模型只通过摄像头图像隐式感知周围环境
        #      而评分时需要精确的 agent 轨迹来做碰撞检测, 所以必须从传感器标注中预计算
        #
        # 5. centerline: PDMPath — 规划路线的车道中心线 (全局 UTM 系)
        #    【数据结构】
        #      _states_se2_array: ndarray (N, 3) — (x, y, heading), N = 50~500+ 个离散点
        #      _progress: ndarray (N,) — 累积弧长 (米), 单调递增, 用于沿路径的进度计算
        #      _interpolator: scipy.interpolate.interp1d — 按弧长插值, 可查询任意距离处的位姿
        #      _linestring: Shapely LineString — 几何表示, 用于距离/投影计算
        #    【数据来源】
        #      PDMClosedPlanner 根据 route_roadblock_ids 从 HD Map 中提取车道几何,
        #      拼接成连续的中心线路径 (处理车道连接、交叉口等拓扑)
        #    【评分时如何使用】
        #      - ego_progress: 模型轨迹沿 centerline 的投影距离 / 参考轨迹的投影距离 → [0, 1]
        #      - driving_direction_compliance: 检查车辆航向与 centerline 切线方向是否一致
        #        (防止逆行、大角度偏离车道方向)
        #    【与训练数据的对比】
        #      训练 JSON 中有 driving_command (如 "turn left", "go straight") 作为高层导航指令
        #      centerline 是这些指令对应的精确几何路径; 模型通过指令隐式学习路径跟踪
        #
        # 6. route_lane_ids: List[str] — 规划路线经过的所有车道 ID
        #    e.g. ['68806', '69531', '69643', ...], 通常 50~150 个 ID
        #    【数据来源】
        #      PDMClosedPlanner._route_lane_dict.keys(), 从 HD Map 的路由规划结果提取
        #    【评分时如何使用】
        #      scorer.score_proposals() 用它判断自车是否在规划路线上行驶
        #      (如果偏离路线太远, driving_direction_compliance 会被扣分)
        #    【为什么和 centerline 分开存？】
        #      centerline 是一条连续曲线 (几何), route_lane_ids 是离散的拓扑标识 (语义)
        #      两者互补: centerline 用于连续距离/方向计算, lane_ids 用于离散的"是否在路线上"判断
        #
        # 7. drivable_area_map: PDMDrivableMap — 可行驶区域的多边形集合 (全局 UTM 系)
        #    【数据结构】
        #      继承自 PDMOccupancyMap:
        #        _tokens: List[str] — 每个多边形的 ID
        #        _geometries: ndarray[Shapely.Polygon] — 20~100+ 个多边形, ~100m 半径范围内
        #        _map_types: List[SemanticMapLayer] — LANE / LANE_CONNECTOR / INTERSECTION / CARPARK_AREA
        #        _str_tree: STRtree — R-tree 空间索引, 支持 O(log n) 的包含/相交查询
        #    【数据来源】
        #      PDMClosedPlanner 从 HD Map (nuplan-maps) 中查询自车附近 100m 内的所有可行驶区域多边形
        #    【评分时如何使用】
        #      drivable_area_compliance (二值门控):
        #        对仿真轨迹的每个时间步, 检查车辆 footprint 的 4 个角点是否都在某个 drivable polygon 内
        #        任何一帧有角点越界 → compliance = 0, 最终得分直接乘以 0 (一票否决)
        #    【与训练数据的对比】
        #      训练时没有显式的可行驶区域信息 — 模型需要从摄像头图像学习道路边界
        #      评分时用 HD Map 的精确几何做 ground truth 判定, 这比视觉感知严格得多
        #
        # ============================================================================

        try:
            # Run closed-loop simulation + scoring:
            # 1. Transform model trajectory from ego frame to global frame
            # 2. Simulate ego driving this trajectory against cached observations
            # 3. Evaluate: collision, drivable area, progress, TTC, comfort, direction
            # 4. Return weighted composite score
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory, # [10, 3], trajectory_sampling=TrajectorySampling(num_poses=10, time_horizon=5.0, interval_length=0.5), 自坐标系
                future_sampling=self.future_sampling, # TrajectorySampling(num_poses=40, time_horizon=4.0, interval_length=0.1)
                simulator=self.simulator,
                scorer=self.scorer,
            )

            final_reward = result.score

            return final_reward

        except Exception as e:
            print(f"Reward calculation failed")

            return 0.0
