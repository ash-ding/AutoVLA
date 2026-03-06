# MetricCache 字段详解

> 定义: `navsim/navsim/planning/metric_caching/metric_cache.py`
> 创建: `MetricCacheProcessor.compute_metric_cache()` 预计算并序列化为 lzma-compressed pickle
> 消费: `pdm_score()` 在评分时加载使用

## 关键坐标系区别

| 方面 | 训练数据 (JSON) | MetricCache | 模型输出 | 评分输入 |
|------|-----------------|-------------|----------|----------|
| **坐标系** | 自车相对系 | 全局 UTM 系 | 自车相对系 | 全局 UTM 系 |
| **原点** | t=0 后轴中心 | UTM zone | t=0 后轴中心 | UTM zone |
| **X 轴** | 车头方向 | 地图东向 | 车头方向 | 地图东向 |
| **Y 轴** | 车左方向 | 地图北向 | 车左方向 | 地图北向 |
| **Heading** | 相对偏航角 | 绝对偏航角 | 相对偏航角 | 绝对偏航角 |

训练时用自车系让模型学习与朝向无关的驾驶策略 (平移/旋转不变性), 评分需要全局系才能与场景中的其他车辆、车道线、可行驶区域进行空间比较。评分时 `pdm_score()` 调用 `relative_to_absolute_poses()` 将模型输出从自车系转换到全局系。

---

## 1. file_path: PosixPath

缓存文件路径。

```
e.g. '.../navmini_metric_cache/2021.06.08.14.35.24_veh-26_02555_03004/unknown/b549b6c92312537d/metric_cache.pkl'
```

层级结构: `{cache_root}/{log_name}/{scenario_type}/{scene_token}/metric_cache.pkl`

scene_token 与训练 JSON 文件名一一对应 (如 `b549b6c92312537d.json`)。

---

## 2. trajectory: InterpolatedTrajectory

PDM-Closed 规划器生成的参考轨迹 (全局 UTM 系)。

### 数据结构

51 个 EgoState 对象 (50 poses @ 10Hz = 5s), 支持时间插值。每个 EgoState 转为 state array 后为 (11,):

| 索引 | 名称 | 含义 | 坐标系 | 示例值 |
|------|------|------|--------|--------|
| 0 | X | 全局 UTM 东向位置 (米) | 全局 | 637045.3 |
| 1 | Y | 全局 UTM 北向位置 (米) | 全局 | 3999490.1 |
| 2 | HEADING | 航向角/偏航角 (弧度, 相对正东逆时针) | 全局 | 1.25 |
| 3 | VELOCITY_X | 后轴纵向速度 (米/秒, 沿车头方向) | 车体 | 6.5 |
| 4 | VELOCITY_Y | 后轴横向速度 (米/秒, 沿车左方向) | 车体 | -0.2 |
| 5 | ACCELERATION_X | 后轴纵向加速度 (米/秒²) | 车体 | 0.8 |
| 6 | ACCELERATION_Y | 后轴横向加速度 (米/秒²) | 车体 | 0.1 |
| 7 | STEERING_ANGLE | 前轮转向角 (弧度, 左转为正) | 车体 | 0.02 |
| 8 | STEERING_RATE | 转向角变化率 (弧度/秒) | 车体 | 0.001 |
| 9 | ANGULAR_VELOCITY | 偏航角速度 (弧度/秒) | 车体 | 0.01 |
| 10 | ANGULAR_ACCELERATION | 偏航角加速度 (弧度/秒²) | 车体 | 0.0 |

所有 11 个维度描述的都是**后轴中心 (rear axle center)** 的状态。nuPlan 用后轴中心作为参考点, 因为它是自行车运动学模型的自然旋转中心 — 后轴不会横向滑移, 运动学方程最简洁:

```
dx/dt = v * cos(heading)
dy/dt = v * sin(heading)
dheading/dt = v * tan(steering_angle) / wheelbase
```

### 与训练数据的区别

- **训练 JSON 的 gt_trajectory**: 10 poses @ 2Hz, 自车系, shape `[10, 3]` (x, y, heading)
  - e.g. `[[3.31, 0.19, 0.12], ..., [39.52, 6.12, 0.16]]`
  - 来源: **人类驾驶员的真实驾驶记录** (从 nuPlan 数据库中提取 GPS/IMU 定位)
  - 用途: SFT 的监督信号 (通过 ActionTokenizer 量化成 action token 作为训练 label)
- **此处 trajectory**: 51 waypoints @ 10Hz, 全局 UTM 系, 包含完整动力学状态 (11维)
  - 来源: **PDM-Closed planner 的规划结果** (算法生成, 和人类驾驶无关)
  - 用途: 评分时作为 reference baseline

两者**完全独立, 没有任何关系**。

### 为什么不直接用 GT 轨迹？

这条轨迹并非人类驾驶员的 GT, 而是 PDM-Closed planner 用 IDM 策略规划的结果:

1. 从 HD Map 提取路由信息和车道中心线
2. 用 IDM (Intelligent Driver Model) 生成纵向速度曲线 — 5 种速度档位 (限速的 20%/40%/60%/80%/100%)
3. 在中心线基础上加横向偏移 (-1m, 0m, +1m)
4. 组合得到 5×3 = 15 条候选轨迹
5. 对每条候选做闭环仿真 + PDM 评分, 选得分最高的那条

NAVSIM 的评分框架设计哲学: 用一个 strong rule-based baseline 作为参考, 而不是用人类 GT。原因是人类驾驶本身有噪声和次优行为 (比如偶尔急刹、不够平滑), PDM-Closed 是一个"理想化的合格司机"。

### 评分时如何使用

`pdm_score()` 中 `get_trajectory_as_array()` 将其重采样到 40 poses @ 10Hz (4s), 与模型轨迹一起 stack 成 `(2, 40, 11)`, 过 simulator 后对比评分。

---

## 3. ego_state: EgoState

t=0 时刻的自车状态 (全局 UTM 系)。

### 数据结构

```
ego_state
├── rear_axle: StateSE2 — (x, y, heading)
│     e.g. (637029.12, 3999487.56, 1.23)  // UTM 东向, 北向, 航向角(弧度)
│
├── dynamic_car_state:
│   ├── velocity: (vx, vy)        e.g. (6.64, -0.18) m/s    // 车体系
│   ├── acceleration: (ax, ay)    e.g. (0.81, 2.36) m/s²    // 车体系
│   ├── angular_velocity: float
│   └── angular_acceleration: float
│
├── tire_steering_angle: float    e.g. 0.02 rad
│
├── car_footprint:
│   └── vehicle_parameters:
│       ├── length: ~4.6m
│       ├── width: ~1.9m
│       └── wheelbase: ~2.7m  // 前轴到后轴的距离
│
└── time_point: TimePoint         时间戳 (微秒)
```

velocity 和 acceleration 是**车体系** — vx 沿车头方向, vy 沿车左方向。正常行驶时 vy 很小 (接近 0), 因为车不会横着走。

wheelbase (轴距) 是自行车运动学模型的关键参数: `dheading/dt = v * tan(steering_angle) / wheelbase`。轴距越长, 同样转向角产生的转弯半径越大。

### 与训练数据的关系

训练 JSON 中的 `velocity`/`acceleration` 就是从同一个 EgoState 提取的:

```
JSON.velocity = [ego_state.vx, ego_state.vy] = [6.64, -0.18]
```

但训练时只用标量速度 `sqrt(vx² + vy²)` 作为 prompt 的一部分。

### 为什么需要它

1. **坐标系转换的锚点**: 模型输出的自车系轨迹 → 全局系, 需要 `ego_state.rear_axle` 作为旋转平移基准
2. **仿真起点**: `PDMSimulator.simulate_proposals()` 从此状态开始做运动学约束仿真
3. **时间对齐**: `ego_state.time_point` 用于将轨迹插值到统一时间网格

---

## 4. observation: PDMObservation

场景中其他交通参与者的时序轨迹 (全局 UTM 系)。

### 它是什么

observation 存储的是未来 5 秒内, 自车周围所有其他交通参与者 (车辆、行人、自行车、路障等) 在每个时间步的精确位置和形状。本质上就是一个"世界的未来剧本" — 不管你的模型怎么开, 其他 agent 都会按照这个剧本走 (**非反应式仿真**, 即 open-loop for others)。

### 数据结构

```
PDMObservation
├── _occupancy_maps: List[PDMOccupancyMap]   # 长度 51 (t=0.0s, 0.1s, ..., 5.0s)
│   └── 每个 PDMOccupancyMap:
│       ├── _tokens: List[str]                # 每个 agent 的 track ID
│       ├── _geometries: ndarray[Polygon]     # 每个 agent 的 Shapely 多边形 (bbox)
│       └── _str_tree: STRtree                # R-tree 空间索引
└── _unique_objects: Dict[str, TrackedObject] # 所有出现过的 agent 的初始状态
```

每个 agent 的状态: `(time, x, y, heading, vx, vy)`, shape `(6,)`

最大 agent 数: 50 vehicles + 25 pedestrians + 10 bicycles + 50 static objects

碰撞检测时, 取仿真轨迹某一时刻的自车 footprint (一个矩形 Polygon), 去对应时刻的 OccupancyMap 的 STRtree 里查询是否与任何 agent 的 Polygon 相交。STRtree 是 R-tree 的一种, O(log n) 复杂度。

### 数据来源与处理

原始数据来自 nuPlan 数据库里的 `tracked_objects` — 激光雷达检测 + 多目标跟踪的结果, 每个 agent 有唯一的 `track_token`、精确的 3D bounding box 和速度。原始采样率约 2Hz。

预处理流程 (`MetricCacheProcessor._interpolate_gt_observation()`):

1. **采样关键帧**: `t = 0.0, 0.5, 1.0, ..., 5.0` → 11 个时间点, 每个时间点从数据库读取所有 agent 的状态
2. **按 agent 聚合**: 同一个 `track_token` 在不同时间点出现的状态拼成序列。至少需要 2 个状态点才能插值, 否则丢弃
3. **三次样条插值到 10Hz**: `t = 0.0, 0.1, 0.2, ..., 5.0` → 51 个时间点, 对每个 agent 的 `(x, y, heading, vx, vy)` 分别插值
4. **重建检测对象**: 每个时间步用插值后的位姿 + 原始尺寸 (长宽高不变) 重建 `OrientedBox`, 区分动态 `Agent` 和 `StaticObject`
5. **构建占用地图**: 每个时间步所有 agent → Shapely Polygon → `PDMOccupancyMap` → STRtree 空间索引

### 为什么要插值到 10Hz 而不是直接用 2Hz？

假设一辆车以 15 m/s 的速度横穿你面前:
- 2Hz: 两帧之间车移动了 7.5 米, 可能直接跳过你的车头, 碰撞检测完全漏掉
- 10Hz: 两帧之间车移动 1.5 米, 足以捕捉到碰撞发生的瞬间

### 与训练数据的对比

训练时模型**看不到任何显式的 agent 信息**。模型感知其他车辆的唯一途径是摄像头图像 — 从像素中隐式理解"前方有车在减速"、"左侧有行人要过马路"。

但评分时需要**精确的几何碰撞检测**, 不能靠视觉估计。所以 observation 从激光雷达标注中预计算, 提供 ground truth 级别的 agent 轨迹。这也意味着: 如果模型从图像中误判了其他车辆的位置或意图, 评分时碰撞照样会被精确检测出来。

---

## 5. centerline: PDMPath

规划路线的车道中心线 (全局 UTM 系)。

### 它是什么

一条沿规划路线的车道中心曲线, 由一系列离散点 `(x, y, heading)` 串联而成。可以理解为导航软件在地图上画的那条引导线, 但精度是厘米级的。

### 数据结构

```
PDMPath
├── _states_se2_array: ndarray (N, 3)
│     每行 = (x, y, heading), N 个离散采样点
│     e.g. N=200, 从当前位置往前延伸几百米
│
├── _progress: ndarray (N,)
│     每个点到起点的累积弧长 (米), 单调递增
│     e.g. [0.0, 0.5, 1.1, 1.8, ..., 487.3]
│     用途: 把"空间位置"映射到"沿路线走了多远"这个一维量
│
├── _interpolator: scipy.interpolate.interp1d
│     输入弧长 → 输出 (x, y, heading)
│     可以查询任意距离处的精确位姿
│
└── _linestring: Shapely LineString
      整条路径的几何表示, 用于:
      - project(): 把任意点投影到路径上, 得到最近点的弧长
      - distance(): 计算点到路径的横向偏移
```

N 的范围 50~500+ 是因为路线长度和弯曲程度差异很大。直路采样点少, 弯道采样点密集。

### 数据来源

PDMClosedPlanner 初始化时从 HD Map 构建:

1. 输入 `route_roadblock_ids` (从 nuPlan scenario 的路由规划获得)
2. 在 HD Map 中查找每个 roadblock 包含的 lane/lane_connector
3. 提取每条 lane 的中心线几何
4. 按拓扑顺序拼接: 处理车道连接、交叉口内的弧线段等
5. 得到一条从当前位置延伸到目的地方向的连续曲线

### 评分时如何使用

**ego_progress (连续值 [0, 1]):**

```
模型轨迹终点投影到 centerline 上 → 得到模型前进弧长 s_pred
参考轨迹终点投影到 centerline 上 → 得到参考前进弧长 s_ref
ego_progress = clip(s_pred / s_ref, 0, 1)
```

比如参考轨迹沿路线走了 40 米, 模型只走了 30 米, progress = 0.75。如果模型停在原地不动, progress ≈ 0。这个指标惩罚"过于保守"的行为。

**driving_direction_compliance (连续值 [0, 1]):**

在仿真轨迹的每个时间步:
1. 把自车位置投影到 centerline, 找到最近点
2. 取该点的 heading (centerline 的切线方向)
3. 计算自车 heading 与切线 heading 的夹角 Δθ
4. 如果 |Δθ| > 阈值 (比如 90°), 说明在逆行或严重偏航

### 与训练数据的对比

训练时模型收到的导航信息是 `driving_command` — 粗粒度的语言指令如 "turn left"、"go straight"。centerline 是这个指令对应的**精确几何实现** — "turn left" 到底是左转多少度、沿哪条车道、走多远, 都编码在 centerline 里。但模型在训练时永远看不到 centerline, 必须从视觉和指令中隐式学会路径跟踪能力。

---

## 6. route_lane_ids: List[str]

规划路线经过的所有车道 ID。

### 它是什么

一个字符串列表, 每个字符串是 HD Map 中一条车道的唯一 ID。构成了从起点到终点的完整路由 — 即"经过哪些路段"的集合。

```
e.g. ['68806', '69531', '69643', ...], 通常 50~150 个 ID
```

### 为什么有 50~150 个 ID 这么多

HD Map 中车道粒度很细:
- 一条直路被切分成多段 lane (每段几十米到上百米)
- 交叉口内的每条通行路径是独立的 `lane_connector`
- 路线两侧相邻车道也包含在内 (允许合理变道)

```
直行段1 (lane_1) → 路口1入口 (lane_2) → 路口1内 (connector_3) → 路口1出口 (lane_4) → ...
```

### 数据来源

`PDMClosedPlanner._route_lane_dict.keys()`, 从 HD Map 的路由规划结果提取。

### 为什么和 centerline 分开存？

两者互补, 解决不同类型的问题:

| | centerline | route_lane_ids |
|---|---|---|
| 表示 | 连续几何曲线 | 离散拓扑标识 |
| 回答的问题 | 走了多远？方向对不对？ | 在不在该走的路上？ |
| 典型场景 | 计算弧长、航向角差 | 判断是否偏离路线 |

举例: 双向四车道, 路线走右侧第二车道。如果模型开到了逆行车道, 从 centerline 看距离可能只有 7 米 (不算远), 但从 route_lane_ids 看, 逆行车道的 ID 不在列表中 → 偏离路线。

### 评分时如何使用

在 `scorer.score_proposals()` 中参与 **driving_direction_compliance** 计算:

1. 取仿真轨迹某时刻的自车位置
2. 在 HD Map 中查询该位置落在哪些 lane 内
3. 检查这些 lane ID 是否在 route_lane_ids 中
4. 完全跑出路线覆盖的所有 lane → 偏离路线

### 与训练数据的对比

训练数据里没有车道 ID 信息。模型从图像中识别车道线、路口结构, 结合 driving_command 推断应该走哪条车道。评分时 route_lane_ids 作为 ground truth 验证模型的车道选择。

---

## 7. drivable_area_map: PDMDrivableMap

可行驶区域的多边形集合 (全局 UTM 系)。

### 它是什么

自车周围 100 米范围内所有"车可以开的地方"的精确边界。由若干个多边形拼成, 覆盖车道、路口、停车场等区域。多边形之外就是不可行驶区域 — 人行道、草坪、建筑物、护栏等。

### 数据结构

```
PDMDrivableMap (继承自 PDMOccupancyMap)
├── _tokens: List[str]
│     每个多边形的唯一 ID, 对应 HD Map 中的 lane/connector/intersection
│
├── _geometries: ndarray[Shapely.Polygon]
│     20~100+ 个多边形, 每个是一条车道或一个路口的精确边界
│     顶点数不等 (直路 ~4 个顶点, 弯道 ~10-20 个)
│     e.g. Polygon([(637020.1, 3999480.2), (637020.5, 3999485.1), ...])
│
├── _map_types: List[SemanticMapLayer]
│     每个多边形的语义类型:
│     - LANE: 普通车道段
│     - LANE_CONNECTOR: 车道连接段 (路口内的转弯路径)
│     - INTERSECTION: 路口区域
│     - CARPARK_AREA: 停车场
│
└── _str_tree: STRtree
      R-tree 空间索引, O(log n) 查询效率
```

### 和 observation 的区别

两者都用了 `PDMOccupancyMap`, 但含义完全不同:

| | observation | drivable_area_map |
|---|---|---|
| 存的是什么 | 其他 agent 的占用区域 | 道路的可行驶区域 |
| 时间维度 | 51 个时间步 (agent 在动) | 单一静态地图 (路不会动) |
| 碰撞检测逻辑 | 自车与 agent **相交** → 碰撞 | 自车 **不在** polygon 内 → 越界 |
| 对应指标 | no_at_fault_collisions | drivable_area_compliance |

### 数据来源

PDMClosedPlanner 从 HD Map (nuplan-maps) 中查询自车附近 100m 内的所有可行驶区域多边形。

100 米半径的原因: 5 秒 × 最高约 20 m/s (72 km/h) = 100 米, 覆盖仿真时间内车辆理论上能到达的最远距离。

### 评分时的具体检测过程

`drivable_area_compliance` 是**二值门控 (一票否决)**:

对仿真轨迹的每个时间步 (40 步, 每 0.1 秒一帧):

1. 根据当前自车位姿 (后轴 x, y, heading) + 车辆尺寸, 计算车辆矩形 footprint 的 **4 个角点**坐标
2. 对每个角点, 在 STRtree 中查询是否被某个 drivable polygon 包含
3. 4 个角点必须**全部**被包含才算合规
4. **任何一帧任何一个角点越界 → 整个轨迹 compliance = 0, 最终得分直接归零**

为什么检查 4 个角点而不是中心点: 转弯时车尾可能甩出路面, 靠边行驶时车身一侧可能擦出路缘, 只查中心点会漏掉这些情况。

### 与训练数据的对比

训练时没有可行驶区域的显式标注。模型必须从摄像头图像中学会识别路面、路缘、路口边界。评分时用 HD Map 的厘米级精确多边形做 ground truth 判定 — 哪怕视觉感知偏差 0.5 米导致车轮压线, compliance 直接判零。这是对模型空间感知精度最严格的考验。

---

## 数据流总结

```
RAW NUPLAN DATA
  │
  ├──→ Metric Cache 创建 (MetricCacheProcessor):
  │      Input: nuPlan scenario (initial_ego_state, tracked_objects, map)
  │      Output: metric_cache.pkl (全局 UTM 系的 reference trajectory, observations, map)
  │
  └──→ 训练 JSON 创建 (preprocessing scripts):
         Input: nuPlan scenario frames
         Output: JSON with 自车系 gt_trajectory + sensor paths

训练阶段:
  JSON + sensor blobs → ActionTokenizer 量化 → action tokens → 训练模型

评分阶段:
  模型生成 action tokens → ActionTokenizer 解码 → 自车系轨迹 [10, 3]
  → pdm_score() 加载 MetricCache
  → relative_to_absolute_poses() 转全局系
  → PDMSimulator 闭环仿真 (运动学约束)
  → PDMScorer 评分 (碰撞 + 可行驶区域 + 进度 + TTC + 舒适度 + 方向)
  → PDMResults (6 个子指标 + 综合得分 0~10)
```
