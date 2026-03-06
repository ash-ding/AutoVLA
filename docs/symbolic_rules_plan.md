# Plan: 在 AutoVLA 中引入符号规则（Neuro-Symbolic Reasoning）

## Context

**Proposal 核心思想**: Choi & Liu 提出将人类可理解的符号规则（symbolic rules）嵌入 VLM 的决策过程。VLM 不再直接从感知映射到控制，而是先生成一组符号规则链（如 `(TrafficLight=Yellow) AND CrossRoad AND (Velocity > 10m/s) → Decelerate`），再由程序执行器转换为控制命令。通过 GRPO 强化学习优化：`r = r_driving - λ * r_complexity`，同时最大化驾驶性能并最小化规则复杂度。

**AutoVLA 现状**: 已有完整的 CoT（Chain-of-Thought）机制——模型在 `<think>...</think>` 中生成自然语言推理，然后在 `<answer>...</answer>` 中输出 action tokens。GRPO 训练管道也已就绪。这为引入符号规则提供了天然的切入点。

**关键洞察**: 符号规则本质上是对现有 CoT 的**结构化升级**——从自由文本推理变为可解析、可执行、可审计的符号逻辑。

---

## 大致思路

### Phase 1: 定义符号规则的 DSL（Domain-Specific Language）

**目标**: 设计一套驾驶场景的符号操作语言

需要定义的核心元素：
- **感知谓词（Perception Predicates）**: `Query(ObjectType, Attribute)` — 如 `Query(TrafficLight, Color)`, `Query(Pedestrian, Distance)`, `Query(Vehicle, RelativeSpeed)`
- **条件表达式**: `=`, `>`, `<`, `AND`, `OR`, `NOT`
- **高层动作（High-level Actions）**: `Accelerate`, `Decelerate`, `MaintainSpeed`, `LaneKeepLeft`, `LaneKeepCenter`, `Stop`, `Yield` 等
- **规则格式**: `CONDITION_1 AND CONDITION_2 → ACTION`

示例规则库：
```
Query(TrafficLight, Count) > 0 → CrossRoad
(TrafficLight = Red) AND CrossRoad → Stop
(TrafficLight = Yellow) AND CrossRoad AND (Velocity > 10) → Decelerate
Query(Pedestrian, Distance) < 15 AND (Pedestrian_in_Crosswalk = True) → Yield
(LeadVehicle_Distance < 8) AND (LeadVehicle_Decelerating = True) → Decelerate
```

**对应代码工作**:
- 新建 `models/symbolic_rules.py` — 规则 DSL 定义、解析器（parser）、验证器
- 新建 `config/rules/` — YAML 格式的规则库定义

---

### Phase 2: 修改 VLM 输出格式，生成符号规则

**目标**: 让模型在 `<think>` 阶段输出结构化的符号规则，而非自由文本

**现有 CoT 格式** (in `dataset_utils/sft_dataset.py` & `models/autovla.py`):
```
<think>
This is a complex scenario. The ego vehicle is approaching an intersection...
Scene analysis: ...
Critical objects: ...
</think>
<answer>
The final output action is: <action_1831><action_42>...
</answer>
```

**新格式**:
```
<think>
RULES:
Query(TrafficLight, Color) = Green
Query(LeadVehicle, Distance) = 25m
Query(LeadVehicle, Speed) > EgoSpeed
(TrafficLight = Green) AND (LeadVehicle_Distance > 15) → MaintainSpeed
SELECTED_ACTION: MaintainSpeed
</think>
<answer>
The final output action is: <action_1831><action_42>...
</answer>
```

**对应代码工作**:
- 修改 `models/autovla.py` — `get_prompt()` 中的 system prompt，指导模型按符号格式输出
- 修改 `dataset_utils/preprocessing/cot_prompts.py` — 更新 CoT prompt 模板
- 新建 `dataset_utils/preprocessing/symbolic_cot_annotation.py` — 用大模型（72B）将现有自然语言 CoT 转换为符号规则格式的标注工具
- 修改 `dataset_utils/sft_dataset.py` — 在 `__getitem__` 中处理新的符号 CoT 格式

---

### Phase 3: 实现符号规则解析器和程序执行器

**目标**: 解析模型输出的符号规则，验证其逻辑一致性，并可选地执行规则

**对应代码工作**:
- 扩展 `models/symbolic_rules.py`:
  - `RuleParser.parse(text) → List[Rule]` — 从模型输出中提取符号规则
  - `RuleExecutor.execute(rules, scene_context) → HighLevelAction` — 执行规则链得到高层动作
  - `RuleValidator.validate(rules) → (is_valid, violations)` — 检查规则是否自洽、是否符合交通法规
- 修改 `models/autovla.py` — 在 `predict()` 中添加规则解析步骤，在返回 trajectory 的同时返回解析后的规则链

---

### Phase 4: 引入规则复杂度奖励到 GRPO 训练

**目标**: 实现 `r = r_driving - λ * r_complexity`

**复杂度度量（r_complexity）候选指标**:
- 规则数量（鼓励简洁）
- 规则中条件子句的总数
- 规则是否可解析（不可解析直接惩罚）
- 规则是否与最终 action token 语义一致（一致性检查）

**对应代码工作**:
- 修改 `models/utils/score.py` — 新增 `compute_rule_complexity(rules_text)` 函数
- 修改 `models/autovla.py` — `GRPOAutoVLA.reward_function()` 中：
  - 现有：`reward = pdm_score - cot_penalty`
  - 改为：`reward = pdm_score - λ_complexity * rule_complexity - λ_validity * (1 - rule_validity)`
  - 其中 `rule_validity` 是规则是否可解析且语义一致的 0/1 奖励
- 修改 GRPO 训练 config 新增 `lambda_complexity`, `lambda_validity` 超参

---

### Phase 5: SFT 数据准备（符号规则标注）

**目标**: 为 SFT 训练准备带符号规则的标注数据

**方法**:
1. 用 Qwen2.5-VL-72B（通过现有的 vLLM/OpenAI annotation pipeline）对训练场景生成符号规则标注
2. 或者：对已有的自然语言 CoT 标注做后处理，用 LLM 转换为符号格式

**对应代码工作**:
- 基于现有 `dataset_utils/preprocessing/vllm_cot_annotation_model.py` 修改 prompt，让 72B 模型输出符号格式
- 编写转换脚本：`tools/preprocessing/convert_cot_to_symbolic.py`

---

### Phase 6: 评估与可视化

**目标**: 评估符号规则的质量和对驾驶性能的影响

**指标**:
- 驾驶性能：PDM score（现有）
- 规则可解析率：生成的规则中有多少能被 parser 正确解析
- 规则一致性：符号规则推导出的高层动作与实际 action token 的语义一致率
- 规则简洁性：平均规则数、平均条件数

**对应代码工作**:
- 修改 `navsim/navsim/agents/autovla_agent.py` — 在评估时记录每个场景的符号规则
- 新建 `tools/eval/evaluate_symbolic_rules.py` — 规则质量评估脚本
- 可选：简单的可视化界面（场景图 + 规则链 + 动作）

---

## 关键修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `models/symbolic_rules.py` (新建) | 规则 DSL、解析器、执行器、验证器 |
| `models/autovla.py` | system prompt 修改、predict() 添加规则解析、GRPOAutoVLA reward 修改 |
| `models/utils/score.py` | 新增 rule_complexity 计算 |
| `dataset_utils/sft_dataset.py` | 支持符号 CoT 格式 |
| `dataset_utils/preprocessing/cot_prompts.py` | 符号规则 prompt 模板 |
| `config/rules/driving_rules.yaml` (新建) | 符号规则库 |
| `config/training/` 下新增配置 | GRPO 训练的符号规则相关超参 |
| `navsim/navsim/agents/autovla_agent.py` | 评估时记录符号规则 |

## 建议的实施顺序

1. **先做 Phase 1 + Phase 3** — 定义好 DSL 和解析器，这是整个系统的基础
2. **Phase 5** — 准备 SFT 标注数据
3. **Phase 2** — 修改模型 prompt 和数据格式，做 SFT 训练
4. **Phase 4** — 在 GRPO 中加入复杂度奖励
5. **Phase 6** — 评估和迭代

## 设计决策

- **执行方式**: Soft constraint — 符号规则作为结构化 CoT，通过 GRPO reward 间接约束行为（规则与动作不一致时扣分），最终动作仍由 action token 决定
- **评估平台**: 先在 NAVSIM 上验证（复用现有 PDM score pipeline），后续按需迁移 CARLA

## 验证方法

1. 单元测试：规则解析器能正确解析各种格式的规则
2. SFT 后模型能生成可解析的符号规则（目标：>80% 可解析率）
3. GRPO 训练后 PDM score 不显著下降（<0.5 分），同时规则更简洁
4. 在 NAVSIM 评估集上运行完整 evaluation pipeline
