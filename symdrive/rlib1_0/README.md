# rlib1.0 — Predefined Propositional Logic CoT
# rlib1.0 —— 预定义命题逻辑 CoT

> Baseline of the symbolic-CoT family. Inherited (mostly) from the original
> AutoVLA codebase, kept under `symdrive/rlib1_0/` so newer variants can be
> compared against it.
>
> symbolic CoT 系列的基线版本。基本继承自 AutoVLA 原始实现,搬到
> `symdrive/rlib1_0/` 下作为后续版本对照的参照系。

## 1. Format / 格式

Five sequential sections inside the model's `<think>...</think>` block:

模型 `<think>...</think>` 块内的 5 段顺序结构:

```
PERCEPTION:
  v_1 = Car {position: Front, lane: EgoLane, distance: Near, motion: Moving, ...}
  tl_1 = TrafficLight {color: Red, ...}
  ...

OPERATIONS:
  Query(tl_1, color) = Red
  Filter(Vehicle, lane = EgoLane) = {v_1}
  EgoQuery(speed) = Slow             # 1.8 m/s
  EgoQuery(acceleration) = Braking   # -1.20 m/s^2
  EgoQuery(instruction) = KeepForward
  ...

FACTS:
  RedLight = True
  LeadVehicleClose = True
  CanStopComfortably = True
  ...

RULES:
  RedLight AND CanStopComfortably AND NOT EgoStopped → KeepLane, DecelerationToZero

ACTION: KeepLane, DecelerationToZero
```

**Key characteristics**:
- **Qualitative attributes only** — distances are `Near/Medium/Far`,
  speeds are `Slow/Medium/Fast`, etc. (bucketized via `rlib/thresholds.yaml`)
- **Ego state is queried via `EgoQuery(...)`** in OPERATIONS, not declared
  as an entity in PERCEPTION
- **Predefined rule library** at `rlib/rules/*.yaml` (TC, CA, PS, SM, EM,
  NV, CM categories) — the teacher VLM is encouraged to pick one of these
  pre-written rules (with `--no-free-rules`) or compose its own
  (`--free-rules`, default)
- Fact vocabulary fixed in `rlib/facts.yaml`

**关键特征**:
- **只用质化属性** —— 距离用 `Near/Medium/Far`,速度用 `Slow/Medium/Fast`
  这种离散标签(通过 `rlib/thresholds.yaml` 把数值分桶得到)
- **Ego 状态通过 OPERATIONS 阶段的 `EgoQuery(...)` 查询**,不是 PERCEPTION 里的一个 entity
- **预定义规则库**在 `rlib/rules/*.yaml`(7 类:TC 交通管制、CA 碰撞避免、
  PS 行人安全、SM 速度管理、EM 紧急情况、NV 导航、CM 舒适性)。teacher VLM
  可以从这些规则里挑(`--no-free-rules`),也可以自己合成(`--free-rules`,默认)
- Fact 词表固化在 `rlib/facts.yaml`

## 2. Verifier / 验证器

[`verifier.py`](verifier.py) defines `SymbolicValidator.validate(output) →
(is_valid, violations, grounding_warnings)`. The boolean output field
`symbolic_valid` written into each sample JSON is `True` iff **all five
structural checks pass** (i.e., `len(violations) == 0`):

[`verifier.py`](verifier.py) 里的 `SymbolicValidator.validate(output) →
(is_valid, violations, grounding_warnings)` 是核心。每个 sample JSON 写出的
`symbolic_valid` 字段为 `True` **当且仅当下面 5 项结构检查全部通过**(等价
于 `len(violations) == 0`):

| # | Check / 检查 | What / 检查什么 |
|---|---|---|
| 1 | `_check_entities` | No duplicate entity IDs; every type/subtype is in `entities.yaml`; every attribute value is a member of the declared vocabulary. <br/>无重复 ID;实体类型/子类型都在 `entities.yaml` 词表里;每个属性的值在声明的合法集合里 |
| 2 | `_check_operations` | Each OPERATIONS line parses (`Query`/`Filter`/`Count`/`Exists`/`Nearest`/`EgoQuery` syntax) and references a declared entity ID or a valid `EgoQuery` attribute. <br/>每条 OPERATIONS 解析正常(`Query`/`Filter`/`Count`/`Exists`/`Nearest`/`EgoQuery` 语法),引用的 entity ID 已在 PERCEPTION 声明 |
| 3 | `_check_facts` | Each fact name appears in `facts.yaml::vocabulary`. <br/>每个 fact 名字必须在 `facts.yaml::vocabulary` 里 |
| 4 | `_check_rules` | For each rule, every condition atom is a declared FACT; the rule head is in the lateral × longitudinal action vocab. <br/>规则体里每个原子都是已声明的 FACT;规则头落在 `actions.yaml` 的 lateral × longitudinal 词表内 |
| 5 | `_check_action` | The final ACTION line matches the head of at least one rule. <br/>ACTION 段和至少一条 RULE 的 head 匹配 |

### `grounding_score` (separate, continuous metric / 独立的连续指标)

After the 5 structural checks, the verifier ALSO measures whether the
`True` facts the model declared are actually supported by the entities /
operations the model wrote down. The score is in [0, 1]:

5 项结构检查之外,verifier 还衡量"模型声称为 True 的 fact 是否真的能从模型
写下的 entity / operation 里推出"。分数在 [0, 1] 区间:

$$
\text{grounding\_score} \;=\; \frac{\text{\#~grounded True facts}}{\text{\#~checkable True facts}}
$$

- **Checkable True facts** = True facts whose `facts.yaml` grounding kind
  is not `judgment` (judgment facts like `CanStopComfortably` can't be
  mechanically verified and are excluded from the denominator).
  <br/>分母 = grounding 类型 `≠ judgment` 的 True facts。`CanStopComfortably`
  这种"判断性"事实没法机械验证,排除在分母外
- **Grounded** = the grounding condition (e.g. "exists Vehicle with
  position=Front, lane=EgoLane, distance ∈ {Near, VeryNear}") evaluates
  True against the declared PERCEPTION/OPERATIONS.
  <br/>分子 = 在 PERCEPTION/OPERATIONS 里能找到匹配 grounding 模板的 fact

A score of `0.50` means **half the checkable True facts are "asserted but
unsupported"** — the model said e.g. `LeadVehicleClose = True` but didn't
declare any matching `Vehicle` entity in PERCEPTION. The specific
unsupported facts are listed in `grounding_warnings`.

`grounding_score = 0.50` 表示**有一半可机械验证的 True facts 是"空中宣称"**
—— 模型写了 `LeadVehicleClose = True` 但 PERCEPTION 里没有对应的 Vehicle 声明。
具体哪些 fact 不达标在 `grounding_warnings` 数组里。

## 3. Output JSON fields / 输出 JSON 关键字段

[`tools/preprocessing/symbolic_cot_sample_generation.py`](../../tools/preprocessing/symbolic_cot_sample_generation.py)
writes these fields per sample:

每个 sample 的 JSON 由
[`tools/preprocessing/symbolic_cot_sample_generation.py`](../../tools/preprocessing/symbolic_cot_sample_generation.py)
写入以下字段:

| Field / 字段 | Meaning / 含义 |
|---|---|
| `cot_format`  | Always `"symbolic"` for this pipeline. <br/>本 pipeline 固定为 `"symbolic"` |
| `cot_output`  | Raw 5-stage text emitted by the teacher VLM. <br/>teacher 输出的 5 段原文 |
| `symbolic_valid` | `True` iff all 5 verifier checks pass. <br/>5 项检查全过则 True |
| `symbolic_violations` | List of specific violations when `symbolic_valid=False`. <br/>失败时,具体违反列表 |
| `grounding_warnings`  | Per-fact warnings for True facts without PERCEPTION/OPERATIONS evidence. <br/>对每条找不到支撑的 True fact 的警告 |
| `grounding_score` | Float in [0, 1] — see formula above. <br/>0~1 之间的浮点数,公式见上 |

## 4. Files / 文件清单

- [`prompt.py`](prompt.py) — assembles the 5-stage prompt fed to the teacher VLM,
  including 4 in-context examples and the RLIB schema sections.
  <br/>组装喂给 teacher VLM 的 5 段 prompt,含 4 个 in-context examples 和 RLIB
  schema 各段
- [`verifier.py`](verifier.py) — `SymbolicSchema` / `SymbolicParser` /
  `SymbolicValidator` (`Validator.validate()` runs the 5 structural checks +
  optional grounding check).
  <br/>`SymbolicSchema` / `SymbolicParser` / `SymbolicValidator`(后者跑 5 项结构
  检查 + 可选的 grounding 检查)
- [`rlib/entities.yaml`](rlib/entities.yaml) — 7 entity types + qualitative attrs.
  <br/>7 种实体类型 + 质化属性
- [`rlib/facts.yaml`](rlib/facts.yaml) — 36 boolean facts with formal grounding conditions.
  <br/>36 条 boolean fact + 形式化 grounding 条件
- [`rlib/operations.yaml`](rlib/operations.yaml) — 6 query operators (Query, Filter, ...).
  <br/>6 个查询算子(Query、Filter 等)
- [`rlib/actions.yaml`](rlib/actions.yaml) — lateral × longitudinal action vocab.
  <br/>横纵向动作词表
- [`rlib/thresholds.yaml`](rlib/thresholds.yaml) — speed / acceleration bucketization.
  <br/>速度 / 加速度的分桶阈值
- [`rlib/rules/*.yaml`](rlib/rules/) — 7 category files of predefined rules.
  <br/>7 类预定义规则文件

## 5. Limitations / 已知局限

- **Loses quantitative information** — bucketizing 8.3 m to "Near" throws
  away the exact value, so the model can't reason about TTC, RSS-style
  safe distance, required deceleration, etc.
  <br/>**量化信息丢失** —— 把 8.3 m 桶化成 "Near",精确数值就消失了,模型没法
  推 TTC、RSS 安全距离、所需减速度这些
- **Grounding is best-effort** — the `grounding_score` measures whether
  the model's PERCEPTION entities support its FACTS, but doesn't verify
  PERCEPTION itself is faithful to the actual scene
  <br/>**Grounding 只是尽力检验** —— `grounding_score` 衡量的是"PERCEPTION ⊢ FACTS"
  这一步,但**不验证 PERCEPTION 本身是否符合真实场景**
- **OPERATIONS layer is largely redundant** — single-attribute `Query`
  calls just repeat what PERCEPTION already declared; only `EgoQuery` is
  truly necessary because ego state isn't in PERCEPTION. This motivated
  rlib1.1.
  <br/>**OPERATIONS 阶段冗余度高** —— 单属性 `Query` 只是把 PERCEPTION 里已经
  写过的东西重复一遍;真正必要的只有 `EgoQuery`(因为 ego 不在 PERCEPTION 里)。
  这是 rlib1.1 改造的动机
