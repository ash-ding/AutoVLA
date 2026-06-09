# rlib1.1 — Propositional Logic CoT, OPERATIONS removed
# rlib1.1 —— 命题逻辑 CoT,去掉 OPERATIONS 阶段

> Minimal-diff cleanup of [rlib1.0](../rlib1_0/README.md): drops the
> redundant OPERATIONS stage and folds ego state into PERCEPTION as a
> first-class entity. Same propositional-logic semantics, shorter CoT,
> cleaner separation of concerns.
>
> [rlib1.0](../rlib1_0/README.md) 的最小化改版:删掉冗余的 OPERATIONS 阶段,
> 把 ego 状态作为一个 entity 并入 PERCEPTION。命题逻辑语义不变,CoT 更短,
> 各阶段职责更干净。

## 1. Changes vs rlib1.0 / 相对 rlib1.0 的改动

| Area / 维度 | rlib1.0 | rlib1.1 |
|---|---|---|
| Number of stages / 阶段数 | 5 (PERCEPTION → **OPERATIONS** → FACTS → RULES → ACTION) | **4** (PERCEPTION → FACTS → RULES → ACTION) |
| Ego state / 自车状态 | Queried via `EgoQuery(speed) = Slow` in OPERATIONS | Declared as `ego = Ego {speed: Slow, acceleration: Braking, instruction: KeepForward}` in PERCEPTION |
| `rlib/operations.yaml` | Required (Query/Filter/Count/Exists/Nearest/EgoQuery) | **Deleted** (loader treats it as optional) |
| `rlib/entities.yaml` | 6 entity types | 7 entity types — adds **`Ego`** with attrs `speed / acceleration / instruction` |
| `rlib/facts.yaml` | Ego-state facts use `kind: ego` grounding (`EgoQuery(...) == value`) | Same fact names, but grounding rewritten to `kind: entity, base_type: Ego, attributes: {...}` |
| OPERATIONS in model output | Required | Allowed but optional — parser warns if present, treats as legacy no-op |

**Motivation / 动机**:
- In rlib1.0, `Query(tl_1, color) = Red` just repeats what PERCEPTION
  already says. The OPERATIONS section was carrying redundant noise that
  pollutes the SFT signal and lengthens the CoT.
  <br/>rlib1.0 里 `Query(tl_1, color) = Red` 只是把 PERCEPTION 里已写过的
  事重复一遍;OPERATIONS 段携带的冗余噪声污染 SFT 信号、还拉长 CoT
- The one thing OPERATIONS was actually needed for — querying ego state
  via `EgoQuery(...)` — is replaced by making `Ego` a normal entity.
  Cleanest unification.
  <br/>OPERATIONS 真正不可或缺的部分是 `EgoQuery(...)` 查 ego 状态。把 `Ego`
  改成普通 entity 之后,这部分需求自动消化

## 2. Format / 格式

4 stages instead of 5; ego is a regular entity in PERCEPTION:

4 段而非 5 段;ego 是 PERCEPTION 里的普通 entity:

```
PERCEPTION:
  tl_1 = TrafficLight {color: Red, position: Front, applies_to: EgoLane}
  v_1  = Car {position: Front, lane: EgoLane, distance: Near, motion: Stationary, signal: BrakeLights}
  r_1  = Intersection {position: Ahead, distance: Near}
  ego  = Ego {speed: Slow, acceleration: Braking, instruction: KeepForward}  # raw: 1.8 m/s, -1.20 m/s^2

FACTS:
  RedLight = True
  ApproachingIntersection = True
  LeadVehicleStopped = True
  CanStopComfortably = True
  EgoMovingSlow = True
  InstructionKeepForward = True

RULES:
  RedLight AND ApproachingIntersection AND CanStopComfortably AND NOT EgoStopped
    → KeepLane, DecelerationToZero

ACTION: KeepLane, DecelerationToZero
```

## 3. Verifier / 验证器

Same `SymbolicValidator.validate(output) → (is_valid, violations,
grounding_warnings)` API as rlib1.0. Same 5 structural checks
(`_check_entities`, `_check_operations`, `_check_facts`, `_check_rules`,
`_check_action`); same `symbolic_valid` semantics.

API 和 rlib1.0 一致(`(is_valid, violations, grounding_warnings)`),5 项结构检查
也照旧(`_check_entities`、`_check_operations`、`_check_facts`、`_check_rules`、
`_check_action`),`symbolic_valid` 含义不变。

Two backward-compatibility tweaks make rlib1.1 tolerate inputs from both
generations:

两处向后兼容微调,使 rlib1.1 同时接受新旧两代输入:

1. **`SymbolicSchema._init_from_rlib`** treats `operations.yaml` as
   optional. If the file is absent (which rlib1.1 deliberately removes),
   the operations vocab is just empty.
   <br/>`SymbolicSchema._init_from_rlib` 把 `operations.yaml` 当作可选文件。
   不存在时(rlib1.1 故意删掉了),operations 词表为空
2. **`SymbolicParser._split_sections`** still recognizes an `OPERATIONS:`
   header in model output (so rlib1.0-style legacy outputs still parse)
   but treats it as informational — the validator simply skips
   `_check_operations` if no operations were declared.
   <br/>parser 依然识别 `OPERATIONS:` 段头(rlib1.0 的旧输出还能解析),但当作
   信息性内容 —— validator 在 operations 列表为空时跳过 `_check_operations`

### `grounding_score` semantics unchanged / `grounding_score` 含义不变

The grounding evaluator now resolves ego-state facts (`EgoStopped`,
`InstructionKeepForward`, etc.) by looking for an `Ego` entity in
PERCEPTION instead of inspecting OPERATIONS. The score formula is
identical:

grounding 评估器现在对 ego 类 fact(`EgoStopped`、`InstructionKeepForward` 等)
转去 PERCEPTION 找 `Ego` entity,不再翻 OPERATIONS。分数公式不变:

$$
\text{grounding\_score} \;=\; \frac{\text{\#~grounded True facts}}{\text{\#~checkable True facts}}
$$

## 4. Output JSON fields / 输出 JSON 关键字段

Same as rlib1.0 — `cot_format = "symbolic"`, `cot_output`,
`symbolic_valid`, `symbolic_violations`, `grounding_warnings`,
`grounding_score`. See [rlib1.0 README §3](../rlib1_0/README.md#3-output-json-fields--输出-json-关键字段)
for the full table.

字段集与 rlib1.0 完全一致 —— `cot_format = "symbolic"`、`cot_output`、
`symbolic_valid`、`symbolic_violations`、`grounding_warnings`、`grounding_score`。
完整说明见 [rlib1.0 README §3](../rlib1_0/README.md#3-output-json-fields--输出-json-关键字段)。

## 5. Limitations / 已知局限

Inherits rlib1.0's two big ones unchanged:

继承 rlib1.0 的两个根本局限,本版本未解决:

- **Still purely qualitative** — `Near` / `Far` / `Slow` / `Fast`. No
  per-sample thresholds, no TTC, no `:=` arithmetic. Move to rlib2.0 for
  that.
  <br/>**依然纯质化** —— `Near`/`Far`/`Slow`/`Fast` 标签,没有逐样本阈值、
  没有 TTC、没有 `:=` 算术。要换到 rlib2.0 才能拿到这些
- **No external grounding** — `grounding_score` only checks whether the
  model's PERCEPTION supports its FACTS, not whether the PERCEPTION
  itself is faithful to the actual scene.
  <br/>**无外部 grounding** —— `grounding_score` 只检查"PERCEPTION ⊢ FACTS",
  不验证 PERCEPTION 是否忠实于真实场景

## 6. Files / 文件清单

Identical layout to rlib1.0, minus `operations.yaml`:

目录结构和 rlib1.0 一致,只是少了 `operations.yaml`:

```
rlib1_1/
├── README.md       ← this file
├── prompt.py       ← 4-stage prompt template (no OPERATIONS section)
├── verifier.py     ← SymbolicSchema/Parser/Validator (operations.yaml optional)
└── rlib/
    ├── entities.yaml   ← + Ego entity type
    ├── facts.yaml      ← ego facts re-grounded via entity attr checks
    ├── actions.yaml    ← unchanged
    ├── thresholds.yaml ← unchanged
    └── rules/          ← unchanged 7 category files
```
