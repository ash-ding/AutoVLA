# rlib2.0 — Datalog¬ + Arithmetic CoT
# rlib2.0 —— Datalog¬ + 算术 CoT

> A fundamentally different design from rlib1.x. Numerical PERCEPTION
> attributes, `:=` grounding expressions on every FACT, a single
> Datalog¬ rule per scene, and a 5-layer Z3-backed verifier. Verifies
> internal logical consistency end-to-end — every FACT's truth value
> must be reproducible by re-executing its declared expression.
>
> 和 rlib1.x 在设计上有本质区别。PERCEPTION 用数值属性,每条 FACT 必须有
> `:=` grounding 表达式,每个场景一条 Datalog¬ 规则,5 层 Z3 验证器。端到端
> 验证内部逻辑自洽 —— 每条 FACT 的真值都必须能通过重新执行其声明的表达式
> 复现出来。

## 1. Why rlib2.0? / 为什么要做 rlib2.0

[rlib1.x](../rlib1_0/README.md) cannot represent quantitative scene
information: distance `Near` could be 2 m or 12 m; speed `Slow` could be
0.5 m/s or 3 m/s. That kills downstream reasoning about TTC, RSS safe
distance, required deceleration, and any threshold-tuning needed for
nuanced decisions. rlib2.0 keeps the raw numerics in PERCEPTION, lets
FACTS derive booleans via arbitrary arithmetic comparisons, and validates
the chain end-to-end with Z3.

[rlib1.x](../rlib1_0/README.md) 无法表达定量场景信息:"距离 Near" 可能是 2 m
也可能是 12 m;"速度 Slow" 可能是 0.5 m/s 也可能是 3 m/s。下游想要 TTC、
RSS 安全距离、所需减速度、阈值微调,统统做不到。rlib2.0 在 PERCEPTION 里
保留原始数值,FACTS 通过任意算术比较推 Boolean,然后用 Z3 端到端验证整条链。

## 2. Changes vs rlib1.x / 相对 rlib1.x 的改动

| Area / 维度 | rlib1.x | rlib2.0 |
|---|---|---|
| **PERCEPTION attrs / 属性** | Qualitative labels (`Near`/`Slow`/...) | **Real numerical values** (`dist: 8.3`, `vel: 2.1`, ...) |
| **FACTS / 事实** | Bare assertions: `RedLight = True` | **Grounded via `:=`**: `red_imminent := tl.color == "red" ∧ tl.dist < 30 → True` |
| **Fact vocab / 词表** | Closed set in `facts.yaml` | **Soft / suggested** — LLM may use canonical names OR invent new ones |
| **OPERATIONS** | Required, contains Query/Filter/EgoQuery | **Optional**, holds intermediate arithmetic derivations (TTC, required_decel, ...) |
| **RULES** | One or more rules; body uses `AND`/`NOT` over FACTS | **Single rule per scene**; body restricted to Boolean atoms from FACTS (no inline arithmetic) |
| **ACTION** | `KeepLane, DecelerationToZero` (free text) | `(LateralAction, LongitudinalAction)` — must literally match rule head |
| **Predefined rules / 预定义规则** | 7 category files under `rules/*.yaml` | **None** — fully free rule composition over the FACT vocabulary |
| **Verifier / 验证器** | 5 structural checks + grounding score | **5 numbered layers L1-L5**, Z3-backed |
| **Camera label policy / 相机标注** | (inherited from upstream — incorrect "FL/FR" labels in nuPlan prompt) | (same image input — unchanged in rlib2.0 prompts; viewer-level fix only) |

## 3. Format (Option C surface syntax) / 格式

5 sections; OPERATIONS is optional and only used when a derived quantity
is reused or non-trivial:

5 段;OPERATIONS 为可选,只有派生量需要复用或计算复杂时才用:

```
[PERCEPTION]
v_1  = Car {dist: 8.3, lat_offset: 0.2, vel: 2.1, lane: ego, signal: brake_lights}
tl_1 = TrafficLight {color: red, dist_to_stopline: 14.5, applies_to: ego_lane}
ego  = Ego {speed: 7.8, accel: -0.5, instruction: KeepForward}

[OPERATIONS]
TTC_lead       := v_1.dist / max(0.1, ego.speed - v_1.vel)        = 1.46
required_decel := ego.speed ** 2 / (2 * tl_1.dist_to_stopline)    = 2.10

[FACTS]
red_imminent      := tl_1.color == "red" ∧ tl_1.dist_to_stopline < 30          → True
can_stop_smoothly := required_decel < 3.0                                      → True
lead_too_close    := TTC_lead < 2.0                                            → True

[RULES]
R: red_imminent ∧ can_stop_smoothly ∧ ¬lead_too_close → (KeepLane, DecelerationToZero)

[ACTION]
KeepLane, DecelerationToZero
```

**Hard rules** (enforced by the verifier):

**硬性规则**(由 verifier 强制):

- Every FACT MUST have a `:=` right-hand side that references PERCEPTION
  or OPERATIONS identifiers — bare `xxx := True` is rejected at L2.
  <br/>每条 FACT 的 `:=` 右侧必须引用 PERCEPTION 或 OPERATIONS 里的标识符 ——
  裸的 `xxx := True` 在 L2 被拒
- RULE body uses **only FACT atoms** (with optional `¬` / `NOT`); no
  inline arithmetic, no `<` / `>` in rule body.
  <br/>RULE body **只能用 FACT 原子**(可选 `¬` / `NOT`);body 里不允许
  inline 算术,不允许 `<` / `>`
- Exactly one rule per scene (multi-rule + `Select:` is rejected — keeps
  the decision chain simple and verifiable).
  <br/>每个场景**恰好一条规则**(不支持多 rule + `Select:` —— 决策链保持简洁可验证)
- ACTION must be a 2-tuple literally equal to the RULE head.
  <br/>ACTION 必须是和 RULE 头完全相等的二元组

## 4. The 5-layer verifier / 5 层验证器

[`verifier.py`](verifier.py) exposes `verify(cot_text, rlib_dir) → dict`.
The dict has `L1..L5` booleans plus a `score = mean(L1..L5)` in [0, 1].

[`verifier.py`](verifier.py) 暴露 `verify(cot_text, rlib_dir) → dict`,字段为
`L1..L5` 五个 boolean 加上 `score = mean(L1..L5)`,落在 [0, 1] 区间。

| Layer / 层 | Check / 检查的事 | Tool / 工具 |
|---|---|---|
| **L1 — Syntax / 语法** | The 5-section template parses; each PERCEPTION/OPERATIONS/FACTS/RULES line matches its regex; ACTION is a 2-tuple. <br/>5 段模板成功 parse;每条 PERCEPTION/OPERATIONS/FACTS/RULES 行匹配各自正则;ACTION 是二元组 | hand-rolled parser |
| **L2 — References / 引用** | Every identifier in OPERATIONS/FACTS expressions resolves (declared in PERCEPTION or earlier in the same section); every atom in RULES is a declared FACT; FACT expressions are non-trivial (must mention at least one PERCEPTION/OPERATIONS identifier — `:= True` is rejected). <br/>OPERATIONS/FACTS 表达式里的每个标识符都已声明;RULES 里的每个原子都是已声明 FACT;FACT 的 `:=` 必须非平凡(至少引用一个 PERCEPTION/OPERATIONS 标识符,纯 `:= True` 被拒) | `ast.parse` + name collection |
| **L3 — Arithmetic / 算术求值** | Re-evaluate each OPERATIONS / FACTS `:=` right-hand side against the declared PERCEPTION values, and compare to the declared result. Numeric tolerance 5 % relative (with 0.05 absolute floor) — accommodates LLM rounding like `0.11` vs the exact `0.1117…`. <br/>对每条 OPERATIONS / FACTS 的 `:=` 右侧在 PERCEPTION 上重新求值,和声明结果对比。数值容差 5 % 相对(绝对下限 0.05),允许 LLM 把 `0.1117…` 写成 `0.11` | restricted `ast` evaluator |
| **L4 — Rule body satisfiability / 规则体可满足** | Pin each FACT to its L3-confirmed boolean, then ask Z3 whether the rule body (e.g. `red_imminent ∧ can_stop_smoothly ∧ ¬lead_too_close`) is satisfied. Catches "the model wrote a rule whose body contradicts its own FACTS". <br/>把每个 FACT 钉死成 L3 验过的 boolean,然后让 Z3 检查规则 body 是否成立。专门抓"模型写的规则 body 跟它自己声明的 FACTS 自相矛盾"这种 case | `z3-solver` |
| **L5 — Action match / 动作匹配** | The `[ACTION]` tuple equals the rule head; both lateral and longitudinal actions are in the actions.yaml vocab. <br/>`[ACTION]` 元组与规则头字面相等;lateral / longitudinal 两个动作都在 actions.yaml 词表内 | string compare |

A score of `1.0` means all five layers pass. A score of `0.6` means three
of five pass (typically L1+L2+L3 — the model wrote syntactically clean
arithmetic — but the rule body doesn't actually follow from the FACTS).
`violations` lists which layer(s) failed and why.

`score = 1.0` 表示 5 层全过。`score = 0.6` 表示 5 层过 3 层(常见组合 L1+L2+L3
全过 —— 模型写了语法干净的算术 —— 但 rule body 实际上没法从 FACTS 推出)。
`violations` 字段列出哪些层失败,以及失败原因。

### How this maps onto the rlib1.x JSON fields / 字段映射

For pipeline compatibility, the same wrapper script
[`tools/preprocessing/symbolic_cot_sample_generation.py`](../../tools/preprocessing/symbolic_cot_sample_generation.py)
still writes `symbolic_valid` / `grounding_score` / etc. when the
`cot_style` is `rlib2.0`. The bridge between rlib2.0's L1-L5 model and
the legacy schema:

为了兼容 pipeline,
[`tools/preprocessing/symbolic_cot_sample_generation.py`](../../tools/preprocessing/symbolic_cot_sample_generation.py)
在 `cot_style=rlib2.0` 下依然写 `symbolic_valid` / `grounding_score` 这些字段。
rlib2.0 的 L1-L5 模型与旧字段对应关系:

| Legacy field / 旧字段 | rlib2.0 mapping / rlib2.0 映射 |
|---|---|
| `symbolic_valid` | `True` iff L1 AND L2 AND L4 AND L5 all pass (the 4 structural / logical layers). L3 is treated as a separate "external grounding" signal — see `grounding_score`. <br/>L1 ∧ L2 ∧ L4 ∧ L5 全过则 True(4 个结构性 / 逻辑性层)。L3 单独作为"外部 grounding"信号 |
| `symbolic_violations` | Concatenation of every layer's failure messages. <br/>所有层失败消息的拼接 |
| `grounding_score` | `1.0 if L3 else 0.0` — binary because rlib2.0's L3 is **exact-or-fail per fact**, not a fraction. (Unlike rlib1.x's continuous fraction.) <br/>`1.0 if L3 通过 else 0.0` —— 二元值,因为 rlib2.0 的 L3 对每个 fact 都是**精确通过/失败**,不像 rlib1.x 那样取分数 |
| `cot_format` | `"symbolic_datalog"` |

> Note / 注意: rlib2.0 generation pipeline is **not yet wired into**
> `symbolic_cot_sample_generation.py` — the verifier exposes the same
> `SymbolicSchema`/`SymbolicParser`/`SymbolicValidator` API, but the
> Datalog¬-specific `verify(text, rlib_dir)` function is the canonical
> entry point. Run it standalone for now via
> [`Z3-based smoke tests`](../../tests/) or by importing `verifier.verify`
> directly. Wiring it into the sample-generation pipeline is the next
> follow-up.
>
> rlib2.0 的生成 pipeline **尚未接入** `symbolic_cot_sample_generation.py` ——
> verifier 暴露和 rlib1.x 同形状的 API,但 Datalog¬ 专属的
> `verify(text, rlib_dir)` 才是正经入口。目前手动测试请用
> [`Z3-based smoke tests`](../../tests/) 或直接 `from symdrive.rlib2_0.verifier
> import verify` 调用。把它接进 pipeline 是下一步 follow-up。

## 5. Files / 文件清单

```
rlib2_0/
├── README.md                ← this file
├── prompt.py                ← 5-stage Option C prompt with 4 new examples
├── verifier.py              ← 5-layer Z3-backed check; `verify()` is the main API
└── rlib/
    ├── entities.yaml        ← each attr has {type, unit} or {type, values} schema
    ├── facts.yaml           ← SOFT vocab (~30 canonical names + reference exprs)
    ├── operations.yaml      ← whitelisted arithmetic / set ops (e.g. min, max, count, Filter)
    └── actions.yaml         ← (lateral × longitudinal) — copied from rlib1.0
                              (no `rules/` subdir — fully free rule composition)
```

## 6. New dependencies / 新增依赖

`z3-solver` for L4 propositional check. Already in
[`requirements.txt`](../../requirements.txt). The rlib2.0 verifier picks
it up lazily — `import z3` only happens inside `_z3_check_rule()`, so the
rest of the verifier still works in environments without z3 installed
(L4 just always fails with a clear error).

`z3-solver` 用于 L4 的命题逻辑检查,已加进
[`requirements.txt`](../../requirements.txt)。verifier 在 `_z3_check_rule()`
里**延迟 import z3**,所以没装 z3 的环境里 verifier 还能跑(只是 L4 会
带着清晰错误信息失败)。

## 7. Limitations / 已知局限

- **PERCEPTION grounding still not externally verified** — same as
  rlib1.x. L3 checks "FACTS ⊨ derived from PERCEPTION", but doesn't check
  "PERCEPTION ⊨ matches the real-world scene". External grounding
  requires tool-grounded data generation (planned for a future rlib).
  <br/>**PERCEPTION 本身仍未外部验证** —— 同 rlib1.x。L3 检查 "FACTS ⊨ 从
  PERCEPTION 推出",但不检查 "PERCEPTION ⊨ 符合真实场景"。要实现外部
  grounding 需要 tool-grounded 数据生成(后续 rlib 版本计划中)
- **LLM-friendliness lower than rlib1.x** — the `:=` arithmetic syntax
  is less standard than predicate names like `LeadVehicleClose`. Expect
  higher syntactic-error rates from off-the-shelf VLMs without
  fine-tuning. Heavy in-context examples in `prompt.py` mitigate this.
  <br/>**对 LLM 友好度低于 rlib1.x** —— `:=` 算术语法不如 `LeadVehicleClose`
  这种命题名常见,直接 zero-shot 用现成 VLM 时语法错误率会更高。`prompt.py`
  里堆 in-context examples 来缓解
- **No symbolic_reasoning bucket in the train_mix data yet** — all
  existing symbolic CoT annotations on disk were generated by rlib1.0.
  Regenerating with rlib2.0 is a follow-up.
  <br/>**目前磁盘上的 symbolic CoT 数据全是 rlib1.0 生成的** —— 用 rlib2.0
  重新生成是下一步 follow-up
