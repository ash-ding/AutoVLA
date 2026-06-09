# Progress — Known Issues / Pits

## 2026-05-11 — Greedy decoding loops on small VL models for structured CoT

**Where**: `tools/preprocessing/symbolic_cot_sample_generation.py` + vLLM backend
([dataset_utils/preprocessing/vllm_cot_annotation_model.py](dataset_utils/preprocessing/vllm_cot_annotation_model.py)).
First hit while running Qwen3-VL-8B-Instruct on navmini for the
model-comparison study.

### Symptom
At `temperature=0` with only `max_tokens` / `temperature` in `SamplingParams`,
Qwen3-VL-8B emits a repetition loop inside the RLIB PERCEPTION block —
~50+ identical entities (e.g. `p_1 = Pedestrian {position: Right, distance:
Near, motion: Walking, location: Sidewalk}`, `p_2 = ...`, …) until it hits
`max_tokens=1500`. The parser then dies on the truncated last entity:

```
Parse error: Cannot parse entity: 'p_53 = Pedestrian {position: Right, distance: Near, motion'
```

Result: `symbolic_valid=False`, `grounding_score=0.0`, and ~40s/scene because
every sample saturates `max_tokens`.

### Root cause
Classic small-VL-model failure mode: greedy decode on a long structured
prompt falls into a repetition attractor. The vLLM annotation model only
threaded `max_tokens` and `temperature` into `SamplingParams` — no
`repetition_penalty`, no `top_p`, no `top_k`. Worked fine with the
GPT-4o-mini OpenAI backend (different code path, and proprietary models
handle greedy on this prompt OK), so the gap was invisible until the first
open-source vLLM run.

### Fix
1. Extended `vllm_cot_annotation_model.CoTAnnotationModel.__init__` to
   forward `top_p`, `top_k`, `repetition_penalty`, `frequency_penalty`,
   `presence_penalty`, `seed` from config to `SamplingParams` *if present*.
   Backwards compatible — existing configs that omit these keys keep the
   old greedy behavior unchanged.
2. In [config/dataset/symbolic-cot-qwen3-vl-8B-nuplan-mini.yaml](config/dataset/symbolic-cot-qwen3-vl-8B-nuplan-mini.yaml)
   set:
   ```yaml
   temperature: 0.1
   top_p: 0.9
   repetition_penalty: 1.05
   seed: 42
   ```
   Slight temperature + top_p + a light repetition penalty kills the
   attractor without making output creative.

### Implication for the model-comparison study
The comparison study uses a **uniform sampling block across all vLLM-backed
open-source VL models** for fairness. Reuse the same four lines above in any
new `symbolic-cot-<model>-nuplan-mini.yaml` config. OpenAI configs
(`symbolic-cot-gpt4o-mini*.yaml`) are unaffected — they go through a
different annotation model with their own request params.

### Also fixed in passing
`symbolic_cot_sample_generation.py` was calling
`get_dataset_tokens(val_dataset)` where `val_dataset` is a
`SymbolicPromptWrapper` that didn't expose `_scene_loader` or `scenes`,
raising `AttributeError` before any inference. Changed to
`get_dataset_tokens(base_dataset)` (both have identical length / index
alignment). Pre-existing latent bug — was hit unconditionally regardless of
`--sample-ids-json` or `--resume`.

---

## 2026-05-11 — Qwen3-VL-8B-Instruct baseline on navmini 1/6

First clean run after the sampling fix. **Selection**:
`partition_indices` with seed=42 + `--num_parts 6 --sample_num 1` → first 66
of 396 navmini scenes after a deterministic shuffle. Token list saved to
`/export/scratch_large/pouya/autovla_dataset/mini_symbolic/navmini_first_sixth_tokens.json`
for reuse across other models in the comparison.

| Category | Count | Notes |
|---|---|---|
| `valid` | 47/66 (71%) | Cleanly passes RLIB schema + grounding |
| `parse failures` | 10/66 (15%) | **Repetition loops still present** — all outputs ~4700+ chars, hit `max_tokens=1500` and got truncated. `repetition_penalty=1.05` weakened but did not eliminate the attractor — 100% → 15% |
| `parsed-but-invalid` | 9/66 (14%) | Schema violations: `Unknown attribute 'subtype'`, hallucinated facts like `EgoMovingFast` not in RLIB vocabulary |

Two failure classes:

- **15% loop residue**: could try `repetition_penalty=1.1` to push it further,
  but the loop rate is also a property of the 8B model — keeping it visible
  is useful signal for the cross-model comparison.
- **14% ontology violations**: the 8B model didn't memorize the closed
  vocabulary in `entities.yaml` / `facts.yaml`. Independent of sampling —
  reflects how well the model follows the schema instructions in the
  symbolic prompt. Expected to differ across models.

Both numbers should be treated as **baseline characteristics** of
Qwen3-VL-8B-Instruct under our sampling, not as bugs to fix. Other models
will surface their own failure profiles.

Throughput notes: ~12-15s/scene single-GPU L40S (TP=1), 17 min wall-clock
for the 66-scene run. Future runs with `--sample-ids-json <token_list>
--dp_size 4` should get ~3-4× speedup on this hardware.

---

## 2026-05-11 — Qwen3-VL-8B-Thinking: incapable of RLIB structured output

**Verdict**: Skip this model from the comparison study. The Thinking variant
**cannot produce structured output at all** in this task — not a tuning
problem, a model-capability problem.

### Setup
Same sampling block as Instruct (temp=0.1, top_p=0.9, rep_penalty=1.05).
Bumped `max_tokens=16384`, `max_model_len=32768` to give thinking room.
Smoke-tested on 1 scene.

### Symptom
1 scene → 6m47s wall-clock → 68,512-char output (~17K tokens, hit `max_tokens`
limit). **Zero RLIB section headers** in the output. The model never
emitted `</think>`, never transitioned to the final answer — burned the
entire budget thinking circularly.

Phrase counts in that one sample:
- `"However"`: 104×, `"Actually"`: 73×, `"But note"`: 58×
- `"in the same direction"`: 173×, `"in the same lane"`: 128×, `"behind"`: 469×
- `PERCEPTION:` / `OPERATIONS:` / `FACTS:` / `RULES:` / `ACTION:`: 0× each

Content was meta-cognitive runaway — the model kept second-guessing what it
saw in the back camera (which way the cars are moving, are they in the same
lane, is ego decelerating...), with no convergence.

### Why
Qwen3-VL-8B-Thinking's chat template **unconditionally** appends
`<|im_start|>assistant\n<think>\n` as the generation prompt — there's no
`enable_thinking=False` knob. The model is trained to think then answer.
On strict-schema generation tasks (output must hit PERCEPTION/OPS/FACTS/
RULES/ACTION exactly), the trained tendency to deliberate freely conflicts
with the structured format. At the 8B scale the model doesn't have the
capacity to both reason fully AND respect the schema; it picks reasoning.

### Was the strip code wrong?
No. `split_thinking` handles both Case A (full `<think>...</think>`) and
Case B (Qwen3's no-leading-tag convention, split on first `</think>`). The
strip would have worked IF the model ever closed thinking — but it didn't.
The split logic is correct and useful for future Thinking-variant runs that
do close their thinking trace. Keeping it.

### Implication for the comparison study
"Thinking 8B fails at structured output" is itself the finding — record it
and move on. Consumers of the symbolic-CoT data (SFT/GRPO downstream) need
clean RLIB output; a model that can't produce it is unusable in the
pipeline, full stop. No need to burn 2 hours validating on all 66 scenes
when 1 sample is this conclusive (0/5 headers, 17K-token max-out).

Smoke-test artifact kept at
`/export/scratch_large/pouya/autovla_dataset/mini_symbolic/qwen3-vl-8B-Thinking/0160a218dc9051bd.json`
as evidence.

---

## 2026-05-11 — Qwen3-VL-32B-Instruct baseline + 8B-vs-32B comparison

Same 66 navmini scenes via the reusable token list. TP=4 across the 4×L40S
(32B bf16 = 67 GB; ~17 GB per GPU after sharding, comfortable with KV cache).
Same sampling block as 8B — no per-model tuning, for fairness.

### Headline numbers

| Model | valid | parse_fail | parsed_invalid | grounding | wall-clock |
|---|---|---|---|---|---|
| 8B Instruct (DP=4 equiv: TP=1) | 47/66 (71%) | 10/66 (15%) | 9/66 (14%) | 0.926 | ~17 min |
| **32B Instruct (TP=4 DP=1)** | **49/66 (74%)** | **13/66 (20%)** | **4/66 (6%)** | **0.984** | **~22 min** |

### Counter-intuitive findings
1. **32B has MORE parse failures** than 8B (13 vs 10). Bigger model is *not*
   strictly more reliable at structured output under our sampling.
2. **32B has FEWER schema-invalid samples** (4 vs 9). When 32B does parse,
   it follows the closed RLIB vocabulary much better.
3. **Grounding score jumps from 0.926 → 0.984**. FACTS that 32B emits are
   almost always supported by entities — much more coherent reasoning.

### Disjoint failure profiles
- 37 scenes valid on both
- 10 scenes valid only on 8B (32B failed)
- 12 scenes valid only on 32B (8B failed)
- Of parse-failures: only **3 overlap** — 7 are 8B-only, 10 are 32B-only

The two models fail on *different scenes*. So 32B isn't a strict upgrade —
it's a different point in the bias-variance tradeoff. This is a useful
ensemble property if multiple-model annotation is ever desired downstream.

### Why 32B still loops
12 of the 13 32B parse-fails are the same pattern as 8B: ~50 sequential
`p_N = Pedestrian {position: Right, distance: Near, motion: Walking,
location: Sidewalk}` (or `v_N = Car {...}`) entities until truncation.
**`repetition_penalty=1.05` is insufficient even at 32B scale.** The
failure is driven by RLIB's repetitive-attribute entity-list template more
than by model capacity. Bigger model didn't help.

The 1 non-loop fail (`1fc1dd0dc3d157ae.json`) is a schema confusion: the
model wrote `Not EgoStopped = True` as a fact (using `Not` as a name prefix)
instead of using `NOT EgoStopped` inside a rule. Distinct failure mode.

### Throughput notes
- Per-scene rate ~18-22s (cold start ~30s) under TP=4
- Total ~22 min wall-clock for 66 scenes
- vs 8B single-GPU @ ~17 min wall-clock for the same 66
- TP=4 PCIe all-reduce overhead is the main cost — DP would be faster
  per-replica but 32B doesn't fit in 1 L40S so DP isn't available
- Useful comparison: **3× larger model, ~30% more wall-clock, all 4 GPUs
  occupied vs. 1**. So the throughput-per-GPU is ~10× lower for ~3-5%
  better validation rate.

### What this means for the comparison study so far

| Model | valid | grounding | cost/usefulness |
|---|---|---|---|
| 8B Instruct | 71% | 0.93 | best throughput-per-GPU, decent baseline |
| 8B Thinking | 0% (1 sample) | n/a | unusable — skip |
| 32B Instruct | 74% | 0.98 | best grounding, marginally better valid rate |
| GPT-4o-mini (upstream config) | TBD | TBD | external API reference |

If sampling were tuned per-model (rep_penalty=1.1 or higher), the
repetition rate might drop for both. But under our uniform sampling block,
**32B's main edge over 8B is grounding quality, not validation rate**.

### Options identified for next moves
1. Bump `repetition_penalty` to 1.1-1.15 and re-run both — would test if the
   shared loop pattern can be suppressed. Cost: breaks the cross-model
   sampling fairness contract.
2. Add GPT-4o-mini on the same 66-scene token list — proprietary-API
   reference point for the comparison.
3. Cross-generation: Qwen2.5-VL-7B + Qwen2.5-VL-72B-AWQ on the same 66 to
   see Qwen2.5 → Qwen3 jump per size class.
4. Skip ahead to downstream — use the existing baselines as SFT training
   data, see which model's CoT trains the best policy. The "useful CoT"
   signal may differ from the "valid CoT" signal.

### Server-VRAM ceiling for the comparison
- Qwen3-VL-235B-A22B-Instruct (bf16, 471 GB) and -FP8 (238 GB) both
  **exceed** the 4× L40S = 180 GB total VRAM. The bf16 needs ~2.6×, the FP8
  ~1.3× more memory than we have. MoE CPU-offload isn't practical because
  routing demands all 128 experts be available per decode step.
- AWQ-INT4 of 235B-A22B (~118 GB) would fit, but no public quant exists yet.
- **Qwen3-VL-32B-Instruct is the largest Qwen3-VL dense model that runs on
  this server.** Going further requires cloud or community INT4 releases.

---

## 2026-05-12 — Overnight screening: 10 models × 66-scene navmini subset

Plan: [docs/screen_model.md](docs/screen_model.md). Same token list as
prior runs (`navmini_first_sixth_tokens.json`). Two lanes in parallel:
GPU (Qwen 72B-AWQ → Qwen3-VL-32B-Thinking sequential) + API (8 GPT
models concurrent). GPT models use API defaults (not low/minimal effort).

### Per-model results

Filled in as each run completes overnight.

| Model | valid | parse_fail | parsed_invalid | grounding (avg) | wall | failure-mode notes |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct (prior) | 47/66 (71%) | 10 (15%) | 9 (14%) | 0.93 | 17m | repetition loops + `subtype`/EgoMovingFast |
| Qwen3-VL-32B-Instruct (prior) | 49/66 (74%) | 13 (20%) | 4 (6%) | 0.98 | 22m | same loops at scale, better grounding |
| **gpt-4o-mini** | **58/66 (87.9%)** | 2 (3.0%) | 6 (9.1%) | 0.89 | 2m54s | 5× `Unknown attribute 'type' for TrafficSign`, 1× invalid Vehicle.position `Rear`, 2× malformed fact lines |
| **gpt-4o** | **60/66 (90.9%)** | 3 (4.5%) | 3 (4.5%) | 0.92 | 2m10s | NEW modes vs Qwen: 1× `Unknown entity 'SUV'`, **1× safety refusal** ("I'm sorry, I can't assist with that request"), 1× markdown decoration (`**PERCEPTION:**` + ``` blocks) — zero loops |
| **gpt-4.1-mini** | **57/66 (86.4%)** | 3 (4.5%) | 6 (9.1%) | 0.93 | 2m21s | 3× `Unknown entity type`; 3× wrong RoadFeature.position values (Right/FrontRight); 2× rule references undeclared fact; 1× TrafficSign.distance out-of-vocab |
| **o4-mini** (Tier 3 reasoning) | **64/66 (97.0%)** | 1 (1.5%) | 1 (1.5%) | 0.97 | 3m22s | 1× truncated mid-entity (cut short, not a loop), 1× hallucinated fact `CrosswalkPresent` |
| **o3** (Tier 3 reasoning) | **64/66 (97.0%)** | **0 (0%)** | 2 (3.0%) | **1.000** | 3m46s | Tied highest valid + **best grounding (perfect)** + **zero parse failures**. 2× attribute-value violations (TrafficLight.color `Off`, TrafficSign.position `Behind`) |
| **gpt-5-mini** (Tier 4 reasoning) | **62/66 (93.9%)** | 1 (1.5%) | 3 (4.5%) | 0.99 | 4m29s | 1× `Unknown entity type`, 2× wrong RoadFeature.position `Right`, 1× hallucinated fact `ParkingArea` |
| **gpt-5** (Tier 4 reasoning) | **64/66 (97.0%)** | 2 (3.0%) | **0 (0%)** | 0.97 | 5m54s | Tied highest valid; zero schema violations. 2× parse failures (entity-type words); whatever it emits, it emits cleanly within RLIB vocab |
| **gpt-4.1** | 45/66 (68.2%) | **15 (22.7%)** | 6 (9.1%) | 0.76 | 11m24s | ⚠ **Surprising regression** — worse than its own mini sibling. 9× malformed fact lines, 5× `Unknown entity type`. Output ~2× longer (avg 1463 ch vs gpt-4o's 784), suggesting temp=1 default makes it verbose-but-noisy on structured tasks |
| **Qwen2.5-VL-72B-Instruct-AWQ** | **66/66 (100%)** | **0 (0%)** | **0 (0%)** | 0.96 | 13m39s | ⚡ **Surprise leader.** Zero schema violations across all 66 scenes. Older-gen (Qwen2.5) at INT4 quant beats every newer Qwen3 variant *and* every GPT model on schema conformance. Avg output 971 ch — concise and clean. Content quality not perfect: one sampled scene had StopSign→TurnLeft+Accel rule mismatch (semantic, not schema), but parser/validator can't catch reasoning errors |

32B-Thinking launched on the same 66; ~5h ETA. Will be the last to finish.

### Full navmini (396 scenes) — GPT models expanded

After 66-scene screening, GPT models were re-run on **all 396 navmini scenes**
for statistical power (Qwen models stay at 66 — local GPU much slower).
Same output dirs with `--resume` so the 66 already-done are not reprocessed.

| Model | valid | parse_fail | parsed_invalid | grounding | wall (full run) | vs 66-scene |
|---|---|---|---|---|---|---|
| **gpt-4o-mini** | 356/396 (89.9%) | 11 (2.8%) | 29 (7.3%) | 0.90 | ~13m | +2pp valid (87.9% → 89.9%); 15× `Unknown attribute 'type' for TrafficSign` dominates schema errors |
| **gpt-4.1-mini** | 330/396 (83.3%) | 31 (7.8%) | 35 (8.8%) | 0.89 | ~13m | -3pp valid (86.4% → 83.3%); 29 of 31 parse_fails are `Unknown entity type` hallucinations |
| **o4-mini** | 385/396 (97.2%) | 9 (2.3%) | 2 (0.5%) | 0.97 | ~14m | +0.2pp valid — extremely stable. 9× `Cannot parse entity` (likely truncations); only 2 schema violations |
| **o3** | 375/396 (94.7%) | 19 (4.8%) | 2 (0.5%) | 0.95 | ~15m | -2.3pp valid (97.0% → 94.7%); ALL 19 parse_fails are `Cannot parse fact` (malformed lines in FACTS section, not loops) |
| **gpt-4o** | 359/396 (90.7%) | 27 (6.8%) | 10 (2.5%) | 0.88 | ~17m | -0.2pp valid — flat (virtually identical to 66-run). 12× missing-headers (markdown decoration or safety refusals), 11× `Unknown entity type` |
| **gpt-5-mini** | 378/396 (95.5%) | 11 (2.8%) | 7 (1.8%) | 0.97 | ~19m | +1.6pp valid (93.9% → 95.5%); varied parse_fails (5 fact, 4 entity, 1 op); 4× wrong RoadFeature.position values dominates schema-invalid |
| **gpt-4.1** | 285/396 (72.0%) | 85 (21.5%) | 26 (6.6%) | 0.78 | ~19m | +3.8pp valid (68.2% → 72.0%); **still weakest GPT model** at 396 scale. 47 fact-parse + 31 unknown-entity-type. Confirms gpt-4.1's verbose/creative tendency under default temp=1 |
| **gpt-5** | **386/396 (97.5%)** | 6 (1.5%) | 4 (1.0%) | 0.98 | ~26m | +0.5pp valid (97.0% → 97.5%); **top GPT at full scale**. 5× entity-parse + 1× unknown type; 2× empty outputs (likely safety refusal returning empty); 2× `Off` TrafficLight.color schema mismatch |

All 8 GPT 396-scene expansions complete.

### Qwen Instruct models — full 396 with updated sampling

Re-run with user-provided official Qwen-recommended sampling (figure 1:
temp=0.7, top_p=0.8, top_k=20, rep_pen=1.0, **presence_penalty=1.5**,
max_tokens=16384). Qwen2.5-72B-AWQ kept its original fairness sampling
block (temp=0.1 / top_p=0.9 / rep_pen=1.05 / max_tokens=1500).

| Model | valid | parse_fail | parsed_invalid | grounding | wall | vs prior |
|---|---|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct-v2** | 313/396 (79.0%) | 29 (7.3%) | 54 (13.6%) | 0.87 | 16m | +8pp vs prior 8B-Inst (71%). All 29 parse_fails are `Unknown entity type` (hallucinated subtypes); 36× `subtype` for TrafficSign/RoadFeature dominates schema-invalid |
| **Qwen3-VL-32B-Instruct-v2** | 296/396 (74.7%) | 57 (14.4%) | 43 (10.9%) | 0.84 | 118m | Essentially flat vs prior 32B-Inst (74%). New params didn't help. 39 fact-parse + 18 unknown-entity-type. Avg pf len 2649 ch suggests verbose-but-malformed outputs with the bigger 16K token budget |
| **Qwen2.5-VL-72B-Instruct-AWQ** (expanded to 396) | **384/396 (97.0%)** | 6 (1.5%) | 6 (1.5%) | 0.94 | 53m (66→396 leg) | -3pp from 100% at 66 (noise of small sample); **still the best Qwen and matches reasoning-tier GPT (o4-mini 97.2%, just shy of gpt-5 97.5%)**. Older-gen INT4 wins over newer Qwen3 variants by ~22-25pp on schema conformance |

Phase 4 (Qwen3-VL-32B-Thinking-v2 on 66, with figure-2 sampling) running.
Phase 5 (Qwen3-VL-8B-Thinking-v2 on 66) pending.

### Qwen Thinking models — 66 scenes with figure-2 sampling

User-provided figure-2 sampling (temp=1.0, top_p=0.95, top_k=20, rep_pen=1.0,
pres_pen=0.0, max_tokens=40960). Both ran on the same 66-token JSON.

| Model | valid | parse_fail | parsed_invalid | grounding | wall | notes |
|---|---|---|---|---|---|---|
| **Qwen3-VL-32B-Thinking-v2** | 60/66 (90.9%) | 1 (1.5%) | 5 (7.6%) | **0.985** | ~7 min/scene (TP=4) | Avg cot_thinking = ~14K tokens (max 22K). 2× `BehindLeft` invented compound position, 1× `RightLane` for Cyclist.lane (vocab mixup), 1× `VeryNear` for TrafficSign, 1× `False` for TrafficLight.color, 1× `(none)` empty-perception parser miss |
| **Qwen3-VL-8B-Thinking-v2** | 46/66 (69.7%) | 9 (13.6%) | 11 (16.7%) | 0.811 | ~1.4 min/scene wall (DP=4 ⇒ ~5.7 min/replica) | ⚡ **Now works** — prior run at temp=0.1 was 0/1 catastrophic meta-cog runaway. New params unlock structured output, but weakest of all models tested |

The same prompt-RLIB-vocab confusions that hit 32B-Thinking show up amplified
in 8B-Thinking (BehindLeft/BehindRight, hallucinated facts like
`EgoAccelerating`/`EgoMovingFast`, plus 6× `Unknown entity type`).

### Final cross-model leaderboard (sorted by valid ratio)

| Rank | Model | Sample | Valid | Grounding | Cost-effectiveness notes |
|---|---|---|---|---|---|
| 1 | **gpt-5** | 396 | **97.5%** | 0.980 | $268 / 10k, $4,949 / 185k |
| 2 | **o4-mini** | 396 | **97.2%** | 0.970 | $136 / 10k, $2,523 / 185k — **best $/quality ratio at reasoning tier** |
| 3 | **Qwen2.5-VL-72B-Instruct-AWQ** | 396 | **97.0%** | 0.935 | Local, no $/scene; 10s/scene on 4× L40S TP=4 — **best local model overall** |
| 4 | **gpt-5-mini** | 396 | 95.5% | 0.968 | $54 / 10k, $990 / 185k — cheapest top-quality |
| 5 | **o3** | 396 | 94.7% | 0.951 | $1,240 / 10k, **$22,940 / 185k** — quality plateaus, cost balloons |
| 6 | **Qwen3-VL-32B-Thinking-v2** | 66 | 90.9% | **0.985** | Local but ~400 s/scene (~40× slower than 72B-AWQ); highest grounding overall |
| 7 | gpt-4o | 396 | 90.7% | 0.881 | $110 / 10k, $2,033 / 185k; 12× safety refusals & markdown |
| 8 | gpt-4o-mini | 396 | 89.9% | 0.901 | $7 / 10k, $122 / 185k — cheapest passable option |
| 9 | gpt-4.1-mini | 396 | 83.3% | 0.891 | $18 / 10k, $326 / 185k; subtype + RoadFeature vocab errors |
| 10 | Qwen3-VL-8B-Instruct-v2 | 396 | 79.0% | 0.872 | New params helped (+8pp vs prior), still mid-pack |
| 11 | Qwen3-VL-32B-Instruct-v2 | 396 | 74.7% | 0.841 | New params did not help; 16K max_tokens budget enables verbose-but-broken outputs |
| 12 | gpt-4.1 | 396 | 72.0% | 0.777 | $88 / 10k, $1,626 / 185k; ⚠ **regression** — worse than its own mini |
| 13 | Qwen3-VL-8B-Thinking-v2 | 66 | 69.7% | 0.811 | Works now (was 0% with old params); weakest |
| 14 | Qwen3-VL-8B-Thinking (prior, temp=0.1) | 1 (smoke) | 0% | — | Catastrophic meta-cog runaway, never closed `</think>` |

### Cross-cutting patterns

1. **The "empty PERCEPTION" parser hole** is the single most common failure
   mode across reasoning models (gpt-5, o4-mini, gpt-5-mini, 32B-Thinking
   all wrote `(none)`, `(no traffic light)`, or `# No vehicles ...` as
   PERCEPTION content when scenes were sparse). A 2-line parser fix (strip
   `#`-comments and `(...)` prose lines) would push o4-mini from 97.2% →
   ~99.5% and gpt-5 from 97.5% → ~98.8%.

2. **RoadFeature.position vocab confusion** affects gpt-4.1, gpt-4.1-mini,
   gpt-5-mini — models default-extend Vehicle's position vocab
   (Right/Left/FrontRight) to RoadFeature, which only allows Ahead/Current/
   Behind.

3. **TrafficSign attribute hallucination** — gpt-4o-mini's biggest weakness
   (15× `Unknown attribute 'type'`). Model wants to add a redundant `type`
   field on top of the subtype name.

4. **Hallucinated entity subtypes** (SUV/Sedan/Pickup) dominate gpt-4.1
   family failures — they invent subtypes outside `entities.yaml`. Reasoning
   models (o3, o4-mini, gpt-5, 32B-Thinking) almost never do this.

5. **`BehindLeft`/`BehindRight` invented compound positions** — Qwen3-VL
   Thinking variants infer compound positions from front-side pattern. RLIB
   schema is asymmetric (Front has 3 subdivisions, Behind only 1) — this is
   a schema-design gap, not a model bug.

6. **Catastrophic loop / runaway** is uniquely a **small-Thinking-model at
   low-temperature** failure (8B-Thinking at temp=0.1). 8B-Instruct, 32B
   variants, and same model at higher temperature all avoid this.

7. **The Qwen2.5 → Qwen3 generation jump is not monotonic** on this task:
   Qwen2.5-VL-72B-AWQ (97.0%) > Qwen3-VL-32B-Instruct (74.7%) ≈ 8B-Instruct
   (79.0%). The older generation's INT4 quantization + larger base might
   be the key — newer Qwen3 variants don't beat it.

### What to do next

- **For production CoT annotation**: pick from top 3 (gpt-5 / o4-mini /
  Qwen2.5-VL-72B-AWQ) based on cost vs. local-compute tradeoff. **o4-mini
  at $136 / 10k samples is the sweet spot.**
- **For research comparison**: keep all results as published baseline.
  Future work should investigate the parser fix (would re-rank top 3).
- **For 32B-Thinking**: not worth the 40× compute cost vs 72B-AWQ for ~6pp
  worse valid rate — unless its 0.985 grounding score is uniquely valuable
  for downstream tasks (e.g., curating tiny gold-standard subsets).

---

## 2026-05-11 — Qwen3-VL-32B-Thinking: works at quality, too slow for batch

**Verdict**: Capable but uneconomical — skipped full 66-scene run.

### Smoke test (1 scene)
TP=4, max_tokens=16384, same sampling block as Instruct variants. Selected
`0160a218dc9051bd` (same scene as 8B-Thinking smoke test for direct comparison).

| Metric | Value |
|---|---|
| Wall-clock | 4 min 27 sec |
| `valid` | True |
| `grounding_score` | **1.000** (highest of any model so far) |
| `cot_thinking` length | 32,458 chars (~8000 tokens) |
| `cot_output` length | 955 chars — clean RLIB structure |
| violations | none |

The 955-char cleaned output has 4 entities, 6 operations, 9 facts, 1 rule,
all schema-valid. The 32K-char thinking trace shows the model reasoning
step by step, self-correcting, and even pushing back on the hint
("Do NOT let the hint influence what you perceive...") — produces something
that *resembles* careful analyst behavior rather than the meta-cognitive
runaway 8B-Thinking exhibited.

### Why 32B-Thinking works where 8B-Thinking failed
- 8B had insufficient capacity to *both* reason fully AND respect the
  schema constraints — picked reasoning, never closed `</think>`.
- 32B reasons more efficiently (8K tokens vs 8B's >17K) and converges to
  the structured answer.
- This is a **capacity threshold** for the Thinking-then-Answer pattern on
  this task: somewhere between 8B and 32B is the minimum size where the
  model can complete both halves of the chat template.

### Why we skipped full 66-scene run
- **~5 hours wall-clock** projected (4m27s × 66 ≈ 4.9 h)
- vs 22 min for 32B-Instruct (13× slower)
- Locks all 4 GPUs for ~5 h
- TP=4 only — 32B doesn't fit on 1 L40S so DP is unavailable
- Sample-1 grounding=1.000 is encouraging but **n=1 isn't statistically
  meaningful**. Could regress on harder scenes (multi-pedestrian, weird
  intersection geometry).

### What we kept
- Smoke-test artifact:
  `/export/scratch_large/pouya/autovla_dataset/mini_symbolic/qwen3-vl-32B-Thinking/0160a218dc9051bd.json`
- Config:
  `config/dataset/symbolic-cot-qwen3-vl-32B-thinking-nuplan-mini.yaml`

If we later want to use 32B-Thinking as the gold-standard annotator
(e.g., for a small high-quality subset to bootstrap distillation), the
infrastructure is ready — just remove the `--num_parts 66 --sample_num 1`
shard knobs to run all 66.
