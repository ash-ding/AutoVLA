# symdrive/

Self-contained, versioned symbolic Chain-of-Thought (CoT) designs for AutoVLA.

Each version subdirectory bundles a complete symbolic-CoT design: the RLIB
ontology, the prompt template the teacher VLM sees, and the verifier that scores
generated CoT. The shared generation pipeline at
[tools/preprocessing/symbolic_cot_sample_generation.py](../tools/preprocessing/symbolic_cot_sample_generation.py)
dispatches between versions based on a `cot_style` config field.

## Versions

| `cot_style` | Folder | Status | Summary |
|---|---|---|---|
| `rlib1.0` | [`rlib1_0/`](rlib1_0/) | Baseline (regression target) | Current implementation, moved here unchanged. Five-stage PL with qualitative attributes (Near/Far/Slow/Fast). `ego` state queried via `EgoQuery(...)` in OPERATIONS. |
| `rlib1.1` | [`rlib1_1/`](rlib1_1/) | Prototype | Identical to rlib1.0 except `OPERATIONS` is removed (4 stages: PERCEPTION → FACTS → RULES → ACTION) and `ego` is declared as an Ego entity in PERCEPTION. Tests whether the OPERATIONS layer adds value. |
| `rlib2.0` | [`rlib2_0/`](rlib2_0/) | Prototype | Datalog¬ + arithmetic. PERCEPTION carries real numerical values; `OPERATIONS` (optional) holds arithmetic / set derivations; `FACTS` use `:=` grounding expressions with `→ True/False` results; `RULES` are Boolean-only over FACT atoms; ACTION is a `(lateral, longitudinal)` tuple. Verifier runs a 5-layer check (L1 syntax → L2 references → L3 arithmetic eval → L4 Z3 propositional → L5 action match). |

## Folder layout (per version)

```
rlibX_Y/
├── __init__.py
├── rlib/                   ← ontology YAMLs (entities, facts, actions, ...)
│   ├── entities.yaml
│   ├── facts.yaml
│   ├── actions.yaml
│   ├── operations.yaml     (optional — rlib1.1 omits it)
│   ├── thresholds.yaml     (rlib1.x only)
│   └── rules/              (predefined Datalog/PL rules; rlib2.0 omits these)
├── prompt.py               ← exposes get_symbolic_cot_prompt, ego_state_to_qualitative,
│                              action_string_to_symbolic
└── verifier.py             ← exposes SymbolicSchema, SymbolicParser, SymbolicValidator,
                              ParseError, compute_symbolic_complexity
                              (rlib2.0 additionally exposes `verify(cot_text, rlib_dir)`)
```

Folder names use **underscores** (`rlib1_0`) because Python package names cannot
contain `.`. The dotted form (`rlib1.0`) lives only in user-facing `cot_style`
config strings.

## Selecting a version

```yaml
# In a dataset config YAML
cot_style: rlib2.0
rlib_dir:  ./symdrive/rlib2_0/rlib
```

Or override at the CLI:

```bash
python -m tools.preprocessing.symbolic_cot_sample_generation \
    --config dataset/symbolic-cot-gpt4o-mini-nuplan-mini-rlib2.0 \
    --cot-style rlib2.0 \
    --rlib_dir ./symdrive/rlib2_0/rlib \
    --output_dir /tmp/out_rlib2.0 --resume
```

## Adding a new version

1. Pick a name (e.g. `rlib2.1`) and create `symdrive/rlib2_1/` with an empty `__init__.py`.
2. Copy or author `rlib/`, `prompt.py`, `verifier.py`. Keep the same public
   names so the pipeline can dispatch (`SymbolicSchema`, `SymbolicParser`,
   `SymbolicValidator`, `ParseError`, `compute_symbolic_complexity`,
   `get_symbolic_cot_prompt`, `ego_state_to_qualitative`,
   `action_string_to_symbolic`).
3. Register the name in [`_pipeline/registry.py`](_pipeline/registry.py) by
   adding it to `_VALID_STYLES`.
4. Add at least one `config/dataset/symbolic-cot-*-rlib2.1.yaml` config.
5. Run a small end-to-end sample (3–5 scenes) to confirm dispatch works.

## Backwards compatibility

Two stub modules redirect the legacy import paths to `rlib1.0`:
- [models/symbolic_rules.py](../models/symbolic_rules.py)
- [dataset_utils/preprocessing/symbolic_cot_prompts.py](../dataset_utils/preprocessing/symbolic_cot_prompts.py)

These keep `pytest tests/test_symbolic_rules.py` and any other rlib1.0-era
caller working unchanged.
