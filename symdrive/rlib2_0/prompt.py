"""
Symbolic Chain-of-Thought (CoT) prompts for driving scenario reasoning — rlib2.0.

Generates structured 5-stage prompts following the Option C surface syntax:

    [PERCEPTION]   numerical entity attributes (incl. ego)
    [OPERATIONS]   optional: arithmetic / set derivations  (name := expr = result)
    [FACTS]        boolean atoms via comparisons           (name := expr → True/False)
    [RULES]        single rule: atoms ∧ ¬atoms → (LateralAction, LongitudinalAction)
    [ACTION]       LateralAction, LongitudinalAction       (literal match with RULES head)

rlib2.0 difference vs rlib1.x:
  - PERCEPTION carries real numerical values, not qualitative buckets.
  - FACTS bodies MUST contain an arithmetic / boolean expression that grounds
    every fact in PERCEPTION or OPERATIONS — no bare `name = True` assertions.
  - RULES bodies are Boolean-only (no inline arithmetic) — they reference FACTS.
  - Single rule per scene (the one that fires); no multi-rule + Select.
  - facts.yaml is a SOFT vocabulary — the LLM may use canonical names or invent
    new ones, but every fact MUST have a `:=` right side referencing PERCEPTION
    or OPERATIONS.

API mirrors rlib1.x for drop-in pipeline compatibility:
  - action_string_to_symbolic(...) — maps GT action string to (lateral, longitudinal)
  - ego_state_to_qualitative(...)  — passes through raw numerics (no bucketize)
  - get_symbolic_cot_prompt(...)   — builds the full prompt dict {type:"text", text:...}
"""

from __future__ import annotations

import math
from pathlib import Path
from functools import lru_cache

import yaml


# ---------------------------------------------------------------------------
# Action string → symbolic mapping (identical to rlib1.x for compat)
# ---------------------------------------------------------------------------

_LATERAL_MAP = {
    "move forward": "KeepLane",
    "turn left": "TurnLeft",
    "change lane to left": "ChangeLaneLeft",
    "turn right": "TurnRight",
    "change lane to right": "ChangeLaneRight",
}

_LONGITUDINAL_MAP = {
    "stop": "Stop",
    "a deceleration to zero": "DecelerationToZero",
    "a constant speed": "ConstantSpeed",
    "a quick deceleration": "QuickDeceleration",
    "a deceleration": "Deceleration",
    "a quick acceleration": "QuickAcceleration",
    "an acceleration": "Acceleration",
}


def action_string_to_symbolic(action_str: str) -> tuple[str, str]:
    """Map free-form GT action string to (lateral, longitudinal) symbolic pair.

    Examples:
        "move forward with a deceleration" -> ("KeepLane", "Deceleration")
        "turn left with an acceleration"   -> ("TurnLeft", "Acceleration")
        "STOP"                             -> ("KeepLane", "Stop")
    """
    action_str = action_str.strip()
    if not action_str or action_str.upper() == "STOP":
        return ("KeepLane", "Stop")

    lateral = "KeepLane"
    longitudinal = "ConstantSpeed"

    lower = action_str.lower()
    for key, val in _LATERAL_MAP.items():
        if lower.startswith(key):
            lateral = val
            break
    for key, val in _LONGITUDINAL_MAP.items():
        if key in lower:
            longitudinal = val
            break

    return (lateral, longitudinal)


# ---------------------------------------------------------------------------
# Ego state → numerical payload (rlib2.0 keeps raw numerics, no bucketize)
# ---------------------------------------------------------------------------

def ego_state_to_qualitative(
    velocity: list[float],
    acceleration: list[float],
    instruction: str,
    rlib_dir: str | Path,  # accepted for API parity; unused here
) -> dict:
    """rlib2.0 returns raw numerical ego state in a dict shaped like rlib1.x.

    The pipeline (SymbolicPromptWrapper) reads {label, raw, unit} from each
    sub-dict; in rlib2.0 we set label = raw (formatted) so the same downstream
    formatting code works.
    """
    speed_mag = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    ax = float(acceleration[0])
    inst_lower = instruction.strip().lower()
    if "left" in inst_lower:
        inst = "TurnLeft"
    elif "right" in inst_lower:
        inst = "TurnRight"
    else:
        inst = "KeepForward"

    return {
        "speed":        {"label": f"{speed_mag:.2f}", "raw": speed_mag, "unit": "m/s"},
        "acceleration": {"label": f"{ax:.2f}",        "raw": ax,        "unit": "m/s^2"},
        "instruction":  inst,
    }


# ---------------------------------------------------------------------------
# RLIB schema → prompt text sections
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_rlib_prompt_sections(rlib_dir: str) -> dict[str, str]:
    """Load rlib2.0 YAMLs and format as prompt text sections."""
    rlib = Path(rlib_dir)

    with open(rlib / "entities.yaml") as f:
        entities = yaml.safe_load(f)
    with open(rlib / "facts.yaml") as f:
        facts_file = yaml.safe_load(f)
    with open(rlib / "actions.yaml") as f:
        actions = yaml.safe_load(f)
    with open(rlib / "operations.yaml") as f:
        ops_file = yaml.safe_load(f)

    # --- Entity type schema (rlib2.0 attrs have type tags) ---
    entity_lines: list[str] = []
    for base_type, cfg in entities.items():
        subtypes = cfg.get("subtypes", [])
        if subtypes:
            entity_lines.append(f"- {base_type} [use one of: {', '.join(subtypes)}]")
        else:
            entity_lines.append(f"- {base_type}")
        for attr_name, attr_spec in cfg.get("attributes", {}).items():
            atype = attr_spec.get("type")
            if atype == "number":
                unit = attr_spec.get("unit", "")
                desc = attr_spec.get("description", "")
                bits = [f"number {unit}".strip()]
                if desc:
                    bits.append(desc)
                entity_lines.append(f"    {attr_name}: {' — '.join(bits)}")
            elif atype == "enum":
                vals = attr_spec.get("values", [])
                entity_lines.append(f"    {attr_name}: enum {{{', '.join(str(v) for v in vals)}}}")
            else:
                entity_lines.append(f"    {attr_name}: {attr_spec}")
    entity_types_text = (
        "ENTITY DECLARATION RULES:\n"
        "1. In PERCEPTION, declare each detected entity as:\n"
        "     <id> = <Type> {attr: value, attr: value, ...}\n"
        "   Use specific subtypes as the class name (Car/Truck/Pedestrian/StopSign/...).\n"
        "2. Numeric attributes (dist, vel, lat_offset, dist_to_stopline, ...) MUST be\n"
        "   real numbers with appropriate units, NOT qualitative labels like 'Near'.\n"
        "   RIGHT: `v_1 = Car {dist: 8.3, vel: 2.1, lane: ego, lat_offset: 0.2, signal: none}`\n"
        "   WRONG: `v_1 = Car {distance: Near, motion: Slow}`\n"
        "3. Enum attributes must use one of the listed string values exactly.\n"
        "4. Ego state MUST be declared as `ego = Ego {speed: <m/s>, accel: <m/s^2>, "
        "instruction: KeepForward|TurnLeft|TurnRight}`.\n"
        "5. ID prefixes (convention): v_ for Vehicle, p_ for Pedestrian, c_ for Cyclist,\n"
        "   tl_ for TrafficLight, ts_ for TrafficSign, r_ for RoadFeature. The Ego id is `ego`.\n"
        + "\n".join(entity_lines)
    )

    # --- OPERATIONS grammar ---
    op_lines = ["OPERATIONS section is OPTIONAL — use only when a quantity is reused or non-trivial.",
                "Allowed arithmetic ops: " + ", ".join(ops_file.get("allowed_operators", {}).get("arithmetic", [])),
                "Allowed comparison ops: " + ", ".join(ops_file.get("allowed_operators", {}).get("comparison", [])),
                "Allowed functions:"]
    for fn, spec in ops_file.get("allowed_functions", {}).items():
        op_lines.append(f"  - {fn}(...): {spec.get('description', '')}")
    op_lines.append("")
    op_lines.append("Line form:")
    op_lines.append("  name := <expression>    = <numeric_or_set_result>")
    operations_text = "\n".join(op_lines)

    # --- FACTS soft vocabulary ---
    fact_lines: list[str] = []
    for entry in facts_file.get("vocabulary", []):
        name = entry["name"]
        desc = entry.get("description", "")
        typ = entry.get("typical_expression", "")
        thresh = entry.get("common_thresholds", {})
        bits = [f"- {name}: {desc}"]
        if typ:
            t_str = f"    typical: {name} := {typ}"
            if thresh:
                t_str += f"   (e.g. {', '.join(f'{k}={v}' for k, v in thresh.items())})"
            bits.append(t_str)
        fact_lines.append("\n".join(bits))
    facts_text = "\n".join(fact_lines)

    # --- ACTIONS ---
    lateral = actions.get("lateral", [])
    longitudinal = actions.get("longitudinal", [])
    actions_text = (
        f"Lateral (choose exactly one): [{', '.join(lateral)}]\n"
        f"Longitudinal (choose exactly one): [{', '.join(longitudinal)}]"
    )

    return {
        "entity_types": entity_types_text,
        "operations":   operations_text,
        "facts":        facts_text,
        "actions":      actions_text,
    }


# ---------------------------------------------------------------------------
# In-context examples (4, mirror rlib1.x scenarios)
# ---------------------------------------------------------------------------

_EXAMPLE_1 = """\
[PERCEPTION]
  tl_1 = TrafficLight {color: red, dist_to_stopline: 14.5, applies_to: ego_lane}
  v_1  = Car {dist: 8.3, lat_offset: 0.2, vel: 0.0, lane: ego, signal: brake_lights}
  r_1  = Intersection {dist: 18.0, width: 12.0}
  ego  = Ego {speed: 1.8, accel: -1.20, instruction: KeepForward}

[OPERATIONS]
  required_decel_for_stop := ego.speed ** 2 / (2 * tl_1.dist_to_stopline)   = 0.11

[FACTS]
  red_imminent       := tl_1.color == "red" ∧ tl_1.dist_to_stopline < 30           → True
  can_stop_smoothly  := required_decel_for_stop < 3.0                              → True
  lead_present       := v_1.lane == "ego"                                          → True
  lead_stopped       := v_1.vel < 0.5                                              → True
  ego_slow           := ego.speed < 3.0                                            → True

[RULES]
  R: red_imminent ∧ can_stop_smoothly ∧ ¬ego_stopped → (KeepLane, DecelerationToZero)

[ACTION]
  KeepLane, DecelerationToZero"""

_EXAMPLE_2 = """\
[PERCEPTION]
  tl_1 = TrafficLight {color: green, dist_to_stopline: 22.0, applies_to: ego_lane}
  p_1  = Pedestrian {dist: 30.0, lat_offset: 3.5, vel: 1.2, on_crosswalk: false, location: sidewalk, motion: walking}
  ego  = Ego {speed: 6.5, accel: 0.10, instruction: KeepForward}

[FACTS]
  green_clear              := tl_1.color == "green"                                 → True
  instruction_keep_forward := ego.instruction == "KeepForward"                      → True
  ped_in_path              := p_1.on_crosswalk == "true" ∧ abs(p_1.lat_offset) < 1.5 → False

[RULES]
  R: green_clear ∧ instruction_keep_forward ∧ ¬ped_in_path → (KeepLane, ConstantSpeed)

[ACTION]
  KeepLane, ConstantSpeed"""

_EXAMPLE_3 = """\
[PERCEPTION]
  tl_1 = TrafficLight {color: yellow, dist_to_stopline: 18.0, applies_to: ego_lane}
  v_1  = Car {dist: 12.0, lat_offset: 0.1, vel: 5.0, lane: ego, signal: brake_lights}
  r_1  = Intersection {dist: 22.0, width: 12.0}
  ego  = Ego {speed: 7.2, accel: -0.80, instruction: KeepForward}

[OPERATIONS]
  required_decel_for_stop := ego.speed ** 2 / (2 * tl_1.dist_to_stopline)   = 1.44

[FACTS]
  yellow_committed_to_stop := tl_1.color == "yellow" ∧ required_decel_for_stop < 3.0   → True
  approaching_intersection := r_1.dist < 50                                            → True
  lead_braking             := v_1.signal == "brake_lights"                             → True

[RULES]
  R: yellow_committed_to_stop ∧ approaching_intersection → (KeepLane, Deceleration)

[ACTION]
  KeepLane, Deceleration"""

_EXAMPLE_4 = """\
[PERCEPTION]
  v_1  = Car {dist: 6.5, lat_offset: 0.0, vel: 3.0, lane: ego, signal: none}
  v_2  = Car {dist: -7.0, lat_offset: 0.1, vel: 6.0, lane: ego, signal: none}
  ego  = Ego {speed: 5.4, accel: -0.20, instruction: KeepForward}

[OPERATIONS]
  rel_vel_lead := ego.speed - v_1.vel                          = 2.40
  TTC_lead     := v_1.dist / max(0.1, rel_vel_lead)            = 2.71

[FACTS]
  lead_present  := v_1.lane == "ego"                           → True
  lead_close    := v_1.dist < 8.0                              → True
  lead_too_close := TTC_lead < 2.0                             → False
  rear_close    := v_2.dist > -8 ∧ v_2.dist < 0                → True

[RULES]
  R: lead_close ∧ ¬lead_too_close ∧ ¬ego_stopped → (KeepLane, Deceleration)

[ACTION]
  KeepLane, Deceleration"""


# ---------------------------------------------------------------------------
# Main prompt function
# ---------------------------------------------------------------------------

def get_symbolic_cot_prompt(
    rlib_dir: str,
    fut_ego_action: str,
    ego_speed: dict,
    ego_acceleration: dict,
    ego_instruction: str,
    nl_cot_reference: str | None = None,
    use_predefined_rules: bool = False,  # rlib2.0: no predefined rules; flag kept for API parity
) -> dict:
    """Build the rlib2.0 (Datalog¬ + arithmetic) symbolic CoT prompt.

    Args:
        rlib_dir: Path to rlib2.0 ontology directory.
        fut_ego_action: Free-form GT action string (e.g. "move forward with a deceleration").
        ego_speed, ego_acceleration: {"label": str, "raw": float, "unit": str} —
            in rlib2.0 `label` already contains the formatted raw value.
        ego_instruction: "KeepForward" / "TurnLeft" / "TurnRight".
        nl_cot_reference: Optional NL CoT trace to use as warm-start reference.
        use_predefined_rules: API parity with rlib1.x; rlib2.0 always free-form.

    Returns:
        {"type": "text", "text": prompt_text}
    """
    sections = _load_rlib_prompt_sections(str(rlib_dir))
    gt_lateral, gt_longitudinal = action_string_to_symbolic(fut_ego_action)

    prompt_text = (
        "Based on the above camera images and ego vehicle state, produce a structured "
        "symbolic reasoning chain in the 5-section Datalog+arithmetic format below.\n\n"

        "=== ENTITY TYPES (rlib2.0 numerical schema) ===\n"
        f"{sections['entity_types']}\n\n"

        "=== OPERATIONS GRAMMAR (optional section) ===\n"
        f"{sections['operations']}\n\n"

        "=== FACTS VOCABULARY (soft — use canonical names where they fit, invent new ones for novel concepts) ===\n"
        "Every FACT line MUST have a `:=` right side referencing PERCEPTION attributes or "
        "earlier OPERATIONS results. Bare assertions like `xxx := True` are FORBIDDEN.\n"
        f"{sections['facts']}\n\n"

        "=== ACTIONS ===\n"
        f"{sections['actions']}\n\n"

        "=== EGO VEHICLE STATE (declare in PERCEPTION as `ego = Ego {...}`) ===\n"
        f"speed       = {ego_speed['raw']:.2f}  {ego_speed['unit']}\n"
        f"accel       = {ego_acceleration['raw']:.2f}  {ego_acceleration['unit']}\n"
        f"instruction = {ego_instruction}\n\n"

        + (
            "=== REFERENCE: NATURAL LANGUAGE REASONING ===\n"
            "Below is a free-form reasoning trace for the same scene. Use it as a reference "
            "for scene understanding and entity identification, but rewrite the reasoning in "
            "the symbolic format above. Do NOT copy it verbatim — translate it into "
            "[PERCEPTION] / [OPERATIONS] / [FACTS] / [RULES] / [ACTION].\n\n"
            f"{nl_cot_reference}\n\n"
            if nl_cot_reference else ""
        ) +

        f"Hint: The ground-truth driving action is **({gt_lateral}, {gt_longitudinal})**.\n"
        "Follow this 4-step workflow:\n"
        "  Step 1 — Write [PERCEPTION] purely from the images and the given ego state. "
        "Numerical attributes must be plausible estimates from what you see (distances in meters, "
        "speeds in m/s). Do NOT let the action hint influence what you perceive.\n"
        "  Step 2 — Optionally write [OPERATIONS] for any derived quantity you'll reuse "
        "(TTC, required_decel, gap, relative velocity, etc.). Skip the section entirely for "
        "simple scenes.\n"
        "  Step 3 — Write [FACTS]: each line is `name := <comparison/boolean expression> → True|False`. "
        "Every right-side expression must reference PERCEPTION attributes or OPERATIONS results.\n"
        "  Step 4 — Write [RULES]: exactly ONE rule whose body is a conjunction of FACT atoms "
        "(positive or NOT-negated) and whose head matches the hint. The rule body must be "
        "satisfied by your declared FACTS. Use `∧` (or `AND`) to conjoin and `¬` (or `NOT`) "
        "to negate. NO inline arithmetic in the rule body — only Boolean atoms.\n"
        "  Step 5 — Write [ACTION] = the exact head of your RULE (lateral, longitudinal).\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Produce exactly these sections in order (OPERATIONS optional):\n"
        "  [PERCEPTION]   numerical entity declarations (include `ego = Ego {...}`)\n"
        "  [OPERATIONS]   (optional) derived quantities: name := expr = result\n"
        "  [FACTS]        boolean atoms: name := expr → True|False\n"
        "  [RULES]        R: atom ∧ ¬atom ∧ ... → (LateralAction, LongitudinalAction)\n"
        "  [ACTION]       LateralAction, LongitudinalAction\n\n"

        "CRITICAL CONSISTENCY RULES:\n"
        "1. ACTION must be the literal tuple from your RULE head — no substitution.\n"
        "2. Every FACT must have a non-trivial `:=` right side grounded in PERCEPTION or OPERATIONS.\n"
        "3. RULES body uses Boolean atoms only — no `<`, `>`, `==` in rule bodies (those belong in FACTS).\n"
        "4. PERCEPTION MUST include the Ego entity with the speed/accel/instruction given above.\n"
        "5. Numerical values in PERCEPTION should be physically plausible (no `dist: 0` for a far car).\n"
        "6. A FACT's truth value must be consistent with evaluating its `:=` expression on PERCEPTION.\n\n"

        "=== EXAMPLES ===\n\n"
        "--- Example 1 (Red light, lead vehicle stopped) ---\n"
        f"{_EXAMPLE_1}\n\n"
        "--- Example 2 (Green light, open road, pedestrian on sidewalk) ---\n"
        f"{_EXAMPLE_2}\n\n"
        "--- Example 3 (Yellow light, approaching intersection) ---\n"
        f"{_EXAMPLE_3}\n\n"
        "--- Example 4 (Close following, lead vehicle moving) ---\n"
        f"{_EXAMPLE_4}\n\n"

        "Now produce the symbolic reasoning for the current scene. "
        "Output ONLY the five sections, no additional text."
    )

    return {"type": "text", "text": prompt_text}
