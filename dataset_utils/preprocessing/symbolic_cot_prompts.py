"""
Symbolic Chain-of-Thought (CoT) prompts for driving scenario reasoning.

Generates structured 4-stage symbolic prompts (PERCEPTION → OPERATIONS → FACTS → RULES → ACTION)
using the RLIB schema definitions. Used by symbolic_cot_sample_generation.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from functools import lru_cache

import yaml


# ---------------------------------------------------------------------------
# Action string → symbolic mapping
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
    """Map free-form action string to (lateral, longitudinal) symbolic pair.

    Examples:
        "move forward with a deceleration" -> ("KeepLane", "Deceleration")
        "turn left with an acceleration"   -> ("TurnLeft", "Acceleration")
        "STOP"                             -> ("KeepLane", "Stop")
    """
    action_str = action_str.strip()
    if not action_str or action_str.upper() == "STOP":
        return ("KeepLane", "Stop")

    # Parse "lateral with longitudinal"
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
# Ego state → qualitative mapping
# ---------------------------------------------------------------------------

def ego_state_to_qualitative(
    velocity: list[float],
    acceleration: list[float],
    instruction: str,
) -> dict[str, str]:
    """Map raw numeric ego state to symbolic EgoQuery vocabulary.

    Args:
        velocity: [vx, vy] in m/s
        acceleration: [ax, ay] in m/s^2
        instruction: e.g. "turn left", "keep forward", "go straight"

    Returns:
        {"speed": ..., "acceleration": ..., "instruction": ...}
    """
    # Speed from velocity magnitude
    speed_mag = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    if speed_mag < 0.3:
        speed = "Stopped"
    elif speed_mag < 3.0:
        speed = "Slow"
    elif speed_mag < 8.0:
        speed = "Medium"
    else:
        speed = "Fast"

    # Acceleration from x-direction (primary driving direction)
    ax = acceleration[0]
    if ax < -0.5:
        accel = "Braking"
    elif ax > 0.5:
        accel = "Accelerating"
    else:
        accel = "Coasting"

    # Instruction mapping
    inst_lower = instruction.strip().lower()
    if "left" in inst_lower:
        inst = "TurnLeft"
    elif "right" in inst_lower:
        inst = "TurnRight"
    else:
        inst = "KeepForward"

    return {"speed": speed, "acceleration": accel, "instruction": inst}


# ---------------------------------------------------------------------------
# RLIB schema → prompt text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_rlib_prompt_sections(rlib_dir: str) -> dict[str, str]:
    """Load RLIB YAML files and format as prompt text sections."""
    rlib = Path(rlib_dir)

    with open(rlib / "entities.yaml") as f:
        entities = yaml.safe_load(f)
    with open(rlib / "operations.yaml") as f:
        operations = yaml.safe_load(f)
    with open(rlib / "facts.yaml") as f:
        facts_file = yaml.safe_load(f)
    with open(rlib / "actions.yaml") as f:
        actions = yaml.safe_load(f)

    # --- Entity types ---
    # Format: base types without subtypes shown normally; types WITH subtypes list
    # subtypes as the valid class names (since you must use the subtype, not the base).
    entity_lines = []
    for base_type, cfg in entities.items():
        subtypes = cfg.get("subtypes", [])
        attrs = cfg.get("attributes", {})
        if subtypes:
            # Show subtypes as the usable class names; base type is just a category label
            entity_lines.append(f"- {base_type} [use one of: {', '.join(subtypes)}]")
        else:
            entity_lines.append(f"- {base_type}")
        for attr_name, attr_vals in attrs.items():
            entity_lines.append(f"    {attr_name}: [{', '.join(str(v) for v in attr_vals)}]")
    entity_types_text = (
        "ENTITY DECLARATION RULES:\n"
        "1. In PERCEPTION, ALWAYS use the specific subtype as the class name — NEVER the base type.\n"
        "   RIGHT: `v_1 = Truck {position: Front}`   WRONG: `v_1 = Vehicle {subtype: Truck}` or `{type: Truck}`\n"
        "   RIGHT: `r_1 = Crosswalk {position: Ahead}` WRONG: `r_1 = RoadFeature {subtype: Crosswalk}` or `{type: Crosswalk}`\n"
        "   RIGHT: `ts_1 = StopSign {position: Front}`  WRONG: `ts_1 = TrafficSign {subtype: StopSign}` or `{type: StopSign}`\n"
        "2. In OPERATIONS, use the base type for filtering: Filter(Vehicle, ...), Filter(RoadFeature, ...).\n"
        "3. Do NOT add `subtype:`, `type:`, or any extra field — only the listed attributes are valid.\n"
        "4. Use ONLY types listed below. Do NOT invent types (no 'Taxi', 'Bike', 'Signal', etc.).\n"
        "ATTRIBUTE VOCABULARY (exact values only):\n"
        "  Vehicle.motion: Stationary/Moving/Approaching/Receding/Decelerating/Accelerating (NOT 'Stopped')\n"
        "  Pedestrian.motion: Standing/Walking/Running/Crossing (NOT 'Waiting')\n"
        "  TrafficLight.color: Red/Yellow/Green/Off/Flashing (NOT True/False)\n"
        "  RoadFeature.position: Ahead/Current/Behind (NOT 'Front' or 'Back')\n"
        + "\n".join(entity_lines)
    )

    # --- Operations ---
    op_lines = []
    for op_name, op_cfg in operations.items():
        syntax = op_cfg.get("syntax", "")
        op_lines.append(f"- {op_name}: {syntax}")
        if op_name == "EgoQuery" and "attributes" in op_cfg:
            for attr, vals in op_cfg["attributes"].items():
                op_lines.append(f"    {attr}: [{', '.join(vals)}]")
    operations_text = "\n".join(op_lines)

    # --- Facts with descriptions ---
    vocabulary = facts_file["vocabulary"]
    groundings = facts_file.get("groundings", {})
    fact_lines = []
    for fact_name in vocabulary:
        desc = groundings.get(fact_name, {}).get("description", "")
        fact_lines.append(f"- {fact_name}: {desc}")
    facts_text = "\n".join(fact_lines)

    # --- Actions ---
    lateral = actions.get("lateral", [])
    longitudinal = actions.get("longitudinal", [])
    actions_text = (
        f"Lateral (choose exactly one): [{', '.join(lateral)}]\n"
        f"Longitudinal (choose exactly one): [{', '.join(longitudinal)}]"
    )

    # --- Rules summary ---
    rules_dir = rlib / "rules"
    rule_lines = []
    if rules_dir.exists():
        for rule_file in sorted(rules_dir.glob("*.yaml")):
            data = yaml.safe_load(open(rule_file))
            if not data or "rules" not in data:
                continue
            for rule in data["rules"]:
                rid = rule["id"]
                priority = rule["priority"]
                preconditions = " AND ".join(rule["preconditions"])
                lat = rule["action"]["lateral"]
                lon = rule["action"]["longitudinal"]
                rule_lines.append(f"{rid} [{priority}]: {preconditions} -> {lat}, {lon}")
    rules_text = "\n".join(rule_lines)

    return {
        "entity_types": entity_types_text,
        "operations": operations_text,
        "facts": facts_text,
        "actions": actions_text,
        "rules_summary": rules_text,
    }


# ---------------------------------------------------------------------------
# In-context examples
# ---------------------------------------------------------------------------

_EXAMPLE_1 = """\
PERCEPTION:
  tl_1 = TrafficLight {color: Red, position: Front, applies_to: EgoLane}
  v_1 = Car {position: Front, lane: EgoLane, distance: Near, motion: Stationary, signal: BrakeLights}
  r_1 = Intersection {position: Ahead, distance: Near}

OPERATIONS:
  Query(tl_1, color) = Red
  Filter(Vehicle, lane = EgoLane) = {v_1}
  EgoQuery(speed) = Slow
  EgoQuery(instruction) = KeepForward

FACTS:
  RedLight = True
  ApproachingIntersection = True
  LeadVehiclePresent = True
  LeadVehicleStopped = True
  CanStopComfortably = True
  EgoMovingSlow = True
  InstructionKeepForward = True

RULES:
  RedLight AND ApproachingIntersection AND CanStopComfortably AND NOT EgoStopped -> KeepLane, DecelerationToZero

ACTION: KeepLane, DecelerationToZero"""

_EXAMPLE_2 = """\
PERCEPTION:
  tl_1 = TrafficLight {color: Green, position: Front, applies_to: EgoLane}
  p_1 = Pedestrian {position: FrontRight, distance: Far, motion: Walking, location: Sidewalk}

OPERATIONS:
  Query(tl_1, color) = Green
  EgoQuery(speed) = Medium
  EgoQuery(instruction) = KeepForward

FACTS:
  GreenLight = True
  InstructionKeepForward = True

RULES:
  InstructionKeepForward AND NOT PathBlocked AND NOT LeadVehicleStopped AND NOT RedLight AND NOT StopSignPresent -> KeepLane, ConstantSpeed

ACTION: KeepLane, ConstantSpeed"""

_EXAMPLE_3 = """\
PERCEPTION:
  tl_1 = TrafficLight {color: Yellow, position: Front, applies_to: EgoLane}
  v_1 = Car {position: Front, lane: EgoLane, distance: Medium, motion: Decelerating}
  r_1 = Intersection {position: Ahead, distance: Near}

OPERATIONS:
  Query(tl_1, color) = Yellow
  EgoQuery(speed) = Medium
  EgoQuery(instruction) = KeepForward

FACTS:
  YellowLight = True
  ApproachingIntersection = True
  LeadVehiclePresent = True
  LeadVehicleBraking = True
  CanStopComfortably = True

RULES:
  YellowLight AND ApproachingIntersection AND CanStopComfortably -> KeepLane, Deceleration

ACTION: KeepLane, Deceleration"""

_EXAMPLE_4 = """\
PERCEPTION:
  v_1 = Car {position: Front, lane: EgoLane, distance: Near, motion: Moving, signal: None}
  v_2 = Car {position: Behind, lane: EgoLane, distance: Near, motion: Approaching}

OPERATIONS:
  Filter(Vehicle, lane = EgoLane) = {v_1}
  EgoQuery(speed) = Medium
  EgoQuery(acceleration) = Coasting
  EgoQuery(instruction) = KeepForward

FACTS:
  LeadVehiclePresent = True
  LeadVehicleClose = True
  RearVehicleApproaching = True
  InstructionKeepForward = True

RULES:
  LeadVehicleClose AND NOT LeadVehicleBraking AND NOT LeadVehicleStopped AND NOT EgoStopped -> KeepLane, Deceleration

ACTION: KeepLane, Deceleration"""


# ---------------------------------------------------------------------------
# Main prompt function
# ---------------------------------------------------------------------------

def get_symbolic_cot_prompt(
    rlib_dir: str,
    fut_ego_action: str,
    ego_speed: str,
    ego_acceleration: str,
    ego_instruction: str,
    nl_cot_reference: str | None = None,
    use_predefined_rules: bool = True,
) -> dict:
    """Build the symbolic CoT reasoning prompt.

    Args:
        rlib_dir: Path to RLIB directory.
        fut_ego_action: Free-form GT action string (e.g. "move forward with a deceleration").
        ego_speed: Qualitative speed ("Stopped"/"Slow"/"Medium"/"Fast").
        ego_acceleration: Qualitative acceleration ("Braking"/"Coasting"/"Accelerating").
        ego_instruction: Qualitative instruction ("KeepForward"/"TurnLeft"/"TurnRight").

    Returns:
        {"type": "text", "text": prompt_text}
    """
    sections = _load_rlib_prompt_sections(rlib_dir)
    gt_lateral, gt_longitudinal = action_string_to_symbolic(fut_ego_action)

    prompt_text = (
        "Based on the above camera images and ego vehicle states, generate a structured symbolic "
        "reasoning chain following the 4-stage format below.\n\n"

        "=== ENTITY TYPES ===\n"
        "Use ID prefixes: v_ for Vehicle/subtypes (Car,Truck,Bus...), tl_ for TrafficLight, "
        "p_ for Pedestrian, c_ for Cyclist/subtypes, ts_ for TrafficSign/subtypes (StopSign...), "
        "r_ for RoadFeature/subtypes (Crosswalk,Intersection...).\n"
        f"{sections['entity_types']}\n\n"

        "=== OPERATIONS ===\n"
        f"{sections['operations']}\n\n"

        "=== FACT VOCABULARY ===\n"
        "Set each relevant fact to True or False based on your perception and operations.\n"
        "FACTS COMPLETENESS RULES:\n"
        "1. Every fact used as a POSITIVE (non-NOT) condition in RULES must appear in FACTS as True.\n"
        "   Before writing RULES, verify each positive condition you plan to use is already in FACTS.\n"
        "2. LeftLaneClear / RightLaneClear: declare True ONLY if your PERCEPTION contains NO vehicle\n"
        "   in that lane at Near or VeryNear distance. Cross-check entity-by-entity.\n"
        "   Example: if you see `v_2 = Car {lane: RightLane, distance: Near}`, then RightLaneClear = False.\n"
        f"{sections['facts']}\n\n"

        "=== ACTIONS ===\n"
        f"{sections['actions']}\n\n"

        + (
            "=== PREDEFINED RULES (higher number = higher priority; when multiple rules match, select the highest-priority one) ===\n"
            f"{sections['rules_summary']}\n\n"
            if use_predefined_rules else
            "=== RULE SYNTAX ===\n"
            "Write your own rule using the facts you declared. Syntax:\n"
            "  condition1 AND condition2 AND NOT condition3 -> LateralAction, LongitudinalAction\n"
            "Conditions must be facts from the FACT VOCABULARY above. Use AND to combine, NOT to negate.\n"
            "Write exactly ONE rule that leads to your chosen action.\n\n"
        ) +

        "=== EGO VEHICLE STATE ===\n"
        f"EgoQuery(speed) = {ego_speed}\n"
        f"EgoQuery(acceleration) = {ego_acceleration}\n"
        f"EgoQuery(instruction) = {ego_instruction}\n\n"

        + (
            "=== REFERENCE: NATURAL LANGUAGE REASONING ===\n"
            "Below is a free-form reasoning trace for the same scene. "
            "Use it as a reference for scene understanding and entity identification, "
            "but rewrite the reasoning in the symbolic format above. "
            "Do NOT copy it verbatim — translate it into PERCEPTION/OPERATIONS/FACTS/RULES/ACTION.\n\n"
            f"{nl_cot_reference}\n\n"
            if nl_cot_reference else ""
        ) +

        f"Hint: The ground truth driving action is **{gt_lateral}, {gt_longitudinal}**.\n"
        "Follow this 3-step workflow:\n"
        "  Step 1 — Write PERCEPTION and FACTS purely from the scene images. "
        "Do NOT let the hint influence what you perceive or what facts you declare.\n"
        + (
            "  Step 2 — Write RULES: find all predefined rules whose (a) every positive condition "
            "is already True in your FACTS, AND (b) action matches the hint. "
            "If multiple rules qualify, select the one with the HIGHEST priority number. "
            "If no predefined rule satisfies both constraints, write a simplified custom rule using "
            "ONLY conditions that are True in your FACTS — drop any condition you cannot support.\n"
            if use_predefined_rules else
            "  Step 2 — Write RULES: compose a single rule from your declared FACTS that leads to "
            "an action matching the hint. Use ONLY facts declared True as positive conditions. "
            "Use NOT for facts that are absent or False. The rule action must match the hint.\n"
        ) +
        "  Step 3 — Write ACTION = the action stated in your selected/written rule, exactly. "
        "Never substitute a different value from the hint.\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Produce exactly five sections in this order:\n"
        "  PERCEPTION:   — list detected entities with typed attributes\n"
        "  OPERATIONS:   — query/filter operations on entities\n"
        "  FACTS:        — boolean facts derived from operations\n"
        "  RULES:        — fact conditions -> action (use AND/NOT)\n"
        "  ACTION:       — final lateral, longitudinal action\n\n"

        "CRITICAL CONSISTENCY RULES:\n"
        "1. ACTION must be identical to the action in your RULE (lateral AND longitudinal).\n"
        "2. Every positive (non-NOT) condition in your RULE must be declared True in FACTS.\n"
        + (
            "   If a condition is not in your FACTS (e.g., GreenLight when no traffic light is "
            "visible in the scene), REMOVE it from the rule.\n"
            if use_predefined_rules else
            "   Do NOT use any fact as a positive condition unless you declared it True in FACTS.\n"
        ) +
        "3. CanStopComfortably is a judgment fact — if you use it in RULES, you must explicitly "
        "declare `CanStopComfortably = True` in FACTS.\n"
        "4. LeadVehicleClose requires a Vehicle in EgoLane at Near or VeryNear distance — "
        "declare it in FACTS whenever that vehicle is present.\n"
        "5. Never write a rule condition that contradicts your declared facts.\n\n"

        "=== EXAMPLES ===\n\n"
        "--- Example 1 (Red light, approaching intersection) ---\n"
        f"{_EXAMPLE_1}\n\n"
        "--- Example 2 (Green light, open road) ---\n"
        f"{_EXAMPLE_2}\n\n"
        "--- Example 3 (Yellow light, approaching intersection) ---\n"
        f"{_EXAMPLE_3}\n\n"
        "--- Example 4 (Close following, moving lead vehicle) ---\n"
        f"{_EXAMPLE_4}\n\n"

        "Now produce the symbolic reasoning for the current scene. "
        "Output ONLY the five sections (PERCEPTION, OPERATIONS, FACTS, RULES, ACTION), "
        "no additional text."
    )

    return {"type": "text", "text": prompt_text}
