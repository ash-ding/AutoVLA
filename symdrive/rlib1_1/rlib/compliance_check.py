#!/usr/bin/env python3
"""RLIB Compliance Check — Self-contained consistency validator.

Checks that all YAML definitions in the RLIB directory are internally consistent.
Does NOT depend on AutoVLA code.

Usage:
    python RLIB/compliance_check.py
    python RLIB/compliance_check.py /path/to/RLIB
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Category prefix → expected category name mapping
# ---------------------------------------------------------------------------

_PREFIX_TO_CATEGORY = {
    "TC": "traffic_control",
    "EM": "emergency",
    "PS": "pedestrian_safety",
    "CA": "collision_avoidance",
    "NV": "navigation",
    "SM": "speed_management",
    "CM": "comfort_meta",
}


def load_yaml(path: Path) -> dict | list:
    with open(path) as f:
        return yaml.safe_load(f)


def run_checks(rlib_dir: Path) -> list[str]:
    """Run all 15 compliance checks. Returns list of error messages."""
    errors: list[str] = []

    # Load files
    entities = load_yaml(rlib_dir / "entities.yaml")
    operations = load_yaml(rlib_dir / "operations.yaml")
    facts_file = load_yaml(rlib_dir / "facts.yaml")
    actions = load_yaml(rlib_dir / "actions.yaml")

    vocabulary: list[str] = facts_file["vocabulary"]
    groundings: dict = facts_file.get("groundings", {})

    lateral_actions: set[str] = set(actions["lateral"])
    longitudinal_actions: set[str] = set(actions["longitudinal"])

    # Build entity type maps
    all_base_types: set[str] = set(entities.keys())
    subtype_to_base: dict[str, str] = {}
    type_attributes: dict[str, dict[str, list]] = {}

    for base_type, cfg in entities.items():
        subtype_to_base[base_type] = base_type
        type_attributes[base_type] = cfg.get("attributes", {})
        for subtype in cfg.get("subtypes", []):
            subtype_to_base[subtype] = base_type

    # EgoQuery attributes
    ego_cfg = operations.get("EgoQuery", {})
    ego_attributes: dict[str, list[str]] = ego_cfg.get("attributes", {})

    # Load all rules from rules/ directory
    all_rules: list[dict] = []
    rules_dir = rlib_dir / "rules"
    if rules_dir.exists():
        for rule_file in sorted(rules_dir.glob("*.yaml")):
            data = load_yaml(rule_file)
            if data and "rules" in data:
                for rule in data["rules"]:
                    rule["_source_file"] = rule_file.name
                    all_rules.append(rule)

    vocab_set = set(vocabulary)

    # ---- Check 1: Rules reference valid facts ----
    for rule in all_rules:
        for precond in rule.get("preconditions", []):
            fact_name = precond.replace("NOT ", "").strip()
            if fact_name not in vocab_set:
                errors.append(
                    f"[Check 1] Rule {rule['id']}: precondition fact '{fact_name}' "
                    f"not in vocabulary"
                )

    # ---- Check 2: Rules have valid lateral actions ----
    for rule in all_rules:
        lat = rule.get("action", {}).get("lateral", "")
        if lat not in lateral_actions:
            errors.append(
                f"[Check 2] Rule {rule['id']}: lateral action '{lat}' not in actions.yaml"
            )

    # ---- Check 3: Rules have valid longitudinal actions ----
    for rule in all_rules:
        lon = rule.get("action", {}).get("longitudinal", "")
        if lon not in longitudinal_actions:
            errors.append(
                f"[Check 3] Rule {rule['id']}: longitudinal action '{lon}' not in actions.yaml"
            )

    # ---- Check 4: Grounding base_type in entities ----
    for fact_name, g in groundings.items():
        for i, cond in enumerate(g.get("conditions", [])):
            bt = cond.get("base_type", "")
            if bt and bt not in all_base_types:
                errors.append(
                    f"[Check 4] Grounding '{fact_name}' condition {i}: "
                    f"base_type '{bt}' not in entities.yaml"
                )

    # ---- Check 5: Grounding subtype valid for base_type ----
    for fact_name, g in groundings.items():
        for i, cond in enumerate(g.get("conditions", [])):
            st = cond.get("subtype", "")
            bt = cond.get("base_type", "")
            if st and bt:
                valid_subtypes = entities.get(bt, {}).get("subtypes", [])
                if st not in valid_subtypes:
                    errors.append(
                        f"[Check 5] Grounding '{fact_name}' condition {i}: "
                        f"subtype '{st}' not in {bt}.subtypes"
                    )

    # ---- Check 6: Grounding attribute keys/values valid ----
    for fact_name, g in groundings.items():
        for i, cond in enumerate(g.get("conditions", [])):
            bt = cond.get("base_type", "")
            if not bt or bt not in all_base_types:
                continue
            valid_attrs = type_attributes.get(bt, {})

            # Check attributes (exact match)
            for attr_key, attr_val in cond.get("attributes", {}).items():
                if attr_key not in valid_attrs:
                    errors.append(
                        f"[Check 6] Grounding '{fact_name}' condition {i}: "
                        f"attribute '{attr_key}' not valid for {bt}"
                    )
                elif attr_val not in valid_attrs[attr_key]:
                    errors.append(
                        f"[Check 6] Grounding '{fact_name}' condition {i}: "
                        f"value '{attr_val}' not valid for {bt}.{attr_key}"
                    )

            # Check attr_in
            for attr_key, attr_vals in cond.get("attr_in", {}).items():
                if attr_key not in valid_attrs:
                    errors.append(
                        f"[Check 6] Grounding '{fact_name}' condition {i}: "
                        f"attr_in key '{attr_key}' not valid for {bt}"
                    )
                else:
                    for v in attr_vals:
                        if v not in valid_attrs[attr_key]:
                            errors.append(
                                f"[Check 6] Grounding '{fact_name}' condition {i}: "
                                f"attr_in value '{v}' not valid for {bt}.{attr_key}"
                            )

    # ---- Check 7: All rule IDs globally unique ----
    seen_ids: set[str] = set()
    for rule in all_rules:
        rid = rule.get("id", "")
        if rid in seen_ids:
            errors.append(f"[Check 7] Duplicate rule ID: '{rid}'")
        seen_ids.add(rid)

    # ---- Check 8: Priority in [0, 200] ----
    for rule in all_rules:
        p = rule.get("priority")
        if not isinstance(p, int) or p < 0 or p > 200:
            errors.append(
                f"[Check 8] Rule {rule.get('id', '?')}: "
                f"priority {p!r} not an integer in [0, 200]"
            )

    # ---- Check 9: Required fields present ----
    required_fields = {"id", "priority", "category", "description",
                       "preconditions", "action", "explanation"}
    for rule in all_rules:
        missing = required_fields - set(rule.keys())
        if missing:
            errors.append(
                f"[Check 9] Rule {rule.get('id', '?')}: missing fields {missing}"
            )

    # ---- Check 10: Action has exactly lateral + longitudinal ----
    for rule in all_rules:
        action = rule.get("action", {})
        keys = set(action.keys())
        expected = {"lateral", "longitudinal"}
        if keys != expected:
            errors.append(
                f"[Check 10] Rule {rule.get('id', '?')}: "
                f"action keys {keys} != {expected}"
            )

    # ---- Check 11: Vocabulary has no duplicates ----
    if len(vocabulary) != len(set(vocabulary)):
        dupes = [v for v in vocabulary if vocabulary.count(v) > 1]
        errors.append(f"[Check 11] Duplicate facts in vocabulary: {set(dupes)}")

    # ---- Check 12: Every vocabulary fact has a grounding ----
    for fact_name in vocabulary:
        if fact_name not in groundings:
            errors.append(
                f"[Check 12] Fact '{fact_name}' in vocabulary but no grounding defined"
            )

    # ---- Check 13: Grounding condition kind valid ----
    valid_kinds = {"entity", "no_entity", "ego", "judgment"}
    for fact_name, g in groundings.items():
        for i, cond in enumerate(g.get("conditions", [])):
            kind = cond.get("kind", "")
            if kind not in valid_kinds:
                errors.append(
                    f"[Check 13] Grounding '{fact_name}' condition {i}: "
                    f"kind '{kind}' not in {valid_kinds}"
                )

    # ---- Check 14: Ego conditions reference valid EgoQuery attributes ----
    for fact_name, g in groundings.items():
        for i, cond in enumerate(g.get("conditions", [])):
            if cond.get("kind") != "ego":
                continue
            attr = cond.get("attribute", "")
            val = cond.get("value", "")
            if attr not in ego_attributes:
                errors.append(
                    f"[Check 14] Grounding '{fact_name}' condition {i}: "
                    f"ego attribute '{attr}' not in EgoQuery"
                )
            elif val not in ego_attributes[attr]:
                errors.append(
                    f"[Check 14] Grounding '{fact_name}' condition {i}: "
                    f"ego value '{val}' not valid for EgoQuery.{attr}"
                )

    # ---- Check 15: Rule category matches file prefix ----
    for rule in all_rules:
        rid = rule.get("id", "")
        category = rule.get("category", "")
        source = rule.get("_source_file", "")

        # Extract prefix from rule ID (e.g., "TC" from "TC-001")
        prefix = rid.split("-")[0] if "-" in rid else ""
        expected_category = _PREFIX_TO_CATEGORY.get(prefix, "")

        if expected_category and category != expected_category:
            errors.append(
                f"[Check 15] Rule {rid}: category '{category}' "
                f"doesn't match prefix '{prefix}' (expected '{expected_category}')"
            )

        # Also check file prefix matches
        file_prefix = source.split("_")[0] if "_" in source else ""
        if file_prefix and prefix and file_prefix != prefix:
            errors.append(
                f"[Check 15] Rule {rid}: ID prefix '{prefix}' "
                f"doesn't match file prefix '{file_prefix}' in {source}"
            )

    return errors


def main():
    if len(sys.argv) > 1:
        rlib_dir = Path(sys.argv[1])
    else:
        rlib_dir = Path(__file__).parent

    print(f"RLIB Compliance Check: {rlib_dir}")
    print("=" * 60)

    errors = run_checks(rlib_dir)

    if errors:
        print(f"\nFAILED — {len(errors)} error(s):\n")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        # Print summary
        facts_file = load_yaml(rlib_dir / "facts.yaml")
        vocab_count = len(facts_file["vocabulary"])
        grounding_count = len(facts_file.get("groundings", {}))

        rules_dir = rlib_dir / "rules"
        rule_count = 0
        file_count = 0
        if rules_dir.exists():
            for rf in sorted(rules_dir.glob("*.yaml")):
                data = load_yaml(rf)
                if data and "rules" in data:
                    file_count += 1
                    rule_count += len(data["rules"])

        print(f"\nPASSED — All 15 checks OK")
        print(f"  Facts:      {vocab_count} vocabulary, {grounding_count} groundings")
        print(f"  Rules:      {rule_count} rules across {file_count} files")
        print(f"  Entities:   {len(load_yaml(rlib_dir / 'entities.yaml'))} base types")
        sys.exit(0)


if __name__ == "__main__":
    main()
