"""
Symbolic Driving Rules DSL — Four-Stage Neuro-Symbolic Reasoning Pipeline.

The VLM generates structured symbolic reasoning inside <think>...</think> blocks,
progressing through four stages:

  Stage 1 — PERCEPTION:  Detect and enumerate entities with typed attributes
  Stage 2 — OPERATIONS:  Query, filter, count, check existence of entities
  Stage 3 — FACTS:       Derive high-level boolean facts from operations
  Stage 4 — RULES:       Match facts to lateral + longitudinal actions

Example output:

    PERCEPTION:
      tl_1 = TrafficLight {color: Yellow, position: Front, applies_to: EgoLane}
      v_1 = Car {position: Front, lane: EgoLane, distance: Near, motion: Decelerating}
      r_1 = Intersection {position: Ahead, distance: VeryNear}

    OPERATIONS:
      Query(tl_1, color) = Yellow
      Filter(Vehicle, lane = EgoLane) = {v_1}
      EgoQuery(speed) = Medium

    FACTS:
      YellowLight = True
      ApproachingIntersection = True
      LeadVehicleClose = True
      CanStopComfortably = True

    RULES:
      YellowLight AND ApproachingIntersection AND CanStopComfortably -> KeepLane, Deceleration

    ACTION: KeepLane, Deceleration
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A detected scene entity, e.g. v_1 = Car {position: Front, lane: EgoLane}."""
    id: str                     # "tl_1", "v_1", "p_1", "r_1"
    entity_type: str            # concrete type: "Car", "TrafficLight", "Intersection"
    base_type: str              # abstract type: "Vehicle", "TrafficLight", "RoadFeature"
    attributes: dict[str, str]  # {"color": "Yellow", "position": "Front"}

    def __str__(self) -> str:
        attrs = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
        return f"{self.id} = {self.entity_type} {{{attrs}}}"


@dataclass
class Operation:
    """An operation on entities, e.g. Query(tl_1, color) = Yellow."""
    op_type: str    # "Query", "Filter", "Count", "Exists", "Nearest", "EgoQuery"
    expression: str # full left-hand side: "Query(tl_1, color)"
    result: str     # right-hand side: "Yellow", "{v_1}", "0", "True"

    def __str__(self) -> str:
        return f"{self.expression} = {self.result}"


@dataclass
class Fact:
    """A high-level boolean fact, e.g. YellowLight = True."""
    name: str       # from predefined vocabulary
    value: bool

    def __str__(self) -> str:
        return f"{self.name} = {self.value}"


@dataclass
class Rule:
    """IF fact conditions THEN action, e.g. YellowLight AND X -> KeepLane, Deceleration."""
    conditions: list[tuple[str, bool]]  # [(fact_name, expected_value), ...]
    lateral_action: str
    longitudinal_action: str

    def __str__(self) -> str:
        parts = []
        for name, val in self.conditions:
            parts.append(name if val else f"NOT {name}")
        cond_str = " AND ".join(parts)
        return f"{cond_str} -> {self.lateral_action}, {self.longitudinal_action}"


@dataclass
class SymbolicOutput:
    """Complete parsed output from the model's <think> block."""
    entities: list[Entity] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    selected_lateral: str = ""
    selected_longitudinal: str = ""

    def __str__(self) -> str:
        lines = ["PERCEPTION:"]
        for e in self.entities:
            lines.append(f"  {e}")
        lines.append("")
        lines.append("OPERATIONS:")
        for o in self.operations:
            lines.append(f"  {o}")
        lines.append("")
        lines.append("FACTS:")
        for f in self.facts:
            lines.append(f"  {f}")
        lines.append("")
        lines.append("RULES:")
        for r in self.rules:
            lines.append(f"  {r}")
        lines.append("")
        lines.append(f"ACTION: {self.selected_lateral}, {self.selected_longitudinal}")
        return "\n".join(lines)


class ParseError(ValueError):
    """Raised when model output cannot be parsed."""
    pass


# ---------------------------------------------------------------------------
# Grounding data classes
# ---------------------------------------------------------------------------

@dataclass
class GroundingCondition:
    """One branch of a fact's OR-disjunction of grounding conditions."""
    kind: str               # "entity" | "no_entity" | "ego" | "judgment"
    base_type: str = ""     # for entity/no_entity
    subtype: str = ""       # optional concrete subtype filter
    attributes: dict[str, str] = field(default_factory=dict)       # exact-match (AND)
    attr_in: dict[str, list[str]] = field(default_factory=dict)    # set-membership (OR within)
    attribute: str = ""     # for ego kind
    value: str = ""         # for ego kind
    description: str = ""   # for judgment kind


@dataclass
class FactGrounding:
    """Parsed grounding definition for one fact."""
    name: str
    description: str
    category: str
    conditions: list[GroundingCondition]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class SymbolicSchema:
    """Loads the DSL schema from a single YAML file or an RLIB directory."""

    def __init__(self, config_path: str | Path, *, _cfg: dict | None = None):
        if _cfg is not None:
            self._init_from_cfg(_cfg)
        else:
            config_path = Path(config_path)
            if config_path.is_dir():
                self._init_from_rlib(config_path)
            else:
                with open(config_path) as f:
                    self._init_from_cfg(yaml.safe_load(f))

    def _init_from_cfg(self, cfg: dict) -> None:
        self.entities_cfg: dict = cfg["entities"]
        self.operations_cfg: dict = cfg.get("operations", {})
        self.fact_vocabulary: set[str] = set(cfg["facts"])
        self.lateral_actions: set[str] = set(cfg["actions"]["lateral"])
        self.longitudinal_actions: set[str] = set(cfg["actions"]["longitudinal"])

        # Build type maps
        self._subtype_to_base: dict[str, str] = {}
        self._all_types: set[str] = set()
        self._type_attributes: dict[str, dict[str, list[str]]] = {}

        for base_type, type_cfg in self.entities_cfg.items():
            self._all_types.add(base_type)
            self._subtype_to_base[base_type] = base_type
            self._type_attributes[base_type] = type_cfg.get("attributes", {})

            for subtype in type_cfg.get("subtypes", []):
                self._all_types.add(subtype)
                self._subtype_to_base[subtype] = base_type

        # EgoQuery valid attributes
        ego_cfg = self.operations_cfg.get("EgoQuery", {})
        self.ego_attributes: dict[str, list[str]] = ego_cfg.get("attributes", {})

        # Fact grounding conditions
        raw_groundings = cfg.get("fact_groundings", {})
        self.fact_groundings: dict[str, FactGrounding] = {}
        for fact_name, fg_cfg in raw_groundings.items():
            cond_list = []
            for c in fg_cfg.get("conditions", []):
                cond_list.append(GroundingCondition(
                    kind=c["kind"],
                    base_type=c.get("base_type", ""),
                    subtype=c.get("subtype", ""),
                    attributes=c.get("attributes", {}),
                    attr_in=c.get("attr_in", {}),
                    attribute=c.get("attribute", ""),
                    value=c.get("value", ""),
                    description=c.get("description", ""),
                ))
            self.fact_groundings[fact_name] = FactGrounding(
                name=fact_name,
                description=fg_cfg.get("description", ""),
                category=fg_cfg.get("category", ""),
                conditions=cond_list,
            )

    def _init_from_rlib(self, rlib_dir: Path) -> None:
        """Load schema from an RLIB directory with split YAML files."""
        with open(rlib_dir / "entities.yaml") as f:
            entities = yaml.safe_load(f)
        with open(rlib_dir / "operations.yaml") as f:
            operations = yaml.safe_load(f)
        with open(rlib_dir / "facts.yaml") as f:
            facts_file = yaml.safe_load(f)
        with open(rlib_dir / "actions.yaml") as f:
            actions = yaml.safe_load(f)

        cfg = {
            "entities": entities,
            "operations": operations,
            "facts": facts_file["vocabulary"],
            "actions": actions,
            "fact_groundings": facts_file.get("groundings", {}),
        }
        self._init_from_cfg(cfg)

    def get_fact_grounding(self, fact_name: str) -> Optional[FactGrounding]:
        """Return grounding conditions for a fact, or None."""
        return self.fact_groundings.get(fact_name)

    def has_grounding(self, fact_name: str) -> bool:
        return fact_name in self.fact_groundings

    def get_base_type(self, entity_type: str) -> Optional[str]:
        """Return the abstract base type for a concrete type, or None."""
        return self._subtype_to_base.get(entity_type)

    def is_valid_type(self, entity_type: str) -> bool:
        return entity_type in self._all_types

    def get_valid_attribute_values(self, base_type: str, attr: str) -> Optional[list[str]]:
        """Return valid values for an attribute of a base type, or None."""
        attrs = self._type_attributes.get(base_type, {})
        return attrs.get(attr)

    def is_valid_fact(self, name: str) -> bool:
        return name in self.fact_vocabulary

    def is_valid_action(self, lateral: str, longitudinal: str) -> bool:
        return lateral in self.lateral_actions and longitudinal in self.longitudinal_actions


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Entity line: "v_1 = Car {position: Front, lane: EgoLane, ...}"
_ENTITY_PATTERN = re.compile(
    r"(\w+)\s*=\s*(\w+)\s*\{([^}]*)\}"
)

# Operation line: "Query(tl_1, color) = Yellow" or "Count(Filter(...)) = 3"
_OPERATION_PATTERN = re.compile(
    r"(.+?)\s*=\s*(.+)$"
)

# Fact line: "YellowLight = True" or "YellowLight = False"
_FACT_PATTERN = re.compile(
    r"(\w+)\s*=\s*(True|False)$"
)

# Rule arrow
_ARROW = re.compile(r"\s*(?:->|→)\s*")

# Section headers
_SECTION_HEADERS = [
    ("PERCEPTION", re.compile(r"(?:^|\n)\s*PERCEPTION\s*:\s*\n", re.IGNORECASE)),
    ("OPERATIONS", re.compile(r"(?:^|\n)\s*OPERATIONS\s*:\s*\n", re.IGNORECASE)),
    ("FACTS", re.compile(r"(?:^|\n)\s*FACTS\s*:\s*\n", re.IGNORECASE)),
    ("RULES", re.compile(r"(?:^|\n)\s*RULES\s*:\s*\n", re.IGNORECASE)),
    ("ACTION", re.compile(r"(?:^|\n)\s*ACTION\s*:\s*", re.IGNORECASE)),
]


class SymbolicParser:
    """Parse four-stage symbolic reasoning text into SymbolicOutput."""

    def __init__(self, schema: SymbolicSchema | str | Path):
        if isinstance(schema, (str, Path)):
            schema = SymbolicSchema(schema)
        self.schema = schema

    def parse(self, text: str) -> SymbolicOutput:
        """Parse full <think> block text. Raises ParseError on failure."""
        text = text.strip()
        sections = self._split_sections(text)

        entities = self._parse_perception(sections.get("PERCEPTION", ""))
        operations = self._parse_operations(sections.get("OPERATIONS", ""))
        facts = self._parse_facts(sections.get("FACTS", ""))
        rules = self._parse_rules(sections.get("RULES", ""))
        lateral, longitudinal = self._parse_action(sections.get("ACTION", ""))

        return SymbolicOutput(
            entities=entities,
            operations=operations,
            facts=facts,
            rules=rules,
            selected_lateral=lateral,
            selected_longitudinal=longitudinal,
        )

    def _split_sections(self, text: str) -> dict[str, str]:
        """Split text into named sections."""
        # Find all section positions
        found: list[tuple[str, int, int]] = []  # (name, header_start, content_start)
        for name, pattern in _SECTION_HEADERS:
            m = pattern.search(text)
            if m:
                found.append((name, m.start(), m.end()))

        if not found:
            raise ParseError("No section headers found (expected PERCEPTION, OPERATIONS, FACTS, RULES, ACTION)")

        # Sort by position
        found.sort(key=lambda x: x[1])

        # Extract content between consecutive headers
        sections = {}
        for i, (name, _, content_start) in enumerate(found):
            if i + 1 < len(found):
                content_end = found[i + 1][1]
            else:
                content_end = len(text)
            sections[name] = text[content_start:content_end].strip()

        return sections

    def _parse_perception(self, text: str) -> list[Entity]:
        """Parse PERCEPTION section into Entity list."""
        entities = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _ENTITY_PATTERN.match(line)
            if not m:
                raise ParseError(f"Cannot parse entity: {line!r}")

            eid = m.group(1)
            etype = m.group(2)
            attrs_str = m.group(3)

            # Parse attributes
            attributes = {}
            for pair in attrs_str.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if ":" not in pair:
                    raise ParseError(f"Invalid attribute format in entity {eid}: {pair!r}")
                key, val = pair.split(":", 1)
                attributes[key.strip()] = val.strip()

            base_type = self.schema.get_base_type(etype)
            if base_type is None:
                raise ParseError(f"Unknown entity type: {etype!r}")

            entities.append(Entity(
                id=eid,
                entity_type=etype,
                base_type=base_type,
                attributes=attributes,
            ))

        return entities

    def _parse_operations(self, text: str) -> list[Operation]:
        """Parse OPERATIONS section into Operation list."""
        operations = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _OPERATION_PATTERN.match(line)
            if not m:
                raise ParseError(f"Cannot parse operation: {line!r}")

            expression = m.group(1).strip()
            result = m.group(2).strip()

            # Determine operation type from expression
            op_type = "Unknown"
            for known_op in ["Query", "Filter", "Count", "Exists", "Nearest", "EgoQuery"]:
                if expression.startswith(known_op + "("):
                    op_type = known_op
                    break

            if op_type == "Unknown":
                raise ParseError(f"Unknown operation type in: {expression!r}")

            operations.append(Operation(op_type=op_type, expression=expression, result=result))

        return operations

    def _parse_facts(self, text: str) -> list[Fact]:
        """Parse FACTS section into Fact list."""
        facts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _FACT_PATTERN.match(line)
            if not m:
                raise ParseError(f"Cannot parse fact: {line!r}")

            name = m.group(1)
            value = m.group(2) == "True"
            facts.append(Fact(name=name, value=value))

        return facts

    def _parse_rules(self, text: str) -> list[Rule]:
        """Parse RULES section into Rule list."""
        rules = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = _ARROW.split(line, maxsplit=1)
            if len(parts) != 2:
                raise ParseError(f"Cannot parse rule (missing ->): {line!r}")

            cond_str, action_str = parts

            # Parse conditions: "YellowLight AND NOT PathBlocked AND ..."
            conditions = []
            for token in re.split(r"\s+AND\s+", cond_str.strip()):
                token = token.strip()
                if token.startswith("NOT "):
                    fact_name = token[4:].strip()
                    conditions.append((fact_name, False))
                else:
                    conditions.append((token, True))

            if not conditions:
                raise ParseError(f"No conditions found in rule: {line!r}")

            # Parse action
            action_parts = [a.strip() for a in action_str.split(",")]
            if len(action_parts) != 2:
                raise ParseError(
                    f"Expected 'LateralAction, LongitudinalAction' but got: {action_str!r}"
                )

            rules.append(Rule(
                conditions=conditions,
                lateral_action=action_parts[0],
                longitudinal_action=action_parts[1],
            ))

        return rules

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Parse ACTION line."""
        text = text.strip()
        if not text:
            raise ParseError("Missing ACTION")
        text = text.splitlines()[0].strip()
        parts = [a.strip() for a in text.split(",")]
        if len(parts) != 2:
            raise ParseError(f"Expected 'LateralAction, LongitudinalAction' but got: {text!r}")
        return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SymbolicValidator:
    """Validate parsed SymbolicOutput for consistency and grounding."""

    def __init__(
        self,
        schema: SymbolicSchema | str | Path,
        grounding_strictness: str = "warn",
    ):
        if isinstance(schema, (str, Path)):
            schema = SymbolicSchema(schema)
        self.schema = schema
        self.grounding_strictness = grounding_strictness  # "none" | "warn" | "strict"

    def validate(self, output: SymbolicOutput) -> tuple[bool, list[str], list[str]]:
        """Validate a SymbolicOutput.

        Returns (is_valid, violations, grounding_warnings).
        """
        violations = []
        violations.extend(self._check_entities(output.entities))
        violations.extend(self._check_operations(output))
        violations.extend(self._check_facts(output.facts))
        violations.extend(self._check_rules(output))
        violations.extend(self._check_action(output))

        grounding_warnings: list[str] = []
        if self.grounding_strictness != "none":
            grounding_warnings = self._check_grounding(output)
            if self.grounding_strictness == "strict":
                violations.extend(grounding_warnings)
                grounding_warnings = []

        return len(violations) == 0, violations, grounding_warnings

    def _check_entities(self, entities: list[Entity]) -> list[str]:
        violations = []
        seen_ids = set()
        for e in entities:
            # Duplicate id
            if e.id in seen_ids:
                violations.append(f"Duplicate entity id: {e.id}")
            seen_ids.add(e.id)

            # Valid type
            if not self.schema.is_valid_type(e.entity_type):
                violations.append(f"Unknown entity type: {e.entity_type!r} (entity {e.id})")
                continue

            # Valid attributes
            for attr, val in e.attributes.items():
                valid_vals = self.schema.get_valid_attribute_values(e.base_type, attr)
                if valid_vals is None:
                    violations.append(
                        f"Unknown attribute '{attr}' for {e.base_type} (entity {e.id})"
                    )
                elif val not in valid_vals:
                    violations.append(
                        f"Invalid value '{val}' for {e.base_type}.{attr} (entity {e.id}). "
                        f"Valid: {valid_vals}"
                    )

        return violations

    def _check_operations(self, output: SymbolicOutput) -> list[str]:
        violations = []
        entity_ids = {e.id for e in output.entities}

        for op in output.operations:
            # For Query operations, check that entity id exists
            if op.op_type == "Query":
                # Extract entity id from "Query(entity_id, attr)"
                m = re.match(r"Query\((\w+),", op.expression)
                if m and m.group(1) not in entity_ids:
                    violations.append(
                        f"Operation references unknown entity: {m.group(1)!r} in {op.expression!r}"
                    )

        return violations

    def _check_facts(self, facts: list[Fact]) -> list[str]:
        violations = []
        seen = set()
        for fact in facts:
            if not self.schema.is_valid_fact(fact.name):
                violations.append(f"Unknown fact: {fact.name!r} (not in vocabulary)")
            if fact.name in seen:
                violations.append(f"Duplicate fact: {fact.name!r}")
            seen.add(fact.name)
        return violations

    def _check_rules(self, output: SymbolicOutput) -> list[str]:
        violations = []
        fact_names = {f.name for f in output.facts}

        for i, rule in enumerate(output.rules):
            # Check conditions reference existing facts
            # NOT conditions for absent facts are OK (implicitly False)
            for cond_name, cond_value in rule.conditions:
                if cond_name not in fact_names and cond_value:
                    violations.append(
                        f"Rule {i+1} references fact '{cond_name}' not in FACTS section"
                    )

            # Check action validity
            if not self.schema.is_valid_action(rule.lateral_action, rule.longitudinal_action):
                violations.append(
                    f"Rule {i+1} has invalid action: {rule.lateral_action}, {rule.longitudinal_action}"
                )

        return violations

    # Longitudinal actions treated as equivalent for rule matching
    _DECEL_CLASS = frozenset({"Deceleration", "QuickDeceleration", "DecelerationToZero"})
    _ACCEL_CLASS = frozenset({"Acceleration", "QuickAcceleration"})

    @classmethod
    def _longitudinal_match(cls, rule_lon: str, selected_lon: str) -> bool:
        """Return True if rule and selected longitudinal actions are equivalent."""
        if rule_lon == selected_lon:
            return True
        if rule_lon in cls._DECEL_CLASS and selected_lon in cls._DECEL_CLASS:
            return True
        if rule_lon in cls._ACCEL_CLASS and selected_lon in cls._ACCEL_CLASS:
            return True
        return False

    def _check_action(self, output: SymbolicOutput) -> list[str]:
        violations = []

        # Valid action
        if not self.schema.is_valid_action(output.selected_lateral, output.selected_longitudinal):
            violations.append(
                f"Invalid selected action: {output.selected_lateral}, {output.selected_longitudinal}"
            )

        # Consistent with at least one rule (with longitudinal equivalence classes)
        if output.rules:
            matched = any(
                r.lateral_action == output.selected_lateral
                and self._longitudinal_match(r.longitudinal_action, output.selected_longitudinal)
                for r in output.rules
            )
            if not matched:
                violations.append(
                    f"Selected action ({output.selected_lateral}, {output.selected_longitudinal}) "
                    f"does not match any rule"
                )

        return violations

    # --- Grounding checks ---

    def _check_grounding(self, output: SymbolicOutput) -> list[str]:
        """Check that True facts are supported by entities/operations."""
        warnings = []
        entity_index = self._build_entity_index(output.entities)
        ego_ops = self._extract_ego_ops(output.operations)

        for fact in output.facts:
            grounding = self.schema.get_fact_grounding(fact.name)
            if grounding is None:
                continue  # no grounding defined — skip

            if not fact.value:
                continue  # only check True facts

            satisfied = self._evaluate_grounding(
                grounding.conditions, entity_index, ego_ops
            )
            if not satisfied:
                warnings.append(
                    f"Grounding: {fact.name} = True but no supporting evidence. "
                    f"Expected: {grounding.description}"
                )

        return warnings

    @staticmethod
    def _build_entity_index(entities: list[Entity]) -> dict[str, list[Entity]]:
        """Index entities by base_type for fast lookup."""
        index: dict[str, list[Entity]] = {}
        for e in entities:
            index.setdefault(e.base_type, []).append(e)
        return index

    @staticmethod
    def _extract_ego_ops(operations: list[Operation]) -> dict[str, str]:
        """Extract EgoQuery results as {attribute: value}."""
        result = {}
        for op in operations:
            if op.op_type == "EgoQuery":
                m = re.match(r"EgoQuery\((\w+)\)", op.expression)
                if m:
                    result[m.group(1)] = op.result
        return result

    def _evaluate_grounding(
        self,
        conditions: list[GroundingCondition],
        entity_index: dict[str, list[Entity]],
        ego_ops: dict[str, str],
    ) -> bool:
        """True if ANY condition branch (OR) is satisfied."""
        for cond in conditions:
            if cond.kind == "judgment":
                return True
            elif cond.kind == "ego":
                if ego_ops.get(cond.attribute) == cond.value:
                    return True
            elif cond.kind == "entity":
                if self._entity_condition_matches(cond, entity_index):
                    return True
            elif cond.kind == "no_entity":
                if not self._entity_condition_matches(cond, entity_index):
                    return True
        return False

    @staticmethod
    def _entity_condition_matches(
        cond: GroundingCondition,
        entity_index: dict[str, list[Entity]],
    ) -> bool:
        """Check if any entity satisfies this condition."""
        candidates = entity_index.get(cond.base_type, [])
        for e in candidates:
            if cond.subtype and e.entity_type != cond.subtype:
                continue
            if not all(e.attributes.get(k) == v for k, v in cond.attributes.items()):
                continue
            if not all(
                e.attributes.get(k) in vs for k, vs in cond.attr_in.items()
            ):
                continue
            return True
        return False


# ---------------------------------------------------------------------------
# Complexity scorer (for GRPO reward)
# ---------------------------------------------------------------------------

def compute_symbolic_complexity(
    think_text: str,
    schema: SymbolicSchema | str | Path,
) -> dict:
    """Compute complexity, validity, and grounding metrics from model output.

    Returns dict with:
        parseable: bool
        valid: bool
        num_entities: int
        num_operations: int
        num_facts: int
        num_rules: int
        num_conditions: int
        action_consistent: bool
        violations: list[str]
        grounding_warnings: list[str]
        stage_completeness: float (0~1, fraction of 4 stages present)
        complexity_score: float (0~1, higher = more verbose)
        grounding_score: float (0~1, fraction of True facts with grounding support)
        ungrounded_facts: list[str]
        judgment_facts: list[str]
    """
    if isinstance(schema, (str, Path)):
        schema = SymbolicSchema(schema)

    parser = SymbolicParser(schema)
    validator = SymbolicValidator(schema, grounding_strictness="warn")

    result = {
        "parseable": False,
        "valid": False,
        "num_entities": 0,
        "num_operations": 0,
        "num_facts": 0,
        "num_rules": 0,
        "num_conditions": 0,
        "action_consistent": False,
        "violations": [],
        "grounding_warnings": [],
        "stage_completeness": 0.0,
        "complexity_score": 1.0,  # max penalty if not parseable
        "grounding_score": 0.0,
        "ungrounded_facts": [],
        "judgment_facts": [],
    }

    try:
        output = parser.parse(think_text)
        result["parseable"] = True
    except ParseError as e:
        result["violations"] = [str(e)]
        return result

    # Validate (includes grounding in warn mode)
    is_valid, violations, grounding_warnings = validator.validate(output)
    result["valid"] = is_valid
    result["violations"] = violations
    result["grounding_warnings"] = grounding_warnings

    # Counts
    result["num_entities"] = len(output.entities)
    result["num_operations"] = len(output.operations)
    result["num_facts"] = len(output.facts)
    result["num_rules"] = len(output.rules)
    result["num_conditions"] = sum(len(r.conditions) for r in output.rules)

    # Action consistency
    for rule in output.rules:
        if (rule.lateral_action == output.selected_lateral
                and rule.longitudinal_action == output.selected_longitudinal):
            result["action_consistent"] = True
            break

    # Stage completeness: how many of the 4 stages have content
    stages_present = sum([
        len(output.entities) > 0,
        len(output.operations) > 0,
        len(output.facts) > 0,
        len(output.rules) > 0,
    ])
    result["stage_completeness"] = stages_present / 4.0

    # Complexity score (0 = minimal, 1 = very verbose)
    max_entities = 6
    max_operations = 8
    max_facts = 10
    max_conditions = 6

    e_ratio = min(result["num_entities"] / max_entities, 1.0)
    o_ratio = min(result["num_operations"] / max_operations, 1.0)
    f_ratio = min(result["num_facts"] / max_facts, 1.0)
    c_ratio = min(result["num_conditions"] / max_conditions, 1.0)
    result["complexity_score"] = 0.25 * e_ratio + 0.25 * o_ratio + 0.25 * f_ratio + 0.25 * c_ratio

    # Grounding score
    entity_index = SymbolicValidator._build_entity_index(output.entities)
    ego_ops = SymbolicValidator._extract_ego_ops(output.operations)
    grounded_count = 0
    checkable_count = 0
    ungrounded_facts = []
    judgment_facts = []

    for fact in output.facts:
        if not fact.value:
            continue
        grounding = schema.get_fact_grounding(fact.name)
        if grounding is None:
            continue  # no definition — skip
        checkable_count += 1
        if any(c.kind == "judgment" for c in grounding.conditions):
            judgment_facts.append(fact.name)
            grounded_count += 1
            continue
        satisfied = validator._evaluate_grounding(
            grounding.conditions, entity_index, ego_ops
        )
        if satisfied:
            grounded_count += 1
        else:
            ungrounded_facts.append(fact.name)

    result["grounding_score"] = (
        grounded_count / checkable_count if checkable_count > 0 else 1.0
    )
    result["ungrounded_facts"] = ungrounded_facts
    result["judgment_facts"] = judgment_facts

    return result
