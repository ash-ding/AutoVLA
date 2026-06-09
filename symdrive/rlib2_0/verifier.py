"""
Symbolic CoT verifier — rlib2.0 (Datalog¬ + arithmetic, 5-layer check).

This module exposes two API surfaces:

(A) The pipeline-facing classes that match the rlib1.x API so that
    tools/preprocessing/symbolic_cot_sample_generation.py can dispatch by
    ``cot_style`` without branching:
      - SymbolicSchema(rlib_dir)       — loads entities.yaml + operations.yaml +
                                          facts.yaml + actions.yaml
      - SymbolicParser(schema)         — exposes .parse(text) → SymbolicOutput
      - SymbolicValidator(schema, ...) — exposes .validate(out) → (is_valid,
                                          violations, grounding_warnings)
      - ParseError                     — re-exported
      - compute_symbolic_complexity    — token-count style metric

(B) The 5-layer rlib2.0 check pipeline, exposed as a single function:
      - verify(cot_text, rlib_dir) → dict {L1..L5: bool, score: float, error}

Layer semantics (rlib2.0-specific):
  L1  syntax       — 5 bracketed sections present, lines parse into AST
  L2  references   — every entity/op/fact name referenced is declared earlier
  L3  arithmetic   — each OPERATIONS and FACTS `:=` right side evaluates to the
                     declared value (numeric tolerance 1e-3 relative, exact for bool)
  L4  rule body    — Z3 propositional check: FACTS truth assignment satisfies
                     the RULES body
  L5  action match — ACTION tuple == RULES head tuple

Surface syntax (Option C):
  [PERCEPTION]
    id = Type {attr: value, attr: value, ...}
    ego = Ego {speed: 7.8, accel: -0.5, instruction: KeepForward}
  [OPERATIONS]                            # optional
    name := <expr> = <numeric_or_set_result>
  [FACTS]
    name := <expr> → True|False
  [RULES]
    R: atom ∧ ¬atom ∧ atom → (LateralAction, LongitudinalAction)
  [ACTION]
    LateralAction, LongitudinalAction
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Exception type — re-exported for pipeline parity with rlib1.x
# ---------------------------------------------------------------------------

class ParseError(ValueError):
    """Raised when CoT text cannot be parsed into the rlib2.0 5-stage AST."""


# ---------------------------------------------------------------------------
# Dataclasses — pipeline-compatible shapes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    id: str
    entity_type: str
    attributes: dict[str, Any]  # raw string or coerced to float when numeric

    def __str__(self) -> str:
        attrs = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
        return f"{self.id} = {self.entity_type} {{{attrs}}}"


@dataclass
class Operation:
    """rlib2.0 OPERATIONS line: name := expr = result."""
    name: str
    expression: str           # raw text of `:=` right side, before `=`
    declared_result: Any      # number, bool, or set-as-frozenset(str)

    # rlib1.x compat shim — used by compute_symbolic_complexity
    op_type: str = "derive"
    result: str = ""

    def __post_init__(self):
        if not self.result:
            self.result = str(self.declared_result)


@dataclass
class Fact:
    name: str
    value: bool
    expression: str  # raw `:=` right side

    def __str__(self) -> str:
        return f"{self.name} := {self.expression} → {self.value}"


@dataclass
class Rule:
    """Single rlib2.0 rule. body = list of (atom_name, expected_value)."""
    conditions: list[tuple[str, bool]]
    lateral_action: str
    longitudinal_action: str

    def __str__(self) -> str:
        parts = [f"{'¬' if not v else ''}{n}" for n, v in self.conditions]
        return f"{' ∧ '.join(parts)} → ({self.lateral_action}, {self.longitudinal_action})"


@dataclass
class SymbolicOutput:
    entities: list[Entity] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    selected_lateral: str = ""
    selected_longitudinal: str = ""


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

class SymbolicSchema:
    """Light schema loader for rlib2.0.

    Only used for L2 (cross-reference) sanity checks. Most validation lives in
    L3 (arithmetic eval) and L4 (Z3 propositional), neither of which needs
    schema beyond the entity-type whitelist and the action vocabulary.
    """

    def __init__(self, rlib_dir: str | Path):
        rlib = Path(rlib_dir)

        with open(rlib / "entities.yaml") as f:
            entities = yaml.safe_load(f) or {}
        with open(rlib / "actions.yaml") as f:
            actions = yaml.safe_load(f) or {}

        # Optional files
        facts_path = rlib / "facts.yaml"
        ops_path = rlib / "operations.yaml"
        facts_file = yaml.safe_load(open(facts_path)) if facts_path.exists() else {}
        ops_file = yaml.safe_load(open(ops_path)) if ops_path.exists() else {}

        self.entities_cfg: dict = entities
        self.operations_cfg: dict = ops_file
        self.actions_cfg: dict = actions

        # Build whitelist of valid entity types (base + subtypes)
        self._all_types: set[str] = set()
        self._subtype_to_base: dict[str, str] = {}
        self._type_attribute_spec: dict[str, dict[str, dict]] = {}
        for base_type, type_cfg in entities.items():
            self._all_types.add(base_type)
            self._subtype_to_base[base_type] = base_type
            self._type_attribute_spec[base_type] = type_cfg.get("attributes", {})
            for subtype in type_cfg.get("subtypes", []) or []:
                self._all_types.add(subtype)
                self._subtype_to_base[subtype] = base_type

        self.lateral_actions: set[str] = set(actions.get("lateral", []))
        self.longitudinal_actions: set[str] = set(actions.get("longitudinal", []))

        # Soft fact vocabulary — names only (not enforced)
        self.fact_suggested_vocabulary: set[str] = {
            entry["name"] for entry in facts_file.get("vocabulary", [])
        }

        # Operations grammar
        self.allowed_arith_ops: set[str] = set(
            ops_file.get("allowed_operators", {}).get("arithmetic", [])
        )
        self.allowed_compare_ops: set[str] = set(
            ops_file.get("allowed_operators", {}).get("comparison", [])
        )
        self.allowed_functions: set[str] = set(
            ops_file.get("allowed_functions", {}).keys()
        )

    def get_base_type(self, type_name: str) -> Optional[str]:
        return self._subtype_to_base.get(type_name)

    def is_valid_lateral(self, name: str) -> bool:
        return name in self.lateral_actions

    def is_valid_longitudinal(self, name: str) -> bool:
        return name in self.longitudinal_actions


# ---------------------------------------------------------------------------
# Parsing — L1
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\[\s*(PERCEPTION|OPERATIONS|FACTS|RULES|ACTION)\s*\]\s*\n?",
    re.IGNORECASE,
)

# id = Type {attr: val, attr: val}
_ENTITY_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\{(.*)\}\s*$"
)

# name := <expr> = <result>     (operations)
_OPERATION_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*:=\s*(.+?)\s*=\s*(.+?)\s*$"
)

# name := <expr> → True|False    (facts)
_FACT_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*:=\s*(.+?)\s*(?:→|->)\s*(True|False)\s*$"
)

# Rule:  R[<id>]: body → (lat, lon)
_RULE_RE = re.compile(
    r"^\s*(?:R(?:[\w]*)?\s*:\s*)?(.+?)\s*(?:→|->)\s*\(?\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)?\s*$"
)


def _normalize_logical_ops(s: str) -> str:
    """Normalize ∧/∨/¬/AND/OR/NOT into a canonical form."""
    s = s.replace("∧", " AND ")
    s = s.replace("∨", " OR ")
    s = s.replace("¬", " NOT ")
    # collapse whitespace
    return re.sub(r"\s+", " ", s).strip()


def _tokenize_rule_body(body: str) -> list[tuple[str, bool]]:
    """Tokenize a rule body of the form `atom AND NOT atom AND atom` into
    [(atom_name, expected_value), ...].

    Only AND is supported (single conjunctive clause). OR would split into
    multiple rules, which rlib2.0 disallows (single-rule design).
    Raises ParseError on OR or on malformed atoms.
    """
    body = _normalize_logical_ops(body)
    if " OR " in body:
        raise ParseError("rlib2.0 RULES body must be a single conjunction; OR is not allowed")
    parts = [p.strip() for p in body.split(" AND ") if p.strip()]
    out: list[tuple[str, bool]] = []
    for p in parts:
        if p.upper().startswith("NOT "):
            atom = p[4:].strip()
            value = False
        else:
            atom = p
            value = True
        m = re.match(r"^([A-Za-z_]\w*)$", atom)
        if not m:
            raise ParseError(
                f"rlib2.0 RULES body atom must be a bare identifier (no arithmetic); got {atom!r}"
            )
        out.append((m.group(1), value))
    return out


class SymbolicParser:
    """rlib2.0 5-stage parser (L1)."""

    def __init__(self, schema: SymbolicSchema | str | Path):
        if isinstance(schema, (str, Path)):
            schema = SymbolicSchema(schema)
        self.schema = schema

    # -- pipeline-compatible API --

    def parse(self, text: str) -> SymbolicOutput:
        """Top-level entry; raises ParseError on syntax failure."""
        text = text.strip()
        sections = self._split_sections(text)

        entities = self._parse_perception(sections.get("PERCEPTION", ""))
        operations = self._parse_operations(sections.get("OPERATIONS", ""))
        facts = self._parse_facts(sections.get("FACTS", ""))
        rules = self._parse_rules(sections.get("RULES", ""))
        lat, lon = self._parse_action(sections.get("ACTION", ""))

        return SymbolicOutput(
            entities=entities,
            operations=operations,
            facts=facts,
            rules=rules,
            selected_lateral=lat,
            selected_longitudinal=lon,
        )

    # -- internals --

    def _split_sections(self, text: str) -> dict[str, str]:
        # Find all bracketed section headers
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            raise ParseError(
                "No bracketed section headers found "
                "(expected [PERCEPTION], [FACTS], [RULES], [ACTION]; [OPERATIONS] optional)"
            )
        sections: dict[str, str] = {}
        for i, m in enumerate(matches):
            name = m.group(1).upper()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[name] = text[start:end].strip()
        return sections

    def _parse_perception(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        if not text:
            return entities
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = _ENTITY_RE.match(line)
            if not m:
                raise ParseError(f"Cannot parse entity line: {line!r}")
            eid, etype, attrs_str = m.group(1), m.group(2), m.group(3)
            attributes = self._parse_attribute_dict(attrs_str, eid)
            entities.append(Entity(id=eid, entity_type=etype, attributes=attributes))
        return entities

    @staticmethod
    def _parse_attribute_dict(attrs_str: str, eid: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        # naive comma split — values are simple scalars or single tokens here
        for pair in _split_top_level_commas(attrs_str):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise ParseError(f"Invalid attribute format in entity {eid!r}: {pair!r}")
            key, val = pair.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Try numeric coercion, then bool, else keep string
            out[key] = _coerce_scalar(val)
        return out

    def _parse_operations(self, text: str) -> list[Operation]:
        ops: list[Operation] = []
        if not text:
            return ops
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = _OPERATION_RE.match(line)
            if not m:
                raise ParseError(f"Cannot parse OPERATIONS line: {line!r}")
            name, expr, result_str = m.group(1), m.group(2), m.group(3)
            ops.append(Operation(
                name=name,
                expression=expr.strip(),
                declared_result=_coerce_scalar(result_str.strip()),
            ))
        return ops

    def _parse_facts(self, text: str) -> list[Fact]:
        facts: list[Fact] = []
        if not text:
            return facts
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = _FACT_RE.match(line)
            if not m:
                raise ParseError(f"Cannot parse FACTS line: {line!r}")
            name, expr, val = m.group(1), m.group(2).strip(), m.group(3)
            facts.append(Fact(name=name, expression=expr, value=(val == "True")))
        return facts

    def _parse_rules(self, text: str) -> list[Rule]:
        rules: list[Rule] = []
        if not text:
            return rules
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = _RULE_RE.match(line)
            if not m:
                raise ParseError(f"Cannot parse RULES line: {line!r}")
            body, lat, lon = m.group(1), m.group(2), m.group(3)
            conditions = _tokenize_rule_body(body)
            rules.append(Rule(
                conditions=conditions,
                lateral_action=lat,
                longitudinal_action=lon,
            ))
        return rules

    def _parse_action(self, text: str) -> tuple[str, str]:
        if not text:
            raise ParseError("Empty ACTION section")
        line = text.strip().splitlines()[0].strip().strip("()")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ParseError(f"ACTION must be 'Lateral, Longitudinal'; got {line!r}")
        return parts[0], parts[1]


def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas not inside braces/brackets/parens."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _coerce_scalar(val: str) -> Any:
    """Best-effort numeric/bool coercion; falls back to stripped string."""
    if val.startswith('"') and val.endswith('"') and len(val) >= 2:
        return val[1:-1]
    if val in ("True", "true"):
        return True
    if val in ("False", "false"):
        return False
    try:
        f = float(val)
        if f.is_integer() and "." not in val and "e" not in val.lower():
            return int(f)
        return f
    except ValueError:
        return val


# ---------------------------------------------------------------------------
# L3 arithmetic eval — restricted AST evaluator
# ---------------------------------------------------------------------------

_ALLOWED_BIN_OPS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
}
_ALLOWED_UNARY_OPS = {ast.UAdd, ast.USub, ast.Not}
_ALLOWED_BOOL_OPS = {ast.And, ast.Or}
_ALLOWED_CMP_OPS = {ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq}
_WHITELISTED_FUNCS = {
    "abs":   abs,
    "min":   min,
    "max":   max,
    "sum":   sum,
    "sqrt":  math.sqrt,
    "count": lambda x: len(x) if hasattr(x, "__len__") else int(x),
}


def _expr_to_python(expr: str) -> str:
    """Translate rlib2.0 surface symbols to Python equivalents for ast.parse."""
    s = expr
    s = s.replace("∧", " and ")
    s = s.replace("∨", " or ")
    s = s.replace("¬", " not ")
    # 'AND/OR/NOT' in upper text → lowercase Python keywords
    s = re.sub(r"\bAND\b", "and", s)
    s = re.sub(r"\bOR\b",  "or",  s)
    s = re.sub(r"\bNOT\b", "not", s)
    return s


def _eval_expr(expr: str, bindings: dict[str, Any]) -> Any:
    """Safely evaluate an arithmetic/boolean expression against `bindings`.

    Bindings:
      - Plain identifiers (e.g. `lead_close`) → bool/number from FACTS/OPERATIONS
      - `entity.attr` access → resolved via bindings[entity][attr]
    """
    py_src = _expr_to_python(expr)
    try:
        tree = ast.parse(py_src, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"syntax error in {expr!r}: {e}") from e

    return _eval_node(tree.body, bindings)


def _eval_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"undefined identifier {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.Attribute):
        # e.g. v_1.dist  ⇒  env['v_1']['dist']
        owner = _eval_node(node.value, env)
        if not isinstance(owner, dict):
            raise ValueError(f"{node.attr!r}: owner is not an entity dict")
        if node.attr not in owner:
            raise ValueError(f"entity missing attribute {node.attr!r}")
        return owner[node.attr]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        v = _eval_node(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.Not):
            return not v
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        a = _eval_node(node.left, env)
        b = _eval_node(node.right, env)
        return _apply_binop(node.op, a, b)
    if isinstance(node, ast.BoolOp) and type(node.op) in _ALLOWED_BOOL_OPS:
        vals = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(vals)
        return any(vals)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _ALLOWED_CMP_OPS:
                raise ValueError(f"comparison op not allowed: {type(op).__name__}")
            right = _eval_node(comparator, env)
            if not _apply_cmpop(op, left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only top-level whitelisted function calls allowed")
        fn = node.func.id
        if fn not in _WHITELISTED_FUNCS:
            raise ValueError(f"function {fn!r} is not whitelisted")
        args = [_eval_node(a, env) for a in node.args]
        return _WHITELISTED_FUNCS[fn](*args)
    raise ValueError(f"unsupported AST node: {type(node).__name__}")


def _apply_binop(op, a, b):
    if isinstance(op, ast.Add):  return a + b
    if isinstance(op, ast.Sub):  return a - b
    if isinstance(op, ast.Mult): return a * b
    if isinstance(op, ast.Div):
        if b == 0:
            raise ValueError("division by zero")
        return a / b
    if isinstance(op, ast.Pow):  return a ** b
    if isinstance(op, ast.Mod):  return a % b
    if isinstance(op, ast.FloorDiv): return a // b
    raise ValueError(f"unhandled binop: {type(op).__name__}")


def _apply_cmpop(op, a, b):
    if isinstance(op, ast.Lt):    return a < b
    if isinstance(op, ast.LtE):   return a <= b
    if isinstance(op, ast.Gt):    return a > b
    if isinstance(op, ast.GtE):   return a >= b
    if isinstance(op, ast.Eq):    return a == b
    if isinstance(op, ast.NotEq): return a != b
    raise ValueError(f"unhandled cmp op: {type(op).__name__}")


# ---------------------------------------------------------------------------
# Validator — wraps layered checks behind the rlib1.x API
# ---------------------------------------------------------------------------

# Tolerance for L3 numerical comparisons. LLM outputs commonly round to 1-2
# decimals (e.g. declares 0.11 when the true value is 0.1117), so we use a 5%
# relative tolerance with a 0.05 absolute floor for values near zero.
_REL_TOLERANCE = 5e-2
_ABS_FLOOR = 5e-2


def _values_match(declared: Any, computed: Any) -> bool:
    if isinstance(declared, bool) or isinstance(computed, bool):
        return bool(declared) == bool(computed)
    if isinstance(declared, (int, float)) and isinstance(computed, (int, float)):
        diff = abs(declared - computed)
        if diff < _ABS_FLOOR:
            return True
        return diff / max(1e-12, abs(computed)) < _REL_TOLERANCE
    return declared == computed


def _bindings_from_perception(entities: list[Entity]) -> dict[str, Any]:
    """Expose entities as `env['v_1']` → {'dist': 8.3, ...} for `v_1.dist`."""
    return {e.id: dict(e.attributes) for e in entities}


def _check_grounding_non_trivial(fact_expr: str, fact_name: str) -> Optional[str]:
    """Reject `name := True`, `name := False`, or numeric-literal-only bodies.

    Returns an error string or None if OK. An expression is considered trivial
    when its parse tree contains zero Name or Attribute nodes.
    """
    try:
        tree = ast.parse(_expr_to_python(fact_expr), mode="eval")
    except SyntaxError as e:
        return f"FACTS[{fact_name}]: parse error in expression: {e}"
    has_ref = any(isinstance(n, (ast.Name, ast.Attribute)) for n in ast.walk(tree))
    if not has_ref:
        return f"FACTS[{fact_name}]: expression {fact_expr!r} is a bare literal — must reference PERCEPTION or OPERATIONS"
    return None


class SymbolicValidator:
    """rlib1.x-shaped wrapper around the 5-layer rlib2.0 check.

    The pipeline calls .validate(output) and expects
    (is_valid, violations, grounding_warnings) — we map the L1-L5 outcome into
    that 3-tuple. L1 failures are surfaced as ParseError by the parser; this
    validator handles L2-L5.
    """

    def __init__(
        self,
        schema: SymbolicSchema,
        grounding_strictness: str = "warn",  # API parity
        strict_action_match: bool = True,    # API parity
    ):
        self.schema = schema
        self.grounding_strictness = grounding_strictness
        self.strict_action_match = strict_action_match

    def validate(
        self, output: SymbolicOutput
    ) -> tuple[bool, list[str], list[str]]:
        violations: list[str] = []
        warnings: list[str] = []

        # ---- L2 cross-reference ----
        decl_entity_ids = {e.id for e in output.entities}
        op_names = [o.name for o in output.operations]
        fact_names = [f.name for f in output.facts]
        valid_ref_for_fact = set(decl_entity_ids) | set(op_names)

        for e in output.entities:
            base = self.schema.get_base_type(e.entity_type)
            if base is None:
                violations.append(
                    f"PERCEPTION[{e.id}]: unknown entity type {e.entity_type!r}"
                )

        # OPERATIONS may only reference earlier OPERATIONS + PERCEPTION entities
        seen_ops: list[str] = []
        for op in output.operations:
            refs = _collect_identifiers(op.expression)
            for ref in refs:
                if ref in decl_entity_ids or ref in seen_ops or ref in _WHITELISTED_FUNCS:
                    continue
                violations.append(
                    f"OPERATIONS[{op.name}]: identifier {ref!r} not declared in PERCEPTION or earlier OPERATIONS"
                )
            seen_ops.append(op.name)

        # FACTS may reference PERCEPTION entities or OPERATIONS names
        seen_facts: list[str] = []
        for f in output.facts:
            refs = _collect_identifiers(f.expression)
            for ref in refs:
                if (
                    ref in decl_entity_ids
                    or ref in op_names
                    or ref in seen_facts
                    or ref in _WHITELISTED_FUNCS
                ):
                    continue
                violations.append(
                    f"FACTS[{f.name}]: identifier {ref!r} not declared in PERCEPTION, OPERATIONS, or earlier FACTS"
                )
            # Non-trivial grounding requirement
            err = _check_grounding_non_trivial(f.expression, f.name)
            if err:
                violations.append(err)
            seen_facts.append(f.name)

        # ---- L3 arithmetic eval ----
        ent_env = _bindings_from_perception(output.entities)
        env: dict[str, Any] = dict(ent_env)  # entities (dicts) + later op/fact values

        for op in output.operations:
            try:
                computed = _eval_expr(op.expression, env)
            except Exception as e:
                violations.append(f"OPERATIONS[{op.name}]: eval failed — {e}")
                continue
            if not _values_match(op.declared_result, computed):
                violations.append(
                    f"OPERATIONS[{op.name}]: declared {op.declared_result!r} != computed {computed!r}"
                )
            env[op.name] = computed

        for f in output.facts:
            try:
                computed = _eval_expr(f.expression, env)
            except Exception as e:
                violations.append(f"FACTS[{f.name}]: eval failed — {e}")
                env[f.name] = f.value
                continue
            if not _values_match(f.value, computed):
                violations.append(
                    f"FACTS[{f.name}]: declared {f.value} != computed {computed!r}"
                )
            env[f.name] = bool(f.value)  # use the declared value downstream

        # ---- L4 rule body satisfiability (Z3) ----
        if not output.rules:
            violations.append("RULES: no rule declared")
        else:
            if len(output.rules) > 1:
                violations.append(
                    f"RULES: rlib2.0 expects a single rule, got {len(output.rules)}"
                )
            rule = output.rules[0]
            try:
                sat = _z3_check_rule(rule, output.facts)
            except Exception as e:
                violations.append(f"RULES: Z3 check failed — {e}")
                sat = False
            if not sat:
                violations.append(
                    f"RULES: rule body {rule!s} not satisfied by declared FACTS"
                )

            # ---- L5 action match ----
            if (rule.lateral_action != output.selected_lateral
                or rule.longitudinal_action != output.selected_longitudinal):
                violations.append(
                    f"ACTION: ({output.selected_lateral}, {output.selected_longitudinal}) "
                    f"!= rule head ({rule.lateral_action}, {rule.longitudinal_action})"
                )

            # Vocab check on action names
            if rule.lateral_action and not self.schema.is_valid_lateral(rule.lateral_action):
                violations.append(
                    f"ACTION: lateral {rule.lateral_action!r} not in actions.yaml vocab"
                )
            if rule.longitudinal_action and not self.schema.is_valid_longitudinal(rule.longitudinal_action):
                violations.append(
                    f"ACTION: longitudinal {rule.longitudinal_action!r} not in actions.yaml vocab"
                )

        is_valid = (len(violations) == 0)
        return is_valid, violations, warnings


def _collect_identifiers(expr: str) -> set[str]:
    """Return the set of bare-name identifiers used in an expression.

    Distinguishes `v_1.dist` (collects only `v_1`) from `lead_close`
    (collects `lead_close`). Whitelisted function names are NOT filtered here;
    the caller does that.
    """
    try:
        tree = ast.parse(_expr_to_python(expr), mode="eval")
    except SyntaxError:
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            # the chain root is a Name; deeper attribute access is unsupported
            base = n
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                out.add(base.id)
        elif isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _z3_check_rule(rule: Rule, facts: list[Fact]) -> bool:
    """L4: under the declared FACT truth assignment, is the rule body satisfied?

    Implementation: build Z3 Bool symbols for every fact named in the rule body,
    pin each to its declared value from `facts`, assert the rule body, check sat.
    Facts referenced in the rule body but not declared in FACTS are treated as
    False (closed-world / NAF).
    """
    import z3
    solver = z3.Solver()
    fact_vals = {f.name: f.value for f in facts}
    z3_syms: dict[str, Any] = {}
    for atom, _expected in rule.conditions:
        if atom not in z3_syms:
            z3_syms[atom] = z3.Bool(atom)
            solver.add(z3_syms[atom] == fact_vals.get(atom, False))
    body_terms = []
    for atom, expected in rule.conditions:
        sym = z3_syms[atom]
        body_terms.append(sym if expected else z3.Not(sym))
    if not body_terms:
        return True
    solver.add(z3.And(*body_terms))
    return solver.check() == z3.sat


# ---------------------------------------------------------------------------
# Complexity (pipeline parity — used as a "cost" signal in result JSON)
# ---------------------------------------------------------------------------

def compute_symbolic_complexity(output: SymbolicOutput) -> dict[str, int]:
    """Crude token-count style complexity metric."""
    num_entities = len(output.entities)
    num_operations = len(output.operations)
    num_facts = len(output.facts)
    num_rules = len(output.rules)
    num_rule_atoms = sum(len(r.conditions) for r in output.rules)
    return {
        "num_entities":    num_entities,
        "num_operations":  num_operations,
        "num_facts":       num_facts,
        "num_rules":       num_rules,
        "num_rule_atoms":  num_rule_atoms,
        "total":           num_entities + num_operations + num_facts + num_rule_atoms,
    }


# ---------------------------------------------------------------------------
# Standalone single-call verifier returning the L1-L5 dict
# ---------------------------------------------------------------------------

def verify(cot_text: str, rlib_dir: str | Path) -> dict:
    """Run the full L1-L5 check on `cot_text` against the rlib2.0 schema at `rlib_dir`.

    Returns:
        {
          "L1": bool,                 # syntax parse
          "L2": bool,                 # references resolved
          "L3": bool,                 # arithmetic agrees with declared values
          "L4": bool,                 # rule body satisfied by FACTS (Z3)
          "L5": bool,                 # ACTION matches RULES head
          "score": float in [0, 1],   # mean of L1..L5
          "violations": [str, ...],
          "error": str | None,
        }
    """
    schema = SymbolicSchema(rlib_dir)
    parser = SymbolicParser(schema)
    validator = SymbolicValidator(schema)

    result = {"L1": False, "L2": False, "L3": False, "L4": False, "L5": False,
              "score": 0.0, "violations": [], "error": None}

    try:
        output = parser.parse(cot_text)
    except ParseError as e:
        result["error"] = f"L1: {e}"
        result["violations"] = [str(e)]
        return result

    result["L1"] = True
    is_valid, violations, _ = validator.validate(output)
    result["violations"] = violations

    # Bucket each violation into its responsible layer (substring heuristics).
    def _layer_of(v: str) -> int:
        if "ACTION:" in v:
            return 5
        if v.startswith("RULES:"):
            return 4
        if "eval failed" in v or "declared" in v and "computed" in v:
            return 3
        # L2: any reference / grounding / type error
        if ("not declared" in v or "unknown entity type" in v
                or "bare literal" in v or "parse error in expression" in v):
            return 2
        return 2  # default: most remaining checks live in L2

    layer_ok = {2: True, 3: True, 4: True, 5: True}
    for v in violations:
        layer_ok[_layer_of(v)] = False

    for k in range(2, 6):
        result[f"L{k}"] = layer_ok[k]
    result["score"] = sum(int(result[f"L{k}"]) for k in range(1, 6)) / 5.0
    return result
