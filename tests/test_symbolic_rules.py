"""Unit tests for the four-stage symbolic driving rules DSL."""

import pytest
from pathlib import Path

from models.symbolic_rules import (
    Entity, Operation, Fact, Rule, SymbolicOutput, ParseError,
    GroundingCondition, FactGrounding,
    SymbolicSchema, SymbolicParser, SymbolicValidator, compute_symbolic_complexity,
)

SCHEMA_PATH = Path(__file__).parent.parent / "RLIB"


@pytest.fixture
def schema():
    return SymbolicSchema(SCHEMA_PATH)

@pytest.fixture
def parser(schema):
    return SymbolicParser(schema)

@pytest.fixture
def validator(schema):
    return SymbolicValidator(schema)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

FULL_EXAMPLE = """\
PERCEPTION:
  tl_1 = TrafficLight {color: Yellow, position: Front, applies_to: EgoLane}
  v_1 = Car {position: Front, lane: EgoLane, distance: Near, motion: Decelerating, signal: BrakeLights}
  p_1 = Pedestrian {position: FrontRight, distance: Medium, motion: Standing, location: Sidewalk}
  r_1 = Intersection {position: Ahead, distance: VeryNear}

OPERATIONS:
  Query(tl_1, color) = Yellow
  Filter(Vehicle, lane = EgoLane) = {v_1}
  Query(v_1, motion) = Decelerating
  Count(Filter(Pedestrian, location = Crosswalk)) = 0
  EgoQuery(speed) = Medium
  EgoQuery(instruction) = KeepForward

FACTS:
  YellowLight = True
  ApproachingIntersection = True
  LeadVehiclePresent = True
  LeadVehicleClose = True
  LeadVehicleBraking = True
  CanStopComfortably = True
  InstructionKeepForward = True

RULES:
  YellowLight AND ApproachingIntersection AND CanStopComfortably -> KeepLane, Deceleration

ACTION: KeepLane, Deceleration
"""

MINIMAL_EXAMPLE = """\
PERCEPTION:
  r_1 = Intersection {position: Current, distance: VeryNear}

OPERATIONS:
  EgoQuery(speed) = Fast
  EgoQuery(instruction) = KeepForward

FACTS:
  GreenLight = True
  AtIntersection = True
  InstructionKeepForward = True
  PathBlocked = False

RULES:
  GreenLight AND NOT PathBlocked -> KeepLane, ConstantSpeed

ACTION: KeepLane, ConstantSpeed
"""

MULTI_RULE_EXAMPLE = """\
PERCEPTION:
  tl_1 = TrafficLight {color: Red, position: Front, applies_to: EgoLane}
  v_1 = Car {position: Front, lane: EgoLane, distance: Near, motion: Stationary, signal: BrakeLights}
  p_1 = Pedestrian {position: Front, distance: Near, motion: Crossing, location: Crosswalk}

OPERATIONS:
  Query(tl_1, color) = Red
  Query(v_1, motion) = Stationary
  Count(Filter(Pedestrian, location = Crosswalk)) = 1
  EgoQuery(speed) = Slow

FACTS:
  RedLight = True
  LeadVehiclePresent = True
  LeadVehicleStopped = True
  PedestrianCrossing = True
  CrosswalkOccupied = True
  EgoMovingSlow = True
  EgoStopped = False

RULES:
  RedLight AND NOT EgoStopped -> KeepLane, DecelerationToZero
  PedestrianCrossing AND CrosswalkOccupied -> KeepLane, Stop

ACTION: KeepLane, Stop
"""


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSymbolicSchema:
    def test_loads_entities(self, schema):
        assert "Vehicle" in schema.entities_cfg
        assert "Pedestrian" in schema.entities_cfg
        assert "TrafficLight" in schema.entities_cfg

    def test_subtype_mapping(self, schema):
        assert schema.get_base_type("Car") == "Vehicle"
        assert schema.get_base_type("Truck") == "Vehicle"
        assert schema.get_base_type("TrafficLight") == "TrafficLight"
        assert schema.get_base_type("Intersection") == "RoadFeature"
        assert schema.get_base_type("FakeType") is None

    def test_valid_types(self, schema):
        assert schema.is_valid_type("Car")
        assert schema.is_valid_type("Vehicle")
        assert schema.is_valid_type("Pedestrian")
        assert not schema.is_valid_type("Airplane")

    def test_attribute_values(self, schema):
        vals = schema.get_valid_attribute_values("Vehicle", "motion")
        assert "Moving" in vals
        assert "Stationary" in vals

        vals = schema.get_valid_attribute_values("Pedestrian", "location")
        assert "Crosswalk" in vals

    def test_fact_vocabulary(self, schema):
        assert schema.is_valid_fact("YellowLight")
        assert schema.is_valid_fact("LeadVehicleClose")
        assert not schema.is_valid_fact("MadeUpFact")

    def test_actions(self, schema):
        assert schema.is_valid_action("KeepLane", "Deceleration")
        assert schema.is_valid_action("TurnLeft", "Acceleration")
        assert not schema.is_valid_action("Fly", "Deceleration")


# ---------------------------------------------------------------------------
# Schema — grounding tests
# ---------------------------------------------------------------------------

class TestSymbolicSchemaGrounding:
    def test_loads_groundings(self, schema):
        assert schema.has_grounding("RedLight")
        assert schema.has_grounding("EgoStopped")
        assert schema.has_grounding("CanStopComfortably")

    def test_all_facts_have_groundings(self, schema):
        for fact_name in schema.fact_vocabulary:
            assert schema.has_grounding(fact_name), f"Missing grounding for {fact_name}"

    def test_entity_grounding_fields(self, schema):
        g = schema.get_fact_grounding("RedLight")
        assert g.category == "traffic_signal"
        assert len(g.conditions) == 1
        c = g.conditions[0]
        assert c.kind == "entity"
        assert c.base_type == "TrafficLight"
        assert c.attributes["color"] == "Red"
        assert "EgoLane" in c.attr_in["applies_to"]

    def test_or_grounding(self, schema):
        g = schema.get_fact_grounding("LeadVehicleBraking")
        assert len(g.conditions) == 2
        kinds = {c.kind for c in g.conditions}
        assert kinds == {"entity"}

    def test_no_entity_grounding(self, schema):
        g = schema.get_fact_grounding("NoTrafficLight")
        assert g.conditions[0].kind == "no_entity"
        assert g.conditions[0].base_type == "TrafficLight"

    def test_ego_grounding(self, schema):
        g = schema.get_fact_grounding("EgoStopped")
        c = g.conditions[0]
        assert c.kind == "ego"
        assert c.attribute == "speed"
        assert c.value == "Stopped"

    def test_judgment_grounding(self, schema):
        g = schema.get_fact_grounding("CanStopComfortably")
        assert g.conditions[0].kind == "judgment"
        assert len(g.conditions[0].description) > 0

    def test_subtype_grounding(self, schema):
        g = schema.get_fact_grounding("EmergencyVehicleNearby")
        assert g.conditions[0].subtype == "EmergencyVehicle"

    def test_nonexistent_returns_none(self, schema):
        assert schema.get_fact_grounding("NonExistentFact") is None


# ---------------------------------------------------------------------------
# Parser — perception
# ---------------------------------------------------------------------------

class TestParsePerception:
    def test_parse_entities(self, parser):
        output = parser.parse(FULL_EXAMPLE)
        assert len(output.entities) == 4

        tl = output.entities[0]
        assert tl.id == "tl_1"
        assert tl.entity_type == "TrafficLight"
        assert tl.base_type == "TrafficLight"
        assert tl.attributes["color"] == "Yellow"

        car = output.entities[1]
        assert car.id == "v_1"
        assert car.entity_type == "Car"
        assert car.base_type == "Vehicle"
        assert car.attributes["lane"] == "EgoLane"

    def test_road_feature(self, parser):
        output = parser.parse(FULL_EXAMPLE)
        rf = output.entities[3]
        assert rf.entity_type == "Intersection"
        assert rf.base_type == "RoadFeature"

    def test_bad_entity_line(self, parser):
        text = "PERCEPTION:\n  garbage line\nOPERATIONS:\nFACTS:\nRULES:\nACTION: KeepLane, Stop"
        with pytest.raises(ParseError, match="Cannot parse entity"):
            parser.parse(text)

    def test_unknown_type(self, parser):
        text = "PERCEPTION:\n  x_1 = Airplane {position: Front}\nOPERATIONS:\nFACTS:\nRULES:\nACTION: KeepLane, Stop"
        with pytest.raises(ParseError, match="Unknown entity type"):
            parser.parse(text)


# ---------------------------------------------------------------------------
# Parser — operations
# ---------------------------------------------------------------------------

class TestParseOperations:
    def test_parse_ops(self, parser):
        output = parser.parse(FULL_EXAMPLE)
        assert len(output.operations) == 6

        assert output.operations[0].op_type == "Query"
        assert output.operations[0].result == "Yellow"

        assert output.operations[1].op_type == "Filter"
        assert output.operations[4].op_type == "EgoQuery"

    def test_unknown_op(self, parser):
        text = "PERCEPTION:\nOPERATIONS:\n  FooBar(x) = 3\nFACTS:\nRULES:\nACTION: KeepLane, Stop"
        with pytest.raises(ParseError, match="Unknown operation"):
            parser.parse(text)


# ---------------------------------------------------------------------------
# Parser — facts
# ---------------------------------------------------------------------------

class TestParseFacts:
    def test_parse_facts(self, parser):
        output = parser.parse(FULL_EXAMPLE)
        assert len(output.facts) == 7
        assert output.facts[0].name == "YellowLight"
        assert output.facts[0].value is True

    def test_false_fact(self, parser):
        output = parser.parse(MINIMAL_EXAMPLE)
        # All facts in minimal are True — test with explicit False
        text = "PERCEPTION:\nOPERATIONS:\nFACTS:\n  RedLight = False\nRULES:\n  NOT RedLight -> KeepLane, ConstantSpeed\nACTION: KeepLane, ConstantSpeed"
        output = parser.parse(text)
        assert output.facts[0].value is False

    def test_bad_fact_line(self, parser):
        text = "PERCEPTION:\nOPERATIONS:\nFACTS:\n  not a fact\nRULES:\nACTION: KeepLane, Stop"
        with pytest.raises(ParseError, match="Cannot parse fact"):
            parser.parse(text)


# ---------------------------------------------------------------------------
# Parser — rules
# ---------------------------------------------------------------------------

class TestParseRules:
    def test_single_rule(self, parser):
        output = parser.parse(FULL_EXAMPLE)
        assert len(output.rules) == 1
        rule = output.rules[0]
        assert len(rule.conditions) == 3
        assert rule.conditions[0] == ("YellowLight", True)
        assert rule.lateral_action == "KeepLane"
        assert rule.longitudinal_action == "Deceleration"

    def test_not_condition(self, parser):
        output = parser.parse(MINIMAL_EXAMPLE)
        rule = output.rules[0]
        assert ("PathBlocked", False) in rule.conditions

    def test_multi_rules(self, parser):
        output = parser.parse(MULTI_RULE_EXAMPLE)
        assert len(output.rules) == 2

    def test_unicode_arrow(self, parser):
        text = MINIMAL_EXAMPLE.replace("->", "→")
        output = parser.parse(text)
        assert len(output.rules) == 1

    def test_no_arrow(self, parser):
        text = "PERCEPTION:\nOPERATIONS:\nFACTS:\nRULES:\n  A then B\nACTION: KeepLane, Stop"
        with pytest.raises(ParseError, match="missing ->"):
            parser.parse(text)


# ---------------------------------------------------------------------------
# Parser — action
# ---------------------------------------------------------------------------

class TestParseAction:
    def test_parse_action(self, parser):
        output = parser.parse(FULL_EXAMPLE)
        assert output.selected_lateral == "KeepLane"
        assert output.selected_longitudinal == "Deceleration"

    def test_missing_action(self, parser):
        text = "PERCEPTION:\nOPERATIONS:\nFACTS:\nRULES:"
        with pytest.raises(ParseError):
            parser.parse(text)


# ---------------------------------------------------------------------------
# Parser — roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_str_roundtrip(self, parser):
        output1 = parser.parse(FULL_EXAMPLE)
        text2 = str(output1)
        output2 = parser.parse(text2)
        assert len(output1.entities) == len(output2.entities)
        assert len(output1.operations) == len(output2.operations)
        assert len(output1.facts) == len(output2.facts)
        assert len(output1.rules) == len(output2.rules)
        assert output1.selected_lateral == output2.selected_lateral
        assert output1.selected_longitudinal == output2.selected_longitudinal

    def test_entity_roundtrip(self, parser):
        output1 = parser.parse(FULL_EXAMPLE)
        text2 = str(output1)
        output2 = parser.parse(text2)
        for e1, e2 in zip(output1.entities, output2.entities):
            assert e1.id == e2.id
            assert e1.entity_type == e2.entity_type
            assert e1.attributes == e2.attributes


# ---------------------------------------------------------------------------
# Parser — missing/empty sections
# ---------------------------------------------------------------------------

class TestMissingSections:
    def test_no_headers(self, parser):
        with pytest.raises(ParseError, match="No section headers"):
            parser.parse("just some random text")

    def test_empty_perception_ok(self, parser):
        """PERCEPTION can be empty (e.g. empty road)."""
        text = "PERCEPTION:\nOPERATIONS:\n  EgoQuery(speed) = Fast\nFACTS:\n  GreenLight = True\nRULES:\n  GreenLight -> KeepLane, ConstantSpeed\nACTION: KeepLane, ConstantSpeed"
        output = parser.parse(text)
        assert len(output.entities) == 0
        assert len(output.operations) == 1


# ---------------------------------------------------------------------------
# Validator tests (structural)
# ---------------------------------------------------------------------------

class TestValidator:
    def test_valid_full(self, validator, parser):
        output = parser.parse(FULL_EXAMPLE)
        is_valid, violations, _ = validator.validate(output)
        assert is_valid, f"Violations: {violations}"

    def test_valid_minimal(self, validator, parser):
        output = parser.parse(MINIMAL_EXAMPLE)
        is_valid, violations, _ = validator.validate(output)
        assert is_valid, f"Violations: {violations}"

    def test_valid_multi_rule(self, validator, parser):
        output = parser.parse(MULTI_RULE_EXAMPLE)
        is_valid, violations, _ = validator.validate(output)
        assert is_valid, f"Violations: {violations}"

    def test_invalid_attribute_value(self, validator):
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle", {"motion": "Flying"})],
            operations=[], facts=[], rules=[],
            selected_lateral="KeepLane", selected_longitudinal="Stop",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("Invalid value" in v for v in violations)

    def test_unknown_attribute(self, validator):
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle", {"color": "Blue"})],
            operations=[], facts=[], rules=[],
            selected_lateral="KeepLane", selected_longitudinal="Stop",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("Unknown attribute" in v for v in violations)

    def test_unknown_fact(self, validator):
        output = SymbolicOutput(
            entities=[], operations=[],
            facts=[Fact("MadeUpFact", True)],
            rules=[], selected_lateral="KeepLane", selected_longitudinal="Stop",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("not in vocabulary" in v for v in violations)

    def test_duplicate_entity(self, validator):
        output = SymbolicOutput(
            entities=[
                Entity("v_1", "Car", "Vehicle", {"position": "Front"}),
                Entity("v_1", "Truck", "Vehicle", {"position": "Behind"}),
            ],
            operations=[], facts=[], rules=[],
            selected_lateral="KeepLane", selected_longitudinal="Stop",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("Duplicate" in v for v in violations)

    def test_duplicate_fact(self, validator):
        output = SymbolicOutput(
            entities=[], operations=[],
            facts=[Fact("RedLight", True), Fact("RedLight", False)],
            rules=[], selected_lateral="KeepLane", selected_longitudinal="Stop",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("Duplicate fact" in v for v in violations)

    def test_rule_references_missing_fact(self, validator):
        output = SymbolicOutput(
            entities=[], operations=[],
            facts=[Fact("GreenLight", True)],
            rules=[Rule(
                conditions=[("GreenLight", True), ("MissingFact", True)],
                lateral_action="KeepLane", longitudinal_action="ConstantSpeed",
            )],
            selected_lateral="KeepLane", selected_longitudinal="ConstantSpeed",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("MissingFact" in v for v in violations)

    def test_rule_not_condition_implicit_false(self, validator):
        """NOT conditions for facts absent from FACTS should NOT be violations."""
        output = SymbolicOutput(
            entities=[], operations=[],
            facts=[Fact("GreenLight", True)],
            rules=[Rule(
                conditions=[("GreenLight", True), ("PathBlocked", False)],
                lateral_action="KeepLane", longitudinal_action="ConstantSpeed",
            )],
            selected_lateral="KeepLane", selected_longitudinal="ConstantSpeed",
        )
        is_valid, violations, _ = validator.validate(output)
        # PathBlocked is NOT in FACTS but used as NOT PathBlocked — implicitly False, OK
        assert not any("PathBlocked" in v for v in violations)

    def test_deceleration_equivalence(self, validator):
        """Deceleration and QuickDeceleration should match each other for rule consistency."""
        output = SymbolicOutput(
            entities=[], operations=[],
            facts=[Fact("LeadVehicleBraking", True)],
            rules=[Rule(
                conditions=[("LeadVehicleBraking", True)],
                lateral_action="KeepLane", longitudinal_action="Deceleration",
            )],
            selected_lateral="KeepLane", selected_longitudinal="QuickDeceleration",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not any("does not match any rule" in v for v in violations)

    def test_action_mismatch(self, validator, parser):
        output = parser.parse(FULL_EXAMPLE)
        output.selected_lateral = "TurnLeft"  # doesn't match rule
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("does not match" in v for v in violations)

    def test_invalid_action(self, validator):
        output = SymbolicOutput(
            entities=[], operations=[],
            facts=[], rules=[],
            selected_lateral="Fly", selected_longitudinal="Hover",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("Invalid selected action" in v for v in violations)

    def test_operation_references_missing_entity(self, validator):
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle", {"position": "Front"})],
            operations=[Operation("Query", "Query(v_99, motion)", "Moving")],
            facts=[], rules=[],
            selected_lateral="KeepLane", selected_longitudinal="Stop",
        )
        is_valid, violations, _ = validator.validate(output)
        assert not is_valid
        assert any("unknown entity" in v for v in violations)


# ---------------------------------------------------------------------------
# Grounding validation tests
# ---------------------------------------------------------------------------

class TestGroundingValidation:
    @pytest.fixture
    def warn_validator(self, schema):
        return SymbolicValidator(schema, grounding_strictness="warn")

    @pytest.fixture
    def strict_validator(self, schema):
        return SymbolicValidator(schema, grounding_strictness="strict")

    @pytest.fixture
    def none_validator(self, schema):
        return SymbolicValidator(schema, grounding_strictness="none")

    # --- Entity grounding ---

    def test_entity_grounded_fact_no_warning(self, warn_validator, parser):
        """YellowLight = True is grounded by tl_1 with color=Yellow."""
        output = parser.parse(FULL_EXAMPLE)
        _, _, warnings = warn_validator.validate(output)
        assert not any("YellowLight" in w for w in warnings)

    def test_entity_ungrounded_fact_warns(self, warn_validator):
        """RedLight = True with no TrafficLight entity should warn."""
        output = SymbolicOutput(
            entities=[],
            operations=[],
            facts=[Fact("RedLight", True)],
            rules=[Rule([("RedLight", True)], "KeepLane", "Stop")],
            selected_lateral="KeepLane",
            selected_longitudinal="Stop",
        )
        is_valid, violations, warnings = warn_validator.validate(output)
        assert is_valid  # warn mode: structural validity unaffected
        assert any("RedLight" in w for w in warnings)

    def test_entity_ungrounded_fact_fails_strict(self, strict_validator):
        """In strict mode, ungrounded fact becomes a violation."""
        output = SymbolicOutput(
            entities=[],
            operations=[],
            facts=[Fact("RedLight", True)],
            rules=[Rule([("RedLight", True)], "KeepLane", "Stop")],
            selected_lateral="KeepLane",
            selected_longitudinal="Stop",
        )
        is_valid, violations, _ = strict_validator.validate(output)
        assert not is_valid
        assert any("RedLight" in v for v in violations)

    def test_grounding_none_mode_skips(self, none_validator):
        """grounding_strictness='none' does not check grounding."""
        output = SymbolicOutput(
            entities=[],
            operations=[],
            facts=[Fact("RedLight", True)],
            rules=[Rule([("RedLight", True)], "KeepLane", "Stop")],
            selected_lateral="KeepLane",
            selected_longitudinal="Stop",
        )
        _, _, warnings = none_validator.validate(output)
        assert len(warnings) == 0

    # --- no_entity grounding ---

    def test_no_entity_satisfied(self, warn_validator):
        """NoTrafficLight = True when no TrafficLight present — should pass."""
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle",
                             {"position": "Front", "lane": "EgoLane", "distance": "Near", "motion": "Moving"})],
            operations=[],
            facts=[Fact("NoTrafficLight", True), Fact("LeadVehiclePresent", True)],
            rules=[Rule([("NoTrafficLight", True)], "KeepLane", "ConstantSpeed")],
            selected_lateral="KeepLane",
            selected_longitudinal="ConstantSpeed",
        )
        _, _, warnings = warn_validator.validate(output)
        assert not any("NoTrafficLight" in w for w in warnings)

    def test_no_entity_violated(self, warn_validator):
        """NoTrafficLight = True but a TrafficLight IS present — contradiction."""
        output = SymbolicOutput(
            entities=[Entity("tl_1", "TrafficLight", "TrafficLight",
                             {"color": "Red", "position": "Front", "applies_to": "EgoLane"})],
            operations=[],
            facts=[Fact("NoTrafficLight", True)],
            rules=[Rule([("NoTrafficLight", True)], "KeepLane", "ConstantSpeed")],
            selected_lateral="KeepLane",
            selected_longitudinal="ConstantSpeed",
        )
        _, _, warnings = warn_validator.validate(output)
        assert any("NoTrafficLight" in w for w in warnings)

    # --- ego grounding ---

    def test_ego_grounded(self, warn_validator):
        """EgoStopped = True with EgoQuery(speed) = Stopped — should pass."""
        output = SymbolicOutput(
            entities=[],
            operations=[Operation("EgoQuery", "EgoQuery(speed)", "Stopped")],
            facts=[Fact("EgoStopped", True)],
            rules=[Rule([("EgoStopped", True)], "KeepLane", "Stop")],
            selected_lateral="KeepLane",
            selected_longitudinal="Stop",
        )
        _, _, warnings = warn_validator.validate(output)
        assert not any("EgoStopped" in w for w in warnings)

    def test_ego_ungrounded(self, warn_validator):
        """EgoStopped = True but EgoQuery(speed) = Fast — should warn."""
        output = SymbolicOutput(
            entities=[],
            operations=[Operation("EgoQuery", "EgoQuery(speed)", "Fast")],
            facts=[Fact("EgoStopped", True)],
            rules=[Rule([("EgoStopped", True)], "KeepLane", "Stop")],
            selected_lateral="KeepLane",
            selected_longitudinal="Stop",
        )
        _, _, warnings = warn_validator.validate(output)
        assert any("EgoStopped" in w for w in warnings)

    # --- judgment grounding ---

    def test_judgment_always_passes(self, strict_validator):
        """CanStopComfortably is judgment — should never be flagged."""
        output = SymbolicOutput(
            entities=[],
            operations=[],
            facts=[Fact("CanStopComfortably", True)],
            rules=[Rule([("CanStopComfortably", True)], "KeepLane", "Deceleration")],
            selected_lateral="KeepLane",
            selected_longitudinal="Deceleration",
        )
        is_valid, violations, _ = strict_validator.validate(output)
        assert not any("CanStopComfortably" in v for v in violations)

    # --- OR logic ---

    def test_or_grounding_first_branch(self, warn_validator):
        """LeadVehicleBraking: satisfied by signal=BrakeLights branch."""
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle",
                             {"position": "Front", "lane": "EgoLane", "signal": "BrakeLights"})],
            operations=[],
            facts=[Fact("LeadVehicleBraking", True)],
            rules=[Rule([("LeadVehicleBraking", True)], "KeepLane", "Deceleration")],
            selected_lateral="KeepLane",
            selected_longitudinal="Deceleration",
        )
        _, _, warnings = warn_validator.validate(output)
        assert not any("LeadVehicleBraking" in w for w in warnings)

    def test_or_grounding_second_branch(self, warn_validator):
        """LeadVehicleBraking: satisfied by motion=Decelerating branch."""
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle",
                             {"position": "Front", "lane": "EgoLane", "motion": "Decelerating"})],
            operations=[],
            facts=[Fact("LeadVehicleBraking", True)],
            rules=[Rule([("LeadVehicleBraking", True)], "KeepLane", "Deceleration")],
            selected_lateral="KeepLane",
            selected_longitudinal="Deceleration",
        )
        _, _, warnings = warn_validator.validate(output)
        assert not any("LeadVehicleBraking" in w for w in warnings)

    # --- False facts not checked ---

    def test_false_fact_not_checked(self, strict_validator):
        """RedLight = False with no entity should NOT raise."""
        output = SymbolicOutput(
            entities=[],
            operations=[],
            facts=[Fact("RedLight", False)],
            rules=[Rule([("RedLight", False)], "KeepLane", "ConstantSpeed")],
            selected_lateral="KeepLane",
            selected_longitudinal="ConstantSpeed",
        )
        is_valid, violations, _ = strict_validator.validate(output)
        assert not any("RedLight" in v for v in violations)

    # --- Full examples grounding ---

    def test_full_example_grounding(self, warn_validator, parser):
        """FULL_EXAMPLE should have no grounding warnings."""
        output = parser.parse(FULL_EXAMPLE)
        _, _, warnings = warn_validator.validate(output)
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    def test_multi_rule_example_grounding(self, warn_validator, parser):
        """MULTI_RULE_EXAMPLE should have no grounding warnings."""
        output = parser.parse(MULTI_RULE_EXAMPLE)
        _, _, warnings = warn_validator.validate(output)
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    # --- attr_in matching ---

    def test_attr_in_matching(self, warn_validator):
        """AdjacentVehicleClose requires lane in [LeftLane, RightLane, Adjacent]."""
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle",
                             {"lane": "LeftLane", "distance": "Near"})],
            operations=[],
            facts=[Fact("AdjacentVehicleClose", True)],
            rules=[Rule([("AdjacentVehicleClose", True)], "KeepLane", "Deceleration")],
            selected_lateral="KeepLane",
            selected_longitudinal="Deceleration",
        )
        _, _, warnings = warn_validator.validate(output)
        assert not any("AdjacentVehicleClose" in w for w in warnings)

    # --- Subtype matching ---

    def test_subtype_matching(self, warn_validator):
        """EmergencyVehicleNearby requires subtype EmergencyVehicle."""
        output = SymbolicOutput(
            entities=[Entity("v_1", "EmergencyVehicle", "Vehicle", {"position": "Behind"})],
            operations=[],
            facts=[Fact("EmergencyVehicleNearby", True)],
            rules=[Rule([("EmergencyVehicleNearby", True)], "KeepLane", "Deceleration")],
            selected_lateral="KeepLane",
            selected_longitudinal="Deceleration",
        )
        _, _, warnings = warn_validator.validate(output)
        assert not any("EmergencyVehicleNearby" in w for w in warnings)

    def test_subtype_mismatch(self, warn_validator):
        """EmergencyVehicleNearby should warn if only regular Car present."""
        output = SymbolicOutput(
            entities=[Entity("v_1", "Car", "Vehicle", {"position": "Behind"})],
            operations=[],
            facts=[Fact("EmergencyVehicleNearby", True)],
            rules=[Rule([("EmergencyVehicleNearby", True)], "KeepLane", "Deceleration")],
            selected_lateral="KeepLane",
            selected_longitudinal="Deceleration",
        )
        _, _, warnings = warn_validator.validate(output)
        assert any("EmergencyVehicleNearby" in w for w in warnings)


# ---------------------------------------------------------------------------
# Complexity scorer tests
# ---------------------------------------------------------------------------

class TestComplexity:
    def test_full_example(self):
        result = compute_symbolic_complexity(FULL_EXAMPLE, SCHEMA_PATH)
        assert result["parseable"] is True
        assert result["valid"] is True
        assert result["num_entities"] == 4
        assert result["num_operations"] == 6
        assert result["num_facts"] == 7
        assert result["num_rules"] == 1
        assert result["action_consistent"] is True
        assert result["stage_completeness"] == 1.0
        assert 0.0 < result["complexity_score"] < 1.0

    def test_minimal_lower_complexity(self):
        full = compute_symbolic_complexity(FULL_EXAMPLE, SCHEMA_PATH)
        mini = compute_symbolic_complexity(MINIMAL_EXAMPLE, SCHEMA_PATH)
        assert mini["complexity_score"] < full["complexity_score"]

    def test_unparseable(self):
        result = compute_symbolic_complexity("garbage", SCHEMA_PATH)
        assert result["parseable"] is False
        assert result["complexity_score"] == 1.0

    def test_stage_completeness(self):
        # Empty perception but other stages present
        text = "PERCEPTION:\nOPERATIONS:\n  EgoQuery(speed) = Fast\nFACTS:\n  GreenLight = True\nRULES:\n  GreenLight -> KeepLane, ConstantSpeed\nACTION: KeepLane, ConstantSpeed"
        result = compute_symbolic_complexity(text, SCHEMA_PATH)
        assert result["stage_completeness"] == 0.75  # 3/4, no entities


# ---------------------------------------------------------------------------
# Complexity — grounding score tests
# ---------------------------------------------------------------------------

class TestComplexityGrounding:
    def test_grounding_score_in_result(self):
        result = compute_symbolic_complexity(FULL_EXAMPLE, SCHEMA_PATH)
        assert "grounding_score" in result
        assert "ungrounded_facts" in result
        assert "judgment_facts" in result
        assert 0.0 <= result["grounding_score"] <= 1.0

    def test_well_grounded_example(self):
        result = compute_symbolic_complexity(FULL_EXAMPLE, SCHEMA_PATH)
        assert result["grounding_score"] == 1.0
        assert len(result["ungrounded_facts"]) == 0

    def test_judgment_fact_tracked(self):
        result = compute_symbolic_complexity(FULL_EXAMPLE, SCHEMA_PATH)
        assert "CanStopComfortably" in result["judgment_facts"]

    def test_ungrounded_fact_lowers_score(self):
        """RedLight = True but no TrafficLight entity → low grounding score."""
        text = """\
PERCEPTION:
  v_1 = Car {position: Front, lane: EgoLane, distance: Near, motion: Stationary}

OPERATIONS:
  EgoQuery(speed) = Slow

FACTS:
  RedLight = True
  EgoMovingSlow = True

RULES:
  RedLight -> KeepLane, Stop

ACTION: KeepLane, Stop"""
        result = compute_symbolic_complexity(text, SCHEMA_PATH)
        assert "RedLight" in result["ungrounded_facts"]
        assert result["grounding_score"] < 1.0

    def test_grounding_warnings_in_result(self):
        result = compute_symbolic_complexity(FULL_EXAMPLE, SCHEMA_PATH)
        assert "grounding_warnings" in result
