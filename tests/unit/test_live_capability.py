from __future__ import annotations

from typing import Any

import pytest

from track_1_agent_under_test.car_guard.capability import (
    LiveToolRegistry,
    SemanticCapabilityBinder,
    ToolArgumentValidationError,
    ToolDefinitionError,
)
from track_1_agent_under_test.car_guard.domain import Goal
from track_1_agent_under_test.car_guard.recipes import (
    REQUIRED_RECIPE_DOMAINS,
    RecipeRegistry,
)


def function_tool(
    name: str,
    *,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    additional_properties: bool | dict[str, Any] = False,
    description: str | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": additional_properties,
        },
    }
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def test_live_registry_strictly_enforces_declared_json_schema() -> None:
    registry = LiveToolRegistry(
        [
            function_tool(
                "configure",
                properties={
                    "mode": {"type": "string", "enum": ["eco", "comfort"]},
                    "level": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 5,
                        "multipleOf": 1,
                    },
                    "options": {
                        "type": "object",
                        "properties": {"quiet": {"type": "boolean"}},
                        "required": ["quiet"],
                        "additionalProperties": False,
                    },
                },
                required=["mode", "level", "options"],
            )
        ]
    )

    arguments = {"mode": "eco", "level": 3, "options": {"quiet": True}}
    assert registry.validate_arguments("configure", arguments) == arguments
    assert registry.required_parameters("configure") == {
        "mode",
        "level",
        "options",
    }
    assert registry.parameter_domain("configure", "mode") == {
        "type": "string",
        "enum": ["eco", "comfort"],
    }

    invalid_cases = [
        ({"level": 3, "options": {"quiet": True}}, "required"),
        ({"mode": "sport", "level": 3, "options": {"quiet": True}}, "enum"),
        ({"mode": "eco", "level": True, "options": {"quiet": True}}, "type"),
        ({"mode": "eco", "level": 6, "options": {"quiet": True}}, "maximum"),
        (
            {"mode": "eco", "level": 3.5, "options": {"quiet": True}},
            "multipleOf",
        ),
        (
            {"mode": "eco", "level": 3, "options": {"quiet": True, "x": 1}},
            "additionalProperties",
        ),
        (
            {
                "mode": "eco",
                "level": 3,
                "options": {"quiet": True},
                "undeclared": 1,
            },
            "additionalProperties",
        ),
    ]
    for invalid, expected_code in invalid_cases:
        with pytest.raises(ToolArgumentValidationError) as error:
            registry.validate_arguments("configure", invalid)
        assert error.value.code == expected_code


def test_omitted_additional_properties_is_closed_and_explicit_schema_is_honored() -> (
    None
):
    closed = {
        "type": "function",
        "function": {
            "name": "closed",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    open_tool = function_tool(
        "open",
        additional_properties={"type": "integer", "minimum": 0},
    )
    registry = LiveToolRegistry([closed, open_tool])

    with pytest.raises(ToolArgumentValidationError) as error:
        registry.validate_arguments("closed", {"x": 1})
    assert error.value.code == "additionalProperties"
    assert registry.validate_arguments("open", {"x": 1}) == {"x": 1}
    with pytest.raises(ToolArgumentValidationError):
        registry.validate_arguments("open", {"x": -1})


def test_registry_rejects_duplicate_or_malformed_live_definitions() -> None:
    duplicate = function_tool("same")
    with pytest.raises(ToolDefinitionError, match="duplicate"):
        LiveToolRegistry([duplicate, duplicate])
    with pytest.raises(ToolDefinitionError, match="parameters"):
        LiveToolRegistry(
            [{"type": "function", "function": {"name": "bad", "parameters": []}}]
        )


def test_optional_false_property_schema_is_pruned_without_weakening_root() -> None:
    official_shape = function_tool(
        "calculate_datetime",
        properties={
            "additionalProperties": False,
            "original_datetime": {
                "type": "object",
                "properties": {"year": {"type": "number"}},
                "required": ["year"],
            },
        },
        required=["original_datetime"],
    )

    registry = LiveToolRegistry([official_shape])

    assert (
        registry.parameter_domain("calculate_datetime", "additionalProperties") is None
    )
    assert registry.validate_arguments(
        "calculate_datetime", {"original_datetime": {"year": 2026}}
    ) == {"original_datetime": {"year": 2026}}
    with pytest.raises(ToolArgumentValidationError) as error:
        registry.validate_arguments(
            "calculate_datetime",
            {
                "original_datetime": {"year": 2026},
                "additionalProperties": False,
            },
        )
    assert error.value.code == "additionalProperties"
    assert (
        official_shape["function"]["parameters"]["properties"]["additionalProperties"]
        is False
    )


def test_optional_false_property_is_pruned_when_parent_implicitly_closed() -> None:
    implicit_closed = {
        "type": "function",
        "function": {
            "name": "implicit_closed",
            "parameters": {
                "type": "object",
                "properties": {
                    "unusable": False,
                    "value": {"type": "string"},
                },
            },
        },
    }

    registry = LiveToolRegistry([implicit_closed])

    assert registry.parameter_domain("implicit_closed", "unusable") is None
    assert registry.validate_arguments("implicit_closed", {"value": "ok"}) == {
        "value": "ok"
    }


@pytest.mark.parametrize(
    "additional_properties",
    [True, {"type": "string"}],
    ids=["unrestricted", "schema-valued"],
)
def test_optional_false_property_schema_is_not_pruned_from_open_parent(
    additional_properties: bool | dict[str, Any],
) -> None:
    open_parent = function_tool(
        "open_parent",
        properties={"reserved": False},
        additional_properties=additional_properties,
    )

    with pytest.raises(ToolDefinitionError, match="invalid entry"):
        LiveToolRegistry([open_parent])


def test_required_false_property_schema_remains_a_definition_error() -> None:
    impossible = function_tool(
        "impossible",
        properties={"value": False},
        required=["value"],
    )

    with pytest.raises(ToolDefinitionError, match="unsatisfiable"):
        LiveToolRegistry([impossible])


def test_live_definitions_are_defensive_copies() -> None:
    registry = LiveToolRegistry([function_tool("get_status")])

    definitions = registry.as_definitions()
    definitions[0]["function"]["name"] = "changed"

    assert registry.has_tool("get_status")
    assert not registry.has_tool("changed")


def test_registry_serializes_only_a_live_schema_valid_official_call() -> None:
    registry = LiveToolRegistry(
        [
            function_tool(
                "set_air_conditioning",
                properties={"on": {"type": "boolean"}},
                required=["on"],
            )
        ]
    )
    assert registry.serialize_official(
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ) == {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    with pytest.raises(ToolArgumentValidationError, match="not live"):
        registry.serialize_official({"tool_name": "set_ac", "arguments": {}})


def test_live_registry_has_no_catalog_difference_or_task_classification_api() -> None:
    registry = LiveToolRegistry([])
    forbidden = {
        "diff_against_full_catalog",
        "infer_missing_component",
        "infer_task_type",
        "missing_tools",
    }
    assert all(not hasattr(registry, name) for name in forbidden)


def test_recipe_registry_is_fixed_complete_and_hash_stable() -> None:
    first = RecipeRegistry()
    second = RecipeRegistry()

    assert first.registry_hash == second.registry_hash

    supported_policy_ids = {
        "POL-002",
        "POL-004",
        "POL-005",
        "POL-007",
        "POL-008",
        "POL-009",
        "POL-010",
        "POL-011",
        "POL-012",
        "POL-013",
        "POL-014",
        "POL-016",
        "POL-017",
        "POL-018",
        "POL-019",
        "POL-021",
        "POL-022",
        "POL-023",
        "POL-024",
    }
    assert all(
        set(recipe.policy_hooks).issubset(supported_policy_ids)
        for recipe in first.recipes
    )
    assert len(first.registry_hash) == 64
    assert REQUIRED_RECIPE_DOMAINS.issubset(first.covered_domains)
    assert len(first) >= 25
    serialized = "\n".join(recipe.model_dump_json() for recipe in first)
    assert "task_id" not in serialized
    assert "removed_part" not in serialized


def test_multi_operation_recipes_do_not_route_broad_aliases_to_wrong_primary() -> None:
    registry = RecipeRegistry()

    fan = registry.get("fan_control")
    charging = registry.get("charging_station")
    assert fan.target_operation("set_fan_airflow_direction").semantic_operation == (
        "set_fan_airflow_direction"
    )
    assert not fan.matches_semantic_operation("set_fan_airflow")
    assert charging.target_operation("calculate_charging_time").semantic_operation == (
        "calculate_charging_time"
    )
    assert not charging.matches_semantic_operation("find_charging_station")


def test_recipe_selection_uses_only_semantic_intent_and_policy() -> None:
    registry = RecipeRegistry()
    goal = Goal(
        goal_id="runtime-goal-id",
        semantic_operation="enable_air_conditioning",
        desired_outcome={"enabled": True},
    )

    by_intent = registry.select_by_intent_and_policy(goal)
    assert [recipe.id for recipe in by_intent] == ["air_conditioning"]
    by_policy = registry.select_by_intent_and_policy(
        "unmapped_semantic_operation", ["POL-005"]
    )
    assert {recipe.id for recipe in by_policy} == {
        "sunroof_control",
        "sunshade_control",
    }


def test_binder_uses_only_goal_relevant_direct_recipe_operations() -> None:
    recipes = RecipeRegistry()
    goal = Goal(
        goal_id="enable-ac",
        semantic_operation="enable_air_conditioning",
        desired_outcome={"enabled": True},
        atomic_group="climate-action",
    )
    live = LiveToolRegistry(
        [
            function_tool("get_climate_settings"),
            function_tool("get_vehicle_window_positions"),
            function_tool(
                "set_air_conditioning",
                properties={"on": {"type": "boolean"}},
                required=["on"],
            ),
            function_tool(
                "send_email",
                properties={"message": {"type": "string"}},
                required=["message"],
            ),
        ]
    )

    relevant = recipes.select_by_intent_and_policy(goal)
    bindings = SemanticCapabilityBinder().bind(goal, relevant, live)

    assert [binding.tool_name for binding in bindings] == [
        "get_climate_settings",
        "get_vehicle_window_positions",
        "set_air_conditioning",
    ]
    assert bindings[-1].arguments == {"on": True}
    assert all(binding.recipe_id == "air_conditioning" for binding in bindings)
    assert all(
        binding.semantic_operation != "close_windows_for_ac" for binding in bindings
    )


@pytest.mark.parametrize(
    ("semantic_operation", "desired_outcome", "tool_name", "parameter", "values"),
    [
        (
            "set_trunk_position",
            {"action": "open"},
            "open_close_trunk_door",
            "action",
            ["OPEN", "CLOSE"],
        ),
        (
            "set_air_circulation",
            {"mode": "fresh_air"},
            "set_air_circulation",
            "mode",
            ["FRESH_AIR", "RECIRCULATION", "AUTO"],
        ),
    ],
)
def test_binder_canonicalizes_unique_spelling_only_live_enum_matches(
    semantic_operation: str,
    desired_outcome: dict[str, str],
    tool_name: str,
    parameter: str,
    values: list[str],
) -> None:
    recipes = RecipeRegistry()
    goal = Goal(
        goal_id="enum-goal",
        semantic_operation=semantic_operation,
        desired_outcome=desired_outcome,
    )
    live = LiveToolRegistry(
        [
            function_tool(
                tool_name,
                properties={parameter: {"type": "string", "enum": values}},
                required=[parameter],
            )
        ]
    )

    binding = SemanticCapabilityBinder().bind(
        goal, recipes.select_by_intent_and_policy(goal), live
    )[-1]

    assert binding.is_complete
    assert binding.arguments == {parameter: values[0]}


def test_binder_does_not_guess_a_non_matching_live_enum_value() -> None:
    recipes = RecipeRegistry()
    goal = Goal(
        goal_id="circulation-goal",
        semantic_operation="set_air_circulation",
        desired_outcome={"mode": "outside"},
    )
    live = LiveToolRegistry(
        [
            function_tool(
                "set_air_circulation",
                properties={
                    "mode": {
                        "type": "string",
                        "enum": ["FRESH_AIR", "RECIRCULATION", "AUTO"],
                    }
                },
                required=["mode"],
            )
        ]
    )

    binding = SemanticCapabilityBinder().bind(
        goal, recipes.select_by_intent_and_policy(goal), live
    )[-1]

    assert not binding.is_complete
    assert binding.arguments == {"mode": "outside"}
    assert binding.argument_error is not None


def test_binder_keeps_distinct_non_ascii_enum_spellings_distinct() -> None:
    recipes = RecipeRegistry()
    goal = Goal(
        goal_id="unicode-enum-goal",
        semantic_operation="set_air_circulation",
        desired_outcome={"mode": "blé"},
    )
    live = LiveToolRegistry(
        [
            function_tool(
                "set_air_circulation",
                properties={
                    "mode": {"type": "string", "enum": ["RED", "BLÅ"]}
                },
                required=["mode"],
            )
        ]
    )

    binding = SemanticCapabilityBinder().bind(
        goal, recipes.select_by_intent_and_policy(goal), live
    )[-1]

    assert not binding.is_complete
    assert binding.arguments == {"mode": "blé"}
    assert binding.argument_error is not None
