from __future__ import annotations

from typing import Any

from track_1_agent_under_test.car_guard.capability import (
    LiveToolRegistry,
    SemanticCapabilityBinder,
    instantiate_triggered_closure,
    prove_goal_feasibility,
)
from track_1_agent_under_test.car_guard.domain import GateOutcome, Goal
from track_1_agent_under_test.car_guard.recipes import RecipeRegistry


def function_tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    *,
    description: str | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def ac_goal() -> Goal:
    return Goal(
        goal_id="ac-goal",
        semantic_operation="enable_air_conditioning",
        desired_outcome={"enabled": True},
        atomic_group="ac-bundle",
    )


def ac_tool() -> dict[str, Any]:
    return function_tool(
        "set_air_conditioning",
        {"on": {"type": "boolean"}},
        ["on"],
    )


def window_tool() -> dict[str, Any]:
    return function_tool(
        "open_close_window",
        {
            "window": {
                "type": "string",
                "enum": [
                    "ALL",
                    "DRIVER",
                    "PASSENGER",
                    "DRIVER_REAR",
                    "PASSENGER_REAR",
                ],
            },
            "percentage": {"type": "number", "minimum": 0, "maximum": 100},
        },
        ["window", "percentage"],
    )


def fan_tool() -> dict[str, Any]:
    return function_tool(
        "set_fan_speed",
        {"level": {"type": "number", "minimum": 0, "maximum": 5}},
        ["level"],
    )


def direct_ac_tools() -> list[dict[str, Any]]:
    return [
        function_tool("get_climate_settings"),
        function_tool("get_vehicle_window_positions"),
        ac_tool(),
    ]


def dumped_bindings(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = RecipeRegistry()
    goal = ac_goal()
    recipes = registry.select_by_intent_and_policy(goal)
    bindings = SemanticCapabilityBinder().bind(goal, recipes, LiveToolRegistry(tools))
    return [binding.model_dump(mode="json") for binding in bindings]


def test_live_tool_order_does_not_change_goal_binding() -> None:
    tools = direct_ac_tools()
    assert dumped_bindings(tools) == dumped_bindings(list(reversed(tools)))


def test_irrelevant_tool_addition_or_deletion_does_not_change_binding() -> None:
    relevant = direct_ac_tools()
    unrelated = function_tool(
        "send_email",
        {
            "content_message": {"type": "string"},
            "email_addresses": {"type": "array", "items": {"type": "string"}},
        },
        ["content_message", "email_addresses"],
    )
    assert dumped_bindings(relevant) == dumped_bindings([unrelated, *relevant])
    assert dumped_bindings(relevant) == dumped_bindings([*relevant, unrelated])


def test_unrelated_tool_description_change_does_not_change_binding() -> None:
    tools = direct_ac_tools()
    changed = [dict(tool) for tool in tools]
    changed[0] = function_tool(
        "get_climate_settings",
        description="Read the current cabin climate configuration.",
    )
    assert dumped_bindings(tools) == dumped_bindings(changed)


def test_untriggered_conditional_branches_do_not_require_setters() -> None:
    registry = RecipeRegistry()
    goal = ac_goal()
    recipes = registry.select_by_intent_and_policy(goal)
    closure = instantiate_triggered_closure(
        goal,
        {
            "window_driver_position": 0,
            "window_passenger_position": 0,
            "window_driver_rear_position": 0,
            "window_passenger_rear_position": 0,
            "fan_speed": 2,
        },
        recipes,
    )

    assert closure.triggered_branch_ids == []
    assert [item.operation.semantic_operation for item in closure.requirements] == [
        "set_air_conditioning"
    ]
    proof = prove_goal_feasibility(goal, closure, LiveToolRegistry([ac_tool()]))
    assert proof.feasible
    assert proof.first_set_allowed
    assert proof.outcome is GateOutcome.ALLOW


def test_triggered_unbindable_branch_blocks_atomic_goal_before_first_set() -> None:
    registry = RecipeRegistry()
    goal = ac_goal()
    recipes = registry.select_by_intent_and_policy(goal)
    closure = instantiate_triggered_closure(
        goal,
        {
            "window_driver_position": 60,
            "window_passenger_position": 0,
            "window_driver_rear_position": 0,
            "window_passenger_rear_position": 0,
            "fan_speed": 2,
        },
        recipes,
    )

    assert closure.triggered_branch_ids == ["close_driver_window_before_ac"]
    assert [item.operation.semantic_operation for item in closure.requirements] == [
        "close_driver_window_for_ac",
        "set_air_conditioning",
    ]
    proof = prove_goal_feasibility(goal, closure, LiveToolRegistry([ac_tool()]))
    assert not proof.feasible
    assert not proof.first_set_allowed
    assert proof.outcome is GateOutcome.UNSUPPORTED_CAPABILITY
    assert proof.checked_operations == ["close_driver_window_for_ac"]


def test_all_triggered_branches_must_bind_for_feasibility() -> None:
    registry = RecipeRegistry()
    goal = ac_goal()
    recipes = registry.select_by_intent_and_policy(goal)
    closure = instantiate_triggered_closure(
        goal,
        {
            "window_driver_position": 60,
            "window_passenger_position": 0,
            "window_driver_rear_position": 0,
            "window_passenger_rear_position": 30,
            "fan_speed": 0,
        },
        recipes,
    )
    proof = prove_goal_feasibility(
        goal,
        closure,
        LiveToolRegistry([ac_tool(), window_tool(), fan_tool()]),
    )

    assert closure.triggered_branch_ids == [
        "close_driver_window_before_ac",
        "close_passenger_rear_window_before_ac",
        "start_fan_before_ac",
    ]
    assert proof.feasible
    assert proof.checked_operations == [
        "close_driver_window_for_ac",
        "close_passenger_rear_window_for_ac",
        "set_minimum_fan_for_ac",
        "set_air_conditioning",
    ]


def test_unknown_branch_evidence_requires_read_without_instantiating_setter() -> None:
    registry = RecipeRegistry()
    goal = ac_goal()
    recipes = registry.select_by_intent_and_policy(goal)
    closure = instantiate_triggered_closure(goal, {"fan_speed": 2}, recipes)

    assert closure.triggered_branch_ids == []
    assert set(closure.unresolved_evidence) == {
        "window_driver_position",
        "window_passenger_position",
        "window_driver_rear_position",
        "window_passenger_rear_position",
    }
    assert all(
        not item.operation.semantic_operation.startswith("close_")
        for item in closure.requirements
    )
    proof = prove_goal_feasibility(goal, closure, LiveToolRegistry([ac_tool()]))
    assert not proof.feasible
    assert proof.outcome is GateOutcome.NEED_READ


def test_changed_unrelated_live_inventory_does_not_change_recipe_registry_hash() -> (
    None
):
    first = RecipeRegistry()
    LiveToolRegistry(direct_ac_tools())
    second = RecipeRegistry()
    LiveToolRegistry([function_tool("unrelated")])
    third = RecipeRegistry()
    assert first.registry_hash == second.registry_hash == third.registry_hash
