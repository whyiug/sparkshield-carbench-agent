from __future__ import annotations

import re
from pathlib import Path

from track_1_agent_under_test.car_guard.domain import OfficialToolCall
from track_1_agent_under_test.car_guard.policy import (
    ActionAuthorization,
    PolicyCompiler,
    PolicyRequest,
)


EXPECTED_RULE_IDS = (
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
)


def test_actual_runtime_policy_compiles_all_numbered_rules() -> None:
    repository = Path(__file__).resolve().parents[2]
    runtime_policy = (
        repository
        / "third_party/car-bench/car_bench/envs/car_voice_assistant/wiki.md"
    ).read_text(encoding="utf-8")

    compiled = PolicyCompiler().compile(runtime_policy)

    assert compiled.rule_ids == EXPECTED_RULE_IDS
    assert all(rule.source_text for rule in compiled.numbered_rules)


def test_semantic_detection_does_not_require_policy_labels() -> None:
    repository = Path(__file__).resolve().parents[2]
    runtime_policy = (
        repository
        / "third_party/car-bench/car_bench/envs/car_voice_assistant/wiki.md"
    ).read_text(encoding="utf-8")
    without_labels = re.sub(
        r"(?:(?:LLM|AUT)-POL:)(?:\d{3}):",
        "",
        runtime_policy,
    )

    compiled = PolicyCompiler().compile(without_labels)

    assert compiled.rule_ids == EXPECTED_RULE_IDS
    assert {rule.source.value for rule in compiled.numbered_rules} == {
        "semantic_text"
    }


def test_absent_numbered_policy_is_not_silently_added() -> None:
    compiled = PolicyCompiler().compile("A short in-car assistant policy.")
    request = PolicyRequest(
        action=OfficialToolCall(
            tool_name="open_close_sunroof", arguments={"percentage": 50}
        ),
        authorization=ActionAuthorization.USER_REQUEST,
        state={"sunshade_position": 0},
    )

    decision = compiled.evaluate(request)

    assert compiled.rule_ids == ()
    assert decision.prerequisite_calls == []


def test_dynamic_location_and_datetime_are_parsed_from_current_text() -> None:
    system_policy = """
    - AUT-POL:009: Weather has to be checked before opening the sunroof.
    CURRENT_LOCATION = {"id":"loc_osl","name":"Oslo"}
    DATETIME = {"year":2026,"month":7,"day":11,"hour":16,"minute":45}
    """
    compiled = PolicyCompiler().compile(system_policy)

    assert compiled.current_location == {"id": "loc_osl", "name": "Oslo"}
    assert compiled.current_datetime == {
        "year": 2026,
        "month": 7,
        "day": 11,
        "hour": 16,
        "minute": 45,
    }
    assert compiled.source_digest == PolicyCompiler().compile(system_policy).source_digest


def test_current_datetime_alias_is_parsed_from_current_text() -> None:
    compiled = PolicyCompiler().compile(
        'CURRENT_DATETIME={"year":2026,"month":7,"day":12,"hour":9,"minute":5}.'
    )

    assert compiled.current_datetime == {
        "year": 2026,
        "month": 7,
        "day": 12,
        "hour": 9,
        "minute": 5,
    }


def test_conflicting_datetime_assignments_fail_closed() -> None:
    compiled = PolicyCompiler().compile(
        'CURRENT_DATETIME={"year":2026,"month":7,"day":12,"hour":9,"minute":5}. '
        'DATETIME={"year":2026,"month":7,"day":12,"hour":10,"minute":5}.'
    )

    assert compiled.current_datetime is None


def test_general_tts_reactive_and_live_schema_constraints_are_always_on() -> None:
    compiled = PolicyCompiler().compile("")
    decision = compiled.evaluate(
        PolicyRequest(
            action=OfficialToolCall(tool_name="set_custom_state"),
            state_changing=True,
            requested_tool_available=False,
            requested_parameters_available=False,
        )
    )

    assert {
        "GEN-TTS",
        "GEN-REACTIVE-ONLY",
        "GEN-LIVE-SCHEMA",
        "GEN-TOOL-DESCRIPTION",
    }.issubset(compiled.general_rule_ids)
    assert {
        "reactive_action_not_authorized",
        "requested_tool_unavailable",
        "requested_parameter_unavailable",
    }.issubset({conflict.code for conflict in decision.conflicts})
    assert "tts_no_markdown_or_visual_lists" in {
        item.code for item in decision.format_decisions
    }


def test_reactive_constraint_recognizes_namespaced_semantic_set() -> None:
    decision = PolicyCompiler().compile("").evaluate(
        PolicyRequest(semantic_operation="vehicle.fan.set_speed")
    )

    assert "reactive_action_not_authorized" in {
        conflict.code for conflict in decision.conflicts
    }


def test_live_tool_description_requires_full_parameter_confirmation() -> None:
    compiled = PolicyCompiler().compile("")
    decision = compiled.evaluate(
        PolicyRequest(
            action=OfficialToolCall(
                tool_name="set_safety_state", arguments={"level": 2, "zone": "ALL"}
            ),
            authorization=ActionAuthorization.USER_REQUEST,
            state_changing=True,
            tool_description="  REQUIRES_CONFIRMATION: changes safety state",
        )
    )

    assert decision.confirmation is not None
    assert decision.confirmation.rule_ids == ["GEN-TOOL-DESCRIPTION"]
    format_decision = next(
        item
        for item in decision.format_decisions
        if item.code == "describe_action_and_all_parameters_before_confirmation"
    )
    assert format_decision.details["arguments"] == {"level": 2, "zone": "ALL"}


def test_route_and_poi_general_presentation_rules_are_structured() -> None:
    system_policy = """
    For multiple alternative routes, present the fastest route and shortest route,
    then only count further route alternatives.
    Tell the user about every point of interest at least by name and ask which one
    the user wants directions to.
    """
    decision = PolicyCompiler().compile(system_policy).evaluate(
        PolicyRequest(
            facts={
                "route_alternative_count": 3,
                "fastest_route_id": "r1",
                "shortest_route_id": "r1",
                "poi_names": ["North Garage", "Central Garage"],
            }
        )
    )

    format_codes = {item.code for item in decision.format_decisions}
    assert "route_alternatives_fastest_and_shortest" in format_codes
    assert "present_all_poi_names" in format_codes
    assert "ask_which_poi_to_navigate_to" in format_codes
