from __future__ import annotations

from collections.abc import Callable

import pytest

from track_1_agent_under_test.car_guard.domain import OfficialToolCall
from track_1_agent_under_test.car_guard.policy import (
    ActionAuthorization,
    ExecutionMode,
    PolicyCompiler,
    PolicyDecision,
    PolicyRequest,
)


RULE_IDS = (
    "002",
    "004",
    "005",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "016",
    "017",
    "018",
    "019",
    "021",
    "022",
    "023",
    "024",
)
SYSTEM_POLICY = "\n\n".join(
    [f"- AUT-POL:{rule_id}: Runtime policy clause {rule_id}." for rule_id in RULE_IDS]
    + [
        (
            '- CURRENT_LOCATION = {"id":"loc_par","name":"Paris",'
            '"position":{"longitude":2.35,"latitude":48.85}}'
        ),
        '- DATETIME = {"year":2025,"month":5,"day":6,"hour":8,"minute":30}',
    ]
)
POLICY = PolicyCompiler().compile(SYSTEM_POLICY)


def action_request(
    tool_name: str,
    arguments: dict | None = None,
    *,
    state: dict | None = None,
    facts: dict | None = None,
    description: str | None = None,
    state_changing: bool | None = None,
) -> PolicyRequest:
    return PolicyRequest(
        action=OfficialToolCall(tool_name=tool_name, arguments=arguments or {}),
        authorization=ActionAuthorization.USER_REQUEST,
        state=state or {},
        facts=facts or {},
        tool_description=description,
        state_changing=state_changing,
    )


def has_read(operation: str) -> Callable[[PolicyDecision], bool]:
    return lambda decision: (
        operation in {call.semantic_operation for call in decision.required_reads}
    )


def has_prerequisite(operation: str) -> Callable[[PolicyDecision], bool]:
    return lambda decision: (
        operation in {call.semantic_operation for call in decision.prerequisite_calls}
    )


def has_format(code: str) -> Callable[[PolicyDecision], bool]:
    return lambda decision: code in {item.code for item in decision.format_decisions}


def has_conflict(code: str) -> Callable[[PolicyDecision], bool]:
    return lambda decision: code in {item.code for item in decision.conflicts}


def test_pol_011_retains_an_ordered_all_window_close_as_a_prerequisite() -> None:
    close_all = OfficialToolCall(
        tool_name="open_close_window",
        arguments={"window": "ALL", "percentage": 0},
    )
    enable_ac = OfficialToolCall(
        tool_name="set_air_conditioning", arguments={"on": True}
    )
    decision = POLICY.evaluate(
        PolicyRequest(
            action=enable_ac,
            semantic_operation="set_air_conditioning",
            ordered_actions=[close_all, enable_ac],
            authorization=ActionAuthorization.USER_REQUEST,
            state={
                "fan_speed": 1,
                "window_driver_position": 50,
                "window_passenger_position": 0,
                "window_driver_rear_position": 30,
                "window_passenger_rear_position": 0,
            },
        )
    )

    window_prerequisites = [
        item
        for item in decision.prerequisite_calls
        if item.semantic_operation == "vehicle.window.set_position"
    ]
    assert [item.arguments for item in window_prerequisites] == [
        {"window": "ALL", "percentage": 0}
    ]


def confirmation_has(rule_id: str) -> Callable[[PolicyDecision], bool]:
    return lambda decision: (
        decision.confirmation is not None and rule_id in decision.confirmation.rule_ids
    )


def is_serial(decision: PolicyDecision) -> bool:
    return decision.execution_mode is ExecutionMode.SERIAL


POLICY_CASES = [
    (
        "POL-002",
        PolicyRequest(),
        has_format("metric_distance_km_and_m"),
    ),
    (
        "POL-004",
        action_request(
            "dangerous_action",
            {"target": "front"},
            description="REQUIRES_CONFIRMATION, changes a safety control",
            state_changing=True,
        ),
        confirmation_has("POL-004"),
    ),
    (
        "POL-005",
        action_request(
            "open_close_sunroof",
            {"percentage": 50},
            state={"sunshade_position": 0, "weather_condition": "sunny"},
        ),
        has_prerequisite("vehicle.sunshade.set_position"),
    ),
    (
        "POL-007",
        action_request(
            "open_close_window",
            {"window": "DRIVER", "percentage": 50},
            state={"air_conditioning": True},
        ),
        confirmation_has("POL-007"),
    ),
    (
        "POL-008",
        action_request(
            "open_close_sunroof",
            {"percentage": 50},
            state={"sunshade_position": 100, "weather_condition": "rainy"},
        ),
        confirmation_has("POL-008"),
    ),
    (
        "POL-009",
        action_request(
            "open_close_sunroof",
            {"percentage": 50},
            state={"sunshade_position": 100},
        ),
        has_read("weather.current.read"),
    ),
    (
        "POL-010",
        action_request(
            "set_window_defrost",
            {"on": True, "defrost_window": "FRONT"},
            state={
                "fan_speed": 0,
                "fan_airflow_direction": "FEET",
                "air_conditioning": False,
            },
        ),
        has_prerequisite("vehicle.fan.set_airflow"),
    ),
    (
        "POL-011",
        action_request(
            "set_air_conditioning",
            {"on": True},
            state={
                "fan_speed": 0,
                "window_driver_position": 50,
                "window_passenger_position": 0,
                "window_driver_rear_position": 0,
                "window_passenger_rear_position": 30,
            },
        ),
        has_prerequisite("vehicle.window.set_position"),
    ),
    (
        "POL-012",
        action_request(
            "set_climate_temperature",
            {"temperature": 25, "seat_zone": "DRIVER"},
            state={
                "climate_temperature_driver": 20,
                "climate_temperature_passenger": 19,
            },
        ),
        has_format("inform_temperature_zone_difference"),
    ),
    (
        "POL-013",
        action_request(
            "set_fog_lights",
            {"on": True},
            state={
                "head_lights_low_beams": False,
                "head_lights_high_beams": True,
                "weather_condition": "cloudy_and_hail",
            },
        ),
        has_prerequisite("vehicle.low_beam.set"),
    ),
    (
        "POL-014",
        action_request(
            "set_head_lights_high_beams",
            {"on": True},
            state={"fog_lights": True},
        ),
        confirmation_has("POL-014"),
    ),
    (
        "POL-016",
        action_request(
            "set_new_navigation",
            {"route_ids": ["route-1"]},
            state={"navigation_active": False},
            facts={"route_start_id": "loc_ber"},
        ),
        has_conflict("route_start_not_current_location"),
    ),
    (
        "POL-017",
        action_request(
            "navigation_add_one_waypoint",
            {"waypoint_id": "loc_bon"},
            state={"navigation_active": False, "waypoints": []},
        ),
        has_conflict("navigation_edit_requires_active_route"),
    ),
    (
        "POL-018",
        action_request(
            "navigation_replace_one_waypoint",
            {"waypoint_id_to_replace": "a", "new_waypoint_id": "b"},
            state={"navigation_active": True, "waypoints": ["a", "c", "d"]},
        ),
        is_serial,
    ),
    (
        "POL-019",
        action_request(
            "navigation_delete_destination",
            {"destination_id_to_delete": "dest"},
            state={"navigation_active": True, "waypoints": ["start", "dest"]},
        ),
        has_conflict("cannot_delete_only_destination"),
    ),
    (
        "POL-021",
        PolicyRequest(
            facts={"route_presented_in_detail": True, "route_includes_toll": True}
        ),
        has_format("disclose_route_toll"),
    ),
    (
        "POL-022",
        PolicyRequest(
            facts={
                "route_segment_count": 3,
                "route_selection_specified": False,
                "selected_segments_include_toll": True,
            }
        ),
        has_format("select_fastest_route_per_segment"),
    ),
    (
        "POL-023",
        action_request(
            "get_entries_from_calendar",
            {"month": 5, "day": 7},
            state_changing=False,
        ),
        has_conflict("non_current_day_request"),
    ),
    (
        "POL-024",
        action_request(
            "get_weather",
            {"location_or_poi_id": "loc_par", "month": 5, "day": 6},
            state_changing=False,
        ),
        has_conflict("weather_time_required"),
    ),
]


@pytest.mark.parametrize(
    ("rule_id", "policy_request", "assertion"),
    POLICY_CASES,
    ids=[case[0] for case in POLICY_CASES],
)
def test_numbered_policy_matrix(
    rule_id: str,
    policy_request: PolicyRequest,
    assertion: Callable[[PolicyDecision], bool],
) -> None:
    decision = POLICY.evaluate(policy_request)

    assert assertion(decision), decision.model_dump(mode="json")
    assert rule_id in decision.applied_rule_ids


def test_policy_prerequisites_are_semantic_and_deterministically_ordered() -> None:
    decision = POLICY.evaluate(
        action_request(
            "set_window_defrost",
            {"on": True, "defrost_window": "ALL"},
            state={
                "fan_speed": 0,
                "fan_airflow_direction": "HEAD",
                "air_conditioning": False,
            },
        )
    )

    assert [call.semantic_operation for call in decision.prerequisite_calls] == [
        "vehicle.fan.set_speed",
        "vehicle.fan.set_airflow",
        "vehicle.ac.set",
    ]
    assert all(call.order == "before_action" for call in decision.prerequisite_calls)


@pytest.mark.parametrize("ordered_level", [1, 2])
def test_pol_011_reuses_sufficient_ordered_fan_action(ordered_level: int) -> None:
    decision = POLICY.evaluate(
        PolicyRequest(
            action=OfficialToolCall(
                tool_name="set_air_conditioning", arguments={"on": True}
            ),
            ordered_actions=[
                OfficialToolCall(
                    tool_name="set_fan_speed", arguments={"level": ordered_level}
                ),
                OfficialToolCall(
                    tool_name="set_air_conditioning", arguments={"on": True}
                ),
                OfficialToolCall(
                    tool_name="set_window_defrost",
                    arguments={"on": True, "defrost_window": "FRONT"},
                ),
            ],
            authorization=ActionAuthorization.USER_REQUEST,
            state={
                "fan_speed": 0,
                "window_driver_position": 0,
                "window_passenger_position": 0,
                "window_driver_rear_position": 0,
                "window_passenger_rear_position": 0,
            },
        )
    )

    assert not any(
        call.semantic_operation == "vehicle.fan.set_speed"
        for call in decision.prerequisite_calls
    )


@pytest.mark.parametrize("ordered_level", [0, "invalid"])
def test_pol_011_still_requires_fan_one_without_sufficient_ordered_action(
    ordered_level: int | str,
) -> None:
    decision = POLICY.evaluate(
        PolicyRequest(
            action=OfficialToolCall(
                tool_name="set_air_conditioning", arguments={"on": True}
            ),
            ordered_actions=[
                OfficialToolCall(
                    tool_name="set_fan_speed", arguments={"level": ordered_level}
                )
            ],
            authorization=ActionAuthorization.USER_REQUEST,
            state={
                "fan_speed": 0,
                "window_driver_position": 0,
                "window_passenger_position": 0,
                "window_driver_rear_position": 0,
                "window_passenger_rear_position": 0,
            },
        )
    )

    assert any(
        call.semantic_operation == "vehicle.fan.set_speed"
        and call.arguments == {"level": 1}
        for call in decision.prerequisite_calls
    )


def test_sunroof_policy_reuses_recipe_canonical_sunshade_evidence() -> None:
    decision = POLICY.evaluate(
        action_request(
            "open_close_sunroof",
            {"percentage": 50},
            state={
                "sunshade_open_percentage": 100,
                "weather_condition": "sunny",
            },
        )
    )

    assert decision.required_reads == []
    assert decision.prerequisite_calls == []


def test_confirmation_covers_policy_side_effect_and_requested_action() -> None:
    decision = POLICY.evaluate(
        action_request(
            "set_head_lights_high_beams",
            {"on": True},
            state={"fog_lights": True},
        )
    )

    assert decision.confirmation is not None
    assert [
        call.semantic_operation for call in decision.confirmation.bundle_operations
    ] == ["vehicle.fog_lights.set", "set_head_lights_high_beams"]
    assert decision.confirmation.include_full_bundle
