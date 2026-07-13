from __future__ import annotations

import json

import pytest

from track_1_agent_under_test.car_guard.runtime import (
    SetCompletionStatus,
    SetCompletionValidator,
)


@pytest.fixture
def validator() -> SetCompletionValidator:
    return SetCompletionValidator()


@pytest.mark.parametrize(
    ("tool_name", "arguments", "returned"),
    [
        (
            "open_close_window",
            {"window": "DRIVER", "percentage": 25},
            {"window": "DRIVER", "percentage": 25},
        ),
        ("open_close_sunroof", {"percentage": 100}, {"percentage": 100}),
        ("open_close_sunshade", {"percentage": 0}, {"percentage": 0}),
        ("open_close_trunk_door", {"action": "OPEN"}, {"action": "OPEN"}),
        ("set_air_conditioning", {"on": True}, {"on": True}),
        ("set_air_circulation", {"mode": "RECIRCULATE"}, {"mode": "RECIRCULATE"}),
        (
            "set_climate_temperature",
            {"temperature": 21, "seat_zone": "DRIVER"},
            {"temperature": 21, "seat_zone": "DRIVER"},
        ),
        ("set_fan_speed", {"level": 2}, {"level": 2}),
        (
            "set_fan_airflow_direction",
            {"direction": "WINDSHIELD"},
            {"direction": "WINDSHIELD"},
        ),
        (
            "set_window_defrost",
            {"on": True, "defrost_window": "FRONT"},
            {"on": True, "defrost_window": "FRONT"},
        ),
        (
            "set_seat_heating",
            {"level": 3, "seat_zone": "DRIVER"},
            {"level": 3, "seat_zone": "DRIVER"},
        ),
        ("set_steering_wheel_heating", {"level": 1}, {"level": 1}),
        ("set_fog_lights", {"on": False}, {"on": False}),
        ("set_head_lights_low_beams", {"on": True}, {"on": True}),
        ("set_head_lights_high_beams", {"on": False}, {"on": False}),
        (
            "set_ambient_lights",
            {"on": True, "lightcolor": "BLUE"},
            {"on": True, "light_color": "BLUE"},
        ),
    ],
)
def test_known_vehicle_set_requires_explicit_matching_result(
    validator: SetCompletionValidator,
    tool_name: str,
    arguments: dict[str, object],
    returned: dict[str, object],
) -> None:
    completion = validator.validate(
        tool_name=tool_name,
        arguments=arguments,
        raw_result=json.dumps({"status": "SUCCESS", "result": returned}),
    )

    assert completion.status is SetCompletionStatus.SUCCESS
    assert completion.can_mark_goal_done
    assert not completion.weak
    assert completion.returned_values == returned


def test_success_envelope_with_different_target_value_is_inconsistent(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="open_close_window",
        arguments={"window": "DRIVER", "percentage": 25},
        raw_result={
            "status": "SUCCESS",
            "result": {"window": "DRIVER", "percentage": 80},
        },
    )

    assert completion.status is SetCompletionStatus.INCONSISTENT
    assert completion.reason == "completion_value_mismatch:percentage"
    assert not completion.can_mark_goal_done


def test_value_comparison_is_type_strict(validator: SetCompletionValidator) -> None:
    completion = validator.validate(
        tool_name="set_air_conditioning",
        arguments={"on": True},
        raw_result={"status": "SUCCESS", "result": {"on": 1}},
    )

    assert completion.status is SetCompletionStatus.INCONSISTENT
    assert not completion.can_mark_goal_done


def test_json_number_int_and_float_forms_are_completion_equivalent(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="open_close_sunshade",
        arguments={"percentage": 100},
        raw_result={"status": "SUCCESS", "result": {"percentage": 100.0}},
    )

    assert completion.status is SetCompletionStatus.SUCCESS
    assert completion.can_mark_goal_done


@pytest.mark.parametrize(
    "raw_result",
    [
        {"status": "FAILURE", "errors": {"SET_001": "rejected"}},
        # Some public navigation failure paths omit status but return errors.
        {"errors": {"NAV_ADD_WP_001": "rejected"}},
    ],
)
def test_explicit_failure_never_completes_goal(
    validator: SetCompletionValidator, raw_result: dict[str, object]
) -> None:
    completion = validator.validate(
        tool_name="set_fan_speed",
        arguments={"level": 2},
        raw_result=raw_result,
    )

    assert completion.status is SetCompletionStatus.FAILURE
    assert not completion.can_mark_goal_done


def test_success_status_with_errors_is_inconsistent(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="set_fan_speed",
        arguments={"level": 2},
        raw_result={
            "status": "SUCCESS",
            "result": {"level": 2},
            "errors": {"SET_FAN_001": "state update failed"},
        },
    )

    assert completion.status is SetCompletionStatus.INCONSISTENT
    assert completion.reason == "success_status_conflicts_with_errors"
    assert not completion.can_mark_goal_done


@pytest.mark.parametrize(
    ("raw_result", "reason"),
    [
        ("{broken", "result_is_malformed_json"),
        ("Error: tool invocation raised", "result_is_malformed_json"),
        ({"result": {"level": 2}}, "top_level_status_missing"),
        (
            {"status": "UNKNOWN", "result": {"level": 2}},
            "top_level_status_is_not_recognized",
        ),
        ({"status": "SUCCESS"}, "success_result_missing_or_unknown"),
        ({"status": "SUCCESS", "result": None}, "success_result_missing_or_unknown"),
        (
            {"status": "SUCCESS", "result": {"level": "unknown"}},
            "completion_field_missing_or_unknown:level",
        ),
    ],
)
def test_malformed_missing_and_unknown_results_are_unverifiable(
    validator: SetCompletionValidator, raw_result: object, reason: str
) -> None:
    completion = validator.validate(
        tool_name="set_fan_speed",
        arguments={"level": 2},
        raw_result=raw_result,
    )

    assert completion.status is SetCompletionStatus.UNVERIFIABLE
    assert completion.reason == reason
    assert not completion.can_mark_goal_done


def test_unknown_set_success_is_weak_and_cannot_auto_complete(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="future_state_changing_tool",
        arguments={"mode": "A"},
        raw_result={"status": "SUCCESS", "result": {"mode": "A"}},
    )

    assert completion.status is SetCompletionStatus.UNVERIFIABLE
    assert completion.reason == "unknown_set_completion_contract"
    assert completion.weak
    assert not completion.can_mark_goal_done
    assert completion.returned_values == {"mode": "A"}


def test_reading_light_does_not_complete_without_returned_on_state(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="set_reading_light",
        arguments={"position": "DRIVER", "on": True},
        raw_result={"status": "SUCCESS", "result": {"position": "DRIVER"}},
    )

    assert completion.status is SetCompletionStatus.UNVERIFIABLE
    assert completion.reason == "result_does_not_report_target_arguments:on"
    assert completion.weak
    assert not completion.can_mark_goal_done


def test_email_uses_explicit_operation_completion_marker(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="send_email",
        arguments={
            "content_message": "Meeting moved to ten.",
            "email_addresses": ["driver@example.test"],
        },
        raw_result={"status": "SUCCESS", "result": {"email_sent": True}},
    )

    assert completion.status is SetCompletionStatus.SUCCESS
    assert completion.can_mark_goal_done


def test_new_navigation_requires_explicit_navigation_set_marker(
    validator: SetCompletionValidator,
) -> None:
    success = validator.validate(
        tool_name="set_new_navigation",
        arguments={"route_ids": ["route_1", "route_2"]},
        raw_result={
            "status": "SUCCESS",
            "result": {
                "navigation_set": True,
                "start_id": "loc_1",
                "waypoints": ["loc_2", "loc_3"],
                "destination_id": "loc_3",
            },
        },
    )
    inconsistent = validator.validate(
        tool_name="set_new_navigation",
        arguments={"route_ids": ["route_1", "route_2"]},
        raw_result={"status": "SUCCESS", "result": {"navigation_set": False}},
    )

    assert success.can_mark_goal_done
    assert inconsistent.status is SetCompletionStatus.INCONSISTENT
    assert not inconsistent.can_mark_goal_done


def test_navigation_add_validates_returned_waypoint_and_route_aliases(
    validator: SetCompletionValidator,
) -> None:
    arguments = {
        "waypoint_id_to_add": "loc_added",
        "waypoint_id_before_new_waypoint": "loc_start",
        "waypoint_id_after_new_waypoint": "loc_end",
        "route_id_leading_to_new_waypoint": "route_in",
        "route_id_leading_away_from_new_waypoint": "route_out",
    }
    completion = validator.validate(
        tool_name="navigation_add_one_waypoint",
        arguments=arguments,
        raw_result={
            "status": "SUCCESS",
            "result": {
                "waypoint_added": True,
                "new_waypoints_id": ["loc_added", "loc_end"],
                "new_routes_id": ["route_in", "route_out"],
            },
        },
    )

    assert completion.status is SetCompletionStatus.SUCCESS
    assert completion.can_mark_goal_done


@pytest.mark.parametrize(
    ("tool_name", "arguments", "returned"),
    [
        (
            "navigation_add_one_waypoint",
            {
                "waypoint_id_to_add": "loc_gamma",
                "waypoint_id_before_new_waypoint": "loc_omega",
                "route_id_leading_to_new_waypoint": "rsy_omega_gamma",
                "expected_post_waypoint_ids": [
                    "loc_alpha",
                    "loc_omega",
                    "loc_gamma",
                ],
                "expected_post_route_ids": [
                    "rsy_alpha_omega",
                    "rsy_omega_gamma",
                ],
            },
            {
                "waypoint_added": True,
                "new_waypoints_id": ["loc_omega", "loc_gamma"],
                "new_routes_id": ["rsy_alpha_omega", "rsy_omega_gamma"],
            },
        ),
        (
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": "loc_beta",
                "new_waypoint_id": "loc_gamma",
                "route_id_leading_to_new_waypoint": "rsy_alpha_gamma",
                "route_id_leading_away_from_new_waypoint": "rsy_gamma_omega",
                "expected_post_waypoint_ids": [
                    "loc_alpha",
                    "loc_gamma",
                    "loc_omega",
                ],
                "expected_post_route_ids": [
                    "rsy_alpha_gamma",
                    "rsy_gamma_omega",
                ],
            },
            {
                "waypoint_replaced": True,
                "new_waypoints": ["loc_alpha", "loc_gamma", "loc_extra"],
                "new_routes": ["rsy_alpha_gamma", "rsy_gamma_omega"],
            },
        ),
    ],
)
def test_navigation_edit_rejects_non_exact_returned_post_state(
    validator: SetCompletionValidator,
    tool_name: str,
    arguments: dict[str, object],
    returned: dict[str, object],
) -> None:
    completion = validator.validate(
        tool_name=tool_name,
        arguments=arguments,
        raw_result={"status": "SUCCESS", "result": returned},
    )

    assert completion.status is SetCompletionStatus.INCONSISTENT
    assert not completion.can_mark_goal_done


def test_navigation_waypoint_delete_requires_consistent_post_state(
    validator: SetCompletionValidator,
) -> None:
    arguments = {
        "waypoint_id_to_delete": "loc_removed",
        "route_id_without_waypoint": "route_direct",
        "expected_post_waypoint_ids": ["loc_start", "loc_destination"],
        "expected_post_route_ids": ["route_direct"],
    }
    valid = validator.validate(
        tool_name="navigation_delete_waypoint",
        arguments=arguments,
        raw_result={
            "status": "SUCCESS",
            "result": {
                "waypoint_deleted": True,
                "new_waypoints": ["loc_start", "loc_destination"],
                "new_routes": ["route_direct"],
            },
        },
    )
    empty_waypoints = validator.validate(
        tool_name="navigation_delete_waypoint",
        arguments=arguments,
        raw_result={
            "status": "SUCCESS",
            "result": {
                "waypoint_deleted": True,
                "new_waypoints": [],
                "new_routes": ["route_direct"],
            },
        },
    )
    wrong_destination = validator.validate(
        tool_name="navigation_delete_waypoint",
        arguments=arguments,
        raw_result={
            "status": "SUCCESS",
            "result": {
                "waypoint_deleted": True,
                "new_waypoints": ["loc_start", "loc_other"],
                "new_routes": ["route_direct"],
            },
        },
    )
    multi_stop = validator.validate(
        tool_name="navigation_delete_waypoint",
        arguments={
            **arguments,
            "expected_post_waypoint_ids": [
                "loc_start",
                "loc_next",
                "loc_destination",
            ],
            "expected_post_route_ids": ["route_direct", "route_next"],
        },
        raw_result={
            "status": "SUCCESS",
            "result": {
                "waypoint_deleted": True,
                "new_waypoints": [
                    "loc_start",
                    "loc_next",
                    "loc_destination",
                ],
                "new_routes": ["route_direct", "route_next"],
            },
        },
    )

    assert valid.can_mark_goal_done
    assert multi_stop.can_mark_goal_done
    assert empty_waypoints.status is SetCompletionStatus.INCONSISTENT
    assert wrong_destination.status is SetCompletionStatus.INCONSISTENT


def test_navigation_replace_rejects_wrong_final_destination(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="navigation_replace_final_destination",
        arguments={
            "new_destination_id": "loc_expected",
            "route_id_leading_to_new_destination": "route_expected",
        },
        raw_result={
            "status": "SUCCESS",
            "result": {
                "destination_replaced": True,
                "new_waypoints": ["loc_other"],
                "new_routes": ["route_expected"],
            },
        },
    )

    assert completion.status is SetCompletionStatus.INCONSISTENT
    assert completion.reason == "completion_value_mismatch:new_destination_id"
    assert not completion.can_mark_goal_done


def test_delete_navigation_requires_inactive_empty_state(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="delete_current_navigation",
        arguments={},
        raw_result={
            "status": "SUCCESS",
            "result": {
                "navigation_active": False,
                "new_waypoints": [],
                "new_routes": [],
            },
        },
    )

    assert completion.status is SetCompletionStatus.SUCCESS
    assert completion.can_mark_goal_done


def test_phone_terminal_without_result_is_not_fabricated_as_success(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="call_phone_by_number",
        arguments={"phone_number": "+1-555-0100"},
        raw_result=None,
    )

    assert validator.is_terminal_without_result("call_phone_by_number")
    assert not validator.expects_separate_result("call_phone_by_number")
    assert completion.terminal_without_result
    assert completion.status is SetCompletionStatus.UNVERIFIABLE
    assert completion.reason == "terminal_side_effect_has_no_separate_result"
    assert not completion.can_mark_goal_done


def test_phone_ack_can_be_validated_if_a_result_is_delivered(
    validator: SetCompletionValidator,
) -> None:
    completion = validator.validate(
        tool_name="call_phone_by_number",
        arguments={"phone_number": "+1-555-0100"},
        raw_result={
            "status": "SUCCESS",
            "result": {"phone_number_called": True},
        },
    )

    assert completion.terminal_without_result
    assert completion.status is SetCompletionStatus.SUCCESS
    assert completion.can_mark_goal_done
