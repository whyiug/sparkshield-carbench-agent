"""Deterministic validation of state-changing tool completion results.

Transport success is not enough to complete a goal.  A known SET contract must
also return values that agree with the requested state change.  Unknown SET
contracts remain deliberately weak so a newly introduced tool cannot create a
false completion claim merely by returning ``status=SUCCESS``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..domain.types import DomainModel, NonEmptyStr


class SetCompletionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    INCONSISTENT = "inconsistent"
    UNVERIFIABLE = "unverifiable"


class SetCompletionResult(DomainModel):
    """The semantic completion decision for one emitted SET call."""

    status: SetCompletionStatus
    reason: NonEmptyStr
    returned_values: dict[str, Any]
    weak: bool = False
    terminal_without_result: bool = False

    @property
    def can_mark_goal_done(self) -> bool:
        return self.status is SetCompletionStatus.SUCCESS and not self.weak


class _Comparison(str, Enum):
    EXACT = "exact"
    TRUE = "true"
    FALSE = "false"
    EMPTY = "empty"
    CONTAINS = "contains"
    ABSENT = "absent"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class _FieldRule:
    result_fields: tuple[str, ...]
    comparison: _Comparison
    argument: str | None = None
    optional_argument: bool = False

    @property
    def label(self) -> str:
        return self.argument or self.result_fields[0]


@dataclass(frozen=True, slots=True)
class _CompletionSpec:
    rules: tuple[_FieldRule, ...]
    marker_covers_unmapped_arguments: bool = False


def _exact(argument: str, *result_fields: str, optional: bool = False) -> _FieldRule:
    return _FieldRule(
        tuple(result_fields or (argument,)),
        _Comparison.EXACT,
        argument,
        optional_argument=optional,
    )


def _contains(argument: str, *result_fields: str, optional: bool = False) -> _FieldRule:
    return _FieldRule(
        tuple(result_fields),
        _Comparison.CONTAINS,
        argument,
        optional_argument=optional,
    )


def _absent(argument: str, *result_fields: str) -> _FieldRule:
    return _FieldRule(tuple(result_fields), _Comparison.ABSENT, argument)


def _last(argument: str, *result_fields: str, optional: bool = False) -> _FieldRule:
    return _FieldRule(
        tuple(result_fields),
        _Comparison.LAST,
        argument,
        optional_argument=optional,
    )


def _constant(comparison: _Comparison, *result_fields: str) -> _FieldRule:
    return _FieldRule(tuple(result_fields), comparison)


_SPECS: dict[str, _CompletionSpec] = {
    "open_close_window": _CompletionSpec((_exact("window"), _exact("percentage"))),
    "open_close_sunroof": _CompletionSpec((_exact("percentage"),)),
    "open_close_sunshade": _CompletionSpec((_exact("percentage"),)),
    "open_close_trunk_door": _CompletionSpec((_exact("action"),)),
    "set_air_conditioning": _CompletionSpec((_exact("on"),)),
    "set_air_circulation": _CompletionSpec((_exact("mode"),)),
    "set_climate_temperature": _CompletionSpec(
        (_exact("temperature"), _exact("seat_zone"))
    ),
    "set_fan_speed": _CompletionSpec((_exact("level"),)),
    "set_fan_airflow_direction": _CompletionSpec((_exact("direction"),)),
    "set_window_defrost": _CompletionSpec((_exact("on"), _exact("defrost_window"))),
    "set_seat_heating": _CompletionSpec((_exact("level"), _exact("seat_zone"))),
    "set_steering_wheel_heating": _CompletionSpec((_exact("level"),)),
    "set_fog_lights": _CompletionSpec((_exact("on"),)),
    "set_head_lights_low_beams": _CompletionSpec((_exact("on"),)),
    "set_head_lights_high_beams": _CompletionSpec((_exact("on"),)),
    # The current result reports the selected light but not whether it was
    # turned on or off.  The unmapped ``on`` argument therefore keeps this
    # contract unverifiable instead of creating a false completion.
    "set_reading_light": _CompletionSpec((_exact("position"),)),
    "set_ambient_lights": _CompletionSpec(
        (_exact("on"), _exact("lightcolor", "lightcolor", "light_color"))
    ),
    "set_new_navigation": _CompletionSpec(
        (_constant(_Comparison.TRUE, "navigation_set"),),
        marker_covers_unmapped_arguments=True,
    ),
    "delete_current_navigation": _CompletionSpec(
        (
            _constant(_Comparison.FALSE, "navigation_active"),
            _constant(_Comparison.EMPTY, "new_waypoints", "new_waypoints_id"),
            _constant(_Comparison.EMPTY, "new_routes", "new_routes_id"),
        )
    ),
    "navigation_add_one_waypoint": _CompletionSpec(
        (
            _constant(_Comparison.TRUE, "waypoint_added"),
            _contains("waypoint_id_to_add", "new_waypoints_id", "new_waypoints"),
            _contains(
                "waypoint_id_after_new_waypoint",
                "new_waypoints_id",
                "new_waypoints",
                optional=True,
            ),
            _contains(
                "route_id_leading_to_new_waypoint", "new_routes_id", "new_routes"
            ),
            _contains(
                "route_id_leading_away_from_new_waypoint",
                "new_routes_id",
                "new_routes",
                optional=True,
            ),
            _exact(
                "expected_post_waypoint_ids",
                "new_waypoints_id",
                "new_waypoints",
                optional=True,
            ),
            _exact(
                "expected_post_route_ids",
                "new_routes_id",
                "new_routes",
                optional=True,
            ),
        ),
        marker_covers_unmapped_arguments=True,
    ),
    "navigation_delete_destination": _CompletionSpec(
        (
            _constant(_Comparison.TRUE, "destination_deleted"),
            _absent("destination_id_to_delete", "new_waypoints", "new_waypoints_id"),
        )
    ),
    "navigation_delete_waypoint": _CompletionSpec(
        (
            _constant(_Comparison.TRUE, "waypoint_deleted"),
            _absent("waypoint_id_to_delete", "new_waypoints", "new_waypoints_id"),
            _contains("route_id_without_waypoint", "new_routes", "new_routes_id"),
            _exact(
                "expected_post_waypoint_ids",
                "new_waypoints",
                "new_waypoints_id",
                optional=True,
            ),
            _exact(
                "expected_post_route_ids",
                "new_routes",
                "new_routes_id",
                optional=True,
            ),
        )
    ),
    "navigation_replace_final_destination": _CompletionSpec(
        (
            _constant(_Comparison.TRUE, "destination_replaced"),
            _last("new_destination_id", "new_waypoints", "new_waypoints_id"),
            _last("route_id_leading_to_new_destination", "new_routes", "new_routes_id"),
            _exact(
                "expected_post_waypoint_ids",
                "new_waypoints",
                "new_waypoints_id",
                optional=True,
            ),
            _exact(
                "expected_post_route_ids",
                "new_routes",
                "new_routes_id",
                optional=True,
            ),
        )
    ),
    "navigation_replace_one_waypoint": _CompletionSpec(
        (
            _constant(_Comparison.TRUE, "waypoint_replaced"),
            _absent("waypoint_id_to_replace", "new_waypoints", "new_waypoints_id"),
            _contains("new_waypoint_id", "new_waypoints", "new_waypoints_id"),
            _contains(
                "route_id_leading_to_new_waypoint", "new_routes", "new_routes_id"
            ),
            _contains(
                "route_id_leading_away_from_new_waypoint",
                "new_routes",
                "new_routes_id",
            ),
            _exact(
                "expected_post_waypoint_ids",
                "new_waypoints",
                "new_waypoints_id",
                optional=True,
            ),
            _exact(
                "expected_post_route_ids",
                "new_routes",
                "new_routes_id",
                optional=True,
            ),
        )
    ),
    "send_email": _CompletionSpec(
        (_constant(_Comparison.TRUE, "email_sent"),),
        marker_covers_unmapped_arguments=True,
    ),
    "call_phone_by_number": _CompletionSpec(
        (_constant(_Comparison.TRUE, "phone_number_called"),),
        marker_covers_unmapped_arguments=True,
    ),
}


TERMINAL_WITHOUT_RESULT_TOOLS = frozenset({"call_phone_by_number"})


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _completion_equal(left: Any, right: Any) -> bool:
    if type(left) in {int, float} and type(right) in {int, float}:
        return bool(left == right)
    return _strict_equal(left, right)


def _is_unknown(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and value.strip().lower() in {"", "unknown", "null", "none"}
    )


def _parse_payload(raw_result: Any) -> tuple[dict[str, Any] | None, str | None]:
    value = raw_result
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None, "result_is_not_utf8"
    if isinstance(value, str):
        if not value.strip():
            return None, "result_is_empty"
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, "result_is_malformed_json"
    if not isinstance(value, Mapping):
        return None, "result_is_not_an_object"
    return dict(value), None


def _returned_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    return dict(result) if isinstance(result, Mapping) else {}


def _field_value(
    returned: Mapping[str, Any], fields: tuple[str, ...]
) -> tuple[bool, Any]:
    for field in fields:
        if field in returned:
            return True, returned[field]
    return False, None


class SetCompletionValidator:
    """Validate an observed SET result against a fixed public result contract."""

    @staticmethod
    def is_terminal_without_result(tool_name: str) -> bool:
        return tool_name in TERMINAL_WITHOUT_RESULT_TOOLS

    @classmethod
    def expects_separate_result(cls, tool_name: str) -> bool:
        return not cls.is_terminal_without_result(tool_name)

    def validate(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        raw_result: Any,
    ) -> SetCompletionResult:
        terminal = self.is_terminal_without_result(tool_name)
        payload, parse_issue = _parse_payload(raw_result)
        if payload is None:
            reason = (
                "terminal_side_effect_has_no_separate_result"
                if terminal and raw_result is None
                else parse_issue or "result_is_unavailable"
            )
            return SetCompletionResult(
                status=SetCompletionStatus.UNVERIFIABLE,
                reason=reason,
                returned_values={},
                weak=tool_name not in _SPECS or terminal,
                terminal_without_result=terminal,
            )

        returned = _returned_mapping(payload)
        status = payload.get("status")
        has_errors = "errors" in payload and not _is_unknown(payload["errors"])
        if (
            has_errors
            and isinstance(status, str)
            and status.strip().upper() == "SUCCESS"
        ):
            return SetCompletionResult(
                status=SetCompletionStatus.INCONSISTENT,
                reason="success_status_conflicts_with_errors",
                returned_values=returned,
                terminal_without_result=terminal,
            )
        if has_errors and (
            not isinstance(status, str) or status.strip().upper() != "SUCCESS"
        ):
            return SetCompletionResult(
                status=SetCompletionStatus.FAILURE,
                reason="tool_reported_failure",
                returned_values=returned,
                terminal_without_result=terminal,
            )
        if not isinstance(status, str) or not status.strip():
            return SetCompletionResult(
                status=SetCompletionStatus.UNVERIFIABLE,
                reason="top_level_status_missing",
                returned_values=returned,
                weak=tool_name not in _SPECS,
                terminal_without_result=terminal,
            )
        normalized_status = status.strip().upper()
        if normalized_status in {"FAILURE", "FAILED", "ERROR"}:
            return SetCompletionResult(
                status=SetCompletionStatus.FAILURE,
                reason="tool_reported_failure",
                returned_values=returned,
                terminal_without_result=terminal,
            )
        if normalized_status != "SUCCESS":
            return SetCompletionResult(
                status=SetCompletionStatus.UNVERIFIABLE,
                reason="top_level_status_is_not_recognized",
                returned_values=returned,
                weak=tool_name not in _SPECS,
                terminal_without_result=terminal,
            )
        if "result" not in payload or _is_unknown(payload["result"]):
            return SetCompletionResult(
                status=SetCompletionStatus.UNVERIFIABLE,
                reason="success_result_missing_or_unknown",
                returned_values={},
                weak=tool_name not in _SPECS,
                terminal_without_result=terminal,
            )

        spec = _SPECS.get(tool_name)
        if spec is None:
            return SetCompletionResult(
                status=SetCompletionStatus.UNVERIFIABLE,
                reason="unknown_set_completion_contract",
                returned_values=returned,
                weak=True,
                terminal_without_result=terminal,
            )
        if not isinstance(payload["result"], Mapping):
            return SetCompletionResult(
                status=SetCompletionStatus.UNVERIFIABLE,
                reason="success_result_is_not_an_object",
                returned_values={},
                terminal_without_result=terminal,
            )

        covered_arguments: set[str] = set()
        for rule in spec.rules:
            if rule.argument is not None:
                if rule.optional_argument and (
                    rule.argument not in arguments or arguments[rule.argument] is None
                ):
                    continue
                if rule.argument not in arguments:
                    return SetCompletionResult(
                        status=SetCompletionStatus.UNVERIFIABLE,
                        reason=f"call_argument_missing:{rule.argument}",
                        returned_values=returned,
                        terminal_without_result=terminal,
                    )
                covered_arguments.add(rule.argument)

            exists, actual = _field_value(returned, rule.result_fields)
            if not exists or _is_unknown(actual):
                return SetCompletionResult(
                    status=SetCompletionStatus.UNVERIFIABLE,
                    reason=f"completion_field_missing_or_unknown:{rule.label}",
                    returned_values=returned,
                    terminal_without_result=terminal,
                )
            if not self._rule_matches(rule, actual, arguments):
                return SetCompletionResult(
                    status=SetCompletionStatus.INCONSISTENT,
                    reason=f"completion_value_mismatch:{rule.label}",
                    returned_values=returned,
                    terminal_without_result=terminal,
                )

        unmapped = set(arguments).difference(covered_arguments)
        if unmapped and not spec.marker_covers_unmapped_arguments:
            return SetCompletionResult(
                status=SetCompletionStatus.UNVERIFIABLE,
                reason=f"result_does_not_report_target_arguments:{','.join(sorted(unmapped))}",
                returned_values=returned,
                weak=True,
                terminal_without_result=terminal,
            )
        if tool_name == "navigation_delete_waypoint" and not (
            self._valid_navigation_delete_result(returned)
        ):
            return SetCompletionResult(
                status=SetCompletionStatus.INCONSISTENT,
                reason="navigation_delete_result_structure_is_inconsistent",
                returned_values=returned,
                terminal_without_result=terminal,
            )
        return SetCompletionResult(
            status=SetCompletionStatus.SUCCESS,
            reason="result_matches_requested_state_change",
            returned_values=returned,
            terminal_without_result=terminal,
        )

    @staticmethod
    def _valid_navigation_delete_result(returned: Mapping[str, Any]) -> bool:
        waypoint_exists, waypoints = _field_value(
            returned, ("new_waypoints", "new_waypoints_id")
        )
        route_exists, routes = _field_value(returned, ("new_routes", "new_routes_id"))
        if (
            not waypoint_exists
            or not route_exists
            or not isinstance(waypoints, (list, tuple))
            or not isinstance(routes, (list, tuple))
            or len(waypoints) < 2
            or len(routes) != len(waypoints) - 1
            or any(not isinstance(value, str) or not value for value in waypoints)
            or any(not isinstance(value, str) or not value for value in routes)
        ):
            return False
        return len(set(waypoints)) == len(waypoints) and len(set(routes)) == len(routes)

    @staticmethod
    def _rule_matches(
        rule: _FieldRule, actual: Any, arguments: Mapping[str, Any]
    ) -> bool:
        if rule.comparison is _Comparison.TRUE:
            return type(actual) is bool and actual is True
        if rule.comparison is _Comparison.FALSE:
            return type(actual) is bool and actual is False
        if rule.comparison is _Comparison.EMPTY:
            return isinstance(actual, (list, tuple, dict, str)) and not actual

        assert rule.argument is not None
        expected = arguments[rule.argument]
        if rule.comparison is _Comparison.EXACT:
            return _completion_equal(actual, expected)
        if not isinstance(actual, Sequence) or isinstance(
            actual, (str, bytes, bytearray)
        ):
            return False
        if rule.comparison is _Comparison.CONTAINS:
            return any(_completion_equal(item, expected) for item in actual)
        if rule.comparison is _Comparison.ABSENT:
            return not any(_completion_equal(item, expected) for item in actual)
        if rule.comparison is _Comparison.LAST:
            return bool(actual) and _completion_equal(actual[-1], expected)
        return False


__all__ = [
    "SetCompletionResult",
    "SetCompletionStatus",
    "SetCompletionValidator",
    "TERMINAL_WITHOUT_RESULT_TOOLS",
]
