from __future__ import annotations

import inspect

import pytest

from track_1_agent_under_test.car_guard.a2a.session_store import SessionState
from track_1_agent_under_test.car_guard.domain import (
    AmbiguitySlot,
    Candidate,
    Goal,
    IntentFrame,
    IntentKind,
)
from track_1_agent_under_test.car_guard.planning import (
    focus_explicit_action_request,
    ground_intent,
    has_explicit_action_request,
    recover_battery_charge_trip_range_intent,
    recover_current_destination_poi_search_intent,
    recover_current_day_calendar_intent,
    recover_driver_warming_intent,
    recover_named_navigation_final_destination_delete_intent,
    recover_named_navigation_waypoint_delete_intent,
    recover_navigation_waypoint_context_intent,
    recover_relative_occupied_seat_heating_intent,
)
from track_1_agent_under_test.car_guard.planning.intent_grounding import (
    recover_named_location_poi_search_intent,
    recover_named_navigation_destination_replacement_intent,
)
from track_1_agent_under_test.car_guard.recipes import RecipeRegistry


REGISTRY = RecipeRegistry()


def frame(
    *goals: Goal,
    slots: dict[str, object] | None = None,
    kind: IntentKind = IntentKind.ACTION,
) -> IntentFrame:
    return IntentFrame(
        language="en",
        call_for_action=True,
        goals=list(goals),
        explicit_slots=slots or {},
        intent_kind=kind,
    )


def goal(
    goal_id: str,
    semantic_operation: str,
    desired_outcome: dict[str, object],
    *,
    depends_on: list[str] | None = None,
) -> Goal:
    return Goal(
        goal_id=goal_id,
        semantic_operation=semantic_operation,
        desired_outcome=desired_outcome,
        depends_on=depends_on or [],
    )


def test_grounding_api_is_inventory_blind() -> None:
    parameters = inspect.signature(ground_intent).parameters

    assert tuple(parameters) == ("raw_user_text", "intent", "registry")
    assert "live_tools" not in parameters
    assert "catalog" not in parameters


def test_grounding_does_not_mutate_frame_or_registry() -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 2}))
    before_frame = extracted.model_dump(mode="json")
    before_registry_hash = REGISTRY.registry_hash

    ground_intent("Set the fan to level two", extracted, REGISTRY)

    assert extracted.model_dump(mode="json") == before_frame
    assert REGISTRY.registry_hash == before_registry_hash


def test_unrelated_time_question_drops_hallucinated_fan_set() -> None:
    extracted = frame(
        goal("fan", "set_fan_speed", {"level": 5}),
        slots={"level": 5},
    )

    result = ground_intent("What time is it?", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.filtered_intent.call_for_action is False
    assert result.filtered_intent.intent_kind is IntentKind.INFORMATION
    assert result.filtered_intent.explicit_slots == {}
    assert result.authorized_action_goal_ids == frozenset()
    assert result.desired_values_by_goal == {}


def test_fan_speed_question_never_authorizes_write_goal() -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 5}))

    result = ground_intent("What is the fan speed?", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.filtered_intent.call_for_action is False
    assert result.authorized_action_goal_ids == frozenset()


def test_fan_speed_question_can_retain_an_invariant_read_goal() -> None:
    extracted = frame(
        goal("fan-read", "read_climate_settings", {}),
        kind=IntentKind.INFORMATION,
    )

    result = ground_intent("What is the fan speed?", extracted, REGISTRY)

    assert [item.goal_id for item in result.filtered_intent.goals] == ["fan-read"]
    assert result.filtered_intent.call_for_action is False
    assert result.filtered_intent.intent_kind is IntentKind.INFORMATION
    assert result.authorized_action_goal_ids == frozenset()


def test_relative_fan_speed_is_recovered_when_model_omits_the_goal() -> None:
    extracted = frame(
        goal(
            "airflow",
            "set_fan_airflow_direction",
            {"direction": "FEET"},
        ),
        slots={"direction": "FEET"},
    )

    result = ground_intent(
        (
            "Could you change the fan so the air blows towards my feet? "
            "And could you bump up the fan speed a little bit, just one level?"
        ),
        extracted,
        REGISTRY,
    )

    assert [item.semantic_operation for item in result.filtered_intent.goals] == [
        "set_fan_airflow_direction",
        "set_fan_speed",
    ]
    relative_goal = result.filtered_intent.goals[1]
    assert relative_goal.desired_outcome == {}
    assert result.relative_fan_speed_deltas_by_goal == {relative_goal.goal_id: 1}
    assert relative_goal.goal_id in result.authorized_action_goal_ids


def test_relative_fan_speed_discards_model_guessed_absolute_level() -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 2}), slots={"level": 2})

    result = ground_intent(
        "Please increase the fan speed level by two.", extracted, REGISTRY
    )

    assert result.filtered_intent.goals[0].desired_outcome == {}
    assert result.filtered_intent.explicit_slots == {}
    assert result.relative_fan_speed_deltas_by_goal == {"fan": 2}


def test_relative_climate_temperature_is_recovered_in_compound_request() -> None:
    extracted = frame(
        goal("ac", "set_air_conditioning", {"enabled": True}),
        slots={"enabled": True},
    )

    result = ground_intent(
        (
            "Turn on the air conditioning and lower the temperature by 4 "
            "degrees for the whole car."
        ),
        extracted,
        REGISTRY,
    )

    assert [item.semantic_operation for item in result.filtered_intent.goals] == [
        "set_air_conditioning",
        "set_climate_temperature",
    ]
    temperature_goal = result.filtered_intent.goals[1]
    assert temperature_goal.desired_outcome == {"seat_zone": "ALL_ZONES"}
    assert result.relative_climate_temperature_deltas_by_goal == {
        temperature_goal.goal_id: -4
    }
    assert result.authorized_action_goal_ids == frozenset(
        {"ac", temperature_goal.goal_id}
    )


def test_relative_temperature_normalizes_real_model_whole_car_value() -> None:
    extracted = frame(
        goal("ac", "set_air_conditioning", {"enabled": True}),
        goal(
            "temperature",
            "set_climate_temperature",
            {"seat_zone": "whole_car"},
        ),
        slots={"enabled": True, "seat_zone": "whole_car"},
    )

    result = ground_intent(
        (
            "Turn on the air conditioning and lower the temperature by 4 "
            "degrees for the whole car."
        ),
        extracted,
        REGISTRY,
    )

    assert result.desired_values_by_goal == {
        "ac": {"enabled": True},
        "temperature": {"seat_zone": "ALL_ZONES"},
    }
    assert result.relative_climate_temperature_deltas_by_goal == {"temperature": -4}


BASE60_INITIAL_UTTERANCE = (
    "I'm feeling a bit cold. Can you set the driver's temperature to 24 degrees "
    "and also turn on the seat heating and steering wheel heating for me?"
)
BASE60_LEVEL_UTTERANCE = (
    "Could you set the seat heating and steering wheel heating to level 2 instead?"
)
HALL58_MATCH_UTTERANCE = (
    "Thanks! Can you also turn on the seat heating for the driver's seat to level "
    "2, and set the steering wheel heating to match that level?"
)


def test_base60_initial_compound_discards_guessed_heat_levels() -> None:
    extracted = frame(
        goal(
            "climate",
            "set_climate_temperature",
            {"temperature": 24, "seat_zone": "DRIVER"},
        ),
        goal(
            "seat",
            "set_seat_heating",
            {"level": 1, "seat_zone": "DRIVER"},
        ),
        goal("steering", "set_steering_wheel_heating", {"level": 1}),
        slots={"temperature": 24, "seat_zone": "DRIVER", "level": 1},
    )

    recovered = recover_driver_warming_intent(
        BASE60_INITIAL_UTTERANCE,
        extracted,
        turn_id="base60-initial",
    )
    result = ground_intent(BASE60_INITIAL_UTTERANCE, recovered, REGISTRY)

    assert recovered.goal_mention_order == ["climate", "seat", "steering"]
    assert [item.semantic_operation for item in recovered.goals] == [
        "set_climate_temperature",
        "set_seat_heating",
        "set_steering_wheel_heating",
    ]
    assert [item.desired_outcome for item in recovered.goals] == [
        {"temperature": 24, "seat_zone": "DRIVER"},
        {"seat_zone": "DRIVER"},
        {},
    ]
    assert recovered.explicit_slots == {
        "temperature": 24,
        "seat_zone": "DRIVER",
    }
    assert result.desired_values_by_goal == {
        "climate": {"temperature": 24, "seat_zone": "DRIVER"},
        "seat": {"seat_zone": "DRIVER"},
        "steering": {},
    }
    assert result.authorized_action_goal_ids == frozenset(
        {"climate", "seat", "steering"}
    )


def test_base60_focused_action_recovers_model_omitted_heating_goals() -> None:
    focused = (
        "Can you set the driver's temperature to 24 degrees and also turn on the "
        "seat heating and steering wheel heating for me?"
    )
    extracted = frame(
        goal(
            "climate",
            "set_climate_temperature",
            {"temperature": 24, "seat_zone": "DRIVER"},
        ),
        slots={"temperature": 24, "seat_zone": "DRIVER"},
    )

    recovered = recover_driver_warming_intent(
        focused,
        extracted,
        turn_id="base60-focused",
    )
    result = ground_intent(focused, recovered, REGISTRY)

    assert [item.semantic_operation for item in result.filtered_intent.goals] == [
        "set_climate_temperature",
        "set_seat_heating",
        "set_steering_wheel_heating",
    ]
    assert result.filtered_intent.goals[1].desired_outcome == {"seat_zone": "DRIVER"}
    assert result.filtered_intent.goals[2].desired_outcome == {}


def test_base60_shared_level_followup_binds_both_heating_goals() -> None:
    extracted = frame(
        goal("seat", "set_seat_heating", {"level": 1}),
        goal("steering", "set_steering_wheel_heating", {"level": 1}),
        slots={"level": 1},
    )

    recovered = recover_driver_warming_intent(
        BASE60_LEVEL_UTTERANCE,
        extracted,
        turn_id="base60-level",
    )
    result = ground_intent(BASE60_LEVEL_UTTERANCE, recovered, REGISTRY)

    assert result.desired_values_by_goal == {
        "seat": {"level": 2},
        "steering": {"level": 2},
    }
    assert recovered.goals[1].depends_on == ["seat"]
    assert result.filtered_intent.goals[1].depends_on == ["seat"]
    assert result.derived_values_by_goal == {}


def test_hall58_match_derives_steering_level_from_unique_seat_source() -> None:
    extracted = frame(
        goal(
            "seat",
            "set_seat_heating",
            {"level": 2, "seat_zone": "DRIVER"},
        ),
        goal("steering", "set_steering_wheel_heating", {"level": 1}),
        slots={"level": 2, "seat_zone": "DRIVER"},
    )

    recovered = recover_driver_warming_intent(
        HALL58_MATCH_UTTERANCE,
        extracted,
        turn_id="hall58-match",
    )
    result = ground_intent(HALL58_MATCH_UTTERANCE, recovered, REGISTRY)

    assert result.desired_values_by_goal == {
        "seat": {"level": 2, "seat_zone": "DRIVER"},
        "steering": {"level": 2},
    }
    assert recovered.goals[1].depends_on == ["seat"]
    assert result.derived_values_by_goal == {
        "steering": {"level": "steering_level_matches_seat_heating_v1"}
    }


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Do not set the driver's temperature to 24 degrees and also turn on "
            "the seat heating and steering wheel heating for me?"
        ),
        (
            "If I feel cold, can you set the driver's temperature to 24 degrees "
            "and also turn on the seat heating and steering wheel heating for me?"
        ),
        (
            "She said, \"Can you set the driver's temperature to 24 degrees and also "
            'turn on the seat heating and steering wheel heating for me?"'
        ),
        (
            "Can you also turn on the seat heating for the driver's seat to level 2, "
            "and set the steering wheel heating to match it?"
        ),
        (
            "Can you also turn on the seat heating for the driver's seat to level 2 "
            "and the passenger's seat to level 3, and set the steering wheel heating "
            "to match that level?"
        ),
    ],
    ids=["negated", "hypothetical", "quoted", "bare-match", "multiple-sources"],
)
def test_driver_warming_recovery_rejects_unsafe_or_ambiguous_text(
    utterance: str,
) -> None:
    extracted = frame(
        goal("seat", "set_seat_heating", {"level": 2}),
        goal("steering", "set_steering_wheel_heating", {"level": 2}),
        slots={"level": 2},
    )

    recovered = recover_driver_warming_intent(
        utterance,
        extracted,
        turn_id="unsafe-driver-warming",
    )

    assert recovered is extracted


def test_base54_compound_recovers_relative_occupied_seat_heating() -> None:
    utterance = (
        "Set the climate temperature to 22 degrees for all zones. "
        "Increase the seat heating by two levels for occupied seats. "
        "Turn on the steering wheel heating to level 2."
    )
    extracted = frame(
        goal(
            "climate",
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "ALL_ZONES"},
        ),
        goal(
            "seat",
            "set_seat_heating",
            {"level": 2, "seat_zone": "ALL_ZONES"},
        ),
        goal("steering", "set_steering_wheel_heating", {"level": 2}),
        slots={"temperature": 22, "seat_zone": "ALL_ZONES", "level": 2},
    )

    recovered = recover_relative_occupied_seat_heating_intent(
        utterance, extracted, turn_id="base54-compound"
    )
    result = ground_intent(utterance, recovered, REGISTRY)

    assert [item.semantic_operation for item in result.filtered_intent.goals] == [
        "set_climate_temperature",
        "set_seat_heating",
        "set_steering_wheel_heating",
    ]
    assert result.desired_values_by_goal == {
        "climate": {"temperature": 22, "seat_zone": "ALL_ZONES"},
        "seat": {},
        "steering": {"level": 2},
    }
    assert result.relative_seat_heating_deltas_by_goal == {"seat": 2}
    assert result.occupied_seat_heating_goal_ids == frozenset({"seat"})
    assert result.authorized_action_goal_ids == frozenset(
        {"climate", "seat", "steering"}
    )


def test_base54_compound_inserts_model_omitted_seat_goal_by_raw_order() -> None:
    utterance = (
        "Set the climate temperature to 22 degrees for all zones. "
        "Increase the seat heating by two levels for occupied seats. "
        "Turn on the steering wheel heating to level 2."
    )
    extracted = frame(
        goal(
            "climate",
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "ALL_ZONES"},
        ),
        goal("steering", "set_steering_wheel_heating", {"level": 2}),
        slots={"temperature": 22, "seat_zone": "ALL_ZONES", "level": 2},
    )

    recovered = recover_relative_occupied_seat_heating_intent(
        utterance, extracted, turn_id="base54-model-omitted-seat"
    )
    result = ground_intent(utterance, recovered, REGISTRY)

    assert [item.semantic_operation for item in result.filtered_intent.goals] == [
        "set_climate_temperature",
        "set_seat_heating",
        "set_steering_wheel_heating",
    ]
    assert result.filtered_intent.goal_mention_order == [
        "climate",
        recovered.goals[1].goal_id,
        "steering",
    ]
    assert result.relative_seat_heating_deltas_by_goal == {
        recovered.goals[1].goal_id: 2
    }


def test_base54_explicit_seat_topic_binds_relative_pronoun() -> None:
    utterance = (
        "For the climate temperature, set it for all zones. "
        "For seat heating, increase it by two levels for occupied seats. "
        "Turn on the steering wheel heating to level 2."
    )
    extracted = frame(
        goal(
            "climate",
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "ALL_ZONES"},
        ),
        goal("steering", "set_steering_wheel_heating", {"level": 2}),
        slots={"temperature": 22, "seat_zone": "ALL_ZONES", "level": 2},
    )

    recovered = recover_relative_occupied_seat_heating_intent(
        utterance, extracted, turn_id="base54-seat-topic-pronoun"
    )
    result = ground_intent(utterance, recovered, REGISTRY)

    seat_goal = next(
        goal
        for goal in result.filtered_intent.goals
        if goal.semantic_operation == "set_seat_heating"
    )
    assert seat_goal.desired_outcome == {}
    assert result.relative_seat_heating_deltas_by_goal == {seat_goal.goal_id: 2}
    assert result.occupied_seat_heating_goal_ids == frozenset({seat_goal.goal_id})
    assert [goal.semantic_operation for goal in recovered.goals] == [
        "set_climate_temperature",
        "set_seat_heating",
        "set_steering_wheel_heating",
    ]


@pytest.mark.parametrize(
    "utterance",
    [
        "Increase it by two levels for occupied seats.",
        "For climate temperature, increase it by two levels for occupied seats.",
        "For steering wheel heating, increase it by two levels for occupied seats.",
        "For seat heating. Increase it by two levels for occupied seats.",
        "For seat heating, increase them by two levels for occupied seats.",
    ],
)
def test_relative_seat_pronoun_requires_same_clause_explicit_topic(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_relative_occupied_seat_heating_intent(
        utterance, extracted, turn_id="unsafe-seat-pronoun"
    )
    result = ground_intent(utterance, extracted, REGISTRY)

    assert recovered is extracted
    assert result.relative_seat_heating_deltas_by_goal == {}


@pytest.mark.parametrize(
    "utterance",
    [
        "Increase the seat heating by two levels for occupied seats.",
        "Increase seat heating for occupied seats by two levels.",
    ],
)
def test_base54_single_retry_recovers_goal_bound_occupied_delta(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_relative_occupied_seat_heating_intent(
        utterance, extracted, turn_id="base54-seat-retry"
    )
    result = ground_intent(utterance, recovered, REGISTRY)

    assert len(result.filtered_intent.goals) == 1
    seat_goal = result.filtered_intent.goals[0]
    assert seat_goal.semantic_operation == "set_seat_heating"
    assert seat_goal.desired_outcome == {}
    assert result.relative_seat_heating_deltas_by_goal == {seat_goal.goal_id: 2}
    assert result.occupied_seat_heating_goal_ids == frozenset({seat_goal.goal_id})
    assert result.filtered_intent.explicit_slots == {}


def test_relative_occupied_seat_heating_discards_model_absolute_level() -> None:
    utterance = "Increase the seat heating by two levels for occupied seats."
    extracted = frame(
        goal(
            "seat",
            "set_seat_heating",
            {"level": 2, "seat_zone": "ALL_ZONES"},
        ),
        slots={"level": 2, "seat_zone": "ALL_ZONES"},
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.desired_values_by_goal == {"seat": {}}
    assert result.filtered_intent.explicit_slots == {}
    assert result.relative_seat_heating_deltas_by_goal == {"seat": 2}
    assert result.occupied_seat_heating_goal_ids == frozenset({"seat"})


@pytest.mark.parametrize(
    "utterance",
    [
        "Do not increase the seat heating by two levels for occupied seats.",
        "If it gets cold, increase seat heating by two levels for occupied seats.",
        'She said, "Increase seat heating by two levels for occupied seats."',
        "She asked me to increase seat heating by two levels for occupied seats.",
        "How do I increase seat heating by two levels for occupied seats?",
        "Set seat heating to level two for occupied seats.",
        "Increase seat heating to level two for occupied seats.",
        "Increase seat heating by four levels for occupied seats.",
        "Increase seat heating by two levels.",
        "Increase seat heating for occupied seats.",
        (
            "Increase seat heating by two levels for occupied seats and steering "
            "wheel heating by one level."
        ),
        (
            "Increase seat heating and steering wheel heating by two levels for "
            "occupied seats."
        ),
        (
            "Increase seat heating by two levels for occupied seats. Increase seat "
            "heating by one level for the driver."
        ),
    ],
)
def test_relative_occupied_seat_heating_rejects_unsafe_or_ambiguous_text(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_relative_occupied_seat_heating_intent(
        utterance, extracted, turn_id="unsafe-relative-seat-heating"
    )
    result = ground_intent(utterance, extracted, REGISTRY)

    assert recovered is extracted
    assert result.relative_seat_heating_deltas_by_goal == {}
    assert result.occupied_seat_heating_goal_ids == frozenset()


def test_session_state_has_goal_bound_relative_occupied_seat_fields() -> None:
    session = SessionState(context_id="base54-session-fields")

    assert session.relative_seat_heating_deltas_by_goal == {}
    assert session.occupied_seat_heating_goal_ids == set()


@pytest.mark.parametrize(
    "utterance",
    [
        "Turn on the AC and drop the temperature by 4 degrees for the whole car.",
        (
            "Can you turn on the AC for me and maybe drop the temperature by like, "
            "4 degrees for the whole car?"
        ),
    ],
    ids=["drop-command", "official-polite-maybe"],
)
def test_relative_temperature_accepts_explicit_drop_variants(utterance: str) -> None:
    extracted = frame(
        goal("ac", "set_air_conditioning", {"enabled": True}),
        goal(
            "temperature",
            "set_climate_temperature",
            {"seat_zone": "whole_car"},
        ),
        slots={"enabled": True, "seat_zone": "whole_car"},
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.desired_values_by_goal == {
        "ac": {"enabled": True},
        "temperature": {"seat_zone": "ALL_ZONES"},
    }
    assert result.relative_climate_temperature_deltas_by_goal == {"temperature": -4}
    assert has_explicit_action_request(utterance)
    assert focus_explicit_action_request(utterance) == utterance


@pytest.mark.parametrize(
    "utterance",
    [
        "Maybe if it gets warm, drop the temperature by 4 degrees.",
        "I might turn on the AC and maybe drop the temperature by 4 degrees.",
        (
            "Could you tell me whether I should turn on the AC and maybe drop the "
            "temperature by 4 degrees for the whole car?"
        ),
        (
            "Can you explain whether I should turn on the AC and maybe drop the "
            "temperature by 4 degrees for the whole car?"
        ),
        "The temperature dropped by 4 degrees.",
        "Do not drop the temperature by 4 degrees.",
    ],
)
def test_drop_temperature_hypothetical_statement_and_negation_fail_closed(
    utterance: str,
) -> None:
    extracted = frame(
        goal(
            "temperature",
            "set_climate_temperature",
            {"seat_zone": "ALL_ZONES"},
        )
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.relative_climate_temperature_deltas_by_goal == {}


def test_multiple_temperature_goals_do_not_cross_bind_one_relative_delta() -> None:
    extracted = frame(
        goal(
            "passenger",
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "PASSENGER"},
        ),
        goal(
            "driver",
            "set_climate_temperature",
            {"temperature": 4, "seat_zone": "DRIVER"},
        ),
        slots={"temperature": 4, "seat_zone": "DRIVER"},
    )

    result = ground_intent(
        "Set passenger to 22 degrees and lower driver by 4 degrees.",
        extracted,
        REGISTRY,
    )

    assert result.relative_climate_temperature_deltas_by_goal == {}


def test_one_model_goal_cannot_hide_two_user_temperature_instructions() -> None:
    extracted = frame(
        goal(
            "passenger",
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "PASSENGER"},
        ),
        slots={"temperature": 22, "seat_zone": "PASSENGER"},
    )

    result = ground_intent(
        (
            "Set passenger temperature to 22 degrees and lower driver temperature "
            "by 4 degrees."
        ),
        extracted,
        REGISTRY,
    )

    assert result.relative_climate_temperature_deltas_by_goal == {}


def test_relative_climate_temperature_discards_guessed_absolute_value() -> None:
    extracted = frame(
        goal(
            "temperature",
            "set_climate_temperature",
            {"temperature": 4},
        ),
        slots={"temperature": 4},
    )

    result = ground_intent(
        "Set the temperature for the whole car to 4 degrees lower than it is now.",
        extracted,
        REGISTRY,
    )

    assert result.desired_values_by_goal == {"temperature": {"seat_zone": "ALL_ZONES"}}
    assert result.filtered_intent.explicit_slots == {}
    assert result.filtered_intent.unresolved_slots == []
    assert result.relative_climate_temperature_deltas_by_goal == {"temperature": -4}


@pytest.mark.parametrize(
    "utterance",
    [
        "If it gets warm, lower the temperature by four degrees.",
        "Do not lower the temperature by four degrees.",
        "Set the temperature to 4 degrees for the whole car.",
        "Lower the temperature to 4 degrees for the whole car.",
        "Set the temperature to 4 degrees lower than the outside temperature.",
        "The temperature was lowered by four degrees.",
    ],
)
def test_non_executable_or_absolute_temperature_phrases_have_no_relative_delta(
    utterance: str,
) -> None:
    extracted = frame(
        goal(
            "temperature",
            "set_climate_temperature",
            {"temperature": 4, "seat_zone": "ALL_ZONES"},
        )
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.relative_climate_temperature_deltas_by_goal == {}


def test_whole_cabin_is_grounded_as_all_climate_zones() -> None:
    extracted = frame(
        goal("temperature", "set_climate_temperature", {"temperature": 22})
    )

    result = ground_intent(
        "Set the temperature to 22 degrees for the entire cabin.",
        extracted,
        REGISTRY,
    )

    assert result.desired_values_by_goal == {
        "temperature": {"temperature": 22, "seat_zone": "ALL_ZONES"}
    }


def test_defrost_anchor_does_not_steal_an_explicit_all_windows_value() -> None:
    extracted = frame(
        goal(
            "windows",
            "set_window_position",
            {"window": "all", "percentage": 0},
        ),
        goal(
            "defrost",
            "set_window_defrost",
            {"enabled": True, "window": "all"},
            depends_on=["windows"],
        ),
        slots={"window": "all", "percentage": 0, "enabled": True},
    )

    result = ground_intent(
        "Close all windows. Then turn on the defrost.", extracted, REGISTRY
    )

    assert [item.goal_id for item in result.filtered_intent.goals] == [
        "windows",
        "defrost",
    ]
    assert result.desired_values_by_goal == {
        "windows": {"window": "all", "percentage": 0},
        "defrost": {"enabled": True},
    }
    assert result.filtered_intent.goals[1].depends_on == ["windows"]


@pytest.mark.parametrize(
    "utterance",
    [
        "If it gets warm, increase the fan speed by one level.",
        "Do not increase the fan speed by one level.",
        "Set the fan speed to level two.",
    ],
)
def test_non_executable_or_absolute_fan_phrases_do_not_create_relative_delta(
    utterance: str,
) -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 2}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.relative_fan_speed_deltas_by_goal == {}


@pytest.mark.parametrize(
    ("utterance", "semantic_operation"),
    [
        ("Can you tell me the current navigation status?", "read_current_navigation"),
        ("What is my charging status?", "read_charging_status"),
    ],
)
def test_direct_read_reused_by_multiple_recipes_is_retained(
    utterance: str, semantic_operation: str
) -> None:
    extracted = frame(goal("read", semantic_operation, {}), kind=IntentKind.INFORMATION)

    result = ground_intent(utterance, extracted, REGISTRY)

    assert [item.goal_id for item in result.filtered_intent.goals] == ["read"]
    assert result.filtered_intent.call_for_action is False
    assert result.authorized_action_goal_ids == frozenset()


def test_base48_named_restaurant_search_normalizes_exact_qwen_frame() -> None:
    utterance = "Can you show me some restaurants in Munich?"
    extracted = IntentFrame(
        language="en",
        call_for_action=True,
        goals=[
            goal(
                "poi",
                "search_poi_at_location",
                {"location_id": "Munich", "category": "restaurants"},
            )
        ],
        explicit_slots={"location": "Munich", "category": "restaurants"},
        intent_kind=IntentKind.ACTION,
    )
    before = extracted.model_dump(mode="json")

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="base48-poi"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert extracted.model_dump(mode="json") == before
    assert recovered.call_for_action is False
    assert recovered.intent_kind is IntentKind.INFORMATION
    assert recovered.goals[0].semantic_operation == "search_poi_at_location"
    assert recovered.goals[0].desired_outcome == {
        "category": "restaurants",
        "location_name": "Munich",
    }
    assert recovered.explicit_slots == {
        "category": "restaurants",
        "location_name": "Munich",
    }
    assert grounded.desired_values_by_goal == {
        "poi": {"category": "restaurants", "location_name": "Munich"}
    }
    assert grounded.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize("verb", ["show me some", "search for", "find"])
def test_named_restaurant_search_recovers_empty_model_frame(verb: str) -> None:
    utterance = f"Can you {verb} restaurants in Munich?"
    extracted = frame(kind=IntentKind.INFORMATION)

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="named-poi"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert [item.semantic_operation for item in grounded.filtered_intent.goals] == [
        "search_poi_at_location"
    ]
    assert next(iter(grounded.desired_values_by_goal.values())) == {
        "category": "restaurants",
        "location_name": "Munich",
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "Do not show me restaurants in Munich.",
        "If needed, show me restaurants in Munich.",
        "Hypothetically, find restaurants in Munich.",
        'She asked, "Show me restaurants in Munich."',
        '"Show me restaurants in Munich" is only an example.',
    ],
)
def test_named_restaurant_search_rejects_non_executable_text(utterance: str) -> None:
    extracted = frame(kind=IntentKind.INFORMATION)

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="named-poi"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "utterance",
    [
        "Show me restaurants in Munich or Berlin.",
        "Show me restaurants and bakeries in Munich.",
        "Show me restaurants in my current city.",
        "Show me a restaurant menu in Munich.",
        "Show me restaurants in Munich tomorrow.",
        "Find restaurants in Munich for dinner.",
        "Search for restaurants in Munich instead of Berlin.",
        "Show me restaurants in Munich sorted by rating.",
        "Find restaurants in Munich around noon.",
        "Show me restaurants in Munich today.",
        "Find restaurants in Munich tonight.",
        "Show me restaurants in Munich open now.",
        "Show me restaurants in Munich next week.",
        "Show me restaurants in Munich on Sunday.",
        "Show me restaurants in Munich after work.",
        "Show me restaurants in Munich on the way.",
        "Show me restaurants in Munich that are cheap.",
        "Show me restaurants in Munich which are cheap.",
    ],
)
def test_named_restaurant_search_rejects_ambiguous_scope(
    utterance: str,
) -> None:
    extracted = frame(kind=IntentKind.INFORMATION)

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="named-poi"
    )

    assert recovered is extracted


def test_named_restaurant_search_empty_frame_rejects_multiword_location() -> None:
    utterance = "Show me restaurants in New York."
    extracted = frame(kind=IntentKind.INFORMATION)

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="named-poi"
    )

    assert recovered is extracted


def test_named_restaurant_search_accepts_exact_multiword_model_literal() -> None:
    utterance = "Show me restaurants in New York."
    extracted = frame(
        goal(
            "poi",
            "search_poi_at_location",
            {"location_id": "New York", "category": "restaurants"},
        ),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="named-poi"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered.goals[0].desired_outcome == {
        "category": "restaurants",
        "location_name": "New York",
    }
    assert grounded.desired_values_by_goal == {
        "poi": {"category": "restaurants", "location_name": "New York"}
    }


def test_named_restaurant_search_model_literal_requires_exact_location_boundary() -> (
    None
):
    utterance = "Show me restaurants in New York next week."
    extracted = frame(
        goal(
            "poi",
            "search_poi_at_location",
            {"location_id": "New York", "category": "restaurants"},
        ),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="named-poi"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert grounded.filtered_intent.goals == []


def test_named_restaurant_search_never_trusts_model_guessed_entity_id() -> None:
    utterance = "Search for restaurants in Munich."
    extracted = frame(
        goal(
            "poi",
            "search_poi_at_location",
            {"location_id": "loc_mun_9995", "category_poi": "restaurants"},
        ),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="named-poi"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert grounded.desired_values_by_goal == {
        "poi": {"category": "restaurants", "location_name": "Munich"}
    }


@pytest.mark.parametrize(
    ("utterance", "extracted"),
    [
        (
            "I'm looking for a good restaurant at my destination. "
            "Can you help me find some options?",
            frame(kind=IntentKind.INFORMATION),
        ),
        (
            "Please find restaurants at the destination.",
            frame(kind=IntentKind.INFORMATION),
        ),
        (
            "Can you search for restaurants near my current destination?",
            frame(kind=IntentKind.INFORMATION),
        ),
        (
            "I'm looking for a good restaurant at my destination. "
            "Can you help me find one?",
            IntentFrame(
                language="en",
                call_for_action=True,
                goals=[
                    goal("navigation", "read_current_navigation", {}),
                    goal(
                        "poi",
                        "search_poi_at_location",
                        {
                            "category_poi": "restaurants",
                            "location_id": "loc_model_guess_123",
                            "phone_number": "+49 000 0000000",
                        },
                    ),
                ],
                explicit_slots={
                    "category": "restaurants",
                    "location_id": "loc_model_guess_123",
                    "phone_number": "+49 000 0000000",
                },
                intent_kind=IntentKind.ACTION,
            ),
        ),
    ],
)
def test_current_destination_restaurant_search_is_recovered_without_model_ids(
    utterance: str,
    extracted: IntentFrame,
) -> None:
    before = extracted.model_dump(mode="json")

    recovered = recover_current_destination_poi_search_intent(
        utterance, extracted, turn_id="base58-current-destination-poi"
    )

    assert extracted.model_dump(mode="json") == before
    assert recovered is not extracted
    assert recovered.call_for_action is False
    assert recovered.intent_kind is IntentKind.INFORMATION
    assert len(recovered.goals) == 1
    assert recovered.goals[0].semantic_operation == (
        "search_poi_at_current_destination"
    )
    assert recovered.goals[0].desired_outcome == {"category": "restaurants"}
    assert recovered.explicit_slots == {"category": "restaurants"}
    assert recovered.explicit_constraints == {}
    assert recovered.explicit_confirmations == []
    assert recovered.unresolved_slots == []
    assert recovered.goal_mention_order == [recovered.goals[0].goal_id]
    assert recovered.intent_source_turn_ids == ["base58-current-destination-poi"]


@pytest.mark.parametrize(
    "utterance",
    [
        "Find restaurants in Munich.",
        "Find restaurants around my current location.",
        "Find restaurants along my route.",
        "Do not find restaurants at my destination.",
        "If needed, find restaurants at my destination.",
        "Maybe find restaurants at my destination.",
        "She asked me to find restaurants at my destination.",
        'She asked, "Find restaurants at my destination."',
        '"Find restaurants at my destination" is only an example.',
        "Find restaurants at my destination and call the first one.",
        "Find restaurants at my destination, then set the temperature to 22.",
        "Find restaurants and bakeries at my destination.",
        "Find restaurants at my destination or my current location.",
        "I found a restaurant at my destination.",
        "Restaurants at my destination are good.",
    ],
)
def test_current_destination_restaurant_search_rejects_unsafe_or_other_scope(
    utterance: str,
) -> None:
    extracted = frame(kind=IntentKind.INFORMATION)

    recovered = recover_current_destination_poi_search_intent(
        utterance, extracted, turn_id="base58-current-destination-poi"
    )

    assert recovered is extracted


def test_current_destination_restaurant_search_rejects_mixed_model_action() -> None:
    utterance = (
        "I'm looking for a good restaurant at my destination. "
        "Can you help me find some options?"
    )
    extracted = frame(
        goal(
            "poi",
            "search_poi_at_location",
            {"category": "restaurants", "location": "destination"},
        ),
        goal(
            "call",
            "call_phone_by_number",
            {"phone_number": "+49 000 0000000"},
        ),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_current_destination_poi_search_intent(
        utterance, extracted, turn_id="base58-current-destination-poi"
    )

    assert recovered is extracted


def test_base48_destination_change_remains_navigation_not_poi_search() -> None:
    utterance = "Okay, can you change my destination to Munich?"
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_final_destination",
            {"new_destination_id": "Munich"},
        )
    )

    recovered = recover_named_location_poi_search_intent(
        utterance, extracted, turn_id="base48-navigation"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert [item.semantic_operation for item in grounded.filtered_intent.goals] == [
        "navigation_replace_final_destination"
    ]
    assert grounded.desired_values_by_goal == {
        "navigation": {"new_destination_name": "Munich"}
    }
    assert grounded.authorized_action_goal_ids == frozenset({"navigation"})


@pytest.mark.parametrize(
    "extracted",
    [
        frame(),
        frame(
            goal(
                "navigation",
                "start_navigation",
                {"destination": "loc_model_guess_123"},
            )
        ),
    ],
)
def test_base48_exact_official_destination_change_is_recovered(
    extracted: IntentFrame,
) -> None:
    utterance = (
        "Actually, I want to change my destination to Munich. "
        "Can you navigate me there?"
    )

    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, extracted, turn_id="base48-official-replacement"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert [item.semantic_operation for item in recovered.goals] == [
        "navigation_replace_final_destination"
    ]
    assert recovered.goals[0].desired_outcome == {"new_destination_name": "Munich"}
    assert grounded.desired_values_by_goal == {
        recovered.goals[0].goal_id: {"new_destination_name": "Munich"}
    }
    assert grounded.authorized_action_goal_ids == frozenset(
        {recovered.goals[0].goal_id}
    )


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Hey there! I'm actually heading to Munich now instead of Paris. "
            "Can you change my navigation destination for me?"
        ),
        (
            "Hey there! I've changed my mind about going to Paris. Can you "
            "switch my navigation destination to Munich instead?"
        ),
        (
            "Hey there! I've changed my mind about Paris. Can you switch my "
            "navigation to Munich instead?"
        ),
        (
            "Yep, that's the current one. So, can you change the destination "
            "to Munich for me?"
        ),
        (
            "Okay, I understand that's the current route. But I want to change "
            "the destination to Munich, please."
        ),
        ("Okay, I need to change my navigation destination. Please set it to Munich."),
        ("I need to change my navigation. Please change the destination to Munich."),
        "Please change my navigation destination to Munich.",
        (
            "I want to change my navigation. My new destination is Munich. "
            "Can you please set that up?"
        ),
        (
            "I want to change my navigation destination. Please set the new "
            "destination to Munich."
        ),
        "Please change my current navigation destination to Munich.",
    ],
    ids=[
        "heading-new-instead-of-old",
        "changed-mind-switch",
        "changed-mind-compact-navigation",
        "current-one-ack",
        "current-route-ack",
        "set-it-anaphora",
        "navigation-then-destination",
        "direct-navigation-destination",
        "declared-new-destination",
        "set-new-destination",
        "direct-current-navigation-destination",
    ],
)
@pytest.mark.parametrize("model_seed", ["empty", "navigation-read"])
def test_base66_formal_destination_replacement_family_is_grounded(
    utterance: str,
    model_seed: str,
) -> None:
    extracted = (
        frame()
        if model_seed == "empty"
        else frame(goal("preserve-base66-read-id", "read_current_navigation", {}))
    )

    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, extracted, turn_id="base66-formal-destination"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    recovered_goal = recovered.goals[0]
    if model_seed == "navigation-read":
        assert recovered_goal.goal_id == "preserve-base66-read-id"
    assert recovered_goal.semantic_operation == "navigation_replace_final_destination"
    assert recovered_goal.desired_outcome == {"new_destination_name": "Munich"}
    assert recovered.goal_mention_order == [recovered_goal.goal_id]
    assert grounded.desired_values_by_goal == {
        recovered_goal.goal_id: {"new_destination_name": "Munich"}
    }
    assert grounded.authorized_action_goal_ids == frozenset({recovered_goal.goal_id})


@pytest.mark.parametrize(
    "utterance",
    [
        "Could you replace the current destination to Munich for me?",
        "So, can you please switch my new navigation destination to Munich?",
        (
            "Okay, that's the current destination. Could you replace my current "
            "navigation destination to Munich instead?"
        ),
        (
            "Ok, I would like to switch my navigation destination. Please set "
            "the new destination to Munich."
        ),
        (
            "I would like to replace my navigation. Please switch the current "
            "destination to Munich."
        ),
        (
            "Okay, I need to switch my navigation. My new destination is Munich. "
            "Please set that up."
        ),
        (
            "I am heading to Munich instead of Paris. Would you replace my "
            "navigation destination?"
        ),
        (
            "I have changed my mind about going to Paris. Would you please "
            "replace my current destination to Munich for me?"
        ),
        (
            "I have changed my mind about Paris. Would you replace my navigation "
            "destination to Munich instead?"
        ),
        (
            "I've changed my mind about going to Paris. Can you change my "
            "navigation to Munich instead?"
        ),
    ],
)
def test_named_destination_replacement_bounded_synonym_family(
    utterance: str,
) -> None:
    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, frame(), turn_id="destination-replacement-synonym-family"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    recovered_goal = recovered.goals[0]
    assert recovered_goal.desired_outcome == {"new_destination_name": "Munich"}
    assert grounded.desired_values_by_goal == {
        recovered_goal.goal_id: {"new_destination_name": "Munich"}
    }
    assert grounded.authorized_action_goal_ids == frozenset({recovered_goal.goal_id})


@pytest.mark.parametrize(
    "utterance",
    [
        "change my navigation destination to Munich, please",
        "Change my navigation destination to Munich, please.",
        "Please change my navigation destination to Munich.",
        "Can you change my current navigation destination to Munich, please?",
    ],
)
def test_base66_navigation_destination_change_preserves_existing_goal_id(
    utterance: str,
) -> None:
    extracted = frame(
        goal(
            "preserve-base66-goal-id",
            "start_navigation",
            {"destination": "loc_model_guess_123"},
        )
    )

    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, extracted, turn_id="base66-navigation-destination"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered.goals[0].goal_id == "preserve-base66-goal-id"
    assert recovered.goals[0].semantic_operation == (
        "navigation_replace_final_destination"
    )
    assert recovered.goals[0].desired_outcome == {"new_destination_name": "Munich"}
    assert recovered.goal_mention_order == ["preserve-base66-goal-id"]
    assert grounded.desired_values_by_goal == {
        "preserve-base66-goal-id": {"new_destination_name": "Munich"}
    }
    assert grounded.authorized_action_goal_ids == frozenset({"preserve-base66-goal-id"})


def test_explicit_destination_change_can_retain_start_navigation_followup() -> None:
    utterance = (
        "I want to change my current destination to Munich and start navigation there."
    )

    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, frame(), turn_id="destination-replacement-followup"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert grounded.authorized_action_goal_ids == frozenset(
        {recovered.goals[0].goal_id}
    )
    assert grounded.desired_values_by_goal == {
        recovered.goals[0].goal_id: {"new_destination_name": "Munich"}
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "Can you navigate me to Munich?",
        "Should I change my destination to Munich?",
        "Can you tell me how to change my destination to Munich?",
        "Do not change my destination to Munich.",
        "If needed, change my destination to Munich.",
        'She asked, "Change my destination to Munich."',
        "Change my destination to Munich or Berlin.",
        "Change my destination to Munich and turn on the air conditioning.",
        "Turn on the air conditioning and change my destination to Munich.",
        "Open the sunroof. Change my destination to Munich.",
        "Change My Destination To Munich And Open Sunroof.",
        "Change My Destination To Munich And Start Engine.",
        "Change My Destination To Munich Next Week.",
        "Change My Destination To Munich Or Berlin.",
        "Change My Destination To Munich And Berlin.",
        "Change My Destination To Munich-or-Berlin.",
        "Change My Destination To Munich-and-Berlin.",
        "Change My Destination To Munich-and-Open-Sunroof.",
        "Change My Destination To Munich-Next-Week.",
        "Change My Destination To Munich_or_Berlin.",
        "Change My Destination To Munich'or'Berlin.",
        "Do not change my navigation destination to Munich, please.",
        "If needed, change my navigation destination to Munich, please.",
        'She asked, "Change my navigation destination to Munich, please."',
        "Change my navigation destination to Munich or Berlin, please.",
        ("Do not change my navigation destination. Please set it to Munich."),
        (
            "If needed, I want to change my navigation. My new destination is "
            "Munich. Can you please set that up?"
        ),
        (
            'She said, "Okay, I need to change my navigation destination. '
            'Please set it to Munich."'
        ),
        (
            '"I want to change my navigation. My new destination is Munich. '
            'Can you please set that up?" is an example.'
        ),
        (
            "Okay, I need to change my navigation destination. Please set it "
            "to Munich or Berlin."
        ),
        (
            "I want to change my navigation. My new destination is Munich or "
            "Berlin. Can you please set that up?"
        ),
        (
            "Hey there! I'm heading to Munich or Berlin now instead of Paris. "
            "Can you change my navigation destination for me?"
        ),
        (
            "Hey there! I'm heading to Munich now instead of Paris. Can you "
            "change my navigation destination and open the sunroof?"
        ),
        (
            "Hey there! I've changed my mind about going to Paris. Can you "
            "switch my navigation destination to Munich or Berlin instead?"
        ),
        (
            "Hey there! I've changed my mind about going to Munich. Can you "
            "switch my navigation destination to Munich instead?"
        ),
        (
            "Hey there! I've changed my mind about Paris. Can you not switch my "
            "navigation to Munich instead?"
        ),
        (
            "If needed, I've changed my mind about Paris. Can you switch my "
            "navigation to Munich instead?"
        ),
        (
            "\"Hey there! I've changed my mind about Paris. Can you switch my "
            'navigation to Munich instead?"'
        ),
        (
            "She said that I've changed my mind about Paris. Can you switch my "
            "navigation to Munich instead?"
        ),
        (
            "Hey there! I've changed my mind about Paris. Can you switch my "
            "navigation to Munich or Berlin instead?"
        ),
        (
            "Hey there! I've changed my mind about Munich. Can you switch my "
            "navigation to Munich instead?"
        ),
        (
            "Hey there! I've changed my mind about Paris. Can you switch my "
            "navigation to Munich instead? Then open the sunroof."
        ),
        "My new destination is Munich. Can you please set that up?",
        "Please set it to Munich.",
        (
            "I'm heading to Munich now instead of Paris. Can you tell me the "
            "current navigation destination?"
        ),
        (
            "I need to change my navigation. Please change the destination to "
            "Munich. Then open the sunroof."
        ),
    ],
)
def test_named_destination_replacement_recovery_rejects_unsafe_text(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, extracted, turn_id="unsafe-destination-replacement"
    )

    assert recovered is extracted


def test_named_destination_replacement_recovery_rejects_unrelated_model_goal() -> None:
    extracted = frame(goal("ac", "set_air_conditioning", {"enabled": True}))

    recovered = recover_named_navigation_destination_replacement_intent(
        "Change my destination to Munich.",
        extracted,
        turn_id="mixed-destination-replacement",
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Hey there! I've changed my mind about going to Essen after Dortmund. "
            "Can you remove Essen from my route so Dortmund is the final stop?"
        ),
        (
            "I want to remove Essen from my current route. Can you make Dortmund "
            "the final destination instead?"
        ),
        (
            "Okay, so I'm currently navigating to Dortmund with a stop in Essen. "
            "I want to remove Essen from the route completely. Just make Dortmund "
            "the final destination."
        ),
        (
            "Okay, let's try this: Cancel Essen as a destination. My trip should "
            "end in Dortmund."
        ),
        (
            "Okay, I need to modify my current navigation. Please remove Essen "
            "from the route and make Dortmund the final destination."
        ),
        (
            "Okay, I need to change my navigation. My current route has Dortmund "
            "and then Essen. I want to remove Essen from the route so that "
            "Dortmund is my final destination."
        ),
    ],
    ids=[
        "official-initial",
        "official-retry-make-instead",
        "official-retry-current-stop",
        "official-retry-cancel-trip-end",
        "official-retry-modify-and-make",
        "official-retry-route-order",
    ],
)
def test_base50_exact_official_final_destination_delete_is_recovered(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, extracted, turn_id="base50-official-delete"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is not extracted
    assert [item.semantic_operation for item in grounded.filtered_intent.goals] == [
        "navigation_delete_final_destination"
    ]
    recovered_goal = grounded.filtered_intent.goals[0]
    assert recovered_goal.desired_outcome == {
        "destination_name_to_delete": "Essen",
        "remaining_destination_name": "Dortmund",
    }
    assert grounded.authorized_action_goal_ids == frozenset({recovered_goal.goal_id})


@pytest.mark.parametrize(
    "utterance",
    ["Remove Rome from my trip.", "Remove Rome from my navigation."],
)
def test_base80_generic_named_navigation_delete_is_recovered(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, extracted, turn_id="base80-generic-named-delete"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is not extracted
    assert len(grounded.filtered_intent.goals) == 1
    recovered_goal = grounded.filtered_intent.goals[0]
    assert recovered_goal.semantic_operation == "navigation_delete_final_destination"
    assert recovered_goal.desired_outcome == {"destination_name_to_delete": "Rome"}
    assert grounded.authorized_action_goal_ids == frozenset({recovered_goal.goal_id})


@pytest.mark.parametrize(
    "utterance",
    [
        "Do not remove Rome from my trip.",
        'She said, "Remove Rome from my trip."',
        "Remove Rome or Paris from my trip.",
        "Remove Rome from my trip and turn on the AC.",
        "Remove my navigation.",
        "How do I remove Rome from my trip?",
    ],
)
def test_base80_generic_named_navigation_delete_rejects_unsafe_text(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, extracted, turn_id="base80-unsafe-generic-delete"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "extracted",
    [
        frame(),
        frame(
            goal(
                "delete-final",
                "navigation_delete_final_destination",
                {"destination_id_to_delete": "loc_ess_model_guess"},
            )
        ),
        frame(
            goal(
                "delete-waypoint",
                "navigation_delete_one_waypoint",
                {"waypoint_id_to_delete": "loc_ess_model_guess"},
            )
        ),
        frame(
            goal(
                "replace-final",
                "navigation_replace_final_destination",
                {"new_destination_id": "loc_dor_model_guess"},
            )
        ),
        frame(goal("read-navigation", "read_current_navigation", {})),
        frame(
            goal(
                "resolve-location",
                "resolve_location",
                {"location_id": "loc_dor_model_guess"},
            )
        ),
    ],
    ids=[
        "empty",
        "final-delete-with-id",
        "waypoint-delete-with-id",
        "final-replace-with-id",
        "navigation-read",
        "location-read-with-id",
    ],
)
def test_named_final_destination_delete_discards_model_ids_and_misclassification(
    extracted: IntentFrame,
) -> None:
    utterance = "Can you remove Essen from my route so Dortmund is the final stop?"

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, extracted, turn_id="base50-model-frame"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert len(grounded.filtered_intent.goals) == 1
    recovered_goal = grounded.filtered_intent.goals[0]
    assert recovered_goal.semantic_operation == "navigation_delete_final_destination"
    assert recovered_goal.desired_outcome == {
        "destination_name_to_delete": "Essen",
        "remaining_destination_name": "Dortmund",
    }
    assert "loc_" not in str(recovered_goal.model_dump(mode="json"))


def test_ground_intent_does_not_synthesize_named_final_destination_delete() -> None:
    utterance = "Can you remove Essen from my route so Dortmund is the final stop?"

    grounded = ground_intent(utterance, frame(), REGISTRY)

    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "utterance",
    [
        "Please remove the waypoint Essen from my current route.",
        "Cancel my current navigation.",
        'She asked, "Can you remove Essen from my route so Dortmund is the final stop?"',
        "Do not remove Essen from my route so Dortmund is the final stop.",
        "If needed, can you remove Essen from my route so Dortmund is the final stop?",
        "She asked me to remove Essen from my route so Dortmund is the final stop.",
        "How do I remove Essen from my route so Dortmund is the final stop?",
        "Can you remove Essen or Berlin from my route so Dortmund is the final stop?",
        "Can you remove Essen from my route so Dortmund or Bremen is the final stop?",
        (
            "Can you remove Essen from my route so Dortmund is the final stop, "
            "and turn on the air conditioning?"
        ),
        (
            "Can you remove Essen and delete Berlin from my route so Dortmund "
            "is the final stop?"
        ),
        "Can you remove Essen from my route so Essen is the final stop?",
        "Can you remove Essen-or-Berlin from my route so Dortmund is the final stop?",
        "Can you remove Essen_or_Berlin from my route so Dortmund is the final stop?",
        "Can you remove Essen/or/Berlin from my route so Dortmund is the final stop?",
        "Can you remove Essen‐or‐Berlin from my route so Dortmund is the final stop?",
        "Can you remove Essen from my route so Dortmund-or-Bremen is the final stop?",
        "Can you remove Essen from my route so Dortmund_or_Bremen is the final stop?",
        "Can you remove Essen from my route so Dortmund/Bremen is the final stop?",
        "Can you remove Essen from my route so Dortmund​or​Bremen is the final stop?",
    ],
)
def test_named_final_destination_delete_recovery_rejects_unsafe_text(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, extracted, turn_id="unsafe-final-destination-delete"
    )

    assert recovered is extracted


def test_generic_named_current_route_delete_is_recovered_for_state_binding() -> None:
    utterance = "Please remove Essen from my current route."

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, frame(), turn_id="generic-named-route-delete"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert len(grounded.filtered_intent.goals) == 1
    recovered_goal = grounded.filtered_intent.goals[0]
    assert recovered_goal.semantic_operation == "navigation_delete_final_destination"
    assert recovered_goal.desired_outcome == {"destination_name_to_delete": "Essen"}
    assert grounded.authorized_action_goal_ids == frozenset({recovered_goal.goal_id})


@pytest.mark.parametrize(
    "operation",
    ["set_air_conditioning", "delete_current_navigation"],
)
def test_named_final_destination_delete_rejects_unrelated_model_goal(
    operation: str,
) -> None:
    extracted = frame(goal("unrelated", operation, {}))
    utterance = "Can you remove Essen from my route so Dortmund is the final stop?"

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, extracted, turn_id="mixed-final-destination-delete"
    )

    assert recovered is extracted


def test_named_final_destination_delete_rejects_mixed_unrelated_model_goal() -> None:
    extracted = frame(
        goal(
            "delete-waypoint",
            "navigation_delete_one_waypoint",
            {"waypoint_id_to_delete": "loc_ess_model_guess"},
        ),
        goal("ac", "set_air_conditioning", {"enabled": True}),
    )
    utterance = "Can you remove Essen from my route so Dortmund is the final stop?"

    recovered = recover_named_navigation_final_destination_delete_intent(
        utterance, extracted, turn_id="mixed-final-destination-delete"
    )

    assert recovered is extracted


def test_named_final_destination_delete_grounding_requires_exact_context_names() -> (
    None
):
    utterance = "Can you remove Essen from my route so Dortmund is the final stop?"
    extracted = frame(
        goal(
            "delete-final",
            "navigation_delete_final_destination",
            {
                "destination_name_to_delete": "Essen",
                "remaining_destination_name": "Berlin",
            },
        )
    )

    grounded = ground_intent(utterance, extracted, REGISTRY)

    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "extracted",
    [
        frame(),
        frame(
            goal(
                "delete-waypoint",
                "navigation_delete_one_waypoint",
                {
                    "waypoint_id": "loc_nur_model_guess",
                    "replacement_route_id": "rll_model_guess",
                },
            )
        ),
        frame(goal("navigation-read", "read_current_navigation", {})),
    ],
    ids=["empty", "delete-with-model-ids", "navigation-read"],
)
def test_base56_exact_named_waypoint_delete_is_recovered(
    extracted: IntentFrame,
) -> None:
    utterance = "Remove Nuremberg from my route. Go straight to Paris."

    recovered = recover_named_navigation_waypoint_delete_intent(
        utterance, extracted, turn_id="base56-named-delete"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert len(grounded.filtered_intent.goals) == 1
    recovered_goal = grounded.filtered_intent.goals[0]
    assert recovered_goal.semantic_operation == "navigation_delete_one_waypoint"
    assert recovered_goal.desired_outcome == {
        "waypoint_name_to_delete": "Nuremberg",
        "remaining_destination_name": "Paris",
    }
    assert grounded.authorized_action_goal_ids == frozenset({recovered_goal.goal_id})
    assert "loc_" not in str(recovered_goal.model_dump(mode="json"))
    assert "rll_" not in str(recovered_goal.model_dump(mode="json"))


def test_base56_named_waypoint_clarification_is_recovered() -> None:
    utterance = "Remove the Nuremberg stop."

    recovered = recover_named_navigation_waypoint_delete_intent(
        utterance, frame(), turn_id="base56-waypoint-clarification"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert len(grounded.filtered_intent.goals) == 1
    recovered_goal = grounded.filtered_intent.goals[0]
    assert recovered_goal.semantic_operation == "navigation_delete_one_waypoint"
    assert recovered_goal.desired_outcome == {"waypoint_name_to_delete": "Nuremberg"}
    assert grounded.authorized_action_goal_ids == frozenset({recovered_goal.goal_id})


@pytest.mark.parametrize(
    ("utterance", "expected_route", "expected_destination"),
    [
        ("Can you please just remove the Brookfield stop from my route?", None, None),
        ("Can you just remove Brookfield from my route, please?", None, None),
        ("Please, just remove the Brookfield stop from my route.", None, None),
        (
            "Can you please remove Brookfield from the route and then find the "
            "shortest route for me?",
            "shortest",
            None,
        ),
        (
            "Now that's done, I also need to remove Brookfield from my route. "
            "It's no longer necessary. Can you find the shortest route after "
            "removing Brookfield?",
            "shortest",
            None,
        ),
        (
            "I'm sorry, I thought you had already removed Brookfield. Can you "
            "please remove Brookfield from the route and then find the shortest "
            "route for me?",
            "shortest",
            None,
        ),
        (
            "Okay, I understand. So, can you please remove Brookfield from my route "
            "now, and then show me the shortest route to Riverton?",
            "shortest",
            "Riverton",
        ),
        (
            "Can you please just take Brookfield out of my route? And then, show me "
            "the shortest route to Riverton.",
            "shortest",
            "Riverton",
        ),
        (
            "Could you please remove Brookfield from my route, then show me the "
            "shortest route to Riverton, please?",
            "shortest",
            "Riverton",
        ),
        (
            "But you haven't removed Brookfield yet. Please remove Brookfield from "
            "my route, and then show me the shortest route to Riverton.",
            "shortest",
            "Riverton",
        ),
        (
            "Okay, I'm getting confused. Can you please remove Brookfield from my "
            "route, then show me the shortest route to Riverton?",
            "shortest",
            "Riverton",
        ),
        (
            "I need to remove Brookfield from my route. Can you please do that?",
            None,
            None,
        ),
        (
            "I need to remove a stop from my route. Please remove Brookfield.",
            None,
            None,
        ),
    ],
)
def test_polite_named_waypoint_delete_variants_are_recovered(
    utterance: str,
    expected_route: str | None,
    expected_destination: str | None,
) -> None:
    seed = frame()
    recovered = recover_named_navigation_waypoint_delete_intent(
        utterance, seed, turn_id="polite-waypoint-delete"
    )

    assert recovered is not seed
    assert len(recovered.goals) == 1
    recovered_goal = recovered.goals[0]
    assert recovered_goal.semantic_operation == "navigation_delete_one_waypoint"
    expected = {"waypoint_name_to_delete": "Brookfield"}
    if expected_route is not None:
        expected["route_choice_alias"] = expected_route
    if expected_destination is not None:
        expected["route_destination_name"] = expected_destination
    assert recovered_goal.desired_outcome == expected


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Can you please remove Brookfield from the route and then find the "
            "shortest route after removing Laketown?"
        ),
        "Can you just remove Brookfield and Laketown from my route, please?",
        (
            "Now that's done, I need to remove Brookfield from my route. It's no "
            "longer necessary. Can you find the shortest route after removing "
            "Brookfield and then lock the doors?"
        ),
        (
            "I'm sorry, I thought you had already removed Laketown. Can you please "
            "remove Brookfield from the route and then find the shortest route for me?"
        ),
        (
            "But you haven't removed Brookfield yet. Please remove Brookfield from "
            "my route, and then show me the shortest route to Riverton and lock "
            "the doors."
        ),
        (
            "Can you please take Brookfield and Laketown out of my route? And then, "
            "show me the shortest route to Riverton."
        ),
        (
            'She said, "I need to remove Brookfield from my route. Can you please '
            'do that?"'
        ),
        (
            "If needed, I need to remove Brookfield from my route. Can you please "
            "do that?"
        ),
    ],
)
def test_polite_named_waypoint_delete_rejects_conflicts_and_extra_actions(
    utterance: str,
) -> None:
    seed = frame()
    assert (
        recover_named_navigation_waypoint_delete_intent(
            utterance, seed, turn_id="unsafe-polite-waypoint-delete"
        )
        is seed
    )


def test_base56_correct_model_goal_grounds_without_recovery() -> None:
    utterance = "Remove Nuremberg from my route. Go straight to Paris."
    extracted = frame(
        goal(
            "delete-waypoint",
            "navigation_delete_one_waypoint",
            {
                "waypoint_name_to_delete": "Nuremberg",
                "remaining_destination_name": "Paris",
            },
        )
    )

    grounded = ground_intent(utterance, extracted, REGISTRY)

    assert grounded.desired_values_by_goal == {
        "delete-waypoint": {
            "waypoint_name_to_delete": "Nuremberg",
            "remaining_destination_name": "Paris",
        }
    }
    assert grounded.authorized_action_goal_ids == frozenset({"delete-waypoint"})


def test_ground_intent_does_not_synthesize_named_waypoint_delete() -> None:
    utterance = "Remove Nuremberg from my route. Go straight to Paris."

    grounded = ground_intent(utterance, frame(), REGISTRY)

    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "utterance",
    [
        "Remove the final destination from my route.",
        "Delete my current navigation.",
        "Remove the intermediate stop from my route.",
        "Do not remove Nuremberg from my route. Go straight to Paris.",
        "If needed, remove Nuremberg from my route. Go straight to Paris.",
        'She said, "Remove Nuremberg from my route. Go straight to Paris."',
        "She asked me to remove Nuremberg from my route. Go straight to Paris.",
        "Remove Nuremberg or Berlin from my route. Go straight to Paris.",
        "Remove Nuremberg from my route. Go straight to Paris or Lyon.",
        "Remove Nuremberg from my route. Go straight to Nuremberg.",
        "Choose the one via A11/A51.",
        (
            "Remove Nuremberg from my route. Go straight to Paris. "
            "Turn on the air conditioning."
        ),
        "Remove Nuremberg and Berlin from my route. Go straight to Paris.",
        "Remove Nuremberg-from-Berlin from my route. Go straight to Paris.",
        "Remove Nuremberg from my route. Go straight to Paris_or_Lyon.",
    ],
)
def test_named_waypoint_delete_recovery_rejects_unsafe_or_ambiguous_text(
    utterance: str,
) -> None:
    extracted = frame()

    recovered = recover_named_navigation_waypoint_delete_intent(
        utterance, extracted, turn_id="unsafe-waypoint-delete"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "operation",
    [
        "navigation_delete_final_destination",
        "delete_current_navigation",
        "set_air_conditioning",
    ],
)
def test_named_waypoint_delete_rejects_wrong_model_operation(operation: str) -> None:
    extracted = frame(goal("wrong-operation", operation, {}))

    recovered = recover_named_navigation_waypoint_delete_intent(
        "Remove Nuremberg from my route. Go straight to Paris.",
        extracted,
        turn_id="wrong-waypoint-delete-operation",
    )

    assert recovered is extracted


def test_named_waypoint_delete_grounding_requires_exact_context_names() -> None:
    utterance = "Remove Nuremberg from my route. Go straight to Paris."
    extracted = frame(
        goal(
            "delete-waypoint",
            "navigation_delete_one_waypoint",
            {
                "waypoint_name_to_delete": "Nuremberg",
                "remaining_destination_name": "Lyon",
            },
        )
    )

    grounded = ground_intent(utterance, extracted, REGISTRY)

    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()


def test_named_destination_replacement_rejects_exact_model_multiword_location() -> None:
    utterance = "Change my destination to New York."
    extracted = frame(
        goal("navigation", "start_navigation", {"destination": "New York"})
    )

    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, extracted, turn_id="multiword-destination-replacement"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "model_location",
    [
        "Munich And Open Sunroof",
        "Munich Plus Berlin",
        "Munich Versus Berlin",
        "Munich Via Berlin",
        "Munich Instead Of Berlin",
        "Munich Next Week",
    ],
)
def test_named_destination_replacement_rejects_unsafe_model_location(
    model_location: str,
) -> None:
    utterance = f"Change My Destination To {model_location}."
    extracted = frame(
        goal(
            "navigation",
            "start_navigation",
            {"destination": model_location},
        )
    )

    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, extracted, turn_id="unsafe-model-destination-replacement"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "utterance",
    [
        "Change my destination to Munich or Berlin?",
        "Change my destination to Munich and Berlin?",
        "Change my destination to Berlin or Munich?",
        "Change my destination to either Munich or Berlin?",
        "Change my destination to Munich, or Berlin?",
        "Change my destination to Munich and Berlin please.",
        "Change my destination to Munich and then Berlin.",
        "Change my destination to Munich and Berlin via A9.",
        "Change my destination to Munich and to Berlin.",
        "Change my destination to Munich and then to Berlin.",
        "Change my destination to Munich and also Berlin.",
        "Change my destination to Munich plus Berlin.",
        "Change my destination to Berlin or to Munich.",
        "Change my destination to one of Munich or Berlin.",
        "Change my destination to Munich; or Berlin.",
        "Change my destination to Munich / Berlin.",
        "Change my destination to Munich and home.",
    ],
)
def test_singular_destination_replacement_rejects_omitted_alternative(
    utterance: str,
) -> None:
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_final_destination",
            {"new_destination_id": "Munich"},
        )
    )

    grounded = ground_intent(utterance, extracted, REGISTRY)

    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()
    assert grounded.desired_values_by_goal == {}


@pytest.mark.parametrize(
    "utterance",
    [
        "Change my destination from Berlin to Munich.",
        "Change my destination to Munich instead of Berlin.",
        "Change my destination to Munich and avoid tolls.",
        "Change my destination to Munich and I mean the city itself.",
        "Change my destination to Munich and I want to go to the city, not restaurant.",
    ],
)
def test_singular_destination_replacement_preserves_explicit_old_to_new_semantics(
    utterance: str,
) -> None:
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_final_destination",
            {"new_destination_id": "Munich"},
        )
    )

    grounded = ground_intent(utterance, extracted, REGISTRY)

    assert grounded.desired_values_by_goal == {
        "navigation": {"new_destination_name": "Munich"}
    }
    assert grounded.authorized_action_goal_ids == frozenset({"navigation"})


def test_base48_route_selection_drops_inherited_model_ids_and_poi_goal() -> None:
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_final_destination",
            {"new_destination_id": "Munich", "route_id": "second_route"},
        ),
        goal(
            "poi",
            "search_poi_at_location",
            {"location_id": "Munich", "category": "restaurants"},
        ),
        goal(
            "new-navigation",
            "set_new_navigation",
            {"route_ids": ["K816", "A46"]},
        ),
    )

    result = ground_intent(
        "I want the second route, the one via K816 and A46.", extracted, REGISTRY
    )

    assert result.filtered_intent.goals == []
    assert result.desired_values_by_goal == {}
    assert result.authorized_action_goal_ids == frozenset()


def test_navigation_create_canonicalizes_qwen_destination_alias() -> None:
    extracted = frame(
        goal(
            "navigation",
            "start_navigation",
            {"destination": "Frankfurt"},
        )
    )

    result = ground_intent(
        "I need to set up navigation to Frankfurt.", extracted, REGISTRY
    )

    assert result.desired_values_by_goal == {"navigation": {"location": "Frankfurt"}}
    assert result.authorized_action_goal_ids == frozenset({"navigation"})


def test_base38_battery_trip_range_recovery_normalizes_exact_qwen_frame() -> None:
    utterance = (
        "Hey there! I'm in Milan right now and I'm thinking of heading to Prague. "
        "Can you tell me if I can make it all the way there without having to "
        "stop and charge up?"
    )
    extracted = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[
            goal("navigation", "read_current_navigation", {}),
            goal("range", "get_ev_range", {}),
            goal("location", "resolve_location", {"location": "Prague"}),
        ],
        explicit_slots={"current_location": "Milan", "destination": "Prague"},
        explicit_constraints={"avoid_charging_stops": True},
        intent_kind=IntentKind.INFORMATION,
    )
    before = extracted.model_dump(mode="json")

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="base38-turn"
    )

    assert extracted.model_dump(mode="json") == before
    assert recovered.call_for_action is False
    assert recovered.intent_kind is IntentKind.INFORMATION
    assert len(recovered.goals) == 1
    assert recovered.goals[0].semantic_operation == ("assess_battery_charge_trip_range")
    assert recovered.goals[0].desired_outcome == {"destination_name": "Prague"}
    assert recovered.explicit_slots == {"destination_name": "Prague"}
    assert recovered.explicit_constraints == {}
    assert recovered.unresolved_slots == []
    assert recovered.goal_mention_order == [recovered.goals[0].goal_id]

    grounded = ground_intent(utterance, recovered, REGISTRY)
    assert [item.semantic_operation for item in grounded.filtered_intent.goals] == [
        "assess_battery_charge_trip_range"
    ]
    assert grounded.desired_values_by_goal == {
        recovered.goals[0].goal_id: {"destination_name": "Prague"}
    }
    assert grounded.authorized_action_goal_ids == frozenset()


def test_base38_recovery_accepts_recipe_shaped_qwen_decomposition() -> None:
    utterance = (
        "Hey there! I'm in Milan right now and I'm thinking of heading to Prague. "
        "Can you tell me if I can make it all the way there without having to "
        "stop and charge up?"
    )
    extracted = IntentFrame(
        language="en",
        call_for_action=True,
        goals=[
            goal(
                "location",
                "resolve_trip_destination",
                {"destination_name": "Prague"},
            ),
            goal(
                "route",
                "find_trip_route",
                {
                    "trip_start_id": "loc_mil_253463",
                    "trip_destination_id": "destination_name",
                },
                depends_on=["location"],
            ),
            goal(
                "range",
                "assess_battery_charge_trip_range",
                {},
                depends_on=["route"],
            ),
        ],
        explicit_slots={"location": "Milan", "destination": "Prague"},
        intent_kind=IntentKind.ACTION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="base38-recipe-shaped"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered.call_for_action is False
    assert recovered.intent_kind is IntentKind.INFORMATION
    assert len(recovered.goals) == 1
    assert recovered.goals[0].desired_outcome == {"destination_name": "Prague"}
    assert [item.semantic_operation for item in grounded.filtered_intent.goals] == [
        "assess_battery_charge_trip_range"
    ]
    assert grounded.desired_values_by_goal == {
        recovered.goals[0].goal_id: {"destination_name": "Prague"}
    }
    assert grounded.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "utterance",
    [
        "Can I make it to Prague without charging?",
        "I'm planning a trip to Prague. Can I make it there without charging?",
        "Is my current battery charge enough to reach Prague?",
        "Will my current range get me to Prague?",
        "Could my car reach Prague on its current charge?",
        (
            "Can you tell me if my current battery can get me to Prague without "
            "charging?"
        ),
        ("Can you tell me if my current range will get me to Prague without charging?"),
        (
            "I'm in Milan and planning a trip to Prague. Can you tell me if I can "
            "make it all the way there without a charge with my current battery?"
        ),
    ],
)
def test_battery_trip_range_recovery_accepts_narrow_feasibility_questions(
    utterance: str,
) -> None:
    extracted = frame(
        goal("range", "get_ev_range", {}),
        goal("destination", "resolve_location", {"location": "Prague"}),
        slots={"destination": "Prague"},
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="battery-trip-turn"
    )

    assert [item.semantic_operation for item in recovered.goals] == [
        "assess_battery_charge_trip_range"
    ]
    assert recovered.goals[0].desired_outcome == {"destination_name": "Prague"}


@pytest.mark.parametrize(
    "utterance",
    [
        "Navigate me to Prague without charging.",
        "Can you route me to Prague without charging?",
        "Can you get me to Prague without charging?",
        "Please drive me to Prague without charging.",
    ],
)
def test_battery_trip_range_recovery_rejects_navigation_commands(
    utterance: str,
) -> None:
    extracted = frame(
        goal("range", "get_ev_range", {}),
        goal("destination", "resolve_location", {"location": "Prague"}),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="battery-trip-turn"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "utterance",
    [
        "I'm planning a trip to Prague with my current battery.",
        "I can make it to Prague without charging.",
        "My battery range for the trip to Prague is interesting.",
    ],
)
def test_battery_trip_range_recovery_rejects_statements(utterance: str) -> None:
    extracted = frame(
        goal("range", "get_ev_range", {}),
        goal("destination", "resolve_location", {"location": "Prague"}),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="battery-trip-turn"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "utterance",
    [
        'He asked, "Can I make it to Prague without charging?"',
        '"Can I make it to Prague without charging?" is only an example.',
        "If I had a larger battery, could I make it to Prague without charging?",
        "Hypothetically, can I make it to Prague without charging?",
        "Maybe I can make it to Prague without charging?",
    ],
)
def test_battery_trip_range_recovery_rejects_reported_meta_or_hypothetical_text(
    utterance: str,
) -> None:
    extracted = frame(
        goal("range", "get_ev_range", {}),
        goal("destination", "resolve_location", {"location": "Prague"}),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="battery-trip-turn"
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "utterance",
    [
        "Can you tell me where to charge in Prague?",
        "What charging stations are on the trip to Prague?",
        "Can I make it to Prague by noon?",
        "What is my battery range?",
    ],
)
def test_battery_trip_range_recovery_rejects_unrelated_questions(
    utterance: str,
) -> None:
    extracted = frame(
        goal("range", "get_ev_range", {}),
        goal("destination", "resolve_location", {"location": "Prague"}),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="battery-trip-turn"
    )

    assert recovered is extracted


def test_battery_trip_range_recovery_rejects_multiple_destinations() -> None:
    utterance = "Can I make it to Prague or Vienna without charging?"
    extracted = frame(
        goal("range", "get_ev_range", {}),
        goal("prague", "resolve_location", {"location": "Prague"}),
        goal("vienna", "resolve_location", {"location": "Vienna"}),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        utterance, extracted, turn_id="battery-trip-turn"
    )

    assert recovered is extracted


def test_battery_trip_range_recovery_rejects_model_hallucinated_destination() -> None:
    extracted = frame(
        goal("range", "get_ev_range", {}),
        goal("destination", "resolve_location", {"location": "Prague"}),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        "Can I make it to Paris without charging?",
        extracted,
        turn_id="battery-trip-turn",
    )

    assert recovered is extracted


def test_battery_trip_range_recovery_rejects_state_changing_model_goal() -> None:
    extracted = frame(
        goal("navigation", "navigate_to", {"location": "Prague"}),
        kind=IntentKind.INFORMATION,
    )

    recovered = recover_battery_charge_trip_range_intent(
        "Can I make it to Prague without charging?",
        extracted,
        turn_id="battery-trip-turn",
    )

    assert recovered is extracted


@pytest.mark.parametrize(
    "utterance",
    [
        "What's on my calendar today?",
        "Show me my calendar for today.",
        "Do I have any meetings today?",
    ],
)
def test_explicit_current_day_calendar_question_recovers_empty_model_frame(
    utterance: str,
) -> None:
    extracted = frame(kind=IntentKind.INFORMATION)

    recovered = recover_current_day_calendar_intent(
        utterance, extracted, turn_id="calendar-turn"
    )
    result = ground_intent(utterance, recovered, REGISTRY)

    assert [item.semantic_operation for item in result.filtered_intent.goals] == [
        "get_entries_from_calendar"
    ]
    assert result.filtered_intent.intent_kind is IntentKind.INFORMATION
    assert result.filtered_intent.call_for_action is False


@pytest.mark.parametrize(
    "utterance",
    [
        "Do not show my calendar today.",
        '"What is on my calendar today?" is only an example.',
        "If I asked, show my calendar today.",
        "What's the weather today? My calendar is synced.",
        "My calendar is ready today.",
    ],
)
def test_calendar_recovery_rejects_non_requests(utterance: str) -> None:
    extracted = frame(kind=IntentKind.INFORMATION)

    recovered = recover_current_day_calendar_intent(
        utterance, extracted, turn_id="calendar-turn"
    )

    assert recovered.goals == []


def test_possessive_name_is_grounded_for_contact_read() -> None:
    extracted = frame(
        goal("contact", "find_contact", {"first_name": "Alex"}),
        kind=IntentKind.INFORMATION,
    )

    result = ground_intent("Find Alex's contact.", extracted, REGISTRY)

    assert [item.goal_id for item in result.filtered_intent.goals] == ["contact"]
    assert result.desired_values_by_goal == {"contact": {"first_name": "Alex"}}


def test_explicit_fan_request_grounds_number_word() -> None:
    extracted = frame(
        goal("fan", "set_fan_speed", {"level": 2}),
        slots={"level": 2},
    )

    result = ground_intent("Set the fan to level two", extracted, REGISTRY)

    assert [item.goal_id for item in result.filtered_intent.goals] == ["fan"]
    assert result.authorized_action_goal_ids == frozenset({"fan"})
    assert result.desired_values_by_goal == {"fan": {"level": 2}}
    assert result.filtered_intent.explicit_slots == {"level": 2}


def test_ambient_color_derives_enabled_without_model_guessing() -> None:
    extracted = frame(
        goal(
            "ambient",
            "set_ambient_lights",
            {"color": "purple", "enabled": True},
        ),
        slots={"color": "purple", "enabled": True},
    )

    result = ground_intent("Change the ambient lights to purple.", extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"ambient"})
    assert result.desired_values_by_goal == {
        "ambient": {"color": "purple", "enabled": True}
    }
    assert result.derived_values_by_goal == {
        "ambient": {"enabled": "ambient_color_selection_enables_lights_v1"}
    }


@pytest.mark.parametrize("model_zone", [None, "both", "ALL_ZONES"])
def test_both_seat_request_grounds_all_zones(model_zone: str | None) -> None:
    desired: dict[str, object] = {"level": 1}
    if model_zone is not None:
        desired["seat_zone"] = model_zone
    extracted = frame(goal("seats", "set_seat_heating", desired))

    result = ground_intent(
        "Turn down the seat heating for both of us to level 1.",
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset({"seats"})
    assert result.desired_values_by_goal == {
        "seats": {"level": 1, "seat_zone": "ALL_ZONES"}
    }


def test_both_me_and_passenger_survives_action_clause_connector() -> None:
    extracted = frame(
        goal(
            "seats",
            "set_seat_heating",
            {"level": "lower", "seat_zone": "both"},
        )
    )

    result = ground_intent(
        "Turn down the seat heating for both me and my passenger.",
        extracted,
        REGISTRY,
    )

    assert result.desired_values_by_goal == {"seats": {"seat_zone": "ALL_ZONES"}}


def test_both_seats_collapses_duplicate_model_goals_after_grounding() -> None:
    extracted = frame(
        goal(
            "driver-seat",
            "set_seat_heating",
            {"level": "lowered", "seat_zone": "driver"},
        ),
        goal(
            "passenger-seat",
            "set_seat_heating",
            {"level": "lowered", "seat_zone": "passenger"},
        ),
    )

    result = ground_intent(
        "Turn down the seat heating for both me and my passenger.",
        extracted,
        REGISTRY,
    )

    assert result.desired_values_by_goal == {
        "driver-seat": {"seat_zone": "ALL_ZONES"},
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "Change the ambient lights to purple but keep them off.",
        "Change the ambient lights to purple while leaving them off.",
        "Change the ambient lights to purple, not on.",
        ("Set the ambient lights to purple without first needing to turn them on."),
    ],
)
def test_ambient_color_never_overrides_explicit_off(utterance: str) -> None:
    extracted = frame(
        goal(
            "ambient",
            "set_ambient_lights",
            {"color": "purple", "enabled": True},
        ),
        slots={"color": "purple", "enabled": True},
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"ambient"})
    assert result.desired_values_by_goal == {
        "ambient": {"color": "purple", "enabled": False}
    }
    assert result.derived_values_by_goal == {}


def test_airflow_direction_alias_is_canonicalized_and_grounded() -> None:
    extracted = frame(
        goal(
            "airflow",
            "set_fan_airflow_direction",
            {"airflow_direction": "windshield"},
        ),
        slots={"airflow_direction": "windshield"},
    )

    result = ground_intent("Direct the air to the windshield.", extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"airflow"})
    assert result.desired_values_by_goal == {"airflow": {"direction": "windshield"}}


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        (
            "Hey there! My front windshield is getting all foggy, can you "
            "turn on the defrost for me?",
            "FRONT",
        ),
        ("Please turn on the rear window defrost.", "REAR"),
    ],
)
def test_explicit_defrost_window_is_grounded_when_model_omits_it(
    utterance: str, expected: str
) -> None:
    extracted = frame(goal("defrost", "set_window_defrost", {"enabled": True}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"defrost"})
    assert result.desired_values_by_goal == {
        "defrost": {"enabled": True, "window": expected}
    }


def test_conflicting_defrost_windows_are_not_derived() -> None:
    extracted = frame(goal("defrost", "set_window_defrost", {"enabled": True}))

    result = ground_intent(
        "Turn on the front window or rear window defrost.", extracted, REGISTRY
    )

    assert result.desired_values_by_goal == {"defrost": {"enabled": True}}


def test_explicit_soc_range_removes_redundant_status_dependency() -> None:
    extracted = frame(
        goal("status", "read_charging_status", {}),
        goal(
            "range",
            "get_distance_by_soc",
            {"initial_soc": 50},
            depends_on=["status"],
        ),
        kind=IntentKind.INFORMATION,
    ).model_copy(update={"explicit_constraints": {"target_soc": 10}})

    result = ground_intent(
        "What is my driving range from 50% charge down to 10% charge?",
        extracted,
        REGISTRY,
    )

    assert [item.goal_id for item in result.filtered_intent.goals] == ["range"]
    assert result.filtered_intent.goals[0].depends_on == []
    assert result.desired_values_by_goal == {
        "range": {"initial_soc": 50, "final_soc": 10}
    }


def test_explicit_soc_range_preserves_separately_requested_charging_status() -> None:
    extracted = frame(
        goal("status", "read_charging_status", {}),
        goal(
            "range",
            "get_distance_by_soc",
            {"initial_soc": 50},
            depends_on=["status"],
        ),
        kind=IntentKind.INFORMATION,
    ).model_copy(update={"explicit_constraints": {"target_soc": 10}})

    result = ground_intent(
        "What is my charging status, and what is my range from 50% to 10%?",
        extracted,
        REGISTRY,
    )

    assert [item.goal_id for item in result.filtered_intent.goals] == [
        "status",
        "range",
    ]
    assert result.filtered_intent.intent_kind is IntentKind.INFORMATION
    assert result.filtered_intent.call_for_action is False


def test_duplicate_soc_endpoint_goals_collapse_to_one_range() -> None:
    extracted = frame(
        goal("range-start", "get_distance_by_soc", {"initial_soc": 50}),
        goal("range-end", "get_distance_by_soc", {"initial_soc": 10}),
        kind=IntentKind.INFORMATION,
    )

    result = ground_intent(
        "Calculate my driving range from 50% to 10%.", extracted, REGISTRY
    )

    assert [item.goal_id for item in result.filtered_intent.goals] == ["range-start"]
    assert result.desired_values_by_goal == {
        "range-start": {"initial_soc": 50, "final_soc": 10}
    }


def test_conflicting_canonical_and_alias_values_are_not_grounded() -> None:
    extracted = frame(
        goal(
            "airflow",
            "set_fan_airflow_direction",
            {"direction": "feet", "airflow_direction": "windshield"},
        )
    )

    result = ground_intent(
        "Direct the air toward feet or windshield.", extracted, REGISTRY
    )

    assert result.authorized_action_goal_ids == frozenset({"airflow"})
    assert result.desired_values_by_goal == {"airflow": {}}


def test_polite_sunroof_request_grounds_followup_percentage_phrase() -> None:
    extracted = frame(
        goal("roof", "set_sunroof_position", {"percentage": 50}),
        slots={"percentage": 50},
    )
    utterance = "Hey, can you open the sunroof a bit? Like, halfway?"

    result = ground_intent(utterance, extracted, REGISTRY)

    assert has_explicit_action_request(utterance)
    assert result.authorized_action_goal_ids == frozenset({"roof"})
    assert result.desired_values_by_goal == {"roof": {"percentage": 50}}


def test_followup_percentage_phrase_does_not_cross_bind_multiple_goals() -> None:
    extracted = frame(
        goal(
            "window",
            "set_window_position",
            {"window": "driver", "percentage": 50},
        ),
        goal("roof", "set_sunroof_position", {"percentage": 50}),
    )

    result = ground_intent(
        "Open the driver window and open the sunroof. Halfway.",
        extracted,
        REGISTRY,
    )

    assert [goal.goal_id for goal in result.filtered_intent.goals] == ["window"]
    assert result.desired_values_by_goal == {"window": {"window": "driver"}}
    assert all(
        "percentage" not in values for values in result.desired_values_by_goal.values()
    )


def test_explicit_action_detector_rejects_negated_and_quoted_commands() -> None:
    assert not has_explicit_action_request("Do not open the sunroof")
    assert not has_explicit_action_request('"Open the sunroof" is an example')


def test_explicit_action_detector_keeps_main_request_before_conditional_sentence() -> (
    None
):
    utterance = (
        "Open the sunroof to 50%. If you need to open the sunshade first, "
        "open it fully to 100%."
    )
    extracted = frame(goal("roof", "set_sunroof_position", {"percentage": 50}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert has_explicit_action_request(utterance)
    assert result.authorized_action_goal_ids == frozenset({"roof"})
    assert result.desired_values_by_goal == {"roof": {"percentage": 50}}
    assert focus_explicit_action_request(utterance) == "Open the sunroof to 50%."


@pytest.mark.parametrize(
    "utterance",
    [
        "If it stops raining, open the sunroof to 50%.",
        "Open the sunroof to 50% unless it stops raining.",
        "Could you maybe open the sunroof to 50%?",
        "It might be nice; open the sunroof to 50%.",
    ],
)
def test_hypothetical_sentence_does_not_authorize_action(utterance: str) -> None:
    extracted = frame(goal("roof", "set_sunroof_position", {"percentage": 50}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert not has_explicit_action_request(utterance)
    assert result.authorized_action_goal_ids == frozenset()
    assert result.filtered_intent.goals == []


def test_later_independent_sentence_can_authorize_after_hypothetical_sentence() -> None:
    extracted = frame(
        goal("roof", "set_sunroof_position", {"percentage": 50}),
        goal("fan", "set_fan_speed", {"level": 2}),
    )

    result = ground_intent(
        "Maybe open the sunroof to 50%. Set the fan to level two.",
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset({"fan"})
    assert result.desired_values_by_goal == {"fan": {"level": 2}}


def test_explicit_fan_request_rejects_injected_different_value() -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 5}))

    result = ground_intent("Set the fan to level two", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


def test_missing_action_parameter_retains_only_unbound_authorized_goal() -> None:
    extracted = frame(goal("roof", "set_sunroof_position", {}))

    result = ground_intent("Open the sunroof.", extracted, REGISTRY)

    assert [item.goal_id for item in result.filtered_intent.goals] == ["roof"]
    assert result.filtered_intent.goals[0].desired_outcome == {}
    assert result.authorized_action_goal_ids == frozenset({"roof"})
    assert result.desired_values_by_goal == {"roof": {}}


def test_turn_on_air_conditioning_grounds_boolean() -> None:
    extracted = frame(
        goal("ac", "set_air_conditioning", {"enabled": True}),
        slots={"enabled": True},
    )

    result = ground_intent("Turn on the air conditioning", extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"ac"})
    assert result.desired_values_by_goal == {"ac": {"enabled": True}}


def test_navigate_to_named_destination_grounds_case_insensitively() -> None:
    extracted = frame(
        goal("nav", "navigate_to", {"location": "Airport"}),
        slots={"location": "Airport"},
    )

    result = ground_intent("Navigate to Airport", extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"nav"})
    assert result.desired_values_by_goal == {"nav": {"location": "Airport"}}


def test_read_goal_drops_model_added_system_values_but_keeps_user_value() -> None:
    extracted = frame(
        goal(
            "weather",
            "get_weather",
            {
                "location_id": "Airport",
                "month": 7,
                "day": 11,
                "hour": 17,
            },
        ),
        kind=IntentKind.INFORMATION,
    )

    result = ground_intent("Check the weather at Airport.", extracted, REGISTRY)

    assert result.filtered_intent.goals[0].desired_outcome == {"location_id": "Airport"}


def test_descriptive_state_is_not_an_action_request() -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 2}))

    result = ground_intent("The fan is set to level two.", extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset()
    assert result.filtered_intent.goals == []


def test_negated_action_is_rejected() -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 2}))

    result = ground_intent("Do not set the fan to level two", extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset()


def test_quoted_example_is_not_an_action_request() -> None:
    extracted = frame(goal("fan", "set_fan_speed", {"level": 2}))

    result = ground_intent(
        '"Set the fan to level two" is an example, not a command.',
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset()
    assert result.filtered_intent.goals == []


def test_values_cannot_move_between_goal_clauses() -> None:
    extracted = frame(
        goal("fan", "set_fan_speed", {"level": 20}),
        goal(
            "temperature",
            "set_climate_temperature",
            {"temperature": 2},
        ),
    )

    result = ground_intent(
        "Set the fan to level two and set the temperature to 20",
        extracted,
        REGISTRY,
    )

    assert result.filtered_intent.goals == []
    assert result.desired_values_by_goal == {}


def test_opposite_boolean_values_are_bound_to_separate_goal_clauses() -> None:
    extracted = frame(
        goal("ac", "set_air_conditioning", {"enabled": True}),
        goal("fog", "set_fog_lights", {"enabled": False}),
    )

    result = ground_intent(
        "Turn on the air conditioning and turn off the fog lights",
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset({"ac", "fog"})
    assert result.desired_values_by_goal == {
        "ac": {"enabled": True},
        "fog": {"enabled": False},
    }


@pytest.mark.parametrize(
    ("utterance", "reported_operation"),
    [
        ("Turn on the high beams.", "set_low_beams"),
        ("Turn on the low beams.", "set_high_beams"),
        ("Turn on the low beams but not the high beams.", "set_high_beams"),
        ("Turn on the low beams rather than the high beams.", "set_high_beams"),
    ],
)
def test_explicit_beam_request_rejects_model_reported_sibling(
    utterance: str,
    reported_operation: str,
) -> None:
    extracted = frame(
        goal("lights", reported_operation, {"enabled": True}),
        slots={"enabled": True},
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()
    assert result.desired_values_by_goal == {}


@pytest.mark.parametrize("reported_operation", ["set_high_beams", "set_low_beams"])
def test_bare_beams_never_authorize_a_model_selected_sibling(
    reported_operation: str,
) -> None:
    extracted = frame(
        goal("lights", reported_operation, {"enabled": True}),
        slots={"enabled": True},
    )

    result = ground_intent("Turn on the beams.", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


def test_sibling_identity_is_scoped_to_the_action_clause() -> None:
    extracted = frame(
        goal("lights", "set_high_beams", {"enabled": True}),
        slots={"enabled": True},
    )

    result = ground_intent(
        "What are high beams? Turn on the low beams.", extracted, REGISTRY
    )

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    ("utterance", "reported_operation"),
    [
        (
            "Delete the final destination from the current navigation.",
            "navigation_delete_one_waypoint",
        ),
        (
            "Delete the waypoint from the current navigation.",
            "navigation_delete_final_destination",
        ),
        (
            "Replace the final destination in the current navigation.",
            "navigation_replace_one_waypoint",
        ),
        (
            "Replace the waypoint in the current navigation.",
            "navigation_replace_final_destination",
        ),
    ],
)
def test_navigation_edit_rejects_cross_recipe_object_identity(
    utterance: str,
    reported_operation: str,
) -> None:
    extracted = frame(goal("navigation", reported_operation, {}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    ("utterance", "reported_operation"),
    [
        (
            "Adjust the sunroof to match the sunshade.",
            "set_sunshade_position",
        ),
        (
            "Adjust the sunshade to match the sunroof.",
            "set_sunroof_position",
        ),
    ],
)
def test_roof_match_rejects_model_reported_reverse_target(
    utterance: str, reported_operation: str
) -> None:
    extracted = frame(goal("roof", reported_operation, {}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


def test_waypoint_delete_cannot_be_promoted_to_whole_navigation_delete() -> None:
    extracted = frame(goal("navigation", "delete_current_navigation", {}))

    result = ground_intent("Delete navigation waypoint old.", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


def test_navigation_start_cannot_be_changed_to_navigation_delete() -> None:
    extracted = frame(goal("navigation", "delete_current_navigation", {}))

    result = ground_intent("Start navigation to airport.", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


def test_navigation_delete_cannot_inherit_start_from_injected_boolean() -> None:
    extracted = frame(
        goal("navigation", "delete_current_navigation", {"enabled": True})
    )

    result = ground_intent("Start navigation to airport.", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    ("utterance", "reported_operation"),
    [
        (
            "Delete the waypoint from the current navigation.",
            "navigation_delete_one_waypoint",
        ),
        (
            "Delete the final destination from the current navigation.",
            "navigation_delete_final_destination",
        ),
        (
            "Replace the waypoint in the current navigation.",
            "navigation_replace_one_waypoint",
        ),
        (
            "Replace the final destination in the current navigation.",
            "navigation_replace_final_destination",
        ),
    ],
)
def test_navigation_edit_accepts_matching_cross_recipe_object_identity(
    utterance: str,
    reported_operation: str,
) -> None:
    extracted = frame(goal("navigation", reported_operation, {}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"navigation"})
    assert result.desired_values_by_goal == {"navigation": {}}


def test_base34_intermediate_stop_retains_replacement_and_name_selector() -> None:
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": "intermediate_stop",
                "new_waypoint_id": "Frankfurt",
                "route_id_leading_to_new_waypoint": "invented-route-to",
                "route_id_leading_away_from_new_waypoint": "invented-route-away",
            },
        )
    )

    result = ground_intent(
        "Hey, can you replace my current intermediate stop with Frankfurt?",
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset({"navigation"})
    assert result.desired_values_by_goal == {
        "navigation": {"new_waypoint_name": "Frankfurt"}
    }


def test_navigation_replacement_city_names_use_separate_selectors() -> None:
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": "Bucharest",
                "new_waypoint_id": "Frankfurt",
            },
        )
    )

    result = ground_intent(
        "Replace intermediate stop Bucharest with Frankfurt.", extracted, REGISTRY
    )

    assert result.desired_values_by_goal == {
        "navigation": {
            "waypoint_name_to_replace": "Bucharest",
            "new_waypoint_name": "Frankfurt",
        }
    }


def test_navigation_change_synonym_selects_replace_not_add() -> None:
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_one_waypoint",
            {"new_waypoint_id": "Frankfurt"},
        )
    )

    result = ground_intent(
        "Can you change my intermediate stop to Frankfurt?", extracted, REGISTRY
    )

    assert result.authorized_action_goal_ids == frozenset({"navigation"})
    assert result.desired_values_by_goal == {
        "navigation": {"new_waypoint_name": "Frankfurt"}
    }


def test_literal_navigation_entity_ids_keep_id_parameter_names() -> None:
    extracted = frame(
        goal(
            "navigation",
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": "loc_buc_567170",
                "new_waypoint_id": "loc_fra_178468",
            },
        )
    )

    result = ground_intent(
        "Replace intermediate stop loc_buc_567170 with loc_fra_178468.",
        extracted,
        REGISTRY,
    )

    assert result.desired_values_by_goal == {
        "navigation": {
            "waypoint_id_to_replace": "loc_buc_567170",
            "new_waypoint_id": "loc_fra_178468",
        }
    }


def test_base36_retains_derived_final_destination_delete_after_bad_id_is_stripped() -> (
    None
):
    extracted = frame(
        goal(
            "navigation",
            "navigation_delete_final_destination",
            {"destination_id": "Cologne"},
        )
    )

    result = ground_intent(
        "Remove the final destination from my current navigation route.",
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset({"navigation"})
    assert result.desired_values_by_goal == {"navigation": {}}


@pytest.mark.parametrize(
    "utterance",
    [
        "Remove the final destination from my current navigation route.",
        "Remove the final stop from my current navigation route.",
    ],
)
def test_base36_remove_synonym_selects_final_destination_delete(
    utterance: str,
) -> None:
    extracted = frame(goal("navigation", "navigation_delete_final_destination", {}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"navigation"})
    assert result.desired_values_by_goal == {"navigation": {}}


@pytest.mark.parametrize(
    ("utterance", "operation", "provided_key", "name", "selector"),
    [
        (
            "Add Paris as a navigation waypoint.",
            "navigation_add_one_waypoint",
            "new_waypoint_id",
            "Paris",
            "new_waypoint_name",
        ),
        (
            "Delete navigation waypoint Bucharest.",
            "navigation_delete_one_waypoint",
            "waypoint_id",
            "Bucharest",
            "waypoint_name_to_delete",
        ),
        (
            "Delete Cologne as the final navigation destination.",
            "navigation_delete_final_destination",
            "destination_id",
            "Cologne",
            "destination_name_to_delete",
        ),
        (
            "Replace the final navigation destination with Paris.",
            "navigation_replace_final_destination",
            "new_destination_id",
            "Paris",
            "new_destination_name",
        ),
    ],
)
def test_navigation_city_names_are_not_retained_as_entity_ids(
    utterance: str,
    operation: str,
    provided_key: str,
    name: str,
    selector: str,
) -> None:
    extracted = frame(goal("navigation", operation, {provided_key: name}))

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.desired_values_by_goal == {"navigation": {selector: name}}


@pytest.mark.parametrize(
    ("utterance", "reported_operation"),
    [
        ("Replace the final stop with Frankfurt.", "navigation_replace_one_waypoint"),
        ("Stop current navigation.", "navigation_delete_one_waypoint"),
        ("Delete the intermediate stop from navigation.", "delete_current_navigation"),
        (
            "Do not replace the intermediate stop with Frankfurt.",
            "navigation_replace_one_waypoint",
        ),
        (
            "If needed, replace the intermediate stop with Frankfurt.",
            "navigation_replace_one_waypoint",
        ),
    ],
)
def test_navigation_stop_alias_does_not_cross_action_or_object_boundaries(
    utterance: str, reported_operation: str
) -> None:
    extracted = frame(
        goal(
            "navigation",
            reported_operation,
            {"new_waypoint_id": "Frankfurt"},
        )
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


def test_whole_navigation_delete_still_authorizes_matching_request() -> None:
    extracted = frame(goal("navigation", "delete_current_navigation", {}))

    result = ground_intent("Delete the current navigation.", extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"navigation"})
    assert result.desired_values_by_goal == {"navigation": {}}


def test_navigation_identity_ignores_explicitly_rejected_sibling() -> None:
    extracted = frame(goal("navigation", "navigation_delete_one_waypoint", {}))

    result = ground_intent(
        "Delete the navigation waypoint rather than the final destination.",
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset({"navigation"})


def test_navigation_edit_identity_is_scoped_to_the_action_clause() -> None:
    extracted = frame(goal("navigation", "navigation_replace_one_waypoint", {}))

    result = ground_intent(
        "What is a waypoint? Replace the final destination in navigation.",
        extracted,
        REGISTRY,
    )

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "utterance",
    [
        "Set the fan speed to auto.",
        "Set the fan airflow direction to auto.",
    ],
)
def test_explicit_fan_sibling_rejects_model_selected_circulation(
    utterance: str,
) -> None:
    extracted = frame(
        goal("fan", "set_air_circulation", {"mode": "AUTO"}),
        slots={"mode": "AUTO"},
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "utterance",
    ["Turn on the reading light.", "Turn on the ambient lights."],
)
def test_cross_recipe_light_identity_rejects_model_selected_fog_light(
    utterance: str,
) -> None:
    extracted = frame(
        goal("lights", "set_fog_lights", {"enabled": True}),
        slots={"enabled": True},
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    ("utterance", "reported_operation"),
    [
        ("Turn on the high beams.", "set_high_beams"),
        ("Turn on the low beams.", "set_low_beams"),
        ("Turn on the fog lights.", "set_fog_lights"),
    ],
)
def test_explicit_exterior_light_request_accepts_matching_sibling(
    utterance: str,
    reported_operation: str,
) -> None:
    extracted = frame(
        goal("lights", reported_operation, {"enabled": True}),
        slots={"enabled": True},
    )

    result = ground_intent(utterance, extracted, REGISTRY)

    assert result.authorized_action_goal_ids == frozenset({"lights"})
    assert result.desired_values_by_goal == {"lights": {"enabled": True}}


def test_shared_recipe_name_does_not_cross_bind_sibling_goal_values() -> None:
    extracted = frame(
        goal("fan", "set_fan_speed", {"level": 2}),
        goal("circulation", "set_air_circulation", {"mode": "RECIRCULATE"}),
    )

    result = ground_intent(
        "Set fan level two and set air circulation to recirculate.",
        extracted,
        REGISTRY,
    )

    assert result.authorized_action_goal_ids == frozenset({"fan", "circulation"})


def test_dropped_dependency_also_drops_dependent_action() -> None:
    extracted = frame(
        goal("invented", "invented_operation", {}),
        goal(
            "fan",
            "set_fan_speed",
            {"level": 2},
            depends_on=["invented"],
        ),
    )

    result = ground_intent("Set the fan to level two", extracted, REGISTRY)

    assert result.filtered_intent.goals == []
    assert result.authorized_action_goal_ids == frozenset()


def test_base64_vague_waypoint_add_preserves_unresolved_disambiguation() -> None:
    add_goal = goal("add-stop", "navigation_add_one_waypoint", {})
    extracted = IntentFrame(
        language="en",
        call_for_action=True,
        goals=[add_goal],
        unresolved_slots=[
            AmbiguitySlot(
                name="new_waypoint_name",
                candidates=[
                    Candidate(candidate_id="stuttgart", value="Stuttgart"),
                    Candidate(candidate_id="karlsruhe", value="Karlsruhe"),
                ],
                goal_id=add_goal.goal_id,
                source_turn_ids=["base64-vague"],
            )
        ],
        intent_source_turn_ids=["base64-vague"],
        intent_kind=IntentKind.ACTION,
    )
    utterance = "Could you add another stop to my route?"

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="base64-vague"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert [item.goal_id for item in grounded.filtered_intent.goals] == ["add-stop"]
    assert grounded.filtered_intent.unresolved_slots == extracted.unresolved_slots
    assert grounded.authorized_action_goal_ids == frozenset({"add-stop"})
    assert grounded.desired_values_by_goal == {"add-stop": {}}


@pytest.mark.parametrize(
    "poi_operation",
    ["search_poi_at_location", "search_poi_at_current_destination"],
)
def test_base64_compound_add_and_next_stop_poi_are_canonicalized(
    poi_operation: str,
) -> None:
    utterance = (
        "Hey there, friend! I'm on a road trip and I need to make a couple of "
        "changes to my route. Could you add Stuttgart as a stop after Cologne? "
        "Also, I'm getting a bit hungry, so could you find a restaurant at my "
        "next stop?"
    )
    extracted = frame(
        goal(
            "add-stop",
            "navigation_add_one_waypoint",
            {
                "waypoint_id_to_add": "loc_stu_model_guess",
                "waypoint_id_before_new_waypoint": "loc_col_model_guess",
                "waypoint_id_after_new_waypoint": "loc_lux_model_guess",
                "route_id_leading_to_new_waypoint": "rll_model_guess_in",
                "route_id_leading_away_from_new_waypoint": "rll_model_guess_out",
            },
        ),
        goal(
            "poi",
            poi_operation,
            {
                "category_poi": "restaurants",
                "location_id": "loc_model_guess",
            },
        ),
    )
    before = extracted.model_dump(mode="json")

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="base64-first-turn"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert extracted.model_dump(mode="json") == before
    assert [item.semantic_operation for item in recovered.goals] == [
        "navigation_add_one_waypoint",
        "search_poi_at_next_navigation_stop",
    ]
    assert recovered.goals[0].desired_outcome == {
        "new_waypoint_name": "Stuttgart",
        "previous_waypoint_name": "Cologne",
    }
    assert recovered.goals[1].desired_outcome == {"category": "restaurants"}
    assert recovered.explicit_slots == {
        "new_waypoint_name": "Stuttgart",
        "previous_waypoint_name": "Cologne",
        "category": "restaurants",
    }
    assert grounded.authorized_action_goal_ids == frozenset({"add-stop"})
    assert grounded.desired_values_by_goal == {
        "add-stop": {
            "new_waypoint_name": "Stuttgart",
            "previous_waypoint_name": "Cologne",
        },
        "poi": {"category": "restaurants"},
    }


@pytest.mark.parametrize(
    ("utterance", "adjacency_parameter", "new_name", "adjacent_name"),
    [
        (
            "Add Stuttgart as a stop after Cologne.",
            "previous_waypoint_name",
            "Stuttgart",
            "Cologne",
        ),
        (
            "I want to add Stuttgart after Cologne.",
            "previous_waypoint_name",
            "Stuttgart",
            "Cologne",
        ),
        (
            "Add Stuttgart to my route after Cologne.",
            "previous_waypoint_name",
            "Stuttgart",
            "Cologne",
        ),
        (
            "Add Stuttgart as a stop before Cologne.",
            "next_waypoint_name",
            "Stuttgart",
            "Cologne",
        ),
        (
            "Add Cologne as a stop after Stuttgart.",
            "previous_waypoint_name",
            "Cologne",
            "Stuttgart",
        ),
        (
            "After Cologne, add Stuttgart as a stop.",
            "previous_waypoint_name",
            "Stuttgart",
            "Cologne",
        ),
        (
            "Before Cologne, add Stuttgart as a stop.",
            "next_waypoint_name",
            "Stuttgart",
            "Cologne",
        ),
    ],
)
def test_base64_named_add_preserves_before_after_roles(
    utterance: str,
    adjacency_parameter: str,
    new_name: str,
    adjacent_name: str,
) -> None:
    recovered = recover_navigation_waypoint_context_intent(
        utterance, frame(), turn_id="base64-add"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered.goals[0].desired_outcome == {
        "new_waypoint_name": new_name,
        adjacency_parameter: adjacent_name,
    }
    assert next(iter(grounded.desired_values_by_goal.values())) == (
        recovered.goals[0].desired_outcome
    )


def test_base64_split_next_stop_restaurant_is_contextual_read() -> None:
    utterance = "Could you find a restaurant at my next intermediate stop?"

    recovered = recover_navigation_waypoint_context_intent(
        utterance,
        frame(kind=IntentKind.INFORMATION),
        turn_id="base64-next-stop-poi",
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered.call_for_action is False
    assert recovered.intent_kind is IntentKind.INFORMATION
    assert recovered.goals[0].semantic_operation == (
        "search_poi_at_next_navigation_stop"
    )
    assert recovered.goals[0].desired_outcome == {"category": "restaurants"}
    assert grounded.authorized_action_goal_ids == frozenset()
    assert next(iter(grounded.desired_values_by_goal.values())) == {
        "category": "restaurants"
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "Do not add Stuttgart as a stop after Cologne.",
        "If needed, add Stuttgart as a stop after Cologne.",
        "Maybe add Stuttgart as a stop after Cologne.",
        "She asked me to add Stuttgart as a stop after Cologne.",
        'She said, "Add Stuttgart as a stop after Cologne."',
        '"Add Stuttgart as a stop after Cologne" is only an example.',
    ],
)
def test_base64_named_add_rejects_non_executable_text(utterance: str) -> None:
    extracted = frame(
        goal(
            "add-stop",
            "navigation_add_one_waypoint",
            {"new_waypoint_id": "Stuttgart", "previous_waypoint_id": "Cologne"},
        )
    )

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="base64-unsafe-add"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "utterance",
    [
        "Add Stuttgart or Karlsruhe as a stop after Cologne.",
        "Add Stuttgart as a stop after Cologne or Luxembourg.",
        "Add Stuttgart as a stop after Cologne and Luxembourg.",
        "Replace Cologne or Bonn with Karlsruhe.",
        "Replace Cologne with Karlsruhe or Stuttgart.",
    ],
)
def test_base64_navigation_edits_reject_unresolved_named_conflicts(
    utterance: str,
) -> None:
    operation = (
        "navigation_replace_one_waypoint"
        if utterance.startswith("Replace")
        else "navigation_add_one_waypoint"
    )
    extracted = frame(
        goal(
            "navigation",
            operation,
            {
                "new_waypoint_id": (
                    "Karlsruhe"
                    if operation.startswith("navigation_replace")
                    else "Stuttgart"
                )
            },
        )
    )

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="base64-name-conflict"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()


def test_base64_named_conflict_preserves_model_disambiguation_boundary() -> None:
    add_goal = goal("add-stop", "navigation_add_one_waypoint", {})
    extracted = IntentFrame(
        language="en",
        call_for_action=True,
        goals=[add_goal],
        unresolved_slots=[
            AmbiguitySlot(
                name="new_waypoint_name",
                candidates=[
                    Candidate(candidate_id="stuttgart", value="Stuttgart"),
                    Candidate(candidate_id="karlsruhe", value="Karlsruhe"),
                ],
                goal_id=add_goal.goal_id,
                source_turn_ids=["base64-conflict"],
            )
        ],
        intent_source_turn_ids=["base64-conflict"],
        intent_kind=IntentKind.ACTION,
    )
    utterance = "Add Stuttgart or Karlsruhe as a stop after Cologne."

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="base64-conflict"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert [item.goal_id for item in grounded.filtered_intent.goals] == ["add-stop"]
    assert grounded.filtered_intent.unresolved_slots == extracted.unresolved_slots
    assert grounded.authorized_action_goal_ids == frozenset({"add-stop"})


def test_base64_replace_ignores_negative_poi_clause_and_keeps_fastest() -> None:
    utterance = (
        "Hmm, not really feeling those options. Actually, could we replace "
        "Cologne with Karlsruhe instead? I know a good spot there, so no need "
        "to search for a restaurant. And when you show me the route options, "
        "always go with the fastest one, okay?"
    )
    extracted = frame(
        goal(
            "replace-stop",
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": "loc_col_model_guess",
                "new_waypoint_id": "loc_kar_model_guess",
                "route_id_leading_to_new_waypoint": "rll_model_guess_in",
                "route_id_leading_away_from_new_waypoint": "rll_model_guess_out",
            },
        ),
        goal(
            "poi",
            "search_poi_at_location",
            {"location_id": "loc_kar_model_guess", "category": "restaurants"},
        ),
    )

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="base64-second-turn"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert len(recovered.goals) == 1
    assert recovered.goals[0].goal_id == "replace-stop"
    assert recovered.goals[0].semantic_operation == "navigation_replace_one_waypoint"
    assert recovered.goals[0].desired_outcome == {
        "waypoint_name_to_replace": "Cologne",
        "new_waypoint_name": "Karlsruhe",
        "route_choice_alias": "fastest",
    }
    assert recovered.explicit_slots == recovered.goals[0].desired_outcome
    assert grounded.authorized_action_goal_ids == frozenset({"replace-stop"})
    assert grounded.desired_values_by_goal == {
        "replace-stop": recovered.goals[0].desired_outcome
    }


@pytest.mark.parametrize("route_choice", ["shortest", "first", "second", "third"])
def test_named_waypoint_edits_retain_one_explicit_route_choice(
    route_choice: str,
) -> None:
    utterance = (
        "Add Gamma after Omega as a new waypoint. "
        f"Use the {route_choice} route."
    )
    extracted = frame(
        goal(
            "add-stop",
            "navigation_add_one_waypoint",
            {"new_waypoint_id": "model_guess"},
        )
    )

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="synthetic-route-choice"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered.goals[0].desired_outcome == {
        "new_waypoint_name": "Gamma",
        "previous_waypoint_name": "Omega",
        "route_choice_alias": route_choice,
    }
    assert grounded.authorized_action_goal_ids == frozenset({"add-stop"})
    assert grounded.desired_values_by_goal == {
        "add-stop": recovered.goals[0].desired_outcome
    }


def test_named_waypoint_edit_does_not_default_an_ambiguous_route_choice() -> None:
    utterance = (
        "Add Gamma after Omega as a new waypoint. "
        "Use the fastest or shortest route."
    )
    extracted = frame(
        goal(
            "add-stop",
            "navigation_add_one_waypoint",
            {"new_waypoint_id": "model_guess"},
        )
    )

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="synthetic-ambiguous-route-choice"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert grounded.authorized_action_goal_ids == frozenset()


@pytest.mark.parametrize(
    "direction",
    ["take the first exit", "take the second left", "take the third turn"],
)
def test_named_waypoint_edit_does_not_treat_driving_direction_as_route_choice(
    direction: str,
) -> None:
    utterance = f"Add Gamma after Omega as a new waypoint, then {direction}."
    extracted = frame(
        goal(
            "add-stop",
            "navigation_add_one_waypoint",
            {"new_waypoint_id": "model_guess"},
        )
    )

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="synthetic-driving-direction"
    )

    assert recovered.goals[0].desired_outcome == {
        "new_waypoint_name": "Gamma",
        "previous_waypoint_name": "Omega",
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "I don't like the old choices. Replace Cologne with Karlsruhe.",
        "Replace Cologne with Karlsruhe, so no need to search for a restaurant.",
        "Replace Cologne with Karlsruhe and don't search for restaurants there.",
    ],
)
def test_base64_replace_negation_is_clause_scoped(utterance: str) -> None:
    recovered = recover_navigation_waypoint_context_intent(
        utterance, frame(), turn_id="base64-clause-negation"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered.goals[0].desired_outcome == {
        "waypoint_name_to_replace": "Cologne",
        "new_waypoint_name": "Karlsruhe",
    }
    assert grounded.authorized_action_goal_ids == frozenset(
        {recovered.goals[0].goal_id}
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "Do not replace Cologne with Karlsruhe.",
        "If needed, replace Cologne with Karlsruhe.",
        "Maybe replace Cologne with Karlsruhe.",
        "She asked me to replace Cologne with Karlsruhe.",
        'She said, "Replace Cologne with Karlsruhe."',
        '"Replace Cologne with Karlsruhe" is only an example.',
    ],
)
def test_base64_named_replace_rejects_non_executable_text(utterance: str) -> None:
    extracted = frame(
        goal(
            "replace-stop",
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": "Cologne",
                "new_waypoint_id": "Karlsruhe",
            },
        )
    )

    recovered = recover_navigation_waypoint_context_intent(
        utterance, extracted, turn_id="base64-unsafe-replace"
    )
    grounded = ground_intent(utterance, recovered, REGISTRY)

    assert recovered is extracted
    assert grounded.filtered_intent.goals == []
    assert grounded.authorized_action_goal_ids == frozenset()
