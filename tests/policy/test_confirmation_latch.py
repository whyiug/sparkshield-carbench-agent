from __future__ import annotations

import pytest

from track_1_agent_under_test.car_guard.domain import (
    ConfirmationScope,
    ConfirmationStatus,
    OfficialToolCall,
)
from track_1_agent_under_test.car_guard.policy import (
    ConfirmationLatch,
    LatchResolutionKind,
)


def call(tool_name: str, **arguments) -> OfficialToolCall:
    return OfficialToolCall(tool_name=tool_name, arguments=arguments)


def scope(
    goal_id: str,
    actions: list[OfficialToolCall],
    *,
    requested: int = 2,
    expires: int = 4,
) -> ConfirmationScope:
    return ConfirmationScope(
        goal_ids=[goal_id],
        ordered_actions=actions,
        requested_at_user_turn=requested,
        expires_after_user_turn=expires,
    )


def test_single_explicit_yes_binds_exact_bundle_and_is_one_shot() -> None:
    actions = [
        call("set_fog_lights", on=False),
        call("set_head_lights_high_beams", on=True),
    ]
    latch = ConfirmationLatch()
    armed = latch.arm(scope("high-beam", actions), confirmation_id="confirm-lights")

    resolution = latch.resolve(
        "Yes, please",
        current_user_turn=3,
        source_turn_id="turn-3",
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED
    assert resolution.confirmation is not None
    assert resolution.confirmation.status is ConfirmationStatus.CONFIRMED
    assert (
        latch.authorized_confirmation(
            goal_ids=["high-beam"],
            ordered_actions=actions,
            current_user_turn=3,
        )
        == resolution.confirmation
    )
    assert (
        latch.consume_authorization(
            goal_ids=["high-beam"],
            ordered_actions=actions,
            current_user_turn=3,
        )
        == resolution.confirmation
    )
    assert armed.confirmation_id not in {
        item.confirmation_id for item in latch.confirmations
    }


@pytest.mark.parametrize(
    "response",
    [
        "Yes, go ahead.",
        "Yes, please go ahead.",
        "Sure, please go ahead.",
        "Please go ahead.",
        "Absolutely.",
        "Yep.",
        "Yeah, that sounds good. Go for it.",
        "Yes, let's do it.",
        "Yes, please do that.",
        "Yes, I want to do that.",
    ],
)
def test_clear_natural_affirmations_confirm_one_exact_pending_bundle(
    response: str,
) -> None:
    actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-roof"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


@pytest.mark.parametrize(
    "response",
    [
        (
            "Yeah, go ahead and open the sunroof to 50%. And if the sunshade "
            "needs to open first, then open it all the way, 100%."
        ),
        "Yes, that's right. Open the sunroof to 50% and the sunshade to 100%.",
        "Yes, that's what I want. Go ahead.",
        (
            "Yes, please go ahead and open the sunroof. I'm okay with the "
            "sunshade opening fully too."
        ),
        "Yes, I still want to open it, even if it's raining.",
        "Yes, please proceed. I've confirmed this multiple times now.",
        "I've already said yes several times. Can you just open it now?",
        "Why do you keep asking? I've already confirmed that I want it open.",
    ],
)
def test_action_restatement_confirms_only_matching_frozen_numbers(
    response: str,
) -> None:
    actions = [
        call("open_close_sunshade", percentage=100),
        call("open_close_sunroof", percentage=50),
    ]
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-roof"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


def test_ambiguous_yes_does_not_choose_between_pending_latches() -> None:
    latch = ConfirmationLatch()
    first = latch.arm(
        scope(
            "open-window", [call("open_close_window", window="DRIVER", percentage=50)]
        ),
        confirmation_id="confirm-window",
    )
    second = latch.arm(
        scope("open-roof", [call("open_close_sunroof", percentage=50)]),
        confirmation_id="confirm-roof",
    )

    resolution = latch.resolve(
        "yes",
        current_user_turn=3,
        source_turn_id="turn-3",
    )

    assert resolution.kind is LatchResolutionKind.AMBIGUOUS
    assert resolution.candidate_confirmation_ids == [
        first.confirmation_id,
        second.confirmation_id,
    ]
    assert all(item.status is ConfirmationStatus.PENDING for item in latch.pending)


def test_targeted_yes_can_select_one_of_multiple_pending_latches() -> None:
    latch = ConfirmationLatch()
    latch.arm(
        scope(
            "open-window", [call("open_close_window", window="DRIVER", percentage=50)]
        ),
        confirmation_id="confirm-window",
    )
    latch.arm(
        scope("open-roof", [call("open_close_sunroof", percentage=50)]),
        confirmation_id="confirm-roof",
    )

    resolution = latch.resolve(
        "confirm",
        current_user_turn=3,
        source_turn_id="turn-3",
        confirmation_id="confirm-roof",
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED
    assert resolution.confirmation is not None
    assert resolution.confirmation.confirmation_id == "confirm-roof"
    assert latch.get("confirm-window").status is ConfirmationStatus.PENDING


@pytest.mark.parametrize(
    ("changed_goals", "changed_actions"),
    [
        (
            ["different-goal"],
            [call("open_close_window", window="DRIVER", percentage=50)],
        ),
        (
            ["open-window"],
            [call("open_close_window", window="DRIVER", percentage=75)],
        ),
        (
            ["open-window"],
            [call("open_close_window", window="PASSENGER", percentage=50)],
        ),
        (
            ["open-window"],
            [call("open_close_sunroof", percentage=50)],
        ),
        (
            ["open-window"],
            [
                call("set_air_conditioning", on=False),
                call("open_close_window", window="DRIVER", percentage=50),
            ],
        ),
    ],
    ids=["goal", "parameter", "target", "action", "bundle"],
)
def test_any_goal_parameter_target_or_action_change_invalidates_latch(
    changed_goals: list[str],
    changed_actions: list[OfficialToolCall],
) -> None:
    original_actions = [call("open_close_window", window="DRIVER", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(
        scope("open-window", original_actions),
        confirmation_id="confirm-window",
    )

    resolution = latch.invalidate_if_scope_changed(
        "confirm-window",
        goal_ids=changed_goals,
        ordered_actions=changed_actions,
        current_user_turn=3,
        source_turn_id="turn-3",
    )

    assert resolution is not None
    assert resolution.kind is LatchResolutionKind.INVALIDATED
    assert latch.get("confirm-window").status is ConfirmationStatus.CANCELLED
    assert latch.active == []


def test_confirmation_id_cannot_override_a_changed_bundle() -> None:
    original_actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(
        scope("open-roof", original_actions),
        confirmation_id="confirm-roof",
    )

    resolution = latch.resolve(
        "yes",
        current_user_turn=3,
        source_turn_id="turn-3",
        confirmation_id="confirm-roof",
        goal_ids=["open-roof"],
        ordered_actions=[call("open_close_sunroof", percentage=100)],
    )

    assert resolution.kind is LatchResolutionKind.INVALIDATED
    assert latch.get("confirm-roof").status is ConfirmationStatus.CANCELLED


def test_confirmed_authorization_is_cleared_when_executed_bundle_changes() -> None:
    original_actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", original_actions))
    latch.resolve("yes", current_user_turn=3, source_turn_id="turn-3")

    authorization = latch.authorized_confirmation(
        goal_ids=["open-roof"],
        ordered_actions=[call("open_close_sunroof", percentage=100)],
        current_user_turn=3,
    )

    assert authorization is None
    assert latch.active == []
    assert latch.confirmations[0].status is ConfirmationStatus.CANCELLED


@pytest.mark.parametrize(
    ("response", "expected_kind", "expected_status"),
    [
        ("no", LatchResolutionKind.REJECTED, ConfirmationStatus.REJECTED),
        ("cancel that", LatchResolutionKind.CANCELLED, ConfirmationStatus.CANCELLED),
    ],
)
def test_no_or_cancel_clears_every_active_latch(
    response: str,
    expected_kind: LatchResolutionKind,
    expected_status: ConfirmationStatus,
) -> None:
    latch = ConfirmationLatch()
    latch.arm(scope("one", [call("set_one", value=1)]), confirmation_id="one")
    latch.arm(scope("two", [call("set_two", value=2)]), confirmation_id="two")

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
    )

    assert resolution.kind is expected_kind
    assert resolution.affected_confirmation_ids == ["one", "two"]
    assert latch.active == []
    assert {item.status for item in latch.confirmations} == {expected_status}


def test_expired_or_pre_request_turn_cannot_confirm() -> None:
    late_latch = ConfirmationLatch()
    late_latch.arm(
        scope("open-roof", [call("open_close_sunroof", percentage=50)], expires=3),
        confirmation_id="late",
    )
    late = late_latch.resolve(
        "yes",
        current_user_turn=4,
        source_turn_id="turn-4",
    )

    early_latch = ConfirmationLatch()
    early_latch.arm(
        scope(
            "open-roof",
            [call("open_close_sunroof", percentage=50)],
            requested=3,
            expires=4,
        ),
        confirmation_id="early",
    )
    early = early_latch.resolve(
        "yes",
        current_user_turn=2,
        source_turn_id="turn-2",
        confirmation_id="early",
        goal_ids=["open-roof"],
        ordered_actions=[call("open_close_sunroof", percentage=50)],
    )

    assert late.kind is LatchResolutionKind.EXPIRED
    assert late_latch.get("late").status is ConfirmationStatus.EXPIRED
    assert early.kind is LatchResolutionKind.INVALIDATED
    assert early_latch.get("early").status is ConfirmationStatus.CANCELLED


def test_non_expressive_or_qualified_yes_does_not_bind() -> None:
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", [call("open_close_sunroof", percentage=50)]))

    resolution = latch.resolve(
        "maybe yes, after I decide",
        current_user_turn=3,
        source_turn_id="turn-3",
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


@pytest.mark.parametrize(
    "response",
    [
        "Yes, if the weather is clear.",
        "Yes, unless it is raining.",
        "Okay, maybe later.",
        "Sure, after I check with my passenger.",
        "Yes, before you do that, tell me the weather.",
        "Yes, when I get there.",
        "Yes, once I get there.",
        "Yes, tomorrow.",
        "Yes, as soon as the rain stops.",
        "Yes, contingent on clear weather.",
        "Yes, depending on clear weather.",
        "Yes, subject to my passenger agreeing.",
    ],
)
def test_conditional_or_deferred_affirmation_does_not_bind(response: str) -> None:
    actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-roof"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_even_if_restatement_remains_an_unconditional_confirmation() -> None:
    actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", actions))

    resolution = latch.resolve(
        "Yes, I still want to open it, even if it is raining.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-roof"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


@pytest.mark.parametrize(
    "response",
    [
        "Yes, navigate via Before Avenue.",
        "Yes, navigate via After Street.",
        "Yes, navigate via Once Road.",
        "Yes, as I said before, go ahead.",
        "Yes, once and for all, go ahead.",
    ],
)
def test_before_after_once_in_names_or_idioms_do_not_block_confirmation(
    response: str,
) -> None:
    actions = [call("set_new_navigation", route_ids=["route-alpha"])]
    latch = ConfirmationLatch()
    latch.arm(scope("navigation", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["navigation"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


@pytest.mark.parametrize(
    "response",
    [
        (
            "Yes. If the weather needs to be clear first, then open it to "
            "50 percent, but wait."
        ),
        (
            "Yes. If my passenger needs to agree first, then open it to "
            "50 percent."
        ),
    ],
)
def test_arbitrary_prerequisite_clause_cannot_promote_a_qualified_yes(
    response: str,
) -> None:
    actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-roof"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


@pytest.mark.parametrize(
    "response",
    [
        "Yes, use the 123.45 kilometer route via Q417.",
        "Yes, the route that takes 1 hour and 17 minutes via route 417.",
    ],
)
def test_frozen_route_description_numbers_may_be_repeated(response: str) -> None:
    actions = [call("set_new_navigation", route_ids=["rll_ori_nor_123456"])]
    latch = ConfirmationLatch()
    latch.arm(scope("resume-route", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["resume-route"],
        ordered_actions=actions,
        confirmation_prompt=(
            "The route via Q417 is 123.45 kilometers and takes 1 hour and "
            "17 minutes. Shall I restart this exact route?"
        ),
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


def test_unpresented_route_description_number_does_not_bind() -> None:
    actions = [call("set_new_navigation", route_ids=["route-1"])]
    latch = ConfirmationLatch()
    latch.arm(scope("resume-route", actions))

    resolution = latch.resolve(
        "Yes, use the 200 kilometer route.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["resume-route"],
        ordered_actions=actions,
        confirmation_prompt="The route is 199.22 kilometers. Shall I restart it?",
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED


@pytest.mark.parametrize(
    ("response", "actions"),
    [
        (
            "Yes, set it to twenty two degrees.",
            [call("set_climate_temperature", temperature=22, seat_zone="ALL")],
        ),
        (
            "Yes, open the sunshade one hundred percent.",
            [call("open_close_sunshade", percentage=100)],
        ),
    ],
)
def test_compound_number_words_match_exact_action_values(
    response: str, actions: list[OfficialToolCall]
) -> None:
    latch = ConfirmationLatch()
    latch.arm(scope("numeric-action", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["numeric-action"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


def test_context_number_cannot_replace_a_numeric_action_target() -> None:
    actions = [
        call("set_climate_temperature", temperature=22, seat_zone="ALL_ZONES")
    ]
    latch = ConfirmationLatch()
    latch.arm(scope("temperature", actions))

    resolution = latch.resolve(
        "Yes, set the temperature to 25 degrees.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["temperature"],
        ordered_actions=actions,
        confirmation_prompt=(
            "The route is 25 kilometers long. Set the temperature to 22 degrees?"
        ),
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


@pytest.mark.parametrize(
    ("frozen_target", "response"),
    [
        ("DRIVER", "Yes, open the passenger window."),
        ("PASSENGER", "Yes, open the driver window."),
        ("DRIVER_REAR", "Yes, open the passenger rear window."),
        ("ALL", "Yes, open the driver window."),
    ],
)
def test_changed_control_target_cannot_confirm_frozen_action(
    frozen_target: str, response: str
) -> None:
    actions = [call("open_close_window", window=frozen_target, percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("window", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["window"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_matching_control_target_can_confirm_frozen_action() -> None:
    actions = [call("open_close_window", window="PASSENGER_REAR", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("window", actions))

    resolution = latch.resolve(
        "Yes, open the passenger rear window to 50 percent.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["window"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


def test_person_reference_is_not_mistaken_for_a_control_target() -> None:
    actions = [call("open_close_window", window="DRIVER", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("window", actions))

    resolution = latch.resolve(
        "Yes, my passenger agrees, go ahead.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["window"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


def test_changed_alphanumeric_route_number_cannot_confirm_frozen_action() -> None:
    actions = [call("set_new_navigation", route_ids=["rll_ori_nor_123456"])]
    latch = ConfirmationLatch()
    latch.arm(scope("resume-route", actions))

    resolution = latch.resolve(
        "Yes, use the route via Q418.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["resume-route"],
        ordered_actions=actions,
        confirmation_prompt="Use the route via Q417?",
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_number_from_sibling_action_cannot_change_another_target() -> None:
    actions = [
        call("open_close_sunshade", percentage=100),
        call("open_close_sunroof", percentage=50),
    ]
    latch = ConfirmationLatch()
    latch.arm(scope("roof-bundle", actions))

    resolution = latch.resolve(
        "Yes, open the sunroof to 100 percent.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["roof-bundle"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_number_from_sibling_action_cannot_use_an_unbound_target_alias() -> None:
    actions = [
        call("open_close_sunshade", percentage=100),
        call("open_close_sunroof", percentage=50),
    ]
    latch = ConfirmationLatch()
    latch.arm(scope("roof-bundle", actions))

    resolution = latch.resolve(
        "Yes, open the roof to 100 percent.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["roof-bundle"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


@pytest.mark.parametrize(
    ("actions", "response"),
    [
        (
            [
                call(
                    "set_climate_temperature",
                    temperature=22,
                    seat_zone="DRIVER",
                ),
                call(
                    "set_climate_temperature",
                    temperature=24,
                    seat_zone="PASSENGER",
                ),
            ],
            (
                "Yes, set the driver temperature to 22 degrees and the "
                "passenger temperature to 24 degrees."
            ),
        ),
        (
            [
                call("open_close_window", window="DRIVER", percentage=25),
                call("open_close_window", window="PASSENGER", percentage=50),
            ],
            (
                "Yes, open the driver window to 25 percent and the passenger "
                "window to 50 percent."
            ),
        ),
    ],
)
def test_distinct_action_anchors_bind_each_exact_numeric_restatement(
    actions: list[OfficialToolCall],
    response: str,
) -> None:
    latch = ConfirmationLatch()
    latch.arm(scope("multi-action", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["multi-action"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED


def test_distinct_action_anchors_reject_swapped_numeric_targets() -> None:
    actions = [
        call(
            "set_climate_temperature",
            temperature=22,
            seat_zone="DRIVER",
        ),
        call(
            "set_climate_temperature",
            temperature=24,
            seat_zone="PASSENGER",
        ),
    ]
    latch = ConfirmationLatch()
    latch.arm(scope("multi-action", actions))

    resolution = latch.resolve(
        (
            "Yes, set the driver temperature to 24 degrees and the passenger "
            "temperature to 22 degrees."
        ),
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["multi-action"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_context_number_without_read_only_route_context_does_not_bind() -> None:
    actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("roof", actions))

    resolution = latch.resolve(
        "Yes, 25.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["roof"],
        ordered_actions=actions,
        confirmation_prompt="The outside temperature is 25 degrees.",
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_prompt_number_cannot_modify_an_action_without_numeric_arguments() -> None:
    actions = [call("set_new_navigation", route_ids=["route-alpha"])]
    latch = ConfirmationLatch()
    latch.arm(scope("navigation", actions))

    resolution = latch.resolve(
        "Yes, open it to 25 percent.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["navigation"],
        ordered_actions=actions,
        confirmation_prompt=(
            "The route is 25 kilometers long. Shall I start this exact route?"
        ),
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_affirmative_with_changed_parameters_does_not_bind() -> None:
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", [call("open_close_sunroof", percentage=50)]))

    resolution = latch.resolve(
        "Yes, but set it to 80 instead.",
        current_user_turn=3,
        source_turn_id="turn-3",
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_affirmative_numeric_restatement_cannot_change_frozen_value() -> None:
    actions = [call("open_close_sunroof", percentage=50)]
    latch = ConfirmationLatch()
    latch.arm(scope("open-roof", actions))

    resolution = latch.resolve(
        "Yes, go ahead and open it to 80%.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-roof"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


@pytest.mark.parametrize(
    "response",
    [
        "Yes, close the passenger window.",
        "Yes, close the driver window.",
        "Yes, close the sunroof.",
        "Yes, open all windows.",
        "Yes, open the front passenger window.",
    ],
)
def test_affirmative_non_numeric_scope_change_does_not_bind(response: str) -> None:
    actions = [
        call("open_close_window", window="DRIVER", percentage=50),
    ]
    latch = ConfirmationLatch()
    latch.arm(scope("open-window", actions))

    resolution = latch.resolve(
        response,
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-window"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.UNRECOGNIZED
    assert len(latch.pending) == 1


def test_affirmative_matching_non_numeric_scope_can_bind() -> None:
    actions = [
        call("open_close_window", window="DRIVER", percentage=50),
    ]
    latch = ConfirmationLatch()
    latch.arm(scope("open-window", actions))

    resolution = latch.resolve(
        "Yes, open the driver window.",
        current_user_turn=3,
        source_turn_id="turn-3",
        goal_ids=["open-window"],
        ordered_actions=actions,
    )

    assert resolution.kind is LatchResolutionKind.CONFIRMED
