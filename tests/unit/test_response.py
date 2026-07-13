from collections.abc import Sequence

import pytest

from track_1_agent_under_test.car_guard.domain import GateOutcome
from track_1_agent_under_test.car_guard.response import (
    ResponseComposer,
    TTSGuard,
    TTSViolation,
)


def test_specific_capability_limitation_names_goal_and_non_execution() -> None:
    text = ResponseComposer().limitation(
        goal="set the fan speed",
        reason="I can't select a fan level with the controls currently available",
        outcome=GateOutcome.UNSUPPORTED_PARAMETER,
    )

    assert "set the fan speed" in text
    assert "haven't performed" in text
    assert "set_fan_speed" not in text


def test_evidence_limitation_does_not_ask_user_for_system_fact() -> None:
    text = ResponseComposer().limitation(
        goal="turn on the air conditioning",
        reason="I couldn't determine the rear window position",
        outcome=GateOutcome.UNAVAILABLE_EVIDENCE,
    )

    assert "safely" in text
    assert "?" not in text
    assert "haven't performed" in text


def test_clarification_lists_all_remaining_choices() -> None:
    text = ResponseComposer().clarification(
        goal="the route", options=["the fastest route", "the shortest route"]
    )

    assert "fastest" in text and "shortest" in text
    assert text.endswith("?")


def test_confirmation_describes_complete_bundle() -> None:
    text = ResponseComposer().confirmation(
        action_bundle=["turn off the fog lights", "turn on the high beams"]
    )

    assert "fog lights" in text and "high beams" in text
    assert text.endswith("?")


def test_bundle_completion_states_verified_actions_and_values() -> None:
    text = ResponseComposer().bundle_completion(
        action_bundle=[
            {
                "tool_name": "open_close_sunshade",
                "arguments": {"percentage": 100.0},
            },
            {
                "tool_name": "open_close_sunroof",
                "arguments": {"percentage": 50.0},
            },
        ],
        policy_operations=["open_sunshade_for_sunroof", "set_sunroof_position"],
    )

    assert text == (
        "Done. I opened the sunshade to 100 percent and opened the sunroof "
        "to 50 percent."
    )
    assert TTSGuard().violations(text) == []


def test_climate_completion_names_celsius_unit() -> None:
    text = ResponseComposer().bundle_completion(
        action_bundle=[
            {
                "tool_name": "set_climate_temperature",
                "arguments": {"temperature": 22, "seat_zone": "ALL_ZONES"},
            }
        ],
        policy_operations=["set_climate_temperature"],
    )

    assert text == "Done. I set the all zones temperature to 22 degrees Celsius."
    assert TTSGuard().violations(text) == []


def test_partial_completion_distinguishes_blocked_goal() -> None:
    text = ResponseComposer().partial_completion(
        completed=["opening the trunk"],
        blocked_goal="start navigation",
        reason="route options were unavailable",
    )

    assert "completed opening the trunk" in text
    assert "start navigation" in text
    assert "not performed" in text


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (
            "en",
            "I found Meson El Segoviano or Casa Pepe in Madrid. Which point of "
            "interest would you like directions to?",
        ),
        (
            "de",
            "Ich habe in Madrid Meson El Segoviano oder Casa Pepe gefunden. "
            "Zu welchem Ort moechtest du navigieren?",
        ),
        (
            "zh",
            "我在Madrid找到了Meson El Segoviano或Casa Pepe。您想导航到哪一个地点？",
        ),
    ],
)
def test_poi_search_results_name_only_contract_is_unchanged(
    language: str,
    expected: str,
) -> None:
    text = ResponseComposer().poi_search_results(
        location="Madrid",
        poi_names=("Meson El Segoviano", "Casa Pepe"),
        language=language,
    )

    assert text == expected


def test_poi_search_results_can_present_evidenced_opening_hours() -> None:
    text = ResponseComposer().poi_search_results(
        location="Madrid",
        poi_names=("Meson El Segoviano", "Casa Pepe"),
        poi_opening_hours=("09:00h - 21:00h", "10:00h - 19:00h"),
    )

    assert text == (
        "I found Meson El Segoviano, with opening hours 09:00h - 21:00h or "
        "Casa Pepe, with opening hours 10:00h - 19:00h in Madrid. Which point "
        "of interest would you like directions to?"
    )
    assert TTSGuard().violations(text) == []


@pytest.mark.parametrize(
    "opening_hours",
    [
        ("", "10:00h - 19:00h"),
        (" 09:00h - 21:00h", "10:00h - 19:00h"),
        ("09:00h - 21:00h ", "10:00h - 19:00h"),
        ("09:00h\n21:00h", "10:00h - 19:00h"),
        ("09:00h\x0021:00h", "10:00h - 19:00h"),
        ("09:00h\u202821:00h", "10:00h - 19:00h"),
        ("`09:00h - 21:00h`", "10:00h - 19:00h"),
        ("x" * 121, "10:00h - 19:00h"),
    ],
)
def test_poi_search_results_rejects_unsafe_opening_hours(
    opening_hours: tuple[str, str],
) -> None:
    with pytest.raises(ValueError, match="safe non-empty"):
        ResponseComposer().poi_search_results(
            location="Madrid",
            poi_names=("Meson El Segoviano", "Casa Pepe"),
            poi_opening_hours=opening_hours,
        )


@pytest.mark.parametrize(
    "opening_hours",
    [
        ("09:00h - 21:00h",),
        ("09:00h - 21:00h", "10:00h - 19:00h", "11:00h - 18:00h"),
        "09:00h - 21:00h",
    ],
)
def test_poi_search_results_requires_one_detail_per_name(
    opening_hours: Sequence[str],
) -> None:
    with pytest.raises(ValueError, match="align with every name"):
        ResponseComposer().poi_search_results(
            location="Madrid",
            poi_names=("Meson El Segoviano", "Casa Pepe"),
            poi_opening_hours=opening_hours,
        )


def test_gate_limitation_hides_internal_semantic_identifier() -> None:
    text = ResponseComposer().gate_limitation(
        goal="set_fan_speed",
        outcome=GateOutcome.UNSUPPORTED_PARAMETER,
    )

    assert "set the fan speed" in text
    assert "set_fan_speed" not in text


def test_prerequisite_limitation_names_missing_action_without_identifiers() -> None:
    text = ResponseComposer().prerequisite_limitation(
        goal="set_sunroof_position",
        prerequisite="open_sunshade_for_sunroof",
    )

    assert "can't open the sunshade" in text
    assert "can't safely move the sunroof" in text
    assert "haven't performed either action" in text
    assert "open_sunshade_for_sunroof" not in text
    assert "set_sunroof_position" not in text
    assert TTSGuard().violations(text) == []


def test_chinese_response_is_voice_safe() -> None:
    text = ResponseComposer().clarification(
        goal="路线", options=["最快路线", "最短路线"], language="zh-CN"
    )

    assert "您想选哪一个" in text
    assert not TTSGuard().violations(text)


def test_navigation_summary_is_detailed_and_voice_safe() -> None:
    text = ResponseComposer().navigation_summary(
        active=True,
        waypoint_names=["Dortmund", "Cologne"],
        vias=["B572"],
        road_types=["country road"],
        aliases=["fastest", "shortest"],
        includes_toll=False,
        distance_km="88.75",
        duration_minutes=67,
        details_available=True,
    )

    assert "Dortmund" in text and "Cologne" in text
    assert "88.75 kilometers" in text
    assert "1 hour and 7 minutes" in text
    assert "from Dortmund to Cologne" in text
    assert "no toll roads" in text
    assert TTSGuard().violations(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "- choose one",
        "Use `set_fan_speed` now",
        "First line\nSecond line",
        "POL-005 requires this",
    ],
)
def test_tts_guard_rejects_internal_or_formatted_text(text: str) -> None:
    with pytest.raises(TTSViolation):
        TTSGuard().ensure(text)


def test_tts_guard_allows_sentence_number_before_period() -> None:
    text = "This will set the fan to level 2. Shall I continue?"
    assert TTSGuard().ensure(text) == text


def test_tts_guard_allows_email_without_weakening_internal_field_checks() -> None:
    guard = TTSGuard()

    assert guard.violations("Send it to grace.nelson3334@gmail.com") == []
    assert guard.violations("Send it to grace.nelson3334@gmail.com.") == []
    assert "internal_identifier" in guard.violations(
        "Send it to user@example.com.object.secret"
    )
    assert "internal_identifier" in guard.violations(
        "Send it to user@mail.example.com"
    )
    assert "internal_identifier" in guard.violations("Read object.secret")
    assert "internal_identifier" in guard.violations("Read .secret")
    assert "multiline" in guard.violations("first\nsecond")


@pytest.mark.parametrize(
    "text",
    [
        "Read object．secret",
        "Never expose tool＿call＿id",
        "Do not call set＿ambient＿lights",
        "Policy POL－005 applies",
        "Read object.\u200bsecret",
        "Policy POL\u200b-005 applies",
        "Never expose task\u200b type",
    ],
)
def test_tts_guard_rejects_unicode_internal_identifier_bypasses(text: str) -> None:
    assert TTSGuard().violations(text)
