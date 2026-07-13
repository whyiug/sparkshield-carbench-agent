from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    Evidence,
    EvidenceSourceKind,
    EvidenceStatus,
)
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _CURRENT_DAY_MEETING_ATOMIC_POSITIVE_RESPONSES,
    _CURRENT_DAY_MEETING_ATOMIC_REJECTION_RE,
    _attendee_meeting_email_request,
    _attendee_meeting_weather_email_request,
    _conditional_named_meeting_email_request,
    _named_late_email_followup,
    _named_meeting_status_request,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft


MEETING_TOPIC = "Team Sync"
ALICE_ID = "con_1011"
BOB_ID = "con_2022"
ALICE_EMAIL = "alice.smith@example.com"
BOB_EMAIL = "bob.jones@example.com"
POLICY = (
    "Use current-day calendar entries only. A send_email tool whose description "
    "starts with REQUIRES_CONFIRMATION requires a fresh explicit yes before use. "
    'CURRENT_LOCATION={"id":"loc_office_100","name":"Main Office"}. '
    'CURRENT_DATETIME={"year":2025,"month":1,"day":10,"hour":9,"minute":15}.'
)
POLICY_WITH_24H_FORMAT = (
    POLICY
    + " 002:The metric system is used and the format of datetime should be in "
    "24h format."
)


def _tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    *,
    description: str = "",
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


CALENDAR_TOOL = _tool(
    "get_entries_from_calendar",
    {
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
        "day": {"type": "integer", "minimum": 1, "maximum": 31},
    },
    ["month", "day"],
)
CONTACT_LOOKUP_TOOL = _tool(
    "get_contact_id_by_contact_name",
    {
        "contact_first_name": {"type": "string"},
        "contact_last_name": {"type": "string"},
    },
)
CONTACT_INFORMATION_TOOL = _tool(
    "get_contact_information",
    {"contact_ids": {"type": "array", "items": {"type": "string"}}},
    ["contact_ids"],
)
SEND_EMAIL_TOOL = _tool(
    "send_email",
    {
        "content_message": {"type": "string"},
        "email_addresses": {"type": "array", "items": {"type": "string"}},
    },
    ["content_message", "email_addresses"],
    description="REQUIRES_CONFIRMATION: sends the exact email to every recipient.",
)
LOCATION_TOOL = _tool(
    "get_location_id_by_location_name",
    {"location": {"type": "string"}},
    ["location"],
)
WEATHER_TOOL = _tool(
    "get_weather",
    {
        "location_or_poi_id": {"type": "string"},
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
        "day": {"type": "integer", "minimum": 1, "maximum": 31},
        "time_hour_24hformat": {"type": "integer", "minimum": 0, "maximum": 23},
    },
    ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
)
FULL_TOOLS = (
    CALENDAR_TOOL,
    CONTACT_LOOKUP_TOOL,
    CONTACT_INFORMATION_TOOL,
    SEND_EMAIL_TOOL,
)
WEATHER_EMAIL_TOOLS = (*FULL_TOOLS, LOCATION_TOOL, WEATHER_TOOL)
MEETING_LOCATION_ID = "loc_conference_3003"


class NoLLMClient:
    def generate(self, **_: Any) -> Any:
        raise AssertionError("the current-day meeting workflow must be deterministic")


class EmptyCalendarIntentClient:
    def generate(self, *, response_model: Any, **_: Any) -> Any:
        if response_model is IntentDraft:
            return SimpleNamespace(
                value=IntentDraft.model_validate(
                    {
                        "language": "en",
                        "intent_kind": "conversation",
                        "call_for_action": False,
                        "goals": [],
                    }
                )
            )
        raise AssertionError("calendar continuation must be deterministic")


def _runtime() -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=32)
    client = NoLLMClient()
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def _calendar_runtime() -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=32)
    client = EmptyCalendarIntentClient()
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def _user_event(
    *,
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...] | None = None,
    system_policy: str | None = None,
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        system_policy=(system_policy or POLICY) if tools is not None else None,
        user_text=text,
        live_tools=tools or (),
    )


def _result_event(
    *, context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def _calendar_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "date": {"year": 2025, "month": 1, "day": 10},
            "meetings": [
                {
                    "start": {"hour": "09", "minute": "00"},
                    "duration": "30min",
                    "location": "Conference Room A",
                    "attendees": [ALICE_ID, BOB_ID],
                    "topic": MEETING_TOPIC,
                }
            ],
        },
    }


def _records() -> dict[str, dict[str, Any]]:
    return {
        ALICE_ID: {
            "id": ALICE_ID,
            "name": {"first_name": "Alice", "last_name": "Smith"},
            "phone_number": "+1 555 0101",
            "email": ALICE_EMAIL,
        },
        BOB_ID: {
            "id": BOB_ID,
            "name": {"first_name": "Bob", "last_name": "Jones"},
            "phone_number": "+1 555 0202",
            "email": BOB_EMAIL,
        },
    }


def _tool_result(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "get_entries_from_calendar":
        assert arguments == {"month": 1, "day": 10}
        return _calendar_result()
    if tool_name == "get_contact_id_by_contact_name":
        assert arguments == {
            "contact_first_name": "Alice",
            "contact_last_name": "Smith",
        }
        return {
            "status": "SUCCESS",
            "result": {"matches": {ALICE_ID: "alice smith"}},
        }
    if tool_name == "get_contact_information":
        contact_ids = arguments.get("contact_ids")
        assert isinstance(contact_ids, list)
        records = _records()
        return {
            "status": "SUCCESS",
            "result": {
                contact_id: deepcopy(records[contact_id]) for contact_id in contact_ids
            },
        }
    if tool_name == "get_location_id_by_location_name":
        assert arguments == {"location": "Conference Room A"}
        return {"status": "SUCCESS", "result": {"id": MEETING_LOCATION_ID}}
    if tool_name == "get_weather":
        assert arguments == {
            "location_or_poi_id": MEETING_LOCATION_ID,
            "month": 1,
            "day": 10,
            "time_hour_24hformat": 9,
        }
        return {
            "status": "SUCCESS",
            "result": {
                "current_slot": {
                    "start_time": "09:00",
                    "end_time": "12:00",
                    "temperature_c": 7,
                    "wind_speed_kph": 18,
                    "humidity_percent": 91,
                    "condition": "cloudy_and_rain",
                },
                "next_slot": None,
            },
        }
    raise AssertionError(f"unexpected read tool: {tool_name}")


def _drive_reads(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    outbound: Any,
) -> Any:
    for step in range(8):
        if not outbound.tool_calls:
            return outbound
        assert len(outbound.tool_calls) == 1
        call = outbound.tool_calls[0]
        tool_name = call["tool_name"]
        assert tool_name != "send_email", "email must wait for fresh confirmation"
        session = runtime.sessions.get(context_id)
        assert session is not None
        read_index = len(session.successful_read_results)
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                message_id=f"{context_id}-read-{read_index}-{step}",
                tool_name=tool_name,
                content=_tool_result(tool_name, call["arguments"]),
            )
        )
    raise AssertionError("meeting reads did not reach a stable response")


def _attendee_preview(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    system_policy: str | None = None,
) -> Any:
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Could you send an email reminder to the attendees of my Team Sync "
                "meeting?"
            ),
            tools=FULL_TOOLS,
            system_policy=system_policy,
        )
    )
    content_prompt = _drive_reads(runtime, context_id=context_id, outbound=outbound)
    assert content_prompt.tool_calls == ()
    assert content_prompt.text is not None
    assert "friendly reminder tone or meeting details" in content_prompt.text
    preview = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-content",
            text=(
                "Could you include a friendly reminder about the meeting today at "
                "9:00 AM in Conference Room A?"
            ),
        )
    )
    assert preview.tool_calls == ()
    assert preview.text is not None
    assert MEETING_TOPIC in preview.text
    assert ALICE_EMAIL in preview.text
    assert BOB_EMAIL in preview.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    bundle = session.execution_bundle
    assert bundle.confirmation_id is not None
    assert bundle.calls[0].arguments["email_addresses"] == [ALICE_EMAIL, BOB_EMAIL]
    assert session.budget.attempted_sets == set()
    return preview


def _select_from_current_day_calendar(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...] = WEATHER_EMAIL_TOOLS,
) -> Any:
    calendar = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar",
            text="Can you show me my meetings for today?",
            tools=tools,
            system_policy=POLICY_WITH_24H_FORMAT,
        )
    )
    summary = _drive_reads(runtime, context_id=context_id, outbound=calendar)
    assert summary.text == (
        "I found 1 calendar entry for today: Team Sync at 09:00 in "
        "Conference Room A."
    )
    selected = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-selection",
            text=(
                "I'm interested in the Team Sync meeting at 9:00 AM in "
                "Conference Room A."
            ),
        )
    )
    assert selected.tool_calls == ()
    assert selected.text == (
        "I selected the Team Sync meeting at 09:00 in Conference Room A. "
        "I will use this exact meeting for your next request."
    )
    return selected


def test_calendar_selection_context_weather_email_end_to_end() -> None:
    runtime = _calendar_runtime()
    context_id = "calendar-selection-weather-email"
    _select_from_current_day_calendar(runtime, context_id=context_id)

    request = (
        "Can you check the weather for Conference Room A at that time, and then "
        "send an email to the attendees about the weather conditions and how it "
        "might affect their travel and the meeting?"
    )
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-request",
            text=request,
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Conference Room A"},
        },
    )
    preview = _drive_reads(runtime, context_id=context_id, outbound=outbound)
    assert preview.text is not None
    assert "I prepared the exact Team Sync meeting email" in preview.text
    assert "today at 09:00 in Conference Room A" in preview.text
    assert "Shall I send this exact email?" in preview.text

    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    reads = [result.tool_name for result in session.successful_read_results]
    assert reads == [
        "get_entries_from_calendar",
        "get_location_id_by_location_name",
        "get_contact_information",
        "get_weather",
    ]
    goal = session.intent.goals[0]
    assert goal.desired_outcome["weather_email_request"] == request
    assert goal.desired_outcome["contextual_attendee_weather_email"] is True
    assert session.execution_bundle is not None
    arguments = session.execution_bundle.calls[0].arguments
    assert session.budget.attempted_sets == set()

    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )
    assert sent.tool_calls == (
        {"tool_name": "send_email", "arguments": arguments},
    )


@pytest.mark.parametrize(
    "delivery_intro",
    ["I'd like to", "can you", "could you please", "will you please"],
)
def test_calendar_selection_accepts_polite_sentence_weather_email_handoff(
    delivery_intro: str,
) -> None:
    runtime = _calendar_runtime()
    context_id = "calendar-selection-polite-weather-email"
    _select_from_current_day_calendar(runtime, context_id=context_id)

    request = (
        "Can you check the weather for Conference Room A at that time? And then, "
        f"{delivery_intro} send an email to the attendees about the weather conditions "
        "and how it might affect their travel and the meeting."
    )
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-request",
            text=request,
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Conference Room A"},
        },
    )
    preview = _drive_reads(runtime, context_id=context_id, outbound=outbound)
    assert preview.text is not None and "Shall I send this exact email?" in preview.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert [result.tool_name for result in session.successful_read_results] == [
        "get_entries_from_calendar",
        "get_location_id_by_location_name",
        "get_contact_information",
        "get_weather",
    ]


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Can you check the weather for the Team Sync meeting in Conference "
            "Room A at 9:00 AM, and then send an email to all attendees about "
            "the weather and how it might affect their travel and the meeting?"
        ),
        (
            "Okay, for the Team Sync meeting in Conference Room A at 9:00 AM, "
            "please check the weather. After that, send an email to all the "
            "attendees about the weather conditions and any potential impacts "
            "on their travel and the meeting."
        ),
        (
            "Can you check the weather for Conference Room A at that time? Then, "
            "I would like to send an email to the attendees about the weather "
            "conditions and how it might affect their travel and the meeting."
        ),
        (
            "Can you check the weather for Conference Room A at that time? And then, "
            "can you please send an email to the attendees about the weather "
            "conditions and how it might affect their travel and the meeting."
        ),
    ],
)
def test_context_weather_email_accepts_meeting_first_shapes(utterance: str) -> None:
    meeting = {
        "topic": MEETING_TOPIC,
        "start_hour": 9,
        "start_minute": 0,
        "location": "Conference Room A",
    }
    assert CARGuardOrchestrator._contextual_meeting_weather_email_shape(utterance)
    assert CARGuardOrchestrator._contextual_meeting_weather_email_matches(
        utterance,
        meeting=meeting,
        candidates=[meeting],
    )


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Could you look up the weather for the meeting time and location, and "
            "then send an email to the attendees about the weather and possible "
            "impacts on their travel and the meeting?"
        ),
        (
            "For the Atlas Planning meeting, can you check the weather for the "
            "specific time and location of that same meeting? After that, please "
            "email all attendees about the weather conditions and how it might "
            "affect their travel and the meeting."
        ),
        (
            "For the Atlas Planning meeting, please look up the forecast at the "
            "specific location and time of the same meeting. Then send an email "
            "to the attendees of that same meeting about the forecast and potential "
            "impacts on their travel and the meeting."
        ),
        (
            "Please check the weather for the specific time and location of the "
            "selected meeting, then email the attendees about the weather and "
            "possible impacts on their trip and the meeting."
        ),
        (
            "Could you look up the weather for the meeting time and location, and "
            "then email the attendees, please, about the weather and possible "
            "impacts on their travel and the meeting?"
        ),
        (
            "Could you look up the weather for the meeting time and location, and "
            "then send the attendees an email about the weather and possible "
            "impacts on their travel and the meeting?"
        ),
    ],
)
def test_context_weather_email_accepts_selected_meeting_reference_families(
    utterance: str,
) -> None:
    meeting = {
        "topic": "Atlas Planning",
        "start_hour": 14,
        "start_minute": 30,
        "location": "North Studio",
    }
    candidates = [
        meeting,
        {
            "topic": "Budget Review",
            "start_hour": 15,
            "start_minute": 0,
            "location": "West Annex",
        },
    ]
    assert CARGuardOrchestrator._contextual_meeting_weather_email_shape(utterance)
    assert CARGuardOrchestrator._contextual_meeting_weather_email_matches(
        utterance,
        meeting=meeting,
        candidates=candidates,
    )


@pytest.mark.parametrize(
    ("utterance", "has_direct_shape"),
    [
        (
            "For the Budget Review meeting, check the weather for the specific "
            "time and location of that same meeting. Then email the attendees "
            "about the weather and possible impacts on their travel and the meeting.",
            True,
        ),
        (
            "For the Atlas Planning meeting, check the weather for North Studio "
            "at 3:15 PM. Then email the attendees about the weather and possible "
            "impacts on their travel and the meeting.",
            True,
        ),
        (
            "For the Atlas Planning meeting, check the weather for West Annex at "
            "the meeting time. Then email the attendees about the weather and "
            "possible impacts on their travel and the meeting.",
            True,
        ),
        (
            "For the Atlas Planning meeting, check the weather for the specific "
            "time and location of that same meeting. Then email the attendees and "
            "Dana Reed about the weather and possible impacts on their travel and "
            "the meeting.",
            True,
        ),
        (
            "For the Atlas Planning meeting, check the weather for the specific "
            "time and location of that same meeting. Then email the attendees about "
            "the weather and possible impacts on their travel and the meeting, then "
            "call Dana Reed.",
            True,
        ),
        (
            "For the Atlas Planning meeting, check the weather for the specific "
            "time and location of that same meeting. Then email the attendees about "
            "the weather and possible impacts on their travel and the meeting, and "
            "update the meeting.",
            True,
        ),
        (
            "Do not check the weather for the specific time and location of the "
            "selected meeting, but email the attendees about the weather and "
            "possible impacts on their travel and the meeting.",
            True,
        ),
        (
            "Hypothetically, for the Atlas Planning meeting, look up the weather "
            "for the specific time and location of that same meeting. Then email "
            "the attendees about possible weather impacts on their travel and the "
            "meeting.",
            False,
        ),
        (
            "Dana said, for the Atlas Planning meeting, look up the weather for the "
            "specific time and location of that same meeting. Then email the "
            "attendees about possible weather impacts on their travel and the meeting.",
            False,
        ),
    ],
)
def test_context_weather_email_selected_reference_guards_remain_strict(
    utterance: str,
    has_direct_shape: bool,
) -> None:
    meeting = {
        "topic": "Atlas Planning",
        "start_hour": 14,
        "start_minute": 30,
        "location": "North Studio",
    }
    candidates = [
        meeting,
        {
            "topic": "Budget Review",
            "start_hour": 15,
            "start_minute": 0,
            "location": "West Annex",
        },
    ]
    assert (
        CARGuardOrchestrator._contextual_meeting_weather_email_shape(utterance)
        is has_direct_shape
    )
    assert not CARGuardOrchestrator._contextual_meeting_weather_email_matches(
        utterance,
        meeting=meeting,
        candidates=candidates,
    )


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Can you check the weather for Other Hall at that time, and then send "
            "an email to the attendees about the weather and possible impacts on "
            "their travel and the meeting?"
        ),
        (
            "Can you check the weather for Conference Room A at 10:00 AM, and "
            "then send an email to the attendees about the weather and possible "
            "impacts on their travel and the meeting?"
        ),
        (
            "Can you check the weather for the Other Review meeting in Conference "
            "Room A at 9:00 AM, and then send an email to the attendees about the "
            "weather and possible impacts on their travel and the meeting?"
        ),
        (
            "Can you check the weather for Conference Room A at that time, and "
            "then send an email to the attendees and Alice Smith about the weather "
            "and possible impacts on their travel and the meeting?"
        ),
        (
            "Can you check the weather for Conference Room A at that time, and "
            "then send an email to the attendees about the weather and possible "
            "impacts on their travel and the meeting. Then call Alice Smith."
        ),
        (
            "Do not check the weather for Conference Room A at that time, but send "
            "an email to the attendees about the weather and possible impacts on "
            "their travel and the meeting."
        ),
    ],
)
def test_context_weather_email_rejects_mismatch_or_scope_expansion(
    utterance: str,
) -> None:
    meeting = {
        "topic": MEETING_TOPIC,
        "start_hour": 9,
        "start_minute": 0,
        "location": "Conference Room A",
    }
    candidates = [
        meeting,
        {
            "topic": "Other Review",
            "start_hour": 10,
            "start_minute": 0,
            "location": "Other Hall",
        },
    ]
    assert CARGuardOrchestrator._contextual_meeting_weather_email_shape(utterance)
    assert not CARGuardOrchestrator._contextual_meeting_weather_email_matches(
        utterance,
        meeting=meeting,
        candidates=candidates,
    )


@pytest.mark.parametrize(
    "extra_action",
    [
        "book a table",
        "read my messages",
        "share my location",
        "resume navigation",
        "play music",
        "lock the doors",
        "adjust the climate",
        "share the forecast with Alice",
        "forward the report to Alice",
        "read my inbox",
        "honk the horn",
        "find a restaurant",
        "make a reservation",
        "schedule an appointment",
    ],
)
@pytest.mark.parametrize("position", ["before", "between"])
def test_context_weather_email_fully_consumes_prefix(
    extra_action: str, position: str
) -> None:
    weather = (
        "check the weather for Conference Room A at that time"
    )
    if position == "before":
        prefix = f"Please {extra_action}, then {weather}, and then "
    else:
        prefix = f"Please {weather}, then {extra_action}, and then "
    utterance = (
        prefix
        + "send an email to the attendees about the weather and possible impacts "
        "on their travel and the meeting."
    )
    meeting = {
        "topic": MEETING_TOPIC,
        "start_hour": 9,
        "start_minute": 0,
        "location": "Conference Room A",
    }
    assert CARGuardOrchestrator._contextual_meeting_weather_email_shape(utterance)
    assert not CARGuardOrchestrator._contextual_meeting_weather_email_matches(
        utterance,
        meeting=meeting,
        candidates=[meeting],
    )


def test_calendar_meeting_selection_requires_immediate_summary() -> None:
    runtime = _calendar_runtime()
    context_id = "calendar-selection-stale"
    calendar = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar",
            text="Can you show me my meetings for today?",
            tools=WEATHER_EMAIL_TOOLS,
            system_policy=POLICY_WITH_24H_FORMAT,
        )
    )
    _drive_reads(runtime, context_id=context_id, outbound=calendar)
    unrelated = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-unrelated",
            text="Thanks for the calendar summary.",
        )
    )
    assert unrelated.tool_calls == ()

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-selection",
            text="I'm interested in the Team Sync meeting.",
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "immediately displayed" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()


def test_calendar_meeting_selection_rejects_mismatched_time() -> None:
    runtime = _calendar_runtime()
    context_id = "calendar-selection-time-mismatch"
    calendar = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar",
            text="Can you show me my meetings for today?",
            tools=WEATHER_EMAIL_TOOLS,
            system_policy=POLICY_WITH_24H_FORMAT,
        )
    )
    _drive_reads(runtime, context_id=context_id, outbound=calendar)
    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-selection",
            text=(
                "I'm interested in the Team Sync meeting at 10:00 AM in "
                "Conference Room A."
            ),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "did not perform an action" in blocked.text


def test_duplicate_calendar_topic_is_not_selected() -> None:
    runtime = _calendar_runtime()
    context_id = "calendar-selection-duplicate-topic"
    calendar = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar",
            text="Can you show me my meetings for today?",
            tools=WEATHER_EMAIL_TOOLS,
            system_policy=POLICY_WITH_24H_FORMAT,
        )
    )
    assert calendar.tool_calls[0]["tool_name"] == "get_entries_from_calendar"
    duplicate = _calendar_result()
    duplicate["result"]["meetings"].append(
        {
            "start": {"hour": "14", "minute": "00"},
            "duration": "30min",
            "location": "Other Hall",
            "attendees": [ALICE_ID],
            "topic": MEETING_TOPIC,
        }
    )
    summary = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar-result",
            tool_name="get_entries_from_calendar",
            content=duplicate,
        )
    )
    assert summary.text is not None and "2 calendar entries" in summary.text
    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-selection",
            text="I'm interested in the Team Sync meeting.",
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "did not perform an action" in blocked.text


def test_context_weather_email_missing_schema_stops_before_new_reads() -> None:
    runtime = _calendar_runtime()
    context_id = "calendar-selection-weather-missing"
    tools = (*FULL_TOOLS, LOCATION_TOOL)
    _select_from_current_day_calendar(runtime, context_id=context_id, tools=tools)
    session = runtime.sessions.get(context_id)
    assert session is not None
    read_count = len(session.successful_read_results)

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-request",
            text=(
                "Can you check the weather for Conference Room A at that time, "
                "and then send an email to the attendees about the weather and "
                "possible impacts on their travel and the meeting?"
            ),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "weather capability is unavailable" in (
        blocked.text
    )
    assert len(session.successful_read_results) == read_count
    assert session.budget.attempted_sets == set()


def test_status_then_named_late_email_uses_current_calendar_and_contact_chain() -> None:
    runtime = _runtime()
    context_id = "meeting-status-then-apology"
    status = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            text="What's the status of my 'Team Sync' meeting?",
            tools=FULL_TOOLS,
        )
    )
    status = _drive_reads(runtime, context_id=context_id, outbound=status)
    assert status.text is not None
    assert "currently in progress" in status.text

    apology = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-apology",
            text=("Send an email to Alice Smith. I need to apologize for being late."),
        )
    )
    preview = _drive_reads(runtime, context_id=context_id, outbound=apology)
    assert preview.text is not None
    assert ALICE_EMAIL in preview.text
    assert "15 minutes late" in preview.text
    assert BOB_EMAIL not in preview.text

    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes, send that exact email.",
        )
    )
    assert len(sent.tool_calls) == 1
    assert sent.tool_calls[0]["tool_name"] == "send_email"
    assert sent.tool_calls[0]["arguments"]["email_addresses"] == [ALICE_EMAIL]
    assert "15 minutes late" in sent.tool_calls[0]["arguments"]["content_message"]


def test_named_late_confirmation_previews_and_sends_exact_24h_body() -> None:
    runtime = _runtime()
    context_id = "meeting-status-apology-official-24h"
    status = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            text="What's the status of my 'Team Sync' meeting?",
            tools=FULL_TOOLS,
            system_policy=POLICY_WITH_24H_FORMAT,
        )
    )
    status = _drive_reads(runtime, context_id=context_id, outbound=status)
    assert status.text is not None
    assert "09:00" in status.text

    apology = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-apology",
            text="Send an email to Alice Smith. Tell her I'm running late and apologize.",
        )
    )
    preview = _drive_reads(runtime, context_id=context_id, outbound=apology)
    assert preview.text == (
        "I prepared the exact Team Sync meeting email for alice.smith@example.com: "
        "Hi Alice, I wanted to reach out regarding our Team Sync meeting that "
        "started at 09:00 today. I'm running about 15 minutes late and apologize "
        "for the delay. I should be there shortly. Thank you for your patience. "
        "Best regards Shall I send this exact email?"
    )
    assert "AM" not in preview.text and "PM" not in preview.text

    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    expected_body = (
        "Hi Alice, I wanted to reach out regarding our Team Sync meeting that "
        "started at 09:00 today. I'm running about 15 minutes late and apologize "
        "for the delay. I should be there shortly. Thank you for your patience. "
        "Best regards"
    )
    assert session.execution_bundle.calls[0].arguments == {
        "content_message": expected_body,
        "email_addresses": [ALICE_EMAIL],
    }

    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )
    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": {
                "content_message": expected_body,
                "email_addresses": [ALICE_EMAIL],
            },
        },
    )


def test_schedule_form_missing_email_capability_stops_before_any_read() -> None:
    runtime = _runtime()
    context_id = "meeting-schedule-no-email"
    tools = (CALENDAR_TOOL, CONTACT_LOOKUP_TOOL, CONTACT_INFORMATION_TOOL)

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Check my schedule for 'Team Sync'. If it has already started, "
                "send an email to Alice Smith to apologize for my delay."
            ),
            tools=tools,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "unavailable" in blocked.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.successful_read_results == []
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Email the attendees of my Team Sync meeting with the weather forecast "
            "and possible travel impacts."
        ),
        (
            "Email the attendees of my Team Sync meeting about the charging stop "
            "and slight delay."
        ),
        (
            "Send an email reminder to the attendees of my Team Sync meeting and "
            "Alice Smith."
        ),
    ],
)
def test_complex_attendee_email_requests_fall_through_specialized_reminder(
    utterance: str,
) -> None:
    assert _attendee_meeting_email_request(utterance) is None


def test_weather_attendee_email_has_its_own_strict_shape() -> None:
    request = (
        "Check the weather for Conference Room A at 9:00 AM and then send an "
        "email to the attendees of my Team Sync meeting about the weather and "
        "possible impacts on their travel and the meeting."
    )
    assert _attendee_meeting_weather_email_request(request) == (
        True,
        MEETING_TOPIC,
        request,
    )


def test_weather_attendee_email_accepts_public_polite_request_shape() -> None:
    request = (
        "Okay, can you check the weather for Frankfurt at 1:30 PM and then send "
        "an email to the attendees of the Risk Management meeting about the "
        "weather conditions and any potential impacts on their travel and the "
        "meeting?"
    )
    parsed = _attendee_meeting_weather_email_request(request)
    assert parsed == (True, "Risk Management", request)
    assert CARGuardOrchestrator._meeting_weather_request_matches_meeting(
        request,
        meeting={
            "topic": "Risk Management",
            "start_hour": 13,
            "start_minute": 30,
            "location": "Frankfurt",
        },
    )


def test_weather_attendee_email_allows_benign_change_and_start_time_language() -> (
    None
):
    request = (
        "Check the weather for Conference Room A at the meeting start time and "
        "then send an email to the attendees of my Team Sync meeting because the "
        "weather may change and impact travel. Include weather details."
    )
    parsed = _attendee_meeting_weather_email_request(request)
    assert parsed == (True, MEETING_TOPIC, request)


@pytest.mark.parametrize(
    "suffix",
    [
        "Then call Alice Smith.",
        "Then lock the doors.",
        "Then play music.",
        "Then unlock the doors.",
        "Then adjust the cabin temperature.",
        "Then check the tire pressure.",
        "Also navigate home.",
        "And text Bob Jones.",
        "And set the fan speed.",
        "And cc Alice Smith.",
        "And copy Alice Smith.",
        "And copying Alice.",
        "And include Alice.",
        "And include alice.",
        "And add alice as a recipient.",
        "Plus Alice Smith.",
        "Plus alice.",
        "And alice about the forecast.",
        "And alice.smith@example.com.",
    ],
)
def test_weather_attendee_email_rejects_extra_actions_and_recipients(
    suffix: str,
) -> None:
    request = (
        "Check the weather for Conference Room A at 9:00 AM and then send an "
        "email to the attendees of my Team Sync meeting about the weather and "
        f"possible impacts on their travel. {suffix}"
    )
    assert _attendee_meeting_weather_email_request(request) == (False, None, None)


@pytest.mark.parametrize(
    "inserted_clause",
    [
        "lock the doors",
        "play music",
        "check the tire pressure",
        "tell a joke",
        "unlock the car",
    ],
)
def test_weather_attendee_email_rejects_actions_inside_weather_prefix(
    inserted_clause: str,
) -> None:
    request = (
        "Check the weather for Conference Room A at 9:00 AM and "
        f"{inserted_clause} and then send an email to the attendees of my Team "
        "Sync meeting about the weather and possible impacts on their travel and "
        "the meeting."
    )
    assert _attendee_meeting_weather_email_request(request) == (False, None, None)


@pytest.mark.parametrize(
    "location",
    [
        "Conference Room A Lock Doors",
        "Conference Room A And Lock Doors",
        "Conference Room A Email Alice",
        "Conference Room A Send Email Alice",
        "Conference Room A Message Alice",
        "Conference Room A Drive Home",
        "Conference Room A Honk Horn",
        "Conference Room A Cool Cabin",
        "Conference Room A Lower Windows",
        "Conference Room A Raise Volume",
        "Frankfurt Turn On Seat Heating",
        "Frankfurt Set Seat Heating",
        "Frankfurt Adjust Fan Speed",
        "Frankfurt Open Sunroof",
        "Frankfurt Activate Cruise Control",
        "Frankfurt Disable Air Conditioning",
        "Frankfurt Change Temperature",
        "Frankfurt Turn On Headlights",
        "Frankfurt Start Climate Control",
        "Frankfurt Unlock Car",
        "Frankfurt Open Trunk",
        "Frankfurt Pause Music",
        "Frankfurt Find Charger",
        "Frankfurt Show Calendar",
        "Frankfurt Defrost Front Window",
        "Frankfurt Increase Fan Speed",
        "Frankfurt Warm Driver Seat",
    ],
)
def test_weather_attendee_email_rejects_embedded_commands_in_location_span(
    location: str,
) -> None:
    request = (
        f"Check weather for {location} at 9:00 AM and then send an email to the "
        "attendees of my Team Sync meeting about the weather and possible impacts "
        "on their travel and the meeting."
    )
    assert _attendee_meeting_weather_email_request(request) == (False, None, None)


@pytest.mark.parametrize(
    ("requested_location", "calendar_location"),
    [
        ("St. John's Hall", "St. John's Hall"),
        ("North Campus Room B", "North Campus Room B"),
        ("Maple Drive", "Maple Drive"),
        ("Lower East Side Conference Center", "Lower East Side Conference Center"),
        ("Cool Springs Cabin", "Cool Springs Cabin"),
        ("Raise Hill Auditorium", "Raise Hill Auditorium"),
        ("Email Innovation Center", "Email Innovation Center"),
        ("Open Sunroof Museum", "Open Sunroof Museum"),
        ("Cruise Control Center", "Cruise Control Center"),
        ("Seat Heating Research Lab", "Seat Heating Research Lab"),
        ("Climate Control Campus", "Climate Control Campus"),
        ("Fan Speed Arena", "Fan Speed Arena"),
    ],
)
def test_weather_attendee_email_exact_location_allows_safe_normalized_variants(
    requested_location: str,
    calendar_location: str,
) -> None:
    request = (
        f"Check weather for {requested_location} at 9:00 AM and then send an email "
        "to the attendees of my Team Sync meeting about the weather and possible "
        "impacts on their travel and the meeting."
    )
    parsed = _attendee_meeting_weather_email_request(request)
    assert parsed == (True, MEETING_TOPIC, request)
    assert CARGuardOrchestrator._meeting_weather_request_matches_meeting(
        request,
        meeting={
            "topic": MEETING_TOPIC,
            "start_hour": 9,
            "start_minute": 0,
            "location": calendar_location,
        },
        requested_location=requested_location,
    )


@pytest.mark.parametrize(
    "location",
    [
        "conference room a",
        "Frankfurt unlock car",
        "Frankfurt open trunk",
        "Frankfurt pause music",
        "Frankfurt find charger",
        "Frankfurt show calendar",
    ],
)
def test_lowercase_non_proper_weather_location_falls_through(location: str) -> None:
    request = (
        f"Check weather for {location} at 9:00 AM and then send an email to the "
        "attendees of my Team Sync meeting about the weather and possible impacts "
        "on their travel and the meeting."
    )
    assert _attendee_meeting_weather_email_request(request) is None


@pytest.mark.parametrize(
    ("requested_time", "calendar_hour", "calendar_minute", "parses", "matches"),
    [
        ("09:00 PM", 9, 0, True, False),
        ("13:30 AM", 13, 30, False, False),
        ("13:30 PM", 13, 30, False, False),
        ("0 AM", 0, 0, False, False),
        ("00 PM", 12, 0, False, False),
        ("9:00", 9, 0, True, True),
        ("9 AM", 9, 0, True, True),
        ("12 AM", 0, 0, True, True),
        ("12 PM", 12, 0, True, True),
        ("1 PM", 13, 0, True, True),
        ("9", 9, 0, True, True),
        ("the meeting start time", 9, 0, True, True),
    ],
)
def test_weather_attendee_email_uses_structured_exact_time_binding(
    requested_time: str,
    calendar_hour: int,
    calendar_minute: int,
    parses: bool,
    matches: bool,
) -> None:
    request = (
        f"Check weather for Conference Room A at {requested_time} and then send "
        "an email to the attendees of my Team Sync meeting about the weather and "
        "possible impacts on their travel and the meeting."
    )
    parsed = _attendee_meeting_weather_email_request(request)
    if parses:
        assert parsed == (True, MEETING_TOPIC, request)
    else:
        assert parsed is None
    assert (
        CARGuardOrchestrator._meeting_weather_request_matches_meeting(
            request,
            meeting={
                "topic": MEETING_TOPIC,
                "start_hour": calendar_hour,
                "start_minute": calendar_minute,
                "location": "Conference Room A",
            },
        )
        is matches
    )


def test_unapproved_but_not_unsafe_weather_location_falls_through() -> None:
    request = (
        "Check weather for conference room a, east wing at 9:00 AM and then send "
        "an email to the attendees of my Team Sync meeting about the weather and "
        "possible impacts on their travel and the meeting."
    )
    assert _attendee_meeting_weather_email_request(request) is None


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Check the weather for Conference Room A at 9:00 AM for my Team Sync "
            "meeting."
        ),
        (
            "Send an email to the attendees of my Team Sync meeting about the "
            "weather, but do not check the forecast."
        ),
        (
            "Check the weather and send an email to Alice Smith about my Team Sync "
            "meeting and possible travel impacts."
        ),
    ],
)
def test_partial_weather_attendee_requests_do_not_enter_specialized_workflow(
    utterance: str,
) -> None:
    assert _attendee_meeting_weather_email_request(utterance) is None


@pytest.mark.parametrize(
    "case",
    ["stale_slot", "humidity", "wind", "extra_field"],
)
def test_meeting_weather_snapshot_rejects_wrong_slot_and_malformed_fields(
    case: str,
) -> None:
    slot: dict[str, Any] = {
        "start_time": "09:00",
        "end_time": "12:00",
        "temperature_c": 7,
        "wind_speed_kph": 18,
        "humidity_percent": 91,
        "condition": "cloudy_and_rain",
    }
    if case == "stale_slot":
        slot.update(start_time="18:00", end_time="21:00")
    elif case == "humidity":
        slot["humidity_percent"] = 101
    elif case == "wind":
        slot["wind_speed_kph"] = -1
    else:
        slot["pressure_hpa"] = 1015
    evidence = Evidence(
        proposition="synthetic_meeting_weather",
        value={"current_slot": slot, "next_slot": None},
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.TOOL,
        source_turn_id="synthetic-weather-result",
        source_tool_call_id="synthetic-weather-call",
        confidence=1.0,
    )
    assert (
        CARGuardOrchestrator._meeting_weather_snapshot_from_evidence(
            evidence, expected_hour=9, expected_minute=30
        )
        is None
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "What's the status of my 'Team Sync' meeting, and please call Alice?",
        "What's the status of my 'Team Sync' meeting, and could you call Alice?",
        "What's the status of my 'Team Sync' meeting; please email the attendees?",
        "When does my 'Team Sync' meeting start, then please navigate home?",
    ],
)
def test_polite_compound_status_requests_fall_through_specialization(
    utterance: str,
) -> None:
    from track_1_agent_under_test.car_guard.runtime.orchestrator import (
        _named_meeting_status_request,
    )

    assert _named_meeting_status_request(utterance) is None


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Remind me to send an email reminder to the attendees of my Team Sync "
            "meeting."
        ),
        "Email me a reminder about my Team Sync meeting and its attendees.",
        "Send me an email reminder about the attendees of my Team Sync meeting.",
    ],
)
def test_self_directed_reminders_do_not_authorize_attendee_email(
    utterance: str,
) -> None:
    assert _attendee_meeting_email_request(utterance) is None


@pytest.mark.parametrize(
    "utterance",
    [
        "What's the status of my 'Team Sync' meeting and then call Alice Smith?",
        "Has my 'Team Sync' meeting started, and also email the attendees?",
        "When does my 'Team Sync' meeting start and navigate to Main Office?",
        "What's the status of my 'Team Sync' meeting? Call Alice Smith.",
    ],
)
def test_compound_meeting_status_requests_fall_through(utterance: str) -> None:
    assert _named_meeting_status_request(utterance) is None


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "If my Team Sync meeting has started, email the attendees that I am "
            "running late."
        ),
        (
            "If my Team Sync meeting has started, email Alice Smith that I am late "
            "and then call Bob Jones."
        ),
    ],
)
def test_partial_or_compound_conditional_email_falls_through(
    utterance: str,
) -> None:
    assert _conditional_named_meeting_email_request(utterance) is None


@pytest.mark.parametrize(
    "utterance",
    [
        "Send an email to Alice Smith. The package is delayed.",
        "Send an email to Alice Smith because Bob is running late.",
        "Send an email to Alice Smith because Bob apologizes for being late.",
        "Send an email to Alice Smith apologizing for the delayed train.",
    ],
)
def test_named_late_followup_requires_first_person_lateness(
    utterance: str,
) -> None:
    assert _named_late_email_followup(utterance) is None


def test_named_late_followup_preserves_first_person_apology() -> None:
    assert _named_late_email_followup(
        "Send an email to Alice Smith. I need to apologize for being late."
    ) == (True, "Alice", "Smith")


def test_attendee_reminder_sends_to_every_verified_calendar_attendee() -> None:
    runtime = _runtime()
    context_id = "meeting-attendee-reminder"
    _attendee_preview(runtime, context_id=context_id)

    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )
    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": {
                "content_message": (
                    "Hi everyone! This is a friendly reminder about our Team Sync "
                    "meeting today at 09:00 in Conference Room A. Looking forward "
                    "to seeing you all there!"
                ),
                "email_addresses": [ALICE_EMAIL, BOB_EMAIL],
            },
        },
    )


@pytest.mark.parametrize(
    ("case_id", "reply"),
    [
        ("send-call", "Yes, send it and call Alice."),
        ("lock-doors", "Yes, and lock doors."),
        ("navigate", "then navigate home"),
        ("email-bob", "also email Bob"),
        ("yes-email-bob", "Yes, also email Bob."),
    ],
)
def test_meeting_email_compound_confirmation_never_sends(
    case_id: str,
    reply: str,
) -> None:
    runtime = _runtime()
    context_id = f"meeting-compound-confirmation-{case_id}"
    _attendee_preview(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-compound",
            text=reply,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "Shall I send this exact email?" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert len(session.confirmation_latch.pending) == 1
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    ("case_id", "reply"),
    [
        ("no-do-not", "No, do not send it."),
        ("no-dont", "No, don't send it."),
        ("please-dont", "Please don't send it."),
        ("cancel-the-email", "Cancel the email."),
        ("cancel-that-email", "Cancel that email."),
        ("stop-title", "Stop."),
        ("stop-lower", "stop"),
        ("stop-the-email", "Stop the email."),
        ("stop-do-not", "Stop, do not send it."),
    ],
)
def test_meeting_email_atomic_rejection_clears_pending_without_send(
    case_id: str,
    reply: str,
) -> None:
    runtime = _runtime()
    context_id = f"meeting-atomic-rejection-{case_id}"
    _attendee_preview(runtime, context_id=context_id)

    rejected = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-reject",
            text=reply,
        )
    )

    assert rejected.tool_calls == ()
    assert rejected.text == "Okay. I did not perform the requested action."
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.confirmation_latch.pending == []
    assert session.confirmation_latch.active == []
    assert session.budget.attempted_sets == set()


def test_meeting_email_positive_allowlist_has_no_rejection_collisions() -> None:
    collisions = {
        response
        for response in _CURRENT_DAY_MEETING_ATOMIC_POSITIVE_RESPONSES
        if _CURRENT_DAY_MEETING_ATOMIC_REJECTION_RE.fullmatch(response) is not None
    }
    assert collisions == set()


@pytest.mark.parametrize(
    "reply",
    [
        "Yes, send it.",
        "Yes, send the exact email.",
        "Yes, send it now.",
        "Please send it.",
        "Send it.",
        "I confirm.",
    ],
)
def test_meeting_email_atomic_confirmation_sends_frozen_email(reply: str) -> None:
    runtime = _runtime()
    context_id = "meeting-atomic-confirmation-" + reply.casefold().replace(" ", "-")
    _attendee_preview(runtime, context_id=context_id)

    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-confirm",
            text=reply,
        )
    )

    assert len(sent.tool_calls) == 1
    assert sent.tool_calls[0]["tool_name"] == "send_email"
    assert sent.tool_calls[0]["arguments"]["email_addresses"] == [
        ALICE_EMAIL,
        BOB_EMAIL,
    ]


def test_attendee_confirmation_previews_and_sends_exact_24h_body() -> None:
    runtime = _runtime()
    context_id = "meeting-attendee-official-24h"
    preview = _attendee_preview(
        runtime,
        context_id=context_id,
        system_policy=POLICY_WITH_24H_FORMAT,
    )
    assert preview.text == (
        "I prepared the exact Team Sync meeting email for alice.smith@example.com, "
        "bob.jones@example.com: Hi everyone! This is a friendly reminder about our "
        "Team Sync meeting today at 09:00 in Conference Room A. Looking forward to "
        "seeing you all there! Shall I send this exact email?"
    )
    assert "AM" not in preview.text and "PM" not in preview.text

    expected_body = (
        "Hi everyone! This is a friendly reminder about our Team Sync meeting "
        "today at 09:00 in Conference Room A. Looking forward to seeing you all "
        "there!"
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert session.execution_bundle.calls[0].arguments == {
        "content_message": expected_body,
        "email_addresses": [ALICE_EMAIL, BOB_EMAIL],
    }

    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )
    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": {
                "content_message": expected_body,
                "email_addresses": [ALICE_EMAIL, BOB_EMAIL],
            },
        },
    )


def test_weather_attendee_email_reuses_calendar_and_sends_verified_forecast() -> (
    None
):
    runtime = _runtime()
    context_id = "meeting-weather-attendees"
    calendar = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar-user",
            text="What's the status of my 'Team Sync' meeting?",
            tools=WEATHER_EMAIL_TOOLS,
            system_policy=POLICY_WITH_24H_FORMAT,
        )
    )
    calendar_summary = _drive_reads(
        runtime, context_id=context_id, outbound=calendar
    )
    assert calendar_summary.text is not None
    assert "Team Sync" in calendar_summary.text

    request = (
        "Check the weather for Conference Room A at 9:00 AM and then send an "
        "email to the attendees of my Team Sync meeting about the weather and "
        "possible impacts on their travel and the meeting."
    )
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-weather-email",
            text=request,
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Conference Room A"},
        },
    )
    preview = _drive_reads(runtime, context_id=context_id, outbound=outbound)
    assert preview.text == (
        "I prepared the exact Team Sync meeting email for "
        "alice.smith@example.com, bob.jones@example.com: Hi team, I wanted to "
        "share a weather update for our Team Sync meeting today at 09:00 in "
        "Conference Room A. The forecast is cloudy with rain with a temperature "
        "around 7 degrees Celsius. Please bring an umbrella, and dress warmly. The "
        "conditions may affect traffic and travel, so consider leaving a little "
        "earlier. Let me know if the weather affects your ability to attend. Best "
        "regards Shall I send this exact email?"
    )
    assert "AM" not in preview.text and "PM" not in preview.text

    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    calendar_reads = [
        result
        for result in session.successful_read_results
        if result.tool_name == "get_entries_from_calendar"
    ]
    assert len(calendar_reads) == 1
    arguments = session.execution_bundle.calls[0].arguments
    assert arguments["email_addresses"] == [ALICE_EMAIL, BOB_EMAIL]
    body = arguments["content_message"]
    assert "Team Sync meeting today at 09:00 in Conference Room A" in body
    assert "cloudy with rain" in body
    assert "7 degrees Celsius" in body
    assert "bring an umbrella" in body
    assert "dress warmly" in body
    assert "traffic and travel" in body

    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )
    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": arguments,
        },
    )


def test_weather_attendee_email_missing_weather_capability_stops_before_reads() -> (
    None
):
    runtime = _runtime()
    context_id = "meeting-weather-missing-tool"
    request = (
        "Check the weather for Conference Room A at 9:00 AM and then send an "
        "email to the attendees of my Team Sync meeting about the weather and "
        "possible impacts on their travel and the meeting."
    )
    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=request,
            tools=(*FULL_TOOLS, LOCATION_TOOL),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text == (
        "I cannot complete that meeting weather email because the current weather "
        "capability is unavailable. Therefore, I did not read any calendar, "
        "contact, location, or weather data, and I did not prepare or send an "
        "email."
    )
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.successful_read_results == []
    assert session.execution_bundle is None


@pytest.mark.parametrize(
    ("missing_tool", "missing_capability"),
    [
        ("send_email", "email"),
        ("get_entries_from_calendar", "calendar"),
        ("get_contact_information", "contact"),
        ("get_location_id_by_location_name", "location"),
    ],
)
def test_weather_attendee_email_names_only_the_missing_capability(
    missing_tool: str,
    missing_capability: str,
) -> None:
    runtime = _runtime()
    context_id = f"meeting-weather-missing-{missing_capability}"
    tools = tuple(
        tool
        for tool in WEATHER_EMAIL_TOOLS
        if tool["function"]["name"] != missing_tool
    )
    request = (
        "Check the weather for Conference Room A at 9:00 AM and then send an "
        "email to the attendees of my Team Sync meeting about the weather and "
        "possible impacts on their travel and the meeting."
    )

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=request,
            tools=tools,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text == (
        "I cannot complete that meeting weather email because the current "
        f"{missing_capability} capability is unavailable. Therefore, I did not "
        "read any calendar, contact, location, or weather data, and I did not "
        "prepare or send an email."
    )
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.successful_read_results == []
    assert session.execution_bundle is None


def test_attendee_body_derivation_includes_user_instruction_provenance() -> None:
    runtime = _runtime()
    context_id = "meeting-attendee-instruction-provenance"
    _attendee_preview(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    body_id = session.derived_value_evidence_by_goal[goal.goal_id]["message"]
    body = session.evidence.evidence[body_id]

    assert body.source_kind is EvidenceSourceKind.DERIVED
    assert (
        body.derivation
        == "friendly_current_meeting_email_body_from_user_instruction_v1"
    )
    assert len(body.derived_from) == 3
    instruction = session.evidence.evidence[body.derived_from[-1]]
    assert instruction.source_kind is EvidenceSourceKind.USER
    assert instruction.source_turn_id == f"{context_id}-content"
    assert instruction.value == (
        "Could you include a friendly reminder about the meeting today at "
        "9:00 AM in Conference Room A?"
    )
    assert session.grounded_value_sources_by_goal[goal.goal_id][
        "message_instruction"
    ] == instruction.source_turn_id


def test_missing_send_email_fails_before_calendar_or_contact_reads() -> None:
    runtime = _runtime()
    context_id = "meeting-no-send-tool"
    tools = (CALENDAR_TOOL, CONTACT_LOOKUP_TOOL, CONTACT_INFORMATION_TOOL)

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=("Send a reminder email to everyone attending my Team Sync meeting."),
            tools=tools,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "unavailable" in blocked.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.successful_read_results == []
    assert session.budget.attempted_sets == set()


def test_attendee_contact_result_with_unrequested_recipient_is_rejected() -> None:
    runtime = _runtime()
    context_id = "meeting-unrequested-recipient"
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=("Send a reminder email to everyone attending my Team Sync meeting."),
            tools=FULL_TOOLS,
        )
    )
    calendar_call = outbound.tool_calls[0]
    outbound = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar",
            tool_name="get_entries_from_calendar",
            content=_calendar_result(),
        )
    )
    assert calendar_call["tool_name"] == "get_entries_from_calendar"
    assert outbound.tool_calls[0]["tool_name"] == "get_contact_information"
    records = _records()
    records["con_3033"] = {
        "id": "con_3033",
        "name": {"first_name": "Mallory", "last_name": "Reed"},
        "phone_number": "+1 555 0303",
        "email": "mallory.reed@example.com",
    }

    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-contacts",
            tool_name="get_contact_information",
            content={"status": "SUCCESS", "result": records},
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.budget.attempted_sets == set()


def test_ambiguous_named_contact_never_reads_contact_details_or_prepares_email() -> (
    None
):
    runtime = _runtime()
    context_id = "meeting-ambiguous-named-contact"
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "If my Team Sync meeting has started, email Alice Smith that I'm "
                "running late."
            ),
            tools=FULL_TOOLS,
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_entries_from_calendar"
    outbound = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar",
            tool_name="get_entries_from_calendar",
            content=_calendar_result(),
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_contact_id_by_contact_name"
    lookup_arguments = outbound.tool_calls[0]["arguments"]
    assert lookup_arguments == {
        "contact_first_name": "Alice",
        "contact_last_name": "Smith",
    }

    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-lookup",
            tool_name="get_contact_id_by_contact_name",
            content={
                "status": "SUCCESS",
                "result": {
                    "matches": {
                        ALICE_ID: "alice smith",
                        "con_4044": "alice smith",
                    }
                },
            },
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "one exact current contact" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert all(
        result.tool_name != "get_contact_information"
        for result in session.successful_read_results
    )
    assert session.execution_bundle is None
    assert session.budget.attempted_sets == set()


def test_select_phase_cancel_clears_authorization_and_cannot_be_resurrected() -> None:
    runtime = _runtime()
    context_id = "meeting-select-cancel"
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Send a reminder email to attendees for whichever meeting is next."
            ),
            tools=FULL_TOOLS,
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_entries_from_calendar"
    calendar = _calendar_result()
    calendar["result"]["meetings"].append(
        {
            "start": {"hour": "11", "minute": "30"},
            "duration": "45min",
            "location": "Project Room B",
            "attendees": [ALICE_ID],
            "topic": "Project Review",
        }
    )
    select = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-calendar",
            tool_name="get_entries_from_calendar",
            content=calendar,
        )
    )
    assert select.tool_calls == ()
    assert select.text is not None and "Which meeting" in select.text

    cancelled = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-cancel",
            text="Never mind.",
        )
    )
    assert cancelled.tool_calls == ()
    assert cancelled.text is not None and "cancelled" in cancelled.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.intent is None
    assert session.authorized_action_goal_ids == set()
    assert session.execution_bundle is None

    stale_yes = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )
    assert all(call["tool_name"] != "send_email" for call in stale_yes.tool_calls)
    assert session.budget.attempted_sets == set()


def test_content_phase_cancel_clears_authorization_and_cannot_be_resurrected() -> None:
    runtime = _runtime()
    context_id = "meeting-content-cancel"
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Could you send an email reminder to the attendees of my Team Sync "
                "meeting?"
            ),
            tools=FULL_TOOLS,
        )
    )
    prompt = _drive_reads(runtime, context_id=context_id, outbound=outbound)
    assert prompt.text is not None
    assert "friendly reminder tone" in prompt.text

    cancelled = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-cancel",
            text="Cancel that reminder.",
        )
    )
    assert cancelled.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.intent is None
    assert session.authorized_action_goal_ids == set()
    assert session.execution_bundle is None

    stale_yes = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes, send it.",
        )
    )
    assert all(call["tool_name"] != "send_email" for call in stale_yes.tool_calls)
    assert session.budget.attempted_sets == set()


def test_unavailable_email_after_status_does_not_deny_completed_calendar_read() -> (
    None
):
    runtime = _runtime()
    context_id = "meeting-status-email-unavailable"
    status = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            text="What's the status of my 'Team Sync' meeting?",
            tools=FULL_TOOLS,
        )
    )
    status = _drive_reads(runtime, context_id=context_id, outbound=status)
    assert status.text is not None and "currently in progress" in status.text

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-email",
            text="Send an email to Alice Smith. I need to apologize for being late.",
            tools=(CALENDAR_TOOL, CONTACT_LOOKUP_TOOL, CONTACT_INFORMATION_TOOL),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "did not read the calendar" not in blocked.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert any(
        result.tool_name == "get_entries_from_calendar"
        for result in session.successful_read_results
    )
    assert session.budget.attempted_sets == set()


def test_stale_yes_after_meeting_evidence_change_never_sends() -> None:
    runtime = _runtime()
    context_id = "meeting-stale-confirmation"
    _attendee_preview(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None
    session.evidence.invalidate_tool_state()

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )

    assert all(call["tool_name"] != "send_email" for call in blocked.tool_calls)
    assert session.budget.attempted_sets == set()
