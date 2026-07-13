from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


OLIVIA_HARRIS_ID = "con_1001"
OLIVIA_CHEN_ID = "con_1002"
WILLIAM_TURNER_ID = "con_2001"
RACHEL_WALKER_ID = "con_3001"
DAVID_HARRIS_ID = "con_4001"

OLIVIA_HARRIS_PHONE = "+1 202-555-0101"
RACHEL_EMAIL = "rachel.walker@example.com"
DAVID_PHONE = "+1 202-555-0401"
DAVID_EMAIL = "david.harris@example.com"

POLICY = (
    "Use only current tool evidence. Calls must use a verified displayed phone. "
    "Email tools whose description starts with REQUIRES_CONFIRMATION need a "
    "fresh explicit yes."
)


def _tool(
    name: str,
    properties: dict[str, Any],
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
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


LOOKUP_TOOL = _tool(
    "get_contact_id_by_contact_name",
    {
        "contact_first_name": {"type": "string"},
        "contact_last_name": {"type": "string"},
    },
)
INFO_TOOL = _tool(
    "get_contact_information",
    {"contact_ids": {"type": "array", "items": {"type": "string"}}},
    ["contact_ids"],
)
CALL_TOOL = _tool(
    "call_phone_by_number",
    {"phone_number": {"type": "string"}},
    ["phone_number"],
)
EMAIL_TOOL = _tool(
    "send_email",
    {
        "content_message": {"type": "string"},
        "email_addresses": {"type": "array", "items": {"type": "string"}},
    },
    ["content_message", "email_addresses"],
    description="REQUIRES_CONFIRMATION: send the exact frozen email.",
)
TOOLS = (LOOKUP_TOOL, INFO_TOOL, CALL_TOOL, EMAIL_TOOL)


RECORDS: dict[str, dict[str, Any]] = {
    OLIVIA_HARRIS_ID: {
        "id": OLIVIA_HARRIS_ID,
        "name": {"first_name": "Olivia", "last_name": "Harris"},
        "phone_number": OLIVIA_HARRIS_PHONE,
        "email": "olivia.harris@example.com",
    },
    OLIVIA_CHEN_ID: {
        "id": OLIVIA_CHEN_ID,
        "name": {"first_name": "Olivia", "last_name": "Chen"},
        "phone_number": "+1 202-555-0102",
        "email": "olivia.chen@example.com",
    },
    WILLIAM_TURNER_ID: {
        "id": WILLIAM_TURNER_ID,
        "name": {"first_name": "William", "last_name": "Turner"},
        "phone_number": "+1 202-555-0201",
        "email": "william.turner@example.com",
    },
    RACHEL_WALKER_ID: {
        "id": RACHEL_WALKER_ID,
        "name": {"first_name": "Rachel", "last_name": "Walker"},
        "phone_number": "+1 202-555-0301",
        "email": RACHEL_EMAIL,
    },
    DAVID_HARRIS_ID: {
        "id": DAVID_HARRIS_ID,
        "name": {"first_name": "David", "last_name": "Harris"},
        "phone_number": DAVID_PHONE,
        "email": DAVID_EMAIL,
    },
}


def _intent(
    semantic_operation: str | None = None,
    desired_outcome: dict[str, Any] | None = None,
) -> IntentDraft:
    goals = (
        [
            {
                "semantic_operation": semantic_operation,
                "desired_outcome": desired_outcome or {},
            }
        ]
        if semantic_operation is not None
        else []
    )
    return IntentDraft.model_validate(
        {
            "language": "en",
            "intent_kind": "information",
            "call_for_action": False,
            "goals": goals,
            "explicit_slots": desired_outcome or {},
        }
    )


class ContactClient:
    def __init__(self, *, hostile: bool = False) -> None:
        self.hostile = hostile
        self.intent_calls = 0
        self.action_calls = 0

    @staticmethod
    def _latest_user(messages: list[dict[str, Any]]) -> str:
        return next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("contact actions must remain deterministic")
        if response_model is not IntentDraft:
            raise AssertionError(f"unexpected response model: {response_model}")
        self.intent_calls += 1
        text = self._latest_user(messages)
        if "William Turner" in text:
            value = _intent(
                "find_contact_ids",
                {
                    "contact_first_name": "William",
                    "contact_last_name": "Turner",
                },
            )
        elif self.hostile and "Olivia" in text:
            value = _intent("find_contact_ids", {"contact_first_name": "Olivia"})
        else:
            value = _intent()
        return SimpleNamespace(value=value)


def _runtime(client: ContactClient | None = None) -> CARGuardOrchestrator:
    configured = client or ContactClient()
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=32),
        client_factory=lambda session: configured,
    )


def _user(
    runtime: CARGuardOrchestrator,
    context_id: str,
    message_id: str,
    text: str,
    *,
    initialize: bool = False,
):
    return runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=message_id,
            system_policy=POLICY if initialize else None,
            user_text=text,
            live_tools=TOOLS if initialize else (),
        )
    )


def _result(
    runtime: CARGuardOrchestrator,
    context_id: str,
    message_id: str,
    tool_name: str,
    content: Any,
):
    return runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=message_id,
            tool_results=({"toolName": tool_name, "content": content},),
        )
    )


def _matches(values: dict[str, str]) -> dict[str, Any]:
    return {"status": "SUCCESS", "result": {"matches": values}}


def _information(*contact_ids: str) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            contact_id: deepcopy(RECORDS[contact_id]) for contact_id in contact_ids
        },
    }


def _assert_call(outbound, tool_name: str, arguments: dict[str, Any]) -> None:
    assert outbound.tool_calls == ({"tool_name": tool_name, "arguments": arguments},)


def _run_olivia_to_ambiguity(runtime: CARGuardOrchestrator, context_id: str):
    lookup = _user(
        runtime, context_id, f"{context_id}-find", "Find Olivia.", initialize=True
    )
    _assert_call(
        lookup,
        "get_contact_id_by_contact_name",
        {"contact_first_name": "Olivia"},
    )
    return _result(
        runtime,
        context_id,
        f"{context_id}-matches",
        "get_contact_id_by_contact_name",
        _matches(
            {
                OLIVIA_HARRIS_ID: "olivia harris",
                OLIVIA_CHEN_ID: "olivia chen",
            }
        ),
    )


def _run_olivia_to_display(
    runtime: CARGuardOrchestrator,
    context_id: str,
    *,
    information: dict[str, Any] | None = None,
):
    ambiguity = _run_olivia_to_ambiguity(runtime, context_id)
    assert ambiguity.tool_calls == ()
    assert ambiguity.text is not None
    assert "Olivia Harris" in ambiguity.text and "Olivia Chen" in ambiguity.text

    info = _user(
        runtime,
        context_id,
        f"{context_id}-select",
        "Olivia Harris.",
    )
    _assert_call(
        info,
        "get_contact_information",
        {"contact_ids": [OLIVIA_HARRIS_ID]},
    )
    return _result(
        runtime,
        context_id,
        f"{context_id}-information",
        "get_contact_information",
        information or _information(OLIVIA_HARRIS_ID),
    )


@pytest.mark.parametrize("call_text", ["Call her.", "Call Olivia Harris."])
def test_base19_bare_lookup_exact_selection_and_verified_terminal_call(
    call_text: str,
) -> None:
    client = ContactClient()
    runtime = _runtime(client)
    context_id = f"base19-{len(call_text)}"

    displayed = _run_olivia_to_display(runtime, context_id)
    assert displayed.text is not None
    assert OLIVIA_HARRIS_PHONE in displayed.text

    called = _user(runtime, context_id, f"{context_id}-call", call_text)
    _assert_call(
        called,
        "call_phone_by_number",
        {"phone_number": OLIVIA_HARRIS_PHONE},
    )
    assert called.terminal
    assert runtime.sessions.get(context_id) is None
    assert runtime.sessions.get_tombstone(context_id) is not None
    replay = _user(runtime, context_id, f"{context_id}-call", call_text)
    assert replay.tool_calls == () and replay.terminal
    assert client.action_calls == 0


def test_base21_find_contact_ids_and_provider_argument_names_are_adopted() -> None:
    client = ContactClient()
    runtime = _runtime(client)
    context_id = "base21-find-contact-ids"

    lookup = _user(
        runtime,
        context_id,
        "base21-user",
        "Hey there! Can you help me find William Turner's contact info?",
        initialize=True,
    )
    _assert_call(
        lookup,
        "get_contact_id_by_contact_name",
        {"contact_first_name": "William", "contact_last_name": "Turner"},
    )
    info = _result(
        runtime,
        context_id,
        "base21-matches",
        "get_contact_id_by_contact_name",
        _matches({WILLIAM_TURNER_ID: "william turner"}),
    )
    _assert_call(
        info,
        "get_contact_information",
        {"contact_ids": [WILLIAM_TURNER_ID]},
    )
    displayed = _result(
        runtime,
        context_id,
        "base21-information",
        "get_contact_information",
        _information(WILLIAM_TURNER_ID),
    )
    assert displayed.tool_calls == ()
    assert displayed.text is not None
    for value in ("William Turner", "+1 202-555-0201", "william.turner@example.com"):
        assert value in displayed.text
    assert client.intent_calls == 1
    assert client.action_calls == 0


def _run_base59_to_recipient(runtime: CARGuardOrchestrator, context_id: str):
    lookup = _user(
        runtime,
        context_id,
        f"{context_id}-recipient",
        "Look up Rachel Walker's contact information. I want to send her an email.",
        initialize=True,
    )
    _assert_call(
        lookup,
        "get_contact_id_by_contact_name",
        {"contact_first_name": "Rachel", "contact_last_name": "Walker"},
    )
    info = _result(
        runtime,
        context_id,
        f"{context_id}-recipient-matches",
        "get_contact_id_by_contact_name",
        _matches({RACHEL_WALKER_ID: "rachel walker"}),
    )
    _assert_call(
        info,
        "get_contact_information",
        {"contact_ids": [RACHEL_WALKER_ID]},
    )
    return _result(
        runtime,
        context_id,
        f"{context_id}-recipient-info",
        "get_contact_information",
        _information(RACHEL_WALKER_ID),
    )


def _run_base59_to_preview(runtime: CARGuardOrchestrator, context_id: str):
    recipient = _run_base59_to_recipient(runtime, context_id)
    assert recipient.tool_calls == ()
    assert recipient.text is not None
    assert "Rachel Walker" in recipient.text and RACHEL_EMAIL in recipient.text

    source_lookup = _user(
        runtime,
        context_id,
        f"{context_id}-source",
        "Share David Harris's contact details.",
    )
    _assert_call(
        source_lookup,
        "get_contact_id_by_contact_name",
        {"contact_first_name": "David", "contact_last_name": "Harris"},
    )
    source_info = _result(
        runtime,
        context_id,
        f"{context_id}-source-matches",
        "get_contact_id_by_contact_name",
        _matches({DAVID_HARRIS_ID: "david harris"}),
    )
    _assert_call(
        source_info,
        "get_contact_information",
        {"contact_ids": [DAVID_HARRIS_ID]},
    )
    return _result(
        runtime,
        context_id,
        f"{context_id}-source-info",
        "get_contact_information",
        _information(DAVID_HARRIS_ID),
    )


def test_base59_two_grounded_records_freeze_then_send_exact_email() -> None:
    client = ContactClient()
    runtime = _runtime(client)
    context_id = "base59-positive"

    preview = _run_base59_to_preview(runtime, context_id)
    assert preview.tool_calls == ()
    assert preview.text is not None
    for value in (RACHEL_EMAIL, "David Harris", DAVID_PHONE, DAVID_EMAIL):
        assert value in preview.text

    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    bundle = session.execution_bundle
    expected = {
        "email_addresses": [RACHEL_EMAIL],
        "content_message": (
            "Hi Rachel,\n\n"
            "I wanted to share David Harris's contact information with you:\n\n"
            "Name: David Harris\n"
            f"Phone: {DAVID_PHONE}\n"
            f"Email: {DAVID_EMAIL}\n\n"
            "Best regards"
        ),
    }
    assert bundle.calls[0].arguments == expected

    sent = _user(runtime, context_id, f"{context_id}-yes", "Yes.")
    _assert_call(sent, "send_email", expected)
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "text",
    [
        "Don't find Olivia.",
        "Alice said Find Olivia.",
        "Hypothetically, Find Olivia.",
        "Find 'Olivia'.",
        "Find Olivia and call her.",
        "Don't find William Turner's contact info.",
        "William said, find William Turner's contact info.",
        "Hypothetically, find William Turner's contact info.",
    ],
)
def test_contact_lookup_rejects_negative_reported_hypothetical_and_composite_text(
    text: str,
) -> None:
    client = ContactClient(hostile=True)
    runtime = _runtime(client)
    outbound = _user(
        runtime,
        f"reject-{abs(hash(text))}",
        f"reject-{abs(hash(text))}-user",
        text,
        initialize=True,
    )
    assert outbound.tool_calls == ()
    session = runtime.sessions.get(f"reject-{abs(hash(text))}")
    assert session is not None and session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    "text",
    ["Call William Turner.", "Don't call her."],
)
def test_displayed_contact_call_rejects_wrong_or_negated_target(text: str) -> None:
    runtime = _runtime()
    context_id = f"call-reject-{len(text)}"
    _run_olivia_to_display(runtime, context_id)

    blocked = _user(runtime, context_id, f"{context_id}-blocked", text)
    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()


def test_displayed_contact_call_rejects_stale_display() -> None:
    runtime = _runtime()
    context_id = "call-stale"
    _run_olivia_to_display(runtime, context_id)
    interposed = _user(runtime, context_id, f"{context_id}-thanks", "Thanks.")
    assert interposed.tool_calls == ()

    blocked = _user(runtime, context_id, f"{context_id}-call", "Call her.")
    assert blocked.tool_calls == ()


@pytest.mark.parametrize(
    "bad_information",
    [
        {
            "status": "SUCCESS",
            "result": {
                OLIVIA_HARRIS_ID: {
                    **RECORDS[OLIVIA_HARRIS_ID],
                    "phone_number": "not-a-phone",
                }
            },
        },
        {
            "status": "SUCCESS",
            "result": {
                OLIVIA_HARRIS_ID: {
                    **RECORDS[OLIVIA_HARRIS_ID],
                    "name": {"first_name": "Olivia", "last_name": "Chen"},
                }
            },
        },
    ],
)
def test_malformed_contact_information_never_becomes_callable(
    bad_information: dict[str, Any],
) -> None:
    runtime = _runtime()
    context_id = f"bad-contact-{len(str(bad_information))}"
    rejected = _run_olivia_to_display(runtime, context_id, information=bad_information)
    assert rejected.tool_calls == ()

    call = _user(runtime, context_id, f"{context_id}-call", "Call her.")
    assert call.tool_calls == ()


def test_base59_rejects_stale_or_negated_source_request() -> None:
    runtime = _runtime()
    context_id = "base59-stale-source"
    _run_base59_to_recipient(runtime, context_id)
    _user(runtime, context_id, f"{context_id}-thanks", "Thanks.")

    stale = _user(
        runtime,
        context_id,
        f"{context_id}-stale",
        "Share David Harris's contact details.",
    )
    assert stale.tool_calls == ()

    runtime = _runtime()
    context_id = "base59-negated-source"
    _run_base59_to_recipient(runtime, context_id)
    negated = _user(
        runtime,
        context_id,
        f"{context_id}-no",
        "Don't share David Harris's contact details.",
    )
    assert negated.tool_calls == ()


@pytest.mark.parametrize("malformation", ["wrong-match", "extra-info", "bad-phone"])
def test_base59_malformed_source_evidence_never_freezes_email(
    malformation: str,
) -> None:
    runtime = _runtime()
    context_id = f"base59-{malformation}"
    _run_base59_to_recipient(runtime, context_id)
    lookup = _user(
        runtime,
        context_id,
        f"{context_id}-source",
        "Share David Harris's contact details.",
    )
    assert lookup.tool_calls

    matches = (
        _matches({DAVID_HARRIS_ID: "david turner"})
        if malformation == "wrong-match"
        else _matches({DAVID_HARRIS_ID: "david harris"})
    )
    info = _result(
        runtime,
        context_id,
        f"{context_id}-matches",
        "get_contact_id_by_contact_name",
        matches,
    )
    if malformation == "wrong-match":
        assert info.tool_calls == ()
    else:
        assert info.tool_calls
        records = _information(DAVID_HARRIS_ID)
        if malformation == "extra-info":
            records["result"][RACHEL_WALKER_ID] = deepcopy(RECORDS[RACHEL_WALKER_ID])
        else:
            records["result"][DAVID_HARRIS_ID]["phone_number"] = "bad"
        preview = _result(
            runtime,
            context_id,
            f"{context_id}-info",
            "get_contact_information",
            records,
        )
        assert preview.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is None
    assert session.budget.attempted_sets == set()


def test_base59_rejected_confirmation_cannot_be_reused() -> None:
    runtime = _runtime()
    context_id = "base59-stale-confirmation"
    _run_base59_to_preview(runtime, context_id)

    rejected = _user(runtime, context_id, f"{context_id}-no", "No.")
    assert rejected.tool_calls == ()
    late_yes = _user(runtime, context_id, f"{context_id}-late-yes", "Yes.")
    assert late_yes.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
