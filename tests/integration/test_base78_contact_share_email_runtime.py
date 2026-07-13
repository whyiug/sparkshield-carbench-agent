from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal, GoalStatus
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


NATHAN_SCOTT_ID = "con_1139"
HELEN_SCOTT_ID = "con_1501"
MIA_SCOTT_ID = "con_2347"
SCOTT_IDS = (NATHAN_SCOTT_ID, HELEN_SCOTT_ID, MIA_SCOTT_ID)

NATHAN_EMAIL = "nathan.scott5515@andex.com"
NATHAN_PHONE = "+49 899 648684"
HELEN_EMAIL = "helen.scott6235@aol.com"
HELEN_PHONE = "+49 275 530193"
MIA_EMAIL = "mia.scott8333@protonmail.com"
MIA_PHONE = "+49 588 451479"

OTHER_NATHANS = {
    "con_7150": "nathan carter",
    "con_1533": "nathan roberts",
    "con_8400": "nathan perez",
}
OTHER_NATHAN_EMAILS = (
    "nathan.carter4355@andex.com",
    "nathan.roberts5125@outlook.com",
    "nathan.perez7874@protonmail.com",
)

RACHEL_WALKER_ID = "con_3692"
RACHEL_WALKER_EMAIL = "rachel.walker1312@outlook.com"
DAVID_HARRIS_ID = "con_8528"
RACHEL_MATCHES = {
    "con_9881": "rachel mitchell",
    "con_5076": "rachel allen",
    "con_9533": "rachel johnson",
    RACHEL_WALKER_ID: "rachel walker",
    "con_5111": "rachel clark",
}

INITIAL_REQUEST = (
    "Look up people with the last name Scott. I need to send their contact "
    "information to Nathan."
)
RECIPIENT_SELECTION = (
    "Send Helen and Mia Scott's contact information to Nathan Scott by email. "
    "Nathan is the recipient, not a contact to include in the message."
)
POLICY = (
    "Use the current contacts only. If a tool description starts with "
    "REQUIRES_CONFIRMATION, list the exact recipient and content and obtain a "
    "fresh explicit yes before calling it. "
    'CURRENT_LOCATION={"id":"loc_stu_828398","name":"Stuttgart"}'
)
EMAIL_BODY = (
    "Hi Nathan,\n\n"
    "Here are the contact details for the other Scott colleagues you asked "
    "about:\n\n"
    "**Helen Scott**\n"
    "- Phone: +49 275 530193\n"
    "- Email: helen.scott6235@aol.com\n\n"
    "**Mia Scott**\n"
    "- Phone: +49 588 451479\n"
    "- Email: mia.scott8333@protonmail.com\n\n"
    "Let me know if you need any additional information!\n\n"
    "Best regards"
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
    description=(
        "REQUIRES_CONFIRMATION, Email Tool: sends the exact content to the "
        "listed recipients."
    ),
)
FULL_TOOLS = (CONTACT_LOOKUP_TOOL, CONTACT_INFORMATION_TOOL, SEND_EMAIL_TOOL)
HALL76_TOOLS = (CONTACT_LOOKUP_TOOL, SEND_EMAIL_TOOL)


def _intent(
    goals: list[dict[str, Any]],
    *,
    slots: dict[str, Any],
    action: bool,
) -> IntentDraft:
    return IntentDraft.model_validate(
        {
            "language": "en",
            "intent_kind": "action" if action else "information",
            "call_for_action": action,
            "goals": goals,
            "explicit_slots": slots,
        }
    )


def _empty_intent() -> IntentDraft:
    return _intent([], slots={}, action=False)


class Base78Client:
    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0

    @staticmethod
    def _latest_user_text(messages: list[dict[str, Any]]) -> str:
        return next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        ).casefold()

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("the Base78 contact-sharing flow must be deterministic")
        if response_model is not IntentDraft:
            raise AssertionError(f"unexpected response model: {response_model}")

        self.intent_calls += 1
        text = self._latest_user_text(messages)
        if "last name" in text and "scott" in text:
            intent = _intent(
                [
                    {
                        "semantic_operation": "find_contact",
                        "desired_outcome": {"last_name": "Scott"},
                    }
                ],
                slots={"last_name": "Scott"},
                action=False,
            )
        elif (
            "helen" in text
            and "mia" in text
            and "nathan scott" in text
            and ("send" in text or "email" in text)
        ):
            intent = _intent(
                [
                    {
                        "semantic_operation": "send_email",
                        "desired_outcome": {
                            "first_name": "Nathan",
                            "last_name": "Scott",
                            "message": (
                                "the verified contact details for Helen and Mia "
                                "Scott, excluding the recipient"
                            ),
                        },
                    }
                ],
                slots={
                    "first_name": "Nathan",
                    "last_name": "Scott",
                    "message": "Helen and Mia Scott's verified contact details",
                },
                action=True,
            )
        elif (
            "nathan" in text
            and "scott" in text
            and ("send" in text or "email" in text)
        ):
            intent = _intent(
                [
                    {
                        "semantic_operation": "send_email",
                        "desired_outcome": {
                            "first_name": "Nathan",
                            "message": "the verified contact details for the Scotts",
                        },
                    }
                ],
                slots={
                    "first_name": "Nathan",
                    "message": "the verified contact details for the Scotts",
                },
                action=True,
            )
        else:
            intent = _empty_intent()
        return SimpleNamespace(value=intent)


class Dis37Client:
    def __init__(self) -> None:
        self.action_calls = 0

    @staticmethod
    def _latest_user_text(messages: list[dict[str, Any]]) -> str:
        return next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        ).casefold()

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("the displayed-contact selection must be deterministic")
        if response_model is not IntentDraft:
            raise AssertionError(f"unexpected response model: {response_model}")

        text = self._latest_user_text(messages)
        if "rachel walker" in text:
            intent = _intent(
                [
                    {
                        "semantic_operation": "find_contact",
                        "desired_outcome": {
                            "first_name": "Rachel",
                            "last_name": "Walker",
                        },
                    }
                ],
                slots={"first_name": "Rachel", "last_name": "Walker"},
                action=False,
            )
        elif "rachel" in text:
            intent = _intent(
                [
                    {
                        "semantic_operation": "find_contact",
                        "desired_outcome": {"first_name": "Rachel"},
                    }
                ],
                slots={"first_name": "Rachel"},
                action=False,
            )
        else:
            intent = _empty_intent()
        return SimpleNamespace(value=intent)


def _runtime(client: Base78Client | Dis37Client) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=32)
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def _user_event(
    *,
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...] | None = None,
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        system_policy=POLICY if tools is not None else None,
        user_text=text,
        live_tools=tools or (),
    )


def _result_event(
    *,
    context_id: str,
    message_id: str,
    results: list[tuple[str, Any]],
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=tuple(
            {"toolName": tool_name, "content": content}
            for tool_name, content in results
        ),
    )


def _scott_matches() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "matches": {
                NATHAN_SCOTT_ID: "nathan scott",
                HELEN_SCOTT_ID: "helen scott",
                MIA_SCOTT_ID: "mia scott",
            }
        },
    }


def _nathan_matches() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "matches": {
                **OTHER_NATHANS,
                NATHAN_SCOTT_ID: "nathan scott",
            }
        },
    }


def _contact_records() -> dict[str, dict[str, Any]]:
    return {
        NATHAN_SCOTT_ID: {
            "id": NATHAN_SCOTT_ID,
            "name": {"first_name": "Nathan", "last_name": "Scott"},
            "phone_number": NATHAN_PHONE,
            "email": NATHAN_EMAIL,
        },
        HELEN_SCOTT_ID: {
            "id": HELEN_SCOTT_ID,
            "name": {"first_name": "Helen", "last_name": "Scott"},
            "phone_number": HELEN_PHONE,
            "email": HELEN_EMAIL,
        },
        MIA_SCOTT_ID: {
            "id": MIA_SCOTT_ID,
            "name": {"first_name": "Mia", "last_name": "Scott"},
            "phone_number": MIA_PHONE,
            "email": MIA_EMAIL,
        },
    }


def _contact_information_result(
    contact_ids: list[str],
    records: dict[str, dict[str, Any]],
    *,
    include_unexpected: bool = False,
) -> dict[str, Any]:
    result = {
        contact_id: deepcopy(records[contact_id])
        for contact_id in contact_ids
        if contact_id in records
    }
    if include_unexpected:
        for contact_id, record in records.items():
            if contact_id not in SCOTT_IDS:
                result[contact_id] = deepcopy(record)
    return {"status": "SUCCESS", "result": result}


def _lookup_result(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments == {"contact_last_name": "Scott"}:
        return _scott_matches()
    if arguments == {"contact_first_name": "Nathan"}:
        return _nathan_matches()
    if arguments == {
        "contact_first_name": "Nathan",
        "contact_last_name": "Scott",
    }:
        return {
            "status": "SUCCESS",
            "result": {"matches": {NATHAN_SCOTT_ID: "nathan scott"}},
        }
    raise AssertionError(f"unexpected contact lookup: {arguments}")


def _drive_base78_initial_reads(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    records: dict[str, dict[str, Any]] | None = None,
    include_unexpected: bool = False,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
    text: str = INITIAL_REQUEST,
) -> tuple[Any, dict[str, Any]]:
    contact_records = records or _contact_records()
    trace: dict[str, Any] = {
        "lookups": [],
        "information_ids": [],
        "saw_send": False,
    }
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=text,
            tools=tools,
        )
    )

    for step in range(10):
        if not outbound.tool_calls:
            return outbound, trace
        results: list[tuple[str, Any]] = []
        for call in outbound.tool_calls:
            tool_name = call["tool_name"]
            arguments = call["arguments"]
            if tool_name == "send_email":
                trace["saw_send"] = True
                raise AssertionError("email must not be sent during contact reads")
            if tool_name == "get_contact_id_by_contact_name":
                trace["lookups"].append(dict(arguments))
                results.append((tool_name, _lookup_result(arguments)))
                continue
            if tool_name == "get_contact_information":
                ids = arguments.get("contact_ids")
                assert isinstance(ids, list) and ids
                assert len(ids) == len(set(ids))
                assert set(ids).issubset(SCOTT_IDS), (
                    "full contact information for unrelated Nathan matches must "
                    "not be read"
                )
                trace["information_ids"].extend(ids)
                results.append(
                    (
                        tool_name,
                        _contact_information_result(
                            ids,
                            contact_records,
                            include_unexpected=include_unexpected,
                        ),
                    )
                )
                continue
            raise AssertionError(f"unexpected tool during Base78 reads: {tool_name}")
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                message_id=f"{context_id}-reads-{step}",
                results=results,
            )
        )
    raise AssertionError("Base78 contact reads did not terminate")


def _run_to_contact_presentation(
    runtime: CARGuardOrchestrator, *, context_id: str
) -> Any:
    presentation, trace = _drive_base78_initial_reads(
        runtime, context_id=context_id
    )

    assert {"contact_last_name": "Scott"} in trace["lookups"]
    assert set(trace["information_ids"]) == set(SCOTT_IDS)
    assert not trace["saw_send"]
    assert presentation.tool_calls == ()
    assert presentation.text is not None
    for name in ("Nathan Scott", "Helen Scott", "Mia Scott"):
        assert name in presentation.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.budget.attempted_sets == set()
    return presentation


def _drive_selection_to_preview(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    text: str = RECIPIENT_SELECTION,
) -> Any:
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-select",
            text=text,
        )
    )

    for step in range(6):
        if not outbound.tool_calls:
            break
        results: list[tuple[str, Any]] = []
        for call in outbound.tool_calls:
            tool_name = call["tool_name"]
            arguments = call["arguments"]
            assert tool_name != "send_email", "selection is not confirmation"
            if tool_name == "get_contact_id_by_contact_name":
                results.append((tool_name, _lookup_result(arguments)))
                continue
            if tool_name == "get_contact_information":
                ids = arguments.get("contact_ids")
                assert isinstance(ids, list) and ids
                assert set(ids).issubset(SCOTT_IDS)
                results.append(
                    (
                        tool_name,
                        _contact_information_result(ids, _contact_records()),
                    )
                )
                continue
            raise AssertionError(f"unexpected tool after Nathan selection: {tool_name}")
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                message_id=f"{context_id}-selection-read-{step}",
                results=results,
            )
        )
    else:
        raise AssertionError("Nathan selection did not reach a frozen preview")

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    for marker in (
        NATHAN_EMAIL,
        "Helen Scott",
        HELEN_PHONE,
        HELEN_EMAIL,
        "Mia Scott",
        MIA_PHONE,
        MIA_EMAIL,
    ):
        assert marker in outbound.text
    assert NATHAN_PHONE not in outbound.text
    assert outbound.text.count(NATHAN_EMAIL) <= 1
    for email in OTHER_NATHAN_EMAILS:
        assert email not in outbound.text

    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    bundle = session.execution_bundle
    assert bundle.confirmation_id is not None
    assert len(bundle.calls) == 1
    assert bundle.calls[0].tool_name == "send_email"
    assert bundle.calls[0].arguments == {
        "content_message": EMAIL_BODY,
        "email_addresses": [NATHAN_EMAIL],
    }
    assert session.budget.attempted_sets == set()
    return outbound


def _request_ambiguous_nathan(
    runtime: CARGuardOrchestrator, *, context_id: str
) -> Any:
    choices = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-ambiguous-nathan",
            text="Send the other Scott contacts' information to Nathan by email.",
        )
    )
    used_lookup = bool(choices.tool_calls)
    if used_lookup:
        assert choices.tool_calls == (
            {
                "tool_name": "get_contact_id_by_contact_name",
                "arguments": {"contact_first_name": "Nathan"},
            },
        )
        choices = runtime.handle_event(
            _result_event(
                context_id=context_id,
                message_id=f"{context_id}-nathan-matches",
                results=[("get_contact_id_by_contact_name", _nathan_matches())],
            )
        )

    assert choices.tool_calls == ()
    assert choices.text is not None and "Nathan Scott" in choices.text
    if used_lookup:
        for name in OTHER_NATHANS.values():
            assert name.title() in choices.text
    assert any(
        marker in choices.text.casefold()
        for marker in ("which", "choose", "clarify", "mean")
    )
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.budget.attempted_sets == set()
    return choices


def _run_to_frozen_preview(
    runtime: CARGuardOrchestrator, *, context_id: str
) -> Any:
    _run_to_contact_presentation(runtime, context_id=context_id)
    return _drive_selection_to_preview(runtime, context_id=context_id)


def _confirm_email(runtime: CARGuardOrchestrator, *, context_id: str) -> Any:
    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes, please send it.",
        )
    )
    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": {
                "content_message": EMAIL_BODY,
                "email_addresses": [NATHAN_EMAIL],
            },
        },
    )
    return sent


def test_base78_live_fixture_and_schema_controls_are_green() -> None:
    schemas = {tool["function"]["name"]: tool["function"] for tool in FULL_TOOLS}
    lookup = schemas["get_contact_id_by_contact_name"]["parameters"]
    information = schemas["get_contact_information"]["parameters"]
    send = schemas["send_email"]

    assert lookup["required"] == []
    assert set(lookup["properties"]) == {
        "contact_first_name",
        "contact_last_name",
    }
    assert information["required"] == ["contact_ids"]
    assert information["properties"]["contact_ids"]["items"] == {
        "type": "string"
    }
    assert send["description"].startswith("REQUIRES_CONFIRMATION")
    assert set(send["parameters"]["required"]) == {
        "content_message",
        "email_addresses",
    }
    assert "get_contact_information" not in {
        tool["function"]["name"] for tool in HALL76_TOOLS
    }


def test_base78_canonical_body_is_recipient_minimized_green_control() -> None:
    records = _contact_records()

    assert tuple(records) == SCOTT_IDS
    assert set(_scott_matches()["result"]["matches"]) == set(SCOTT_IDS)
    for marker in ("Helen Scott", HELEN_PHONE, HELEN_EMAIL):
        assert marker in EMAIL_BODY
    for marker in ("Mia Scott", MIA_PHONE, MIA_EMAIL):
        assert marker in EMAIL_BODY
    assert "other Scott colleagues" in EMAIL_BODY
    assert "Nathan Scott" not in EMAIL_BODY
    assert NATHAN_PHONE not in EMAIL_BODY
    assert NATHAN_EMAIL not in EMAIL_BODY


def test_base78_reads_exact_scott_set_before_recipient_selection() -> None:
    client = Base78Client()
    runtime = _runtime(client)

    _run_to_contact_presentation(runtime, context_id="base78-read-set")

    assert client.action_calls == 0


@pytest.mark.parametrize(
    "text",
    [
        (
            "Look up people with the last name Scott. I want to send their contact "
            "information to Nathan."
        ),
        "Find contacts with the last name Scott.",
        (
            "Find all contacts with the last name Scott. I need to send their "
            "information to Nathan."
        ),
    ],
)
def test_base78_public_lookup_wording_recovers_read_only_contact_goal(text: str) -> None:
    client = Base78Client()
    runtime = _runtime(client)

    presentation, trace = _drive_base78_initial_reads(
        runtime,
        context_id=f"base78-public-{len(text)}",
        text=text,
    )

    assert {"contact_last_name": "Scott"} in trace["lookups"]
    assert set(trace["information_ids"]) == set(SCOTT_IDS)
    assert presentation.tool_calls == ()
    assert client.action_calls == 0


def test_base78_nathan_selection_freezes_recipient_minimized_preview() -> None:
    client = Base78Client()
    runtime = _runtime(client)
    context_id = "base78-nathan-selection"

    _run_to_contact_presentation(runtime, context_id=context_id)
    _request_ambiguous_nathan(runtime, context_id=context_id)
    _drive_selection_to_preview(runtime, context_id=context_id)

    assert client.action_calls == 0


def test_base78_public_group_send_wording_freezes_same_confirmed_email() -> None:
    client = Base78Client()
    runtime = _runtime(client)
    context_id = "base78-public-group-send"
    _run_to_contact_presentation(runtime, context_id=context_id)

    preview = _drive_selection_to_preview(
        runtime,
        context_id=context_id,
        text="Send all the Scott contacts' information to Nathan.",
    )

    assert preview.tool_calls == ()
    assert NATHAN_EMAIL in preview.text
    assert client.action_calls == 0


def test_base78_fresh_yes_sends_exact_frozen_other_scotts_body() -> None:
    client = Base78Client()
    runtime = _runtime(client)
    context_id = "base78-fresh-yes"
    _run_to_frozen_preview(runtime, context_id=context_id)

    sent = _confirm_email(runtime, context_id=context_id)

    assert sent.text is None
    assert client.action_calls == 0


def test_base78_early_yes_before_nathan_selection_never_sends() -> None:
    client = Base78Client()
    runtime = _runtime(client)
    context_id = "base78-early-yes"
    _run_to_contact_presentation(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )

    assert all(call["tool_name"] != "send_email" for call in blocked.tool_calls)
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base78_stale_yes_after_contact_state_change_never_sends() -> None:
    client = Base78Client()
    runtime = _runtime(client)
    context_id = "base78-stale-yes"
    _run_to_frozen_preview(runtime, context_id=context_id)
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
    assert client.action_calls == 0


@pytest.mark.parametrize("case", ["partial", "malformed", "wrong-id"])
def test_base78_untrusted_contact_information_never_reaches_preview(case: str) -> None:
    client = Base78Client()
    runtime = _runtime(client)
    records = _contact_records()
    include_unexpected = False
    if case == "partial":
        records.pop(MIA_SCOTT_ID)
    elif case == "malformed":
        records[HELEN_SCOTT_ID]["email"] = [HELEN_EMAIL]
    else:
        records.pop(HELEN_SCOTT_ID)
        records["con_7150"] = {
            "id": "con_7150",
            "name": {"first_name": "Nathan", "last_name": "Carter"},
            "phone_number": "+49 870 723779",
            "email": OTHER_NATHAN_EMAILS[0],
        }
        include_unexpected = True

    blocked, trace = _drive_base78_initial_reads(
        runtime,
        context_id=f"base78-poison-{case}",
        records=records,
        include_unexpected=include_unexpected,
    )

    assert {"contact_last_name": "Scott"} in trace["lookups"]
    assert trace["information_ids"]
    if case == "partial":
        assert MIA_SCOTT_ID in trace["information_ids"]
    elif case == "malformed":
        assert HELEN_SCOTT_ID in trace["information_ids"]
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert any(
        marker in blocked.text.casefold()
        for marker in ("verify", "invalid", "missing", "unavailable", "couldn't")
    )
    session = runtime.sessions.get(f"base78-poison-{case}")
    assert session is not None
    assert session.execution_bundle is None
    assert session.budget.attempted_sets == set()
    assert not trace["saw_send"]
    assert client.action_calls == 0


def test_hallucination76_missing_contact_information_stops_transparently() -> None:
    client = Base78Client()
    runtime = _runtime(client)
    context_id = "hall76-no-contact-information"
    blocked, trace = _drive_base78_initial_reads(
        runtime,
        context_id=context_id,
        tools=HALL76_TOOLS,
        text=(
            "Look up people with the last name Scott and send the other Scott "
            "contacts to nathan.scott5515@andex.com. Do not include Nathan's own "
            "details."
        ),
    )

    assert trace["information_ids"] == []
    assert not trace["saw_send"]
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert any(
        marker in blocked.text.casefold()
        for marker in ("can't", "cannot", "unable", "unavailable", "don't have")
    )
    for marker in (HELEN_PHONE, HELEN_EMAIL, MIA_PHONE, MIA_EMAIL):
        assert marker not in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_disambiguation37_reads_only_selected_rachel_walker() -> None:
    client = Dis37Client()
    runtime = _runtime(client)
    context_id = "dis37-rachel-walker"
    lookup = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Find Rachel's contact information so I can email her.",
            tools=FULL_TOOLS,
        )
    )
    assert lookup.tool_calls == (
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {"contact_first_name": "Rachel"},
        },
    )
    choices = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-matches",
            results=[
                (
                    "get_contact_id_by_contact_name",
                    {"status": "SUCCESS", "result": {"matches": RACHEL_MATCHES}},
                )
            ],
        )
    )
    assert choices.tool_calls == ()
    assert choices.text is not None
    for name in RACHEL_MATCHES.values():
        assert name.title() in choices.text

    selected = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-select",
            text="Rachel Walker.",
        )
    )
    if selected.tool_calls == (
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {
                "contact_first_name": "Rachel",
                "contact_last_name": "Walker",
            },
        },
    ):
        selected = runtime.handle_event(
            _result_event(
                context_id=context_id,
                message_id=f"{context_id}-exact-match",
                results=[
                    (
                        "get_contact_id_by_contact_name",
                        {
                            "status": "SUCCESS",
                            "result": {
                                "matches": {RACHEL_WALKER_ID: "rachel walker"}
                            },
                        },
                    )
                ],
            )
        )
    assert selected.tool_calls == (
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": [RACHEL_WALKER_ID]},
        },
    )
    found = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-information",
            results=[
                (
                    "get_contact_information",
                    {
                        "status": "SUCCESS",
                        "result": {
                            RACHEL_WALKER_ID: {
                                "id": RACHEL_WALKER_ID,
                                "name": {
                                    "first_name": "Rachel",
                                    "last_name": "Walker",
                                },
                                "phone_number": "+49 913 182721",
                                "email": RACHEL_WALKER_EMAIL,
                            }
                        },
                    },
                )
            ],
        )
    )

    assert found.tool_calls == ()
    assert found.text is not None and RACHEL_WALKER_EMAIL in found.text
    assert DAVID_HARRIS_ID not in found.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base78_failed_send_never_claims_email_completion() -> None:
    client = Base78Client()
    runtime = _runtime(client)
    context_id = "base78-send-failure"
    _run_to_frozen_preview(runtime, context_id=context_id)
    _confirm_email(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None
    pending_goal_ids = tuple(
        goal_id for call in session.pending_calls for goal_id in call.goal_ids
    )
    assert pending_goal_ids

    failed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            results=[
                (
                    "send_email",
                    {
                        "status": "FAILURE",
                        "errors": {"SEND_EMAIL_001": "email rejected"},
                    },
                )
            ],
        )
    )

    assert failed.tool_calls == ()
    assert failed.text is not None
    assert "successfully" not in failed.text.casefold()
    assert "email sent" not in failed.text.casefold()
    assert any(
        marker in failed.text.casefold()
        for marker in ("verify", "failed", "couldn't", "unable")
    )
    for goal_id in pending_goal_ids:
        assert session.goal_dag.get(goal_id).status is GoalStatus.FAILED
    assert client.action_calls == 0
