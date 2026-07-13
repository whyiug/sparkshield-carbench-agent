from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from track_1_agent_under_test.car_guard.capability import LiveToolRegistry
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    Evidence,
    EvidenceSourceKind,
    EvidenceStatus,
    EvidenceStore,
    Goal,
)
from track_1_agent_under_test.car_guard.validation import (
    CriticDecision,
    CriticInput,
    CriticInputRejected,
    CriticReasonCode,
    CriticVerdict,
    GeneralCritic,
    build_critic_input,
)


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "set_fan_level",
                "description": "Set the cabin fan level.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "integer",
                            "description": "Requested level.",
                            "enum": [1, 2, 3],
                            "x-private-note": "must not reach the critic",
                        }
                    },
                    "required": ["level"],
                    "additionalProperties": False,
                    "x-private-root": {"value": True},
                },
                "x-provider-metadata": "not model input",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Read outside weather.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]


def _goals() -> list[Goal]:
    return [
        Goal(goal_id="goal-precheck", semantic_operation="read_cabin_state"),
        Goal(
            goal_id="goal-fan",
            semantic_operation="set_fan_level",
            desired_outcome={"level": 2},
            depends_on=["goal-precheck"],
        ),
    ]


def _proposal() -> DecisionProposal:
    return DecisionProposal.model_validate(
        {
            "kind": "tool_set",
            "goal_ids": ["goal-fan"],
            "tool_calls": [
                {
                    "tool_name": "set_fan_level",
                    "arguments": {"level": 2},
                    "call_id": "internal-call-7",
                    "goal_id": "goal-fan",
                    "argument_sources": {"level": "turn-user-2"},
                }
            ],
            "evidence_used": ["internal-evidence-id"],
            "policy_rules_used": ["internal-policy-id"],
            "needs_critic": True,
        }
    )


def _known_evidence(proposition: str = "cabin.mode") -> Evidence:
    return Evidence(
        proposition=proposition,
        value="manual",
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.TOOL,
        source_turn_id="turn-tool-1",
        source_tool_call_id="call-get-1",
        confidence=1.0,
    )


def _view(*, evidence: EvidenceStore | None = None) -> CriticInput:
    return build_critic_input(
        policy="Confirm safety-sensitive changes before execution.",
        conversation=[
            {
                "role": "user",
                "content": "Set the fan to level two.",
                "turn_id": "turn-user-2",
                "private_metadata": "not model input",
            },
            {
                "role": "assistant",
                "content": {"tool_calls": [{"name": "raw-call"}]},
            },
            {
                "role": "tool",
                "content": "unknown raw result",
                "tool_call_id": "raw-result-id",
            },
        ],
        live_tools=LiveToolRegistry(_tools()),
        semantic_goals=_goals(),
        candidate_action=_proposal(),
        evidence=evidence or EvidenceStore(),
    )


def test_builder_projects_only_public_candidate_and_referenced_schema() -> None:
    view = _view()

    assert [tool.name for tool in view.live_tools] == ["set_fan_level"]
    assert [goal.goal_id for goal in view.semantic_goals] == [
        "goal-precheck",
        "goal-fan",
    ]
    assert view.conversation[0].model_dump() == {
        "role": "user",
        "content": "Set the fan to level two.",
        "turn_id": "turn-user-2",
    }
    assert len(view.conversation) == 1

    payload = view.model_dump(mode="json")
    serialized = json.dumps(payload)
    assert "get_weather" not in serialized
    assert "x-private" not in serialized
    assert "x-provider" not in serialized
    assert "internal-call-7" not in serialized
    assert "internal-evidence-id" not in serialized
    assert "internal-policy-id" not in serialized
    assert payload["candidate_action"]["tool_calls"] == [
        {
            "tool_name": "set_fan_level",
            "arguments": {"level": 2},
            "goal_id": "goal-fan",
        }
    ]


def test_unknown_and_stale_tool_observations_cannot_enter_known_evidence() -> None:
    store = EvidenceStore()
    store.add(_known_evidence())
    store.add(
        Evidence(
            proposition="outside.weather",
            status=EvidenceStatus.UNKNOWN,
            source_kind=EvidenceSourceKind.TOOL,
            source_turn_id="turn-tool-2",
            source_tool_call_id="call-get-2",
            confidence=0.0,
        )
    )
    view = _view(evidence=store)
    assert [item.proposition for item in view.known_evidence] == ["cabin.mode"]
    assert "status" not in view.known_evidence[0].model_dump()

    store.invalidate_tool_state()
    assert _view(evidence=store).known_evidence == ()


def test_explicit_conflict_observation_cannot_enter_known_evidence() -> None:
    store = EvidenceStore()
    store.add(_known_evidence("cabin.mode"))
    store.add(
        Evidence(
            proposition="cabin.mode",
            value=None,
            status=EvidenceStatus.CONFLICT,
            source_kind=EvidenceSourceKind.TOOL,
            source_turn_id="turn-tool-2",
            source_tool_call_id="call-get-2",
            confidence=0.0,
        )
    )

    assert _view(evidence=store).known_evidence == ()


def test_unbound_candidate_is_rejected_before_a_critic_can_be_constructed() -> None:
    proposal = _proposal().model_copy(
        update={
            "tool_calls": [
                _proposal().tool_calls[0].model_copy(
                    update={"tool_name": "set_absent_control"}
                )
            ]
        }
    )

    with pytest.raises(CriticInputRejected, match="critic input rejected"):
        build_critic_input(
            policy="policy",
            conversation=[],
            live_tools=LiveToolRegistry(_tools()),
            semantic_goals=_goals(),
            candidate_action=proposal,
            evidence=EvidenceStore(),
        )


def test_every_nested_dto_forbids_unlisted_fields() -> None:
    payload = _view().model_dump(mode="json")
    modified = deepcopy(payload)
    modified["candidate_action"]["tool_calls"][0]["call_id"] = "not-public"
    with pytest.raises(ValidationError):
        CriticInput.model_validate(modified)

    modified = deepcopy(payload)
    modified["live_tools"][0]["parameters"][0]["schema_extension"] = True
    with pytest.raises(ValidationError):
        CriticInput.model_validate(modified)

    modified = deepcopy(payload)
    modified["conversation"][0]["metadata"] = {"private": True}
    with pytest.raises(ValidationError):
        CriticInput.model_validate(modified)


class FakeCriticClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.response)


def test_general_critic_uses_structured_client_and_safe_serialized_view() -> None:
    client = FakeCriticClient(CriticVerdict(decision=CriticDecision.ACCEPT))

    verdict = GeneralCritic(client).review(_view())

    assert verdict.decision is CriticDecision.ACCEPT
    assert client.calls[0]["response_model"] is CriticVerdict
    assert client.calls[0]["critic"] is True
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert sent == _view().model_dump(mode="json")


@pytest.mark.parametrize("decision", ["approve", "revise", "retry", "allow"])
def test_verdict_accepts_only_four_public_decisions(decision: str) -> None:
    with pytest.raises(ValidationError):
        CriticVerdict.model_validate({"decision": decision})


def test_non_accept_verdict_requires_a_bounded_reason() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(decision=CriticDecision.ASK)
    with pytest.raises(ValidationError):
        CriticVerdict.model_validate(
            {
                "decision": "decline",
                "reason_code": "policy_risk",
                "reason": "not allowed",
                "extra": True,
            }
        )


def test_second_replan_request_is_deterministically_closed() -> None:
    requested = CriticVerdict(
        decision=CriticDecision.REPLAN_ONCE,
        reason_code=CriticReasonCode.DEPENDENCY_GAP,
        reason="The declared precondition is not reflected in the candidate.",
    )
    client = FakeCriticClient(requested)
    critic = GeneralCritic(client)

    first = critic.review(_view(), replan_already_used=False)
    second = critic.review(_view(), replan_already_used=True)

    assert first.decision is CriticDecision.REPLAN_ONCE
    assert first.requires_revision
    assert second.decision is CriticDecision.DECLINE
    assert second.reason_code is CriticReasonCode.REPLAN_LIMIT_REACHED
    assert not second.requires_revision


def test_general_critic_revalidates_fake_client_output() -> None:
    client = FakeCriticClient(
        {
            "decision": "ask",
            "reason_code": "unresolved_ambiguity",
            "reason": "Choose one candidate.",
            "unexpected": "field",
        }
    )

    with pytest.raises(ValidationError):
        GeneralCritic(client).review(_view())
