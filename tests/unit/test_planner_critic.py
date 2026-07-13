from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from track_1_agent_under_test.car_guard.domain import (
    CommitDecision,
    DecisionProposal,
    GateOutcome,
    Goal,
)
from track_1_agent_under_test.car_guard.planning.action_planner import ActionPlanner
from track_1_agent_under_test.car_guard.validation import (
    CriticInput,
    should_run_general_critic,
)


class FakeClient:
    def __init__(self, proposal: DecisionProposal) -> None:
        self.proposal = proposal
        self.messages = None

    def generate(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(value=self.proposal)


def get_proposal() -> DecisionProposal:
    return DecisionProposal.model_validate(
        {
            "kind": "tool_get",
            "goal_ids": ["goal-1"],
            "tool_calls": [
                {
                    "tool_name": "get_status",
                    "arguments": {},
                    "goal_id": "goal-1",
                }
            ],
        }
    )


def test_action_planner_assigns_stable_internal_call_id() -> None:
    proposal = get_proposal()
    client = FakeClient(proposal)
    planner = ActionPlanner(client)
    kwargs = {
        "policy": "policy",
        "conversation": [],
        "live_tools": [],
        "live_bindings": [],
        "conditional_closure": {},
        "goal_dag": {"goals": []},
        "evidence": {},
        "ambiguities": [],
        "relevant_recipes": [],
        "budget": {},
    }
    # The real runtime supplies a GoalDAG; this lightweight stand-in exposes
    # the same JSON serialization path used by the test.
    from track_1_agent_under_test.car_guard.domain import GoalDAG

    kwargs["goal_dag"] = GoalDAG()
    first = planner.propose_next(**kwargs)
    second = planner.propose_next(**kwargs)

    assert first.tool_calls[0].call_id is not None
    assert first.tool_calls[0].call_id == second.tool_calls[0].call_id
    assert "semantic_goals" in client.messages[1]["content"]


def test_runtime_call_id_factory_overrides_model_supplied_id() -> None:
    proposal = get_proposal().model_copy(
        update={
            "tool_calls": [
                get_proposal().tool_calls[0].model_copy(update={"call_id": "model-id"})
            ]
        }
    )
    client = FakeClient(proposal)
    from track_1_agent_under_test.car_guard.domain import GoalDAG

    planned = ActionPlanner(client).propose_next(
        policy="policy",
        conversation=[],
        live_tools=[],
        live_bindings=[],
        conditional_closure={},
        goal_dag=GoalDAG(),
        evidence={},
        ambiguities=[],
        relevant_recipes=[],
        budget={},
        call_id_factory=lambda: "session-call-7",
    )

    assert planned.tool_calls[0].call_id == "session-call-7"


def test_critic_dto_rejects_non_allowlisted_fields() -> None:
    proposal = get_proposal()
    payload = {
        "policy": "policy",
        "conversation": [],
        "live_tools": [],
        "semantic_goals": [
            Goal(goal_id="goal-1", semantic_operation="read_status")
        ],
        "candidate_action": proposal,
        "score": 1,
    }
    with pytest.raises(ValidationError):
        CriticInput.model_validate(payload)


@pytest.mark.parametrize(
    "outcome",
    [
        GateOutcome.UNSUPPORTED_CAPABILITY,
        GateOutcome.UNSUPPORTED_PARAMETER,
        GateOutcome.UNAVAILABLE_EVIDENCE,
        GateOutcome.NEED_READ,
    ],
)
def test_deterministic_limitations_never_enter_critic(outcome: GateOutcome) -> None:
    proposal = get_proposal()
    decision = CommitDecision(outcome=outcome)

    assert not should_run_general_critic(
        proposal,
        decision,
        enable_critic=True,
        unresolved_candidate_count=3,
        chain_length=4,
    )


def test_state_change_uses_critic_when_gate_did_not_deterministically_stop() -> None:
    proposal = DecisionProposal.model_validate(
        {
            "kind": "tool_set",
            "goal_ids": ["goal-1"],
            "tool_calls": [
                {
                    "tool_name": "set_control",
                    "arguments": {"on": True},
                    "argument_sources": {"on": "user-turn-1"},
                }
            ],
        }
    )
    decision = CommitDecision(
        outcome=GateOutcome.ALLOW,
        normalized_calls=[
            {"tool_name": "set_control", "arguments": {"on": True}}
        ],
    )

    assert should_run_general_critic(proposal, decision, enable_critic=True)
