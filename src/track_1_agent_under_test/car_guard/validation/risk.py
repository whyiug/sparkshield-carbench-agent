"""General product-risk routing for optional critic calls."""

from __future__ import annotations

from ..domain import CommitDecision, DecisionProposal, GateOutcome, ProposalKind


_DETERMINISTIC_NO_CRITIC = {
    GateOutcome.UNSUPPORTED_CAPABILITY,
    GateOutcome.UNSUPPORTED_PARAMETER,
    GateOutcome.UNAVAILABLE_EVIDENCE,
    GateOutcome.NEED_READ,
}


def should_run_general_critic(
    proposal: DecisionProposal,
    decision: CommitDecision,
    *,
    enable_critic: bool,
    unresolved_candidate_count: int = 0,
    chain_length: int = 1,
) -> bool:
    if not enable_critic or decision.outcome in _DETERMINISTIC_NO_CRITIC:
        return False
    if proposal.kind is ProposalKind.TOOL_SET:
        return True
    if unresolved_candidate_count >= 2 or chain_length >= 3:
        return True
    return proposal.needs_critic or decision.outcome in {
        GateOutcome.POLICY_CONFLICT,
        GateOutcome.NEED_USER_DISAMBIGUATION,
        GateOutcome.INVALID_PROPOSAL,
    }
