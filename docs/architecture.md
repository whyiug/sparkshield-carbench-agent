# CAR-Guard Architecture

CAR-Guard is an intent-first, evidence-backed in-car tool agent. Runtime state is
isolated by A2A `context_id`; the evaluator remains the only component that executes
vehicle tools.

## Runtime Flow

1. The A2A adapter parses the current system policy, user turn, live tool definitions,
   and returned observations.
2. The inventory-blind intent stage converts policy and conversation into semantic
   goals and a stable Goal DAG. Live tools are absent from this interface.
3. The invariant recipe registry selects operations from intent and policy. The live
   binder then maps only those frozen operations onto the current definitions.
4. Purpose-specific reads populate the evidence store. Conditional policy branches are
   instantiated from observed state, and atomic feasibility is proven before a first
   state change.
5. The resolver eliminates candidates in policy, explicit user, preference, heuristic,
   context, and user-clarification order.
6. The planner proposes one response mode. The deterministic pre-commit gate validates
   intent, live schema, provenance, ambiguity, evidence, policy, confirmation, closure,
   ordering, idempotency, and budget.
7. A general critic is optional for state changes and general intent or policy risk. It
   is never used to reason about unsupported capability, unsupported parameters, or
   unavailable required evidence.
8. The serializer emits either tool calls or natural text, never both. State changes are
   serial; only independent reads with distinct names may be parallel.

## Trust Boundaries

The runtime DTO contains only current policy, conversation, live definitions, returned
observations, derived intent/evidence state, candidate action, and context isolation
metadata. Development evaluation data, scoring components, reference answers, and the
core benchmark package are outside the production dependency and image boundary.

Model output is an untrusted proposal. It cannot create an outbound call without passing
the deterministic gate. All IDs and derived values retain source provenance, successful
state changes are idempotency tombstoned, and terminal side effects remain replay-safe
until session TTL expiry.

## Deployment

The production build has an independent lock and Dockerfile-specific context allowlist.
The runtime image contains only the Track 1 server, thin executor, CAR-Guard package,
and three required shared protocol modules. It is Linux amd64, CPU-only, non-root, and
receives only declared `AGENT_*` configuration. Evaluator credentials are never passed
to the Agent container.
