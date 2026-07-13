# ADR-001: Competition Compliance Boundary

- Status: Accepted
- Date: 2026-07-11
- Scope: Track 1 CAR-Guard runtime and production image
- Required release review: one author and one independent reviewer

## Context

CAR-Guard is evaluated through a live A2A boundary. The runtime may reason from
the current request and the capabilities that are actually exposed, but it must
not identify benchmark cases, reconstruct removed components, consume evaluator
internals, or optimize against scoring implementation details. Development tools
may inspect public evaluation artifacts offline; that code and data are outside
the production trust boundary.

This ADR turns that distinction into an enforceable source, dependency, data,
and image contract. There are no per-task exceptions and no inline suppressions
for production code.

## Decision

### Runtime input allowlist

Planner, critic, validator, and policy components may receive only these fields:

| Field | Provenance |
| --- | --- |
| `policy` | Current system policy |
| `conversation` | Current conversation and roles |
| `live_tools` | Tool definitions exposed in the current turn |
| `tool_results` | Results observed in the current conversation |
| `derived_intent_state` | Goals, ambiguity, evidence, and confirmation state derived only from rows above |
| `candidate_action` | Proposed action for the current turn |
| `context_id` | Opaque identifier used only for session isolation |

DTO constructors must reject extra fields. Runtime routing may depend on user
intent, policy, risk, and observed evidence. It may not depend on a benchmark
category, known task identifier, missing-component label, reference answer,
score, evaluator state, or a comparison between a complete catalog and the live
inventory.

The validator asks whether the candidate is supported by policy, conversation,
observed evidence, and the current live schema. It never predicts whether a
benchmark judge will accept the answer.

### Behavior matrix

| Design | Decision | Boundary |
| --- | --- | --- |
| Stable policy compilation | Allowed | Current policy remains authoritative for dynamic values |
| General planner, critic, and validator | Allowed | Input uses the runtime DTO allowlist only |
| Intent-selected invariant operation recipes | Allowed | Registry is identical for every scenario |
| Deterministic derivation from observed results | Allowed | Every derived parameter retains provenance |
| Live schema feasibility check | Allowed | Runs after an intent-first plan identifies a required capability |
| Offline analysis of public evaluator output | Development only | Must remain outside production paths and image |
| Complete static tool catalog | Not shipped | Runtime must not diff it against live tools or results |
| Missing-part or task-category routing | Prohibited | No classifier, prompt branch, recipe, model route, or fallback may implement it |
| Score-aware output repair | Prohibited | No reward, subscore, or judge-aware feedback reaches runtime decisions |
| Evaluator, user simulator, or benchmark environment import | Prohibited | Enforced by AST import and dependency checks |
| Direct vehicle-state access | Prohibited | Outbound execution uses only currently exposed official tools |

### Production source allowlist

The source allowlist is defined once as `PRODUCTION_PATH_ALLOWLIST` in
`scripts/check_runtime_compliance.py`:

```text
src/track_1_agent_under_test/
src/logging_utils.py
src/tool_call_types.py
src/turn_metrics.py
```

Adding another runtime path requires changing the allowlist, passing compliance
tests, and reviewing why the dependency belongs in the Agent. Repository-level
`tests/`, `docs/`, evaluator and Track 2 sources, scenarios,
third-party benchmark code, task data, results, logs, caches, and secrets are
development assets and are not production inputs.

The production Python environment consists of base project dependencies plus
the `track-1-agent` optional dependency group. The evaluator dependency group is
not part of that environment. Direct imports of evaluator modules or the
benchmark package fail compliance even if a developer environment happens to
make them importable.

### Production image allowlist

The final image may contain only:

1. The resolved Track 1 virtual environment, built from the production
   dependency set.
2. Files selected by the production source allowlist above.
3. Minimal license or image metadata required for distribution.

Dependency manifests, build caches, and temporary source dependencies may be
used in a builder stage but must not be copied into the final stage. Evaluator,
Track 2, `agentbeats`, benchmark `third_party`, tests, docs, task/result data,
`.env*`, VCS metadata, and logs must not appear in the final filesystem.

The current source scan is the M0 guard for this design. T18 must make Docker
copy operations match this allowlist and verify the built filesystem and SBOM;
source compliance alone is not evidence that an image is compliant.

## Enforcement

`python scripts/check_runtime_compliance.py` performs three checks:

1. It scans only the production source allowlist by default. Prompt, recipe,
   config, documentation shipped with the runtime, and Python text are checked
   for benchmark-only metadata and public task identifiers.
2. It parses Python with `ast` and rejects static or constant dynamic imports of
   evaluator, user-simulator, reward-calculator, or benchmark environment code.
3. It parses `pyproject.toml` and rejects banned direct dependencies from base
   dependencies or the `track-1-agent` extra. Development-only extras are not
   treated as production dependencies.

Negative assertions under repository-level `tests/**` are excluded by default
because they must contain the strings they prove are rejected. This exception
does not apply to a `tests` directory nested under production source. There are
no source annotations, task IDs, filenames, or scenario-specific suppressions
that can waive a finding.

CI runs the scanner before the compliance unit suite. A finding blocks merge and
image publication. A generic false positive must be corrected by narrowing a
rule with a corresponding regression test and independent review; it must not
be bypassed with a task-specific allow rule.

## Verification and release review

Run locally:

```bash
python scripts/check_runtime_compliance.py
python -m unittest discover -s tests/compliance -p 'test_*.py' -v
```

Before release, two reviewers must record that they checked:

- Runtime DTO fields and constructors still implement the data allowlist.
- Intent extraction is independent of live inventory and skill registry content
  is invariant across scenarios.
- No complete-catalog comparison or task/category/missing-part branch exists.
- The scanner, counterfactual tests, and runtime contract tests pass.
- The built image file list and dependency SBOM match the image allowlist.
- Agent and evaluator credentials are separate and no secret is present in
  source, logs, layers, image config, or release artifacts.

## Consequences

The runtime cannot use evaluator internals for convenience, and new production
paths require explicit review. This adds a small CI cost and deliberately makes
benchmark-aware shortcuts difficult to introduce accidentally.

Static analysis cannot prove the absence of a dynamically constructed import or
validate a built image. Those risks are covered by runtime design review,
counterfactual tests, and the T18 image filesystem/SBOM gate. LLM output remains
untrusted and cannot replace source-to-sink review of a runtime decision.
