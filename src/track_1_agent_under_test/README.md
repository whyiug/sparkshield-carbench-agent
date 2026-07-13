# Track 1 Agent Under Test

This package contains CAR-Guard, an intent-first, evidence-backed Track 1 agent.
It preserves the official A2A boundary and puts every model-generated state
change behind deterministic capability, schema, evidence, policy, confirmation,
and idempotency checks.

## What This Agent Demonstrates

- Parses evaluator messages into policy/user text, tool definitions, and tool
  results.
- Maintains conversation history per `context_id`.
- Calls a LiteLLM-compatible model using the configured `AGENT_*` route.
- Returns either a user-facing text Part or a data Part with
  `{"tool_calls": [...]}`, never both.
- Never executes CAR-bench tools directly; the evaluator executes tool calls and
  returns tool results on the next turn.

## Turn Contract

The high-level contract is:

| Turn situation | Evaluator sends | Agent returns |
| --- | --- | --- |
| First task turn | text Part with `System: ... User: ...`, data Part with `{"tools": [...]}` | text Part or tool-call data Part |
| After agent tool calls | data Part with `{"tool_results": [...]}` | text Part or more tool-call data |
| After agent text response | next simulated user text Part | text Part or tool-call data |

For exact schemas and helper functions, read the
[development guide](../../docs/development-guide.md), especially:

- [Inbound messages](../../docs/development-guide.md#inbound-messages--what-your-agent-receives)
- [Outbound messages](../../docs/development-guide.md#outbound-messages--what-your-agent-should-return)
- [Agent executor contract](../../docs/development-guide.md#agent-executor-contract)

## Configuration

Keep evaluator and Agent credentials separate in `.env`:

```bash
GEMINI_API_KEY=...
AGENT_LLM=gemini/gemini-3.5-flash
AGENT_PROVIDER=gemini
AGENT_API_BASE=
AGENT_API_KEY=...
AGENT_TEMPERATURE=0
AGENT_STRUCTURED_OUTPUT_MODE=json_schema
AGENT_ENABLE_CRITIC=false
```

`GEMINI_API_KEY` is consumed by the local evaluator only. CAR-Guard consumes
only `AGENT_*`; even when both use the same provider they should use separate,
low-budget, revocable keys. Empty optional values are treated as unset. Explicit
non-routing CLI values override environment values, and environment values
override defaults. The CLI cannot change `provider` or `api_base` on an already
resolved route because that could redirect its credential; switch routes by
setting a complete `AGENT_LLM`/`AGENT_PROVIDER`/`AGENT_API_KEY` route and an
`AGENT_API_BASE` only when that provider route requires one.

Remote API bases must use HTTPS. Plain HTTP is accepted only for loopback
development endpoints (`localhost`, `127.0.0.1`, or `::1`), so a bearer key is
never intentionally sent to an external plaintext endpoint. LiteLLM message and
exception logging is redacted. Docker and submission scenarios continue to pass
only the selected generic `AGENT_*` route, avoiding injection of every profile's
credential into the production container.

`AGENT_STRUCTURED_OUTPUT_MODE` accepts `auto`, `json_schema`, or `json_object`.
In `auto`, custom OpenAI-compatible endpoints receive `json_object` plus an
explicit JSON Schema instruction, while native routes keep Pydantic schema mode.
The forced modes are useful for provider compatibility diagnosis; all returned
objects are still validated by the same Pydantic model before CAR-Guard uses
them.

The intent stage never receives live tools. The production image never contains
the evaluator, public tasks/results, Track 2 code, or the benchmark core package.

## Run

Run the service directly (loopback is the default):

```bash
uv run python src/track_1_agent_under_test/server.py --port 8080
```

The production Docker image explicitly passes `--host 0.0.0.0`; local runs do
not expose the service beyond `127.0.0.1` unless that override is supplied.
Boolean CLI settings support both forms, for example `--thinking` and
`--no-thinking`. A CLI value is applied only when explicitly present, so
`--temperature 0.0` and `--no-enable-critic` override environment values.

Local smoke:

```bash
uv run car-bench-run scenarios/track_1_agent_under_test/local_smoke.toml --show-logs
```

Docker smoke:

```bash
uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit
```

## Read More

- [Main README](../../README.md): setup, validation modes, and submission shape.
- [Development guide](../../docs/development-guide.md): detailed A2A turn
  contract.
- [Harnessing guide](../../docs/agent-under-test-harnessing.md): what advanced
  internal harnesses may and may not do.
