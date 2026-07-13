# Third-Party CAR-bench Dependency

This repository wraps the upstream [CAR-bench](https://github.com/CAR-bench/car-bench)
benchmark, but does not vendor the benchmark source into git.

## Setup

Clone the local dependency before installing the evaluator extra or building the
green evaluator image:

```bash
./scripts/setup_car_bench.sh
```

The script checks out the repository's pinned CAR-bench commit into
`third_party/car-bench/`. That directory is a local ignored dependency. If an
existing checkout is at another commit, the script fails instead of silently
changing it.

The green evaluator imports CAR-bench from this path. Purple reference agents do
not need the CAR-bench checkout at runtime.

## Running the Benchmark

After setup, install the normal extras and run any scenario from `scenarios/`:

```bash
uv sync --extra track-1-agent --extra car-bench-evaluator
uv run car-bench-run scenarios/track_1_agent_under_test/local_smoke.toml
```
