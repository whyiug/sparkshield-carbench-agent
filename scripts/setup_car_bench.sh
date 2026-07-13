#!/bin/bash
# Clone the external CAR-bench repository required by the evaluator.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CAR_BENCH_DIR="$PROJECT_ROOT/third_party/car-bench"
CAR_BENCH_REPOSITORY="https://github.com/CAR-bench/car-bench.git"
CAR_BENCH_COMMIT="3868ee57ff9d0de05acd784c85bb16eb6fc7102b"

if [ -d "$CAR_BENCH_DIR" ]; then
    actual_commit="$(git -C "$CAR_BENCH_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [ "$actual_commit" = "$CAR_BENCH_COMMIT" ]; then
        echo "car-bench is already pinned at $CAR_BENCH_COMMIT"
        exit 0
    fi
    echo "car-bench exists at an unexpected commit: ${actual_commit:-unknown}" >&2
    echo "Expected $CAR_BENCH_COMMIT. Move the existing directory aside and rerun." >&2
    exit 1
fi

mkdir -p "$(dirname "$CAR_BENCH_DIR")"

echo "Fetching pinned car-bench commit $CAR_BENCH_COMMIT..."
git init --quiet "$CAR_BENCH_DIR"
git -C "$CAR_BENCH_DIR" remote add origin "$CAR_BENCH_REPOSITORY"
git -C "$CAR_BENCH_DIR" fetch --quiet --depth 1 origin "$CAR_BENCH_COMMIT"
git -C "$CAR_BENCH_DIR" checkout --quiet --detach FETCH_HEAD


echo ""
echo "✅ Setup complete! car-bench is ready at:"
echo "   $CAR_BENCH_DIR"
echo ""
echo "📝 Note: Tasks and mock data are automatically loaded from HuggingFace"
