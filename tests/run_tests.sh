#!/bin/bash
# Run all MeanFlow unit tests
#
# Usage:
#   ./tests/run_tests.sh           # Run all tests
#   ./tests/run_tests.sh -v        # Verbose output
#   ./tests/run_tests.sh -k jvp    # Run only tests matching "jvp"
#   ./tests/run_tests.sh --cov     # Run with coverage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Default arguments
PYTEST_ARGS="-v"

# Pass through any additional arguments
if [ $# -gt 0 ]; then
    PYTEST_ARGS="$@"
fi

echo "Running MeanFlow unit tests..."
echo "================================"

python -m pytest tests/ $PYTEST_ARGS

echo ""
echo "================================"
echo "All tests completed!"
