#!/usr/bin/env bash
#
# Setup git hooks for DenoisingZoo
#
# Usage: ./scripts/setup-hooks.sh
#

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/hooks"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Setting up git hooks..."

# Check hooks directory exists
if [[ ! -d "$HOOKS_DIR" ]]; then
    echo "Error: hooks/ directory not found at $HOOKS_DIR"
    exit 1
fi

# Install pre-commit hook
if [[ -f "$HOOKS_DIR/pre-commit" ]]; then
    cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "  Installed: pre-commit (runs unit tests)"
else
    echo "  Warning: hooks/pre-commit not found, skipping"
fi

echo ""
echo "Done! Hooks installed to .git/hooks/"
echo ""
echo "To skip hooks temporarily: git commit --no-verify"
echo "To uninstall: rm .git/hooks/pre-commit"
