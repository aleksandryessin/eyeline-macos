#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
cd "$PROJECT_DIR"
exec uv run --project "$PROJECT_DIR" --python 3.12 eyeline doctor --probe-output "$@"
