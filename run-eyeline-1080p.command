#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
exec "$SCRIPT_DIR/run-eyeline.command" --config "$SCRIPT_DIR/config/zoom-1080p.yaml" "$@"
