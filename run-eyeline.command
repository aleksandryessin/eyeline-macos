#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
export EYELINE_HOME="${EYELINE_HOME:-${HOME}/Library/Application Support/EyeLine}"
export EYELINE_MODEL_DIR="${EYELINE_MODEL_DIR:-${EYELINE_HOME}/models}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${EYELINE_HOME}/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${EYELINE_HOME}/cache/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${EYELINE_HOME}/cache/uv}"

mkdir -p "$EYELINE_MODEL_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  print -u2 "EyeLine requires uv. Install it from https://docs.astral.sh/uv/ and retry."
  exit 1
fi

exec uv run --project "$SCRIPT_DIR" --python 3.12 eyeline run "$@"
