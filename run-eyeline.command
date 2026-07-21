#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
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

models_ready() {
  [[ -f "$EYELINE_MODEL_DIR/face_landmarker.task" ]] &&
    [[ -f "$EYELINE_MODEL_DIR/weights/warping_model/flx/12/L/L.index" ]] &&
    [[ -f "$EYELINE_MODEL_DIR/weights/warping_model/flx/12/R/R.index" ]]
}

if ! models_ready; then
  print "EyeLine model assets are missing; downloading pinned, checksum-verified assets…"
  if ! uv run --project "$SCRIPT_DIR" --python 3.12 python \
    tools/download_models.py --model-dir "$EYELINE_MODEL_DIR"; then
    print -u2 "EyeLine could not download or verify its model assets. Check the network error above; existing files were not replaced unless checksum verification succeeded."
    exit 1
  fi
  if ! models_ready; then
    print -u2 "EyeLine model bootstrap completed without the required Face Landmarker and L/R checkpoints. Remove only incomplete model files and retry."
    exit 1
  fi
fi

exec uv run --project "$SCRIPT_DIR" --python 3.12 eyeline run "$@"
