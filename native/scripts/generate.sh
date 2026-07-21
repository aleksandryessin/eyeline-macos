#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
NATIVE_DIR=${SCRIPT_DIR:h}

if ! command -v xcodegen >/dev/null 2>&1; then
  print -u2 "xcodegen is required. Install it with: brew install xcodegen"
  exit 2
fi

cd "$NATIVE_DIR"
xcodegen generate --spec project.yml
print "Generated $NATIVE_DIR/EyeLine.xcodeproj"
