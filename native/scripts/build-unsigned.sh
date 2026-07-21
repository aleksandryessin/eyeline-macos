#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
NATIVE_DIR=${SCRIPT_DIR:h}
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

"$SCRIPT_DIR/generate.sh"

xcodebuild \
  -project "$NATIVE_DIR/EyeLine.xcodeproj" \
  -scheme EyeLineHost \
  -configuration Debug \
  -destination "platform=macOS,arch=arm64" \
  -derivedDataPath "$NATIVE_DIR/DerivedData" \
  CODE_SIGNING_ALLOWED=NO \
  build
