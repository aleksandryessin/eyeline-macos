#!/bin/zsh
set -euo pipefail

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

print "Xcode:"
xcodebuild -version
print ""
print "XcodeGen:"
if command -v xcodegen >/dev/null 2>&1; then
  xcodegen --version
else
  print "not installed (brew install xcodegen)"
fi
print ""
print "Signing identities (activation requires Apple Development):"
security find-identity -v -p codesigning 2>/dev/null || true
print ""
print "Unsigned CI build: native/scripts/build-unsigned.sh"
