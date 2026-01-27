#!/usr/bin/env bash

set -euo pipefail

TARGET_DIR="${1:-.}"

echo "Searching for *.Identifier files in: $TARGET_DIR"
echo

find "$TARGET_DIR" -type f -name "*.Identifier" -print -delete

echo
echo "Done."