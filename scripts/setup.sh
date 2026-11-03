#!/usr/bin/env bash
# Clone the upstream C engine at a pinned commit, build it, and run the weightless test suite.
# Needs only a C99 compiler, make and OpenMP. macOS: run `brew install libomp` first.
set -euo pipefail

readonly UPSTREAM_URL="https://github.com/FareedKhan-dev/kimi-k3-in-c.git"
readonly UPSTREAM_SHA="81bb6c23ccee2154bc35cad922588ab66494a9a9"  # main @ 2026-10-02 (v1.1.0)
readonly ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly UPSTREAM_DIR="$ROOT/third_party/kimi-k3-in-c"

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  git clone --quiet "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi
git -C "$UPSTREAM_DIR" fetch --quiet origin
git -C "$UPSTREAM_DIR" checkout --quiet "$UPSTREAM_SHA"

make -C "$UPSTREAM_DIR" -j
make -C "$UPSTREAM_DIR" test

echo "Built and tested upstream at $(git -C "$UPSTREAM_DIR" rev-parse --short HEAD)"
