#!/usr/bin/env bash
# Clone upstream C2C at a pinned commit and install it into a local venv.
set -euo pipefail

readonly UPSTREAM_URL="https://github.com/thu-nics/C2C.git"
readonly UPSTREAM_SHA="5eac72b90faf7b3dc2638e874fe7cba1e74c8bcf"  # main @ 2026-10-28
readonly ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly UPSTREAM_DIR="$ROOT/third_party/C2C"
readonly EXTRAS="${C2C_EXTRAS:-evaluation}"  # set C2C_EXTRAS=training,evaluation for Phase 3

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  git clone --quiet "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi
git -C "$UPSTREAM_DIR" fetch --quiet origin
git -C "$UPSTREAM_DIR" checkout --quiet "$UPSTREAM_SHA"

if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e "$UPSTREAM_DIR[$EXTRAS]" huggingface_hub

echo "Ready. Upstream at $(git -C "$UPSTREAM_DIR" rev-parse --short HEAD). Activate with: source .venv/bin/activate"
