#!/usr/bin/env bash
# Start mlx_lm.server on 127.0.0.1 with the official Qwen3.8-27B 4-bit checkpoint (never modified).
# The local helper adds the Qwen3.8 RFT adapter by default (+9.3 pts heldout-v2, +10 pts CodeContests pilot).
# Overrides: MODEL=... PORT=... ADAPTER=<dir> or ADAPTER=none for the plain base model.
# Evaluation scripts start their own servers and do not use this script.
set -euo pipefail
readonly ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-official-mlx-4bit}"
readonly PORT="${PORT:-8080}"
readonly ADAPTER="${ADAPTER:-$ROOT/adapters/qwen38-rft-v1}"
readonly ADAPTER_RECORD="$ROOT/results/delegations/current-adapter.txt"

mkdir -p "$(dirname "$ADAPTER_RECORD")"
args=(--model "$MODEL" --host 127.0.0.1 --port "$PORT")
if [[ "$ADAPTER" == "none" ]]; then
  echo "none" > "$ADAPTER_RECORD"
else
  if [[ ! -f "$ADAPTER/adapters.safetensors" ]]; then
    echo "Refusing to start: no adapter at $ADAPTER (set ADAPTER=none for the base model)" >&2
    exit 1
  fi
  echo "$ADAPTER" > "$ADAPTER_RECORD"
  args+=(--adapter-path "$ADAPTER")
fi
exec "$ROOT/.venv/bin/mlx_lm.server" "${args[@]}"
