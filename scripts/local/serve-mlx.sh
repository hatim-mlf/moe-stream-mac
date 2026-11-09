#!/usr/bin/env bash
# Start the local helper (mlx_lm.server on 127.0.0.1). Evaluation scripts start their own servers instead.
#
#   HELPER=kat       (default) KAT-Coder-V2.5-Dev OptiQ 4-bit: Polyglot 34/60 with 3-try repair in 70 min.
#   HELPER=qwen-rft  Qwen3.8-27B official 4-bit + RFT adapter: Polyglot 31/60 in 135 min (results/20261109T013000Z-polyglot).
#   HELPER=qwen      Qwen3.8-27B official 4-bit, no adapter.
# Other overrides: PORT=...
# The served model is recorded in results/delegations/current-adapter.txt (read by tools/delegate.py).
set -euo pipefail
readonly ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly HELPER="${HELPER:-kat}"
readonly PORT="${PORT:-8080}"
readonly QWEN="$HOME/models/Qwen3.8-27B-official-mlx-4bit"
readonly RECORD="$ROOT/results/delegations/current-adapter.txt"

case "$HELPER" in
  kat)      model="$HOME/models/KAT-Coder-V2.5-Dev-OptiQ-4bit"; adapter="" ;;
  qwen-rft) model="$QWEN"; adapter="$ROOT/adapters/qwen38-rft-v1" ;;
  qwen)     model="$QWEN"; adapter="" ;;
  *) echo "Unknown HELPER=$HELPER (use kat, qwen-rft or qwen)" >&2; exit 1 ;;
esac

if [[ ! -f "$model/config.json" ]]; then
  echo "Refusing to start: no model at $model" >&2
  exit 1
fi
args=(--model "$model" --host 127.0.0.1 --port "$PORT")
if [[ -n "$adapter" ]]; then
  if [[ ! -f "$adapter/adapters.safetensors" ]]; then
    echo "Refusing to start: no adapter at $adapter" >&2
    exit 1
  fi
  args+=(--adapter-path "$adapter")
fi

mkdir -p "$(dirname "$RECORD")"
echo "$HELPER $model${adapter:+ + $adapter}" > "$RECORD"
exec "$ROOT/.venv/bin/mlx_lm.server" "${args[@]}"
