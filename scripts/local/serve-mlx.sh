#!/usr/bin/env bash
# Start mlx_lm.server on 127.0.0.1 with the Track A default model (override: MODEL=... PORT=...).
set -euo pipefail
readonly ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly MODEL="${MODEL:-$HOME/models/Qwen3.6-35B-A3B-4bit}"
readonly PORT="${PORT:-8080}"
exec "$ROOT/.venv/bin/mlx_lm.server" --model "$MODEL" --host 127.0.0.1 --port "$PORT"
