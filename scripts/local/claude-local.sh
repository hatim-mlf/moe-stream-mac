#!/usr/bin/env bash
# Run Claude Code against the local model: MLX server (:8080) <- LiteLLM proxy (:4000) <- claude.
# Starts the proxy if it isn't running. The MLX server must already be up (scripts/local/serve-mlx.sh).
set -euo pipefail
readonly ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly PROXY_PORT="${PROXY_PORT:-4000}"
readonly LOG="$ROOT/results/litellm.log"
readonly KEY_FILE="$ROOT/config/.litellm_key"  # gitignored; only processes on this Mac use it

if [[ ! -s "$KEY_FILE" ]]; then
  (umask 077; echo "sk-local-$(openssl rand -hex 24)" >"$KEY_FILE")
fi
LITELLM_MASTER_KEY="$(cat "$KEY_FILE")"
export LITELLM_MASTER_KEY

if ! curl -sf -m 2 http://127.0.0.1:8080/v1/models >/dev/null; then
  echo "MLX server is not running on :8080. Start it first: bash scripts/local/serve-mlx.sh" >&2
  exit 1
fi
if ! curl -s -m 2 -o /dev/null "http://127.0.0.1:$PROXY_PORT/health/liveliness"; then
  nohup "$ROOT/.venv/bin/litellm" --config "$ROOT/config/litellm.yaml" --host 127.0.0.1 --port "$PROXY_PORT" >"$LOG" 2>&1 &
  for _ in $(seq 1 30); do
    curl -s -m 2 -o /dev/null "http://127.0.0.1:$PROXY_PORT/health/liveliness" && break
    sleep 1
  done
fi

export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
export ANTHROPIC_MODEL="local-qwen"
export ANTHROPIC_DEFAULT_SONNET_MODEL="local-qwen"
export ANTHROPIC_DEFAULT_OPUS_MODEL="local-qwen"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="local-qwen"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
exec claude "$@"
