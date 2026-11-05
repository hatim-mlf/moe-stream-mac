#!/usr/bin/env bash
# Overnight RFT sampling: base model, 8 samples at T=0.7 for each of the 250 training tasks (resumable).
set -euo pipefail
cd "$(dirname "$0")/../.."
R=results/20261105T213000Z-rft-base-k8
.venv/bin/mlx_lm.server --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --host 127.0.0.1 --port 8080 > "$R/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 2; done
.venv/bin/python tools/ft_rft.py sample data/tasks-v1 data/tasks-v2 data/tasks-v3 data/tasks-v4 data/tasks-v5 \
  --k 8 --temperature 0.7 --seed 20261105 --out "$R/rft-samples.jsonl"
echo "sampling done"
