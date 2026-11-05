#!/usr/bin/env bash
# Eval-resolution chain: base greedy re-score, then 5 samples at T=0.6 for base and v5-A step 128.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=results/20261105T201500Z-eval-resolution
M=/Users/hatim/models/Qwen3.6-35B-A3B-4bit
A=$(pwd)/results/20261105T194500Z-v5-configA/adapter-s128
serve() {  # $1 = adapter dir or empty
  local extra=(); [ -n "$1" ] && extra=(--adapter-path "$1")
  .venv/bin/mlx_lm.server --model "$M" ${extra[@]+"${extra[@]}"} --host 127.0.0.1 --port 8080 > "$R/server-$2.log" 2>&1 &
  echo $! > "$R/server.pid"
  for _ in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null && return 0; sleep 2; done
  echo "server did not start" >&2; return 1
}
stop() { kill "$(cat "$R/server.pid")"; sleep 4; }
serve "" base
.venv/bin/python tools/ft_eval.py --split heldout-v2 --attempts 1 --temperature 0 --samples 1 --out $R/base-today-greedy.json > $R/base-greedy.log 2>&1
.venv/bin/python tools/ft_eval.py --split heldout-v2 --attempts 1 --temperature 0.6 --seed 7 --samples 5 --out $R/base-s5-t06.json > $R/base-s5.log 2>&1
stop
serve "$A" v5A
.venv/bin/python tools/ft_eval.py --split heldout-v2 --attempts 1 --temperature 0.6 --seed 7 --samples 5 --out $R/v5A-s5-t06.json > $R/v5A-s5.log 2>&1
stop
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-v2-first.json $R/base-today-greedy.json > $R/base-reproducibility.txt
.venv/bin/python tools/ft_eval.py --paired $R/base-s5-t06.json $R/v5A-s5-t06.json > $R/paired-s5.txt
echo done
