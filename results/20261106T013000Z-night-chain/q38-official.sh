#!/bin/bash
# After methods.sh and the bf16 fetch: convert official Qwen3.8-27B to MLX 6-bit, delete bf16, eval it.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=results/20261106T013000Z-night-chain
DB=/Users/hatim/models/Qwen3.8-27B-official-bf16
D6=/Users/hatim/models/Qwen3.8-27B-official-mlx-6bit
BASE_S5=results/20261105T201500Z-eval-resolution/base-s5-t06.json
log() { echo "$(date +%H:%M:%S) $*"; }
while pgrep -f "night-chain/methods.sh|fetch-model.sh Qwen/Qwen3.8-27B" >/dev/null; do sleep 30; done
grep -q "FAILED\|mismatch" "$DB.fetch.log" && { log "bf16 fetch failed, see $DB.fetch.log"; exit 1; }
log "convert to 6-bit"
.venv/bin/python -m mlx_lm convert --hf-path "$DB" --mlx-path "$D6" -q --q-bits 6 --q-group-size 64 > "$R/q38off-convert.log" 2>&1 &
CONV=$!; .venv/bin/python tools/wired_watch.py "$CONV" > "$R/q38off-convert-watch.log" 2>&1 &
wait "$CONV" || { log "convert FAILED"; exit 1; }
[ -f "$D6/config.json" ] && rm -rf "$DB" && log "bf16 deleted"
.venv/bin/mlx_lm.server --model "$D6" --host 127.0.0.1 --port 8080 > "$R/q38off-server.log" 2>&1 &
S=$!; trap 'kill $S 2>/dev/null' EXIT
.venv/bin/python tools/wired_watch.py "$S" > "$R/q38off-watch.log" 2>&1 &
for _ in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 2; done
.venv/bin/python tools/ft_eval.py --split heldout-v2 --attempts 1 --temperature 0.6 --seed 7 --samples 5 --out "$R/q38off-s5.json" > "$R/q38off-s5.log" 2>&1
.venv/bin/python tools/ft_eval.py --split heldout-v2 --attempts 3 --temperature 0.6 --seed 7 --samples 5 --out "$R/q38off-repair3.json" > "$R/q38off-repair3.log" 2>&1
.venv/bin/python tools/ft_eval.py --paired "$BASE_S5" "$R/q38off-s5.json" > "$R/q38off-paired.txt" 2>&1
log "done: $(tail -1 $R/q38off-s5.log) | repair3: $(tail -1 $R/q38off-repair3.log)"
