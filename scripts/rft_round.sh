#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SERVER="$ROOT/.venv/bin/mlx_lm.server"
MODEL="/Users/hatim/models/Qwen3.6-35B-A3B-4bit"
BASE_EVAL="$ROOT/results/20261105T201500Z-eval-resolution/base-s5-t06.json"
PORT=8080
DRY_RUN=0
NAME=""
SAMPLE_COUNT=0
SAMPLES=()
SERVER_PID=""
CONFIG_TMP=""

usage() {
  echo "Usage: scripts/rft_round.sh SAMPLES... --name NAME [--dry-run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name)
      if [ "$#" -lt 2 ]; then usage; exit 2; fi
      NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      SAMPLES+=("$1")
      SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
      shift
      ;;
  esac
done

if [ "$SAMPLE_COUNT" -eq 0 ] || [ -z "$NAME" ]; then
  usage
  exit 2
fi
case "$NAME" in
  *[!A-Za-z0-9._-]*) echo "NAME may contain only letters, digits, dot, underscore, and hyphen" >&2; exit 2 ;;
esac

cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DATA_DIR="$ROOT/data/finetune/rft-$NAME"
RUN_DIR="$ROOT/results/${STAMP}-rft-$NAME"
TRAIN_ROWS="<number-of-train-rows>"
STEPS="<max(1,train-rows/4)>"

print_command() {
  printf '+'
  for item in "$@"; do
    printf ' %q' "$item"
  done
  printf '\n'
}

print_redirect() {
  destination="$1"
  shift
  printf '+'
  for item in "$@"; do
    printf ' %q' "$item"
  done
  printf ' > %q 2>&1\n' "$destination"
}

print_background() {
  destination="$1"
  shift
  printf '+'
  for item in "$@"; do
    printf ' %q' "$item"
  done
  printf ' > %q 2>&1 &\n' "$destination"
}

show_dataset_plan() {
  print_command command -v lsof
  print_command lsof -nP -iTCP:"$PORT" -sTCP:LISTEN
  build=("$PYTHON" "$ROOT/tools/ft_rft.py" build --samples)
  for sample_path in "${SAMPLES[@]}"; do build+=("$sample_path"); done
  build+=(--tasks "$ROOT/data/tasks-v1" "$ROOT/data/tasks-v2" "$ROOT/data/tasks-v3" \
    "$ROOT/data/tasks-v4" "$ROOT/data/tasks-v5" --out "$DATA_DIR" \
    --max-pass-fraction 0.875 --per-task 2)
  print_command "${build[@]}"
  print_command mkdir -p "$RUN_DIR" "$RUN_DIR/scale20" "$RUN_DIR/scale4"
  print_command "$PYTHON" -c \
    'from pathlib import Path; import sys; print(sum(1 for _ in Path(sys.argv[1]).open(encoding="utf-8")))' \
    "$DATA_DIR/train.jsonl"
}

show_training_plan() {
  scale="$1"
  scale_dir="$RUN_DIR/scale$scale"
  config="$ROOT/docs/finetune/lora.yaml"
  if [ "$scale" = 4 ]; then
    config="$scale_dir/lora-scale4.yaml"
    print_redirect "$config" sed 's/^  scale: 20\.0$/  scale: 4.0/' \
      "$ROOT/docs/finetune/lora.yaml"
    print_command grep -q '^  scale: 4\.0$' "$config"
  fi
  print_command mkdir -p "$scale_dir/checkpoints"
  training=(env FROZEN_GDN_KERNEL=1 MLX_CACHE_LIMIT_GB=0 "$PYTHON" \
    "$ROOT/tools/ft_lora_lowmem.py" --train --config "$config" --model "$MODEL" \
    --data "$DATA_DIR" --num-layers 1 --iters "$TRAIN_ROWS" --val-batches 10 \
    --steps-per-eval "$STEPS" --save-every "$STEPS" \
    --adapter-path "$scale_dir/checkpoints")
  print_redirect "$scale_dir/train.log" "${training[@]}"
  print_command cat "$scale_dir/train.log"
  print_command "$PYTHON" "$ROOT/tools/rft_round_metrics.py" select-checkpoint \
    --log "$scale_dir/train.log" --checkpoints "$scale_dir/checkpoints" \
    --adapter "$scale_dir/adapter" --chosen-step "$scale_dir/chosen-step.txt"
}

show_evaluation_plan() {
  scale="$1"
  scale_dir="$RUN_DIR/scale$scale"
  print_background "$scale_dir/server.log" "$SERVER" --model "$MODEL" \
    --adapter-path "$scale_dir/adapter" --host 127.0.0.1 --port "$PORT"
  print_command "$PYTHON" -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2).read()'
  print_command sleep 2
  echo "  (repeat the model endpoint probe and sleep for up to 300 attempts)"
  print_command "$PYTHON" "$ROOT/tools/ft_eval.py" --split heldout-v2 --attempts 1 \
    --temperature 0.6 --seed 7 --samples 5 --out "$scale_dir/heldout-v2.json"
  print_command kill '<server-pid-started-by-this-script>'
  print_command wait '<server-pid-started-by-this-script>'
  print_redirect "$scale_dir/paired.txt" "$PYTHON" "$ROOT/tools/ft_eval.py" \
    --paired "$BASE_EVAL" "$scale_dir/heldout-v2.json"
  print_command cat "$scale_dir/paired.txt"
}

show_plan() {
  show_dataset_plan
  show_training_plan 20
  show_training_plan 4
  show_evaluation_plan 20
  show_evaluation_plan 4
  print_command "$PYTHON" "$ROOT/tools/rft_round_metrics.py" write-metrics \
    --data "$DATA_DIR" --run-dir "$RUN_DIR" --name "$NAME" --stamp "$STAMP"
  print_command rm -f "$RUN_DIR/scale4/lora-scale4.yaml"
}

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  if [ -n "$CONFIG_TMP" ] && [ -f "$CONFIG_TMP" ]; then
    rm -f "$CONFIG_TMP"
  fi
}

stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

wait_for_server() {
  attempts=0
  while [ "$attempts" -lt 300 ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "MLX server exited before /v1/models became ready" >&2
      tail -n 60 "$SERVER_LOG" >&2
      return 1
    fi
    if "$PYTHON" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2).read()' \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    attempts=$((attempts + 1))
  done
  echo "Timed out waiting for MLX server" >&2
  tail -n 60 "$SERVER_LOG" >&2
  return 1
}

train_adapter() {
  scale="$1"
  scale_dir="$RUN_DIR/scale$scale"
  config="$ROOT/docs/finetune/lora.yaml"
  if [ "$scale" = 4 ]; then config="$RUN_DIR/scale4/lora-scale4.yaml"; fi
  mkdir -p "$scale_dir/checkpoints"
  if [ "$scale" = 4 ]; then
    CONFIG_TMP="$config"
    sed 's/^  scale: 20\.0$/  scale: 4.0/' "$ROOT/docs/finetune/lora.yaml" > "$CONFIG_TMP"
    if ! grep -q '^  scale: 4\.0$' "$CONFIG_TMP"; then
      echo "Could not set lora_parameters.scale to 4.0" >&2
      return 1
    fi
  fi
  steps_per_eval=$((ITERS / 4))
  if [ "$steps_per_eval" -lt 1 ]; then steps_per_eval=1; fi
  env FROZEN_GDN_KERNEL=1 MLX_CACHE_LIMIT_GB=0 "$PYTHON" \
    "$ROOT/tools/ft_lora_lowmem.py" --train --config "$config" --model "$MODEL" \
    --data "$DATA_DIR" --num-layers 1 --iters "$ITERS" --val-batches 10 \
    --steps-per-eval "$steps_per_eval" --save-every "$steps_per_eval" \
    --adapter-path "$scale_dir/checkpoints" > "$scale_dir/train.log" 2>&1 || {
      status=$?
      cat "$scale_dir/train.log" >&2
      return "$status"
    }
  cat "$scale_dir/train.log"
  "$PYTHON" "$ROOT/tools/rft_round_metrics.py" select-checkpoint \
    --log "$scale_dir/train.log" --checkpoints "$scale_dir/checkpoints" \
    --adapter "$scale_dir/adapter" --chosen-step "$scale_dir/chosen-step.txt"
}

evaluate_adapter() {
  scale="$1"
  scale_dir="$RUN_DIR/scale$scale"
  SERVER_LOG="$scale_dir/server.log"
  "$SERVER" --model "$MODEL" --adapter-path "$scale_dir/adapter" \
    --host 127.0.0.1 --port "$PORT" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server
  "$PYTHON" "$ROOT/tools/ft_eval.py" --split heldout-v2 --attempts 1 \
    --temperature 0.6 --seed 7 --samples 5 --out "$scale_dir/heldout-v2.json"
  stop_server
  "$PYTHON" "$ROOT/tools/ft_eval.py" --paired "$BASE_EVAL" \
    "$scale_dir/heldout-v2.json" > "$scale_dir/paired.txt"
  cat "$scale_dir/paired.txt"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$DRY_RUN" -eq 1 ]; then
  show_plan
  exit 0
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof is required to check port $PORT" >&2
  exit 1
fi
listeners="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$listeners" ]; then
  echo "Refusing to start: something is already listening on port $PORT" >&2
  echo "$listeners" >&2
  exit 1
fi

"$PYTHON" "$ROOT/tools/ft_rft.py" build --samples "${SAMPLES[@]}" \
  --tasks "$ROOT/data/tasks-v1" "$ROOT/data/tasks-v2" "$ROOT/data/tasks-v3" \
  "$ROOT/data/tasks-v4" "$ROOT/data/tasks-v5" --out "$DATA_DIR" \
  --max-pass-fraction 0.875 --per-task 2
mkdir -p "$RUN_DIR/scale20" "$RUN_DIR/scale4"
ITERS="$("$PYTHON" -c 'from pathlib import Path; import sys; print(sum(1 for _ in Path(sys.argv[1]).open(encoding="utf-8")))' "$DATA_DIR/train.jsonl")"
if [ "$ITERS" -lt 1 ]; then
  echo "RFT dataset has no train rows: $DATA_DIR/train.jsonl" >&2
  exit 1
fi
train_adapter 20
train_adapter 4
evaluate_adapter 20
evaluate_adapter 4
"$PYTHON" "$ROOT/tools/rft_round_metrics.py" write-metrics \
  --data "$DATA_DIR" --run-dir "$RUN_DIR" --name "$NAME" --stamp "$STAMP"
echo "RFT round complete: $RUN_DIR"
