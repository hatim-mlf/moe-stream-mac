#!/usr/bin/env bash
# Build llama.cpp at a pinned tag, CPU-only (no Metal/GPU) for a fair CPU-vs-CPU baseline.
set -euo pipefail

readonly LLAMA_URL="https://github.com/ggml-org/llama.cpp.git"
readonly LLAMA_TAG="v0.8.0"  # 2026-10-28, supports qwen35 / qwen35moe
readonly ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly LLAMA_DIR="$ROOT/third_party/llama.cpp"

if ! command -v cmake >/dev/null; then
  echo "cmake not found (macOS: brew install cmake)" >&2
  exit 1
fi

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --quiet --depth 1 --branch "$LLAMA_TAG" "$LLAMA_URL" "$LLAMA_DIR"
fi

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$LLAMA_DIR/build" -j --target llama-cli llama-bench >/dev/null

echo "llama.cpp $LLAMA_TAG built (CPU-only): $LLAMA_DIR/build/bin/{llama-cli,llama-bench}"
