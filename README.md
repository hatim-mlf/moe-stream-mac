# Run a 122B MoE model on a 32 GB Mac

Two small inference engines that run **Qwen 3.5 / 3.6 mixture-of-experts models far larger than your RAM** on an
Apple-silicon Mac, by keeping only the small "trunk" of the model in memory and streaming the experts each
token needs from the SSD:

- **C engine** (`engine/src`): portable C99 + OpenMP + NEON, CPU only, float32 math. Verified token-for-token
  against MLX in float32.
- **MLX hybrid engine** (`engine/hybrid`): the same streaming idea, with the math on the GPU through
  [MLX](https://github.com/ml-explore/mlx). Much faster at reading long prompts. Bit-identical to plain MLX.

Both are driven from one terminal chat, `engine/cli/qchat.py`.

> Built in two days as an experiment, mostly by Claude (Anthropic) orchestrating a local Qwen model that wrote
> much of the C and Python under review. Everything that was tried, including what failed, is in
> [docs/REPORT.md](docs/REPORT.md).

## Results (MacBook Pro M1 Max, 32 GB, with Xcode, browsers etc. open)

| Model | Disk | Engine | RAM used | Prompt reading | Reply |
|---|---|---|---|---|---|
| Qwen3.5-122B-A10B **8-bit** | 131 GB | C engine | 5.5–6.7 GB | 1.2 tok/s | **1.5 tok/s** |
| Qwen3.5-122B-A10B 8-bit | 131 GB | MLX hybrid | ~2–8 GB + GPU | **10.4 tok/s** (240-token prompt) | 1.24 tok/s |
| Qwen3.6-35B-A3B 4-bit | 20 GB | MLX hybrid | ~2.5 GB | 10–18 tok/s | 6.3–7.4 tok/s |
| Qwen3.6-35B-A3B 4-bit | 20 GB | C engine | 1.4 GB | ~4 tok/s | 6.2 tok/s |
| *for comparison:* 35B 4-bit | 20 GB | plain MLX (whole model in RAM) | ~19 GB | ~660 tok/s | ~49 tok/s |

Plain MLX cannot load the 122B at all on this machine (131 GB of weights vs 32 GB of RAM).

**Correctness:** the C engine matches MLX float32 to a relative error of ~1e-6 (35B) and ~2e-6 (122B), with the
same top token at every position checked (including a 240-token prompt). The hybrid engine is bit-identical to
plain MLX in bf16.

**Limits, honestly:** the 122B needs ~3.8 GB of expert data per generated token. That data doesn't fit in RAM,
so it comes from the SSD: ~0.6 s/token is about the floor on this Mac for any engine. On long answers under
heavy memory pressure (swap nearly full), macOS starts swapping the engine's own memory and speed drops.

## Requirements

- Apple-silicon Mac (M1 or later), macOS 14+. Tested on an M1 Max, 32 GB.
- Free disk for the model (20 GB for the 35B 4-bit, 131 GB for the 122B 8-bit). A fast internal SSD matters.
- Xcode command-line tools, Homebrew `libomp` (C engine), Python 3.12+ (tested with 3.14).

## Install

```bash
git clone https://github.com/hatim-mlf/moe-stream-mac.git
cd moe-stream-mac
brew install libomp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make -C engine lib          # builds engine/build/libqengine.dylib (C engine)
make -C engine test         # unit tests, no model needed: should print ALL ... PASSED x4
```

> Use a Python that is NOT Anaconda's: MLX picks up Anaconda's MPI library and aborts. Homebrew's
> `python3` works.

## Download a model

Any MLX-quantized (affine 4- or 8-bit) **qwen3_5_moe** checkpoint works. A resumable, checksum-verified
downloader is included (it survives Wi-Fi drops and downloads 4 files at once):

```bash
.venv/bin/python tools/fetch_model.py mlx-community/Qwen3.6-35B-A3B-4bit      # 20 GB, good first test
.venv/bin/python tools/fetch_model.py mlx-community/Qwen3.5-122B-A10B-8bit     # 131 GB
```

Models go to `~/models/<name>`.

## Chat

```bash
.venv/bin/python engine/cli/qchat.py --model ~/models/Qwen3.5-122B-A10B-8bit                 # MLX hybrid (GPU)
.venv/bin/python engine/cli/qchat.py --model ~/models/Qwen3.5-122B-A10B-8bit --engine cpu    # C engine
```

| Option | Default | Meaning |
|---|---|---|
| `--engine gpu\|cpu` | gpu | MLX hybrid (fast prompts, bf16) or C engine (float32, exact) |
| `--threads N` | 6 | CPU threads (C engine). 6 of 10 cores keeps the Mac responsive; 8 is fastest |
| `--nice N` | 10 | scheduling priority; your other apps go first |
| `--ctx N` | 4096 | context length in tokens |
| `--temp X` | 0 | 0 = greedy (deterministic, the verified setting); otherwise top-k 20 / top-p 0.95 |
| `--think` | off | let the model reason first (shown dimmed) |
| `--system "..."` | | system prompt |
| `--prompt "..."` | | answer one prompt and exit |

In the chat: `/threads N`, `/think on|off`, `/temp X`, `/system TEXT`, `/reset`, `/stats`, `/help`, `/exit`.
Every reply ends with a line of stats: prompt and reply tok/s, context used, RAM, expert GB read per token.
Multi-turn chats only append tokens: earlier turns are never re-processed.

## Verify it yourself

```bash
make -C engine build/run_model
# C engine: whole model vs MLX float32 (loads MLX one layer at a time, so it fits in RAM)
.venv/bin/python engine/tests/model_check.py tokenize ~/models/Qwen3.6-35B-A3B-4bit engine/build/p.i32
OMP_NUM_THREADS=8 engine/build/run_model ~/models/Qwen3.6-35B-A3B-4bit engine/build/p.i32 12 engine/build/out.bin
.venv/bin/python engine/tests/model_check.py check ~/models/Qwen3.6-35B-A3B-4bit engine/build/out.bin
# -> WHOLE MODEL MATCHES MLX (engine generation = MLX greedy)

# MLX hybrid vs plain MLX (35B; needs ~20 GB free; it explains how to create its reference file first)
.venv/bin/python engine/hybrid/hybrid_vs_mlx.py ~/models/Qwen3.6-35B-A3B-4bit
```

Details: [docs/C_ENGINE.md](docs/C_ENGINE.md), [docs/MLX_ENGINE.md](docs/MLX_ENGINE.md).

## How it works (short)

Qwen 3.5/3.6 MoE layers: 3 of every 4 use linear attention (Gated DeltaNet), 1 uses full attention, and each
has 256 experts of which the router picks 8 per token, plus 1 shared expert. Only ~3% of the expert weights are
used for any one token. Both engines:

1. keep the **trunk** (attention, routers, shared experts, embeddings, output head) in memory;
2. after each layer's router picks its experts, **read only those** from the safetensors files with parallel
   `pread`s, in disk order;
3. compute, then move on to the next layer.

The C engine does everything on the CPU (OpenMP threads + a NEON 4/8-bit kernel). The hybrid hands the math to
MLX's Metal kernels and, for long prompts, reads each expert **once per prompt** while prefetching the next
layer in the background.

## Repository layout

```
engine/src, engine/include   C engine (C99). Parts adapted from kimi-k3-in-c (Apache-2.0), see engine/NOTICE
engine/tests                 C engine tests + MLX references
engine/hybrid                MLX hybrid engine + its tests and benchmarks
engine/cli/qchat.py          terminal chat for both engines
tools/fetch_model.py         resumable, verified Hugging Face downloader
tools/delegate.py, qwen_commit.py   how the local Qwen was used as a coding helper (see docs/REPORT.md)
scripts/local                optional: serve a local model with MLX, shell helper, benchmark (docs/LOCAL_HELPER.md)
docs/                        REPORT.md (full story), C_ENGINE.md, MLX_ENGINE.md, delegation log, briefs
results/                     every measurement, as metrics.json
```

## Contributing

Ideas, measurements on other Macs, and fixes are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). The most
valuable open problems are listed there (memory churn under real workloads, next-layer expert prediction,
more model families).

## License and credits

Dual-licensed under **MIT or Apache-2.0**, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)). Exception: files adapted from
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) by Fareed Khan (`engine/src/io/k3_st.*`,
`engine/src/io/k3_portable_io.h`, `engine/third_party/json.h`) remain **Apache-2.0 only**, see `engine/NOTICE`.
This project started from that engine's design (trunk in RAM, experts streamed from SSD) and rebuilt the model
code for Qwen 3.5/3.6.

GPU kernels and model definitions: [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm). Models: [Qwen](https://huggingface.co/Qwen), MLX conversions by
[mlx-community](https://huggingface.co/mlx-community).
