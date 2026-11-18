# Brief 50: verify Strata's Apple-silicon claim on the M1 Max 32 GB

Status: planned (2026-11-18). Needs the Mac (Phase 3) and the user's OK for a >1 GB download.

## The claim

[Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT, v0.2.15, 16 Nov 2026) runs **one model family,
Qwen3.8-Flash-Next** (GGUF, 24,576 experts, 10 used per token), with its own Metal engine (not MLX, not llama.cpp).
On Macs whose memory can't hold the experts it streams them from the SSD (`docs/MAC.md`: it streams
"when the experts are more than 60% of Metal's share").

Self-reported numbers that matter to us (docs/MAC.md):

| Mac | Pack | Decode | Prompt |
|---|---|---|---|
| **M1 Max, 34 GB** (one line only) | Q2_0 | **7.9 tok/s** | 186 tok/s (9,940-token prompt) |
| M4 Mac mini 32 GB, streamed, no MTP | Q2_0 (37.6 GB) | 14.1 short / 13.5 story / 13.1 after 6K doc | 93.6 tok/s @ 4K |
| M4 Mac mini 32 GB | IQ3_XXS (47.0 GB) | 10.0 / 9.2 / 7.7 | 65.8 tok/s @ 4K |
| M4 Mac mini 32 GB | Gyro-S (58.5 GB file) | 9.14 story | 109 tok/s @ 4K; 18.6 GB held |
| M4 32 GB, llama.cpp CPU | Q2_0 / IQ3_XXS | 6.3 / 4.1 | 21.2 / 8.7 |

Caveats in their own docs: Mac support is "new", streaming is "measured at 32 GB only", model must be on the
internal SSD, Qwen3.6 does not run on the Mac engine, and a 0.2.15 bug shows "Drafting: MTP" when nothing drafts.

## What we check (in this order, stop at the first failure)

1. **Runs at all**: Q2_0 on our M1 Max 32 GB with normal apps open (Xcode, browsers, etc.; don't close them).
2. **Speed vs their M1 Max line**: decode on a short question, a story (~500 tokens), and after a ~6K-token
   document; prompt speed at 4K and ~10K tokens. Medians of 5. Record with MTP drafting on and off
   (`--no-mtp` or the equivalent flag; their M4 table is no-MTP, check which one the M1 Max line used).
3. **Is it correct?** Same Q2_0 GGUF under llama.cpp (`llama-cli`, greedy, temp 0): first 64 greedy tokens should
   match on 5 prompts, or drift only after a near-tie. Strata's own kernels are what we're testing, so a mismatch
   early on means its Metal math is wrong, not just quantization noise.
4. **Memory**: peak RSS + wired, swap used before/after, `vm_stat` pageins (shows how much SSD streaming happens).
5. **Baselines on the same Mac, same session**: llama.cpp Metal and CPU on the same GGUF (`llama-bench -p 4096 -n 128`).
   Next to our own numbers (35B 6.3–7.4 tok/s hybrid, 122B 8-bit 1.24–1.5 tok/s). Note: that's a
   different model and quantization, so it's context, not a head-to-head.
6. **How far can we go** (only if 1–3 pass): IQ3_XXS (47 GB), then Gyro-S (58.5 GB). Gyro-M (92 GB) and
   UD-Q4_K_XL (82.6 GB) need ≥96/128 GB per their table; try one only if IQ3_XXS still decodes ≥5 tok/s.
   Quality check for each step: 30-case code comparison (`tools/qwen38_code_compare.py add-model`) through
   Strata's OpenAI API at `http://127.0.0.1:8080/v1`, against the Qwen3.8-27B 4-bit baseline (11/30).

## Disk and downloads (needs approval)

- Q2_0: ~66 GB download (per Strata) + ~5 GB MTP draft + possibly a 34–40 GB experts copy, so **~110 GB on disk**.
- The Mac has ~252 GB free and the rule is one big model at a time: move or delete the 122B 8-bit (131 GB) first,
  or confirm it stays.
- Each later pack replaces the previous one.

## Install (source build, so we can read what it runs)

```bash
cd ~/src && git clone https://github.com/Niko1221/Strata && cd Strata
git checkout v0.2.15                 # pin; record the commit (c775728)
less setup.sh                        # read before running
./setup.sh                           # pick Q2_0; put the model on the internal SSD
strata launch                        # server + chat page at http://127.0.0.1:8080/
```

Don't use the `curl | sh` installer. Keep the clone outside this repo (third-party, never committed).

## Output

`results/<UTC>-strata-m1max/metrics.json`: Strata version + commit, pack, flags, for each prompt/condition
median decode and prompt tok/s, peak memory, swap, pageins, greedy-match count vs llama.cpp, and the llama.cpp
baseline rows. No raw logs, no weights.
