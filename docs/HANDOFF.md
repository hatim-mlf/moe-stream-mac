# Handoff (2026-11-06, Claude near weekly quota)

Running unattended (check logs in results/20261106T013000Z-night-chain/):
1. methods.sh: base Qwen3.6 with --guide checklist, --plan-first, guide+repair3 (m-*.json, m-*-paired.txt).
   repair3 done: 32.0% -> 39.3% pass@3 (paired CI -0.7..+17.3; ft_eval --paired refuses different retry caps:
   compute the paired bootstrap by hand as in the session).
2. q38-official.sh: waits for methods + bf16 fetch, converts official Qwen/Qwen3.8-27B to MLX 6-bit
   (~/models/Qwen3.8-27B-official-mlx-6bit), deletes bf16, evals 5-sample and repair3 (q38off-*).
3. Codex C32 (brief 48): DPO on the model's own passing-vs-failing pairs. Review, run tests on the Mac, then
   `scripts/dpo_round.sh`.

Results so far (heldout-v2, 5 samples T=0.6 vs base 32.0%): SFT on Codex refs 14.7% (worse); RFT scale 20/4
28.7%/30.0%; Qwen3.8-27B DavidAU 33.3% (5x slower); base + repair loop x3 39.3% (best).
Next order: finish methods (2,3) -> combine best methods -> DPO (4b) -> RFT/DPO on top of the repair loop.
Models: docs/MODELS.md. Memory: docs/finetune/MEMORY_MODEL.md. After reboot: sudo sysctl iogpu.wired_limit_mb=31744.
