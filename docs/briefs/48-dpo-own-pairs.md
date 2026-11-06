# Brief 48: preference training from the model's own passing vs failing answers (Codex)

<task>
Step 4b of docs/finetune (method plan). `results/20261105T213000Z-rft-base-k8/rft-samples.jsonl` holds 2,000 samples
of base Qwen3.6-35B-A3B on 250 training tasks (fields: task_id, sample, passed, code, failure_tail); 118 tasks
have both passing and failing samples. Build:
1. `tools/ft_pairs.py --samples ... --tasks data/tasks-v1..v5 --out data/finetune/pairs-v1`: for each task with
   ≥1 pass and ≥1 fail, up to 2 (chosen=passing code, rejected=failing code) pairs, prompt built exactly like
   tools/ft_rft.py / ft_tasks_to_sft.py, keep the v5 task split (valid tasks → valid.jsonl), leak check.
2. `tools/ft_dpo.py`: DPO loss (Rafailov et al. 2023, beta default 0.1) for mlx_lm LoRA training, used by
   `tools/ft_lora_lowmem.py` when env `DPO=1`. Reference log-probs = the same model with LoRA disabled
   (precompute once per pair before training, or compute with adapters scaled to 0), completion tokens only
   (reuse tools/ft_lowmem_loss.py's completion slicing). Batch = one pair. Keep FROZEN_GDN_KERNEL compatible.
3. `scripts/dpo_round.sh` modelled on scripts/rft_round.sh: train 1-layer LoRA scale 4 with DPO on pairs-v1,
   pick lowest-val checkpoint, 5-sample paired eval vs results/20261105T201500Z-eval-resolution/base-s5-t06.json.
   Do not run it (the lead runs it on the Mac).
</task>
<tests>
tools/test_ft_dpo.py with the tiny model from tools/test_gdn_checkpoint.py: DPO loss equals a hand-computed value
on fixed log-probs; gradient pushes chosen log-prob up / rejected down after one step; pairs builder split and
leak check. You have no Metal: write tests, `py_compile`, `bash -n`; the lead runs pytest.
</tests>
<constraints>No commit. Do not modify ft_eval.py, data/tasks-v*. Functions < 50 lines, no silent excepts.</constraints>
