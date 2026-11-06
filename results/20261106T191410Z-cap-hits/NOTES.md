# Token-cap hits on Qwen3.6-35B-A3B base (heldout-v2, 30 tasks x 5 samples, T=0.6, seed 7)

150 model calls: cap hits (finish_reason=length) 0/150 = 0.0%; answers with repetition score > 0.5: 0%.
Completion tokens: mean 523, median 455, p90 915 (cap 8,192). Pass rate this run 30.7% (46/150 runs;
the earlier identical-settings run was 32.0%, i.e. sampling noise of ~1-2 points).
Decision: the base model does not loop on our coding eval, so the MIT "remove the repetition direction" weight edit
(docs/research/weight-surgery-claim.md; their 0.8B hit the cap on 93% of MATH-500) has nothing to fix here. Dropped.
