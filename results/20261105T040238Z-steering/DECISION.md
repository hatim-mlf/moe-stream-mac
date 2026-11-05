# Steering config frozen before held-out evaluation (2026-11-05)
Dev (15 tasks, base 11/15): L12 11/11/12, L20 12/12/12, L28 11/13/11 at alpha 0.05/0.1/0.2 (alpha in units of the
layer's residual norm). Chosen: layer 20, alpha 0.1 (centre of the only stable plateau; the L28 0.1 peak is flanked
by base-level scores and looks like noise). Held-out: heldout-v2 first attempt, plus 5 random directions with the
same norm (seeds 1-5) and the public mini. Selection made without looking at any held-out result.
