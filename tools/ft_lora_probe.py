#!/usr/bin/env python3
"""Run mlx_lm.lora while printing training seconds/step and MLX peak bytes."""
import time

import mlx.core as mx
from mlx_lm import lora


def timed_train(*args, **kwargs):
    steps = kwargs["args"].iters
    started = time.perf_counter()
    try:
        return _original_train(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        print(f"training_seconds={elapsed:.3f} training_steps={steps} s_per_step={elapsed / steps:.3f}")


_original_train = lora.train
lora.train = timed_train
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])  # else macOS compresses weights and stalls
import os
if os.environ.get("MLX_CACHE_LIMIT_GB"):
    mx.set_cache_limit(int(float(os.environ["MLX_CACHE_LIMIT_GB"]) * 2**30))
mx.reset_peak_memory()
lora.main()
mx.eval(mx.array(0))
peak = mx.get_peak_memory()
print(f"mlx_peak_memory_bytes={peak} mlx_peak_memory_gib={peak / 1024**3:.3f}")
