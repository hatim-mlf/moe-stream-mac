#!/usr/bin/env python3
"""Run mlx_lm.lora with completion-only loss and memory reporting."""

import os
import time
from typing import Any

import mlx.core as mx
from mlx_lm import lora

from ft_lowmem_loss import completion_only_loss
import mem_guard


_original_train = lora.train
_original_evaluate = lora.evaluate


def timed_train(*args: Any, **kwargs: Any) -> Any:
    steps = kwargs["args"].iters
    started = time.perf_counter()
    kwargs["loss"] = completion_only_loss
    # The loss uses concrete row boundaries to slice variable-length completions.
    mx.disable_compile()
    try:
        return _original_train(*args, **kwargs)
    finally:
        mx.enable_compile()
        elapsed = time.perf_counter() - started
        print(
            f"training_seconds={elapsed:.3f} training_steps={steps} "
            f"s_per_step={elapsed / steps:.3f}"
        )


def lowmem_evaluate(*args: Any, **kwargs: Any) -> Any:
    kwargs["loss"] = completion_only_loss
    return _original_evaluate(*args, **kwargs)


lora.train = timed_train
lora.evaluate = lowmem_evaluate
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
if os.environ.get("MLX_CACHE_LIMIT_GB"):
    mx.set_cache_limit(int(float(os.environ["MLX_CACHE_LIMIT_GB"]) * 2**30))
mx.reset_peak_memory()
_total_memory = mem_guard.total_memory_bytes()
_headroom = mem_guard.headroom_bytes_from_env(os.environ)
_limit = _total_memory - _headroom
print(
    f"mem_guard: limit {_limit / 2**30:.2f} GiB "
    f"(total {_total_memory / 2**30:.2f}, headroom {_headroom / 2**30:.2f})",
    flush=True,
)
mem_guard.start(total_bytes=_total_memory, headroom_bytes=_headroom)
lora.main()
mx.eval(mx.array(0))
peak = mx.get_peak_memory()
print(f"mlx_peak_memory_bytes={peak} mlx_peak_memory_gib={peak / 1024**3:.3f}")
