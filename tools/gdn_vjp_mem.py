"""Isolated GDN backward memory probe using the model's real head sizes."""

import argparse
import os
import sys
from collections.abc import Callable
from functools import partial

import mlx.core as mx
from mlx_lm.models import gated_delta

import gdn_vjp
import mem_guard


BATCH = 1
KEY_HEADS = 16
VALUE_HEADS = 32
HEAD_DIM = 128
Array = mx.array


def _inputs(timesteps: int) -> tuple[Array, Array, Array, Array, Array, Array]:
    mx.random.seed(0)
    q = mx.random.normal(
        (BATCH, timesteps, KEY_HEADS, HEAD_DIM)
    ).astype(mx.bfloat16)
    k = mx.random.normal(
        (BATCH, timesteps, KEY_HEADS, HEAD_DIM)
    ).astype(mx.bfloat16)
    q, k = gated_delta.normalize_qk(
        q, k, inv_scale=HEAD_DIM**-0.5, eps=1e-6
    )
    v = mx.random.normal(
        (BATCH, timesteps, VALUE_HEADS, HEAD_DIM)
    ).astype(mx.bfloat16)
    g = mx.random.uniform(0.9, 1.0, (BATCH, timesteps, VALUE_HEADS))
    beta = mx.random.uniform(0.05, 0.95, (BATCH, timesteps, VALUE_HEADS))
    state = mx.zeros(
        (BATCH, VALUE_HEADS, HEAD_DIM, HEAD_DIM), dtype=mx.float32
    )
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def _loss(
    fn: Callable[..., tuple[Array, Array]],
    state: Array,
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
) -> Array:
    y, final_state = fn(q, k, v, g, beta, state, None)
    return y.astype(mx.float32).sum() + final_state.sum()


def main(argv: list[str] | None = None) -> int:
    _total_memory = mem_guard.total_memory_bytes()
    _headroom = mem_guard.headroom_bytes_from_env(os.environ)
    print(mem_guard.startup_limit_message(_total_memory, _headroom), flush=True)
    mem_guard.start(total_bytes=_total_memory, headroom_bytes=_headroom)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timesteps", type=int)
    parser.add_argument(
        "--impl", choices=("ops", "chunked", "staged", "kernel"),
        default="chunked"
    )
    args = parser.parse_args(argv)
    if args.timesteps <= 0:
        parser.error("timesteps must be positive")
    q, k, v, g, beta, state = _inputs(args.timesteps)
    if args.impl == "ops":
        fn = gated_delta.gated_delta_ops
    else:
        fn = partial(
            gdn_vjp.gated_delta_chunked,
            chunk=gdn_vjp.DEFAULT_CHUNK,
            impl=args.impl,
        )
    loss = partial(_loss, fn, state)
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    value, grads = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(
        q, k, v, g, beta
    )
    mx.eval(value, grads)
    peak = mx.get_peak_memory() - base
    mib = peak / 2**20
    print(
        f"T={args.timesteps} impl={args.impl}: backward_peak={mib:.0f} MiB "
        f"MiB_per_token={mib / args.timesteps:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
