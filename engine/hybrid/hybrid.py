"""Hybrid engine: model trunk on the GPU (MLX), MoE experts streamed from SSD on demand.

The trunk (attention / linear-attention layers, routers, shared experts, embeddings, lm_head) is loaded
into unified memory and runs on the GPU. Each layer's routed experts stay on disk: for every forward pass
the router's choices for ALL tokens in the batch are collected, each needed expert is read ONCE (parallel
preadv, engine/hybrid/expert_reader.py) and MLX's own gather_qmm kernels run them on the GPU.
A batched prompt therefore reads each expert at most once, however long the prompt is.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwiGLU, _gather_sort, _scatter_unsort

from expert_reader import ExpertReader

PROJS = ("gate_proj", "up_proj", "down_proj")
SORT_MIN_INDICES = 64          # same threshold as mlx_lm SwitchGLU
PREFETCH_MIN_TOKENS = 32       # batches this long use whole layers, read ahead in disk order


class StreamStats:
    def __init__(self) -> None:
        self.read_s = 0.0
        self.bytes = 0
        self.experts = 0


class LayerPrefetcher:
    """Reads a WHOLE layer's experts (all E, disk order) in the background while the GPU computes the
    previous layer. Used for long batches (prompts), where nearly every expert is needed anyway."""

    def __init__(self, reader: ExpertReader, stats: StreamStats):
        self._reader, self._stats = reader, stats
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending: dict = {}

    def _read_layer(self, layer: int) -> dict:
        t0 = time.perf_counter()
        experts = np.arange(self._reader.num_experts(layer, PROJS[0]))
        out = {}
        for proj in PROJS:                                   # numpy only: MLX arrays must be made on the main thread
            out[proj] = self._reader.read_sorted(layer, proj, experts)
            self._stats.bytes += sum(a.nbytes for a in out[proj].values())
        self._stats.read_s += time.perf_counter() - t0      # background time (overlaps GPU work)
        self._stats.experts += len(experts)
        return out

    def request(self, layer: int) -> None:
        if layer not in self._pending and 0 <= layer < self._reader.n_layers:
            self._pending[layer] = self._pool.submit(self._read_layer, layer)

    def take(self, layer: int) -> dict:
        self.request(layer)
        raw = self._pending.pop(layer).result()
        return {proj: (mx.array(r["weight"]), mx.array(r["scales"]).view(mx.bfloat16), mx.array(r["biases"]).view(mx.bfloat16))
                for proj, r in raw.items()}

    def clear(self) -> None:
        for fut in self._pending.values():
            fut.result()
        self._pending.clear()


class StreamingSwitchGLU(nn.Module):
    """Drop-in replacement for mlx_lm SwitchGLU whose expert weights live on disk."""

    def __init__(self, reader: ExpertReader, layer: int, group_size: int, bits: dict, stats: StreamStats,
                 prefetcher: LayerPrefetcher):
        super().__init__()
        self._reader, self._layer, self._gs, self._bits, self._stats = reader, layer, group_size, bits, stats
        self._prefetcher = prefetcher
        self._act = SwiGLU()

    def _load(self, experts: np.ndarray) -> dict:
        t0 = time.perf_counter()
        out = {}
        for proj in PROJS:
            raw = self._reader.read_sorted(self._layer, proj, experts)
            out[proj] = (mx.array(raw["weight"]), mx.array(raw["scales"]).view(mx.bfloat16),
                         mx.array(raw["biases"]).view(mx.bfloat16))
            self._stats.bytes += sum(a.nbytes for a in raw.values())
        self._stats.read_s += time.perf_counter() - t0
        self._stats.experts += len(experts)
        return out

    def _proj(self, x: mx.array, proj: str, w: dict, idx: mx.array, sorted_idx: bool) -> mx.array:
        q, s, b = w[proj]
        return mx.gather_qmm(x, q, s, b, rhs_indices=idx, transpose=True,
                             group_size=self._gs, bits=self._bits[proj], sorted_indices=sorted_idx)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        ids = np.array(indices)                                # waits for the router (one sync per layer)
        if ids.shape[-2] >= PREFETCH_MIN_TOKENS:               # long batch: whole layer, already read ahead
            w = self._prefetcher.take(self._layer)
            self._prefetcher.request(self._layer + 1)          # next layer reads while the GPU computes this one
            idx = mx.array(ids.astype(np.uint32))
        else:                                                  # decode: only the experts the router picked
            uniq, inv = np.unique(ids, return_inverse=True)
            w = self._load(uniq)
            idx = mx.array(inv.reshape(ids.shape).astype(np.uint32))
        x = mx.expand_dims(x, (-2, -3))
        do_sort = idx.size >= SORT_MIN_INDICES
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, idx)
        up = self._proj(x, "up_proj", w, idx, do_sort)
        gate = self._proj(x, "gate_proj", w, idx, do_sort)
        y = self._proj(self._act(up, gate), "down_proj", w, idx, do_sort)
        if do_sort:
            y = _scatter_unsort(y, inv_order, ids.shape)
        return y.squeeze(-2)


class HybridModel:
    """Load a qwen3_5_moe MLX checkpoint with experts streamed from disk. API mirrors the C engine."""

    def __init__(self, model_dir: Path, read_threads: int = 8):
        self.model, self.tokenizer = load(str(model_dir), lazy=True)
        self.reader = ExpertReader(model_dir, threads=read_threads)
        self.stats = StreamStats()
        self.prefetcher = LayerPrefetcher(self.reader, self.stats)
        for i, layer in enumerate(self.model.language_model.model.layers):
            old = layer.mlp.switch_mlp
            bits = {p: getattr(old, p).bits for p in PROJS}
            layer.mlp.switch_mlp = StreamingSwitchGLU(self.reader, i, old.gate_proj.group_size, bits, self.stats,
                                                      self.prefetcher)
            del old                                            # lazy expert arrays are dropped unread
        mx.eval(self.model.parameters())                       # loads the trunk only
        self.reset()

    def reset(self) -> None:
        self.cache = make_prompt_cache(self.model)
        self.position = 0

    def forward(self, tokens: list[int]) -> mx.array:
        """Feed tokens (a whole prompt or one token); return the logits after the LAST one (float32)."""
        if len(tokens) >= PREFETCH_MIN_TOKENS:
            self.prefetcher.request(0)                          # start reading layer 0 during embedding/attention
        logits = self.model(mx.array(tokens)[None], cache=self.cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        self.position += len(tokens)
        return logits

    def forward_all(self, tokens: list[int]) -> mx.array:
        """Like forward but returns logits for EVERY position [len(tokens), vocab] (for verification)."""
        logits = self.model(mx.array(tokens)[None], cache=self.cache)[0].astype(mx.float32)
        mx.eval(logits)
        self.position += len(tokens)
        return logits
