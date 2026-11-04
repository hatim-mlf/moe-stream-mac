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

from expert_cache import ExpertCache
from expert_reader import ExpertReader
from predictive import PredictivePrefetcher

PROJS = ("gate_proj", "up_proj", "down_proj")
SORT_MIN_INDICES = 64          # same threshold as mlx_lm SwitchGLU
PREFETCH_MIN_TOKENS = 32       # batches this long use whole layers, read ahead in disk order
PREDICT_EXPERTS = 12           # next-layer experts read ahead (measured recall: 8 -> 83%, 12 -> 94%, 16 -> 97%)
MLX_CACHE_LIMIT_BYTES = 1 << 30  # streamed expert buffers are freed every layer; MLX's default cache would hoard
                               # them (up to most of RAM), squeezing the file cache and making the GPU swap


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
                 prefetcher: LayerPrefetcher, cache: ExpertCache):
        super().__init__()
        self._reader, self._layer, self._gs, self._bits, self._stats = reader, layer, group_size, bits, stats
        self._prefetcher, self._cache = prefetcher, cache
        self._next_gate = None          # router of layer + 1, set by HybridModel (None for the last layer)
        self._predictive = None
        self._act = SwiGLU()

    def _read_missing(self, missing: np.ndarray) -> dict:
        """Read the missing experts (sorted ids) from disk; return {expert: {proj: (w, s, b)}} as separate MLX arrays."""
        t0 = time.perf_counter()
        per_expert = {int(e): {} for e in missing}
        for proj in PROJS:
            raw = self._reader.read_sorted(self._layer, proj, missing)
            self._stats.bytes += sum(a.nbytes for a in raw.values())
            for i, e in enumerate(missing):     # one array per expert, so the cache can evict them independently
                per_expert[int(e)][proj] = (mx.array(raw["weight"][i]),
                                            mx.array(raw["scales"][i]).view(mx.bfloat16),
                                            mx.array(raw["biases"][i]).view(mx.bfloat16))
        self._stats.read_s += time.perf_counter() - t0
        self._stats.experts += len(missing)
        return per_expert

    def _experts(self, experts: np.ndarray) -> dict:
        """{expert: {proj: (w, s, b)}} for sorted ids: cache hits stay on the GPU, misses are read from disk."""
        found = {int(e): self._cache.get((self._layer, int(e))) for e in experts}
        missing = np.array([e for e, v in found.items() if v is None], dtype=np.int64)
        if len(missing):
            for e, v in self._read_missing(missing).items():
                found[e] = v
                self._cache.put((self._layer, e), v, sum(a.nbytes for t in v.values() for a in t))
        return found

    def _load(self, experts: np.ndarray) -> dict:
        """Stacked {proj: (w, s, b)} for gather_qmm (small batches only: stacking copies the weights)."""
        found = self._experts(experts)
        order = [found[int(e)] for e in experts]
        return {proj: tuple(mx.stack([v[proj][k] for v in order]) for k in range(3)) for proj in PROJS}

    def _qmm(self, x: mx.array, proj: str, w: tuple) -> mx.array:
        return mx.quantized_matmul(x, w[0], w[1], w[2], transpose=True, group_size=self._gs, bits=self._bits[proj])

    def _predicted_experts(self, ids: np.ndarray) -> dict:
        """Needed experts via the predictive prefetcher (read ahead during the previous layer), as MLX arrays."""
        t0 = time.perf_counter()
        rows = self._predictive.take(self._layer, np.unique(ids))
        self._stats.read_s += time.perf_counter() - t0       # only the time the GPU path actually waited
        out = {}
        for e, per_proj in rows.items():
            out[e] = {proj: (mx.array(w), mx.array(sc).view(mx.bfloat16), mx.array(b).view(mx.bfloat16))
                      for proj, (w, sc, b) in per_proj.items()}
            self._stats.bytes += sum(a.nbytes for t in per_proj.values() for a in t)
        return out

    def _single_token(self, x: mx.array, ids: np.ndarray) -> mx.array:
        """One token: run each selected expert straight from its own arrays, no weight copies."""
        found = self._predicted_experts(ids) if self._predictive is not None else self._experts(np.unique(ids))
        x2 = x.reshape(1, -1)
        outs = []
        for e in ids.reshape(-1):
            w = found[int(e)]
            h = self._act(self._qmm(x2, "up_proj", w["up_proj"]), self._qmm(x2, "gate_proj", w["gate_proj"]))
            outs.append(self._qmm(h, "down_proj", w["down_proj"]))
        return mx.concatenate(outs).reshape(*ids.shape, -1)

    def _proj(self, x: mx.array, proj: str, w: dict, idx: mx.array, sorted_idx: bool) -> mx.array:
        q, s, b = w[proj]
        return mx.gather_qmm(x, q, s, b, rhs_indices=idx, transpose=True,
                             group_size=self._gs, bits=self._bits[proj], sorted_indices=sorted_idx)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        single = indices.size // indices.shape[-1] == 1
        if single and self._predictive is not None and self._next_gate is not None:
            nxt = self._next_gate(x)                           # next layer's router on THIS layer's input
            mx.eval(indices, nxt)                              # same sync as before: no extra GPU round trip
            guess = np.argpartition(np.array(nxt.astype(mx.float32)).reshape(-1), -PREDICT_EXPERTS)[-PREDICT_EXPERTS:]
            self._predictive.request(self._layer + 1, guess)   # read layer L+1 while the GPU works on layer L
        ids = np.array(indices)                                # waits for the router (one sync per layer)
        if single:                                             # a single token (generation)
            return self._single_token(x, ids)
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

    def __init__(self, model_dir: Path, read_threads: int = 8, cache_gb: float = 0.0, predict: bool = False):
        # predict off by default: 92% hit rate, but fresh read buffers per layer churn ~10 GB/token and macOS
        # compresses the GPU trunk (3 s/token vs 0.82). Re-enable once read buffers are reused.  # GPU cache off: it steals RAM from the macOS file cache (measured 0.81 vs 5.5 s/token)
        mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        self.model, self.tokenizer = load(str(model_dir), lazy=True)
        self.reader = ExpertReader(model_dir, threads=read_threads)
        self.stats = StreamStats()
        self.prefetcher = LayerPrefetcher(self.reader, self.stats)
        self.expert_cache = ExpertCache(int(cache_gb * 1e9))   # hot experts stay resident in GPU memory
        self.predictive = PredictivePrefetcher(self.reader, PROJS) if predict else None
        for i, layer in enumerate(self.model.language_model.model.layers):
            old = layer.mlp.switch_mlp
            bits = {p: getattr(old, p).bits for p in PROJS}
            layer.mlp.switch_mlp = StreamingSwitchGLU(self.reader, i, old.gate_proj.group_size, bits, self.stats,
                                                      self.prefetcher, self.expert_cache)
            del old                                            # lazy expert arrays are dropped unread
        layers = self.model.language_model.model.layers
        for i, layer in enumerate(layers):                     # wire each layer to the next layer's router
            layer.mlp.switch_mlp._predictive = self.predictive
            layer.mlp.switch_mlp._next_gate = layers[i + 1].mlp.gate if i + 1 < len(layers) else None
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
