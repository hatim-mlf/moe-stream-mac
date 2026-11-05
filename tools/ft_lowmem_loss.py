"""Completion-only cross entropy with a bounded-vocabulary projection."""

from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import gather_last_axis


CHUNK = 512


def _logits_from_head(language_model: nn.Module, hidden: mx.array) -> mx.array:
    """Apply the same output projection used by mlx_lm's Qwen3.5 TextModel."""
    if language_model.args.tie_word_embeddings:
        return language_model.model.embed_tokens.as_linear(hidden)

    logits = language_model.lm_head(hidden)
    if language_model.sharding_group is not None:
        logits = gather_last_axis(logits, language_model.sharding_group)
    return logits


def _chunk_cross_entropy(
    language_model: nn.Module, hidden: mx.array, targets: mx.array
) -> mx.array:
    logits = _logits_from_head(language_model, hidden)
    return nn.losses.cross_entropy(logits, targets).astype(mx.float32).sum()


def _row_loss(
    language_model: nn.Module,
    hidden: mx.array,
    targets: mx.array,
    row: int,
    start: int,
    end: int,
) -> mx.array:
    total = mx.array(0.0, dtype=mx.float32)
    for chunk_start in range(start, end, CHUNK):
        chunk_end = min(chunk_start + CHUNK, end)
        chunk_hidden = hidden[row, chunk_start:chunk_end, :]
        chunk_targets = targets[row, chunk_start:chunk_end]
        # targets are closed over: checkpoint differentiates every positional argument.
        head_ce = partial(_chunk_cross_entropy, language_model, targets=chunk_targets)
        chunk_loss = mx.checkpoint(head_ce)(chunk_hidden)
        total = total + chunk_loss
    return total


def completion_only_loss(
    model: nn.Module, batch: mx.array, lengths: mx.array
) -> tuple[mx.array, mx.array]:
    """Return masked next-token CE and token count, matching trainer.default_loss."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    language_model = model.language_model
    hidden = language_model.model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ntoks = mask.sum()
    token_spans = lengths.tolist()

    ce_sum = mx.array(0.0, dtype=mx.float32)
    for row, (span_start, span_end) in enumerate(token_spans):
        start = max(int(span_start), 1) - 1
        end = min(int(span_end), targets.shape[1])
        ce_sum = ce_sum + _row_loss(language_model, hidden, targets, row, start, end)

    return ce_sum / ntoks, ntoks
