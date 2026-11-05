"""Use the Gated DeltaNet inference kernel for frozen lower decoder layers."""

from contextvars import ContextVar
from typing import Any, Callable, Optional

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5


MARKER = "_frozen_gdn_kernel"
_ACTIVE_MARKED_GDN: ContextVar[bool] = ContextVar(
    "frozen_gdn_kernel_active", default=False
)
_ORIGINAL_UPDATE: Optional[Callable[..., Any]] = None
_ORIGINAL_GDN_CALL: Optional[Callable[..., Any]] = None


def _marked_update(*args: Any, **kwargs: Any) -> Any:
    if _ORIGINAL_UPDATE is None:
        raise RuntimeError("frozen_gdn.install() has not patched the update")
    if _ACTIVE_MARKED_GDN.get():
        kwargs["use_kernel"] = True
    return _ORIGINAL_UPDATE(*args, **kwargs)


def _marked_gdn_call(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    if _ORIGINAL_GDN_CALL is None:
        raise RuntimeError("frozen_gdn.install() has not patched GatedDeltaNet")
    if not getattr(module, MARKER, False):
        return _ORIGINAL_GDN_CALL(module, *args, **kwargs)
    token = _ACTIVE_MARKED_GDN.set(True)
    try:
        return _ORIGINAL_GDN_CALL(module, *args, **kwargs)
    finally:
        _ACTIVE_MARKED_GDN.reset(token)


def _decoder_layers(model: nn.Module) -> list[nn.Module]:
    supported_models = (
        qwen3_5.Model,
        qwen3_5.TextModel,
        qwen3_5.Qwen3_5TextModel,
    )
    if not isinstance(model, supported_models):
        raise TypeError("frozen_gdn supports only qwen3_5-family models")
    return model.layers


def _has_trainable_parameters(module: nn.Module) -> bool:
    return bool(tree_flatten(module.trainable_parameters()))


def _patch_qwen3_5() -> None:
    global _ORIGINAL_UPDATE, _ORIGINAL_GDN_CALL

    if qwen3_5.gated_delta_update is not _marked_update:
        _ORIGINAL_UPDATE = qwen3_5.gated_delta_update
        qwen3_5.gated_delta_update = _marked_update
    if qwen3_5.GatedDeltaNet.__call__ is not _marked_gdn_call:
        _ORIGINAL_GDN_CALL = qwen3_5.GatedDeltaNet.__call__
        qwen3_5.GatedDeltaNet.__call__ = _marked_gdn_call


def install(model: nn.Module) -> None:
    """Mark frozen GDN layers below the first trainable decoder layer."""
    layers = _decoder_layers(model)
    if not _has_trainable_parameters(model):
        raise ValueError("frozen_gdn requires at least one trainable parameter")

    first_trainable = next(
        (
            index
            for index, layer in enumerate(layers)
            if _has_trainable_parameters(layer)
        ),
        len(layers),
    )
    _patch_qwen3_5()
    for index, layer in enumerate(layers):
        gdn = getattr(layer, "linear_attn", None)
        if isinstance(gdn, qwen3_5.GatedDeltaNet):
            setattr(gdn, MARKER, index < first_trainable)
