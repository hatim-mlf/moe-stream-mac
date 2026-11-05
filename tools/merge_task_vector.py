#!/usr/bin/env python3
"""Memory-bounded merger for MLX affine-quantized safetensors checkpoints.

For each logical tensor, this computes tau = dequant(finetune) - dequant(base)
in float32, then merges base + lambda * tau and requantizes with the base
tensor's own affine bits and group size. MLX affine groups use a signed scale
and choose the endpoint with the larger absolute value as the exact edge:

    scale = max((max(group) - min(group)) / (2**bits - 1), eps)
    if abs(min(group)) > abs(max(group)):
        edge = min(group); scale stays positive
    else:
        edge = max(group); scale is negated
    q0 = round(edge / scale)
    scale = edge / q0 if q0 != 0 else scale
    bias = edge if q0 != 0 else 0
    q = clip(round((w - bias) / scale), 0, 2**bits - 1)
    dequant(w) = scale * q + bias

The packed uint32 values are low-bits-first. Constant groups use scale 1 and
q=0, preserving the constant value through bias. TIES for one task vector is
magnitude trimming only: retain the largest |tau| entries and zero the rest,
with no rescaling of retained entries. DARE randomly drops delta entries and
rescales survivors by 1/(1-p).

Safetensors headers are read up front, while payloads are memory-mapped and
visited in bounded row chunks. Stacked switch-expert weights are chunked along
their leading expert axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


GIB = 1024**3
DEFAULT_CHUNK_ELEMENTS = 1_048_576
RAW_COPY_CHUNK = 8 * 1024 * 1024

# Safetensors dtype names mapped to their on-disk NumPy interpretation.
# BF16 is represented as uint16 here and converted with explicit bit logic so
# the merger does not need a platform-specific bfloat16 NumPy extension.
RAW_DTYPES: dict[str, np.dtype[Any]] = {
    "BOOL": np.dtype(np.uint8),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "U16": np.dtype(np.uint16),
    "I16": np.dtype(np.int16),
    "U32": np.dtype(np.uint32),
    "I32": np.dtype(np.int32),
    "U64": np.dtype(np.uint64),
    "I64": np.dtype(np.int64),
    "F16": np.dtype(np.float16),
    "BF16": np.dtype(np.uint16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str
    data_start: int
    data_end: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start


@dataclass(frozen=True)
class TensorInfo:
    """One logical tensor, backed by either one dense key or an MLX triple."""

    name: str
    shape: tuple[int, ...]
    quantized: bool
    bits: int | None
    group_size: int | None
    weight_key: str
    scales_key: str | None = None
    biases_key: str | None = None

    @property
    def component_keys(self) -> tuple[str, ...]:
        if self.quantized:
            assert self.scales_key is not None and self.biases_key is not None
            return self.weight_key, self.scales_key, self.biases_key
        return (self.weight_key,)


@dataclass
class OutputShard:
    handle: Any
    header_bytes: int

    def seek(self, offset: int) -> int:
        return self.handle.seek(offset)

    def write(self, value: bytes) -> int:
        return self.handle.write(value)

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class Checkpoint:
    """Header inventory plus lazy memory maps for one safetensors directory."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        config_path = self.directory / "config.json"
        index_path = self.directory / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            raise ValueError(
                f"{self.directory} must contain config.json and model.safetensors.index.json"
            )
        self.config: dict[str, Any] = json.loads(config_path.read_text())
        self.index: dict[str, Any] = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = self.index.get("weight_map", {})
        if not self.weight_map:
            raise ValueError(f"No weight_map entries in {index_path}")
        self.headers: dict[str, dict[str, Any]] = {}
        self.header_bytes: dict[str, bytes] = {}
        self.entries: dict[str, TensorEntry] = {}
        self._mapped_shards: dict[str, np.memmap] = {}
        self._read_handles: dict[str, Any] = {}
        self._write_handles: dict[str, Any] = {}
        self._read_headers()
        self.logical_tensors = self._make_logical_inventory()
        self.shard_size = sum(
            (self.directory / shard).stat().st_size
            for shard in sorted(set(self.weight_map.values()))
        )

    def _read_headers(self) -> None:
        shard_names = sorted(set(self.weight_map.values()))
        for shard in shard_names:
            path = self.directory / shard
            if not path.is_file():
                raise ValueError(f"Missing safetensors shard: {path}")
            with path.open("rb") as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Truncated safetensors header: {path}")
                header_length = struct.unpack("<Q", prefix)[0]
                raw = f.read(header_length)
                if len(raw) != header_length:
                    raise ValueError(f"Truncated safetensors header JSON: {path}")
            try:
                header = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid safetensors header in {path}: {exc}") from exc
            self.headers[shard] = header
            self.header_bytes[shard] = prefix + raw
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                if key not in self.weight_map:
                    raise ValueError(f"Header key {key!r} is absent from the index for {path}")
                if self.weight_map[key] != shard:
                    raise ValueError(f"Index assigns {key!r} to a different shard than {path}")
                try:
                    dtype = str(value["dtype"])
                    shape = tuple(int(x) for x in value["shape"])
                    start, end = (int(x) for x in value["data_offsets"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Malformed tensor header for {key!r} in {path}") from exc
                if dtype not in RAW_DTYPES:
                    raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {key}")
                if start < 0 or end < start:
                    raise ValueError(f"Invalid data offsets for {key!r} in {path}")
                expected = math.prod(shape) * RAW_DTYPES[dtype].itemsize
                if end - start != expected:
                    raise ValueError(
                        f"Header byte count for {key!r} is {end - start}, expected {expected}"
                    )
                self.entries[key] = TensorEntry(key, shape, dtype, shard, start, end)
        missing_from_headers = set(self.weight_map) - set(self.entries)
        if missing_from_headers:
            sample = sorted(missing_from_headers)[:8]
            raise ValueError(f"Index keys missing from shard headers ({len(missing_from_headers)}): {sample}")

    def _quant_config(self, weight_key: str) -> tuple[int, int, str]:
        stem = weight_key[: -len(".weight")]
        config = self.config.get("quantization", {})
        if not isinstance(config, dict):
            raise ValueError(f"{self.directory}: config quantization must be an object")
        default_bits = config.get("bits")
        default_group = config.get("group_size")
        override = config.get(stem, {})
        if not isinstance(override, dict):
            override = {}
        bits = override.get("bits", default_bits)
        group_size = override.get("group_size", default_group)
        mode = override.get("mode", config.get("mode", "affine"))
        if bits is None or group_size is None:
            raise ValueError(f"Missing bits/group_size quantization config for {weight_key}")
        bits, group_size = int(bits), int(group_size)
        if mode != "affine":
            raise ValueError(f"Only MLX affine quantization is supported; {weight_key} uses {mode}")
        if bits <= 0 or bits > 16 or 32 % bits != 0 or group_size <= 0:
            raise ValueError(f"Unsupported affine quantization settings for {weight_key}: {bits}/{group_size}")
        return bits, group_size, str(mode)

    def _make_logical_inventory(self) -> dict[str, TensorInfo]:
        inventory: dict[str, TensorInfo] = {}
        component_keys: set[str] = set()
        for key in sorted(self.weight_map):
            if not key.endswith(".weight"):
                continue
            stem = key[: -len(".weight")]
            scales_key, biases_key = stem + ".scales", stem + ".biases"
            has_scales = scales_key in self.weight_map
            has_biases = biases_key in self.weight_map
            if has_scales != has_biases:
                raise ValueError(f"Incomplete MLX quantization triple for {key}")
            if has_scales:
                weight_entry = self.entries[key]
                scale_entry = self.entries[scales_key]
                bias_entry = self.entries[biases_key]
                if weight_entry.dtype != "U32":
                    raise ValueError(f"Packed MLX weight {key} must have U32 dtype")
                if scale_entry.dtype not in ("F16", "BF16", "F32") or bias_entry.dtype not in (
                    "F16",
                    "BF16",
                    "F32",
                ):
                    raise ValueError(f"MLX scales and biases for {key} must be floating point")
                bits, group_size, _ = self._quant_config(key)
                logical_shape = self._logical_shape(weight_entry, scale_entry, bias_entry, bits, group_size)
                inventory[key] = TensorInfo(
                    key,
                    logical_shape,
                    True,
                    bits,
                    group_size,
                    key,
                    scales_key,
                    biases_key,
                )
                component_keys.update((key, scales_key, biases_key))
            else:
                entry = self.entries[key]
                inventory[key] = TensorInfo(key, entry.shape, False, None, None, key)
                component_keys.add(key)
        for key in sorted(self.weight_map):
            if key in component_keys:
                continue
            if key.endswith((".scales", ".biases")):
                raise ValueError(f"Unpaired MLX quantization metadata tensor: {key}")
            entry = self.entries[key]
            inventory[key] = TensorInfo(key, entry.shape, False, None, None, key)
        return inventory

    @staticmethod
    def _logical_shape(
        weight: TensorEntry,
        scales: TensorEntry,
        biases: TensorEntry,
        bits: int,
        group_size: int,
    ) -> tuple[int, ...]:
        if weight.shape != scales.shape and weight.shape[:-1] != scales.shape[:-1]:
            raise ValueError(f"Packed weight/scales leading dimensions differ for {weight.name}")
        if scales.shape != biases.shape:
            raise ValueError(f"Scale/bias shapes differ for {weight.name}")
        if len(weight.shape) != len(scales.shape) or not weight.shape:
            raise ValueError(f"Packed weight/scales ranks differ for {weight.name}")
        logical_last = scales.shape[-1] * group_size
        packed_factor = logical_last // weight.shape[-1] if weight.shape[-1] else 0
        if (
            logical_last % weight.shape[-1] != 0
            or packed_factor != 32 // bits
            or weight.shape[-1] * packed_factor != logical_last
        ):
            raise ValueError(
                f"Packed shape, groups, or configured bits disagree for {weight.name}: "
                f"weight={weight.shape}, scales={scales.shape}, bits={bits}, group={group_size}"
            )
        # The stored scales have one entry for each quantization group along
        # the final axis; all leading dimensions are kept intact.
        if weight.shape[:-1] != scales.shape[:-1]:
            raise ValueError(f"Packed weight/scales leading dimensions differ for {weight.name}")
        return weight.shape[:-1] + (logical_last,)

    def array(self, key: str) -> np.ndarray[Any, Any]:
        entry = self.entries[key]
        shard_map = self._mapped_shards.get(entry.shard)
        if shard_map is None:
            shard_map = np.memmap(self.directory / entry.shard, dtype=np.uint8, mode="r")
            self._mapped_shards[entry.shard] = shard_map
        data_base = len(self.header_bytes[entry.shard])
        dtype = RAW_DTYPES[entry.dtype]
        return np.ndarray(
            shape=entry.shape,
            dtype=dtype,
            buffer=shard_map,
            offset=data_base + entry.data_start,
            order="C",
        )

    def read_component(self, key: str, first_slice: slice | None = None) -> np.ndarray[Any, Any]:
        arr = self.array(key)
        if first_slice is not None and arr.ndim:
            arr = arr[first_slice]
        return arr

    def read_float(self, key: str, first_slice: slice | None = None) -> np.ndarray[Any, Any]:
        entry = self.entries[key]
        arr = self.read_component(key, first_slice)
        if entry.dtype == "BF16":
            bits = np.asarray(arr, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
            return bits.view(np.float32)
        if entry.dtype in ("F16", "F32", "F64"):
            return np.asarray(arr, dtype=np.float32)
        raise ValueError(f"Tensor {key} is not a floating-point tensor ({entry.dtype})")

    def read_handle(self, shard: str) -> Any:
        if shard not in self._read_handles:
            self._read_handles[shard] = (self.directory / shard).open("rb")
        return self._read_handles[shard]

    def close(self) -> None:
        for handle in self._read_handles.values():
            handle.close()
        self._read_handles.clear()
        self._mapped_shards.clear()


def pack_qvalues(qvalues: np.ndarray[Any, Any], bits: int) -> np.ndarray[Any, Any]:
    """Pack last-axis integer values low-bits-first into uint32 words."""
    if bits <= 0 or 32 % bits:
        raise ValueError(f"bits must divide 32, got {bits}")
    per_word = 32 // bits
    if qvalues.shape[-1] % per_word:
        raise ValueError(f"Last dimension {qvalues.shape[-1]} is not divisible by {per_word}")
    q = np.asarray(qvalues, dtype=np.uint32)
    packed = np.zeros(q.shape[:-1] + (q.shape[-1] // per_word,), dtype=np.uint32)
    for lane in range(per_word):
        packed |= (q[..., lane::per_word] & np.uint32((1 << bits) - 1)) << np.uint32(lane * bits)
    return packed


def unpack_qvalues(packed: np.ndarray[Any, Any], bits: int) -> np.ndarray[Any, Any]:
    """Unpack MLX low-bits-first uint32 values along the last axis."""
    if bits <= 0 or 32 % bits:
        raise ValueError(f"bits must divide 32, got {bits}")
    per_word = 32 // bits
    shifts = np.arange(per_word, dtype=np.uint32) * np.uint32(bits)
    mask = np.uint32((1 << bits) - 1)
    unpacked = (np.asarray(packed, dtype=np.uint32)[..., None] >> shifts) & mask
    return unpacked.reshape(unpacked.shape[:-2] + (unpacked.shape[-2] * per_word,))


def dequantize_affine(
    packed: np.ndarray[Any, Any],
    scales: np.ndarray[Any, Any],
    biases: np.ndarray[Any, Any],
    bits: int,
    group_size: int,
) -> np.ndarray[Any, Any]:
    """Return float32 weights from an MLX affine packed triple."""
    unpacked = unpack_qvalues(packed, bits)
    if unpacked.shape[:-1] != scales.shape[:-1] or scales.shape != biases.shape:
        raise ValueError("Packed weights, scales, and biases have incompatible shapes")
    logical_last = scales.shape[-1] * group_size
    if unpacked.shape[-1] != logical_last:
        raise ValueError(
            f"Unpacked last dimension {unpacked.shape[-1]} does not match {scales.shape[-1]} groups x {group_size}"
        )
    grouped_q = unpacked.reshape(unpacked.shape[:-1] + (scales.shape[-1], group_size)).astype(np.float32)
    scale32 = np.asarray(scales, dtype=np.float32)[..., None]
    bias32 = np.asarray(biases, dtype=np.float32)[..., None]
    return (grouped_q * scale32 + bias32).reshape(unpacked.shape)


def quantize_affine(
    values: np.ndarray[Any, Any],
    bits: int,
    group_size: int,
    *,
    scale_dtype: str = "F32",
    bias_dtype: str = "F32",
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Quantize float values into MLX affine uint32/scales/biases arrays.

    Groups are formed along the last axis. The signed scale and bias follow
    MLX's affine endpoint rule; rounding is nearest with ties-to-even.
    """
    if bits <= 0 or 32 % bits:
        raise ValueError(f"bits must divide 32, got {bits}")
    values32 = np.asarray(values, dtype=np.float32)
    if values32.ndim == 0 or values32.shape[-1] % group_size:
        raise ValueError(f"Last dimension must be divisible by group_size={group_size}")
    per_word = 32 // bits
    if values32.shape[-1] % per_word:
        raise ValueError(f"Last dimension must be divisible by {per_word} packed values")
    grouped = values32.reshape(values32.shape[:-1] + (values32.shape[-1] // group_size, group_size))
    minv = grouped.min(axis=-1).astype(np.float32)
    maxv = grouped.max(axis=-1).astype(np.float32)
    n_bins = np.float32((1 << bits) - 1)
    eps = np.float32(1e-7)
    scale = np.maximum((maxv - minv) / n_bins, eps).astype(np.float32)
    use_min_edge = np.abs(minv) > np.abs(maxv)
    edge = np.where(use_min_edge, minv, maxv).astype(np.float32)
    scale = np.where(use_min_edge, scale, -scale).astype(np.float32)
    q0 = np.rint(edge / scale).astype(np.float32)
    has_edge_code = q0 != 0
    scale = np.where(has_edge_code, edge / np.where(has_edge_code, q0, np.float32(1.0)), scale)
    bias = np.where(has_edge_code, edge, np.float32(0.0)).astype(np.float32)
    # MLX checkpoints can store affine parameters in BF16. Compute q against
    # the values that will actually be stored so dequantization uses the same
    # scale and bias. The default remains F32 for parity with mx.quantize on
    # float32 inputs.
    scale = _float32_after_storage_cast(scale, scale_dtype)
    bias = _float32_after_storage_cast(bias, bias_dtype)
    scale = scale.astype(np.float32)
    normalized = (grouped - bias[..., None]) / scale[..., None]
    qvalues = np.rint(normalized).clip(0, (1 << bits) - 1).astype(np.uint32)
    return pack_qvalues(qvalues.reshape(values32.shape), bits), scale, bias


def _bf16_bits(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Cast float32 to bfloat16 bit patterns using round-to-nearest-even."""
    values32 = np.asarray(values, dtype=np.float32)
    word = values32.view(np.uint32)
    lsb = (word >> np.uint32(16)) & np.uint32(1)
    rounded = word + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _float32_after_storage_cast(values: np.ndarray[Any, Any], dtype_name: str) -> np.ndarray[Any, Any]:
    values32 = np.asarray(values, dtype=np.float32)
    if dtype_name == "BF16":
        bits = _bf16_bits(values32).astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    if dtype_name == "F16":
        return values32.astype(np.float16).astype(np.float32)
    if dtype_name in ("F32", "F64"):
        return values32
    raise ValueError(f"Unsupported scale/bias storage dtype: {dtype_name}")


def _raw_bytes(values: np.ndarray[Any, Any], dtype_name: str) -> bytes:
    if dtype_name == "BF16":
        return _bf16_bits(values).tobytes(order="C")
    target_dtype = RAW_DTYPES[dtype_name]
    return np.asarray(values).astype(target_dtype, copy=False).tobytes(order="C")


def _iter_slices(
    tensor: TensorInfo,
    max_elements: int = DEFAULT_CHUNK_ELEMENTS,
    expert_chunk: int = 8,
) -> Iterator[slice | None]:
    if not tensor.shape:
        yield None
        return
    if len(tensor.shape) == 1:
        width = tensor.shape[0]
        step = max(1, max_elements)
    else:
        width = math.prod(tensor.shape[1:])
        step = max(1, max_elements // max(1, width))
        if tensor.shape[0] == 256 and ("switch_mlp" in tensor.name or ".experts." in tensor.name):
            step = min(step, expert_chunk)
    for start in range(0, tensor.shape[0], step):
        yield slice(start, min(start + step, tensor.shape[0]))


def _read_quantized(checkpoint: Checkpoint, tensor: TensorInfo, first_slice: slice | None) -> np.ndarray[Any, Any]:
    assert tensor.quantized and tensor.bits is not None and tensor.group_size is not None
    packed = checkpoint.read_component(tensor.weight_key, first_slice)
    scales = checkpoint.read_float(tensor.scales_key or "", first_slice)
    biases = checkpoint.read_float(tensor.biases_key or "", first_slice)
    return dequantize_affine(packed, scales, biases, tensor.bits, tensor.group_size)


def _read_logical_float(checkpoint: Checkpoint, tensor: TensorInfo, first_slice: slice | None) -> np.ndarray[Any, Any]:
    if tensor.quantized:
        return _read_quantized(checkpoint, tensor, first_slice)
    return checkpoint.read_float(tensor.weight_key, first_slice)


def _chunk_pair(
    base: Checkpoint,
    finetune: Checkpoint,
    base_tensor: TensorInfo,
    finetune_tensor: TensorInfo,
    first_slice: slice | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    return (
        _read_logical_float(base, base_tensor, first_slice),
        _read_logical_float(finetune, finetune_tensor, first_slice),
    )


def _chunk_elements(tensor: TensorInfo, first_slice: slice | None) -> int:
    if first_slice is None or not tensor.shape:
        return 1
    return (first_slice.stop - first_slice.start) * math.prod(tensor.shape[1:]) if len(tensor.shape) > 1 else first_slice.stop - first_slice.start


def module_for_tensor(name: str) -> str:
    if "mlp.gate." in name:
        return "router"
    if "shared_expert" in name:
        return "shared"
    if "switch_mlp." in name or ".experts." in name:
        return "experts"
    if ".self_attn." in name or ".linear_attn." in name:
        return "attn"
    return "other"


def effective_lambda(name: str, lam: float, overrides: Mapping[str, float | None] | None = None) -> float:
    overrides = overrides or {}
    selected = overrides.get(module_for_tensor(name))
    return lam if selected is None else float(selected)


def _tokenizer_hash(checkpoint: Checkpoint) -> str:
    path = checkpoint.directory / "tokenizer.json"
    if not path.is_file():
        raise ValueError(f"Missing tokenizer.json in {checkpoint.directory}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


def _read_optional_json(directory: Path, name: str) -> dict[str, Any]:
    path = directory / name
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _tokenizer_semantics(checkpoint: Checkpoint) -> dict[str, Any]:
    """Read tokenizer compatibility fields without loading model weights."""
    path = checkpoint.directory / "tokenizer.json"
    if not path.is_file():
        raise ValueError(f"Missing tokenizer.json in {checkpoint.directory}")
    tokenizer = json.loads(path.read_text())
    if not isinstance(tokenizer, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    model = tokenizer.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError(f"Missing tokenizer model object in {path}")
    raw_vocab = model.get("vocab")
    if isinstance(raw_vocab, Mapping):
        vocab = {str(token): int(token_id) for token, token_id in raw_vocab.items()}
    elif isinstance(raw_vocab, list):
        # Unigram tokenizers store (token, score) rows; row position is the ID.
        vocab = {
            str(row[0]): token_id
            for token_id, row in enumerate(raw_vocab)
            if isinstance(row, (list, tuple)) and row
        }
    else:
        raise ValueError(f"Cannot read token -> id vocabulary from {path}")

    tokenizer_config = _read_optional_json(checkpoint.directory, "tokenizer_config.json")
    generation_config = _read_optional_json(checkpoint.directory, "generation_config.json")
    added = tokenizer.get("added_tokens", [])
    if not isinstance(added, list):
        raise ValueError(f"Expected added_tokens to be a list in {path}")
    added_tokens = sorted(
        (_canonical_json(item) for item in added),
        key=lambda item: (int(item.get("id", -1)), str(item.get("content", "")))
        if isinstance(item, Mapping)
        else (0, json.dumps(item, sort_keys=True)),
    )
    decoder = tokenizer_config.get("added_tokens_decoder", {})
    if isinstance(decoder, Mapping):
        added_decoder = sorted(
            [
                {"id": int(token_id), **_canonical_json(item)}
                for token_id, item in decoder.items()
                if isinstance(item, Mapping)
            ],
            key=lambda item: (int(item["id"]), str(item.get("content", ""))),
        )
    else:
        added_decoder = _canonical_json(decoder)

    special_names = (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    )

    def special_identity(value: Any) -> Any:
        if isinstance(value, Mapping) and "content" in value:
            return value["content"]
        if isinstance(value, list):
            return [special_identity(item) for item in value]
        return value

    special_tokens = {
        name: _canonical_json(special_identity(tokenizer_config[name]))
        for name in special_names
        if name in tokenizer_config
    }

    eos_values: list[Any] = []
    for source in (checkpoint.config, checkpoint.config.get("text_config", {}), tokenizer_config, generation_config):
        if isinstance(source, Mapping) and source.get("eos_token_id") is not None:
            value = source["eos_token_id"]
            if isinstance(value, list):
                eos_values.extend(value)
            else:
                eos_values.append(value)
    eos_token = tokenizer_config.get("eos_token")
    eos_content = special_identity(eos_token) if eos_token is not None else None
    if eos_values:
        eos_ids = sorted({int(value) for value in eos_values})
    elif eos_content is not None:
        all_vocab = dict(vocab)
        for item in added_tokens:
            if isinstance(item, Mapping) and "content" in item and "id" in item:
                all_vocab[str(item["content"])] = int(item["id"])
        if isinstance(added_decoder, list):
            for item in added_decoder:
                if "content" in item and "id" in item:
                    all_vocab[str(item["content"])] = int(item["id"])
        token_id = all_vocab.get(str(eos_content))
        eos_ids = [] if token_id is None else [token_id]
    else:
        eos_ids = []

    chat_template = tokenizer_config.get("chat_template", tokenizer.get("chat_template"))
    if chat_template is None:
        template_path = checkpoint.directory / "chat_template.jinja"
        if template_path.is_file():
            chat_template = template_path.read_text()

    return {
        "vocab": vocab,
        "added_tokens": added_tokens,
        "added_tokens_decoder": added_decoder,
        "special_tokens": special_tokens,
        "eos_ids": eos_ids,
        "chat_template": _canonical_json(chat_template),
    }


def _tokenizer_semantic_differences(base: Checkpoint, finetune: Checkpoint) -> list[str]:
    left, right = _tokenizer_semantics(base), _tokenizer_semantics(finetune)
    fields = (
        ("vocab", "vocabulary"),
        ("added_tokens", "added tokens"),
        ("added_tokens_decoder", "added token decoder"),
        ("special_tokens", "special tokens"),
        ("eos_ids", "EOS ids"),
        ("chat_template", "chat_template"),
    )
    return [label for field, label in fields if left[field] != right[field]]


LANGUAGE_MODEL_PREFIX = "language_model."


def _is_language_model_tensor(name: str) -> bool:
    return name.startswith(LANGUAGE_MODEL_PREFIX)


def _config_value(config: Mapping[str, Any], name: str) -> Any:
    if name in config and config[name] is not None:
        return config[name]
    for subkey in ("text_config", "vision_config"):
        nested = config.get(subkey)
        if isinstance(nested, Mapping) and nested.get(name) is not None:
            return nested[name]
    return None


ARCH_FIELDS = (
    "model_type",
    "num_hidden_layers",
    "num_experts",
    "num_local_experts",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_experts_per_tok",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)


def compatibility_issues(
    base: Checkpoint,
    finetune: Checkpoint,
    base_tokenizer_hash: str,
    finetune_tokenizer_hash: str,
    allow_tokenizer_mismatch: bool = False,
    text_only: bool = False,
) -> list[str]:
    issues: list[str] = []
    for field in ARCH_FIELDS:
        left, right = _config_value(base.config, field), _config_value(finetune.config, field)
        if left is not None and right is not None and left != right:
            issues.append(f"config {field} differs: base={left!r}, finetune={right!r}")
    left_arch = base.config.get("architectures")
    right_arch = finetune.config.get("architectures")
    if left_arch is not None and right_arch is not None and left_arch != right_arch:
        issues.append(f"config architectures differs: base={left_arch!r}, finetune={right_arch!r}")
    for subconfig in ("text_config", "vision_config"):
        left = base.config.get(subconfig)
        right = finetune.config.get(subconfig)
        if (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.get("model_type") is not None
            and right.get("model_type") is not None
            and left.get("model_type") != right.get("model_type")
        ):
            issues.append(
                f"config {subconfig}.model_type differs: "
                f"base={left.get('model_type')!r}, finetune={right.get('model_type')!r}"
            )
    tokenizer_differences = _tokenizer_semantic_differences(base, finetune)
    if tokenizer_differences and not allow_tokenizer_mismatch:
        issues.append(
            "tokenizer semantics differ: " + ", ".join(tokenizer_differences)
            + " (use --allow-tokenizer-mismatch to override)"
        )

    base_keys, ft_keys = set(base.logical_tensors), set(finetune.logical_tensors)
    if text_only:
        base_keys = {key for key in base_keys if _is_language_model_tensor(key)}
        ft_keys = {key for key in ft_keys if _is_language_model_tensor(key)}
    only_base, only_ft = sorted(base_keys - ft_keys), sorted(ft_keys - base_keys)
    if only_base:
        issues.append(f"logical key-set mismatch: {len(only_base)} base-only keys; examples={only_base[:8]}")
    if only_ft:
        issues.append(f"logical key-set mismatch: {len(only_ft)} finetune-only keys; examples={only_ft[:8]}")
    common = sorted(base_keys & ft_keys)
    mismatches = []
    for key in common:
        left, right = base.logical_tensors[key], finetune.logical_tensors[key]
        if left.shape != right.shape:
            mismatches.append((key, left.shape, right.shape))
    if mismatches:
        issues.append(
            f"dequantized shape mismatch: {len(mismatches)} tensors; examples={mismatches[:8]}"
        )
    return issues


def print_tokenizer_hashes(base_hash: str, finetune_hash: str) -> None:
    print(f"Tokenizer SHA-256 base:     {base_hash}", flush=True)
    print(f"Tokenizer SHA-256 finetune: {finetune_hash}", flush=True)


class ModuleStats:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, float | int]] = {}

    def add(self, module: str, tau: np.ndarray[Any, Any], base: np.ndarray[Any, Any]) -> None:
        record = self.values.setdefault(
            module,
            {"tensors": 0, "elements": 0, "sum_abs_tau": 0.0, "sum_abs_base": 0.0},
        )
        record["elements"] = int(record["elements"]) + int(tau.size)
        record["sum_abs_tau"] = float(record["sum_abs_tau"]) + float(np.abs(tau).sum(dtype=np.float64))
        record["sum_abs_base"] = float(record["sum_abs_base"]) + float(np.abs(base).sum(dtype=np.float64))

    def tensor_seen(self, module: str) -> None:
        record = self.values.setdefault(
            module,
            {"tensors": 0, "elements": 0, "sum_abs_tau": 0.0, "sum_abs_base": 0.0},
        )
        record["tensors"] = int(record["tensors"]) + 1

    def report(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for module in sorted(self.values):
            row = self.values[module]
            elements = int(row["elements"])
            mean_tau = float(row["sum_abs_tau"]) / elements if elements else 0.0
            mean_base = float(row["sum_abs_base"]) / elements if elements else 0.0
            result[module] = {
                "tensors": int(row["tensors"]),
                "elements": elements,
                "mean_abs_tau": mean_tau,
                "mean_abs_base": mean_base,
                "mean_abs_tau_over_base": mean_tau / mean_base if mean_base else 0.0,
            }
        return result


def _format_stats(stats: Mapping[str, Mapping[str, float | int]]) -> None:
    print("module statistics: tensors, elements, mean|tau|, mean|base|, mean|tau|/mean|base|", flush=True)
    for module, row in sorted(stats.items()):
        print(
            f"  {module:8s} {int(row['tensors']):5d} {int(row['elements']):12d} "
            f"{float(row['mean_abs_tau']):.7g} {float(row['mean_abs_base']):.7g} "
            f"{float(row['mean_abs_tau_over_base']):.7g}",
            flush=True,
        )


def select_sample_tensors(
    base: Checkpoint,
    finetune: Checkpoint,
    issues: Sequence[str],
    sample_count: int,
    text_only: bool = False,
) -> list[str]:
    keys = sorted(set(base.logical_tensors) & set(finetune.logical_tensors))
    if text_only:
        keys = [key for key in keys if _is_language_model_tensor(key)]
    keys = [key for key in keys if base.logical_tensors[key].shape == finetune.logical_tensors[key].shape]
    chosen: list[str] = []
    seen_modules: set[str] = set()
    # Prefer one representative per module, then fill up with the next common
    # tensors. This is header-only selection; payload reads happen afterwards.
    for key in keys:
        module = module_for_tensor(key)
        if module not in seen_modules:
            chosen.append(key)
            seen_modules.add(module)
            if len(chosen) >= sample_count:
                return chosen
    for key in keys:
        if key not in chosen:
            chosen.append(key)
            if len(chosen) >= sample_count:
                break
    return chosen


def dry_run(
    base: Checkpoint,
    finetune: Checkpoint,
    base_hash: str,
    finetune_hash: str,
    allow_tokenizer_mismatch: bool = False,
    sample_count: int = 8,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
    text_only: bool = False,
) -> dict[str, Any]:
    print_tokenizer_hashes(base_hash, finetune_hash)
    issues = compatibility_issues(
        base, finetune, base_hash, finetune_hash, allow_tokenizer_mismatch, text_only
    )
    samples = select_sample_tensors(base, finetune, issues, sample_count, text_only)
    sample_stats = ModuleStats()
    for name in samples:
        bt, ft = base.logical_tensors[name], finetune.logical_tensors[name]
        first_slice = next(_iter_slices(bt, chunk_elements), None)
        b, f = _chunk_pair(base, finetune, bt, ft, first_slice)
        sample_stats.tensor_seen(module_for_tensor(name))
        sample_stats.add(module_for_tensor(name), f - b, b)
        print(f"sample tensor: {name} (sampled {b.size} of {math.prod(bt.shape)} elements)", flush=True)
    print("DRY-RUN sample delta statistics (first bounded chunk of each sampled tensor):", flush=True)
    _format_stats(sample_stats.report())
    if issues:
        raise ValueError("Checkpoint compatibility failed:\n  - " + "\n  - ".join(issues))
    print("DRY-RUN compatibility: PASS; no output files written", flush=True)
    return sample_stats.report()


def _copy_raw_component(source: Checkpoint, key: str, output_handles: Mapping[str, OutputShard], output_entries: Mapping[str, TensorEntry]) -> None:
    entry = source.entries[key]
    out_entry = output_entries[key]
    if entry.nbytes != out_entry.nbytes:
        raise ValueError(f"Cannot raw-copy {key}: source and base output byte counts differ")
    src = source.read_handle(entry.shard)
    dst = output_handles[out_entry.shard]
    source_start = len(source.header_bytes[entry.shard]) + entry.data_start
    dest_start = output_header_size(output_handles, out_entry.shard) + out_entry.data_start
    # output_header_size returns the already-created header prefix length.
    for position in range(0, entry.nbytes, RAW_COPY_CHUNK):
        count = min(RAW_COPY_CHUNK, entry.nbytes - position)
        src.seek(source_start + position)
        block = src.read(count)
        if len(block) != count:
            raise ValueError(f"Short read while copying {key}")
        dst.seek(dest_start + position)
        dst.write(block)


def output_header_size(output_handles: Mapping[str, OutputShard], shard: str) -> int:
    return output_handles[shard].header_bytes


def _write_array_chunk(
    output_handles: Mapping[str, OutputShard],
    output_entries: Mapping[str, TensorEntry],
    key: str,
    first_slice: slice | None,
    values: np.ndarray[Any, Any],
) -> None:
    entry = output_entries[key]
    handle = output_handles[entry.shard]
    header_bytes = output_header_size(output_handles, entry.shard)
    dtype = RAW_DTYPES[entry.dtype]
    if first_slice is None or not entry.shape:
        byte_offset = 0
    else:
        row_bytes = math.prod(entry.shape[1:]) * dtype.itemsize if len(entry.shape) > 1 else dtype.itemsize
        byte_offset = first_slice.start * row_bytes
    payload = _raw_bytes(values, entry.dtype)
    handle.seek(header_bytes + entry.data_start + byte_offset)
    handle.write(payload)


def _output_headers(base: Checkpoint, out: Path) -> tuple[dict[str, OutputShard], dict[str, TensorEntry]]:
    output_handles: dict[str, OutputShard] = {}
    for shard in sorted(set(base.weight_map.values())):
        prefix = base.header_bytes[shard]
        out_path = out / shard
        with out_path.open("wb") as f:
            f.write(prefix)
            f.truncate((base.directory / shard).stat().st_size)
        handle = out_path.open("r+b")
        output_handles[shard] = OutputShard(handle, len(prefix))
    return output_handles, dict(base.entries)


def _write_tensor_copy(
    base: Checkpoint,
    tensor: TensorInfo,
    output_handles: Mapping[str, OutputShard],
    output_entries: Mapping[str, TensorEntry],
) -> None:
    for key in tensor.component_keys:
        _copy_raw_component(base, key, output_handles, output_entries)


def _write_dense_tensor(
    base: Checkpoint,
    tensor: TensorInfo,
    first_slice: slice | None,
    merged: np.ndarray[Any, Any],
    output_handles: Mapping[str, OutputShard],
    output_entries: Mapping[str, TensorEntry],
) -> None:
    _write_array_chunk(output_handles, output_entries, tensor.weight_key, first_slice, merged)


def _write_quantized_tensor_chunk(
    base: Checkpoint,
    tensor: TensorInfo,
    first_slice: slice | None,
    merged: np.ndarray[Any, Any],
            output_handles: Mapping[str, OutputShard],
    output_entries: Mapping[str, TensorEntry],
) -> None:
    assert tensor.bits is not None and tensor.group_size is not None
    scale_dtype = output_entries[tensor.scales_key or ""].dtype
    bias_dtype = output_entries[tensor.biases_key or ""].dtype
    packed, scales, biases = quantize_affine(
        merged,
        tensor.bits,
        tensor.group_size,
        scale_dtype=scale_dtype,
        bias_dtype=bias_dtype,
    )
    _write_array_chunk(output_handles, output_entries, tensor.weight_key, first_slice, packed)
    _write_array_chunk(output_handles, output_entries, tensor.scales_key or "", first_slice, scales)
    _write_array_chunk(output_handles, output_entries, tensor.biases_key or "", first_slice, biases)


def _make_output_directory(base: Checkpoint, out: Path) -> None:
    if out.exists():
        raise ValueError(f"Output path already exists: {out}")
    if not out.parent.is_dir():
        raise ValueError(f"Output parent directory does not exist: {out.parent}")
    out.mkdir()
    for source in base.directory.iterdir():
        if not source.is_file() or source.suffix == ".safetensors":
            continue
        if source.name == "model.safetensors.index.json":
            continue
        shutil.copy2(source, out / source.name)
    shutil.copy2(base.directory / "model.safetensors.index.json", out / "model.safetensors.index.json")


def _check_disk_space(base: Checkpoint, out: Path) -> None:
    free = shutil.disk_usage(out.parent).free
    required = base.shard_size + 2 * GIB
    if free < required:
        raise ValueError(
            f"Insufficient free space at {out.parent}: {free / GIB:.2f} GiB available, "
            f"need at least {required / GIB:.2f} GiB (base shard size + 2 GiB)"
        )


def _add_chunk_stats(stats: ModuleStats, name: str, b: np.ndarray[Any, Any], f: np.ndarray[Any, Any]) -> None:
    stats.add(module_for_tensor(name), f - b, b)


def _scan_tensor(
    base: Checkpoint,
    finetune: Checkpoint,
    bt: TensorInfo,
    ft: TensorInfo,
    stats: ModuleStats,
    chunk_elements: int,
) -> bool:
    module = module_for_tensor(bt.name)
    stats.tensor_seen(module)
    tau_zero = True
    for first_slice in _iter_slices(bt, chunk_elements):
        b, f = _chunk_pair(base, finetune, bt, ft, first_slice)
        tau = f - b
        tau_zero = tau_zero and bool(np.all(tau == 0))
        stats.add(module, tau, b)
    return tau_zero


def _ties_keep_count(total: int, fraction: float) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("--ties-trim must be between 0 and 1")
    return int(math.ceil(total * fraction)) if fraction else 0


def _ties_threshold_bits(
    base: Checkpoint,
    finetune: Checkpoint,
    bt: TensorInfo,
    ft: TensorInfo,
    keep_count: int,
    chunk_elements: int,
) -> int | None:
    if keep_count <= 0:
        return None
    rank = keep_count
    prefix = 0
    for shift in (24, 16, 8, 0):
        histogram = np.zeros(256, dtype=np.int64)
        for first_slice in _iter_slices(bt, chunk_elements):
            b, f = _chunk_pair(base, finetune, bt, ft, first_slice)
            magnitude_bits = np.abs(f - b).view(np.uint32).reshape(-1)
            if shift == 24:
                candidates = magnitude_bits
            else:
                candidates = magnitude_bits[(magnitude_bits >> np.uint32(shift + 8)) == np.uint32(prefix)]
            if candidates.size:
                byte_values = ((candidates >> np.uint32(shift)) & np.uint32(0xFF)).astype(np.int64)
                histogram += np.bincount(byte_values, minlength=256)
        selected = None
        for bucket in range(255, -1, -1):
            if rank > int(histogram[bucket]):
                rank -= int(histogram[bucket])
            else:
                selected = bucket
                break
        if selected is None:
            raise ValueError(f"TIES threshold selection failed for {bt.name}")
        prefix = (prefix << 8) | selected
    return prefix


def _apply_ties_chunk(
    tau: np.ndarray[Any, Any],
    threshold_bits: int | None,
    keep_count_remaining: int,
) -> tuple[np.ndarray[Any, Any], int]:
    if threshold_bits is None:
        return np.zeros_like(tau), keep_count_remaining
    bits = np.abs(tau).view(np.uint32)
    keep = bits > np.uint32(threshold_bits)
    greater = int(np.count_nonzero(keep))
    remaining = keep_count_remaining - greater
    ties = np.flatnonzero(bits.reshape(-1) == np.uint32(threshold_bits))
    tie_take = min(max(remaining, 0), int(ties.size))
    if tie_take:
        flat_keep = keep.reshape(-1)
        flat_keep[ties[:tie_take]] = True
    remaining -= tie_take
    return np.where(keep, tau, np.float32(0.0)), remaining


def _apply_update(
    tau: np.ndarray[Any, Any],
    method: str,
    dare_p: float,
    rng: np.random.Generator,
    ties_threshold: int | None,
    ties_remaining: int,
) -> tuple[np.ndarray[Any, Any], int]:
    if method == "linear":
        return tau, ties_remaining
    if method == "dare":
        if dare_p == 0:
            return tau, ties_remaining
        keep = rng.random(tau.shape) >= dare_p
        return tau * keep.astype(np.float32) / np.float32(1.0 - dare_p), ties_remaining
    if method == "ties":
        # A missing threshold represents either keep-none (remaining == 0) or
        # keep-all (remaining is the original tensor size).
        if ties_threshold is None:
            if ties_remaining > 0:
                return tau, max(0, ties_remaining - tau.size)
            return np.zeros_like(tau), 0
        return _apply_ties_chunk(tau, ties_threshold, ties_remaining)
    raise ValueError(f"Unknown merge method {method!r}")


def merge_checkpoints(
    base_path: str | Path,
    finetune_path: str | Path,
    output_path: str | Path,
    *,
    lam: float = 1.0,
    lam_attn: float | None = None,
    lam_experts: float | None = None,
    lam_shared: float | None = None,
    lam_router: float | None = None,
    method: str = "linear",
    dare_p: float = 0.9,
    ties_trim: float = 1.0,
    seed: int = 0,
    allow_tokenizer_mismatch: bool = False,
    text_only: bool = False,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Merge two compatible checkpoints and write a base-layout output tree."""
    if method not in ("linear", "dare", "ties"):
        raise ValueError("method must be linear, dare, or ties")
    if not 0.0 <= dare_p < 1.0:
        raise ValueError("--dare-p must be in [0, 1)")
    if not 0.0 <= ties_trim <= 1.0:
        raise ValueError("--ties-trim must be between 0 and 1")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    base = Checkpoint(base_path)
    finetune = Checkpoint(finetune_path)
    out = Path(output_path).expanduser().resolve()
    base_hash = _tokenizer_hash(base)
    finetune_hash = _tokenizer_hash(finetune)
    if verbose:
        print_tokenizer_hashes(base_hash, finetune_hash)
    issues = compatibility_issues(
        base, finetune, base_hash, finetune_hash, allow_tokenizer_mismatch, text_only
    )
    if issues:
        base.close()
        finetune.close()
        raise ValueError("Checkpoint compatibility failed:\n  - " + "\n  - ".join(issues))
    _check_disk_space(base, out)
    _make_output_directory(base, out)
    output_handles, output_entries = _output_headers(base, out)
    stats = ModuleStats()
    overrides: dict[str, float | None] = {
        "attn": lam_attn,
        "experts": lam_experts,
        "shared": lam_shared,
        "router": lam_router,
    }
    rng = np.random.default_rng(seed)
    merge_names = {
        name
        for name in base.logical_tensors
        if not text_only or (_is_language_model_tensor(name) and name in finetune.logical_tensors)
    }
    copied_names = set(base.logical_tensors) - merge_names

    try:
        for name in sorted(base.logical_tensors):
            bt = base.logical_tensors[name]
            if name not in merge_names:
                _write_tensor_copy(base, bt, output_handles, output_entries)
                continue
            ft = finetune.logical_tensors[name]
            tensor_lam = effective_lambda(name, lam, overrides)
            tau_zero = _scan_tensor(base, finetune, bt, ft, stats, chunk_elements)
            if tensor_lam == 0.0 or tau_zero:
                _write_tensor_copy(base, bt, output_handles, output_entries)
                continue

            keep_count = _ties_keep_count(math.prod(bt.shape), ties_trim) if method == "ties" else 0
            ties_threshold = (
                _ties_threshold_bits(base, finetune, bt, ft, keep_count, chunk_elements)
                if method == "ties" and keep_count < math.prod(bt.shape)
                else None
            )
            ties_remaining = keep_count
            for first_slice in _iter_slices(bt, chunk_elements):
                b, f = _chunk_pair(base, finetune, bt, ft, first_slice)
                tau, ties_remaining = _apply_update(
                    f - b,
                    method,
                    dare_p,
                    rng,
                    ties_threshold,
                    ties_remaining,
                )
                if (
                    tensor_lam == 1.0
                    and (
                        method == "linear"
                        or (method == "dare" and dare_p == 0.0)
                        or (method == "ties" and keep_count == math.prod(bt.shape))
                    )
                ):
                    # Avoid cancellation in base + (finetune - base): at lambda
                    # one the exact input to requantization is dequant(finetune).
                    merged = f
                else:
                    merged = b + np.float32(tensor_lam) * tau
                if bt.quantized:
                    _write_quantized_tensor_chunk(
                        base, bt, first_slice, merged, output_handles, output_entries
                    )
                else:
                    _write_dense_tensor(base, bt, first_slice, merged, output_handles, output_entries)

        for handle in output_handles.values():
            handle.flush()
            handle.close()
        output_handles.clear()
    finally:
        for handle in output_handles.values():
            handle.close()
        base.close()
        finetune.close()

    metadata = {
        "base": str(base.directory),
        "finetune": str(finetune.directory),
        "output": str(out),
        "args": {
            "lam": lam,
            "lam_attn": lam_attn,
            "lam_experts": lam_experts,
            "lam_shared": lam_shared,
            "lam_router": lam_router,
            "method": method,
            "dare_p": dare_p,
            "ties_trim": ties_trim,
            "seed": seed,
            "allow_tokenizer_mismatch": allow_tokenizer_mismatch,
            "text_only": text_only,
        },
        "tokenizer_sha256": {"base": base_hash, "finetune": finetune_hash},
        "copied_keys_count": sum(
            len(base.logical_tensors[name].component_keys) for name in copied_names
        ),
        "copied_tensors_count": len(copied_names),
        "module_statistics": stats.report(),
        "quantization": {
            "format": "MLX affine, uint32 packed low-bits-first",
            "formula": (
                "n_bins=2**bits-1; scale=max((max-min)/n_bins,1e-7); "
                "if abs(min)>abs(max): edge=min and scale positive; else: edge=max and scale negative; "
                "q0=round(edge/scale); if q0!=0: scale=edge/q0 and bias=edge, else bias=0; "
                "q=clip(round((w-bias)/scale),0,n_bins); dequant=scale*q+bias"
            ),
            "rounding": "float32 round-to-nearest, ties-to-even",
        },
    }
    (out / "merge_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if verbose:
        print("Merge complete. Per-module statistics:")
        _format_stats(stats.report())
        print(f"Output: {out}")
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--finetune", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lam-attn", type=float)
    parser.add_argument("--lam-experts", type=float)
    parser.add_argument("--lam-shared", type=float)
    parser.add_argument("--lam-router", type=float)
    parser.add_argument("--method", choices=("linear", "dare", "ties"), default="linear")
    parser.add_argument("--dare-p", type=float, default=0.9)
    parser.add_argument("--ties-trim", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-tensors", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        base = Checkpoint(args.base)
        finetune = Checkpoint(args.finetune)
        try:
            base_hash = _tokenizer_hash(base)
            finetune_hash = _tokenizer_hash(finetune)
            if args.dry_run:
                dry_run(
                    base,
                    finetune,
                    base_hash,
                    finetune_hash,
                    allow_tokenizer_mismatch=args.allow_tokenizer_mismatch,
                    sample_count=args.sample_tensors,
                    text_only=args.text_only,
                )
                return 0
        finally:
            base.close()
            finetune.close()
        merge_checkpoints(
            args.base,
            args.finetune,
            args.out,
            lam=args.lam,
            lam_attn=args.lam_attn,
            lam_experts=args.lam_experts,
            lam_shared=args.lam_shared,
            lam_router=args.lam_router,
            method=args.method,
            dare_p=args.dare_p,
            ties_trim=args.ties_trim,
            seed=args.seed,
            allow_tokenizer_mismatch=args.allow_tokenizer_mismatch,
            text_only=args.text_only,
        )
        return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
