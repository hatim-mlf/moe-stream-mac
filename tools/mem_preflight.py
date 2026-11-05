#!/usr/bin/env python3
"""Token-length and system-wired preflight for the local LoRA training data."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mem_budget import (
    DEFAULT_OS_WIRED_GIB,
    GIB,
    MEM_GUARD_LIMIT_GIB,
    PREFLIGHT_MARGIN_GIB,
    Shapes,
    allocator_limits,
    load_shapes,
    peak_bytes,
    trainer_positions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path.home() / "models" / "Qwen3.6-35B-A3B-4bit"
DEFAULT_DATA_DIR = ROOT / "data" / "finetune"
DEFAULT_TRAINING_CONFIG = ROOT / "docs" / "finetune" / "lora.yaml"
DEFAULT_MAX_SEQ_LENGTH = 10240
GUARD_THRESHOLD_GIB = MEM_GUARD_LIMIT_GIB - PREFLIGHT_MARGIN_GIB


@dataclass(frozen=True)
class RowPrediction:
    label: str
    token_count: int
    model_tokens: int
    wired_bytes: int


@dataclass(frozen=True)
class PreflightResult:
    predictions: tuple[RowPrediction, ...]
    largest_fitting_tokens: int
    threshold_bytes: int
    cache_cap_bytes: int
    cache_occupancy_bytes: int
    layers: int

    @property
    def over_budget(self) -> tuple[RowPrediction, ...]:
        return tuple(
            row for row in self.predictions
            if row.wired_bytes >= self.threshold_bytes
        )


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield line_number, row


def tokenize_rows(data_dir: Path, tokenizer: Any) -> tuple[tuple[str, int], ...]:
    lengths: list[tuple[str, int]] = []
    for split in ("train", "valid"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"required preflight data file is missing: {path}")
        for line_number, row in _read_jsonl(path):
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number}: row needs chat-template messages")
            encoded = tokenizer.apply_chat_template(
                messages, tools=row.get("tools"), return_dict=False
            )
            lengths.append((f"{path.name}:{line_number}", len(encoded)))
    if not lengths:
        raise ValueError(f"no train or valid rows found in {data_dir}")
    return tuple(lengths)


def load_chat_tokenizer(model_dir: Path) -> Any:
    try:
        from mlx_lm.utils import load_tokenizer
    except ImportError:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("preflight needs mlx_lm or transformers for the tokenizer") from exc
        return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return load_tokenizer(model_dir)


def _wired_bytes(
    shapes: Shapes, model_tokens: int, layers: int, cache_cap_bytes: int
) -> int:
    active = peak_bytes(shapes, model_tokens, layers)
    reserve = int(DEFAULT_OS_WIRED_GIB * GIB)
    return active + cache_cap_bytes + reserve


def _largest_fitting_tokens(
    shapes: Shapes, layers: int, max_sequence: int, cache_cap_bytes: int,
    threshold_bytes: int,
) -> int:
    low, high, best = 1, max_sequence, 0
    while low <= high:
        middle = (low + high) // 2
        model_tokens = trainer_positions(middle, max_sequence)
        wired = _wired_bytes(shapes, model_tokens, layers, cache_cap_bytes)
        if wired < threshold_bytes:
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return best


def predict_rows(
    shapes: Shapes,
    rows: Iterable[tuple[str, int]],
    layers: int,
    max_sequence: int,
    cache_limit_gib: float | None,
) -> PreflightResult:
    limits = allocator_limits(cache_limit_gib)
    cap = limits.cache_cap_bytes
    threshold = int(GUARD_THRESHOLD_GIB * GIB)
    predictions = tuple(
        RowPrediction(
            label=label,
            token_count=min(token_count, max_sequence),
            model_tokens=trainer_positions(min(token_count, max_sequence), max_sequence),
            wired_bytes=_wired_bytes(
                shapes,
                trainer_positions(min(token_count, max_sequence), max_sequence),
                layers,
                cap,
            ),
        )
        for label, token_count in rows
    )
    fit = _largest_fitting_tokens(
        shapes, layers, max_sequence, cap, threshold
    )
    return PreflightResult(
        predictions, fit, threshold, cap, limits.occupancy_bytes, layers
    )


def _print_result(result: PreflightResult, max_sequence: int) -> None:
    print(
        f"Preflight guard threshold: {GUARD_THRESHOLD_GIB:.1f} GiB "
        f"(< {MEM_GUARD_LIMIT_GIB:.1f} GiB guard minus {PREFLIGHT_MARGIN_GIB:.1f} GiB margin)"
    )
    print(
        f"Cache occupancy now: {result.cache_occupancy_bytes / GIB:.3f} GiB; "
        f"cap used in wired bound: {result.cache_cap_bytes / GIB:.3f} GiB"
    )
    print("Trainer calls clear_cache after each evaluated step at threshold 0 bytes.")
    print(
        f"Largest fitting row length for L={result.layers}: "
        f"{result.largest_fitting_tokens} tokens"
    )
    if result.over_budget:
        print(
            "Hint: with the default MLX cache the bound includes the full cache cap; "
            "measured runs fit with MLX_CACHE_LIMIT_GB=0 (docs/finetune/MEMORY_MODEL.md)."
        )
        print("Rows at or over budget:")
        for row in result.over_budget:
            print(
                f"  {row.label}: row={row.token_count} tokens, model T={row.model_tokens}, "
                f"system wired bound={row.wired_bytes / GIB:.3f} GiB"
            )
        return
    longest = max(result.predictions, key=lambda row: row.token_count)
    print(
        f"All {len(result.predictions)} rows fit; longest is {longest.label} "
        f"({longest.token_count} tokens capped at {max_sequence}; "
        f"wired bound {longest.wired_bytes / GIB:.3f} GiB)."
    )


def run_preflight(
    data_dir: Path,
    model_dir: Path,
    layers: int,
    max_sequence: int,
    cache_limit_gib: float | None,
    config_path: Path | None = None,
) -> PreflightResult:
    config = config_path or model_dir / "config.json"
    shapes = load_shapes(config)
    if layers == -1:
        layers = shapes.layers
    if not 0 <= layers <= shapes.layers:
        raise ValueError(f"layers must be between 0 and {shapes.layers}, or -1 for all")
    tokenizer = load_chat_tokenizer(model_dir)
    rows = tokenize_rows(data_dir, tokenizer)
    return predict_rows(shapes, rows, layers, max_sequence, cache_limit_gib)


def _read_training_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"training config must be a YAML mapping: {path}")
    return value


def lora_preflight_options(argv: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("-c", "--config")
    parsed, _ = parser.parse_known_args(argv)
    config_path = Path(parsed.config) if parsed.config else DEFAULT_TRAINING_CONFIG
    yaml_values = _read_training_yaml(config_path) if config_path.is_file() else {}
    model = parsed.model or yaml_values.get("model") or DEFAULT_MODEL_DIR
    data = parsed.data or yaml_values.get("data") or DEFAULT_DATA_DIR
    layers = parsed.num_layers
    if layers is None:
        layers = int(yaml_values.get("num_layers", 16))
    max_sequence = parsed.max_seq_length
    if max_sequence is None:
        max_sequence = int(yaml_values.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH))
    cache_value = os.environ.get("MLX_CACHE_LIMIT_GB")
    cache_limit = None if cache_value is None else float(cache_value)
    return {
        "model_dir": Path(model),
        "data_dir": Path(data),
        "layers": layers,
        "max_sequence": max_sequence,
        "cache_limit_gib": cache_limit,
    }


def lora_preflight(argv: list[str]) -> int:
    if os.environ.get("MEM_PREFLIGHT") == "0":
        print("Memory preflight disabled by MEM_PREFLIGHT=0", flush=True)
        return 0
    if "--help" in argv or "-h" in argv:
        return 0
    try:
        options = lora_preflight_options(argv)
        result = run_preflight(**options)
    except Exception as exc:
        print(f"Memory preflight failed closed: {exc}", file=sys.stderr, flush=True)
        return 3
    _print_result(result, options["max_sequence"])
    if result.over_budget:
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--cache-limit-gb", type=float, default=None)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.max_seq_length < 2:
        parser.error("--max-seq-length must be at least 2")
    if args.cache_limit_gb is not None and (
        not math.isfinite(args.cache_limit_gb) or args.cache_limit_gb < 0
    ):
        parser.error("--cache-limit-gb must be finite and nonnegative")
    try:
        result = run_preflight(
            args.data, args.model, args.layers, args.max_seq_length,
            args.cache_limit_gb, args.config,
        )
    except Exception as exc:
        print(f"preflight error: {exc}", file=sys.stderr)
        return 3
    _print_result(result, args.max_seq_length)
    return 3 if result.over_budget else 0


if __name__ == "__main__":
    raise SystemExit(main())
