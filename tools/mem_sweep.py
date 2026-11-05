"""Subprocess sweep and least-squares fitting for mem_profile measurements."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

GIB = 2**30
STEP_PHASES = ("load", "forward", "value_and_grad", "optimizer_update")


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(number < 1 for number in values):
        raise argparse.ArgumentTypeError("all comma-separated values must be positive")
    return values


def _is_oom(returncode: int, output: str) -> bool:
    lowered = output.lower()
    markers = (
        "out of memory", "failed to allocate", "allocation failed",
        "insufficient memory", "bad_alloc", "outofmemory",
    )
    return returncode in (-9, 137) or any(marker in lowered for marker in markers)


def _child_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_sweep(
    token_values: Sequence[int],
    layer_values: Sequence[int],
    repeat: int,
    output_path: str | Path,
    child_runner: Callable[..., Any] | None = None,
    mode: str = "trainer-step",
    cache_limit_gb: float | None = None,
) -> list[dict[str, Any]]:
    if repeat < 1 or not token_values or not layer_values:
        raise ValueError("sweep needs positive repeat and non-empty token/layer grids")
    if mode not in ("step", "trainer-step"):
        raise ValueError("mode must be step or trainer-step")
    if cache_limit_gb is not None:
        if not math.isfinite(cache_limit_gb) or cache_limit_gb < 0:
            raise ValueError("cache-limit-gb must be a finite float >= 0")
        if not math.isfinite(cache_limit_gb * GIB):
            raise ValueError("cache-limit-gb is too large")
    runner = subprocess.run if child_runner is None else child_runner
    points = sorted(
        ((tokens, layers) for tokens in token_values for layers in layer_values),
        key=lambda point: (point[0] * point[1], point[0], point[1]),
    )
    output = Path(output_path).resolve()
    results = []
    for repeat_index in range(repeat):
        for point_index, (tokens, layers) in enumerate(points):
            results.append(_run_sweep_child(
                runner, output, tokens, layers, repeat_index, point_index,
                mode, cache_limit_gb,
            ))
    return results


def _sweep_record(
    tokens: int,
    layers: int,
    mode: str,
    cache_limit_gb: float | None,
    status: str,
    exit_code: int | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    config = {
        "tokens": tokens, "layers": layers, "mode": mode,
        "cache_limit_gb": "default" if cache_limit_gb is None else cache_limit_gb,
    }
    result: dict[str, Any] = {"config": config, "status": status}
    if exit_code is not None:
        result["exit_code"] = exit_code
    if status == "error" and detail is not None:
        result["detail"] = detail[-3000:]
    return result


def _run_sweep_child(
    runner: Callable[..., Any],
    output: Path,
    tokens: int,
    layers: int,
    repeat_index: int,
    point_index: int,
    mode: str,
    cache_limit_gb: float | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mem-profile-") as temporary:
        child_output = Path(temporary) / "step.jsonl"
        profile_script = Path(__file__).with_name("mem_profile.py").resolve()
        command = [sys.executable, str(profile_script), mode]
        command.extend(["--tokens", str(tokens), "--layers", str(layers)])
        if mode == "step":
            command.extend(["--gdn-checkpoint", "1"])
        if cache_limit_gb is not None:
            command.extend(["--cache-limit-gb", str(cache_limit_gb)])
        command.extend(["--out", str(child_output)])
        child = runner(command, capture_output=True, text=True)
        transcript = f"{getattr(child, 'stdout', '')}\n{getattr(child, 'stderr', '')}"
        if child.returncode == 0:
            records = _child_records(child_output)
            _append_child_records(output, records, repeat_index, point_index)
            result = _sweep_record(tokens, layers, mode, cache_limit_gb, "complete")
        else:
            status = "guard_stop" if child.returncode == 86 else (
                "oom" if _is_oom(child.returncode, transcript) else "error"
            )
            _append_child_records(
                output, _child_records(child_output), repeat_index, point_index
            )
            result = _sweep_record(
                tokens, layers, mode, cache_limit_gb, status,
                int(child.returncode), transcript,
            )
        append_jsonl(output, result)
        return result


def _append_child_records(
    output: Path,
    records: Sequence[dict[str, Any]],
    repeat_index: int,
    point_index: int,
) -> None:
    for record in records:
        record.setdefault("config", {}).update(
            {"sweep_repeat": repeat_index, "sweep_point": point_index}
        )
        append_jsonl(output, record)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _record_coordinates(record: Mapping[str, Any]) -> tuple[int, int, Mapping[str, Any]]:
    config = record.get("config", record)
    return int(config["tokens"]), int(config["layers"]), config


def _completed_points(records: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, float]]:
    phase_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    completed: list[tuple[int, int, float]] = []
    required = set(STEP_PHASES)
    for record in records:
        if record.get("status") is not None:
            continue
        tokens, layers, config = _record_coordinates(record)
        if config.get("command") == "trainer-step":
            if record.get("phase") == "overall" and "peak_memory_bytes" in record:
                completed.append((tokens, layers, float(record["peak_memory_bytes"])))
            continue
        if "phase" not in record:
            peak = record.get("peak_memory_bytes", record.get("peak"))
            if peak is not None:
                completed.append((tokens, layers, float(peak)))
            continue
        identity = (
            tokens, layers, config.get("run_id"), config.get("sweep_repeat"),
            config.get("sweep_point"),
        )
        group = phase_groups.setdefault(identity, {"phases": set(), "peaks": []})
        group["phases"].add(record["phase"])
        if "peak_memory_bytes" in record:
            group["peaks"].append(float(record["peak_memory_bytes"]))
    for identity, group in phase_groups.items():
        if required.issubset(group["phases"]) and group["peaks"]:
            completed.append((identity[0], identity[1], max(group["peaks"])))
    return completed


def fit_coefficients(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    points = _completed_points(records)
    if len(points) < 4:
        raise ValueError("fit requires at least four completed points")
    token_counts = np.asarray([point[0] for point in points], dtype=np.float64)
    layer_counts = np.asarray([point[1] for point in points], dtype=np.float64)
    peaks_gib = np.asarray([point[2] / GIB for point in points], dtype=np.float64)
    design = np.column_stack((
        np.ones_like(token_counts), layer_counts, token_counts, layer_counts * token_counts
    ))
    coefficients, _, rank, _ = np.linalg.lstsq(design, peaks_gib, rcond=None)
    if rank < 4:
        raise ValueError("completed points do not span all four fit terms")
    residuals = peaks_gib - design @ coefficients
    total = float(np.square(peaks_gib - peaks_gib.mean()).sum())
    r_squared = 1.0 if total == 0 else 1.0 - float(np.square(residuals).sum()) / total
    return {
        "W_gib": float(coefficients[0]),
        "a_gib_per_layer": float(coefficients[1]),
        "b_mib_per_token": float(coefficients[2] * 1024),
        "c_mib_per_layer_token": float(coefficients[3] * 1024),
        "residuals_gib": residuals.tolist(),
        "r_squared": r_squared,
        "points": len(points),
    }


def print_fit(result: Mapping[str, Any]) -> None:
    print(f"W: {result['W_gib']:.9f} GiB")
    print(f"a: {result['a_gib_per_layer']:.9f} GiB/layer")
    print(f"b: {result['b_mib_per_token']:.9f} MiB/token")
    print(f"c: {result['c_mib_per_layer_token']:.9f} MiB/(layer·token)")
    residuals = ", ".join(f"{value:.9f}" for value in result["residuals_gib"])
    print(f"residuals_gib: [{residuals}]")
    print(f"R²: {result['r_squared']:.9f}")
