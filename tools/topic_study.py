#!/usr/bin/env python3
"""Measure how well topic-specific expert sets generalize across routing traces.

Usage:
  .venv/bin/python tools/topic_study.py [--budgets 4 6 8 12 16] [--expert-mb 10.0]

The study uses leave-one-out coding traces, a mixed non-coding baseline, a
per-trace upper bound, and a cross-topic check. Results are written to a
timestamped directory under ``results/``.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from expert_profile import load_traces


LAYERS = 48
EXPERTS = 256
TOP_K = 8
RECORD_BYTES = 4 * (1 + TOP_K)
NON_CODING_NAMES = (
    "trace122.bin",
    "trace122_explain.bin",
    "trace122_chat.bin",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--budgets", type=float, nargs="+", default=[4, 6, 8, 12, 16])
    parser.add_argument("--expert-mb", type=float, default=10.0)
    args = parser.parse_args()
    if args.expert_mb <= 0 or any(budget <= 0 for budget in args.budgets):
        parser.error("budgets and expert-mb must be positive")
    return args


def read_counts(path):
    return load_traces([path], LAYERS, EXPERTS, TOP_K)


def top_pairs(counts, size):
    values = counts.ravel()
    used = np.flatnonzero(values > 0)
    order = np.lexsort((used % EXPERTS, used // EXPERTS, -values[used]))
    selected = used[order[:size]]
    mask = np.zeros(values.size, dtype=bool)
    mask[selected] = True
    return mask


def hit_rate(counts, selected):
    values = counts.ravel()
    total = int(values.sum(dtype=np.int64))
    return int(values[selected].sum(dtype=np.int64)) / total if total else 0.0


def average_hit(counts_by_path, paths, selected):
    return float(np.mean([hit_rate(counts_by_path[path], selected) for path in paths]))


def build_metrics(budgets, expert_mb, coding_paths, other_paths, counts_by_path):
    shape = (LAYERS, EXPERTS)
    coding_total = sum(
        (counts_by_path[path] for path in coding_paths), np.zeros(shape, dtype=np.int64)
    )
    other_total = sum(
        (counts_by_path[path] for path in other_paths), np.zeros(shape, dtype=np.int64)
    )
    rows = []
    for budget in budgets:
        requested = math.floor(budget * 1000 / expert_mb)
        mixed_set = top_pairs(other_total, requested)
        coding_set = top_pairs(coding_total, requested)
        folds = []
        upper = []
        for held_out in coding_paths:
            train = coding_total - counts_by_path[held_out]
            folds.append(hit_rate(counts_by_path[held_out], top_pairs(train, requested)))
            own_set = top_pairs(counts_by_path[held_out], requested)
            upper.append(hit_rate(counts_by_path[held_out], own_set))
        rows.append({
            "budget_gb": budget,
            "requested_pairs": requested,
            "coding_loo": {"mean": float(np.mean(folds)), "min": min(folds), "max": max(folds)},
            "mixed_set_baseline_mean": average_hit(counts_by_path, coding_paths, mixed_set),
            "upper_bound_mean": float(np.mean(upper)),
            "coding_set_on_non_coding_mean": average_hit(counts_by_path, other_paths, coding_set),
        })
    distinct = int(np.count_nonzero(coding_total))
    per_trace = [int(np.count_nonzero(counts_by_path[path])) for path in coding_paths]
    return rows, distinct, float(np.mean(per_trace)), per_trace


def discover_traces(root):
    coding_dir = root / "engine" / "build" / "coding"
    core_paths = sorted(coding_dir.glob("*.trace"))
    incomplete = [path for path in core_paths if path.stat().st_size % RECORD_BYTES]
    for path in incomplete:
        print(f"Skipping incomplete trace: {path.relative_to(root)}")
    core_paths = [path for path in core_paths if path not in incomplete]
    if len(core_paths) < 4:
        raise SystemExit(f"need at least 4 complete coding/*.trace files; found {len(core_paths)}")
    coding_paths = list(core_paths)
    extra_coding = root / "engine" / "build" / "trace122_code.bin"
    if extra_coding.is_file() and extra_coding.stat().st_size % RECORD_BYTES == 0:
        coding_paths.append(extra_coding)
    other_paths = [root / "engine" / "build" / name for name in NON_CODING_NAMES]
    missing = [path for path in other_paths if not path.is_file()]
    if missing:
        raise SystemExit("missing non-coding trace(s): " + ", ".join(str(path) for path in missing))
    return coding_paths, other_paths, len(core_paths)


def print_report(rows, coding_paths, core_count, distinct, average_distinct):
    extra_count = len(coding_paths) - core_count
    print(f"Coding traces used: {len(coding_paths)} ({core_count} coding/*.trace"
          f" + {extra_count} separate coding-like trace)")
    print(f"Distinct (layer, expert) pairs used by all coding traces: {distinct:,}")
    print(f"Distinct (layer, expert) pairs per coding trace (mean): {average_distinct:.1f}")
    print("Budget GB | Coding LOO mean (min–max) | Mixed-set baseline | Upper bound | Coding set on non-coding")
    for row in rows:
        loo = row["coding_loo"]
        print(f"{row['budget_gb']:9g} | {loo['mean']:.2%} ({loo['min']:.2%}–{loo['max']:.2%})"
              f" | {row['mixed_set_baseline_mean']:.2%}"
              f" | {row['upper_bound_mean']:.2%}"
              f" | {row['coding_set_on_non_coding_mean']:.2%}")


def write_metrics(root, args, coding_paths, other_paths, rows, distinct, average_distinct, per_trace):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / "results" / f"{timestamp}-topic-study"
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "coding_trace_count": len(coding_paths),
        "coding_trace_paths": [str(path.relative_to(root)) for path in coding_paths],
        "non_coding_trace_paths": [str(path.relative_to(root)) for path in other_paths],
        "layers": LAYERS,
        "experts_per_layer": EXPERTS,
        "top_k": TOP_K,
        "expert_mb": args.expert_mb,
        "distinct_layer_expert_pairs_all_coding": distinct,
        "distinct_layer_expert_pairs_per_trace_mean": average_distinct,
        "distinct_layer_expert_pairs_per_trace": per_trace,
        "budgets": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"JSON: {out_dir.relative_to(root) / 'metrics.json'}")


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    coding_paths, other_paths, core_count = discover_traces(root)
    counts_by_path = {path: read_counts(path) for path in coding_paths + other_paths}
    rows, distinct, average_distinct, per_trace = build_metrics(
        args.budgets, args.expert_mb, coding_paths, other_paths, counts_by_path
    )
    print_report(rows, coding_paths, core_count, distinct, average_distinct)
    write_metrics(root, args, coding_paths, other_paths, rows, distinct, average_distinct, per_trace)


if __name__ == "__main__":
    main()
