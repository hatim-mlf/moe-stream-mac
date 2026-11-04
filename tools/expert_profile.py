"""Build and inspect per-expert usage profiles from routing traces.

Usage:
  .venv/bin/python tools/expert_profile.py build --layers 48 --experts 256 --top-k 8 --out <profile.npy> <trace> [<trace> ...]
  .venv/bin/python tools/expert_profile.py show <profile.npy> [--expert-mb 10.0]

Import ``load_traces(paths, layers, experts, top_k)`` to build the counts
array, or ``coverage(counts, fraction)`` to measure uses covered by the
hottest fraction of layer/expert pairs.
"""

import argparse
import math
from pathlib import Path

import numpy as np


def load_traces(paths, layers, experts, top_k) -> np.ndarray:
    """Read routing traces and return int64 counts indexed by layer and expert."""
    if layers <= 0 or experts <= 0 or top_k <= 0:
        raise ValueError("layers, experts, and top_k must all be positive")
    counts = np.zeros((layers, experts), dtype=np.int64)
    record_bytes = 4 * (1 + top_k)
    for path in paths:
        path = Path(path)
        size = path.stat().st_size
        if size % record_bytes:
            raise ValueError(
                f"{path}: file size {size} is not a multiple of "
                f"the {record_bytes}-byte record size"
            )
        records = np.fromfile(path, dtype="<i4").reshape(-1, 1 + top_k)
        bad_layers = np.flatnonzero((records[:, 0] < 0) | (records[:, 0] >= layers))
        if bad_layers.size:
            index = int(bad_layers[0])
            value = int(records[index, 0])
            raise ValueError(f"{path}: record {index} has layer {value}, expected [0, {layers})")
        expert_ids = records[:, 1:]
        bad_rows = np.flatnonzero(np.any((expert_ids < 0) | (expert_ids >= experts), axis=1))
        if bad_rows.size:
            index = int(bad_rows[0])
            value = int(expert_ids[index, np.flatnonzero(
                (expert_ids[index] < 0) | (expert_ids[index] >= experts)
            )[0]])
            raise ValueError(
                f"{path}: record {index} has expert id {value}, expected [0, {experts})"
            )
        np.add.at(counts, (records[:, :1], expert_ids), 1)
    return counts


def coverage(counts, fraction) -> float:
    """Return the fraction of all uses covered by the hottest pair fraction."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    values = np.asarray(counts).ravel()
    if values.size == 0:
        raise ValueError("counts must contain at least one layer/expert pair")
    total = int(values.sum(dtype=np.int64))
    if total == 0:
        return 0.0
    keep = math.ceil(values.size * fraction)
    if keep == 0:
        return 0.0
    hottest = np.partition(values, values.size - keep)[-keep:]
    return int(hottest.sum(dtype=np.int64)) / total


def _build(args):
    counts = load_traces(args.traces, args.layers, args.experts, args.top_k)
    out = Path(args.out)
    np.save(out, counts)
    saved_path = str(out) if str(out).endswith(".npy") else f"{out}.npy"
    records = int(counts.sum(dtype=np.int64)) // args.top_k
    tokens = records // args.layers if records % args.layers == 0 else records / args.layers
    print(f"Records: {records:,}")
    print(f"Tokens: {tokens}")
    print(f"Distinct (layer, expert) pairs used: {np.count_nonzero(counts):,}")
    print(f"Out: {saved_path}")


def _show(args):
    counts = np.load(args.profile, allow_pickle=False)
    if counts.ndim != 2 or counts.size == 0:
        raise ValueError(f"{args.profile}: profile must be a nonempty 2D counts array")
    total = int(counts.sum(dtype=np.int64))
    print(f"Total uses: {total:,}")
    print(f"Distinct (layer, expert) pairs used: {np.count_nonzero(counts):,}")
    print("Coverage:")
    print(f"{'Top pairs':>10} {'Pairs':>10} {'Uses covered':>14} {'Size (GB)':>11}")
    for percent in (1, 5, 10, 15, 20, 30):
        pairs = math.ceil(counts.size * percent / 100)
        covered = coverage(counts, percent / 100)
        size_gb = pairs * args.expert_mb / 1000
        print(f"{percent:>9}% {pairs:>10,} {covered:>13.2%} {size_gb:>11.2f}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a profile from routing traces")
    build.add_argument("--layers", type=int, required=True)
    build.add_argument("--experts", type=int, required=True)
    build.add_argument("--top-k", type=int, required=True)
    build.add_argument("--out", required=True)
    build.add_argument("traces", nargs="+", help="routing trace files")
    build.set_defaults(func=_build)
    show = commands.add_parser("show", help="show profile coverage and memory sizes")
    show.add_argument("profile")
    show.add_argument("--expert-mb", type=float, default=10.0)
    show.set_defaults(func=_show)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "show" and args.expert_mb < 0:
            raise ValueError("expert_mb must be nonnegative")
        args.func(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
