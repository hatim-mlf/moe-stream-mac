#!/usr/bin/env python3
"""Select the best saved RFT adapter and summarize a completed round."""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ft_night import parse_loss_lines

SCALES = ("20", "4")
TRAIN_SECONDS_RE = re.compile(r"training_seconds=([0-9]+(?:\.[0-9]+)?)")
PAIRED_RE = re.compile(
    r"^first_pass: base=([+-]?[0-9.]+), adapter=([+-]?[0-9.]+), "
    r"delta=([+-]?[0-9.]+), paired-bootstrap-95%=\(([^,]+), ([^)]+)\)"
)


def val_losses(log_path: Path) -> list[dict[str, float | int]]:
    parsed = parse_loss_lines(log_path.read_text(encoding="utf-8"))
    return [{"step": row["step"], "val_loss": row["val_loss"]}
            for row in parsed if "val_loss" in row]


def select_checkpoint(log_path: Path, checkpoints: Path, adapter: Path,
                      chosen_step_path: Path) -> int:
    losses = val_losses(log_path)
    saved = [row for row in losses
             if (checkpoints / f"{row['step']:07d}_adapters.safetensors").is_file()]
    if not saved:
        raise ValueError(f"no validation loss has a saved checkpoint in {checkpoints}")
    best = min(saved, key=lambda row: (row["val_loss"], row["step"]))
    source = checkpoints / f"{best['step']:07d}_adapters.safetensors"
    config_path = checkpoints / "adapter_config.json"
    if not source.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"best checkpoint or adapter_config.json is missing in {checkpoints}")
    adapter.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, adapter / "adapters.safetensors")
    config: Any = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"adapter config must be a JSON object: {config_path}")
    config["adapter_path"] = str(adapter)
    (adapter / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    chosen_step_path.write_text(f"{best['step']}\n", encoding="utf-8")
    print(f"selected step {best['step']} with validation loss {best['val_loss']}")
    return int(best["step"])


def parse_paired_line(path: Path) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PAIRED_RE.match(line)
        if match:
            base, adapter, delta, low, high = match.groups()
            return {"line": line, "base": float(base), "adapter": float(adapter),
                    "delta": float(delta), "ci95": [float(low), float(high)]}
    raise ValueError(f"first_pass paired line missing from {path}")


def adapter_metrics(run_dir: Path, scale: str) -> dict[str, Any]:
    scale_dir = run_dir / f"scale{scale}"
    log_path = scale_dir / "train.log"
    log_text = log_path.read_text(encoding="utf-8")
    seconds_match = TRAIN_SECONDS_RE.search(log_text)
    if not seconds_match:
        raise ValueError(f"training_seconds missing from {log_path}")
    step = int((scale_dir / "chosen-step.txt").read_text(encoding="utf-8").strip())
    losses = val_losses(log_path)
    chosen = next((row for row in losses if row["step"] == step), None)
    if chosen is None:
        raise ValueError(f"chosen step {step} has no validation loss in {log_path}")
    return {"scale": float(scale), "train_seconds": float(seconds_match.group(1)),
            "chosen_step": step, "chosen_val_loss": chosen["val_loss"],
            "val_losses": losses,
            "paired": parse_paired_line(scale_dir / "paired.txt")}


def write_metrics(data_dir: Path, run_dir: Path, name: str, stamp: str) -> Path:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = {"name": name, "started_utc": stamp,
               "dataset": {"path": str(data_dir), "counts": manifest["counts"]},
               "adapters": {f"scale{scale}": adapter_metrics(run_dir, scale)
                            for scale in SCALES}}
    output = run_dir / "metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return output


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select-checkpoint")
    select.add_argument("--log", type=Path, required=True)
    select.add_argument("--checkpoints", type=Path, required=True)
    select.add_argument("--adapter", type=Path, required=True)
    select.add_argument("--chosen-step", type=Path, required=True)
    metrics = commands.add_parser("write-metrics")
    metrics.add_argument("--data", type=Path, required=True)
    metrics.add_argument("--run-dir", type=Path, required=True)
    metrics.add_argument("--name", required=True)
    metrics.add_argument("--stamp", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.command == "select-checkpoint":
        select_checkpoint(args.log, args.checkpoints, args.adapter, args.chosen_step)
    else:
        output = write_metrics(args.data, args.run_dir, args.name, args.stamp)
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
