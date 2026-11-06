#!/usr/bin/env python3
"""Sample passing student solutions and build rejection-sampled SFT data."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K = 8
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 20261105
DEFAULT_MODEL = "default_model"
DEFAULT_PER_TASK = 2
FAILURE_TAIL_CHARS = 400
UNREACHABLE_REQUEST_LIMIT = 3
HINT_PREFIX = "A previous attempt failed the tests with:\n"
HINT_SUFFIX = "\nWrite a corrected solution."
V5_MANIFESTS = ROOT / "data/finetune/v5"


def load_helpers() -> tuple[Any, Any, Any, Any]:
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import ft_build_dataset
    import ft_eval
    import ft_tasks_check
    import ft_tasks_to_sft
    return ft_eval, ft_tasks_check, ft_tasks_to_sft, ft_build_dataset


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="sample and test student completions")
    sample.add_argument("directories", nargs="+", type=Path)
    sample.add_argument("--k", type=int, default=DEFAULT_K)
    sample.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    sample.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--endpoint", default=None)
    sample.add_argument("--limit", type=int)
    sample.add_argument("--only", metavar="ID")
    hinted = commands.add_parser("hint-sample", help="resample tasks with no passing samples")
    hinted.add_argument("--samples", type=Path, required=True)
    hinted.add_argument("directories", nargs="+", type=Path)
    hinted.add_argument("--k", type=int, default=4)
    hinted.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    hinted.add_argument("--seed", type=int, default=DEFAULT_SEED)
    hinted.add_argument("--out", type=Path, required=True)
    hinted.add_argument("--endpoint", default=None)
    build = commands.add_parser("build", help="build SFT rows from passing samples")
    build.add_argument("--samples", type=Path, nargs="+", action="extend", required=True)
    build.add_argument("--tasks", nargs="+", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK)
    build.add_argument("--max-pass-fraction", type=float, default=1.0)
    return parser


def collect_folders(directories: list[Path], only: str | None, limit: int | None,
                    checker: Any) -> list[Path]:
    folders: list[Path] = []
    seen: set[str] = set()
    for item in directories:
        directory = rooted(item).resolve()
        if not directory.is_dir():
            raise ValueError(f"task directory does not exist: {directory}")
        for folder in checker.read_task_folders(directory):
            if folder.name in seen:
                raise ValueError(f"duplicate task id across directories: {folder.name}")
            seen.add(folder.name)
            folders.append(folder)
    folders.sort(key=lambda path: path.name)
    if only is not None:
        folders = [folder for folder in folders if folder.name == only]
        if not folders:
            raise ValueError(f"task id not found: {only}")
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be nonnegative")
        folders = folders[:limit]
    if not folders:
        raise ValueError("no task folders selected")
    return folders


def task_inputs(folder: Path, checker: Any, to_sft: Any) -> tuple[dict, str, dict]:
    meta = json.loads((folder / "meta.json").read_text())
    checker.validate_meta(folder, meta)
    prompt = to_sft.task_prompt(folder, meta)
    runner = checker.runner_task(folder, meta["language"])
    return meta, prompt, runner


def read_done_pairs(output: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not output.exists():
        return done
    with output.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                key = (row["task_id"], row["sample"])
                if not isinstance(key[0], str) or type(key[1]) is not int:
                    raise ValueError("task_id must be text and sample must be an integer")
                done.add(key)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid sample output at {output}:{line_number}: {error}") from error
    return done


def failure_text(value: Any) -> str:
    message = str(value).strip() or type(value).__name__
    return message[-FAILURE_TAIL_CHARS:]


def is_unreachable(error: Exception) -> bool:
    if not isinstance(error, RuntimeError) or "Cannot reach MLX server" not in str(error):
        return False
    cause = error.__cause__
    return cause is not None and not isinstance(cause, urllib.error.HTTPError)


def sample_one(folder: Path, meta: dict, prompt: str, runner: dict, index: int,
               args: argparse.Namespace, endpoint: str, helpers: tuple[Any, ...],
               test_python: Path | None, hinted: bool = False) -> tuple[dict, bool]:
    ft_eval, checker, _, _ = helpers
    started = time.perf_counter()
    reply = ""
    code = ""
    passed = False
    failure = ""
    unreachable = False
    try:
        response = ft_eval.request_model(prompt, endpoint, DEFAULT_MODEL, None,
                                         args.temperature, args.seed + index)
        if not isinstance(response, str):
            raise TypeError("MLX reply content must be text")
        reply = response
    except Exception as error:
        failure = failure_text(error)
        unreachable = is_unreachable(error)
    else:
        try:
            code = ft_eval.extract_code(reply)
            status, output = checker.run_candidate(ft_eval, runner, code, test_python)
            passed = status == 0 and bool(code.strip())
            if not passed:
                failure = failure_text(output or f"tests failed with exit code {status}")
        except Exception as error:
            failure = failure_text(error)
    record = {"task_id": meta["id"], "batch": meta.get("source", ""), "sample": index,
              "k": args.k, "passed": passed, "failure_tail": failure,
              "reply_length": len(reply), "seconds": round(time.perf_counter() - started, 3),
              "code": code}
    if hinted:
        record["hinted"] = True
    return record, unreachable


def setup_failure(folder: Path, meta: dict, index: int, k: int, error: Exception,
                  hinted: bool = False) -> dict:
    record = {"task_id": meta.get("id", folder.name), "batch": meta.get("source", ""),
              "sample": index, "k": k, "passed": False,
              "failure_tail": failure_text(error), "reply_length": 0, "seconds": 0.0,
              "code": ""}
    if hinted:
        record["hinted"] = True
    return record


def append_setup_failures(stream: Any, folder: Path, k: int, done: set[tuple[str, int]],
                          error: Exception, hinted: bool = False) -> int:
    meta = json.loads((folder / "meta.json").read_text())
    written = 0
    for index in range(k):
        if (meta["id"], index) in done:
            continue
        stream.write(json_line(setup_failure(folder, meta, index, k, error, hinted)))
        stream.flush()
        done.add((meta["id"], index))
        written += 1
    print(f"{meta['id']}: task setup failed: {failure_text(error)}", file=sys.stderr)
    return written


def json_line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"


def first_failure_hint(rows: list[dict]) -> str:
    failures = sorted((row for row in rows if not row["passed"]),
                      key=lambda row: row["sample"])
    if not failures:
        raise ValueError("hint task has no failing sample")
    failure = str(failures[0].get("failure_tail", "")).strip()[-FAILURE_TAIL_CHARS:]
    return failure or "No failure details were recorded."


def hinted_prompt(prompt: str, rows: list[dict]) -> str:
    return prompt + "\n\n" + HINT_PREFIX + first_failure_hint(rows) + HINT_SUFFIX


def sampling_command(args: argparse.Namespace, hinted: bool = False) -> int:
    try:
        if args.k < 1:
            raise ValueError("--k must be at least 1")
        if args.temperature < 0:
            raise ValueError("--temperature must be nonnegative")
        helpers = load_helpers()
        ft_eval, checker, to_sft, _ = helpers
        folders = collect_folders(args.directories, getattr(args, "only", None),
                                  getattr(args, "limit", None), checker)
        prior_samples: dict[str, list[dict]] = {}
        if hinted:
            prior_samples = read_sample_rows(args.samples)
            folders = [folder for folder in folders
                       if folder.name in prior_samples and
                       not any(row["passed"] for row in prior_samples[folder.name])]
        test_python = checker.python_with_test_dependencies(ft_eval)
        endpoint = args.endpoint or ft_eval.API
        output = rooted(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        done = read_done_pairs(output)
        unreachable_streak = 0
        written = 0
        with output.open("a", encoding="utf-8") as stream:
            for folder in folders:
                try:
                    meta, prompt, runner = task_inputs(folder, checker, to_sft)
                except Exception as error:
                    written += append_setup_failures(stream, folder, args.k, done, error, hinted)
                    continue
                if hinted:
                    prompt = hinted_prompt(prompt, prior_samples[meta["id"]])
                for index in range(args.k):
                    if (meta["id"], index) in done:
                        continue
                    record, unreachable = sample_one(folder, meta, prompt, runner, index,
                                                     args, endpoint, helpers, test_python, hinted)
                    stream.write(json_line(record))
                    stream.flush()
                    done.add((meta["id"], index))
                    written += 1
                    unreachable_streak = unreachable_streak + 1 if unreachable else 0
                    print(f"{meta['id']} sample {index}: {'PASS' if record['passed'] else 'FAIL'}")
                    if unreachable_streak >= UNREACHABLE_REQUEST_LIMIT:
                        print(f"MLX server unreachable for {unreachable_streak} consecutive requests; stopping.",
                              file=sys.stderr)
                        return 1
        print(f"wrote {written} new sample(s) to {output}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"RFT sampling failed: {error}", file=sys.stderr)
        return 1


def sample_command(args: argparse.Namespace) -> int:
    return sampling_command(args)


def hint_sample_command(args: argparse.Namespace) -> int:
    if args.k < 1:
        print("RFT sampling failed: --k must be at least 1", file=sys.stderr)
        return 1
    return sampling_command(args, hinted=True)


def read_sample_rows(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    seen: set[tuple[str, int]] = set()
    with rooted(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                task_id, index = row["task_id"], row["sample"]
                if not isinstance(task_id, str) or type(index) is not int or index < 0:
                    raise ValueError("task_id must be text and sample a nonnegative integer")
                if not isinstance(row.get("passed"), bool):
                    raise ValueError("passed must be a boolean")
                key = (task_id, index)
                if key in seen:
                    raise ValueError(f"duplicate sample pair {key}")
                seen.add(key)
                grouped.setdefault(task_id, []).append(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid sample row at {path}:{line_number}: {error}") from error
    for rows in grouped.values():
        rows.sort(key=lambda row: row["sample"])
    return grouped


def read_sample_files(paths: list[Path]) -> dict[str, list[dict]]:
    """Merge sample files, assigning each file a distinct sample-index range per task."""
    merged: dict[str, list[dict]] = {}
    offsets: dict[str, int] = {}
    for path in paths:
        grouped = read_sample_rows(path)
        for task_id, rows in grouped.items():
            offset = offsets.get(task_id, 0)
            span = max(max(int(row.get("k", 0)), row["sample"] + 1) for row in rows)
            total = offset + span
            merged.setdefault(task_id, []).extend(
                {**row, "sample": offset + row["sample"], "k": total} for row in rows
            )
            offsets[task_id] = total
    for rows in merged.values():
        rows.sort(key=lambda row: row["sample"])
    return merged


def selected_codes(rows: list[dict], per_task: int) -> list[tuple[int, str]]:
    unique: dict[str, int] = {}
    for row in rows:
        code = row.get("code", "")
        if row["passed"] and isinstance(code, str) and code.strip():
            normalized = code.strip()
            unique.setdefault(normalized, row["sample"])
    choices = sorted(((sample, code) for code, sample in unique.items()),
                     key=lambda item: (len(item[1]), item[0]))
    return choices[:per_task]


def valid_task_ids() -> set[str]:
    valid: set[str] = set()
    paths = sorted(V5_MANIFESTS.glob("*-manifest.json"))
    if not paths:
        raise ValueError(f"no v5 manifests found in {V5_MANIFESTS}")
    for path in paths:
        data = json.loads(path.read_text())
        ids = data.get("valid_task_ids")
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"invalid valid_task_ids in {path}")
        valid.update(ids)
    return valid


def row_for_sample(folder: Path, meta: dict, code: str, to_sft: Any,
                   ft_eval: Any, dataset: Any) -> dict:
    prompt = to_sft.task_prompt(folder, meta)
    fence = "c" if meta["language"] == "c" else "python"
    row = dataset.make_messages(prompt, dataset.code_fence(code, fence))
    row["messages"][0]["content"] = ft_eval.SYSTEM
    return row


def build_rows(folders: list[Path], samples: dict[str, list[dict]], per_task: int,
               helpers: tuple[Any, ...],
               selectable_samples: dict[str, list[dict]] | None = None
               ) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    ft_eval, checker, to_sft, dataset = helpers
    rows: dict[str, list[dict]] = {}
    stats: dict[str, dict] = {}
    for folder in folders:
        meta = json.loads((folder / "meta.json").read_text())
        checker.validate_meta(folder, meta)
        task_samples = samples.get(meta["id"], [])
        k = max((max(int(row.get("k", 0)), row["sample"] + 1) for row in task_samples), default=0)
        passes = sum(row["passed"] for row in task_samples)
        stats[meta["id"]] = {"k": k, "passes": passes,
                              "pass@1": passes / k if k else None,
                              "pass@k": bool(passes) if k else None}
        candidates = (selectable_samples.get(meta["id"], [])
                      if selectable_samples is not None else task_samples)
        chosen = selected_codes(candidates, per_task)
        if chosen:
            rows[meta["id"]] = [row_for_sample(folder, meta, code, to_sft, ft_eval, dataset)
                                for _, code in chosen]
    return rows, stats


def make_summary(stats: dict[str, dict]) -> dict:
    sampled = [info for info in stats.values() if info["k"] > 0]
    sample_ks = sorted({info["k"] for info in sampled})
    zero = sorted(task_id for task_id, info in stats.items()
                  if info["k"] > 0 and info["passes"] == 0)
    distribution = {"sample_k": sample_ks[0] if len(sample_ks) == 1 else None,
                    "sample_ks": sample_ks,
                    "tasks_with_0_passes": sum(info["passes"] == 0 for info in sampled),
                    "tasks_with_1_to_k_minus_1_passes": sum(
                        1 <= info["passes"] < info["k"] for info in sampled),
                    "tasks_with_k_passes": sum(
                        info["passes"] == info["k"] for info in sampled)}
    return {"per_task": stats, "distribution": distribution,
            "zero_pass_tasks": zero,
            "unsampled_tasks": sorted(task_id for task_id, info in stats.items() if info["k"] == 0)}


def write_outputs(out: Path, rows: dict[str, list[dict]], valid_ids: set[str],
                  stats: dict[str, dict], to_sft: Any) -> dict:
    train_ids = sorted(task_id for task_id in rows if task_id not in valid_ids)
    valid_ids_written = sorted(task_id for task_id in rows if task_id in valid_ids)
    train_rows = [row for task_id in train_ids for row in rows[task_id]]
    valid_rows = [row for task_id in valid_ids_written for row in rows[task_id]]
    train_data, valid_data = to_sft.jsonl_bytes(train_rows), to_sft.jsonl_bytes(valid_rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_bytes(train_data)
    (out / "valid.jsonl").write_bytes(valid_data)
    manifest = {"task_ids": sorted(rows), "train_task_ids": train_ids,
                "valid_task_ids": valid_ids_written,
                "counts": {"train_rows": len(train_rows), "valid_rows": len(valid_rows),
                           "train_tasks": len(train_ids), "valid_tasks": len(valid_ids_written)},
                "sha256": {"train.jsonl": hashlib.sha256(train_data).hexdigest(),
                           "valid.jsonl": hashlib.sha256(valid_data).hexdigest()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(make_summary(stats), indent=2) + "\n")
    return manifest


def run_leak_check() -> None:
    result = subprocess.run([sys.executable, str(ROOT / "tools/ft_leak_check.py")], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError("ft_leak_check.py failed")


def build_command(args: argparse.Namespace) -> int:
    try:
        if args.per_task < 1:
            raise ValueError("--per-task must be at least 1")
        if not math.isfinite(args.max_pass_fraction):
            raise ValueError("--max-pass-fraction must be finite")
        helpers = load_helpers()
        _, checker, to_sft, _ = helpers
        folders = collect_folders(args.tasks, None, None, checker)
        samples = read_sample_files(args.samples)
        from rft_filter import filter_by_pass_rate

        eligible = filter_by_pass_rate(samples, args.max_pass_fraction)
        rows, stats = build_rows(folders, samples, args.per_task, helpers, eligible)
        out = rooted(args.out)
        manifest = write_outputs(out, rows, valid_task_ids(), stats, to_sft)
        run_leak_check()
        print(f"wrote {manifest['counts']['train_rows']} train and "
              f"{manifest['counts']['valid_rows']} valid rows to {out}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"RFT build failed: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sample":
        return sample_command(args)
    if args.command == "hint-sample":
        return hint_sample_command(args)
    return build_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
