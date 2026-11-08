#!/usr/bin/env python3
"""Prepare, run, and score the sequential Qwen3.8 base-versus-RFT coding benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = Path("/Users/hatim/models/Qwen3.8-27B-official-mlx-4bit")
RFT_RUN = ROOT / "results/20261107T141610Z-rft-qwen38-rft-k2"
RFT_ADAPTER = RFT_RUN / "scale4/adapter"
# Third candidate (2026-11-08): Solstice-AI TURBO Fable Cold Fusion, 4-bit oQ4e MLX, used as-is without adapter.
TURBO_MODEL = Path("/Users/hatim/models/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-MLX-oQ4e-1M")
BASE_MANIFEST = ROOT / "results/q38-official-manifest.json"
SERVER = ROOT / ".venv/bin/mlx_lm.server"
PY3 = Path("/opt/homebrew/bin/python3")
CPP = Path("/usr/bin/clang++")
SANDBOX = Path("/usr/bin/sandbox-exec")
API = "https://datasets-server.huggingface.co/rows"
DATASET = "deepmind/code_contests"
PORT = 8080
SEED = 20261107
MAX_TOKENS = 4096
BOOTSTRAPS = 10000
# The user asked to let the active single-model run proceed without a memory stop threshold.
FREE_RAM_FLOOR = 0
OUT_LIMIT = 8 * 1024 * 1024
TIERS = {
    "easy": ["1579_A", "1581_A", "1582_A", "1607_C", "1619_B"],
    "medium": ["1575_D", "1580_A", "1594_D", "1604_D", "1622_C"],
    "hard": ["1591_D", "1598_E", "1600_E", "1613_D", "1620_E"],
}
LANGS = {
    "cpp17": {"id": 2, "ext": ".cpp", "label": "C++17"},
    "python3": {"id": 3, "ext": ".py", "label": "Python 3"},
}
SYSTEM = (
    "You are a competitive-programming code generator. Produce a correct, efficient, "
    "complete program. Read input from stdin and write output to stdout. Use the language "
    "and version requested by the user. Return source code only, without Markdown or "
    "explanation. Follow the statement exactly."
)


def sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def command_output(command: list[str], timeout: float = 30) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return result.stdout


def memory_free() -> int | None:
    try:
        output = command_output(["/usr/bin/memory_pressure"], timeout=10)
        match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def base_identity() -> dict[str, Any]:
    manifest = load_json(BASE_MANIFEST)
    if Path(manifest["path"]).resolve() != BASE_MODEL.resolve():
        raise RuntimeError("The saved official model manifest points to a different base.")
    if (RFT_RUN / "scale4/chosen-step.txt").read_text().strip() != "152":
        raise RuntimeError("Expected the selected RFT checkpoint to be step 152.")
    adapter_files = []
    for path in sorted(RFT_ADAPTER.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            data = path.read_bytes()
            adapter_files.append({
                "path": path.relative_to(RFT_ADAPTER).as_posix(),
                "size_bytes": len(data),
                "sha256": sha(data),
            })
    adapter_identity = json.dumps(adapter_files, sort_keys=True, separators=(",", ":"))
    return {
        "base": {
            "path": str(BASE_MODEL),
            "directory_sha256": manifest["directory_sha256"],
        },
        "rft": {
            "base_path": str(BASE_MODEL),
            "base_directory_sha256": manifest["directory_sha256"],
            "adapter_path": str(RFT_ADAPTER),
            "adapter_directory_sha256": sha(adapter_identity),
            "adapter_files": adapter_files,
            "training_run": str(RFT_RUN.relative_to(ROOT)),
            "selected_step": 152,
        },
    }


def fetch_test_split() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        query = urllib.parse.urlencode({
            "dataset": DATASET,
            "config": "default",
            "split": "test",
            "offset": offset,
            "length": 100,
        })
        request = urllib.request.Request(
            f"{API}?{query}", headers={"User-Agent": "qwen38-code-comparison/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        total = int(payload["num_rows_total"])
        page = payload.get("rows", [])
        if not page:
            break
        rows.extend(item["row"] for item in page)
        offset += len(page)
    if total != len(rows):
        raise RuntimeError(f"Dataset API returned {len(rows)} rows, expected {total}.")
    return rows, {
        "dataset": DATASET,
        "split": "test",
        "rows": total,
        "api": API,
        "retrieved_utc": now(),
    }


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("name", "")).split(".", 1)[0].strip()


def test_cases(row: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group_name in ("public_tests", "private_tests", "generated_tests"):
        group = row.get(group_name) or {}
        inputs, outputs = group.get("input") or [], group.get("output") or []
        if len(inputs) != len(outputs):
            raise ValueError(f"{task_id(row)} has unpaired {group_name} data.")
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            if not isinstance(inp, str) or not isinstance(out, str):
                raise ValueError(f"{task_id(row)} has non-text tests.")
            result.append({"source": group_name, "source_index": i,
                           "input": inp, "expected": out})
    if not result:
        raise ValueError(f"{task_id(row)} has no tests.")
    return result


def sandbox_profile(temp: Path) -> str:
    # Generated programs cannot access this user's home, use the network, or write elsewhere.
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        "(deny file-read* (subpath \"/Users/hatim\"))\n"
        f"(deny file-write* (require-not (subpath \"{temp}\")))\n"
    )


def set_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (16, 17))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUT_LIMIT, OUT_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def run_program(
    command: list[str], temp: Path, profile: str, inp: str, timeout: float
) -> dict[str, Any]:
    out_path, err_path = temp / "stdout.bin", temp / "stderr.bin"
    env = os.environ.copy()
    env.update({
        "HOME": str(temp), "TMPDIR": str(temp), "XDG_CACHE_HOME": str(temp),
        "PATH": "/usr/bin:/bin:/opt/homebrew/bin", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1", "LC_ALL": "C", "OMP_NUM_THREADS": "1",
    })
    try:
        with out_path.open("wb") as out, err_path.open("wb") as err:
            proc = subprocess.run(
                [str(SANDBOX), "-p", profile, *command],
                cwd=temp, env=env, input=inp.encode("utf-8"),
                stdout=out, stderr=err, timeout=timeout, check=False,
                preexec_fn=set_limits,
            )
        size = out_path.stat().st_size
        output = out_path.read_bytes()[:OUT_LIMIT + 1].decode("utf-8", errors="replace")
        stderr = err_path.read_bytes()[:65536].decode("utf-8", errors="replace")
        if size > OUT_LIMIT:
            return {"status": "output_limit", "returncode": proc.returncode,
                    "stdout": output, "stderr": stderr}
        return {
            "status": "ok" if proc.returncode == 0 else "nonzero_exit",
            "returncode": proc.returncode, "stdout": output, "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "stdout": "",
                "stderr": f"Timed out after {timeout:.1f}s."}
    except Exception as exc:
        return {"status": "runner_error", "returncode": None, "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}"}


def write_bits_compat(temp: Path) -> Path:
    """Provide GCC's common umbrella header when compiling portable C++17 with Apple Clang."""
    include_root = temp / "compat"
    bits_dir = include_root / "bits"
    bits_dir.mkdir(parents=True, exist_ok=True)
    headers = (
        "algorithm array atomic bitset cassert cctype cerrno cfenv cfloat ciso646 "
        "climits clocale cmath codecvt complex csetjmp csignal cstdarg cstddef cstdint "
        "cstdio cstdlib cstring ctgmath ctime cwchar cwctype deque exception fstream "
        "functional future initializer_list iomanip ios iosfwd iostream istream iterator "
        "limits list locale map memory mutex new numeric ostream queue random ratio regex "
        "scoped_allocator set sstream stack stdexcept streambuf string string_view strstream "
        "system_error thread tuple typeindex typeinfo type_traits "
        "unordered_map unordered_set utility valarray variant vector"
    ).split()
    content = "#pragma once\n" + "".join(f"#include <{name}>\n" for name in headers)
    (bits_dir / "stdc++.h").write_text(content, encoding="utf-8")
    return include_root


def ref_candidates(row: dict[str, Any], language_id: int) -> list[str]:
    solutions = row.get("solutions") or {}
    return [
        code for lang, code in zip(solutions.get("language") or [],
                                   solutions.get("solution") or [])
        if lang == language_id and isinstance(code, str) and code.strip()
    ]


def validate_reference(
    code: str, lang: str, cases: list[dict[str, Any]], time_limit: float
) -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="qwen38-ref-", dir="/private/tmp") as raw:
        temp = Path(raw)
        profile = sandbox_profile(temp)
        if lang == "cpp17":
            source, binary = temp / "ref.cpp", temp / "ref"
            source.write_text(code, encoding="utf-8")
            include_root = write_bits_compat(temp)
            # Compilation is isolated in a disposable directory; execution is sandboxed.
            proc = subprocess.run(
                [str(CPP), "-std=c++17", "-O2", "-pipe", "-I", str(include_root),
                 str(source), "-o", str(binary)],
                cwd=temp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=120, check=False,
            )
            if proc.returncode:
                return False, {"reason": "compile", "log": proc.stdout[-3000:]}
            command = [str(binary)]
        else:
            source = temp / "ref.py"
            source.write_text(code, encoding="utf-8")
            command = [str(PY3), "-I", "-B", "-S", str(source)]
        timeout = min(15.0, max(4.0, time_limit * 3 + 1))
        for i, case in enumerate(cases):
            got = run_program(command, temp, profile, case["input"], timeout)
            if got["status"] != "ok":
                return False, {"reason": got["status"], "test": i,
                               "stderr": got["stderr"][-1500:]}
            if got["stdout"].split() != case["expected"].split():
                return False, {
                    "reason": "oracle mismatch", "test": i,
                    "expected": case["expected"][:300], "actual": got["stdout"][:300],
                }
        return True, {"reason": "all cases match", "checked": len(cases)}


def prepare(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run folder already exists: {run_dir}")
    for required in (BASE_MODEL, RFT_ADAPTER, BASE_MANIFEST, SERVER, PY3, CPP, SANDBOX):
        if not required.exists():
            raise FileNotFoundError(f"Required local file/tool is missing: {required}")
    rows, dataset = fetch_test_split()
    row_map = {task_id(row): (i, row) for i, row in enumerate(rows)}
    selected: list[dict[str, Any]] = []
    refs_by_id: dict[str, dict[str, str]] = {}
    refs_evidence: dict[str, Any] = {}
    used_contests: set[int] = set()
    for tier, ids in TIERS.items():
        for problem_id in ids:
            if problem_id not in row_map:
                raise RuntimeError(f"Frozen problem {problem_id} not found in test split.")
            row_index, row = row_map[problem_id]
            rating = int(row.get("cf_rating") or 0)
            expected_tier = (
                "easy" if 800 <= rating <= 1200
                else "medium" if 1300 <= rating <= 1800
                else "hard" if rating >= 1900
                else None
            )
            langs = set((row.get("solutions") or {}).get("language") or [])
            if (expected_tier != tier or not {2, 3}.issubset(langs)
                    or "interactive" in [str(t).lower() for t in row.get("cf_tags", [])]
                    or row.get("input_file") or row.get("output_file")):
                raise RuntimeError(f"Frozen task {problem_id} fails its preregistered eligibility.")
            contest = int(row.get("cf_contest_id") or 0)
            if contest in used_contests:
                raise RuntimeError(f"Selected task set repeats contest {contest}.")
            used_contests.add(contest)
            cases = test_cases(row)
            time_limit = float((row.get("time_limit") or {}).get("seconds", 2))
            references: dict[str, str] = {}
            evidence: dict[str, Any] = {"test_count": len(cases), "languages": {}}
            for lang, spec in LANGS.items():
                accepted = None
                failures = []
                for rank, code in enumerate(ref_candidates(row, spec["id"])[:10]):
                    valid, detail = validate_reference(code, lang, cases, time_limit)
                    if valid:
                        accepted = code
                        evidence["languages"][lang] = {
                            "candidate_rank": rank, "source_sha256": sha(code), **detail,
                        }
                        break
                    failures.append({"rank": rank, **detail})
                if accepted is None:
                    raise RuntimeError(
                        f"No reference for {problem_id}/{lang} matched every frozen test: "
                        f"{json.dumps(failures[:3], ensure_ascii=False)}"
                    )
                references[lang] = accepted
            refs_by_id[problem_id] = references
            refs_evidence[problem_id] = evidence
            selected.append({
                "task_id": problem_id, "name": row["name"], "tier": tier, "rating": rating,
                "contest": contest, "row_index": row_index, "row": row,
                "cases": cases, "refs": references, "ref_evidence": evidence,
            })
            print(f"[prepare] references validated {problem_id} ({len(cases)} inputs)",
                  flush=True)

    run_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for problem_index, item in enumerate(selected):
        title = item["name"].split(".", 1)[-1]
        slug = re.sub(r"[^a-z0-9]+", "_",
                      f"{item['task_id']}_{title}".lower()).strip("_")[:100]
        task_dir = run_dir / "tasks" / item["tier"] / slug
        task_dir.mkdir(parents=True)
        desc = str(item["row"].get("description") or "").strip()
        tests_text = "".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n"
                             for c in item["cases"])
        (task_dir / "tests.jsonl").write_text(tests_text, encoding="utf-8")
        ref_dir = task_dir / "references"
        ref_dir.mkdir()
        for lang, code in item["refs"].items():
            (ref_dir / f"reference{LANGS[lang]['ext']}").write_text(code, encoding="utf-8")
        prompts = {}
        for lang, spec in LANGS.items():
            prompt = (
                f"Solve the following programming problem in {spec['label']}.\n"
                "Read from standard input and write the required output to standard output.\n\n"
                f"Problem:\n{desc}\n"
            )
            path = task_dir / f"prompt_{lang}.txt"
            path.write_text(prompt, encoding="utf-8")
            prompts[lang] = {
                "path": str(path.relative_to(run_dir)), "sha256": sha(prompt),
            }
        task_meta = {
            "task_id": item["task_id"], "name": item["name"], "tier": item["tier"],
            "rating": item["rating"], "contest": item["contest"],
            "source_row_index": item["row_index"], "description": desc,
            "description_sha256": sha(desc),
            "tests_path": str((task_dir / "tests.jsonl").relative_to(run_dir)),
            "tests_sha256": sha(tests_text), "test_count": len(item["cases"]),
            "test_counts_by_source": {
                group: len((item["row"].get(group) or {}).get("input") or [])
                for group in ("public_tests", "private_tests", "generated_tests")
            },
            "time_limit_seconds": float((item["row"].get("time_limit") or {}).get("seconds", 2)),
            "prompts": prompts, "reference_evidence": item["ref_evidence"],
            "reference_hashes": {lang: sha(code) for lang, code in item["refs"].items()},
            "problem_index": problem_index,
        }
        write_json(task_dir / "task.json", task_meta)
        records.append({key: task_meta[key] for key in (
            "task_id", "name", "tier", "rating", "contest", "source_row_index",
            "tests_path", "tests_sha256", "test_count", "test_counts_by_source",
            "prompts", "reference_hashes", "problem_index",
        )})

    training_root = ROOT / "data/finetune/qwen38-rft-k2"
    source_files = [p for p in training_root.rglob("*") if p.is_file()
                    and p.suffix.lower() in {".jsonl", ".json", ".md", ".txt"}]
    exact_matches = []
    for path in source_files:
        content = path.read_text(encoding="utf-8", errors="ignore").lower()
        for item in selected:
            if item["task_id"].lower() in content or item["name"].lower() in content:
                exact_matches.append({"file": str(path.relative_to(ROOT)),
                                      "task_id": item["task_id"]})
    versions = {
        "python": sys.version, "platform": platform.platform(),
        "compiler": command_output([str(CPP), "--version"]).splitlines()[:2],
        "mlx_lm": command_output([
            str(ROOT / ".venv/bin/python"), "-c",
            "import mlx_lm; print(getattr(mlx_lm, '__version__', 'unknown'))",
        ]).strip(),
    }
    manifest = {
        "status": "prepared", "created_utc": now(),
        "run_dir": str(run_dir.relative_to(ROOT)), "dataset": dataset,
        "models": base_identity(), "environment": versions,
        "protocol": {
            "system_prompt": SYSTEM, "system_prompt_sha256": sha(SYSTEM),
            "temperature": 0, "seed_formula": "20261107 + problem_index*2 + language_index",
            "max_tokens": MAX_TOKENS, "thinking": False, "attempts": 1,
            "repair_feedback": False, "checklist": False, "plan_first": False,
            "prompt_cache_size": 0, "prompt_cache_bytes": 0,
            "prompt_concurrency": 1, "decode_concurrency": 1,
            "languages": {key: spec["label"] for key, spec in LANGS.items()},
            "primary_metric": "task-language pass@1; all frozen cases must pass",
            "bootstrap": {"samples": BOOTSTRAPS, "seed": SEED,
                          "cluster": "problem; both language cases stay together"},
            "sandbox": "network denied; reads under /Users/hatim denied; writes confined to temp",
        },
        "task_selection": {
            "tiers": {"easy": "800-1200", "medium": "1300-1800", "hard": "1900+"},
            "tasks": records, "reference_validation": refs_evidence,
            "rft_data_exact_id_title_audit": {
                "files_checked": len(source_files), "matches": exact_matches,
                "interpretation": "Exact ID/title scan only; it does not test semantic or pretraining overlap.",
            },
        },
        "state": {"base_generated": False, "base_scored": False,
                  "rft_generated": False, "rft_scored": False},
        "memory_before_inference_free_percent": memory_free(),
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def assert_no_server() -> None:
    listing = command_output(["/bin/ps", "-axo", "pid=,command="])
    active = [line.strip() for line in listing.splitlines()
              if "mlx_lm.server" in line and "qwen38_code_compare.py" not in line]
    if active:
        raise RuntimeError("Refusing to load a model while another local server is active:\n"
                           + "\n".join(active))
    result = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-iTCP:8080", "-sTCP:LISTEN"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        raise RuntimeError(f"Port {PORT} is already in use:\n{result.stdout}")


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
    time.sleep(2)
    assert_no_server()


def wait_server(proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 300
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"MLX server exited with {proc.returncode} during startup.")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"MLX server readiness timed out: {last}")


def extract_code(reply: str) -> tuple[str, str]:
    fence = chr(96) * 3
    pattern = re.escape(fence) + r"[A-Za-z0-9_+-]*\s*\n(.*?)\n" + re.escape(fence)
    blocks = re.findall(pattern, reply, flags=re.DOTALL)
    if len(blocks) == 1:
        return blocks[0].strip() + "\n", "one_fenced_block"
    if len(blocks) > 1:
        return "", "multiple_fenced_blocks"
    return reply.strip() + "\n", "raw_source"


def result_dir(run_dir: Path, task: dict[str, Any], model: str, lang: str) -> Path:
    return run_dir / task["tests_path"].split("/tests.jsonl")[0] / model / lang


def generate(run_dir: Path, manifest: dict[str, Any], model: str) -> None:
    assert_no_server()
    model_dir = run_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "server.log"
    args = [
        str(SERVER), "--model", str(TURBO_MODEL if model == "turbo" else BASE_MODEL), "--host", "127.0.0.1",
        "--port", str(PORT), "--prompt-cache-size", "0", "--prompt-cache-bytes", "0",
        "--prompt-concurrency", "1", "--decode-concurrency", "1",
    ]
    if model == "rft":
        args.extend(["--adapter-path", str(RFT_ADAPTER)])
    proc: subprocess.Popen[bytes] | None = None
    with log_path.open("ab") as log:
        log.write((f"\n[{now()}] starting {model}: {' '.join(args)}\n").encode())
        log.flush()
        proc = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            wait_server(proc)
            free = memory_free()
            print(f"[{model}] server ready; system memory free {free}%.", flush=True)
            if free is not None and free < FREE_RAM_FLOOR:
                raise MemoryError(f"System memory free fell below {FREE_RAM_FLOOR}%.")
            total = len(manifest["task_selection"]["tasks"]) * len(LANGS)
            done = 0
            for task in manifest["task_selection"]["tasks"]:
                task_meta = load_json(run_dir / task["tests_path"].replace("tests.jsonl", "task.json"))
                for lang_index, lang in enumerate(LANGS):
                    out = result_dir(run_dir, task, model, lang)
                    out.mkdir(parents=True, exist_ok=True)
                    gen_path = out / "generation.json"
                    if gen_path.exists() and load_json(gen_path).get("status") == "complete":
                        done += 1
                        continue
                    prompt_info = task_meta["prompts"][lang]
                    prompt = (run_dir / prompt_info["path"]).read_text(encoding="utf-8")
                    seed = SEED + int(task_meta["problem_index"]) * 2 + lang_index
                    body = {
                        "model": "default_model",
                        "messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": prompt}],
                        "temperature": 0, "seed": seed, "max_tokens": MAX_TOKENS,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    start = time.monotonic()
                    item: dict[str, Any] = {
                        "status": "complete", "model": model, "task_id": task["task_id"],
                        "language": lang, "seed": seed, "prompt_sha256": prompt_info["sha256"],
                        "system_sha256": manifest["protocol"]["system_prompt_sha256"],
                        "request": body, "requested_utc": now(), "memory_before": memory_free(),
                    }
                    try:
                        request = urllib.request.Request(
                            "http://127.0.0.1:8080/v1/chat/completions",
                            data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"}, method="POST",
                        )
                        with urllib.request.urlopen(request, timeout=3600) as response:
                            api = json.load(response)
                        choice = (api.get("choices") or [{}])[0]
                        reply = str((choice.get("message") or {}).get("content") or "")
                        code, extraction = extract_code(reply)
                        item.update({
                            "api_response": api, "reply": reply, "reply_sha256": sha(reply),
                            "code_sha256": sha(code), "extraction": extraction,
                            "finish_reason": choice.get("finish_reason"),
                            "usage": api.get("usage"),
                        })
                        (out / f"solution{LANGS[lang]['ext']}").write_text(code, encoding="utf-8")
                    except Exception as exc:
                        item.update({"status": "request_error",
                                     "error": f"{type(exc).__name__}: {exc}"})
                        (out / f"solution{LANGS[lang]['ext']}").write_text("", encoding="utf-8")
                    item["response_seconds"] = round(time.monotonic() - start, 3)
                    item["memory_after"] = memory_free()
                    write_json(gen_path, item)
                    done += 1
                    print(f"[{model}] {done}/{total} {task['task_id']} {lang}: "
                          f"{item['status']} in {item['response_seconds']:.1f}s; "
                          f"memory free {item['memory_after']}%.", flush=True)
                    if item["memory_after"] is not None and item["memory_after"] < FREE_RAM_FLOOR:
                        raise MemoryError("Memory free percentage fell below safety floor.")
        finally:
            stop_server(proc)
            print(f"[{model}] server stopped and exit verified.", flush=True)
    manifest["state"][f"{model}_generated"] = True
    manifest[f"memory_after_{model}_generation_free_percent"] = memory_free()
    write_json(run_dir / "manifest.json", manifest)


def score_one(
    run_dir: Path, task: dict[str, Any], model: str, lang: str
) -> dict[str, Any]:
    out = result_dir(run_dir, task, model, lang)
    task_meta = load_json(run_dir / task["tests_path"].replace("tests.jsonl", "task.json"))
    generation = load_json(out / "generation.json") if (out / "generation.json").exists() else {}
    source_path = out / f"solution{LANGS[lang]['ext']}"
    code = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    tests_path = run_dir / task["tests_path"]
    cases = [json.loads(line) for line in tests_path.read_text(encoding="utf-8").splitlines()]
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen38-score-", dir="/private/tmp") as raw:
        temp = Path(raw)
        profile = sandbox_profile(temp)
        source = temp / ("candidate.cpp" if lang == "cpp17" else "candidate.py")
        source.write_text(code, encoding="utf-8")
        compile_info: dict[str, Any]
        command: list[str] | None
        if generation.get("status") != "complete" or not code.strip():
            compile_info = {"status": "no_source"}
            command = None
        elif lang == "cpp17":
            binary = temp / "candidate"
            include_root = write_bits_compat(temp)
            env = os.environ.copy()
            env.update({"TMPDIR": str(temp), "HOME": str(temp), "LC_ALL": "C"})
            compiled = subprocess.run(
                [str(CPP), "-std=c++17", "-O2", "-pipe", "-I", str(include_root),
                 str(source), "-o", str(binary)],
                cwd=temp, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=120, check=False,
            )
            compile_info = {"status": "ok" if compiled.returncode == 0 else "failed",
                            "returncode": compiled.returncode, "log": compiled.stdout[-10000:],
                            "compiler": str(CPP),
                            "compat_header": "local bits/stdc++.h shim",
                            "flags": ["-std=c++17", "-O2", "-pipe"]}
            command = [str(binary)] if compiled.returncode == 0 else None
        else:
            compile_info = {"status": "ready", "interpreter": str(PY3),
                            "flags": ["-I", "-B", "-S"]}
            command = [str(PY3), "-I", "-B", "-S", str(source)]
        if command is not None:
            limit = min(15.0, max(4.0, float(task_meta["time_limit_seconds"]) * 3 + 1))
            for i, case in enumerate(cases):
                got = run_program(command, temp, profile, case["input"], limit)
                status = got["status"]
                match = status == "ok" and got["stdout"].split() == case["expected"].split()
                if status == "ok" and not match:
                    status = "wrong_answer"
                row = {
                    "test_index": i, "source": case["source"],
                    "source_index": case["source_index"],
                    "status": "pass" if match else status,
                    "expected_sha256": sha(case["expected"]),
                    "actual_sha256": sha(got["stdout"]),
                    "returncode": got.get("returncode"),
                }
                if not match:
                    row["expected_preview"] = case["expected"][:1000]
                    row["actual_preview"] = got["stdout"][:1000]
                    # Keep complete diagnostics for a future repair prompt. They stay
                    # inside the per-attempt artifacts and are not sent in one-shot runs.
                    row["expected_full"] = case["expected"]
                    row["actual_full"] = got["stdout"]
                    row["stderr"] = got["stderr"][-65536:]
                results.append(row)
                if status in {"timeout", "runner_error", "output_limit"}:
                    results.extend({
                        "test_index": j, "source": cases[j]["source"],
                        "source_index": cases[j]["source_index"],
                        "status": "not_run_after_runtime_failure",
                    } for j in range(i + 1, len(cases)))
                    break
        else:
            results = [{
                "test_index": i, "source": case["source"],
                "source_index": case["source_index"], "status": "not_run_" + compile_info["status"],
            } for i, case in enumerate(cases)]
        (out / "test_results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for r in results),
            encoding="utf-8",
        )
    passed = sum(r["status"] == "pass" for r in results)
    summary = {
        "model": model, "task_id": task["task_id"], "tier": task["tier"], "language": lang,
        "task_pass": passed == len(cases), "passed_test_inputs": passed,
        "total_test_inputs": len(cases), "compile": compile_info,
        "generation_status": generation.get("status"), "finish_reason": generation.get("finish_reason"),
        "response_seconds": generation.get("response_seconds"), "usage": generation.get("usage"),
        "test_results": str((out / "test_results.jsonl").relative_to(run_dir)),
    }
    write_json(out / "summary.json", summary)
    return summary


def score_model(run_dir: Path, manifest: dict[str, Any], model: str) -> None:
    for task in manifest["task_selection"]["tasks"]:
        for lang in LANGS:
            out = result_dir(run_dir, task, model, lang)
            if (out / "summary.json").exists():
                continue
            summary = score_one(run_dir, task, model, lang)
            print(f"[{model}] {task['task_id']} {lang}: "
                  f"{summary['passed_test_inputs']}/{summary['total_test_inputs']} tests; "
                  f"{'PASS' if summary['task_pass'] else 'FAIL'}", flush=True)
    manifest["state"][f"{model}_scored"] = True
    write_json(run_dir / "manifest.json", manifest)


def percentile(vals: list[float], p: float) -> float:
    x = (len(vals) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return vals[lo] if lo == hi else vals[lo] * (hi - x) + vals[hi] * (x - lo)


def analyze(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for task in manifest["task_selection"]["tasks"]:
        for lang in LANGS:
            b = load_json(result_dir(run_dir, task, "base", lang) / "summary.json")
            r = load_json(result_dir(run_dir, task, "rft", lang) / "summary.json")
            cases.append({
                "task_id": task["task_id"], "tier": task["tier"], "language": lang,
                "base": bool(b["task_pass"]), "rft": bool(r["task_pass"]),
                "base_test_inputs": b["passed_test_inputs"],
                "rft_test_inputs": r["passed_test_inputs"],
                "total_test_inputs": b["total_test_inputs"],
            })
    problem_ids = sorted({r["task_id"] for r in cases})
    by_problem = {
        problem: [float(r["rft"]) - float(r["base"]) for r in cases if r["task_id"] == problem]
        for problem in problem_ids
    }
    rng = random.Random(SEED)
    boot = []
    for _ in range(BOOTSTRAPS):
        sample = [rng.choice(problem_ids) for _ in problem_ids]
        values = [d for problem in sample for d in by_problem[problem]]
        boot.append(sum(values) / len(values))
    boot.sort()
    n = len(cases)
    bpass = sum(r["base"] for r in cases)
    rpass = sum(r["rft"] for r in cases)
    wins = sum(r["rft"] and not r["base"] for r in cases)
    losses = sum(r["base"] and not r["rft"] for r in cases)
    groups = {}
    for label, pred in (
        ("C++17", lambda r: r["language"] == "cpp17"),
        ("Python 3", lambda r: r["language"] == "python3"),
        ("Easy", lambda r: r["tier"] == "easy"),
        ("Medium", lambda r: r["tier"] == "medium"),
        ("Hard", lambda r: r["tier"] == "hard"),
    ):
        group = [r for r in cases if pred(r)]
        groups[label] = {
            "n": len(group), "base": sum(r["base"] for r in group),
            "rft": sum(r["rft"] for r in group),
        }
    return {
        "created_utc": now(), "n_problems": len(problem_ids), "n_cases": n,
        "base": {"passed": bpass, "total": n, "rate": bpass / n},
        "rft": {"passed": rpass, "total": n, "rate": rpass / n},
        "delta": (rpass - bpass) / n, "wins": wins, "losses": losses, "ties": n - wins - losses,
        "bootstrap_95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "bootstrap_samples": BOOTSTRAPS, "bootstrap_seed": SEED,
        "groups": groups, "case_results": cases,
    }


def report_text(run_dir: Path, manifest: dict[str, Any], result: dict[str, Any]) -> str:
    def pc(v: float) -> str:
        return f"{100*v:.1f}%"
    lo, hi = result["bootstrap_95"]
    lines = [
        "# Qwen3.8 base vs RFT coding comparison", "",
        f"**Status:** complete; scored {result['created_utc']}.", "",
        "## Result", "",
        f"On the same {result['n_cases']} task-language cases, the base passed "
        f"**{result['base']['passed']}/{result['n_cases']} ({pc(result['base']['rate'])})** "
        f"and the RFT adapter passed **{result['rft']['passed']}/{result['n_cases']} "
        f"({pc(result['rft']['rate'])})**. Paired RFT-minus-base change: "
        f"**{result['delta']*100:+.1f} percentage points** "
        f"(problem-clustered 95% bootstrap interval **{lo*100:+.1f} to {hi*100:+.1f} pp**).",
        "",
        f"RFT-only wins: **{result['wins']}**; base-only wins: **{result['losses']}**; "
        f"ties: **{result['ties']}**. This 15-problem pilot is useful directional evidence, "
        "not a precise estimate for all coding tasks.", "",
        "## By language and difficulty", "", "| Group | Base | RFT | Change |", "|---|---:|---:|---:|",
    ]
    for label in ("C++17", "Python 3", "Easy", "Medium", "Hard"):
        g = result["groups"][label]
        lines.append(
            f"| {label} ({g['n']} cases) | {g['base']}/{g['n']} ({pc(g['base']/g['n'])}) | "
            f"{g['rft']}/{g['n']} ({pc(g['rft']/g['n'])}) | "
            f"{(g['rft']-g['base'])*100/g['n']:+.1f} pp |"
        )
    lines.extend(["", "## Per-problem results", "",
                  "| Tier | Problem | Rating | C++ base/RFT | Python base/RFT | Tests |",
                  "|---|---|---:|---|---|---:|"])
    outcomes = {(r["task_id"], r["language"]): r for r in result["case_results"]}
    for task in manifest["task_selection"]["tasks"]:
        c = outcomes[(task["task_id"], "cpp17")]
        p = outcomes[(task["task_id"], "python3")]
        lines.append(
            f"| {task['tier'].title()} | {task['name']} | {task['rating']} | "
            f"{'pass' if c['base'] else 'fail'} / {'pass' if c['rft'] else 'fail'} | "
            f"{'pass' if p['base'] else 'fail'} / {'pass' if p['rft'] else 'fail'} | "
            f"{task['test_count']} |"
        )
    audit = manifest["task_selection"]["rft_data_exact_id_title_audit"]
    lines.extend([
        "", "## Method and independent verification", "",
        f"- Base: {manifest['models']['base']['path']} (saved checkpoint hash "
        f"{manifest['models']['base']['directory_sha256']}).",
        f"- Fine-tuned candidate: the same base plus RFT scale-4 adapter, step "
        f"{manifest['models']['rft']['selected_step']} "
        f"({manifest['models']['rft']['adapter_path']}).",
        "- One completion per problem and language; temperature 0; same prompt and decoding "
        "settings; no repair feedback, checklist, plan-first step, or tools.",
        "- Base and RFT servers ran one at a time. Each was stopped and checked before the next loaded.",
        "- Accepted C++ and Python references matched every public, private, and generated test pair "
        "before scoring. Candidate programs ran with network denied, reads from this user's home "
        "denied, and writes confined to a temporary directory.",
        f"- Local RFT text-data audit: {len(audit['matches'])} exact ID/title matches among "
        f"{audit['files_checked']} files. This does not test semantic overlap.",
        "",
        "## How this relates to the earlier scores", "",
        "This result is not directly comparable to the earlier 40.0% base and 49.3% RFT scores. "
        "Those used a different 30-task held-out suite with five sampled completions per task. "
        "This comparison uses 15 problems, two languages, and one deterministic completion per case. "
        "CodeContests is public, so pretraining exposure cannot be ruled out.", "",
        "## Artifacts", "",
        f"- Run folder: {run_dir.relative_to(ROOT)}",
        "- Prompt: QWEN38_BASE_VS_RFT_CODING_COMPARISON_PROMPT.md",
        "- Runner: tools/qwen38_code_compare.py",
        f"- Scoring data: {run_dir.relative_to(ROOT)}/analysis.json",
        "",
        "## Sources", "",
        "- [DeepMind CodeContests repository](https://github.com/google-deepmind/code_contests)",
        "- [CodeContests dataset card](https://huggingface.co/datasets/deepmind/code_contests)",
        "- [Official problem and test schema](https://github.com/google-deepmind/code_contests/blob/main/contest_problem.proto)",
        "- [Hugging Face Dataset Viewer rows API](https://huggingface.co/docs/dataset-viewer/en/rows)",
    ])
    return "\n".join(lines) + "\n"


def run(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    manifest = load_json(run_dir / "manifest.json")
    manifest["status"] = "running"
    write_json(run_dir / "manifest.json", manifest)
    for model in ("base", "rft"):
        if not manifest["state"].get(f"{model}_generated"):
            generate(run_dir, manifest, model)
            manifest = load_json(run_dir / "manifest.json")
        if not manifest["state"].get(f"{model}_scored"):
            score_model(run_dir, manifest, model)
            manifest = load_json(run_dir / "manifest.json")
    result = analyze(run_dir, manifest)
    write_json(run_dir / "analysis.json", result)
    (ROOT / "QWEN38_BASE_VS_RFT_CODING_COMPARISON.md").write_text(
        report_text(run_dir, manifest, result), encoding="utf-8"
    )
    manifest["status"] = "complete"
    manifest["completed_utc"] = now()
    write_json(run_dir / "manifest.json", manifest)
    print(f"Complete: base {result['base']['passed']}/{result['n_cases']}; "
          f"RFT {result['rft']['passed']}/{result['n_cases']}; "
          f"change {result['delta']*100:+.1f} pp.", flush=True)


def add_turbo(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    manifest = load_json(run_dir / "manifest.json")
    config = TURBO_MODEL / "config.json"
    manifest["turbo_model"] = {"path": str(TURBO_MODEL), "config_sha256": sha(config.read_bytes()),
                               "added_utc": now()}
    write_json(run_dir / "manifest.json", manifest)
    if not manifest["state"].get("turbo_generated"):
        generate(run_dir, manifest, "turbo")
        manifest = load_json(run_dir / "manifest.json")
    if not manifest["state"].get("turbo_scored"):
        score_model(run_dir, manifest, "turbo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--run-dir", type=Path, required=True)
    r = subs.add_parser("run")
    r.add_argument("--run-dir", type=Path, required=True)
    t = subs.add_parser("add-turbo", help="generate and score the TURBO model on an existing run")
    t.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            data = prepare(args.run_dir)
            print(f"Prepared {len(data['task_selection']['tasks'])} tasks and "
                  f"{sum(t['test_count'] for t in data['task_selection']['tasks'])} test inputs. "
                  f"Run folder: {data['run_dir']}", flush=True)
        elif args.command == "add-turbo":
            add_turbo(args.run_dir)
        else:
            run(args.run_dir)
        return 0
    except Exception as exc:
        print(f"qwen38_code_compare: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
