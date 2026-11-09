"""Run the Aider Polyglot exercises against a local mlx_lm server with a test-and-repair loop."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polyglot_bench as pb  # noqa: E402
import qwen38_code_compare as cc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path("/Users/hatim/data-ext/polyglot-benchmark")
MODELS: dict[str, dict[str, Path | None]] = {
    "rft": {
        "model": Path("/Users/hatim/models/Qwen3.8-27B-official-mlx-4bit"),
        "adapter": ROOT / "adapters" / "qwen38-rft-v1",
    },
    "kat": {
        "model": Path("/Users/hatim/models/KAT-Coder-V2.5-Dev-OptiQ-4bit"),
        "adapter": None,
    },
}
CONDITIONS: dict[str, dict[str, Any]] = {
    "repair3": {"attempts": 3, "think": False},
    "think": {"attempts": 1, "think": True},
}
MAX_TOKENS = 8192
SEED = 20261109
SYSTEM = "You are an expert programmer. Follow the instructions exactly."


def solve(
    ex: dict[str, Any],
    ask: Callable[[list[dict[str, str]], bool], tuple[str, dict[str, Any]]],
    work_parent: Path,
    attempts: int,
    think: bool,
) -> dict[str, Any]:
    work = pb.prepare_workdir(ex, work_parent)
    prompt = pb.build_prompt(ex)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    records: list[dict[str, Any]] = []
    total_seconds = 0.0
    total_tokens = 0
    passed = False
    for n in range(1, attempts + 1):
        reply, info = ask(messages, think)
        files = pb.parse_reply(reply, list(ex["solution_files"]))
        if not files:
            passed = False
            test_output = (
                "No file blocks found in your reply: give each file name on its own line "
                "followed by a fenced code block."
            )
        else:
            try:
                pb.write_files(work, files, allowed=ex["solution_files"])
            except ValueError as e:
                passed = False
                test_output = str(e)
            else:
                passed, test_output = pb.run_tests(work, ex)
        records.append(
            {
                "attempt": n,
                "passed": passed,
                "files": sorted(files),
                "test_output": test_output,
                "seconds": info["seconds"],
                "completion_tokens": info["completion_tokens"],
                "forced": info["forced"],
                "reply": reply,
            }
        )
        total_seconds += info["seconds"]
        total_tokens += info["completion_tokens"]
        if passed:
            break
        if n < attempts:
            copy = dict(ex)
            current: dict[str, str] = {}
            for fname in ex["solution_files"]:
                p = work / fname
                current[fname] = p.read_text() if p.exists() else ""
            copy["solution_files"] = current
            prompt = (
                pb.build_prompt(copy)
                + "\n\nYour previous version of these files failed the tests. Test output:\n```\n"
                + test_output
                + "\n```\nFix the code so the tests pass. Reply with the complete content of every file you "
                + "change, the file name on its own line followed by a fenced code block."
            )
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return {
        "lang": ex["lang"],
        "name": ex["name"],
        "attempts": records,
        "attempts_used": len(records),
        "pass_first": records[0]["passed"] if records else False,
        "pass_final": records[-1]["passed"] if records else False,
        "seconds": total_seconds,
        "completion_tokens": total_tokens,
    }


def result_path(out: Path, model: str, condition: str, lang: str, name: str) -> Path:
    return out / model / condition / f"{lang}-{name}.json"


def pending(
    out: Path, model: str, condition: str, exercises: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    return [
        (lang, name)
        for lang, name in exercises
        if not result_path(out, model, condition, lang, name).exists()
    ]


def summarize(out: Path) -> dict[str, Any]:
    s: dict[str, Any] = {}
    for model_dir in sorted(out.iterdir()):
        if not model_dir.is_dir():
            continue
        for cond_dir in sorted(model_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            files = sorted(cond_dir.glob("*.json"))
            if not files:
                continue
            n = 0
            pass_first = 0
            pass_final = 0
            seconds = 0.0
            tokens = 0
            by_lang: dict[str, dict[str, int]] = {}
            for f in files:
                data = json.loads(f.read_text())
                n += 1
                pf = bool(data.get("pass_first"))
                pfl = bool(data.get("pass_final"))
                pass_first += 1 if pf else 0
                pass_final += 1 if pfl else 0
                seconds += float(data.get("seconds", 0.0))
                tokens += int(data.get("completion_tokens", 0))
                lang = data.get("lang", "")
                if lang not in by_lang:
                    by_lang[lang] = {"n": 0, "pass_first": 0, "pass_final": 0}
                by_lang[lang]["n"] += 1
                by_lang[lang]["pass_first"] += 1 if pf else 0
                by_lang[lang]["pass_final"] += 1 if pfl else 0
            s.setdefault(model_dir.name, {})[cond_dir.name] = {
                "n": n,
                "pass_first": pass_first,
                "pass_final": pass_final,
                "seconds": seconds,
                "completion_tokens": tokens,
                "by_lang": by_lang,
            }
    return s


_tokenizer_cache: dict[str, Any] = {}


def make_ask(model_key: str, seed: int) -> Callable[[list[dict[str, str]], bool], tuple[str, dict[str, Any]]]:
    import urllib.request

    def ask(messages: list[dict[str, str]], think: bool) -> tuple[str, dict[str, Any]]:
        t0 = time.monotonic()
        if not think:
            payload = {
                "model": "default_model",
                "messages": messages,
                "temperature": 0,
                "seed": seed,
                "max_tokens": MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = urllib.request.Request(
                f"http://127.0.0.1:{cc.PORT}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                data = json.loads(resp.read().decode())
            reply = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            forced = False
        else:
            import qwen38_rft_workflow_eval as wf  # noqa: F401 (lazy heavy import)

            if model_key not in _tokenizer_cache:
                from transformers import AutoTokenizer

                _tokenizer_cache[model_key] = AutoTokenizer.from_pretrained(
                    str(MODELS[model_key]["model"])
                )
            tokenizer = _tokenizer_cache[model_key]
            result = wf.think_with_budget(messages, seed, tokenizer)
            reply = result["content"]
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            forced = result.get("forced", False)
        seconds = time.monotonic() - t0
        return reply, {"seconds": seconds, "completion_tokens": completion_tokens, "forced": forced}

    return ask


def run(model_key: str, condition: str, out: Path, limit: int | None) -> None:
    exercises = pb.list_exercises(BENCH_ROOT)
    if limit is not None:
        exercises = exercises[:limit]
    todo = pending(out, model_key, condition, exercises)
    if not todo:
        return
    cc.assert_no_server()
    model_cfg = MODELS[model_key]
    model = model_cfg["model"]
    adapter = model_cfg["adapter"]
    cond_cfg = CONDITIONS[condition]
    attempts = cond_cfg["attempts"]
    think = cond_cfg["think"]
    log_path = out / model_key / f"server-{condition}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab")
    cmd = [
        str(cc.SERVER),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(cc.PORT),
        "--prompt-cache-size",
        "0",
        "--prompt-concurrency",
        "1",
        "--decode-concurrency",
        "1",
    ]
    if adapter is not None:
        cmd += ["--adapter-path", str(adapter)]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    try:
        cc.wait_server(proc)
        for (lang, name) in todo:
            idx = exercises.index((lang, name))
            seed = SEED + idx
            ex = pb.load_exercise(BENCH_ROOT, lang, name)
            r = solve(
                ex,
                make_ask(model_key, seed),
                out / "work" / model_key / condition / lang,
                attempts,
                think,
            )
            r["model"] = model_key
            r["condition"] = condition
            r["seed"] = seed
            r["utc"] = cc.now()
            r["memory_free"] = cc.memory_free()
            path = result_path(out, model_key, condition, lang, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(r, indent=2))
            tmp.rename(path)
            status = "PASS" if r["pass_final"] else "FAIL"
            print(
                f"[{model_key}/{condition}] {idx + 1}/{len(exercises)} {lang} {name}: "
                f"{status} first={r['pass_first']} attempts={r['attempts_used']} {r['seconds']:.1f}s",
                flush=True,
            )
    finally:
        cc.stop_server(proc)
        log_file.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--model", choices=["rft", "kat"], required=True)
    p_run.add_argument("--condition", choices=["repair3", "think"], required=True)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--limit", type=int, default=None)
    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.model, args.condition, args.out, args.limit)
    elif args.cmd == "summary":
        print(json.dumps(summarize(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
