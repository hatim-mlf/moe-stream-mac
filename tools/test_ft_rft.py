import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_rft

FIXTURES = Path(__file__).parent / "fixtures/ft_tasks"
VALID_ID = "b1_c_crcbyte"
TRAIN_ID = "b1_c_affine8"


def task_folders() -> list[Path]:
    return sorted(path for path in FIXTURES.iterdir() if path.is_dir())


def canned_reply(prompt: str) -> str:
    if "add_one" in prompt:
        return "```python\ndef add_one(value):\n    return value + 1\n```"
    if ft_rft.HINT_PREFIX in prompt:
        return "```python\ndef is_even(value):\n    return value % 2 == 0\n```"
    return "```python\ndef is_even(value):\n    return True\n```"


def fake_server() -> tuple[ThreadingHTTPServer, threading.Thread, str, list[dict]]:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            prompt = body["messages"][1]["content"]
            content = canned_reply(prompt)
            payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    return server, thread, endpoint, requests


def sample_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_sample_request_prompts(requests: list[dict], by_task: dict[str, dict]) -> None:
    eval_mod, _, to_sft, _ = ft_rft.load_helpers()
    expected = {folder.name: to_sft.task_prompt(folder, json.loads((folder / "meta.json").read_text()))
                for folder in task_folders()}
    assert by_task[VALID_ID]["reply_length"] == len(canned_reply(expected[VALID_ID]))
    assert by_task[TRAIN_ID]["reply_length"] == len(canned_reply(expected[TRAIN_ID]))
    for body in requests:
        prompt = body["messages"][1]["content"]
        task_id = VALID_ID if "add_one" in prompt else TRAIN_ID
        assert prompt == expected[task_id]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["model"] == "default_model" and body["seed"] == 20261105
    assert eval_mod.extract_code(canned_reply(expected[VALID_ID])).startswith("def add_one")


def test_sample_checks_replies_and_resumes(tmp_path: Path, monkeypatch: Any) -> None:
    eval_mod = ft_rft.load_helpers()[0]
    try:
        server, thread, endpoint, requests = fake_server()
    except PermissionError:
        server = thread = None
        endpoint = "http://127.0.0.1:8080/v1/chat/completions"
        requests = []

        def request_stub(prompt: str, url: str, model: str, adapter: Any,
                         temperature: float, seed: int) -> str:
            requests.append({"messages": [{"role": "system", "content": eval_mod.SYSTEM},
                                           {"role": "user", "content": prompt}],
                             "model": model, "seed": seed,
                             "chat_template_kwargs": {"enable_thinking": False}})
            return canned_reply(prompt)

        monkeypatch.setattr(eval_mod, "request_model", request_stub)
    output = tmp_path / "samples.jsonl"
    try:
        result = ft_rft.main(["sample", str(FIXTURES), "--k", "1",
                              "--seed", "20261105", "--endpoint", endpoint, "--out", str(output)])
        assert result == 0
        rows = sample_lines(output)
        assert len(rows) == 2
        by_task = {row["task_id"]: row for row in rows}
        assert by_task[VALID_ID]["passed"] is True
        assert by_task[TRAIN_ID]["passed"] is False
        assert all(row["sample"] == 0 and row["batch"] == "codex-batch1" for row in rows)
        assert 0 < len(by_task[TRAIN_ID]["failure_tail"]) <= 400
        assert len(requests) == 2
        assert_sample_request_prompts(requests, by_task)
        assert ft_rft.main(["sample", str(FIXTURES), "--k", "1",
                            "--endpoint", endpoint, "--out", str(output)]) == 0
        assert len(requests) == 2
        assert len(sample_lines(output)) == 2
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_sampling_stops_after_three_unreachable_requests(tmp_path: Path,
                                                          monkeypatch: Any, capsys: Any) -> None:
    eval_mod = ft_rft.load_helpers()[0]

    def unreachable(*_args: Any) -> str:
        error = RuntimeError("Cannot reach MLX server at fake endpoint: refused")
        error.__cause__ = ConnectionRefusedError("refused")
        raise error

    monkeypatch.setattr(eval_mod, "request_model", unreachable)
    output = tmp_path / "unreachable.jsonl"
    result = ft_rft.main(["sample", str(FIXTURES), "--only", VALID_ID, "--k", "8",
                          "--endpoint", "http://127.0.0.1:1/v1/chat/completions",
                          "--out", str(output)])
    assert result == 1
    rows = sample_lines(output)
    assert [row["sample"] for row in rows] == [0, 1, 2]
    assert all(not row["passed"] and "Cannot reach MLX server" in row["failure_tail"] for row in rows)
    assert "unreachable for 3 consecutive requests" in capsys.readouterr().err


def build_sample(task_id: str, sample: int, passed: bool, code: str, k: int) -> dict:
    return {"task_id": task_id, "batch": "codex-batch1", "sample": sample, "k": k,
            "passed": passed, "failure_tail": "" if passed else "test failed",
            "reply_length": len(code), "seconds": 0.01, "code": code}


def test_build_deduplicates_caps_and_uses_v5_split(tmp_path: Path) -> None:
    first = "def add_one(value):\n    return value + 1\n"
    shortest = "def add_one(x): return x+1\n"
    other = "def add_one(value):\n    return 1 + value\n"
    samples = [build_sample(VALID_ID, 0, True, shortest, 5),
               build_sample(VALID_ID, 1, True, shortest, 5),
               build_sample(VALID_ID, 2, True, first, 5),
               build_sample(VALID_ID, 3, True, other, 5),
               build_sample(VALID_ID, 4, False, "", 5),
               build_sample(TRAIN_ID, 0, True,
                            "def is_even(value): return value % 2 == 0\n", 2),
               build_sample(TRAIN_ID, 1, False, "", 2)]
    sample_file = tmp_path / "samples.jsonl"
    sample_file.write_text("".join(json.dumps(row) + "\n" for row in samples))
    output = tmp_path / "rft"
    result = ft_rft.main(["build", "--samples", str(sample_file), "--tasks",
                          str(FIXTURES), "--out", str(output), "--per-task", "2"])
    assert result == 0
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(line) for line in (output / "valid.jsonl").read_text().splitlines()]
    assert len(train) == 1
    assert len(valid) == 2
    assert "def add_one(x): return x+1" in valid[0]["messages"][-1]["content"]
    assert valid[0]["messages"][-1]["content"] != valid[1]["messages"][-1]["content"]
    assert valid[0]["messages"][0]["content"] == ft_rft.load_helpers()[0].SYSTEM
    assert all("return True" not in row["messages"][-1]["content"] for row in train + valid)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["train_task_ids"] == [TRAIN_ID]
    assert manifest["valid_task_ids"] == [VALID_ID]
    assert manifest["counts"] == {"train_rows": 1, "valid_rows": 2,
                                   "train_tasks": 1, "valid_tasks": 1}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["per_task"][VALID_ID]["pass@1"] == pytest.approx(4 / 5)
    assert summary["per_task"][VALID_ID]["pass@k"] is True
    assert summary["per_task"][TRAIN_ID]["pass@1"] == pytest.approx(1 / 2)
    assert summary["distribution"]["tasks_with_1_to_k_minus_1_passes"] == 2


def test_build_fails_when_leak_check_fails(tmp_path: Path, monkeypatch: Any) -> None:
    sample_file = tmp_path / "samples.jsonl"
    sample_file.write_text(json.dumps(build_sample(TRAIN_ID, 0, False, "", 1)) + "\n")

    def fail_check():
        raise RuntimeError("ft_leak_check.py failed")

    monkeypatch.setattr(ft_rft, "run_leak_check", fail_check)
    result = ft_rft.main(["build", "--samples", str(sample_file), "--tasks",
                          str(FIXTURES), "--out", str(tmp_path / "rejected")])
    assert result == 1


def test_build_max_pass_fraction_drops_eight_of_eight_tasks(
        tmp_path: Path, monkeypatch: Any) -> None:
    rows = [build_sample(VALID_ID, index, True, "def add_one(x): return x + 1", 8)
            for index in range(8)]
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(ft_rft, "run_leak_check", lambda: None)

    result = ft_rft.main(["build", "--samples", str(samples), "--tasks", str(FIXTURES),
                          "--out", str(tmp_path / "filtered"), "--max-pass-fraction", "0.875"])

    assert result == 0
    manifest = json.loads((tmp_path / "filtered" / "manifest.json").read_text())
    assert VALID_ID not in manifest["task_ids"]
    summary = json.loads((tmp_path / "filtered" / "summary.json").read_text())
    assert summary["per_task"][VALID_ID]["passes"] == 8
    assert summary["per_task"][VALID_ID]["pass@k"] is True


def test_hint_sample_only_resamples_zero_pass_tasks_and_builds_original_prompt(
        tmp_path: Path, monkeypatch: Any) -> None:
    eval_mod = ft_rft.load_helpers()[0]
    prompts: list[str] = []

    def request_stub(prompt: str, _url: str, _model: str, _adapter: Any,
                     _temperature: float, _seed: int) -> str:
        prompts.append(prompt)
        return canned_reply(prompt)

    monkeypatch.setattr(eval_mod, "request_model", request_stub)
    prior_rows = [build_sample(VALID_ID, 0, True, "def add_one(x): return x + 1", 1),
                  build_sample(TRAIN_ID, 1, False, "", 2),
                  build_sample(TRAIN_ID, 0, False, "", 2)]
    prior_rows[1]["failure_tail"] = "later failure"
    prior_rows[2]["failure_tail"] = "x" * 500
    prior = tmp_path / "prior.jsonl"
    prior.write_text("".join(json.dumps(row) + "\n" for row in prior_rows))
    hinted = tmp_path / "hinted.jsonl"

    result = ft_rft.main(["hint-sample", "--samples", str(prior), str(FIXTURES),
                          "--k", "1", "--seed", "7", "--endpoint", "http://fake",
                          "--out", str(hinted)])

    assert result == 0
    rows = sample_lines(hinted)
    assert len(rows) == 1
    assert rows[0]["task_id"] == TRAIN_ID and rows[0]["passed"] is True
    assert rows[0]["hinted"] is True
    assert len(prompts) == 1
    original = ft_rft.load_helpers()[2].task_prompt(
        FIXTURES / TRAIN_ID, json.loads((FIXTURES / TRAIN_ID / "meta.json").read_text()))
    assert prompts[0].startswith(original)
    assert prompts[0].endswith(ft_rft.HINT_PREFIX + "x" * 400 + ft_rft.HINT_SUFFIX)

    monkeypatch.setattr(ft_rft, "run_leak_check", lambda: None)
    out = tmp_path / "merged-build"
    assert ft_rft.main(["build", "--samples", str(prior), "--samples", str(hinted), "--tasks",
                        str(FIXTURES), "--out", str(out)]) == 0
    built = [json.loads(line) for filename in ("train.jsonl", "valid.jsonl")
             for line in (out / filename).read_text().splitlines()]
    train_row = next(row for row in built
                     if row["messages"][1]["content"] == original)
    assert "A previous attempt failed the tests with:" not in train_row["messages"][1]["content"]
    assert "def is_even(value):\n    return value % 2 == 0" in train_row["messages"][-1]["content"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["per_task"][TRAIN_ID]["passes"] == 1
    assert summary["per_task"][TRAIN_ID]["k"] == 3
