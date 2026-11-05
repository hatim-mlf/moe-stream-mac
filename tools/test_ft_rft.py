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
