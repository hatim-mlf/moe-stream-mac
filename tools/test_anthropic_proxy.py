"""Tests for tools/anthropic_proxy.py: a localhost-only forwarder that adds the API key (brief 51 step 8).

  .venv/bin/python -m pytest -q tools/test_anthropic_proxy.py
"""
import http.server
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROXY = Path(__file__).parent / "anthropic_proxy.py"
REAL_KEY = "sk-ant-test-real-key"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_upstream(seen: list) -> http.server.ThreadingHTTPServer:
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
            out = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args: object) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start_proxy(tmp_path: Path, upstream: str) -> tuple[subprocess.Popen, int]:
    key = tmp_path / "key"
    key.write_text(REAL_KEY + "\n")
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(PROXY), "--port", str(port), "--upstream", upstream,
                             "--key-file", str(key)], stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return proc, port
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError(proc.stderr.read())


def post(port: int, path: str) -> tuple[int, bytes]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b'{"model": "m"}', method="POST",
                                 headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_proxy_swaps_in_real_key_and_forwards_messages(tmp_path: Path) -> None:
    seen: list = []
    up = fake_upstream(seen)
    proc, port = start_proxy(tmp_path, f"http://127.0.0.1:{up.server_address[1]}")
    try:
        status, body = post(port, "/v1/messages?beta=true")
        assert status == 200 and json.loads(body) == {"ok": True}
        hdrs = {k.lower(): v for k, v in seen[0]["headers"].items()}
        assert hdrs["x-api-key"] == REAL_KEY
        assert hdrs["anthropic-version"] == "2023-06-01"
        assert seen[0]["path"] == "/v1/messages?beta=true" and seen[0]["body"] == {"model": "m"}
    finally:
        proc.kill()
        up.shutdown()
    assert REAL_KEY not in proc.stderr.read()


def test_proxy_refuses_other_paths(tmp_path: Path) -> None:
    seen: list = []
    up = fake_upstream(seen)
    proc, port = start_proxy(tmp_path, f"http://127.0.0.1:{up.server_address[1]}")
    try:
        assert post(port, "/v1/files")[0] == 403
        assert seen == []
    finally:
        proc.kill()
        up.shutdown()


def test_proxy_refuses_empty_key_file(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("\n")
    out = subprocess.run([sys.executable, str(PROXY), "--port", str(free_port()), "--key-file", str(key)],
                         capture_output=True, text=True, timeout=10)
    assert out.returncode != 0 and "empty" in out.stderr.lower()
