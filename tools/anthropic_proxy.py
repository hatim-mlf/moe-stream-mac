"""Localhost forwarder to the Anthropic API that adds the API key (brief 51 step 8, the Opus reference arm).

  .venv/bin/python tools/anthropic_proxy.py --port 8091 [--key-file ~/.config/anthropic/eval-key]

pi runs in a sandbox that may only reach 127.0.0.1; it talks to this proxy with a dummy key. The proxy replaces
the key with the one read from --key-file and forwards only POST /v1/messages* to --upstream, streaming the reply.
So the sandboxed agent never sees the key and cannot reach any other host (no looking up solutions online).
The key is never logged.
"""
import argparse
import http.server
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_UPSTREAM = "https://api.anthropic.com"
DEFAULT_KEY_FILE = Path.home() / ".config/anthropic/eval-key"
ALLOWED_PREFIX = "/v1/messages"
DROP_HEADERS = {"host", "content-length", "connection", "x-api-key", "authorization", "accept-encoding",
                "transfer-encoding"}
HOP_HEADERS = {"transfer-encoding", "connection", "content-length"}
CHUNK = 65536
UPSTREAM_TIMEOUT_S = 900


def make_handler(upstream: str, key: str) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not self.path.startswith(ALLOWED_PREFIX):
                self.reply_error(403, b'{"error": "only /v1/messages is forwarded"}')
                return
            headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP_HEADERS}
            headers["x-api-key"] = key
            req = urllib.request.Request(upstream + self.path, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as resp:
                    self.relay(resp.status, resp.headers.items(), resp)
            except urllib.error.HTTPError as err:
                self.relay(err.code, err.headers.items(), err)

        def relay(self, status: int, headers: list, stream: object) -> None:
            self.send_response(status)
            for k, v in headers:
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := stream.read1(CHUNK):
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True

        def reply_error(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    args = parser.parse_args()
    key = args.key_file.expanduser().read_text().strip()
    if not key:
        print(f"API key file {args.key_file} is empty", file=sys.stderr)
        return 1
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.upstream.rstrip("/"), key))
    print(f"anthropic_proxy on 127.0.0.1:{args.port} -> {args.upstream}", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
