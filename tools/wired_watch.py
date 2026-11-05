"""Protect a non-Python process from exhausting wired memory."""

import os
import sys
import signal
import time
import mem_guard


def watch(
    pid: int,
    *,
    read_wired=None,
    total_bytes=None,
    headroom_bytes=None,
    poll_seconds=None,
    is_alive=None,
    kill=None,
    sleep=None,
    log=print,
) -> str:
    """Watch a process and kill it if wired memory exceeds the limit."""
    if read_wired is None:
        read_wired = mem_guard.read_wired_bytes
    if total_bytes is None:
        total_bytes = mem_guard.total_memory_bytes()
    if headroom_bytes is None:
        headroom_bytes = mem_guard.headroom_bytes_from_env(os.environ)
    if poll_seconds is None:
        poll_seconds = mem_guard.POLL_SECONDS
    if is_alive is None:
        def _is_alive(p: int) -> bool:
            try:
                os.kill(p, 0)
                return True
            except ProcessLookupError:
                return False
        is_alive = _is_alive
    if kill is None:
        kill = lambda p: os.kill(p, signal.SIGKILL)
    if sleep is None:
        sleep = time.sleep

    while True:
        if not is_alive(pid):
            return "exited"
        try:
            wired = read_wired()
        except Exception as e:
            log(f"Error reading wired memory for pid {pid}: {e}")
            kill(pid)
            return "killed"
        if mem_guard.is_breach(wired, total_bytes, headroom_bytes):
            limit_gib = (total_bytes - headroom_bytes) / (1024 ** 3)
            wired_gib = wired / (1024 ** 3)
            log(f"pid={pid} wired={wired_gib:.2f}GiB limit={limit_gib:.2f}GiB")
            kill(pid)
            return "killed"
        sleep(poll_seconds)


def main() -> None:
    """CLI entry point for wired_watch."""
    if len(sys.argv) != 2:
        print("Usage: python tools/wired_watch.py PID", file=sys.stderr)
        sys.exit(1)
    pid = int(sys.argv[1])
    result = watch(pid)
    if result == "killed":
        sys.exit(86)
    sys.exit(0)


if __name__ == "__main__":
    main()
