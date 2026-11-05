"""Fail-safe watcher for system wired memory during MLX training."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping

DEFAULT_HEADROOM_GB = 2.5
MIN_HEADROOM_GB = 1.0
POLL_SECONDS = 0.2
EXIT_CODE = 86
_GIB = 2**30
_PANIC_NOTE = "the run was stopped to avoid a kernel watchdog panic"


def parse_vm_stat(text: str) -> int:
    """Return wired bytes from vm_stat output, requiring its declared page size."""
    page_match = re.search(r"\(page size of\s+(\d+)\s+bytes\)", text)
    wired_match = re.search(
        r"^\s*Pages wired down:\s*(\d{1,3}(?:,\d{3})*|\d+)\.?\s*$",
        text,
        re.MULTILINE,
    )
    if page_match is None or wired_match is None:
        raise ValueError("vm_stat output is missing a valid page size or wired-page count")
    page_size = int(page_match.group(1))
    wired_pages = int(wired_match.group(1).replace(",", ""))
    if page_size <= 0:
        raise ValueError("vm_stat page size must be positive")
    return wired_pages * page_size


def read_wired_bytes() -> int:
    """Read the current system wired-page count from macOS."""
    result = subprocess.run(
        ["/usr/bin/vm_stat"], check=True, capture_output=True, text=True
    )
    return parse_vm_stat(result.stdout)


def total_memory_bytes() -> int:
    """Read installed system memory from sysctl."""
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True
    )
    return int(result.stdout.strip())


def headroom_bytes_from_env(env: Mapping[str, str]) -> int:
    """Read and validate the wired-memory safety margin in GiB."""
    raw = env.get("MEM_GUARD_HEADROOM_GB")
    try:
        gib = DEFAULT_HEADROOM_GB if raw is None else float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("MEM_GUARD_HEADROOM_GB must be a finite number") from exc
    if not math.isfinite(gib) or gib < MIN_HEADROOM_GB:
        raise ValueError(
            f"MEM_GUARD_HEADROOM_GB must be finite and at least {MIN_HEADROOM_GB} GiB"
        )
    try:
        return int(gib * _GIB)
    except OverflowError as exc:
        raise ValueError("MEM_GUARD_HEADROOM_GB is too large") from exc


def startup_limit_message(total_bytes: int, headroom_bytes: int) -> str:
    """Format the startup line describing the wired-memory safety limit."""
    limit = total_bytes - headroom_bytes
    return (
        f"mem_guard: limit {limit / _GIB:.2f} GiB "
        f"(total {total_bytes / _GIB:.2f}, headroom {headroom_bytes / _GIB:.2f})"
    )


def is_breach(wired: int, total: int, headroom: int) -> bool:
    """Return whether system wired memory reached the safe limit."""
    return wired >= total - headroom


def _breach_message(wired: int, total: int, headroom: int) -> str:
    limit = total - headroom
    return (
        f"wired {wired / _GIB:.2f} GiB >= limit {limit / _GIB:.2f} GiB "
        f"(total {total / _GIB:.2f}, headroom {headroom / _GIB:.2f}); {_PANIC_NOTE}"
    )


def _read_error_message(error: Exception) -> str:
    return f"system wired-memory check failed: {type(error).__name__}: {error}; {_PANIC_NOTE}"


def abort(message: str) -> None:
    """Print the breach reason and exit the whole process, including from a thread."""
    print(f"mem_guard: {message}", file=sys.stderr, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_CODE)


def start(
    read_wired: Callable[[], int] = read_wired_bytes,
    total_bytes: int | None = None,
    headroom_bytes: int | None = None,
    poll_seconds: float = POLL_SECONDS,
    on_breach: Callable[[str], None] = abort,
) -> threading.Thread:
    """Check immediately, then watch wired memory in a daemon thread."""
    total = total_memory_bytes() if total_bytes is None else total_bytes
    headroom = (
        headroom_bytes_from_env(os.environ)
        if headroom_bytes is None
        else headroom_bytes
    )
    stopped = threading.Event()

    def poll() -> None:
        while not stopped.wait(poll_seconds):
            try:
                wired = read_wired()
            except Exception as exc:
                stopped.set()
                on_breach(_read_error_message(exc))
                return
            if is_breach(wired, total, headroom):
                stopped.set()
                on_breach(_breach_message(wired, total, headroom))
                return

    thread = threading.Thread(target=poll, name="mem-guard", daemon=True)
    try:
        wired = read_wired()
    except Exception as exc:
        stopped.set()
        on_breach(_read_error_message(exc))
    else:
        if is_breach(wired, total, headroom):
            stopped.set()
            on_breach(_breach_message(wired, total, headroom))
    thread.start()
    return thread
