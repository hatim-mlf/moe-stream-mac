"""Tests for tools/mem_guard.py (system wired-memory watchdog for training runs)."""

import threading

import pytest

import mem_guard

GIB = 2**30
PAGE = 16384

VM_STAT_SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              426540.
Pages active:                            740267.
Pages inactive:                          456573.
Pages speculative:                       305223.
Pages throttled:                              0.
Pages wired down:                        111760.
Pages purgeable:                          23307.
"""


def test_parse_vm_stat_returns_wired_bytes() -> None:
    assert mem_guard.parse_vm_stat(VM_STAT_SAMPLE) == 111760 * PAGE


@pytest.mark.parametrize(
    "text",
    [
        "",
        VM_STAT_SAMPLE.replace("page size of 16384 bytes", "page size unknown"),
        VM_STAT_SAMPLE.replace("Pages wired down:", "Pages wired:"),
    ],
)
def test_parse_vm_stat_refuses_unknown_format(text: str) -> None:
    with pytest.raises(ValueError):
        mem_guard.parse_vm_stat(text)


def test_headroom_default() -> None:
    expected = int(mem_guard.DEFAULT_HEADROOM_GB * GIB)
    assert mem_guard.headroom_bytes_from_env({}) == expected


def test_headroom_from_env() -> None:
    assert mem_guard.headroom_bytes_from_env({"MEM_GUARD_HEADROOM_GB": "3"}) == 3 * GIB


@pytest.mark.parametrize("value", ["abc", "", "0.5", "-2", "nan"])
def test_headroom_refuses_bad_or_too_small(value: str) -> None:
    with pytest.raises(ValueError):
        mem_guard.headroom_bytes_from_env({"MEM_GUARD_HEADROOM_GB": value})


def test_is_breach_boundary() -> None:
    total, headroom = 32 * GIB, 2 * GIB
    assert not mem_guard.is_breach(30 * GIB - 1, total, headroom)
    assert mem_guard.is_breach(30 * GIB, total, headroom)
    assert mem_guard.is_breach(31 * GIB, total, headroom)


def _run_guard(readings: list, poll_seconds: float = 0.001) -> list:
    """Start the guard with scripted readings; return the on_breach calls."""
    calls = []
    fired = threading.Event()
    values = iter(readings)

    def read_wired() -> int:
        value = next(values, readings[-1])
        if isinstance(value, Exception):
            raise value
        return value

    def on_breach(message: str) -> None:
        calls.append(message)
        fired.set()

    thread = mem_guard.start(
        read_wired=read_wired,
        total_bytes=32 * GIB,
        headroom_bytes=2 * GIB,
        poll_seconds=poll_seconds,
        on_breach=on_breach,
    )
    assert thread.daemon
    fired.wait(timeout=2.0)
    return calls


def test_guard_fires_once_when_wired_crosses_limit() -> None:
    calls = _run_guard([20 * GIB, 25 * GIB, 31 * GIB])
    assert len(calls) == 1
    assert "31.00" in calls[0] and "30.00" in calls[0]


def test_guard_fires_immediately_when_already_over_limit() -> None:
    calls = _run_guard([31 * GIB], poll_seconds=10.0)
    assert len(calls) == 1


def test_guard_fires_when_reader_fails() -> None:
    calls = _run_guard([20 * GIB, OSError("vm_stat failed")])
    assert len(calls) == 1
    assert "vm_stat failed" in calls[0]


def test_guard_stays_quiet_under_limit() -> None:
    calls = []
    thread = mem_guard.start(
        read_wired=lambda: 10 * GIB,
        total_bytes=32 * GIB,
        headroom_bytes=2 * GIB,
        poll_seconds=0.001,
        on_breach=calls.append,
    )
    thread.join(timeout=0.1)
    assert thread.is_alive()
    assert calls == []


def test_read_wired_bytes_on_this_machine() -> None:
    wired = mem_guard.read_wired_bytes()
    assert 0 < wired < mem_guard.total_memory_bytes()
