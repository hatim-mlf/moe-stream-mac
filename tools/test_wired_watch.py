"""Tests for tools/wired_watch.py (written by the lead before delegation)."""

import wired_watch

GIB = 2**30


def _run(readings, alive_steps=100):
    calls, logs, state = [], [], {"n": 0}
    values = iter(readings)

    def read():
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    def alive(_pid):
        state["n"] += 1
        return state["n"] <= alive_steps

    result = wired_watch.watch(
        1234, read_wired=read, total_bytes=32 * GIB, headroom_bytes=2 * GIB, poll_seconds=0,
        is_alive=alive, kill=calls.append, sleep=lambda _s: None, log=logs.append,
    )
    return result, calls, logs


def test_kills_on_breach():
    result, calls, logs = _run([20 * GIB, 31 * GIB])
    assert result == "killed" and calls == [1234]
    assert "1234" in logs[0] and "31.00" in logs[0] and "30.00" in logs[0]


def test_returns_exited_when_process_ends():
    result, calls, _ = _run([20 * GIB] * 10, alive_steps=3)
    assert result == "exited" and calls == []


def test_kills_when_reader_fails():
    result, calls, logs = _run([20 * GIB, OSError("vm_stat failed")])
    assert result == "killed" and calls == [1234] and "vm_stat failed" in logs[0]
