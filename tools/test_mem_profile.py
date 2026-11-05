from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from tools import mem_profile
from tools.mem_profile import build_row, run_step, run_trainer_step
from tools.mem_sweep import GIB, fit_coefficients, run_sweep
from tools.test_gdn_checkpoint import _make_tiny_moe


class FixedTokenizer:
    def apply_chat_template(
        self, messages: Sequence[Mapping[str, Any]], tools: Any = None,
        return_dict: bool = False,
    ) -> list[int]:
        content = messages[-1]["content"]
        length, first = (int(value) for value in content.split(":"))
        return list(range(first, first + length))


def _rows(lengths: tuple[int, ...] = (64,)) -> list[dict[str, Any]]:
    return [
        {"messages": [{"role": "user", "content": f"{length}:{100 * index + 1}"}]}
        for index, length in enumerate(lengths)
    ]


def test_build_row_truncates_and_masks_last_half() -> None:
    rows = _rows((8, 12))
    short_tokens, _ = build_row(4, FixedTokenizer(), rows)
    tokens, mask = build_row(9, FixedTokenizer(), rows)
    assert short_tokens == [101, 102, 103, 104]
    assert len(tokens) == 9
    assert tokens == list(range(101, 110))
    assert mask == [False, False, False, False, True, True, True, True, True]


def test_build_row_refuses_when_all_rows_are_short() -> None:
    with pytest.raises(ValueError, match="requested 65"):
        build_row(65, FixedTokenizer(), _rows((12, 30)))


def test_step_returns_all_phases_and_completion_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_tiny_moe()
    tokenizer = FixedTokenizer()
    rows = _rows()
    tokens, _ = build_row(8, tokenizer, rows)
    batch = mx.array([tokens], dtype=mx.int32)
    lengths = mx.array([[4, 8]], dtype=mx.int32)
    from tools.ft_lowmem_loss import completion_only_loss
    mem_guard = mem_profile._import_local_module("mem_guard")

    cache_limits = []
    monkeypatch.setattr(mx, "set_cache_limit", cache_limits.append)
    monkeypatch.setattr(mem_guard, "read_wired_bytes", lambda: 1234)

    expected, _ = completion_only_loss(model, batch, lengths)
    mx.eval(expected)

    result = run_step(
        8,
        4,
        gdn_checkpoint=0,
        grad_checkpoint=False,
        model=model,
        tokenizer=tokenizer,
        row_source=rows,
        model_prepared=True,
        cache_limit_gb=0.25,
    )
    assert cache_limits == [int(0.25 * GIB)]
    assert result["config"]["cache_limit_gb"] == 0.25
    assert set(result["phases"]) == {
        "load", "forward", "value_and_grad", "optimizer_update"
    }
    for phase in result["phases"].values():
        for name in (
            "peak_memory_bytes", "active_memory_bytes", "cache_memory_bytes",
            "system_wired_bytes", "wall_time_seconds",
        ):
            assert phase[name] >= 0
    np.testing.assert_allclose(result["loss"], float(expected.item()), atol=1e-6, rtol=0)


def test_trainer_step_runs_eval_and_training_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_tiny_moe()
    measurements = []
    mem_guard = mem_profile._import_local_module("mem_guard")

    cache_limits = []
    monkeypatch.setattr(mx, "set_cache_limit", cache_limits.append)
    monkeypatch.setattr(mem_guard, "read_wired_bytes", lambda: 5678)
    result = run_trainer_step(
        8, 4,
        gdn_checkpoint=0,
        grad_checkpoint=False,
        model=model,
        tokenizer=FixedTokenizer(),
        row_source=_rows(),
        model_prepared=True,
        cache_limit_gb=0.0,
        on_measurement=measurements.append,
    )
    phases = [record["phase"] for record in result["phases"]]
    assert phases == ["load", "after_eval", "after_train_step", "overall"]
    assert cache_limits == [0]
    assert result["config"]["cache_limit_gb"] == 0.0
    assert all(record["peak_memory_bytes"] >= 0 for record in measurements)
    assert all(record["system_wired_bytes"] == 5678 for record in measurements)
    assert all(record["max_system_wired_bytes"] == 5678 for record in measurements)


def test_fit_recovers_coefficients_from_synthetic_points() -> None:
    expected = (5.0, 0.25, 1.5, 0.125)
    points = []
    for tokens in (256, 512, 1000):
        for layers in (1, 2, 4):
            peak_gib = (
                expected[0]
                + expected[1] * layers
                + expected[2] * tokens / 1024
                + expected[3] * layers * tokens / 1024
            )
            config = {
                "command": "trainer-step", "tokens": tokens, "layers": layers,
                "run_id": f"{tokens}-{layers}",
            }
            points.extend((
                {"config": config, "phase": "after_eval", "peak_memory_bytes": 10000 * GIB},
                {"config": config, "phase": "overall", "peak_memory_bytes": peak_gib * GIB},
            ))
    fitted = fit_coefficients(points)
    assert fitted["points"] == 9
    assert fitted["W_gib"] == pytest.approx(expected[0], abs=1e-6)
    assert fitted["a_gib_per_layer"] == pytest.approx(expected[1], abs=1e-6)
    assert fitted["b_mib_per_token"] == pytest.approx(expected[2], abs=1e-6)
    assert fitted["c_mib_per_layer_token"] == pytest.approx(expected[3], abs=1e-6)


def test_sweep_records_mem_guard_exit_86(tmp_path: Path) -> None:
    calls = []

    def fake_child(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        return type("ChildResult", (), {"returncode": 86, "stdout": "", "stderr": "guard"})()

    output = tmp_path / "sweep.jsonl"
    results = run_sweep([256], [1], 1, output, child_runner=fake_child)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert calls
    assert results[0]["status"] == "guard_stop"
    assert records[0]["status"] == "guard_stop"
    assert calls[0][2] == "trainer-step"
    assert records[0]["config"]["cache_limit_gb"] == "default"

    step_output = tmp_path / "step-sweep.jsonl"
    run_sweep(
        [256], [1], 1, step_output, child_runner=fake_child,
        mode="step", cache_limit_gb=0.5,
    )
    assert calls[1][2] == "step"
    assert calls[1][calls[1].index("--gdn-checkpoint") + 1] == "1"
    assert calls[1][calls[1].index("--cache-limit-gb") + 1] == "0.5"
