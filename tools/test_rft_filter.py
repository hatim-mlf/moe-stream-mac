"""Tests for tools/rft_filter.py (written by the lead before delegation)."""

import pytest

from rft_filter import filter_by_pass_rate


def _rows(passes: int, total: int) -> list[dict]:
    return [{"passed": index < passes} for index in range(total)]


def test_drops_always_solved_tasks() -> None:
    samples = {"easy": _rows(8, 8), "zone": _rows(3, 8), "hard": _rows(0, 8)}
    assert list(filter_by_pass_rate(samples, 0.875)) == ["zone", "hard"]


def test_keeps_all_at_one() -> None:
    samples = {"easy": _rows(8, 8), "zone": _rows(3, 8)}
    assert list(filter_by_pass_rate(samples, 1.0)) == ["easy", "zone"]


def test_boundary_is_inclusive() -> None:
    assert list(filter_by_pass_rate({"t": _rows(7, 8)}, 0.875)) == ["t"]


def test_does_not_mutate_and_copies_lists() -> None:
    rows = _rows(2, 8)
    samples = {"t": rows}
    result = filter_by_pass_rate(samples, 0.5)
    assert result["t"] == rows and result["t"] is not rows
    assert samples == {"t": rows}


def test_drops_empty_tasks() -> None:
    assert filter_by_pass_rate({"empty": [], "t": _rows(1, 2)}, 1.0) == {"t": _rows(1, 2)}


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.5", None, True, float("nan")])
def test_rejects_bad_fraction(bad: object) -> None:
    with pytest.raises(ValueError):
        filter_by_pass_rate({"t": _rows(1, 2)}, bad)  # type: ignore[arg-type]
