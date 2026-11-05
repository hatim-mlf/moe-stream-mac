"""Filter samples by pass rate.

This module provides a single public function to filter a mapping of task
identifiers to sample rows based on a maximum allowed pass fraction.
"""

import math


def filter_by_pass_rate(
    samples: dict[str, list[dict]], max_pass_fraction: float
) -> dict[str, list[dict]]:
    """Filter tasks whose pass fraction is <= max_pass_fraction.

    Args:
        samples: A mapping from task id to a list of sample rows. Each row
            is a dict containing a boolean key "passed".
        max_pass_fraction: The maximum allowed pass fraction (inclusive).
            Must be a float in [0.0, 1.0]. Booleans are not accepted.

    Returns:
        A new dict containing only the tasks that meet the pass fraction
        criterion, with their row lists copied. The order of task ids is
        preserved.

    Raises:
        ValueError: If max_pass_fraction is not a valid float in [0.0, 1.0].
    """
    # Validate max_pass_fraction
    # Booleans are a subclass of int in Python, so we must explicitly reject them.
    if isinstance(max_pass_fraction, bool):
        raise ValueError("max_pass_fraction must be a float, not a bool.")
    
    if not isinstance(max_pass_fraction, (int, float)):
        raise ValueError("max_pass_fraction must be a number.")
    
    if not math.isfinite(max_pass_fraction) or not 0.0 <= max_pass_fraction <= 1.0:
        raise ValueError("max_pass_fraction must be in [0.0, 1.0].")

    result: dict[str, list[dict]] = {}

    for task_id, rows in samples.items():
        # Tasks with an empty row list are dropped
        if not rows:
            continue

        total_rows = len(rows)

        passed_count = 0
        for row in rows:
            if row.get("passed") is True:
                passed_count += 1

        pass_fraction = passed_count / total_rows

        # Check if pass fraction is within the allowed limit
        if pass_fraction <= max_pass_fraction:
            result[task_id] = list(rows)

    return result
