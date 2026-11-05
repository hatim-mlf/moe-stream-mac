"""Fixed longest-row tokenization used by the memory measurements."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "finetune" / "train.jsonl"
RowSource = str | Path | Iterable[Mapping[str, Any]] | None


def iter_rows(source: RowSource) -> Iterator[Mapping[str, Any]]:
    if source is None:
        source = DATA_PATH
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    yield from source


def row_source_name(source: RowSource) -> str:
    if source is None:
        return str(DATA_PATH)
    if isinstance(source, (str, Path)):
        return str(source)
    return "injected"


def build_row(
    token_count: int,
    tokenizer: Any,
    row_source: RowSource = None,
) -> tuple[list[int], list[bool]]:
    """Return the longest chat-templated row, truncated to T, and its last-half mask."""
    if token_count < 2:
        raise ValueError("tokens must be at least 2 for next-token loss")
    longest: list[int] = []
    for row in iter_rows(row_source):
        encoded = tokenizer.apply_chat_template(
            row["messages"], tools=row.get("tools"), return_dict=False
        )
        if len(encoded) > len(longest):
            longest = [int(token) for token in encoded]
    if len(longest) < token_count:
        raise ValueError(
            f"Longest chat-template row has {len(longest)} tokens; requested {token_count}"
        )
    selected = longest[:token_count]
    start = token_count // 2
    mask = [index >= start for index in range(token_count)]
    return selected, mask
