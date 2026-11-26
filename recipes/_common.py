"""Small shared helpers for the recipes.

Deliberately thin. Anything that looks like real orchestration belongs in a
recipe where you can see it, not hidden in here -- the point of a recipe is
that the training flow is readable top to bottom.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

__all__ = ["load_jsonl", "batched", "Timer", "DEMO_SFT", "DEMO_GRPO", "fmt"]


def load_jsonl(path: str | Path, *, key: str = "messages") -> list[Any]:
    """Read JSONL, pulling one field out of each line.

    Blank lines are skipped. A malformed line fails loudly with its line
    number: silently dropping training data is how you end up wondering why a
    run used 900 examples instead of 1000.
    """
    out: list[Any] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
            if key not in record:
                raise ValueError(f"{path}:{number} has no {key!r} field")
            out.append(record[key])
    if not out:
        raise ValueError(f"{path} contained no usable records")
    return out


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL keeping whole records (GRPO needs the reward fields too)."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
    if not out:
        raise ValueError(f"{path} contained no usable records")
    return out


def batched(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """Yield fixed-size batches; the final one may be short."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class Timer:
    """Context manager reporting elapsed seconds.

    Used to feed ``sc.metric(timings=...)``, which is what makes the step
    breakdown show up on the platform.
    """

    def __init__(self) -> None:
        self.seconds = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


def fmt(value: float | None, digits: int = 4) -> str:
    """Format a metric that might be absent, without printing 'None'."""
    return "n/a" if value is None else f"{value:.{digits}f}"


#: A tiny SFT set so every recipe runs with --demo and no data prep.
DEMO_SFT: list[list[dict[str, str]]] = [
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ],
    [
        {"role": "user", "content": "Name a primary colour."},
        {"role": "assistant", "content": "Red is a primary colour."},
    ],
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
    ],
    [
        {"role": "user", "content": "How many days in a week?"},
        {"role": "assistant", "content": "There are seven days in a week."},
    ],
    [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": "Sunlight scatters off air molecules, and blue scatters most."},
    ],
    [
        {"role": "user", "content": "Translate 'hello' to French."},
        {"role": "assistant", "content": "'Hello' is 'bonjour' in French."},
    ],
    [
        {"role": "user", "content": "What is 10 divided by 4?"},
        {"role": "assistant", "content": "10 divided by 4 is 2.5."},
    ],
    [
        {"role": "user", "content": "Is water wet?"},
        {"role": "assistant", "content": "Water makes other things wet, so yes in everyday terms."},
    ],
]

#: Prompts plus the reference answer a reward function can check.
DEMO_GRPO: list[dict[str, Any]] = [
    {"messages": [{"role": "user", "content": "What is 2+2? Reply with the number only."}], "answer": "4"},
    {"messages": [{"role": "user", "content": "What is 5+3? Reply with the number only."}], "answer": "8"},
    {"messages": [{"role": "user", "content": "What is 9-4? Reply with the number only."}], "answer": "5"},
    {"messages": [{"role": "user", "content": "What is 6*7? Reply with the number only."}], "answer": "42"},
]
