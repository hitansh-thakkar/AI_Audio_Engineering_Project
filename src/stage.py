"""Stage protocol + instrumentation.

Every stage implements the same interface so stages are swappable and independently
testable. The orchestrator wraps each run() in a timer + peak-memory tracker and appends
a StageResult to the RunManifest — the benchmark reads this manifest, so instrumentation
lives in the code from Phase 0, not bolted on later.

Peak memory is measured with `tracemalloc` (Python-heap peak). This captures NumPy/pure-
Python allocations; native/torch allocations are not fully reflected — documented limitation.
"""
from __future__ import annotations

import gc
import random
import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import StageResult


@runtime_checkable
class Stage(Protocol):
    name: str
    version: str

    def load(self) -> None:
        """Lazily load weights/clients. No heavy work at import time."""
        ...

    def run(self, inp: Any) -> Any:
        """Pure-ish transform: input dataclass -> output dataclass.
        Must not raise on recoverable failures — log, degrade, record in notes."""
        ...


def set_seeds(seed: int) -> dict:
    """Seed everything we can and return the record for the manifest."""
    random.seed(seed)
    np.random.seed(seed)
    seeds = {"python": seed, "numpy": seed}
    try:
        import torch
        torch.manual_seed(seed)
        seeds["torch"] = seed
    except Exception:
        pass
    return seeds


@contextmanager
def measure(name: str, version: str, sink: list[StageResult],
            notes: list[str] | None = None) -> Iterator[list[str]]:
    """Time + peak-memory a block, appending a StageResult to `sink`.

    Yields a mutable notes list the caller can append degradations to.
    """
    local_notes = notes if notes is not None else []
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield local_notes
    finally:
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sink.append(StageResult(
            name=name, version=version,
            wall_time_s=round(wall, 4),
            peak_mem_mb=round(peak / (1024 * 1024), 2),
            notes=list(local_notes),
        ))
