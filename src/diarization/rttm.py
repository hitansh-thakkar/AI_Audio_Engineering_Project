"""RTTM read/write + overlap computation (NIST RTTM format)."""
from __future__ import annotations

from pathlib import Path

from ..types import SpeakerTurn


def read_rttm(path: str | Path) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            turns.append(SpeakerTurn(start=start, end=start + dur, speaker=spk))
    return sorted(turns, key=lambda t: t.start)


def write_rttm(turns: list[SpeakerTurn], uri: str, path: str | Path) -> None:
    lines = [
        f"SPEAKER {uri} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>\n"
        for t in sorted(turns, key=lambda t: t.start)
    ]
    Path(path).write_text("".join(lines), encoding="utf-8")


def compute_overlaps(turns: list[SpeakerTurn]) -> list[tuple[float, float]]:
    """Regions where >=2 distinct speakers are simultaneously active."""
    edges: list[tuple[float, int]] = []
    for t in turns:
        edges.append((t.start, 1))
        edges.append((t.end, -1))
    edges.sort()
    overlaps: list[tuple[float, float]] = []
    active = 0
    region_start: float | None = None
    for time, delta in edges:
        prev = active
        active += delta
        if prev < 2 <= active:
            region_start = time
        elif prev >= 2 > active and region_start is not None:
            if time > region_start:
                overlaps.append((round(region_start, 3), round(time, 3)))
            region_start = None
    return overlaps
