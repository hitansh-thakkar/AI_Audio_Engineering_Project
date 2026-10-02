"""Voice-activity segmentation.

Default: energy-based split (librosa) — the local, dependency-light fallback used when
pyannote's VAD/segmentation is not installed. pyannote plugs in via the embedding/diarize
backend when available.
"""
from __future__ import annotations

import librosa
import numpy as np


def energy_vad(samples: np.ndarray, sr: int, top_db: int = 30,
               min_dur: float = 0.25) -> list[tuple[float, float]]:
    """Return [(start_s, end_s)] of speech-active regions."""
    if samples.size == 0:
        return []
    intervals = librosa.effects.split(samples, top_db=top_db)
    out: list[tuple[float, float]] = []
    for i0, i1 in intervals:
        start, end = i0 / sr, i1 / sr
        if end - start >= min_dur:
            out.append((round(start, 3), round(end, 3)))
    if not out:  # everything below threshold — treat whole clip as one region
        out = [(0.0, round(len(samples) / sr, 3))]
    return out


def window_segments(regions: list[tuple[float, float]], win: float = 1.5,
                    hop: float = 0.75) -> list[tuple[float, float]]:
    """Slide fixed windows over VAD regions — gives embeddings something to cluster."""
    segs: list[tuple[float, float]] = []
    for start, end in regions:
        if end - start <= win:
            segs.append((start, end))
            continue
        t = start
        while t + win <= end + 1e-6:
            segs.append((round(t, 3), round(t + win, 3)))
            t += hop
        if segs and segs[-1][1] < end - 1e-3:
            segs.append((round(end - win, 3), round(end, 3)))
    return segs
