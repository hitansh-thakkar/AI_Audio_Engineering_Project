"""Attribute separated window-streams to diarization speakers.

The separator runs on a context window around each overlap region and produces anonymous
streams (one per source). We must decide which stream belongs to which speaker. The
reliable evidence is the SOLO regions inside the window: there the mixture equals that one
speaker, so the correct stream reproduces the mixture there. We score each stream against
the mixture over each speaker's solo mask (scale-invariant SNR) and greedily assign the
best pairs one-to-one. Streams with no confident match are left unassigned (noted — errors
flow, not halt).
"""
from __future__ import annotations

import numpy as np

from ..types import DiarizationResult


def solo_intervals(turn: tuple[float, float],
                   overlaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the parts of `turn` not covered by any overlap region (its solo intervals)."""
    solo = [turn]
    for o0, o1 in overlaps:
        nxt: list[tuple[float, float]] = []
        for s0, s1 in solo:
            if o1 <= s0 or o0 >= s1:          # no intersection
                nxt.append((s0, s1))
                continue
            if s0 < o0:
                nxt.append((s0, o0))
            if o1 < s1:
                nxt.append((o1, s1))
        solo = nxt
    return [(a, b) for a, b in solo if b - a > 1e-3]


def _solo_mask(diar: DiarizationResult, speaker: str, window: tuple[float, float],
               n: int, sr: int) -> np.ndarray:
    """Boolean mask (length n) of samples inside `window` where only `speaker` is active."""
    w0, w1 = window
    mask = np.zeros(n, dtype=bool)
    overlaps = list(diar.overlap_regions)
    for turn in diar.turns:
        if turn.speaker != speaker:
            continue
        for s0, s1 in solo_intervals((turn.start, turn.end), overlaps):
            a, b = max(s0, w0), min(s1, w1)
            if b > a:
                i0, i1 = int((a - w0) * sr), int((b - w0) * sr)
                mask[i0:min(i1, n)] = True
    return mask


def _si_snr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est.astype(np.float64) - est.mean()
    ref = ref.astype(np.float64) - ref.mean()
    s = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    noise = est - s
    return float(10 * np.log10((np.dot(s, s) + eps) / (np.dot(noise, noise) + eps)))


def _speakers_in_window(diar: DiarizationResult, window: tuple[float, float]) -> list[str]:
    w0, w1 = window
    out: list[str] = []
    for turn in diar.turns:
        if turn.start < w1 and turn.end > w0 and turn.speaker not in out:
            out.append(turn.speaker)
    return out


def attribute_streams(streams: list[np.ndarray], window: tuple[float, float],
                      diar: DiarizationResult, mix: np.ndarray, sr: int,
                      notes: list[str] | None = None) -> dict[int, str]:
    """Return {stream_index -> speaker} for streams that confidently match a speaker's
    solo audio. Greedy one-to-one on scale-invariant SNR; unmatched streams are omitted."""
    notes = [] if notes is None else notes
    if not streams:
        return {}
    speakers = _speakers_in_window(diar, window)
    min_samples = sr // 10                                 # need >=100 ms of solo evidence
    masks = {spk: _solo_mask(diar, spk, window, len(mix), sr) for spk in speakers}

    pairs: list[tuple[float, int, str]] = []
    for ki, stream in enumerate(streams):
        n = min(len(stream), len(mix))
        for spk in speakers:
            m = masks[spk][:n]
            if m.sum() < min_samples:
                continue
            pairs.append((_si_snr(stream[:n][m], mix[:n][m]), ki, spk))
    pairs.sort(key=lambda p: p[0], reverse=True)

    taken_k: set[int] = set()
    taken_spk: set[str] = set()
    mapping: dict[int, str] = {}
    for _score, ki, spk in pairs:
        if ki in taken_k or spk in taken_spk:
            continue
        mapping[ki] = spk
        taken_k.add(ki)
        taken_spk.add(spk)
    if len(mapping) < len(streams):
        notes.append(f"{len(streams) - len(mapping)} separated stream(s) in window {window} "
                     "left unattributed (insufficient solo evidence)")
    return mapping
