"""Reconstruct a clean, full-length audio track per speaker.

For each speaker: start from silence the length of the whole clip; copy the original
audio into that speaker's SOLO turns; and inside overlap regions, copy the speaker's
ASSIGNED separated source instead of the mix. If a region has no assigned source for
the speaker (separation unavailable or unmatched), fall back to the mix so the track is
still complete — a graceful degradation, not a crash.
"""
from __future__ import annotations

import numpy as np

from ..types import AudioClip, SeparatedSource, SpeakerTurn
from .assign import solo_intervals


def build_speaker_tracks(clip: AudioClip, turns: list[SpeakerTurn],
                         overlap_regions: list[tuple[float, float]],
                         separated: list[SeparatedSource]) -> dict[str, np.ndarray]:
    sr = clip.sample_rate
    speakers = sorted({t.speaker for t in turns})
    tracks: dict[str, np.ndarray] = {spk: np.zeros_like(clip.samples) for spk in speakers}
    overlaps = list(overlap_regions)

    src_map: dict[tuple[tuple[float, float], str], np.ndarray] = {}
    for src in separated:
        if src.speaker in tracks:
            src_map[(src.region, src.speaker)] = src.samples

    for spk in speakers:
        for turn in (t for t in turns if t.speaker == spk):
            for s0, s1 in solo_intervals((turn.start, turn.end), overlaps):
                i0, i1 = int(s0 * sr), int(s1 * sr)
                tracks[spk][i0:i1] = clip.samples[i0:i1]
            for region in overlaps:
                r0, r1 = region
                a, b = max(turn.start, r0), min(turn.end, r1)
                if b - a <= 1e-3:
                    continue
                i0, i1 = int(a * sr), int(b * sr)
                length = i1 - i0
                src_samples = src_map.get((region, spk))
                if src_samples is None:
                    tracks[spk][i0:i1] = clip.samples[i0:i1]   # mix fallback
                    continue
                off = int((a - r0) * sr)
                seg = src_samples[off:off + length]
                if len(seg) < length:
                    seg = np.pad(seg, (0, length - len(seg)))
                tracks[spk][i0:i1] = seg[:length]
    return tracks
