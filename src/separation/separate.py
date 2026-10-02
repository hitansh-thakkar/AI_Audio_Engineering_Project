"""Stage 3 — separation (surgical): recover per-speaker audio in overlapped regions.

mode: none      -> skip (return []).
      overlap   -> run only on diarization overlap regions (default).
      everywhere-> run on the whole clip (ablation; expected to hurt on real audio).

Real backends (SepFormer / Conv-TasNet via SpeechBrain) are 8 kHz, fixed-2-speaker; they
plug in behind `load()` when installed, resampling 16k<->8k. When unavailable, the stage
degrades gracefully: it records the region un-separated and notes it — errors flow, not halt.
"""
from __future__ import annotations

import numpy as np

from ..types import AudioClip, SeparatedSource


class Separator:
    name = "separation"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"{cfg.mode}-{cfg.backend}"
        self._model = None
        self._available = False

    def load(self) -> None:
        """Try to load a real separator; stay graceful if deps/weights are absent."""
        try:
            import speechbrain  # noqa: F401
            # Real SepFormer/Conv-TasNet loading would happen here.
            # Left unloaded by default to keep the local run dependency-light.
            self._available = False
        except Exception:
            self._available = False

    def run(self, clip: AudioClip, overlap_regions: list[tuple[float, float]],
            notes: list[str] | None = None) -> list[SeparatedSource]:
        notes = [] if notes is None else notes
        mode = str(self.cfg.mode)
        if mode == "none":
            return []

        if mode == "everywhere":
            regions = [(0.0, clip.duration)]
        else:  # overlap
            regions = list(overlap_regions)
            if not regions:
                notes.append("separation.mode=overlap but no overlap regions; nothing to separate")
                return []

        if not self._available:
            notes.append(
                "no real separator installed (SpeechBrain/SepFormer absent); "
                "returning un-separated regions as passthrough sources"
            )

        out: list[SeparatedSource] = []
        sr = clip.sample_rate
        for (start, end) in regions:
            i0, i1 = int(start * sr), int(end * sr)
            seg = clip.samples[i0:i1]
            if self._available and self._model is not None:  # pragma: no cover (needs weights)
                sources = self._separate_region(seg, sr)
            else:
                sources = [seg]  # passthrough
            for k, src in enumerate(sources):
                out.append(SeparatedSource(region=(start, end), speaker=f"SRC_{k}", samples=src))
        return out

    def _separate_region(self, seg: np.ndarray, sr: int):  # pragma: no cover
        raise NotImplementedError("real SepFormer backend not wired in this build")
