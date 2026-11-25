"""Overlapped-speech detection (who is talking *at the same time*).

Our hand-applied Spectral/GMM clustering answers "who spoke when" from a single stream and
cannot, by construction, flag simultaneous speech. pyannote's segmentation model can, so we
use it *only* to populate `overlap_regions` — the clustering still owns speaker attribution
(the compulsory Milestone-2 deliverable). Separation then fires on those regions.

pyannote's `speaker-diarization-community-1` is gated-but-free: the user accepts the
click-through licence on Hugging Face and provides a free HF_TOKEN. Until then the model
returns HTTP 403; we surface a clear, actionable note and return no overlaps rather than
crashing — the pipeline still produces one track per speaker, just without overlap splitting.
"""
from __future__ import annotations

from typing import Any

from ..config import load_env
from ..types import AudioClip

_ACCEPT_HINT = (
    "accept the free licence at https://hf.co/pyannote/speaker-diarization-community-1 "
    "(and its segmentation dependency) and set HF_TOKEN"
)


def overlap_regions_from_annotation(annotation) -> list[tuple[float, float]]:
    """Convert a pyannote diarization Annotation into merged (start, end) overlap regions."""
    # pyannote.audio 4 pipelines return a DiarizeOutput wrapping the Annotation; 3.x returned it bare
    annotation = getattr(annotation, "speaker_diarization", annotation)
    overlap = annotation.get_overlap()          # pyannote.core Timeline of simultaneous speech
    return [(round(seg.start, 3), round(seg.end, 3)) for seg in overlap]


class OverlapDetector:
    name = "overlap_detector"

    def __init__(self, cfg, run_cfg):
        self.cfg = cfg
        self.run_cfg = run_cfg
        self._pipe: Any = None
        self._load_error: str | None = None

    @property
    def enabled(self) -> bool:
        return str(getattr(self.cfg, "overlap_detector", "none")) == "pyannote"

    def _load(self, notes: list[str]) -> bool:
        if self._pipe is not None:
            return True
        if self._load_error is not None:
            return False
        import os
        try:
            import torchaudio

            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
            from pyannote.audio import Pipeline
            load_env()
            token = os.environ.get("HF_TOKEN")
            if not token:
                self._load_error = f"HF_TOKEN not set; {_ACCEPT_HINT}"
                notes.append(f"overlap detection disabled: {self._load_error}")
                return False
            self._pipe = Pipeline.from_pretrained(str(self.cfg.model), token=token)
            return True
        except Exception as exc:                 # gated 403 / offline / missing dep
            kind = type(exc).__name__
            detail = str(exc).replace("\n", " ")[:180]
            self._load_error = f"could not load pyannote ({kind}: {detail}); {_ACCEPT_HINT}"
            notes.append(f"overlap detection unavailable: {self._load_error}")
            return False

    def detect(self, clip: AudioClip, notes: list[str] | None = None) -> list[tuple[float, float]]:
        notes = [] if notes is None else notes
        if not self.enabled:
            return []
        if not self._load(notes):
            return []
        try:
            import torch
            waveform = torch.from_numpy(clip.samples).unsqueeze(0)   # (channel, time)
            annotation = self._pipe({"waveform": waveform, "sample_rate": clip.sample_rate})
            regions = overlap_regions_from_annotation(annotation)
            if not regions:
                notes.append("overlap detector ran; no overlapping speech found")
            return regions
        except Exception as exc:                 # keep the run alive; record and pass through
            detail = str(exc).replace("\n", " ")[:180]
            notes.append(f"overlap detection failed at inference ({type(exc).__name__}: {detail}); "
                         "continuing without overlap regions")
            return []
