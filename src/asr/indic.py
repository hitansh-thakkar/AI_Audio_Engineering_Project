"""Stage 4 — ASR via AI4Bharat IndicConformer (optional, localized Indic ASR).

`ai4bharat/indic-conformer-600m-multilingual` (MIT) is the proposal's "localized" ASR for
the 22 official Indian languages, in native script. It is a ~2.4 GB NeMo/transformers model
and needs an explicit decoding language (no auto-detect), so it is an opt-in backend
(`asr.backend: indicconformer`, `asr.language: hi|bn|...`) — not the free default.

Transcribes each per-speaker track (one segment per track; CTC gives text without word
timings). Degrades gracefully to placeholder text if the model/deps aren't installed — the
pipeline keeps running and records why, so errors flow rather than halt.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from ..config import load_env
from ..separation.stitch import build_speaker_tracks
from ..types import AudioClip, DiarizationResult, SeparatedSource, Transcript, TranscriptSegment


class IndicConformerASR:
    name = "asr"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"indicconformer-{cfg.model}"
        self._model: Any = None
        self._load_error: str | None = None

    def load(self) -> None:
        load_env()
        tok = os.environ.get("HF_TOKEN")   # gated-but-free repo: needs an accepted licence + token
        try:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(
                str(self.cfg.model), trust_remote_code=True, token=tok)
        except Exception as exc:
            self._load_error = (f"IndicConformer unavailable ({type(exc).__name__}: {str(exc)[:160]}); "
                                "accept the gated licence on HF, set HF_TOKEN, "
                                "and `pip install nemo_toolkit[asr]`")

    def run(self, clip: AudioClip, diarization: DiarizationResult,
            separated: list[SeparatedSource] | None = None,
            notes: list[str] | None = None) -> Transcript:
        notes = [] if notes is None else notes
        if self._model is None:
            notes.append(f"ASR unavailable: {self._load_error or 'model not loaded'}; placeholder text")
            return Transcript([
                TranscriptSegment(t.start, t.end, t.speaker, "<asr-unavailable>")
                for t in sorted(diarization.turns, key=lambda turn: turn.start)
            ])

        lang = str(getattr(self.cfg, "language", None) or "hi")
        if lang in ("auto", ""):
            lang = "hi"
            notes.append("IndicConformer needs an explicit language; defaulted to 'hi'")
        tracks = build_speaker_tracks(
            clip, diarization.turns, diarization.overlap_regions, separated or [])
        out: list[TranscriptSegment] = []
        for speaker, samples in sorted(tracks.items()):
            turns = [t for t in diarization.turns if t.speaker == speaker]
            if not turns:
                continue
            start, end = min(t.start for t in turns), max(t.end for t in turns)
            try:
                wav = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
                text = str(self._model(wav, lang, "ctc")).strip()
            except Exception as exc:
                notes.append(f"IndicConformer failed for {speaker}: {type(exc).__name__}: {exc}")
                text = "<asr-unavailable>"
            out.append(TranscriptSegment(start, end, speaker, text, lang=lang))
        notes.append(f"IndicConformer ({self.cfg.model}, lang={lang}) transcribed {len(tracks)} track(s)")
        return Transcript(sorted(out, key=lambda s: (s.start, s.speaker)))
