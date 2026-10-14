"""Stage 4 — ASR via AI4Bharat IndicWav2Vec (the proposal's named localized Indic ASR).

Transformers-native Wav2Vec2 CTC (no NeMo), greedy decoding (no external KenLM / pyctcdecode
dependency), emitting native-script text (e.g. Devanagari) per speaker — matching the
proposal's "map regional phonemes directly to native scripts". The model is gated-but-free
on Hugging Face (accept the licence + set HF_TOKEN, exactly like pyannote).

One segment per speaker track (CTC gives no word timestamps). Long tracks are chunked so the
encoder doesn't blow up on memory. Degrades to placeholder text if the model/licence/token is
unavailable — errors flow, not halt.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..config import load_env
from ..separation.stitch import build_speaker_tracks
from ..types import AudioClip, DiarizationResult, SeparatedSource, Transcript, TranscriptSegment

_CHUNK_SECONDS = 30


class IndicWav2VecASR:
    name = "asr"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"indicwav2vec-{cfg.model}"
        self._model: Any = None
        self._proc: Any = None
        self._load_error: str | None = None

    def load(self) -> None:
        load_env()
        tok = os.environ.get("HF_TOKEN")
        try:
            from transformers import (
                AutoModelForCTC,
                Wav2Vec2CTCTokenizer,
                Wav2Vec2FeatureExtractor,
                Wav2Vec2Processor,
            )
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            try:
                hf_logging.disable_progress_bar()
            except Exception:
                pass
            mid = str(self.cfg.model)
            try:
                # Plain processor (greedy CTC) — avoids the repo's KenLM decoder, which would
                # otherwise pull pyctcdecode/kenlm (hard to install on Windows).
                self._proc = Wav2Vec2Processor.from_pretrained(mid, token=tok)
            except Exception:
                fe = Wav2Vec2FeatureExtractor.from_pretrained(mid, token=tok)
                tk = Wav2Vec2CTCTokenizer.from_pretrained(mid, token=tok)
                self._proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tk)
            self._model = AutoModelForCTC.from_pretrained(mid, token=tok).eval()
        except Exception as exc:
            self._load_error = (f"IndicWav2Vec unavailable ({type(exc).__name__}: {str(exc)[:160]}); "
                                "accept the gated-but-free licence on HF and set HF_TOKEN")

    def _transcribe(self, samples: np.ndarray, sr: int) -> str:
        import torch
        parts: list[str] = []
        step = _CHUNK_SECONDS * sr
        for i in range(0, max(1, len(samples)), step):
            chunk = samples[i:i + step]
            if chunk.size < sr // 20:            # skip < 50 ms slivers
                continue
            inputs = self._proc(chunk.astype(np.float32), sampling_rate=sr,
                                return_tensors="pt").input_values
            with torch.no_grad():
                ids = torch.argmax(self._model(inputs).logits, dim=-1)
            parts.append(self._proc.batch_decode(ids)[0].strip())
        return " ".join(p for p in parts if p).strip()

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
        tracks = build_speaker_tracks(
            clip, diarization.turns, diarization.overlap_regions, separated or [])
        out: list[TranscriptSegment] = []
        for speaker, samples in sorted(tracks.items()):
            turns = [t for t in diarization.turns if t.speaker == speaker]
            if not turns:
                continue
            start, end = min(t.start for t in turns), max(t.end for t in turns)
            try:
                text = self._transcribe(samples, clip.sample_rate)
            except Exception as exc:
                notes.append(f"IndicWav2Vec failed for {speaker}: {type(exc).__name__}: {exc}")
                text = "<asr-unavailable>"
            out.append(TranscriptSegment(start, end, speaker, text, lang=lang))
        notes.append(f"IndicWav2Vec ({self.cfg.model}) transcribed {len(tracks)} speaker track(s)")
        return Transcript(sorted(out, key=lambda s: (s.start, s.speaker)))
