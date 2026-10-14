"""Stage 4 — ASR via faster-whisper (CTranslate2): local, free, word-timestamped.

Builds one full-length track per speaker (original audio in solo turns, the assigned
separated source inside overlaps — same `build_speaker_tracks` the output uses) and runs
faster-whisper on each with its built-in Silero VAD (skips the silent gaps) and native
word timestamps — so we get speaker-attributed, word-aligned text without the heavy WhisperX
dependency. Language is auto-detected per track (code-switched Hindi/English is transcribed
in Whisper's detected mode); set `asr.language` to force one.

Degrades gracefully: if the model can't load or a track fails, it records a note and emits a
placeholder segment rather than crashing — errors flow, not halt.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..separation.stitch import build_speaker_tracks
from ..types import AudioClip, DiarizationResult, SeparatedSource, Transcript, TranscriptSegment, Word


class FasterWhisperASR:
    name = "asr"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"faster-whisper-{cfg.model}"
        self._model: Any = None
        self._load_error: str | None = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                str(self.cfg.model),
                device=str(getattr(self.cfg, "device", "cpu")),
                compute_type=str(getattr(self.cfg, "compute_type", "int8")),
            )
        except Exception as exc:
            self._load_error = f"faster-whisper unavailable: {type(exc).__name__}: {exc}"

    def _language(self) -> str | None:
        lang = getattr(self.cfg, "language", None)
        return None if lang in (None, "", "auto") else str(lang)

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

        tracks = build_speaker_tracks(
            clip, diarization.turns, diarization.overlap_regions, separated or [])
        language = self._language()
        out: list[TranscriptSegment] = []
        for speaker, samples in sorted(tracks.items()):
            try:
                segments, info = self._model.transcribe(
                    samples.astype(np.float32), language=language,
                    word_timestamps=True, vad_filter=True, beam_size=5)
                detected = getattr(info, "language", None)
                for seg in segments:
                    text = (seg.text or "").strip()
                    if not text:
                        continue
                    words = [Word(w.word.strip(), float(w.start), float(w.end))
                             for w in (seg.words or [])]
                    out.append(TranscriptSegment(float(seg.start), float(seg.end), speaker,
                                                 text, words=words, lang=detected))
            except Exception as exc:
                notes.append(f"faster-whisper failed for {speaker}: {type(exc).__name__}: {exc}")
        if not out:
            notes.append("faster-whisper produced no text (silence / VAD removed everything)")
        notes.append(f"faster-whisper ({self.cfg.model}) transcribed {len(tracks)} speaker track(s)")
        return Transcript(sorted(out, key=lambda s: (s.start, s.speaker)))
