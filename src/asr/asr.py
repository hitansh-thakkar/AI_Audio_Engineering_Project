"""Stage 4 — ASR + alignment (STUB for the pre-midsem vertical slice).

Produces a structurally valid, speaker-attributed Transcript (one segment per diarized
turn) with placeholder text, so the pipeline runs end-to-end. The real faster-whisper /
Tara / IndicConformer backend + WhisperX alignment lands in the post-midsem phase, behind
this same interface.
"""
from __future__ import annotations

from ..types import AudioClip, DiarizationResult, SeparatedSource, Transcript, TranscriptSegment

PLACEHOLDER = "<asr-pending>"


class ASRStub:
    name = "asr"
    version = "stub-0.1"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self) -> None:
        pass

    def run(self, clip: AudioClip, diarization: DiarizationResult,
            separated: list[SeparatedSource] | None = None,
            notes: list[str] | None = None) -> Transcript:
        notes = [] if notes is None else notes
        notes.append("ASR is a stub (post-midsem): placeholder text per diarized turn")
        segments = [
            TranscriptSegment(start=t.start, end=t.end, speaker=t.speaker, text=PLACEHOLDER)
            for t in sorted(diarization.turns, key=lambda t: t.start)
        ]
        return Transcript(segments=segments)
