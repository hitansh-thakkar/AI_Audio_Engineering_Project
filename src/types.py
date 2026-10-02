"""Core data contracts threaded through every pipeline stage.

Stages communicate through these typed dataclasses — never loose dicts.
See docs/ARCHITECTURE.md for the rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioClip:
    samples: np.ndarray        # float32, mono
    sample_rate: int           # canonical 16000 after preprocessing
    source_path: str

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def clip_id(self) -> str:
        from pathlib import Path
        return Path(self.source_path).stem


@dataclass
class SpeakerTurn:
    start: float               # seconds
    end: float
    speaker: str               # "SPEAKER_00", ...

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    overlap_regions: list[tuple[float, float]] = field(default_factory=list)
    num_speakers: int = 0

    def to_timeline_json(self) -> list[dict]:
        """REQUIRED Milestone-2 deliverable: time->speaker timeline."""
        def fmt(t: float) -> str:
            m, s = divmod(int(round(t)), 60)
            return f"{m:02d}:{s:02d}"
        return [
            {"start": fmt(t.start), "end": fmt(t.end),
             "start_s": round(t.start, 3), "end_s": round(t.end, 3), "speaker": t.speaker}
            for t in sorted(self.turns, key=lambda x: x.start)
        ]


@dataclass
class SeparatedSource:
    region: tuple[float, float]
    speaker: str               # assigned back to a diarization speaker
    samples: np.ndarray


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str
    words: list[Word] = field(default_factory=list)
    lang: str | None = None    # detected/segment language, incl. "mixed"


@dataclass
class Transcript:
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()


@dataclass
class Report:
    summary: str
    tags: list[str]
    dialogue: list[TranscriptSegment]   # speaker-separated, cleaned
    raw_transcript: Transcript


@dataclass
class StageResult:
    name: str
    version: str
    wall_time_s: float
    peak_mem_mb: float
    notes: list[str] = field(default_factory=list)   # recoverable failures, degradations


@dataclass
class RunManifest:
    git_sha: str
    config_hash: str
    seeds: dict
    stages: list[StageResult] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
