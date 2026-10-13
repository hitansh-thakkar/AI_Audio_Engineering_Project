"""Sarvam speech-to-text backend with word timestamps per diarized speaker track."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import soundfile as sf

from ..config import load_env
from ..separation.stitch import build_speaker_tracks
from ..types import AudioClip, DiarizationResult, SeparatedSource, Transcript, TranscriptSegment, Word


class SarvamASR:
    name = "asr"
    version = "sarvam"

    def __init__(self, cfg):
        self.cfg = cfg
        self._client = None
        self._load_error: str | None = None

    def load(self) -> None:
        load_env()
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            self._load_error = "SARVAM_API_KEY is not set"
            return
        try:
            from sarvamai import SarvamAI

            self._client = SarvamAI(api_subscription_key=key)
        except Exception as exc:
            self._load_error = f"could not initialize Sarvam: {type(exc).__name__}: {exc}"

    def run(self, clip: AudioClip, diarization: DiarizationResult,
            separated: list[SeparatedSource] | None = None,
            notes: list[str] | None = None) -> Transcript:
        notes = [] if notes is None else notes
        if self._client is None:
            notes.append(f"Sarvam ASR unavailable: {self._load_error or 'client not loaded'}")
            return Transcript([
                TranscriptSegment(t.start, t.end, t.speaker, "<asr-unavailable>")
                for t in sorted(diarization.turns, key=lambda turn: turn.start)
            ])

        tracks = build_speaker_tracks(
            clip, diarization.turns, diarization.overlap_regions, separated or [])
        segments: list[TranscriptSegment] = []
        with tempfile.TemporaryDirectory(prefix="dhvani_sarvam_") as temp_dir:
            for speaker, samples in sorted(tracks.items()):
                audio_path = Path(temp_dir) / f"{speaker}.wav"
                sf.write(audio_path, samples, clip.sample_rate, subtype="PCM_16")
                try:
                    with audio_path.open("rb") as audio_file:
                        response = self._client.speech_to_text.transcribe(
                            file=audio_file,
                            model=str(self.cfg.model),
                            language_code="unknown",
                            input_audio_codec="wav",
                        )
                    segments.extend(self._response_segments(response, speaker, clip.duration))
                except Exception as exc:
                    notes.append(f"Sarvam failed for {speaker}: {type(exc).__name__}: {exc}")
                    segments.append(TranscriptSegment(0.0, clip.duration, speaker, "<asr-unavailable>"))
        notes.append(f"Sarvam transcribed {len(tracks)} full-length speaker WAVs")
        return Transcript(segments=sorted(segments, key=lambda segment: (segment.start, segment.speaker)))

    def _response_segments(self, response, speaker: str, duration: float) -> list[TranscriptSegment]:
        timestamps = getattr(response, "timestamps", None)
        transcript = str(getattr(response, "transcript", "") or "").strip()
        if timestamps is None or not timestamps.words:
            return [TranscriptSegment(0.0, duration, speaker, transcript or "<empty>")]
        words = [
            Word(text=text, start=float(start), end=float(end))
            for text, start, end in zip(
                timestamps.words, timestamps.start_time_seconds, timestamps.end_time_seconds,
                strict=False,
            )
        ]
        return [TranscriptSegment(words[0].start, words[-1].end, speaker,
                                  " ".join(word.text for word in words), words=words)]
