"""Stage 4 — Gemini Cloud Multilingual Speech-to-Text.

Transcribes audio directly using Google Gemini Flash Multimodal Audio API
with zero local GPU requirement, supporting 20+ Indian languages and code-switching.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any

import soundfile as sf

from ..config import load_env
from ..separation.stitch import build_speaker_tracks
from ..types import AudioClip, DiarizationResult, SeparatedSource, Transcript, TranscriptSegment


class GeminiASR:
    name = "asr"
    version = "gemini-asr"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._client: Any = None
        self._load_error: str | None = None
        self.model_name = getattr(cfg, "model", "gemini-2.0-flash")
        if "2.5" in str(self.model_name):
            self.model_name = "gemini-2.0-flash"

    def load(self) -> None:
        load_env()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            self._load_error = "GEMINI_API_KEY is not set"
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
        except Exception as exc:
            self._load_error = f"could not initialize Gemini client: {type(exc).__name__}: {exc}"

    def run(self, clip: AudioClip, diarization: DiarizationResult,
            separated: list[SeparatedSource] | None = None,
            notes: list[str] | None = None) -> Transcript:
        notes = [] if notes is None else notes
        if self._client is None:
            notes.append(f"Gemini ASR unavailable: {self._load_error or 'client not loaded'}")
            return Transcript([
                TranscriptSegment(t.start, t.end, t.speaker, "<asr-unavailable>")
                for t in sorted(diarization.turns, key=lambda turn: turn.start)
            ])

        tracks = build_speaker_tracks(clip, diarization.turns, diarization.overlap_regions, separated or [])
        segments: list[TranscriptSegment] = []

        for _index, turn in enumerate(sorted(diarization.turns, key=lambda item: item.start)):
            start_sample = max(0, int(round(turn.start * clip.sample_rate)))
            end_sample = min(len(clip.samples), int(round(turn.end * clip.sample_rate)))
            track = tracks.get(turn.speaker)
            samples = (
                track[start_sample:end_sample]
                if track is not None
                else clip.samples[start_sample:end_sample]
            )
            duration = max(0.0, turn.end - turn.start)

            if len(samples) < 160:  # < 10ms
                continue

            seg = self.transcribe_audio_chunk(
                samples, clip.sample_rate, turn.speaker, turn.start, duration, notes
            )
            segments.append(seg)

        notes.append(f"Gemini ASR transcribed {len(segments)} speaker turns via multimodal cloud audio")
        return Transcript(segments=sorted(segments, key=lambda s: (s.start, s.speaker)))

    def transcribe_audio_chunk(self, samples, sr: int, speaker: str, start: float, duration: float,
                               notes: list[str] | None = None) -> TranscriptSegment:
        from google.genai import types

        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        prompt = (
            "You are an Indian Automatic Speech Recognition system for spontaneous Indian speech, "
            "rural dialects, and code-switching (e.g. Hinglish, Tanglish, Benglish).\n"
            "Transcribe this audio recording verbatim. Return strictly JSON:\n"
            "{\n"
            '  "transcript": "transcribed speech in authentic script/words",\n'
            '  "language": "detected language, e.g. Hindi, English, Hinglish, Bengali",\n'
            '  "is_code_switched": true/false,\n'
            '  "translation": "fluent English translation"\n'
            "}"
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            data = json.loads(response.text or "{}")
            text = str(data.get("transcript", "") or "").strip() or "<silence>"
            lang = data.get("language") or "Unknown"
            is_cs = bool(data.get("is_code_switched", False))
            trans = data.get("translation")
            return TranscriptSegment(
                start=start,
                end=start + duration,
                speaker=speaker,
                text=text,
                lang=lang,
                translation=trans,
                is_code_switched=is_cs,
            )
        except Exception as exc:
            if notes is not None:
                notes.append(f"Gemini ASR chunk {speaker} ({start:.2f}s) failed: {exc}")
            return TranscriptSegment(start, start + duration, speaker, "<asr-failed>")
