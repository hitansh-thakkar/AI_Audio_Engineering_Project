"""Glue between the Streamlit UI and the pipeline.

Keeps all non-presentation logic (transcoding uploads, running the pipeline, shaping the
run manifest into something the UI can render) here so it can be unit-tested without a
browser. The Streamlit module (`app.py`) only renders what these functions return.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..pipeline import run_pipeline
from ..types import RunManifest

# Categorical, color-blind-friendly hues. Each speaker keeps one color across the whole UI
# (label, waveform, download) so color encodes identity — not decoration.
SPEAKER_COLORS = ["#4EA8DE", "#F4A259", "#8AC926", "#E36588", "#9D8DF1", "#49C5B6"]

TARGET_SR = 16000


def speaker_color(index: int) -> str:
    return SPEAKER_COLORS[index % len(SPEAKER_COLORS)]


def transcode_to_wav(src: str | Path, out_path: str | Path) -> Path:
    """Load any audio the system can decode and write a 16 kHz mono WAV the pipeline reads.

    Uploads arrive in many formats (mp3, m4a, flac, ...); normalizing up front means the
    pipeline always receives a clean, readable WAV.
    """
    samples, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, samples.astype(np.float32), TARGET_SR)
    return out_path


def envelope(path: str | Path, buckets: int = 400) -> np.ndarray:
    """Downsampled amplitude envelope (0..1) for drawing a compact waveform."""
    samples, _ = sf.read(str(path))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.abs(samples.astype(np.float32))
    if samples.size == 0:
        return np.zeros(buckets, dtype=np.float32)
    step = max(1, samples.size // buckets)
    env = np.array([samples[i:i + step].max() for i in range(0, samples.size, step)][:buckets])
    peak = float(env.max()) if env.size else 0.0
    return (env / peak).astype(np.float32) if peak > 0 else env


@dataclass
class SpeakerResult:
    name: str
    path: str
    color: str
    speaking_seconds: float


@dataclass
class ProcessResult:
    clip_id: str
    out_dir: str
    duration: float
    num_speakers: int
    num_overlap_regions: int
    num_separated_sources: int
    speakers: list[SpeakerResult]
    timeline: list[dict]
    notes: list[str]
    manifest_path: str


def _speaking_seconds(timeline: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in timeline:
        dur = float(row.get("end_s", 0.0)) - float(row.get("start_s", 0.0))
        totals[row["speaker"]] = totals.get(row["speaker"], 0.0) + max(0.0, dur)
    return totals


def _shape(manifest: RunManifest, out_dir: Path, clip_id: str) -> ProcessResult:
    art = manifest.artifacts
    timeline_path = out_dir / "timeline.json"
    timeline = (json.loads(timeline_path.read_text(encoding="utf-8"))
                if timeline_path.exists() else [])
    talk = _speaking_seconds(timeline)
    duration = 0.0
    processed = out_dir / "processed.wav"
    if processed.exists():
        info = sf.info(str(processed))
        duration = info.frames / info.samplerate

    speakers: list[SpeakerResult] = []
    for idx, path in enumerate(sorted(art.get("speaker_audio", []))):
        name = Path(path).stem
        speakers.append(SpeakerResult(name=name, path=path, color=speaker_color(idx),
                                      speaking_seconds=round(talk.get(name, 0.0), 2)))
    notes: list[str] = []
    for stage in manifest.stages:
        notes.extend(f"[{stage.name}] {n}" for n in stage.notes)
    return ProcessResult(
        clip_id=clip_id, out_dir=str(out_dir), duration=round(duration, 2),
        num_speakers=int(art.get("num_speakers", len(speakers))),
        num_overlap_regions=int(art.get("num_overlap_regions", 0)),
        num_separated_sources=int(art.get("num_separated_sources", 0)),
        speakers=speakers, timeline=timeline, notes=notes,
        manifest_path=str(out_dir / "manifest.json"))


def run_on_file(audio_path: str | Path, config_path: str | Path,
                out_root: str | Path) -> ProcessResult:
    """Run the full pipeline on one audio file and shape the result for the UI."""
    manifest = run_pipeline(str(audio_path), config=str(config_path), out=str(out_root))
    clip_id = Path(audio_path).stem
    return _shape(manifest, Path(out_root) / clip_id, clip_id)
