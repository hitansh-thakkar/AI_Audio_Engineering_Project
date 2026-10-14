"""Stage 1-only service used by the standalone preprocessing web UI."""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from ..asr import make_asr
from ..config import load_config
from ..diarization import Diarizer
from ..postprocessing import make_postprocessor
from ..preprocessing import Preprocessor
from ..preprocessing.preprocess import spectral_denoise
from ..separation import Separator
from ..separation.stitch import build_speaker_tracks
from ..types import (
    AudioClip,
    DiarizationResult,
    SeparatedSource,
    SpeakerTurn,
    Transcript,
    TranscriptSegment,
    Word,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Every stage reads the SAME config as the Streamlit app / main pipeline (config-first):
# real diarization + pyannote overlap detection + faster-whisper ASR + Gemini report.
LIVE_CFG = REPO_ROOT / "config" / "live.yaml"


def process_upload(
    source_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Condition the upload per the live config and return download metadata.

    Uses config/live.yaml preprocessing (config-first) — notably `denoise: none` on the
    diarization path (spectral denoise there distorts speaker embeddings and over-counts
    speakers); the final per-speaker tracks are denoised later via `denoise_output`.
    """
    source_path = Path(source_path)
    job_id = uuid.uuid4().hex
    job_dir = Path(output_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "processed.wav"

    cfg = load_config(LIVE_CFG).preprocessing
    notes: list[str] = []
    preprocessor = Preprocessor(cfg)
    preprocessor.load()
    clip = preprocessor.run(source_path, notes)
    sf.write(output_path, clip.samples, clip.sample_rate)

    input_info = sf.info(str(source_path))
    output_info = sf.info(str(output_path))
    result: dict[str, object] = {
        "job_id": job_id,
        "filename": source_path.name,
        "output_filename": f"{source_path.stem}_preprocessed.wav",
        "download_name": str(output_path.name),
        "input": {"sample_rate": input_info.samplerate, "channels": input_info.channels},
        "output": {
            "sample_rate": output_info.samplerate,
            "channels": output_info.channels,
            "duration_s": round(output_info.frames / output_info.samplerate, 2),
            "peak": round(float(np.max(np.abs(clip.samples))) if clip.samples.size else 0.0, 4),
        },
        "settings": {
            "denoise": str(cfg.denoise),
            "denoise_output": str(getattr(cfg, "denoise_output", "none")),
            "normalize": str(cfg.normalize),
            "highpass_hz": int(getattr(cfg, "highpass_hz", 0) or 0),
            "trim_silence": bool(cfg.trim_silence),
        },
        "notes": notes,
        "processed_path": str(output_path),
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _job_dir(job_id: str, output_root: str | Path) -> Path:
    if not job_id or Path(job_id).name != job_id:
        raise ValueError("invalid job id")
    path = (Path(output_root) / job_id).resolve()
    if Path(output_root).resolve() not in path.parents:
        raise ValueError("invalid job path")
    if not path.is_dir():
        raise FileNotFoundError("job not found")
    return path


def run_diarization(job_id: str, output_root: str | Path) -> dict[str, object]:
    """Run Stage 2 on the conditioned clip and persist the timeline JSON."""
    job_dir = _job_dir(job_id, output_root)
    audio_path = job_dir / "processed.wav"
    samples, sample_rate = sf.read(str(audio_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    clip = AudioClip(samples=samples, sample_rate=sample_rate, source_path=str(audio_path))

    cfg = load_config(LIVE_CFG)   # real diarization + pyannote overlap detection
    notes: list[str] = []
    diarizer = Diarizer(cfg.diarization, cfg.run)
    diarizer.load()
    result = diarizer.run(clip, notes)
    payload = {
        "job_id": job_id,
        "duration_s": round(clip.duration, 2),
        "num_speakers": result.num_speakers,
        "turns": result.to_timeline_json(),
        "overlap_regions": [
            {"start_s": round(start, 3), "end_s": round(end, 3)}
            for start, end in result.overlap_regions
        ],
        "notes": notes,
    }
    (job_dir / "timeline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_separation(job_id: str, output_root: str | Path) -> dict[str, object]:
    """Run Stage 3 only when Stage 2 found merged/overlapping speech."""
    job_dir = _job_dir(job_id, output_root)
    timeline = json.loads((job_dir / "timeline.json").read_text(encoding="utf-8"))
    overlap_regions = [
        (float(region["start_s"]), float(region["end_s"]))
        for region in timeline["overlap_regions"]
    ]
    samples, sample_rate = sf.read(str(job_dir / "processed.wav"), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    clip = AudioClip(samples=samples, sample_rate=sample_rate, source_path=str(job_dir / "processed.wav"))
    turns = [
        SpeakerTurn(float(turn["start_s"]), float(turn["end_s"]), str(turn["speaker"]))
        for turn in timeline["turns"]
    ]
    diarization = DiarizationResult(
        turns=turns,
        overlap_regions=overlap_regions,
        num_speakers=int(timeline["num_speakers"]),
    )
    cfg = load_config(LIVE_CFG)
    notes: list[str] = []
    separated = []
    if overlap_regions:
        separator = Separator(cfg.separation)
        separator.load()
        separated = separator.run(clip, diarization, notes)
    else:
        notes.append("no merged speech intervals; separation skipped")

    separated_dir = job_dir / "separated"
    separated_dir.mkdir(exist_ok=True)
    separated_files: list[dict[str, object]] = []
    for index, source in enumerate(separated):
        path = separated_dir / f"source_{index:03d}_{source.speaker}.wav"
        sf.write(path, source.samples, clip.sample_rate)
        separated_files.append({
            "speaker": source.speaker,
            "filename": path.name,
            "region": {"start_s": source.region[0], "end_s": source.region[1]},
            "url": f"/api/file?job={job_id}&kind=separated&name={path.name}",
            "waveform_url": f"/api/waveform?job={job_id}&kind=separated&name={path.name}",
        })

    speaker_dir = job_dir / "speakers"
    speaker_dir.mkdir(exist_ok=True)
    tracks = build_speaker_tracks(clip, turns, overlap_regions, separated)
    denoise_output = str(getattr(cfg.preprocessing, "denoise_output", "none")) == "spectral"
    speaker_files: list[dict[str, object]] = []
    for speaker, track in sorted(tracks.items()):
        if denoise_output:                       # clean the final per-speaker track (market noise)
            track = spectral_denoise(track, clip.sample_rate)
        path = speaker_dir / f"{speaker}.wav"
        sf.write(path, track, clip.sample_rate)
        speaker_files.append({
            "speaker": speaker,
            "url": f"/api/file?job={job_id}&kind=speakers&name={path.name}",
            "waveform_url": f"/api/waveform?job={job_id}&kind=speakers&name={path.name}",
        })

    payload = {
        "job_id": job_id,
        "skipped": not bool(overlap_regions),
        "separated_sources": separated_files,
        "speaker_files": speaker_files,
        "notes": notes,
    }
    (job_dir / "separation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_asr(job_id: str, output_root: str | Path) -> dict[str, object]:
    """Run the configured ASR backend (faster-whisper by default) on the speaker tracks."""
    job_dir = _job_dir(job_id, output_root)
    timeline = json.loads((job_dir / "timeline.json").read_text(encoding="utf-8"))
    samples, sample_rate = sf.read(str(job_dir / "processed.wav"), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    clip = AudioClip(samples=samples, sample_rate=sample_rate, source_path=str(job_dir / "processed.wav"))
    turns = [
        SpeakerTurn(float(turn["start_s"]), float(turn["end_s"]), str(turn["speaker"]))
        for turn in timeline["turns"]
    ]
    overlap_regions = [
        (float(region["start_s"]), float(region["end_s"]))
        for region in timeline["overlap_regions"]
    ]
    separated: list[SeparatedSource] = []
    separation_path = job_dir / "separation.json"
    if separation_path.exists():
        separation = json.loads(separation_path.read_text(encoding="utf-8"))
        for source in separation["separated_sources"]:
            source_path = job_dir / "separated" / str(source["filename"])
            source_samples, _ = sf.read(str(source_path), dtype="float32")
            separated.append(SeparatedSource(
                region=(float(source["region"]["start_s"]), float(source["region"]["end_s"])),
                speaker=str(source["speaker"]), samples=source_samples,
            ))
    cfg = load_config(LIVE_CFG)
    diarization = DiarizationResult(
        turns=turns, overlap_regions=overlap_regions,
        num_speakers=int(timeline["num_speakers"]),
    )
    notes: list[str] = []
    asr = make_asr(cfg.asr)          # respects cfg.asr.backend (free faster_whisper by default)
    asr.load()
    transcript = asr.run(clip, diarization, separated, notes)
    segments = [
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": segment.speaker,
            "text": segment.text,
            "lang": segment.lang,
            "words": [
                {"text": word.text, "start": round(word.start, 3), "end": round(word.end, 3)}
                for word in segment.words
            ],
        }
        for segment in transcript.segments
    ]
    payload: dict[str, object] = {
        "job_id": job_id,
        "backend": str(cfg.asr.backend),
        "model": str(cfg.asr.model),
        "segments": segments,
        "notes": notes,
    }
    (job_dir / "transcript.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_llm(job_id: str, output_root: str | Path) -> dict[str, object]:
    """Run Stage 5 (LLM post-processing) on the transcript -> report (summary, tags, dialogue).

    Rebuilds the Transcript from transcript.json and routes it through the configured backend
    (Gemini by default); degrades to a minimal report if the LLM is unavailable.
    """
    job_dir = _job_dir(job_id, output_root)
    transcript_json = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    segments = [
        TranscriptSegment(
            start=float(s["start"]), end=float(s["end"]), speaker=str(s["speaker"]),
            text=str(s["text"]), lang=s.get("lang"),
            words=[Word(str(w["text"]), float(w["start"]), float(w["end"]))
                   for w in s.get("words", [])],
        )
        for s in transcript_json.get("segments", [])
    ]
    transcript = Transcript(segments=segments)
    cfg = load_config(LIVE_CFG)
    notes: list[str] = []
    post = make_postprocessor(cfg.postprocessing)
    post.load()
    report = post.run(transcript, notes)
    payload: dict[str, object] = {
        "job_id": job_id,
        "backend": str(cfg.postprocessing.backend),
        "model": str(cfg.postprocessing.model),
        "summary": report.summary,
        "tags": report.tags,
        "dialogue": [
            {"start": round(s.start, 3), "end": round(s.end, 3),
             "speaker": s.speaker, "text": s.text}
            for s in report.dialogue
        ],
        "notes": notes,
    }
    (job_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def make_upload_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "dhvani_preprocess_ui" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path
