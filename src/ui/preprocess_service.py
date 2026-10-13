"""Stage 1-only service used by the standalone preprocessing web UI."""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from ..config import load_config
from ..diarization import Diarizer
from ..preprocessing import Preprocessor
from ..separation import Separator
from ..separation.stitch import build_speaker_tracks
from ..types import AudioClip, DiarizationResult, SpeakerTurn

REPO_ROOT = Path(__file__).resolve().parents[2]


def process_upload(
    source_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Run the fixed Stage 1 recipe and return download metadata."""
    source_path = Path(source_path)
    job_id = uuid.uuid4().hex
    job_dir = Path(output_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "processed.wav"

    cfg = OmegaConf.create({
        "target_sample_rate": 16000,
        "mono": True,
        "highpass_hz": 70,
        "denoise": "spectral",
        "normalize": "peak",
        "target_lufs": -23.0,
        "trim_silence": True,
        "trim_top_db": 30,
        "chunk_seconds": 0,
    })
    notes: list[str] = []
    preprocessor = Preprocessor(cfg)
    preprocessor.load()
    clip = preprocessor.run(source_path, notes)
    sf.write(output_path, clip.samples, clip.sample_rate)

    input_info = sf.info(str(source_path))
    output_info = sf.info(str(output_path))
    result = {
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
            "denoise": "spectral",
            "normalize": "peak",
            "highpass_hz": 70,
            "trim_silence": True,
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

    cfg = load_config(REPO_ROOT / "config" / "default.yaml")
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
    notes: list[str] = []
    separated = []
    if overlap_regions:
        cfg = load_config(REPO_ROOT / "config" / "default.yaml")
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
            "region": {"start_s": source.region[0], "end_s": source.region[1]},
            "url": f"/api/file?job={job_id}&kind=separated&name={path.name}",
            "waveform_url": f"/api/waveform?job={job_id}&kind=separated&name={path.name}",
        })

    speaker_dir = job_dir / "speakers"
    speaker_dir.mkdir(exist_ok=True)
    tracks = build_speaker_tracks(clip, turns, overlap_regions, separated)
    speaker_files: list[dict[str, object]] = []
    for speaker, track in sorted(tracks.items()):
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


def make_upload_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "dhvani_preprocess_ui" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path