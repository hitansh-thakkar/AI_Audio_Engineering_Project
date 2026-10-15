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
from ..diarization.embeddings import embed_segments
from ..diarization.vad import energy_vad, window_segments
from ..preprocessing import Preprocessor
from ..separation import Separator
from ..separation.stitch import build_speaker_tracks
from ..asr.sarvam import SarvamASR
from ..types import AudioClip, DiarizationResult, SeparatedSource, SpeakerTurn
from .viz import energy_vad_chart, embedding_heatmap, speaker_gantt, separation_snr_chart

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


def run_diarization(
    job_id: str,
    output_root: str | Path,
) -> dict[str, object]:
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

    # ── matplotlib plots ────────────────────────────────────────────────────
    plots_dir = job_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_urls: dict[str, str] = {}

    # 1. Energy VAD chart — re-run VAD (cheap, no model) so we have the raw regions
    try:
        top_db = int(getattr(cfg.diarization, "vad_top_db", 30))
        vad_regions = energy_vad(samples, sample_rate, top_db=top_db)
        vad_path = plots_dir / "energy_vad.png"
        energy_vad_chart(samples, sample_rate, vad_regions, vad_path)
        plot_urls["energy_vad"] = f"/api/plot?job={job_id}&name=energy_vad.png"
    except Exception as exc:
        notes.append(f"energy VAD chart skipped: {exc}")

    # 2. Embedding similarity heatmap — re-extract embeddings (same path the Diarizer took)
    try:
        backend_str = str(cfg.diarization.embedding)
        win = float(getattr(cfg.diarization, "window_seconds", 1.5))
        hop = float(getattr(cfg.diarization, "window_hop", 0.75))
        vad_regions_for_emb = energy_vad(samples, sample_rate, top_db=top_db)
        segments = window_segments(vad_regions_for_emb, win=win, hop=hop)
        if len(segments) >= 2:
            embeddings = embed_segments(samples, sample_rate, segments,
                                        backend=backend_str, notes=notes)
            # build per-segment speaker labels from result turns
            speaker_map = {spk: idx for idx, spk in enumerate(sorted({t.speaker for t in result.turns}))}
            labels = np.zeros(len(segments), dtype=int)
            for seg_idx, (seg_start, _) in enumerate(segments):
                mid = seg_start + win / 2
                for turn in result.turns:
                    if turn.start <= mid <= turn.end:
                        labels[seg_idx] = speaker_map.get(turn.speaker, 0)
                        break
            heatmap_path = plots_dir / "embedding_heatmap.png"
            embedding_heatmap(embeddings, labels, heatmap_path)
            plot_urls["embedding_heatmap"] = f"/api/plot?job={job_id}&name=embedding_heatmap.png"
    except Exception as exc:
        notes.append(f"embedding heatmap skipped: {exc}")

    # 3. Speaker Gantt chart
    try:
        turns_for_gantt = result.to_timeline_json()
        overlaps_for_gantt = [
            {"start_s": s, "end_s": e} for s, e in result.overlap_regions
        ]
        gantt_path = plots_dir / "speaker_gantt.png"
        speaker_gantt(turns_for_gantt, overlaps_for_gantt, clip.duration, gantt_path)
        plot_urls["speaker_gantt"] = f"/api/plot?job={job_id}&name=speaker_gantt.png"
    except Exception as exc:
        notes.append(f"speaker gantt skipped: {exc}")

    backend_label = "pyannote.ai precision-2" if any("pyannote.ai" in n for n in notes) else str(cfg.diarization.backend)
    payload = {
        "job_id": job_id,
        "backend": backend_label,
        "duration_s": round(clip.duration, 2),
        "num_speakers": result.num_speakers,
        "turns": result.to_timeline_json(),
        "overlap_regions": [
            {"start_s": round(start, 3), "end_s": round(end, 3)}
            for start, end in result.overlap_regions
        ],
        "plots": plot_urls,
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
            "filename": path.name,
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

    # ── matplotlib separation chart ─────────────────────────────────────────
    plot_urls: dict[str, str] = {}
    if separated:
        try:
            plots_dir = job_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            sources_for_viz = [
                {"speaker": src.speaker, "samples": src.samples}
                for src in separated
            ]
            snr_path = plots_dir / "separation_sources.png"
            separation_snr_chart(
                sources_for_viz, snr_path,
                clip_samples=clip.samples, sr=clip.sample_rate,
            )
            plot_urls["separation_sources"] = (
                f"/api/plot?job={job_id}&name=separation_sources.png"
            )
        except Exception as exc:
            notes.append(f"separation chart skipped: {exc}")

    payload["plots"] = plot_urls

    (job_dir / "separation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_asr(job_id: str, output_root: str | Path) -> dict[str, object]:
    """Run the Sarvam ASR adapter on the completed speaker-track artifacts."""
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
    cfg = load_config(REPO_ROOT / "config" / "live.yaml")
    diarization = DiarizationResult(
        turns=turns, overlap_regions=overlap_regions,
        num_speakers=int(timeline["num_speakers"]),
    )
    notes: list[str] = []
    asr = SarvamASR(cfg.asr)
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
    payload = {
        "job_id": job_id,
        "backend": str(cfg.asr.backend),
        "model": str(cfg.asr.model),
        "segments": segments,
        "notes": notes,
    }
    (job_dir / "transcript.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def make_upload_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "dhvani_preprocess_ui" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path