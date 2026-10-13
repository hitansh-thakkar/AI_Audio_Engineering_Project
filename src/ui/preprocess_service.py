"""Stage 1-only service used by the standalone preprocessing web UI."""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from ..preprocessing import Preprocessor


def process_upload(
    source_path: str | Path,
    output_root: str | Path,
    *,
    denoise: str = "spectral",
    normalize: str = "peak",
    highpass_hz: int = 70,
    trim_silence: bool = True,
) -> dict[str, object]:
    """Run only the configurable preprocessing stage and return download metadata."""
    source_path = Path(source_path)
    job_id = uuid.uuid4().hex
    job_dir = Path(output_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "processed.wav"

    cfg = OmegaConf.create({
        "target_sample_rate": 16000,
        "mono": True,
        "highpass_hz": max(0, int(highpass_hz)),
        "denoise": denoise if denoise in {"none", "spectral"} else "spectral",
        "normalize": normalize if normalize in {"none", "peak", "lufs"} else "peak",
        "target_lufs": -23.0,
        "trim_silence": bool(trim_silence),
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
            "denoise": str(cfg.denoise),
            "normalize": str(cfg.normalize),
            "highpass_hz": int(cfg.highpass_hz),
            "trim_silence": bool(cfg.trim_silence),
        },
        "notes": notes,
        "processed_path": str(output_path),
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def make_upload_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "dhvani_preprocess_ui" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path