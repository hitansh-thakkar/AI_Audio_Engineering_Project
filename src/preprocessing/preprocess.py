"""Stage 1 — audio conditioning: resample to 16 kHz mono, normalize, trim silence.

Note: `denoise` (demucs/dns) is a music/vocal split used ONLY as a denoiser — it does
NOT separate speakers. Off by default. Real DNS backend plugs in behind the toggle.
"""
from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf

from ..types import AudioClip


class Preprocessor:
    name = "preprocessing"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"sr{cfg.target_sample_rate}-{cfg.normalize}-denoise_{cfg.denoise}"

    def load(self) -> None:
        pass

    def run(self, source, notes: list[str] | None = None) -> AudioClip:
        notes = [] if notes is None else notes
        sr_target = int(self.cfg.target_sample_rate)

        if isinstance(source, AudioClip):
            y, sr, path = source.samples.astype(np.float32), source.sample_rate, source.source_path
        else:
            y, sr = sf.read(str(source))
            y = y.astype(np.float32)
            path = str(source)

        # mono
        if self.cfg.mono and y.ndim > 1:
            y = y.mean(axis=1)

        # resample
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        # denoise (optional; real backend not wired here — graceful no-op + note)
        if str(self.cfg.denoise) != "none":
            notes.append(f"denoise '{self.cfg.denoise}' requested but backend not installed; skipped")

        # normalize
        mode = str(self.cfg.normalize)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if mode == "peak" and peak > 0:
            y = y * (0.9 / peak)
        elif mode == "lufs":
            # True LUFS needs pyloudnorm; fall back to peak and note it.
            if peak > 0:
                y = y * (0.9 / peak)
            notes.append("lufs normalization unavailable (pyloudnorm missing); used peak")
        # mode == "none": leave as-is

        # trim leading/trailing silence
        if bool(self.cfg.trim_silence) and y.size:
            y, _ = librosa.effects.trim(y, top_db=int(self.cfg.trim_top_db))

        return AudioClip(samples=y.astype(np.float32), sample_rate=sr, source_path=path)


def chunk_audio(clip: AudioClip, chunk_seconds: float) -> list[AudioClip]:
    """Split a clip into contiguous pieces of at most `chunk_seconds`.

    Long broadcasts are processed in windows (the demo handles 30–120 s clips, not whole
    broadcasts). Returns `[clip]` unchanged when it already fits in one window or chunking
    is disabled (chunk_seconds falsy / <= 0). Chunks keep the parent's source_path; the
    orchestrator offsets each chunk's timestamps by chunk_index * chunk_seconds.
    """
    if not chunk_seconds or chunk_seconds <= 0:
        return [clip]
    step = int(chunk_seconds * clip.sample_rate)
    n = len(clip.samples)
    if step <= 0 or n <= step:
        return [clip]
    return [
        AudioClip(samples=clip.samples[start:start + step], sample_rate=clip.sample_rate,
                  source_path=clip.source_path)
        for start in range(0, n, step)
    ]
