"""Stage 1 — audio conditioning: resample to 16 kHz mono, high-pass, denoise, normalize, trim.

The proposal's audio is "low-cost mobile phones ... surrounded by ambient village or market
noise", so conditioning matters: an optional high-pass kills mains/handling rumble, a
dependency-light spectral-gating denoiser attenuates stationary background noise, and LUFS
normalization (ITU-R BS.1770) gives consistent loudness across recordings. Each step is a
config toggle and degrades gracefully. `denoise: demucs/dns` remain optional heavy backends.
"""
from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf

from ..types import AudioClip


def highpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass — removes sub-cutoff rumble (mains hum, mic handling)."""
    if cutoff_hz <= 0 or y.size == 0:
        return y
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * sr
    b, a = butter(order, min(cutoff_hz / nyq, 0.99), btype="high")
    padlen = 3 * max(len(a), len(b))
    if y.size <= padlen:
        return y
    return filtfilt(b, a, y).astype(np.float32)


def spectral_denoise(y: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256,
                     over_sub: float = 1.5, floor: float = 0.05,
                     noise_pct: float = 10.0) -> np.ndarray:
    """Spectral-gating denoise (no model). Estimate the noise magnitude per frequency bin from
    the quietest frames, then over-subtract it with a spectral floor. Good for stationary
    ambient/market noise; pure numpy+librosa so it stays offline and dependency-light."""
    if y.size < n_fft:
        return y
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(stft), np.angle(stft)
    frame_energy = mag.sum(axis=0)
    thresh = np.percentile(frame_energy, noise_pct)
    quiet = mag[:, frame_energy <= thresh]
    noise_mag = (quiet.mean(axis=1, keepdims=True) if quiet.size
                 else mag.min(axis=1, keepdims=True))
    clean_mag = np.maximum(mag - over_sub * noise_mag, floor * mag)
    out = librosa.istft(clean_mag * np.exp(1j * phase), hop_length=hop, length=len(y))
    return out.astype(np.float32)


def measure_lufs(y: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770 via pyloudnorm)."""
    import pyloudnorm as pyln
    return float(pyln.Meter(sr).integrated_loudness(y.astype(np.float64)))


def lufs_normalize(y: np.ndarray, sr: int, target_lufs: float = -23.0,
                   peak_ceiling: float = 0.97) -> np.ndarray:
    """Scale to a target integrated loudness, then guard against clipping (true-peak-ish)."""
    if y.size == 0:
        return y
    loudness = measure_lufs(y, sr)
    if not np.isfinite(loudness):            # silent / all-zero input
        return y
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    out = y * gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(np.float32)


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

        # high-pass (optional): strip sub-cutoff rumble from mobile mics / mains hum
        hp = float(getattr(self.cfg, "highpass_hz", 0) or 0)
        if hp > 0:
            y = highpass(y, sr, hp)

        # denoise (optional)
        denoise = str(self.cfg.denoise)
        if denoise == "spectral":
            y = spectral_denoise(y, sr)
        elif denoise != "none":
            notes.append(f"denoise '{denoise}' backend not installed; skipped (use 'spectral')")

        # normalize
        mode = str(self.cfg.normalize)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if mode == "peak" and peak > 0:
            y = y * (0.9 / peak)
        elif mode == "lufs":
            try:
                y = lufs_normalize(y, sr, float(getattr(self.cfg, "target_lufs", -23.0)))
            except Exception as exc:                 # pyloudnorm missing / degenerate input
                if peak > 0:
                    y = y * (0.9 / peak)
                notes.append(f"lufs normalization unavailable ({type(exc).__name__}); used peak")
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
