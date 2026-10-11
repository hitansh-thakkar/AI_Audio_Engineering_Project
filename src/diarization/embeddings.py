"""Speaker-embedding extraction for the clustering step.

The hand-applied Spectral/GMM clustering (compulsory Milestone-2) is only as good as the
embeddings it sees. Two backends:
- `ecapa`: SpeechBrain ECAPA-TDNN speaker embeddings (open model, no gated licence) — far
  more discriminative than spectral features; the recommended real-mode default.
- `mfcc` / `auto`: MFCC-statistics (mean+std) — dependency-light, fully offline fallback used
  when ECAPA can't load. The clustering algorithm is identical either way.
"""
from __future__ import annotations

from typing import Any

import librosa
import numpy as np

_ECAPA: Any = None
_ECAPA_ERROR: str | None = None


def _mfcc_stats(seg: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    if seg.size < sr // 20:  # pad very short segments
        seg = np.pad(seg, (0, max(0, sr // 20 - seg.size)))
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def _load_ecapa(notes: list[str]):
    """Lazily load (and cache) the ECAPA encoder. Returns None if unavailable."""
    global _ECAPA, _ECAPA_ERROR
    if _ECAPA is not None:
        return _ECAPA
    if _ECAPA_ERROR is not None:
        return None
    try:
        from pathlib import Path

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path("models") / "ecapa-voxceleb"),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,   # Windows: avoid symlink-privilege failures
        )
        return _ECAPA
    except Exception as exc:
        _ECAPA_ERROR = f"ECAPA unavailable ({type(exc).__name__}); using MFCC embeddings"
        notes.append(_ECAPA_ERROR)
        return None


def _ecapa_vec(encoder, seg: np.ndarray, sr: int) -> np.ndarray:
    import torch
    if seg.size < sr // 10:                      # pad < 100 ms so the encoder is stable
        seg = np.pad(seg, (0, sr // 10 - seg.size))
    with torch.no_grad():
        v = encoder.encode_batch(torch.from_numpy(seg.astype(np.float32))[None, :])
    return v.squeeze().cpu().numpy().astype(np.float32)


def _normalize_rows(vecs: list[np.ndarray], dim: int) -> np.ndarray:
    if not vecs:
        return np.zeros((0, dim), dtype=np.float32)
    emb = np.vstack(vecs)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def embed_segments(samples: np.ndarray, sr: int, segments: list[tuple[float, float]],
                   backend: str = "auto", notes: list[str] | None = None) -> np.ndarray:
    """Return an (n_segments, dim) L2-normalized embedding matrix."""
    notes = [] if notes is None else notes
    if backend not in ("auto", "mfcc", "ecapa", "pyannote"):
        notes.append(f"unknown embedding backend '{backend}'; using MFCC")
        backend = "mfcc"

    encoder = _load_ecapa(notes) if backend == "ecapa" else None
    if backend == "ecapa" and encoder is not None:
        vecs = [_ecapa_vec(encoder, samples[int(s * sr):int(e * sr)], sr) for s, e in segments]
        return _normalize_rows(vecs, 192)

    # mfcc / auto / pyannote-not-wired / ecapa-unavailable -> MFCC statistics
    vecs = [_mfcc_stats(samples[int(s * sr):int(e * sr)], sr) for s, e in segments]
    return _normalize_rows(vecs, 2 * 20)


def embed_waveform(samples: np.ndarray, sr: int, backend: str = "auto",
                   notes: list[str] | None = None) -> np.ndarray:
    """L2-normalized embedding of a standalone waveform (whole array as one segment).

    Used to compare a separated source against a speaker's voiceprint; same MFCC-stats
    representation as the clustering embedder so the spaces are comparable.
    """
    if samples.size == 0:
        return np.zeros(2 * 20, dtype=np.float32)
    seconds = len(samples) / sr
    return embed_segments(samples, sr, [(0.0, seconds)], backend, notes)[0]
