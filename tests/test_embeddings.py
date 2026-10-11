"""Embedding backends: ECAPA when available, graceful MFCC fallback, always L2-normalized."""
import numpy as np

from src.diarization import embeddings
from src.diarization.embeddings import embed_segments


def _clip():
    t = np.arange(16000) / 16000
    return (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 16000


def test_mfcc_backend_is_offline_and_normalized():
    y, sr = _clip()
    emb = embed_segments(y, sr, [(0.0, 0.5), (0.5, 1.0)], backend="mfcc")
    assert emb.shape == (2, 40)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_ecapa_falls_back_to_mfcc_when_unavailable(monkeypatch):
    monkeypatch.setattr(embeddings, "_load_ecapa", lambda notes: notes.append("ecapa down") or None)
    y, sr = _clip()
    notes: list[str] = []
    emb = embed_segments(y, sr, [(0.0, 1.0)], backend="ecapa", notes=notes)
    assert emb.shape == (1, 40)             # fell back to MFCC dim
    assert any("ecapa down" in n for n in notes)
