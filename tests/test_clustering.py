"""The compulsory Milestone-2 clustering must actually separate speakers."""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.diarization.clustering import cluster_embeddings


def _two_blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=+2.0, scale=0.3, size=(15, 8))
    b = rng.normal(loc=-2.0, scale=0.3, size=(15, 8))
    emb = np.vstack([a, b]).astype(np.float32)
    truth = np.array([0] * 15 + [1] * 15)
    # L2-normalize like the real embedder does
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, truth


def test_spectral_recovers_two_speakers():
    emb, truth = _two_blobs()
    labels = cluster_embeddings(emb, method="spectral", num_speakers=2, seed=42)
    assert adjusted_rand_score(truth, labels) > 0.9


def test_gmm_recovers_two_speakers():
    emb, truth = _two_blobs()
    labels = cluster_embeddings(emb, method="gmm", num_speakers=2, seed=42)
    assert adjusted_rand_score(truth, labels) > 0.9


def test_auto_k_finds_two():
    emb, _ = _two_blobs()
    labels = cluster_embeddings(emb, method="spectral", num_speakers=None,
                                min_speakers=1, max_speakers=5, seed=42)
    assert len(set(labels.tolist())) == 2


def test_single_segment_is_one_speaker():
    emb = np.ones((1, 8), dtype=np.float32)
    labels = cluster_embeddings(emb, method="spectral", seed=42)
    assert labels.tolist() == [0]
