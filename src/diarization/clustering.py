"""REQUIRED (proposal Milestone 2): hand-applied unsupervised clustering of speaker
embeddings — "specifically Spectral Clustering or Gaussian Mixture Models (GMMs)."

No prior speaker templates. If the speaker count is unknown, it is estimated:
- spectral: silhouette score over a candidate range of k
- gmm:      Bayesian Information Criterion (BIC) over a candidate range of k

This is the production clustering path; pyannote's end-to-end clusterer is kept only as an
optional `builtin` comparison elsewhere.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture


def _candidate_ks(n: int, min_sp, max_sp) -> list[int]:
    lo = int(min_sp) if min_sp else 2
    hi = int(max_sp) if max_sp else min(8, n - 1)
    lo = max(2, lo)
    hi = max(lo, min(hi, n - 1))
    return list(range(lo, hi + 1))


def _cosine_affinity(emb: np.ndarray) -> np.ndarray:
    """Non-negative cosine-similarity affinity. Speaker embeddings are directional, so cosine
    separates voices far better than the Euclidean nearest-neighbors graph."""
    return np.clip(cosine_similarity(emb), 0.0, None)


def _estimate_k(emb: np.ndarray, ks: list[int], seed: int) -> int:
    """Pick the speaker count by the best cosine silhouette over candidate k (shared by both
    clustering backends — silhouette on the embedding geometry is a more reliable counter than
    GMM's BIC, which badly over-/under-counts speakers). Auto-counting is still imperfect; set
    diarization.num_speakers (UI 'Number of speakers') when known for the cleanest result."""
    best_k, best_score = ks[0], -2.0
    n = emb.shape[0]
    for k in ks:
        if not 2 <= k <= n - 1:  # silhouette undefined here (e.g. 2 windows); keep ks[0]
            continue
        labels = _fit_spectral(emb, k, seed)
        if len(set(labels)) < 2 or len(set(labels)) > n - 1:
            continue
        score = silhouette_score(emb, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _fit_spectral(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    )
    return sc.fit_predict(_cosine_affinity(emb))


def cluster_embeddings(
    emb: np.ndarray,
    method: str = "spectral",
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
    seed: int = 42,
    notes: list[str] | None = None,
) -> np.ndarray:
    """Cluster (n, d) embeddings into speaker labels (np.ndarray of ints)."""
    notes = [] if notes is None else notes
    n = emb.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int)
    if n == 1:
        return np.zeros((1,), dtype=int)

    if num_speakers is not None:
        k = max(1, int(num_speakers))
    else:
        ks = _candidate_ks(n, min_speakers, max_speakers)
        if not ks:
            notes.append("too few segments to estimate speaker count; assuming 1")
            return np.zeros((n,), dtype=int)
        k = _estimate_k(emb, ks, seed)

    k = min(k, n)
    if k <= 1:
        return np.zeros((n,), dtype=int)

    if method == "gmm":
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed)
        return gmm.fit_predict(emb).astype(int)
    if method == "spectral":
        return _fit_spectral(emb, k, seed).astype(int)
    notes.append(f"unknown clustering method '{method}'; using spectral")
    return _fit_spectral(emb, k, seed).astype(int)
