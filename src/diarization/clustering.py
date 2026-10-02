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
from sklearn.mixture import GaussianMixture


def _candidate_ks(n: int, min_sp, max_sp) -> list[int]:
    lo = int(min_sp) if min_sp else 2
    hi = int(max_sp) if max_sp else min(8, n - 1)
    lo = max(2, lo)
    hi = max(lo, min(hi, n - 1))
    return list(range(lo, hi + 1))


def _estimate_k_spectral(emb: np.ndarray, ks: list[int], seed: int) -> int:
    best_k, best_score = ks[0], -1.0
    for k in ks:
        labels = _fit_spectral(emb, k, seed)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _estimate_k_gmm(emb: np.ndarray, ks: list[int], seed: int) -> int:
    best_k, best_bic = ks[0], np.inf
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed)
        gmm.fit(emb)
        bic = gmm.bic(emb)
        if bic < best_bic:
            best_k, best_bic = k, bic
    return best_k


def _fit_spectral(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = emb.shape[0]
    n_neighbors = min(10, max(2, n - 1))
    sc = SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors", n_neighbors=n_neighbors,
        assign_labels="kmeans", random_state=seed,
    )
    return sc.fit_predict(emb)


def cluster_embeddings(emb: np.ndarray, method: str = "spectral", num_speakers=None,
                       min_speakers=None, max_speakers=None, seed: int = 42,
                       notes: list[str] | None = None) -> np.ndarray:
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
        k = (_estimate_k_spectral(emb, ks, seed) if method == "spectral"
             else _estimate_k_gmm(emb, ks, seed))

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
