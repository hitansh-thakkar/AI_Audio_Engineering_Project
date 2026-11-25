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
    for k in ks:
        labels = _fit_spectral(emb, k, seed)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _merge_similar(emb: np.ndarray, labels: np.ndarray, thr: float, floor: int) -> np.ndarray:
    """Greedily merge the two clusters with the most similar centroids while their cosine
    is >= thr and more than `floor` clusters remain. Returns labels renumbered 0..k-1."""
    labels = labels.copy()
    while True:
        ids = sorted(set(labels.tolist()))
        if len(ids) <= max(1, floor):
            break
        cents = np.stack([emb[labels == i].mean(axis=0) for i in ids])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
        sim = cents @ cents.T
        np.fill_diagonal(sim, -np.inf)
        a, b = np.unravel_index(int(np.argmax(sim)), sim.shape)
        if sim[a, b] < thr:
            break
        labels[labels == ids[b]] = ids[a]
    _, renumbered = np.unique(labels, return_inverse=True)
    return renumbered.astype(int)


def _fit_spectral(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    sc = SpectralClustering(
        n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=seed,
    )
    return sc.fit_predict(_cosine_affinity(emb))


def cluster_embeddings(emb: np.ndarray, method: str = "spectral", num_speakers=None,
                       min_speakers=None, max_speakers=None, seed: int = 42,
                       notes: list[str] | None = None,
                       single_speaker_cos: float | None = None) -> np.ndarray:
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
        # Silhouette is undefined for k=1 and tends to over-count on short clips. When
        # configured, cluster at the estimated k and then merge clusters whose centroids
        # still point the same way (same voice). Off (None) by default so benchmark numbers
        # are unchanged.
        if single_speaker_cos is not None and k > 1:
            labels = _fit_spectral(emb, min(k, n), seed).astype(int)
            floor = int(min_speakers) if min_speakers else 1
            merged = _merge_similar(emb, labels, single_speaker_cos, floor)
            n_before, n_after = len(set(labels)), len(set(merged))
            if n_after < n_before:
                notes.append(f"centroid merge (cos >= {single_speaker_cos}): {n_before} -> {n_after} speakers")
            return merged

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
