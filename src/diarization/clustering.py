"""REQUIRED (proposal Milestone 2): hand-applied unsupervised clustering of speaker
embeddings — "specifically Spectral Clustering or Gaussian Mixture Models (GMMs)."

No prior speaker templates. If the speaker count is unknown, it is estimated:
- count_method="silhouette": best cosine silhouette over a candidate range of k.
- count_method="eigengap":   largest normalized-Laplacian eigengap over the range (the
  classic spectral-clustering count heuristic; far more stable than silhouette, which
  over-counts when many embedding windows come from the same voice — the failure mode
  that made "everything before the overlap = speaker 1, everything after = speaker 2").

The affinity feeding spectral clustering can be:
- affinity="cosine":  raw non-negative cosine similarity (legacy).
- affinity="refined": cosine similarity pruned to each row's nearest neighbours and
  symmetrised. Pruning removes weak cross-speaker edges so the graph cut lands on the
  speaker boundary instead of the time boundary.

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


def _refined_affinity(emb: np.ndarray, prune_frac: float = 0.3) -> np.ndarray:
    """Cosine affinity pruned to each row's top-`prune_frac` neighbours, then symmetrised.

    Weak edges between windows of *different* speakers are what let a spectral cut fall on a
    time boundary; pruning them to the strongest neighbours sharpens the block structure so
    the cut lands on the speaker boundary. (Row-wise k-NN graph, symmetrised by max — the
    standard refinement from auto-tuning spectral-clustering diarization.)
    """
    aff = _cosine_affinity(emb)
    n = aff.shape[0]
    keep = max(1, int(round(prune_frac * n)))
    pruned = np.zeros_like(aff)
    for i in range(n):
        idx = np.argpartition(aff[i], -keep)[-keep:]
        pruned[i, idx] = aff[i, idx]
    out = np.maximum(pruned, pruned.T)          # symmetrise
    np.fill_diagonal(out, 1.0)
    return out


def _affinity(emb: np.ndarray, mode: str) -> np.ndarray:
    return _refined_affinity(emb) if mode == "refined" else _cosine_affinity(emb)


def _fit_spectral_aff(aff: np.ndarray, k: int, seed: int) -> np.ndarray:
    sc = SpectralClustering(
        n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=seed,
    )
    return sc.fit_predict(aff)


def _estimate_k_silhouette(emb: np.ndarray, aff: np.ndarray, ks: list[int], seed: int) -> int:
    """Pick the speaker count by the best cosine silhouette over candidate k."""
    best_k, best_score = ks[0], -2.0
    for k in ks:
        labels = _fit_spectral_aff(aff, k, seed)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _estimate_k_eigengap(aff: np.ndarray, ks: list[int]) -> int:
    """Pick k at the largest gap between consecutive eigenvalues of the normalized Laplacian.

    L = I - D^{-1/2} A D^{-1/2}; with k well-separated clusters the k smallest eigenvalues are
    near 0 and eigenvalue k+1 jumps up, so the biggest gap in the candidate range marks k.
    """
    d = aff.sum(axis=1)
    d[d <= 0] = 1e-12
    dinv = 1.0 / np.sqrt(d)
    lap = np.eye(aff.shape[0]) - (aff * dinv[:, None]) * dinv[None, :]
    lap = (lap + lap.T) / 2.0                   # guard numerical asymmetry
    eigvals = np.sort(np.linalg.eigvalsh(lap))
    gaps = {k: float(eigvals[k] - eigvals[k - 1]) for k in ks if k < len(eigvals)}
    return max(gaps, key=lambda k: gaps[k]) if gaps else ks[0]


def cluster_embeddings(emb: np.ndarray, method: str = "spectral", num_speakers=None,
                       min_speakers=None, max_speakers=None, seed: int = 42,
                       count_method: str = "silhouette", affinity: str = "cosine",
                       notes: list[str] | None = None) -> np.ndarray:
    """Cluster (n, d) embeddings into speaker labels (np.ndarray of ints).

    `count_method` / `affinity` only matter when `num_speakers` is unknown and `method` is
    spectral; defaults keep the legacy behaviour so existing callers/tests are unaffected.
    """
    notes = [] if notes is None else notes
    n = emb.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int)
    if n == 1:
        return np.zeros((1,), dtype=int)

    aff = _affinity(emb, affinity)

    if num_speakers is not None:
        k = max(1, int(num_speakers))
    else:
        ks = _candidate_ks(n, min_speakers, max_speakers)
        if not ks:
            notes.append("too few segments to estimate speaker count; assuming 1")
            return np.zeros((n,), dtype=int)
        if count_method == "eigengap":
            k = _estimate_k_eigengap(aff, ks)
        else:
            k = _estimate_k_silhouette(emb, aff, ks, seed)

    k = min(k, n)
    if k <= 1:
        return np.zeros((n,), dtype=int)

    if method == "gmm":
        return GaussianMixture(
            n_components=k, covariance_type="diag", random_state=seed).fit_predict(emb).astype(int)
    if method != "spectral":
        notes.append(f"unknown clustering method '{method}'; using spectral")
    return _fit_spectral_aff(aff, k, seed).astype(int)
