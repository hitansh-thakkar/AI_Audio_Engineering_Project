"""Link speaker labels across chunks.

Long clips are diarized chunk-by-chunk, so each chunk invents its own SPEAKER_00/01/...
labels — the same person is a different label in every chunk. We reconcile them by embedding
each per-chunk speaker's audio and clustering those embeddings (the same compulsory
Spectral/GMM step), so one voice collapses to one global label across the whole broadcast.
"""
from __future__ import annotations

import numpy as np

from .clustering import cluster_embeddings
from .embeddings import embed_waveform


def link_speakers(labeled_audio: dict[str, np.ndarray], sr: int, embedding: str = "auto",
                  clustering: str = "spectral", seed: int = 42,
                  notes: list[str] | None = None) -> dict[str, str]:
    """Map each provisional (per-chunk) label to a global SPEAKER_XX label.

    `labeled_audio` maps a provisional label -> that speaker's concatenated audio in its chunk.
    """
    notes = [] if notes is None else notes
    labels = sorted(labeled_audio)
    if len(labels) <= 1:
        return {label: "SPEAKER_00" for label in labels}

    emb = np.vstack([embed_waveform(labeled_audio[label], sr, embedding, notes)
                     for label in labels])
    clusters = cluster_embeddings(emb, method=clustering, num_speakers=None,
                                  seed=seed, notes=notes)
    return {label: f"SPEAKER_{int(clusters[i]):02d}" for i, label in enumerate(labels)}
