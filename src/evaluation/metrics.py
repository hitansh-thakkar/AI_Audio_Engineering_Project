"""Metric implementations: DER (diarization), SI-SNRi / SDRi (separation), WER/CER (ASR).

DER here is a frame-based, no-collar implementation with optimal (Hungarian) speaker-label
mapping, scored on the dominant speaker per frame. This is a standard simplification that is
exact for single-speaker-per-frame audio (our main case) and is documented as such; full
md-eval overlap scoring is a post-midsem refinement.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types import SpeakerTurn
from .normalize import normalize_text


# --------------------------------------------------------------------- DER
def _frame_labels(turns: list[SpeakerTurn], n_frames: int, frame: float) -> list[str | None]:
    """Dominant speaker per frame (None = non-speech)."""
    labels: list[str | None] = [None] * n_frames
    for t in turns:
        i0 = max(0, int(round(t.start / frame)))
        i1 = min(n_frames, int(round(t.end / frame)))
        for i in range(i0, i1):
            if labels[i] is None:  # first writer wins (dominant); keeps it deterministic
                labels[i] = t.speaker
    return labels


def der(ref: list[SpeakerTurn], hyp: list[SpeakerTurn], frame: float = 0.01) -> dict:
    """Diarization Error Rate = (missed + false_alarm + confusion) / reference_speech."""
    max_end = max([t.end for t in ref] + [t.end for t in hyp] + [0.0])
    n = int(np.ceil(max_end / frame)) if max_end > 0 else 0
    if n == 0:
        return {"der": 0.0, "missed": 0.0, "false_alarm": 0.0, "confusion": 0.0}

    r = _frame_labels(ref, n, frame)
    h = _frame_labels(hyp, n, frame)

    ref_speakers = sorted({x for x in r if x is not None})
    hyp_speakers = sorted({x for x in h if x is not None})

    # optimal hyp->ref label mapping by co-occurrence
    mapping: dict[str, str] = {}
    if ref_speakers and hyp_speakers:
        cost = np.zeros((len(hyp_speakers), len(ref_speakers)))
        ri = {s: i for i, s in enumerate(ref_speakers)}
        hi = {s: i for i, s in enumerate(hyp_speakers)}
        for rf, hf in zip(r, h, strict=True):
            if rf is not None and hf is not None:
                cost[hi[hf], ri[rf]] -= 1  # maximize overlap
        rows, cols = linear_sum_assignment(cost)
        mapping = {hyp_speakers[row]: ref_speakers[col] for row, col in zip(rows, cols, strict=True)}

    ref_speech = sum(1 for x in r if x is not None)
    missed = fa = conf = 0
    for rf, hf in zip(r, h, strict=True):
        if rf is not None and hf is None:
            missed += 1
        elif rf is None and hf is not None:
            fa += 1
        elif rf is not None and hf is not None and mapping.get(hf) != rf:
            conf += 1

    denom = ref_speech if ref_speech > 0 else 1
    return {
        "der": round((missed + fa + conf) / denom, 4),
        "missed": round(missed / denom, 4),
        "false_alarm": round(fa / denom, 4),
        "confusion": round(conf / denom, 4),
    }


# ------------------------------------------------------------ separation
def _align(*arrs: np.ndarray) -> list[np.ndarray]:
    m = min(len(a) for a in arrs)
    return [a[:m].astype(np.float64) for a in arrs]


def si_snr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    est, ref = _align(estimate, reference)
    est = est - est.mean()
    ref = ref - ref.mean()
    s = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    noise = est - s
    return float(10 * np.log10((np.dot(s, s) + eps) / (np.dot(noise, noise) + eps)))


def si_snri(estimate: np.ndarray, reference: np.ndarray, mixture: np.ndarray) -> float:
    """Scale-invariant SNR improvement: SI-SNR(est) - SI-SNR(mixture)."""
    return round(si_snr(estimate, reference) - si_snr(mixture, reference), 4)


def sdri(estimate: np.ndarray, reference: np.ndarray, mixture: np.ndarray, eps: float = 1e-8) -> float:
    """SDR improvement. The estimate is least-squares rescaled to the reference first, so the
    metric is scale-invariant — essential because neural separators emit arbitrary scale
    (otherwise a correctly-separated-but-loud source scores hugely negative)."""
    def sdr(e: np.ndarray, r: np.ndarray) -> float:
        e, r = _align(e, r)
        alpha = np.dot(e, r) / (np.dot(e, e) + eps)   # optimal scale of e onto r
        e = alpha * e
        return float(10 * np.log10((np.dot(r, r) + eps) / (np.dot(r - e, r - e) + eps)))
    return round(sdr(estimate, reference) - sdr(mixture, reference), 4)


# -------------------------------------------------------------------- ASR
def wer(reference: str, hypothesis: str, lang: str | None = None) -> float:
    import jiwer
    ref, hyp = normalize_text(reference, lang), normalize_text(hypothesis, lang)
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(float(jiwer.wer(ref, hyp)), 4)


def cer(reference: str, hypothesis: str, lang: str | None = None) -> float:
    import jiwer
    ref, hyp = normalize_text(reference, lang), normalize_text(hypothesis, lang)
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(float(jiwer.cer(ref, hyp)), 4)
