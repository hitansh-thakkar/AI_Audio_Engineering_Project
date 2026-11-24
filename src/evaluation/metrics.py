"""Metric implementations: DER/JER (diarization), SI-SNRi / SDRi (separation), WER/CER (ASR),
ROUGE-L / factual consistency (LLM stage).

DER uses pyannote.metrics (the reference implementation of NIST md-eval semantics): optimal
speaker mapping, overlap scored (skip_overlap=False), collar 0 by default. The earlier
hand-rolled frame scorer kept only the *first* speaker per frame, which silently ignored
overlapped speech — exactly the regions this project is about.

Corpus-level WER/CER must be *pooled* (sum of edits / sum of reference units), not the mean of
per-clip ratios: a 2-word clip with 1 error would otherwise weigh as much as a 40-word clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import SpeakerTurn
from .normalize import normalize_text


# --------------------------------------------------------------------- DER
def _annotation(turns: list[SpeakerTurn]):  # -> pyannote.core.Annotation (lazy import)
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for i, t in enumerate(turns):
        if t.end > t.start:
            ann[Segment(float(t.start), float(t.end)), i] = t.speaker
    return ann


def der(
    ref: list[SpeakerTurn], hyp: list[SpeakerTurn], collar: float = 0.0, skip_overlap: bool = False
) -> dict:
    """Diarization Error Rate = (missed + false_alarm + confusion) / total reference speech.

    Components are returned as fractions of reference speech so they sum to `der`. Reference
    speech counts overlapped regions once *per speaker* (md-eval convention).
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    comp = metric(_annotation(ref), _annotation(hyp), detailed=True)
    total = comp["total"]
    if total <= 0:
        return {
            "der": 0.0 if not hyp else 1.0,
            "missed": 0.0,
            "false_alarm": 0.0,
            "confusion": 0.0,
            "total": 0.0,
        }
    return {
        "der": round(comp["diarization error rate"], 4),
        "missed": round(comp["missed detection"] / total, 4),
        "false_alarm": round(comp["false alarm"] / total, 4),
        "confusion": round(comp["confusion"] / total, 4),
        "total": round(total, 4),
        # absolute seconds, so corpus-level DER can be pooled the same way as WER
        "missed_s": comp["missed detection"],
        "false_alarm_s": comp["false alarm"],
        "confusion_s": comp["confusion"],
    }


def jer(ref: list[SpeakerTurn], hyp: list[SpeakerTurn], collar: float = 0.0) -> float:
    from pyannote.metrics.diarization import JaccardErrorRate

    return round(float(JaccardErrorRate(collar=collar)(_annotation(ref), _annotation(hyp))), 4)


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
        alpha = np.dot(e, r) / (np.dot(e, e) + eps)  # optimal scale of e onto r
        e = alpha * e
        return float(10 * np.log10((np.dot(r, r) + eps) / (np.dot(r - e, r - e) + eps)))

    return round(sdr(estimate, reference) - sdr(mixture, reference), 4)


def pit_si_snri(
    estimates: list[np.ndarray], references: list[np.ndarray], mixture: np.ndarray
) -> tuple[list[float], tuple[int, ...]]:
    """Permutation-invariant SI-SNRi: best assignment of estimates to references.

    This measures the *separator's* quality independent of the pipeline's speaker-assignment
    step (which is scored separately). Brute-force over permutations — fine for 2-3 sources.
    """
    from itertools import permutations

    best: tuple[float, tuple[int, ...], list[float]] = (-np.inf, (), [])
    for perm in permutations(range(len(estimates)), len(references)):
        vals = [si_snri(estimates[p], references[i], mixture) for i, p in enumerate(perm)]
        if float(np.mean(vals)) > best[0]:
            best = (float(np.mean(vals)), perm, vals)
    return best[2], best[1]


# -------------------------------------------------------------------- ASR
@dataclass(frozen=True)
class EditCounts:
    """Raw edit counts so corpus-level rates can be pooled correctly."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    ref_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.ref_units == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_units


def word_edits(
    reference: str, hypothesis: str, lang: str | None = None, *, fold_nukta: bool = False
) -> EditCounts:
    import jiwer

    ref = normalize_text(reference, lang, fold_nukta=fold_nukta)
    hyp = normalize_text(hypothesis, lang, fold_nukta=fold_nukta)
    if not ref:
        n = len(hyp.split())
        return EditCounts(0, 0, 0, n, 0)
    if not hyp:
        n = len(ref.split())
        return EditCounts(0, 0, n, 0, n)
    o = jiwer.process_words(ref, hyp)
    return EditCounts(
        o.hits, o.substitutions, o.deletions, o.insertions, o.hits + o.substitutions + o.deletions
    )


def char_edits(
    reference: str, hypothesis: str, lang: str | None = None, *, fold_nukta: bool = False
) -> EditCounts:
    """Character edits over *whitespace-free* text: CER then measures spelling, not word
    segmentation (which WER already captures). Matters for agglutinative ta/ml/te/kn, where
    compound splitting varies between transcription conventions."""
    import jiwer

    ref = normalize_text(reference, lang, fold_nukta=fold_nukta).replace(" ", "")
    hyp = normalize_text(hypothesis, lang, fold_nukta=fold_nukta).replace(" ", "")
    if not ref:
        return EditCounts(0, 0, 0, len(hyp), 0)
    if not hyp:
        return EditCounts(0, 0, len(ref), 0, len(ref))
    # jiwer.process_characters treats each string as a sequence of characters.
    o = jiwer.process_characters(ref, hyp)
    return EditCounts(
        o.hits, o.substitutions, o.deletions, o.insertions, o.hits + o.substitutions + o.deletions
    )


def pooled(counts: list[EditCounts]) -> EditCounts:
    return EditCounts(
        *(
            sum(getattr(c, f) for c in counts)
            for f in ("hits", "substitutions", "deletions", "insertions", "ref_units")
        )
    )


def wer(reference: str, hypothesis: str, lang: str | None = None) -> float:
    return round(word_edits(reference, hypothesis, lang).rate, 4)


def cer(reference: str, hypothesis: str, lang: str | None = None) -> float:
    return round(char_edits(reference, hypothesis, lang).rate, 4)


# -------------------------------------------------------- Summary / LLM
def _ws_tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


class _UnicodeTokenizer:
    """rouge_score's default tokenizer keeps only [a-z0-9] — it deletes every Indic character,
    so ROUGE on Devanagari/Bengali/... text is silently computed on empty strings. Use the
    project normalizer + whitespace split instead."""

    def tokenize(self, text: str) -> list[str]:
        return _ws_tokenize(text)


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 between reference and generated text (Unicode-safe tokenization)."""
    if not reference or not hypothesis:
        return 0.0
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=_UnicodeTokenizer())
        return round(float(scorer.score(reference, hypothesis)["rougeL"].fmeasure), 4)
    except ImportError:
        pass
    ref_tok, hyp_tok = _ws_tokenize(reference), _ws_tokenize(hypothesis)
    m, n = len(ref_tok), len(hyp_tok)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if ref_tok[i] == hyp_tok[j] else max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    prec, rec = lcs / n, lcs / m
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0


def factual_consistency(summary: str, transcript: str) -> float:
    """Measure factual grounding rate of summary against source transcript."""
    from ..postprocessing.guardrails import check_factual_consistency
    from ..types import Transcript, TranscriptSegment

    t = Transcript(segments=[TranscriptSegment(0.0, 1.0, "SPEAKER_00", transcript)])
    score, _, _ = check_factual_consistency(summary, t)
    return score
