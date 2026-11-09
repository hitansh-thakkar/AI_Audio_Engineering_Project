"""Milestone 4 — Semantic Post-Processing & Guardrailing tests.

Covers:
  - Factual consistency guardrail (pass / fail on invented numbers)
  - ROUGE-L metric computation
  - Factual consistency evaluation metric
  - PostProcessorStub has all three guardrail fields
  - create_postprocessor factory returns correct class per backend
  - OllamaPostProcessor falls back gracefully (offline)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.postprocessing.guardrails import check_factual_consistency
from src.postprocessing.postprocess import PostProcessorStub, create_postprocessor
from src.types import Transcript, TranscriptSegment


# ─── helpers ────────────────────────────────────────────────────────────────

def _seg(text: str, speaker: str = "SPEAKER_00", start: float = 0.0, end: float = 5.0) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, speaker=speaker, text=text)


def _transcript(*texts: str) -> Transcript:
    segs = [_seg(t, start=i * 5.0, end=(i + 1) * 5.0) for i, t in enumerate(texts)]
    return Transcript(segments=segs)


# ─── Guardrail: grounded summary passes ─────────────────────────────────────

def test_guardrail_passes_grounded_summary() -> None:
    """A summary that only uses words/numbers present in the transcript should PASS."""
    transcript = _transcript(
        "The mango price rose to 50 rupees per kilogram in the market today.",
        "Farmers in the village discussed the drought affecting crop yield.",
    )
    summary = "Market prices for mango reached 50 rupees. Farmers discussed drought and crop issues."
    score, status, hallucinations = check_factual_consistency(summary, transcript)

    assert status == "PASSED", f"Expected PASSED but got {status!r}; hallucinations={hallucinations}"
    assert score >= 0.70, f"Score {score} should be >= 0.70 for grounded summary"
    assert hallucinations == [], f"Expected no hallucinations, got {hallucinations}"


# ─── Guardrail: invented number is flagged ───────────────────────────────────

def test_guardrail_flags_invented_numbers() -> None:
    """A number that appears in the summary but NOT the transcript should be flagged."""
    transcript = _transcript(
        "Rainfall was low this season.",
        "Wheat production was good.",
    )
    # 42 appears nowhere in the transcript
    summary = "According to the report, 42 farmers attended the meeting and production improved."
    score, status, hallucinations = check_factual_consistency(summary, transcript)

    # At least one hallucination flagged for '42'
    assert any("42" in h for h in hallucinations), (
        f"Expected '42' to be flagged as hallucination, got: {hallucinations}"
    )
    assert status in {"PASSED_WITH_WARNINGS", "FLAGGED_UNGROUNDED"}, (
        f"Status should not be PASSED when ungrounded number present; got {status!r}"
    )


# ─── Guardrail: empty summary always passes ──────────────────────────────────

def test_guardrail_empty_summary_passes() -> None:
    """An empty or whitespace-only summary trivially passes."""
    transcript = _transcript("Some speech content here.")
    score, status, hallucinations = check_factual_consistency("", transcript)
    assert status == "PASSED"
    assert score == 1.0
    assert hallucinations == []


# ─── ROUGE-L metric ─────────────────────────────────────────────────────────

def test_rouge_l_computes_correctly() -> None:
    """ROUGE-L F1 for identical strings is 1.0; for unrelated strings is very low."""
    from src.evaluation.metrics import rouge_l

    perfect = rouge_l("the cat sat on the mat", "the cat sat on the mat")
    assert perfect == pytest.approx(1.0, abs=1e-6), f"Perfect match should be 1.0, got {perfect}"

    no_overlap = rouge_l("alpha beta gamma delta", "one two three four five")
    assert no_overlap < 0.1, f"Zero-overlap ROUGE-L should be near 0, got {no_overlap}"

    partial = rouge_l("the cat sat on the mat", "the cat sat on a carpet")
    assert 0.3 < partial < 1.0, f"Partial overlap should be between 0.3 and 1.0, got {partial}"


# ─── Factual consistency evaluation metric ───────────────────────────────────

def test_factual_consistency_metric() -> None:
    """src.evaluation.metrics.factual_consistency wraps the guardrails module."""
    from src.evaluation.metrics import factual_consistency

    score = factual_consistency(
        "Market prices rose this season.",
        "Market prices rose this season due to drought.",
    )
    # Grounded summary → score near 1.0
    assert 0.5 <= score <= 1.0, f"Expected high score for grounded text, got {score}"


# ─── PostProcessorStub has all guardrail fields ──────────────────────────────

def test_postprocessor_stub_has_guardrail_fields() -> None:
    """PostProcessorStub.run() must return a Report with all three guardrail fields."""
    cfg = SimpleNamespace(backend="stub")
    stub = PostProcessorStub(cfg)
    stub.load()
    transcript = _transcript("hello world", "testing one two three")
    report = stub.run(transcript)

    assert hasattr(report, "consistency_score"), "Report missing consistency_score"
    assert hasattr(report, "guardrail_status"), "Report missing guardrail_status"
    assert hasattr(report, "hallucinations_detected"), "Report missing hallucinations_detected"

    assert isinstance(report.consistency_score, float), "consistency_score must be float"
    assert report.guardrail_status in {"PASSED", "PASSED_WITH_WARNINGS", "FLAGGED_UNGROUNDED"}, (
        f"guardrail_status has unexpected value: {report.guardrail_status!r}"
    )
    assert isinstance(report.hallucinations_detected, list), "hallucinations_detected must be a list"


# ─── create_postprocessor factory ───────────────────────────────────────────

def test_create_postprocessor_factory_stub() -> None:
    """create_postprocessor returns PostProcessorStub for backend='stub'."""
    cfg = SimpleNamespace(backend="stub")
    post = create_postprocessor(cfg)
    assert isinstance(post, PostProcessorStub), (
        f"Expected PostProcessorStub, got {type(post).__name__}"
    )


def test_create_postprocessor_factory_gemini() -> None:
    """create_postprocessor returns GeminiPostProcessor for backend='gemini'."""
    from src.postprocessing.gemini import GeminiPostProcessor

    cfg = SimpleNamespace(backend="gemini", env_var="NON_EXISTENT_API_KEY_XYZ")
    post = create_postprocessor(cfg)
    assert isinstance(post, GeminiPostProcessor), (
        f"Expected GeminiPostProcessor, got {type(post).__name__}"
    )


def test_create_postprocessor_factory_ollama() -> None:
    """create_postprocessor returns OllamaPostProcessor for backend='ollama'."""
    from src.postprocessing.ollama import OllamaPostProcessor

    cfg = SimpleNamespace(backend="ollama", model="llama3", host="http://localhost:11434")
    post = create_postprocessor(cfg)
    assert isinstance(post, OllamaPostProcessor), (
        f"Expected OllamaPostProcessor, got {type(post).__name__}"
    )


# ─── OllamaPostProcessor graceful offline fallback ───────────────────────────

def test_ollama_offline_fallback() -> None:
    """OllamaPostProcessor.load() with an unreachable server must NOT raise."""
    from src.postprocessing.ollama import OllamaPostProcessor

    cfg = SimpleNamespace(
        backend="ollama",
        model="llama3",
        host="http://127.0.0.1:19999",  # almost certainly nothing listening here
    )
    post = OllamaPostProcessor(cfg)
    # load() should catch connection errors and mark itself as unavailable
    post.load()   # must not raise

    # run() should degrade gracefully to a stub-style report
    transcript = _transcript("test fallback speech")
    notes: list[str] = []
    report = post.run(transcript, notes)

    assert report.summary != "", "Should produce some summary even in fallback mode"
    assert isinstance(report.hallucinations_detected, list)
    assert any("ollama" in n.lower() or "unavailable" in n.lower() for n in notes), (
        f"Expected an 'ollama unavailable' note; got notes={notes}"
    )
