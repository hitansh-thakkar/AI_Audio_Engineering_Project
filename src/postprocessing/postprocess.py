"""Stage 5 — Semantic Post-Processing & Structuring (LLM Guardrailing).

Orchestrates post-processing backends (Google Gemini API, local Ollama, or stub fallback)
to clean noise-induced dropouts, translate regional/code-switched speech, and produce
a structured executive summary report with factual consistency guardrailing.
"""
from __future__ import annotations

from typing import Any

from ..types import Report, Transcript


class PostProcessorStub:
    name = "postprocessing"
    version = "stub-0.1"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def load(self) -> None:
        pass

    def run(self, transcript: Transcript, notes: list[str] | None = None) -> Report:
        notes = [] if notes is None else notes
        backend = getattr(self.cfg, "backend", "stub")
        notes.append(f"LLM post-processing is a stub; backend '{backend}' not called")
        speakers = sorted({s.speaker for s in transcript.segments})
        summary = (
            f"[stub report] {len(transcript.segments)} speaker turns across "
            f"{len(speakers)} speaker(s). Real LLM summary pending."
        )
        return Report(
            summary=summary,
            tags=["Stub Report"],
            dialogue=list(transcript.segments),
            raw_transcript=transcript,
            languages=[],
            language_breakdown={},
            consistency_score=1.0,
            guardrail_status="PASSED",
            hallucinations_detected=[],
        )


def create_postprocessor(cfg: Any):
    """Factory creating the appropriate Stage 5 post-processor based on configuration."""
    backend = str(getattr(cfg, "backend", "gemini")).lower()
    if backend == "gemini":
        from .gemini import GeminiPostProcessor
        return GeminiPostProcessor(cfg)
    elif backend == "ollama":
        from .ollama import OllamaPostProcessor
        return OllamaPostProcessor(cfg)
    else:
        return PostProcessorStub(cfg)
