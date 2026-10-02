"""Stage 5 — LLM post-processing (STUB for the pre-midsem vertical slice).

Emits a structurally valid Report (executive summary + keyword tags + speaker-separated
dialogue) built directly from the transcript — no external calls. The real Gemini backend
(and optional local Ollama) lands post-midsem behind this interface, with the no-new-facts
guardrail + factual-consistency check. Degrades to exactly this minimal report if the LLM
is unavailable at runtime.
"""
from __future__ import annotations

from ..types import Report, Transcript


class PostProcessorStub:
    name = "postprocessing"
    version = "stub-0.1"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self) -> None:
        pass

    def run(self, transcript: Transcript, notes: list[str] | None = None) -> Report:
        notes = [] if notes is None else notes
        notes.append(f"LLM post-processing is a stub (post-midsem); backend '{self.cfg.backend}' not called")
        speakers = sorted({s.speaker for s in transcript.segments})
        summary = (
            f"[stub report] {len(transcript.segments)} speaker turns across "
            f"{len(speakers)} speaker(s). Real LLM summary pending."
        )
        return Report(summary=summary, tags=[], dialogue=list(transcript.segments),
                      raw_transcript=transcript)
