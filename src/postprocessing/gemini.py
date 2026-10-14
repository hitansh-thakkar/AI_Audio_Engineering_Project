"""Stage 5 — LLM post-processing via the Google Gemini API (free tier, default).

Turns the fragmented, code-switched, speaker-attributed transcript into the proposal's
report: executive summary + keyword tags + cleaned speaker-separated dialogue.

Guardrails (golden rule #5 — do not fabricate):
- System instruction forbids introducing any fact not present in the transcript; the model
  may only clean/normalize/reorganize per-segment text and summarize.
- Per-segment cleaning preserves the original speaker + timings (the LLM returns cleaned
  text keyed by segment index), so it cannot invent turns or move speech around.
- A zero-weight factual-consistency check flags summary tokens absent from the transcript.

Degrades to the minimal stub report (never crashes) if the key/package/call is unavailable.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any

from ..config import load_env
from ..evaluation.normalize import normalize_text
from ..types import Report, Transcript

SYSTEM_INSTRUCTION = (
    "You are a transcript editor for chaotic, code-switched (Hindi/English/regional) Indian "
    "audio. You may ONLY clean, normalize code-switching, and repair obvious noise-induced "
    "errors in the provided transcript, and summarize what is actually said. You MUST NOT "
    "invent names, facts, numbers, topics, or dialogue absent from the transcript. Return "
    "ONLY JSON: {\"summary\": str, \"tags\": [str], \"segments\": [{\"i\": int, \"text\": str}]}. "
    "'summary' is a concise executive summary grounded only in the transcript. 'tags' are 3-8 "
    "keyword topics. 'segments' gives cleaned text for each numbered input segment (same i); "
    "omit an entry to keep the original text."
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_text(text).lower()))


_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "lakh", "crore", "million", "billion",
}


def _numeric_facts(text: str) -> set[str]:
    low = normalize_text(text).lower()
    low = re.sub(r"speaker[_ ]?\d+", " ", low)   # speaker labels are structural, not facts
    nums = set(re.findall(r"\d+", low))
    nums |= {w for w in _NUMBER_WORDS if re.search(rf"\b{w}\b", low)}
    return nums


def factual_consistency(summary: str, transcript_text: str) -> float:
    """Fraction of the summary's numeric facts (prices, rates, counts — the real fabrication
    risk here) that are grounded in the transcript (1.0 = all grounded). A zero-weight,
    language-agnostic proxy for the no-new-facts guardrail: it flags invented numbers without
    penalising legitimate abstractive/cross-lingual rephrasing (the system prompt is the
    primary guard). No model needed."""
    facts = _numeric_facts(summary)
    if not facts:
        return 1.0
    grounded = _numeric_facts(transcript_text)
    return round(len(facts & grounded) / len(facts), 3)


def build_report(parsed: dict, transcript: Transcript, notes: list[str],
                 min_consistency: float = 0.5) -> Report:
    cleaned = {int(e["i"]): str(e.get("text", "")).strip()
               for e in parsed.get("segments", []) if "i" in e}
    dialogue = [
        replace(seg, text=cleaned[i]) if cleaned.get(i) else seg
        for i, seg in enumerate(transcript.segments)
    ]
    summary = str(parsed.get("summary", "")).strip() or "(no summary produced)"
    tags = [str(t).strip() for t in parsed.get("tags", []) if str(t).strip()][:8]
    score = factual_consistency(summary, transcript.text)
    if score < min_consistency:
        notes.append(f"factual-consistency low ({score}); summary may contain ungrounded content")
    else:
        notes.append(f"factual-consistency {score}")
    return Report(summary=summary, tags=tags, dialogue=dialogue, raw_transcript=transcript)


def _minimal_report(transcript: Transcript, notes: list[str], reason: str) -> Report:
    notes.append(f"{reason}; emitting minimal report from the raw transcript")
    speakers = sorted({s.speaker for s in transcript.segments})
    summary = (f"[fallback] {len(transcript.segments)} turns across {len(speakers)} speaker(s); "
               "LLM post-processing unavailable.")
    return Report(summary=summary, tags=[], dialogue=list(transcript.segments),
                  raw_transcript=transcript)


class GeminiPostProcessor:
    name = "postprocessing"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"gemini-{cfg.model}"
        self._client: Any = None
        self._load_error: str | None = None

    def load(self) -> None:
        load_env()
        key = os.environ.get(str(getattr(self.cfg, "api_key_env", "GEMINI_API_KEY")))
        if not key:
            self._load_error = f"{getattr(self.cfg, 'api_key_env', 'GEMINI_API_KEY')} not set"
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
        except Exception as exc:
            self._load_error = f"google-genai unavailable: {type(exc).__name__}: {exc}"

    def run(self, transcript: Transcript, notes: list[str] | None = None) -> Report:
        notes = [] if notes is None else notes
        if not transcript.segments:
            return _minimal_report(transcript, notes, "empty transcript")
        if self._client is None:
            return _minimal_report(transcript, notes, f"Gemini unavailable: {self._load_error}")
        numbered = "\n".join(f"{i} | {s.speaker} | {s.text}"
                             for i, s in enumerate(transcript.segments))
        try:
            from google.genai import types
            resp = self._client.models.generate_content(
                model=str(self.cfg.model),
                contents=f"Transcript segments (index | speaker | text):\n{numbered}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.2),
            )
            parsed = json.loads(resp.text)
        except Exception as exc:
            return _minimal_report(transcript, notes, f"Gemini call failed ({type(exc).__name__}: {exc})")
        notes.append(f"Gemini ({self.cfg.model}) structured the report")
        return build_report(parsed, transcript, notes)
