"""Stage 5 — Gemini Multilingual Post-Processing & Intelligence.

Performs multilingual language detection, code-switching identification,
English translation, phoneme error normalization, and structured executive summary
using Google Gemini API (configured through ``postprocessing.model``).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from ..config import load_env
from ..types import Report, Transcript, TranscriptSegment
from .guardrails import check_factual_consistency


def _heuristic_detect_language(text: str) -> tuple[str, bool]:
    """Offline heuristic language and code-switch detection based on Unicode ranges and script mixing."""
    if not text or text.startswith("<"):
        return "Unknown", False

    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_bengali = bool(re.search(r"[\u0980-\u09FF]", text))
    has_tamil = bool(re.search(r"[\u0B80-\u0BFF]", text))
    has_telugu = bool(re.search(r"[\u0C00-\u0C7F]", text))
    has_kannada = bool(re.search(r"[\u0C80-\u0CFF]", text))
    has_malayalam = bool(re.search(r"[\u0D00-\u0D7F]", text))
    has_gujarati = bool(re.search(r"[\u0A80-\u0AFF]", text))
    has_punjabi = bool(re.search(r"[\u0A00-\u0A7F]", text))
    has_latin = bool(re.search(r"[a-zA-Z]{2,}", text))

    indic_scripts = sum([
        has_devanagari, has_bengali, has_tamil, has_telugu,
        has_kannada, has_malayalam, has_gujarati, has_punjabi
    ])

    if has_devanagari:
        if has_latin:
            return "Hinglish (Hindi + English)", True
        return "Hindi", False
    if has_bengali:
        if has_latin:
            return "Benglish (Bengali + English)", True
        return "Bengali", False
    if has_tamil:
        if has_latin:
            return "Tanglish (Tamil + English)", True
        return "Tamil", False
    if has_telugu:
        if has_latin:
            return "Telugu + English", True
        return "Telugu", False
    if has_kannada:
        return "Kannada" + (" + English" if has_latin else ""), has_latin
    if has_malayalam:
        return "Malayalam" + (" + English" if has_latin else ""), has_latin
    if has_gujarati:
        return "Gujarati" + (" + English" if has_latin else ""), has_latin
    if has_punjabi:
        return "Punjabi" + (" + English" if has_latin else ""), has_latin

    if has_latin:
        # Check for common code-switched Hindi/Urdu keywords in Latin script
        hinglish_words = {"haan", "nahin", "nahi", "kya", "hai", "hain", "acha", "achha", "theek",
                          "aur", "bhi", "lekin", "mera", "meri", "aap", "tum", "hum", "kaise", "karo"}
        words = {w.lower() for w in re.findall(r"[a-zA-Z]+", text)}
        if len(words.intersection(hinglish_words)) >= 2:
            return "Hinglish (Romanized)", True
        return "English", False

    return "Regional", indic_scripts > 1


class GeminiPostProcessor:
    name = "postprocessing"
    version = "gemini"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._client: Any = None
        self._load_error: str | None = None
        self.model_name: str = getattr(cfg, "model", "gemini-3.8-flash")

    def load(self) -> None:
        load_env()
        env_var = getattr(self.cfg, "api_key_env", "GEMINI_API_KEY")
        key = os.environ.get(env_var)
        if not key:
            self._load_error = f"{env_var} is not set in environment"
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
        except Exception as exc:
            self._load_error = f"could not initialize Gemini: {type(exc).__name__}: {exc}"

    def run(self, transcript: Transcript, notes: list[str] | None = None) -> Report:
        notes = [] if notes is None else notes

        if not transcript.segments:
            return Report(
                summary="No transcript segments available to process.",
                tags=[],
                dialogue=[],
                raw_transcript=transcript,
                languages=[],
                language_breakdown={},
            )

        # Apply offline heuristic detection first as baseline
        for seg in transcript.segments:
            if not seg.lang or seg.lang.lower() in {"unknown", "none"}:
                heuristic_lang, is_cs = _heuristic_detect_language(seg.text)
                seg.lang = heuristic_lang
                seg.is_code_switched = is_cs

        if self._client is None:
            err = self._load_error or "client not initialized"
            notes.append(f"Gemini LLM unavailable ({err}); using heuristic language tagging")
            return self._build_fallback_report(transcript)

        # Call Gemini API for deep multilingual understanding and synthesis
        try:
            return self._call_gemini(transcript, notes)
        except Exception as exc:
            notes.append(
                f"Gemini API call failed ({type(exc).__name__}: {exc}); falling back to heuristic tags"
            )
            return self._build_fallback_report(transcript)

    def _call_gemini(self, transcript: Transcript, notes: list[str]) -> Report:
        from google.genai import types

        turns_payload = [
            {
                "index": i,
                "speaker": seg.speaker,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "raw_text": seg.text,
                "preliminary_lang": seg.lang or "unknown",
            }
            for i, seg in enumerate(transcript.segments)
        ]

        prompt = f"""You are an advanced computational speech linguist specializing in Indian speech,
regional dialects, and code-switching (e.g. Hinglish, Benglish, Tanglish).
Below is a multi-speaker transcript recorded from authentic, chaotic conversational Indian audio:

{json.dumps(turns_payload, ensure_ascii=False, indent=2)}

Perform the following tasks:
1. Multi-Language & Code-Switching Identification:
   - Identify language(s) spoken (e.g., "Hindi", "English", "Hinglish (Hindi + English)", "Bengali", "Tamil").
   - Set `is_code_switched` to true if the speaker transitions between English and an Indian language.
2. Transcription Cleaning & Repair:
   - Correct phoneme dropouts, syntax fragments, and noise glitches while preserving authentic scripts.
3. English Translation:
   - Provide a natural, fluent English translation for every segment with non-English or code-switched speech.
4. Executive Summary:
   - Provide a concise 2-4 sentence executive summary detailing who spoke what and the core narrative.
5. Topic Tags:
   - Extract 3-6 relevant tags (e.g., ["Community Radio", "Market Dispute", "Weather Alert", "Agriculture"]).
6. Language Breakdown:
   - Estimate dialogue percentage for each language (e.g., {{"Hindi": "60%", "English": "40%"}}).
7. Guardrail:
   - Rely strictly on spoken words; do not invent or extrapolate unmentioned facts.

Return strictly valid JSON conforming to this schema:
{{
  "summary": "string",
  "tags": ["string"],
  "languages_detected": ["string"],
  "language_breakdown": {{"Language": "percentage"}},
  "segments": [
    {{
      "index": 0,
      "speaker": "SPEAKER_00",
      "cleaned_text": "string",
      "lang": "string",
      "is_code_switched": true,
      "translation": "string"
    }}
  ]
}}"""

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                http_options=types.HttpOptions(timeout=25000),
            ),
        )

        resp_text = response.text or "{}"
        data = json.loads(resp_text)

        summary = data.get("summary", "")
        tags = data.get("tags", [])
        languages = data.get("languages_detected", [])
        language_breakdown = data.get("language_breakdown", {})
        enriched_segments = data.get("segments", [])

        # Map back to transcript segments
        idx_map = {item.get("index", idx): item for idx, item in enumerate(enriched_segments)}
        dialogue: list[TranscriptSegment] = []

        for i, orig_seg in enumerate(transcript.segments):
            enrichment = idx_map.get(i)
            if enrichment:
                cleaned_text = enrichment.get("cleaned_text") or orig_seg.text
                lang = enrichment.get("lang") or orig_seg.lang or "Unknown"
                is_cs = bool(enrichment.get("is_code_switched", orig_seg.is_code_switched))
                translation = enrichment.get("translation")
                updated_seg = TranscriptSegment(
                    start=orig_seg.start,
                    end=orig_seg.end,
                    speaker=orig_seg.speaker,
                    text=cleaned_text,
                    words=orig_seg.words,
                    lang=lang,
                    translation=translation,
                    is_code_switched=is_cs,
                )
            else:
                updated_seg = orig_seg
            dialogue.append(updated_seg)

        # Apply Factual Consistency Guardrail check
        consistency_score, status, hallucinations = check_factual_consistency(
            summary, transcript, dialogue
        )
        if status != "PASSED":
            notes.append(f"Guardrail check: {status} (score={consistency_score})")

        lang_str = ", ".join(languages) or "detected"
        notes.append(
            f"Gemini ({self.model_name}) analyzed multilingual transcript: "
            f"{len(dialogue)} turns, languages: {lang_str}"
        )

        return Report(
            summary=summary,
            tags=tags,
            dialogue=dialogue,
            raw_transcript=transcript,
            languages=languages,
            language_breakdown=language_breakdown,
            consistency_score=consistency_score,
            guardrail_status=status,
            hallucinations_detected=hallucinations,
        )

    def _build_fallback_report(self, transcript: Transcript) -> Report:
        speakers = sorted({s.speaker for s in transcript.segments})
        langs = sorted({s.lang for s in transcript.segments if s.lang and s.lang != "Unknown"})
        summary = (
            f"Conversation with {len(transcript.segments)} turns across {len(speakers)} speaker(s) "
            f"({', '.join(speakers)}). Languages identified: {', '.join(langs) or 'mixed regional'}."
        )
        return Report(
            summary=summary,
            tags=["Spontaneous Speech", "Multi-Speaker"] + ([l for l in langs if " " not in l]),
            dialogue=list(transcript.segments),
            raw_transcript=transcript,
            languages=langs,
            language_breakdown={l: f"{round(100 / max(1, len(langs)))}%" for l in langs},
            consistency_score=1.0,
            guardrail_status="PASSED",
            hallucinations_detected=[],
        )
