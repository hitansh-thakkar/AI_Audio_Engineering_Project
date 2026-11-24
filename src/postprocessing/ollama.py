"""Local LLM Post-Processing backend using Ollama (e.g. Qwen2.5, Llama-3.2).

Runs fully offline on local hardware or server instance via http://localhost:11434.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..types import Report, Transcript, TranscriptSegment
from .gemini import _heuristic_detect_language
from .guardrails import check_factual_consistency


class OllamaPostProcessor:
    name = "postprocessing"
    version = "ollama"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.host = getattr(cfg, "host", "http://localhost:11434").rstrip("/")
        self.model_name = getattr(cfg, "model", "qwen2.5:7b")
        self._load_error: str | None = None
        self._available: bool = False

    def load(self) -> None:
        # Probe Ollama API endpoint to check availability
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    self._available = True
        except Exception as exc:
            self._load_error = f"Ollama service unavailable at {self.host} ({type(exc).__name__}: {exc})"

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

        # Baseline heuristic language detection
        for seg in transcript.segments:
            if not seg.lang or seg.lang.lower() in {"unknown", "none"}:
                heuristic_lang, is_cs = _heuristic_detect_language(seg.text)
                seg.lang = heuristic_lang
                seg.is_code_switched = is_cs

        if not self._available:
            notes.append(
                f"Ollama local LLM unavailable ({self._load_error}); using heuristic language tagging"
            )
            return self._build_fallback_report(transcript)

        try:
            return self._call_ollama(transcript, notes)
        except Exception as exc:
            notes.append(
                f"Ollama generation failed ({type(exc).__name__}: {exc}); falling back to heuristic tags"
            )
            return self._build_fallback_report(transcript)

    def _call_ollama(self, transcript: Transcript, notes: list[str]) -> Report:
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

        req_body = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # 7B Q4 models spill to CPU on 4 GB GPUs; a report can take >60 s there.
        timeout = float(getattr(self.cfg, "timeout_s", 60.0))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            data = json.loads(res_data.get("response", "{}"))

        summary = data.get("summary", "")
        tags = data.get("tags", [])
        languages = data.get("languages_detected", [])
        language_breakdown = data.get("language_breakdown", {})
        enriched_segments = data.get("segments", [])

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

        notes.append(
            f"Ollama ({self.model_name}) analyzed multilingual transcript: "
            f"{len(dialogue)} turns, languages: {', '.join(languages) or 'detected'}"
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

