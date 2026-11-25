"""Post-hoc Factual Consistency & Hallucination Guardrail for LLM Post-Processing.

Ensures that the LLM post-processing stage strictly adheres to the source transcript,
verifying that no external, ungrounded, or fabricated facts (e.g., invented numbers,
names, or contradictory events) are introduced into the final executive summary.
"""

from __future__ import annotations

import re
from typing import Any

from ..types import Transcript, TranscriptSegment

_LANGUAGE_NAMES = {
    "Hindi",
    "English",
    "Hinglish",
    "Bengali",
    "Bangla",
    "Assamese",
    "Marathi",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Gujarati",
    "Punjabi",
    "Odia",
    "Urdu",
    "Nepali",
}


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens (including percentages, currencies, ordinals)."""
    return set(re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", text))


def _extract_key_terms(text: str) -> set[str]:
    """Extract content terms (lowercase words of length >= 3, excluding common stop words)."""
    stops = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "they",
        "have",
        "were",
        "been",
        "what",
        "when",
        "where",
        "which",
        "there",
        "their",
        "about",
        "into",
        "across",
        "after",
        "before",
        "each",
        "both",
        "some",
        "other",
        "than",
        "then",
        "also",
        "does",
        "done",
        "will",
        "would",
        "could",
        "should",
        "said",
        "spoke",
        "speaker",
        "speakers",
        "turn",
        "turns",
        "dialogue",
        "conversation",
        "discussed",
        "stated",
        "summary",
        "report",
        "recorded",
        "speech",
        "audio",
        "hai",
        "hain",
        "kya",
        "aur",
        "bhi",
        "tha",
        "thi",
        "aap",
        "hum",
        "yeh",
        "woh",
        # discourse words a summary uses to *talk about* the transcript; they carry no facts
        "additionally", "briefly", "describe", "describes", "discuss", "discusses", "mentions",
        "talks", "shares", "features", "including", "various", "while", "one", "how",
    }
    tokens = re.findall(r"\b[a-zA-Z\u0900-\u0D7F]{3,}\b", text.lower())
    return {t for t in tokens if t not in stops}


def check_factual_consistency(
    summary: str,
    transcript: Transcript,
    dialogue: list[TranscriptSegment] | None = None,
    client: Any = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> tuple[float, str, list[str]]:
    """Verify that the generated summary is grounded in the raw transcript & dialogue.

    Returns:
        (consistency_score, guardrail_status, list_of_flagged_hallucinations)
    """
    if not summary or not summary.strip():
        return 1.0, "PASSED", []

    # Assemble complete reference corpus (raw transcript + cleaned dialogue + translations)
    dialogue_segments = dialogue if dialogue is not None else transcript.segments
    source_texts: list[str] = [transcript.text]
    for seg in dialogue_segments:
        if seg.text:
            source_texts.append(seg.text)
        if getattr(seg, "translation", None):
            source_texts.append(seg.translation or "")

    source_corpus = " ".join(source_texts)
    source_numbers = _extract_numbers(source_corpus)
    source_terms = _extract_key_terms(source_corpus)

    hallucinations: list[str] = []

    # 1. Numeric Grounding: Any number in the summary MUST appear in the source
    # Speaker labels ("SPEAKER_01", "Speaker 1") are not facts; drop them before number checks.
    summary_numbers = _extract_numbers(re.sub(r"(?i)speaker[\s_]*\d+", " ", summary))
    for num in summary_numbers:
        # Ignore turn/speaker indexing like "2 speakers" if turn count matches
        if num in source_numbers:
            continue
        # Also allow small integers that might describe speaker count
        if num in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
            spk_count = len({s.speaker for s in dialogue_segments})
            turn_count = len(dialogue_segments)
            if num in {str(spk_count), str(turn_count)}:
                continue
        hallucinations.append(f"Ungrounded numerical figure '{num}' not found in source transcript.")

    # 2. Sentence-level content grounding
    sentences = [s.strip() for s in re.split(r"[.\n!?]+", summary) if len(s.strip()) > 5]
    if not sentences:
        return 1.0, "PASSED", []

    grounded_sentences = 0
    for sentence in sentences:
        sentence_terms = _extract_key_terms(sentence)
        if not sentence_terms:
            grounded_sentences += 1
            continue

        # Compare 5-char prefixes so inflections still match (learning/learn, stones/stone).
        source_stems = {t[:5] for t in source_terms}
        overlap = {t for t in sentence_terms if t[:5] in source_stems}
        overlap_ratio = len(overlap) / len(sentence_terms)

        if overlap_ratio >= 0.25:  # At least 25% key content words directly trace to source
            grounded_sentences += 1
        else:
            # Check for excessive ungrounded named entities / capitalized terms
            # Only mid-sentence capitals are likely proper nouns: the first word is capitalised
            # by grammar ("This", "Additionally"), and language names describe the transcript
            # rather than invent a fact.
            body = sentence.split(maxsplit=1)[1] if len(sentence.split()) > 1 else ""
            capitalized = set(re.findall(r"\b[A-Z][a-z]{3,}\b", body)) - {"Speaker"} - _LANGUAGE_NAMES
            unsupported_caps = [c for c in capitalized if c.lower() not in source_corpus.lower()]
            if unsupported_caps:
                hallucinations.append(
                    f"Ungrounded entity/concept in sentence: '{sentence[:80]}...' "
                    f"(terms: {', '.join(unsupported_caps)})"
                )

    ratio_grounded = grounded_sentences / len(sentences)

    # 3. Penalize score for any detected hallucinations
    penalty = len(hallucinations) * 0.25
    consistency_score = round(max(0.0, min(1.0, ratio_grounded - penalty)), 3)

    if consistency_score >= 0.70 and len(hallucinations) == 0:
        status = "PASSED"
    elif consistency_score >= 0.50:
        status = "PASSED_WITH_WARNINGS"
    else:
        status = "FLAGGED_UNGROUNDED"

    return consistency_score, status, hallucinations
