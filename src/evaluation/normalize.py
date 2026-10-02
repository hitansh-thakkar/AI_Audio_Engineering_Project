"""Script-safe text normalization for WER/CER.

CRITICAL: Whisper's default normalizer strips Devanagari matras and inflates Devanagari WER.
Use NFC (+ indic-nlp-library normalization when installed) instead — every reported ASR
number depends on this. Implemented once, used everywhere.
"""
from __future__ import annotations

import unicodedata

_INDIC = None
try:  # optional; improves Indic normalization when present
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _INDIC = IndicNormalizerFactory()
except Exception:
    _INDIC = None


def normalize_text(text: str, lang: str | None = None) -> str:
    """NFC + (optional) Indic normalization + whitespace collapse. Never the Whisper default."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    if _INDIC is not None and lang and lang not in ("en", "english"):
        try:
            text = _INDIC.get_normalizer(lang).normalize(text)
            text = unicodedata.normalize("NFC", text)
        except Exception:
            pass
    return " ".join(text.split())
