"""Script-safe text normalization for WER/CER.

CRITICAL: Whisper's default normalizer strips Devanagari matras and inflates Devanagari WER.
This module never touches combining marks (Unicode category M*), which carry vowel signs and
viramas in every Indic script. Implemented once, used everywhere.

Pipeline (order matters):
  1. NFC                         — canonical composition (precomposed nukta letters such as
                                   U+095B "ज़" are composition-excluded, so NFC yields ज + ◌़).
  2. indic-nlp normalization     — per-language canonicalisation (e.g. chandrabindu/anusvara
                                   variants, two-part vowel signs). Skipped for unsupported langs.
  3. casefold                    — only affects Latin text (English / code-switch registers).
  4. punctuation & symbols → " " — Unicode P* and S* (includes danda । and double danda ॥).
                                   Without this, "एक।" ≠ "एक" counts as a substitution.
  5. drop format chars (Cf)      — ZWJ/ZWNJ are orthographic variants, not lexical content.
  6. optional nukta folding      — sensitivity analysis only; nukta is lexically meaningful
                                   in some words, so it is OFF for headline numbers.
  7. whitespace collapse.
"""

from __future__ import annotations

import unicodedata
from typing import Any

_INDIC: Any = None
try:  # indic-nlp-library is a hard requirement for reported numbers (see requirements.txt)
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    _INDIC = IndicNormalizerFactory()
except ImportError:
    _INDIC = None

# Combining nukta code points across Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati,
# Oriya, Kannada). Only removed when fold_nukta=True.
_NUKTAS = frozenset("़়਼઼଼಼")

# indic-nlp language codes; others pass through steps 1, 3-7 only. Urdu is excluded: its
# indic-nlp normalizer requires `urduhack` (TensorFlow); NFC + punctuation strip suffices for
# Perso-Arabic here. Sindhi references in dhvani_eval are Devanagari, so the Devanagari
# (Hindi) normalizer is used for it via _LANG_ALIAS.
_INDIC_NLP_LANGS = frozenset(
    {"hi", "mr", "ne", "sa", "kok", "mai", "bn", "as", "pa", "gu", "or", "ta", "te", "kn", "ml"}
)
_LANG_ALIAS = {"sd": "hi"}
_CACHE: dict[str, Any] = {}


def indic_normalizer_available() -> bool:
    return _INDIC is not None


def _indic_normalize(text: str, lang: str | None) -> str:
    lang = _LANG_ALIAS.get(lang or "", lang)
    if _INDIC is None or not lang or lang not in _INDIC_NLP_LANGS:
        return text
    if lang not in _CACHE:
        try:
            _CACHE[lang] = _INDIC.get_normalizer(lang)
        except (ImportError, NotImplementedError, KeyError, ValueError):
            _CACHE[lang] = None  # unsupported here; steps 1, 3-7 still apply
    norm = _CACHE[lang]
    return text if norm is None else unicodedata.normalize("NFC", norm.normalize(text))


def normalize_text(text: str, lang: str | None = None, *, fold_nukta: bool = False) -> str:
    """NFC + Indic normalization + casefold + punctuation strip. Never the Whisper default."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _indic_normalize(text, lang)
    text = text.casefold()
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        elif cat == "Cf" or (fold_nukta and ch in _NUKTAS):
            continue
        else:
            out.append(ch)
    return " ".join("".join(out).split())


# ---------------------------------------------------------------- script diagnostics
# Expected Unicode script (first word of unicodedata.name) per language code.
EXPECTED_SCRIPT: dict[str, str] = {
    "hi": "DEVANAGARI",
    "mr": "DEVANAGARI",
    "ne": "DEVANAGARI",
    "sa": "DEVANAGARI",
    "sd": "DEVANAGARI",  # dhvani_eval Sindhi references are Devanagari (IndicVoices convention)
    "bn": "BENGALI",
    "as": "BENGALI",  # Assamese uses the Bengali-Assamese block
    "pa": "GURMUKHI",
    "gu": "GUJARATI",
    "or": "ORIYA",
    "ta": "TAMIL",
    "te": "TELUGU",
    "kn": "KANNADA",
    "ml": "MALAYALAM",
    "ur": "ARABIC",
    "en": "LATIN",
}


def script_share(text: str, lang: str) -> float | None:
    """Fraction of letters in `text` that are in the language's expected script.

    Separates *script/language-ID failures* (e.g. Whisper emitting Devanagari for Malayalam)
    from ordinary recognition errors — both inflate WER, but for different reasons.
    Code-switched Hindi legitimately mixes LATIN, so callers should not apply it there.
    """
    expected = EXPECTED_SCRIPT.get(lang)
    if expected is None:
        return None
    letters = [c for c in text if unicodedata.category(c)[0] in ("L", "M")]
    if not letters:
        return None
    hits = sum(1 for c in letters if unicodedata.name(c, "").startswith(expected))
    return hits / len(letters)
