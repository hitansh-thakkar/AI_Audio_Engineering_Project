"""ASR stage backends + a config-driven factory.

Default path is free/local: faster-whisper (CTranslate2). IndicConformer / IndicWav2Vec are
optional localized Indic backends; Gemini ASR uses the free API tier; Sarvam is an OPTIONAL
PAID cloud backend (off the default path). Heavy backends are imported lazily.
"""
from .asr import ASRStub
from .gemini_asr import GeminiASR
from .sarvam import SarvamASR


def make_asr(cfg):
    """Select an ASR backend from `asr.backend` (golden rule #2: no hardcoded choice)."""
    backend = str(getattr(cfg, "backend", "faster_whisper"))
    if backend == "faster_whisper":
        from .whisper import FasterWhisperASR
        return FasterWhisperASR(cfg)
    if backend == "indicwav2vec":
        from .indicwav2vec import IndicWav2VecASR
        return IndicWav2VecASR(cfg)
    if backend == "indicconformer":              # gated; needs NeMo
        from .indic import IndicConformerASR
        return IndicConformerASR(cfg)
    if backend == "gemini":
        return GeminiASR(cfg)
    if backend == "sarvam":                      # OPTIONAL PAID — not the default
        return SarvamASR(cfg)
    return ASRStub(cfg)


__all__ = ["ASRStub", "GeminiASR", "SarvamASR", "make_asr"]
