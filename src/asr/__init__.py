"""ASR stage backends + a config-driven factory.

Default path is free/local: faster-whisper (CTranslate2). IndicConformer is the optional
localized Indic backend; Sarvam is an OPTIONAL PAID cloud backend (off the default path).
"""
from .asr import ASRStub


def make_asr(cfg):
    """Select an ASR backend from `asr.backend` (golden rule #2: no hardcoded choice)."""
    backend = str(getattr(cfg, "backend", "faster_whisper"))
    if backend == "indicwav2vec":                # proposal's localized Indic ASR (default)
        from .indicwav2vec import IndicWav2VecASR
        return IndicWav2VecASR(cfg)
    if backend == "faster_whisper":              # multilingual alternative (code-switch/English)
        from .whisper import FasterWhisperASR
        return FasterWhisperASR(cfg)
    if backend == "indicconformer":              # multilingual Indic (gated; needs NeMo)
        from .indic import IndicConformerASR
        return IndicConformerASR(cfg)
    if backend == "sarvam":                      # OPTIONAL PAID — not the default
        from .sarvam import SarvamASR
        return SarvamASR(cfg)
    return ASRStub(cfg)


__all__ = ["ASRStub", "make_asr"]
