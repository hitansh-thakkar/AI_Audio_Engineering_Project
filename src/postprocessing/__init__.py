"""LLM post-processing backends + a config-driven factory.

Default is the free Google Gemini API; `ollama` is the optional fully-offline local backend;
`api` (paid frontier/Sarvam) and the no-op `stub` sit behind the same interface. All degrade
to a minimal report rather than crashing (golden rule #4).
"""
from .postprocess import PostProcessorStub


def make_postprocessor(cfg):
    """Select an LLM backend from `postprocessing.backend` (golden rule #2)."""
    backend = str(getattr(cfg, "backend", "gemini"))
    if backend == "gemini":
        from .gemini import GeminiPostProcessor
        return GeminiPostProcessor(cfg)
    if backend == "ollama":
        from .local_llm import OllamaPostProcessor
        return OllamaPostProcessor(cfg)
    # "api" (paid) is not implemented on the free path; fall back to the stub with a clear name.
    return PostProcessorStub(cfg)


__all__ = ["PostProcessorStub", "make_postprocessor"]
