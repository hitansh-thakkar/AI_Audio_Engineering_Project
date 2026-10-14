"""Stage 5 — optional fully-offline LLM post-processing via a local Ollama server.

Same contract/guardrails as the Gemini backend, but calls http://localhost:11434 (no API
key, no cloud) so the pipeline can run entirely offline when required (golden rule #7). Uses
stdlib urllib — no extra dependency. Degrades to the minimal report if Ollama isn't running.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..types import Report, Transcript
from .gemini import SYSTEM_INSTRUCTION, _minimal_report, build_report


class OllamaPostProcessor:
    name = "postprocessing"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"ollama-{cfg.model}"
        self._host = str(getattr(cfg, "host", "http://localhost:11434")).rstrip("/")

    def load(self) -> None:
        return

    def run(self, transcript: Transcript, notes: list[str] | None = None) -> Report:
        notes = [] if notes is None else notes
        if not transcript.segments:
            return _minimal_report(transcript, notes, "empty transcript")
        numbered = "\n".join(f"{i} | {s.speaker} | {s.text}"
                             for i, s in enumerate(transcript.segments))
        payload = json.dumps({
            "model": str(self.cfg.model),
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user",
                 "content": f"Transcript segments (index | speaker | text):\n{numbered}"},
            ],
            "format": "json", "stream": False, "options": {"temperature": 0.2},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self._host}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(body["message"]["content"])
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            return _minimal_report(transcript, notes,
                                   f"Ollama unavailable at {self._host} ({type(exc).__name__}: {exc})")
        notes.append(f"Ollama ({self.cfg.model}) structured the report")
        return build_report(parsed, transcript, notes)
