# Handoff

**Goal:** Choose and fix a local LLM backend for stage 5 (post-processing) that satisfies the proposal's "localized foundational LLM" requirement.

**Done**
- `main` reset to `origin/main` (`1c7845f added milestone 4`); `git diff --quiet origin/main` passed. Local commit `2046a82` still lives on branch `Hitansh`. GitHub Desktop stash dropped (recoverable via `git stash apply 15f7ac9` until gc).
- Proposal (`Audio_Engineering_AI_Project_Proposal.docx`) audited against repo; gaps listed below.
- Airavata + OpenHathi tested via Ollama (Q4_K_M, RTX 3050 4 GB). Harness + raw outputs: `C:\Users\Hitansh\AppData\Local\Temp\claude\1db972064d451009\19adc057-89b9-4b9d-82e5-e1a4ab3096a8\scratchpad\llm_eval\` (`run_eval.py`, `fixtures.py`, `Modelfile.airavata`, `results/*.json`). Scratchpad may be cleared; copy it out if needed.
- Verdict: neither fits as default. Airavata gives hollow/placeholder JSON and makes things up (e.g. "26 rupees", "dams of the Prime Minister"), supports Hindi only, and has a 4K context. OpenHathi is a base model: on the repo prompt it returns `{"version":2,"type":"json"}`, loops on plain instructions, and works only with few-shot plain-text summaries.
- Ollama models installed: `airavata-tulu` (Tulu template), `openhathi-base`, plus the `hf.co/...` originals (~8.4 GB). Not yet removed.

**Broken on main (not yet fixed)**
- `src/postprocessing/ollama.py`: no `num_ctx` (Ollama default silently truncated a 2,050-token prompt to 1,026); `timeout=60.0` at ~line 141 is too short (observed 25–180 s); relies on the model's template (community GGUFs ship `{{ .Prompt }}`).
- `config/default.yaml:62`: `postprocessing.model` is shared, so `backend: ollama` sends `gemini-3.8-flash` to Ollama. The model ID itself is also unverified.
- `src/pipeline.py:229`: ASR is `SarvamASR` or `ASRStub`; there is no faster-whisper implementation, so the default run produces a stub transcript.
- `config/default.yaml:25`: diarization default `precision2` is paid pyannote.ai, which violates CLAUDE.md rule 0.
- Missing: loaders for Vaani / Nirantar / AIR-RS-DB, and the System Benchmark notebook. CLAUDE.md is out of sync with `main`.

**Decisions made**
- `main` must equal `origin/main`; local changes on this machine are discarded.
- Proposal model names ("such as") are examples, not mandates.

**Next steps** (awaiting user's pick; last question unanswered)
1. Ask: remove the test models, fix the Ollama bugs on a branch, and/or test other candidate local models with the same harness.
2. If fixing: branch off `main`; add per-backend model keys, `num_ctx`, configurable timeout, and a shipped Modelfile or `/api/chat`.
3. Optionally turn the 10 documented failure cases into `tests/fixtures/llm/` regression checks (offered, not yet accepted).
4. Then tackle the ASR stub (`src/pipeline.py:229`) and the paid diarization default.

**Files to read first**
- `CLAUDE.md`, `config/default.yaml`, `src/postprocessing/ollama.py`, `src/postprocessing/postprocess.py`, `src/pipeline.py`
