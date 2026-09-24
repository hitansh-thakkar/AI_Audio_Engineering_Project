# PLAN.md — Build Plan

Phased plan for building the Dhvani "Who Spoke What and When" pipeline. Each phase has concrete tasks and **acceptance criteria** (the definition of done). Build in order; each phase should leave `main` in a runnable state.

**Guiding principle:** vertical slices first. A working end-to-end run with stub models beats one perfect stage. Get the plumbing and the measurement harness in early, because the ablation study — not any single model — is the deliverable that earns the grade.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Phase 0 — Scaffold & contracts

Goal: an end-to-end pipeline runs on a 5-second clip using **stub** models, emitting a (nonsense but structurally valid) report.

- [ ] Repo skeleton per `README.md` layout; `pyproject.toml` / `requirements.txt`; `ruff` + `mypy` + `pytest` configured.
- [ ] Define core dataclasses in `src/types.py`: `AudioClip`, `SpeechSegment`, `SpeakerTurn`, `DiarizationResult`, `SeparatedSource`, `TranscriptSegment`, `Transcript`, `Report`, `RunManifest`. (See contracts in `docs/ARCHITECTURE.md`.)
- [ ] Define the `Stage` protocol (`load()`, `run()`, `name`, `version`) and a `StageResult` wrapper that carries timing + peak-memory metadata.
- [ ] Config system: load `config/default.yaml` via OmegaConf; a model registry maps stage → model id + params.
- [ ] Stub implementations for all five stages (return fixed/trivial outputs).
- [ ] `src/pipeline.py` orchestrator: runs stages in order, threads the dataclasses through, writes a `RunManifest` (git SHA, config hash, per-stage timing/memory, seeds).
- [ ] CLI: `python -m src.pipeline --audio <path> --config <path>`.
- [ ] Smoke test: end-to-end run on `tests/fixtures/tiny.wav` passes.

**Acceptance:** `python -m src.pipeline --audio tests/fixtures/tiny.wav` produces a `Report` and a `RunManifest` with timing for all five stages. CI green (lint, type, smoke test).

---

## Phase 1 — Preprocessing

Goal: real audio conditioning.

- [ ] `src/preprocessing/`: resample to 16 kHz mono, peak/loudness normalization (target ~ -23 LUFS or peak-normalize), silence trimming.
- [ ] Optional denoise toggle (`preprocessing.denoise: none | demucs | dns`). Demucs here isolates the **vocal stem from background music** — it does NOT separate speakers. Document this clearly in the code.
- [ ] Chunking utility for long files (the demo processes 30–120 s clips, not whole broadcasts).
- [ ] Unit tests on a fixture with known sample rate / channels.

**Acceptance:** arbitrary input audio (any sr/channels) is normalized to canonical 16 kHz mono; denoise toggle works; tests pass.

---

## Phase 2 — Diarization (backbone)

Goal: real "who spoke when," with overlap detection and speaker counting.

- [ ] `src/diarization/`: wrap `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0). Read HF token from env; surface gated-model / 401 errors clearly (segmentation agreement gotcha).
- [ ] Emit `DiarizationResult`: list of `SpeakerTurn(start, end, speaker_id)` + an overlap mask (regions flagged as overlapping speech).
- [ ] Support `num_speakers` / `min_speakers` / `max_speakers` passthrough from config.
- [ ] Optional `precision-2` backend behind `diarization.backend: community1 | precision2` — but note `precision2` is a **paid pyannoteAI cloud API**, off the free/default path. `community1` (local, CPU, free HF token) is the default.
- [ ] **Oracle mode:** load ground-truth RTTM when available and expose it through the same interface (needed for the ablation).
- [ ] Manual-clustering exercise (Milestone 2 pedagogy): a separate script that extracts pyannote embeddings and runs spectral clustering / GMM by hand. Keep it out of the production path; it exists to satisfy the milestone and to compare against the built-in clustering.

**Acceptance:** on a real multi-speaker clip, produces sensible speaker turns + overlap mask; oracle mode returns ground truth; DER computes against a reference RTTM.

---

## Phase 3 — Separation (surgical)

Goal: recover per-speaker audio in overlapped regions.

- [ ] `src/separation/`: wrap SepFormer (`speechbrain/sepformer-wsj02mix`) and Conv-TasNet as two selectable backends (large vs. light — the benchmark wants both).
- [ ] `separation.mode: none | overlap | everywhere`:
  - `overlap` (default): run only on regions in the diarization overlap mask.
  - `everywhere`: blind separation of the whole clip (for the ablation — expected to hurt).
  - `none`: skip entirely.
- [ ] Handle sample-rate mismatch (checkpoints are 8 kHz; resample in/out and document the quality cost).
- [ ] Handle the fixed-2-speaker limitation: define behavior when overlap involves >2 speakers (document as a known limitation; do not pretend to solve it).
- [ ] Map separated sources back onto diarized speaker identities (assignment by embedding similarity or temporal overlap).
- [ ] Compute SI-SNRi / SDRi where reference sources exist (synthetic mixtures for eval).

**Acceptance:** overlapped segments yield separated streams routed to the right speakers; SI-SNRi/SDRi computed on a synthetic-mixture eval set; all three modes runnable.

---

## Phase 4 — ASR + alignment

Goal: code-switched transcripts with word-level timestamps.

- [ ] `src/asr/`: runtime is **faster-whisper (CTranslate2, int8)** — local, runs on GPU or CPU. `asr.model` selects the weights:
  - `large-v3` (baseline; faster-whisper auto-downloads the CT2 conversion; int8 ≈ 2–4 GB VRAM). **Config default — runs out of the box.**
  - Tara (`Trelis/tara`, Hindi/Hinglish, mixed-code mode) — convert to CT2 (`ct2-transformers-converter`) or run via transformers int8. **Recommended for Hindi-belt audio.**
  - `distil-large-v3` / `medium` / `small` — fallbacks for weak hardware / tight disk.
  - `indicconformer` (`ai4bharat/indic-conformer-600m-multilingual` via AI4Bharat NeMo, optional/heavy, for non-Hindi Indic languages).
  - (The Sarvam Saaras-v3 API option has been removed — it is a paid cloud service, off the local path.)
- [ ] Transcribe each single-speaker diarized turn; for overlapped regions, transcribe the separated streams from Phase 3.
- [ ] Word-level alignment via WhisperX (`whisperx` wav2vec2 forced alignment) → `TranscriptSegment(start, end, text, speaker, words[])`.
- [ ] Reconcile ASR timestamps with diarization using pyannote community-1's exclusive-speaker mode.
- [ ] **Script-safe normalization** for eval (NFC + `indic-nlp-library`); never the Whisper default normalizer for Devanagari. WER + CER via `jiwer`, reported per register: pure-Hindi, pure-English, code-switched.

**Acceptance:** speaker-attributed, word-timestamped transcript for a real clip; WER/CER computed under the script-safe normalizer; at least two ASR backends comparable through config.

---

## Phase 5 — LLM post-processing

Goal: clean, structured, human-readable report.

- [ ] `src/postprocessing/`: LLM client behind an interface. Default `postprocessing.backend: ollama` — a **local** model served at `http://localhost:11434` (`postprocessing.model` picked by hardware tier, e.g. `qwen3:4b`/`qwen3:8b`/`gemma3:12b`; see `docs/LOCAL_SETUP.md`). An optional `api` backend (paid frontier/Sarvam) may sit behind the same interface but is off by default. This stage only cleans/summarizes text, so a 7–9B local model suffices.
- [ ] Prompt framework that: normalizes code-switching, repairs noise-induced dropouts, identifies core topics (e.g., market-rate disputes, weather), and produces the report structure: **executive summary + keyword tags + speaker-separated dialogue**.
- [ ] **Guardrail:** system prompt forbids introducing facts not present in the transcript; add a post-hoc factual-consistency check (e.g., entailment/keyword-overlap against the source transcript).
- [ ] Do NOT use Airavata / OpenHathi (2023–24, Hindi-only, GPT-3.5-era).
- [ ] Emit final `Report` (Markdown + JSON).

**Acceptance:** a fragmented raw transcript becomes a coherent structured report; guardrail check flags any hallucinated content; output validates against the `Report` schema.

---

## Phase 6 — Evaluation & benchmark (Deliverable 2)

Goal: the measurement that makes this a research project.

- [ ] `src/evaluation/`: metric implementations — SI-SNRi/SDRi, DER/JER + speaker-count error, WER/CER (script-safe, per register), summary metrics (ROUGE/BERTScore + factual-consistency).
- [ ] Per-module **latency + peak memory** instrumentation surfaced from the `RunManifest`.
- [ ] **Ablation runner** (`config/ablation.yaml`): sweep `separation.mode ∈ {none, overlap, everywhere}` × `diarization ∈ {real, oracle}`; collect final WER/CER + intermediate metrics for each cell.
- [ ] `notebooks/benchmark.ipynb`: tables + plots. **Money plot:** final WER vs. pipeline configuration. Secondary: error contribution per stage, latency/memory per module.
- [ ] Curate a small eval set: a few hand-checked multi-speaker clips + a synthetic-mixture set (for separation references) drawn from IndicVoices/Vaani slices.

**Acceptance:** the notebook reproduces the full ablation matrix end-to-end and renders the headline plot; every metric traces back to a logged run manifest.

---

## Phase 7 — Docs, polish, reproducibility

- [ ] Docstrings on every public class/function; module READMEs where useful.
- [ ] `RunManifest` captures git SHA, config hash, model versions, seeds, environment.
- [ ] A `make demo` / single command reproduces one full run + the report.
- [ ] Final pass on `README.md` and `docs/` so a grader can run it cold.
- [ ] Record known limitations honestly (WSJ0-2Mix domain gap, >2-speaker overlap, DER on chaotic audio).

**Acceptance:** a fresh clone + `pip install` + one command runs the demo; the benchmark notebook runs top-to-bottom without manual fixes.

---

## Non-goals (explicitly out of scope)

- Training separation/ASR models from scratch (WSJ0-2Mix needs LDC-licensed data; corpora are for eval/fine-tuning slices only).
- Real-time / streaming inference.
- Speaker *identification* (attaching real names) — diarization is anonymous "Speaker A/B" only.
- Productionization (serving, scaling). This is a research pipeline.

## Risk register

| Risk | Mitigation |
| --- | --- |
| "AIR-RS-DB" dataset doesn't exist as named | Confirm with instructor; default to Shrutilipi. See `docs/DATASETS.md`. |
| Separation checkpoints (8 kHz, 2-spk, synthetic) fail on real audio | Apply on overlap only; treat blind mode as an ablation showing the failure. |
| pyannote gated-model access / 401 | Document HF token + both user agreements; fail loud with a clear message. |
| High DER on chaotic radio (20–45%) | Expected — report it, use oracle diarization to isolate downstream error. |
| Instructor requires literal separate→diarize→transcribe order | Keep that as `baseline` config; add reordered config + ablation to justify. |
