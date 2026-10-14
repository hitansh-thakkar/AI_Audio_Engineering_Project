# Dhvani — Project Summary

**"Who Spoke What and When"**
Team HTMN · Advanced Computational Speech Engineering

---

## What We Are Building

A pipeline that takes raw, messy, multi-speaker Indian audio — the kind from community radio, rural panel discussions, phone-in programs — and produces a clean, speaker-attributed, summarized transcript. The challenge: speakers overlap each other, switch between Hindi/regional and English mid-sentence, and the recordings are full of background noise. Standard ASR tools collapse under these conditions.

Our approach is systems engineering: we chain pretrained specialist models together, measure how errors from one stage affect the next, and run a rigorous ablation study across pipeline configurations.

---

## The 5-Stage Pipeline

```
Raw Audio (.wav)
      │
      ▼
[1] Preprocessing     → resample to 16 kHz mono, peak-normalize, trim silence
      │
      ▼
[2] Diarization       → who spoke when (Spectral Clustering / GMM on MFCC embeddings)
      │                  outputs: timeline.json  ← Milestone 2 deliverable
      ▼
[3] Separation        → SepFormer on overlapping regions only
      │                  outputs: per-speaker .wav files
      ▼
[4] ASR + Alignment   → faster-whisper (local, free) → word-timestamped text
      │                  per speaker; code-switch aware   [implemented]
      ▼
[5] LLM Post-Process  → Google Gemini (free tier) → summary + tags + dialogue
      │                  no-new-facts guardrail            [implemented]
      ▼
report.json / report.md + manifest.json (latency, memory, config hash per stage)
```

**Design choice:** we diarize first, then separate only the overlapping regions. Off-the-shelf separators (trained on 8 kHz, 2-speaker, synthetic data) introduce artifacts when applied blindly to real audio. Diarizing first tells us exactly where overlaps occur and how many speakers exist — making separation targeted and the error-propagation study clean.

---

## What Is Done (Milestones 1 & 2)

### Preprocessing — fully implemented

- Resamples any input to 16 kHz mono
- Peak normalization (LUFS falls back to peak with a logged note)
- Silence trimming via librosa
- Handles chunking for long files

### Diarization — fully implemented (Milestone 2 centerpiece)

This is the compulsory hand-rolled implementation the course requires.

- Energy VAD to detect speech regions
- Windowed segments → MFCC-stats speaker embeddings (pyannote embeddings plug in as an upgrade)
- **Spectral Clustering** (sklearn): silhouette score selects speaker count automatically
- **GMM** (sklearn): BIC selects speaker count automatically
- No prior speaker templates; both methods implemented and selectable via config
- Adjacent same-speaker turns merged (gap < 0.5 s)
- Oracle mode: load ground-truth RTTM for the ablation study
- **Emits `timeline.json`** — the required Milestone 2 deliverable
  ```json
  [{"start": "00:00", "end": "00:04", "speaker": "SPEAKER_00"},
   {"start": "00:03", "end": "00:05", "speaker": "SPEAKER_02"}, ...]
  ```
- Preliminary real DER ≈ 0.30 on 10-clip eval set (MFCC fallback embeddings; expected range 20–45% on chaotic audio)

### Separation — interface + metrics implemented

- SepFormer (SpeechBrain, `speechbrain/sepformer-wsj02mix`) wired and weights downloaded
- Three modes: `none`, `overlap` (default — surgical), `everywhere` (ablation baseline)
- Graceful passthrough when model unavailable — the pipeline keeps running and logs a note
- Evaluation harness implemented: **SI-SNRi** and **SDRi** on the 24-mixture overlap set

### Evaluation framework — fully implemented

- **DER** (frame-based, no collar, Hungarian label mapping) for diarization
- **SI-SNRi / SDRi** for separation quality
- **WER / CER** with Indic-script-safe normalization (ready for ASR stage)
- Per-run `RunManifest`: git SHA, config hash, seeds, per-stage wall time + peak memory

### Pipeline infrastructure

- Full end-to-end run works today (ASR and LLM stages produce structurally valid stub output)
- Typed dataclass contracts between every stage (`src/types.py`)
- Every model ID and behavioral toggle is in `config/default.yaml` — no hardcoded values
- Graceful degradation: a failing stage logs a note and passes through rather than crashing
- `ruff` + `mypy` clean; **53 tests pass** (unit + end-to-end smoke)

### Dataset — `dhvani_eval/` (224 wav files, ~46 MB)

Built from real speech corpora, reproducibly:

| Subset              | Contents                                                   |
| ------------------- | ---------------------------------------------------------- |
| `sequential/`       | 10 multi-speaker clips + RTTM ground truth + transcripts   |
| `sources/`          | 30 isolated utterances (Assamese / Bengali / Hindi)        |
| `overlap_mixtures/` | 24 synthetic 2-speaker overlaps + ground-truth stems       |
| `registers/`        | 40 English + 24 Hindi–English code-switch clips            |
| `vaani/`            | 24 real Project Vaani clips (rural dialect diversity)      |
| `nirantar/`         | 24 real Nirantar clips (spontaneous conversational speech) |

Required corpora status:

- IndicVoices ✅ — base of the entire eval set
- Project Vaani ✅ — 24-clip slice via `fetch_vaani.py`
- Nirantar ✅ — 24-clip slice via `fetch_nirantar.py`

---

## Recently Completed

### ASR + Alignment — implemented

- **faster-whisper** (CTranslate2, int8; local + free) behind the `asr` factory; native word
  timestamps (no separate WhisperX needed); per-speaker, code-switch aware (language auto-detect).
- Optional **IndicConformer** (native-script Indic) and **Sarvam** (PAID) backends, off the default path.
- WER/CER use the Indic-script-safe normalizer, reportable per register.

### LLM Post-Processing — implemented

- **Google Gemini** (free tier) default via `google-genai`; optional fully-offline **Ollama**.
- Produces executive summary + keyword tags + cleaned speaker-separated dialogue.
- No-new-facts guardrail (system prompt) + a zero-weight numeric-fact grounding check.

### Stronger diarization — implemented

- Default embeddings moved MFCC → **ECAPA**; **eigengap** speaker counting + **refined affinity**
  (DER median 0.34→0.08, speaker-count accuracy 0.2→0.8); fixed a dead pyannote overlap detector.

## What Is Left

### Ablation study (the graded research payload)

- Final WER vs. pipeline configuration × {real, oracle} diarization × {none, overlap, everywhere} separation
- Runs across all dataset subsets and languages
- Everything feeds into the **System Benchmark Notebook** (the second course deliverable)

---

## Current State at a Glance

| Component                                | Status         |
| ---------------------------------------- | -------------- |
| Audio preprocessing                      | ✅ Done        |
| Spectral Clustering diarization          | ✅ Done        |
| GMM diarization                          | ✅ Done        |
| Timeline JSON output (M2 deliverable)    | ✅ Done        |
| DER evaluation metric                    | ✅ Done        |
| Separation interface + SepFormer backend | ✅ Done        |
| SI-SNRi / SDRi metrics                   | ✅ Done        |
| Per-run manifest (timing + memory)       | ✅ Done        |
| Dataset (3 of 4 corpora)                 | ✅ Done        |
| 53 tests passing, ruff + mypy clean      | ✅ Done        |
| ASR (faster-whisper, local/free)         | ✅ Done        |
| LLM post-processing (Gemini + guardrail) | ✅ Done        |
| ECAPA embeddings + eigengap counting     | ✅ Done        |
| Ablation study                           | ⏳ Remaining   |
| Benchmark notebook                       | ⏳ Remaining   |
