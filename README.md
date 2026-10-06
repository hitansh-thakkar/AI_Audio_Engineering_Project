# Dhvani — "Who Spoke What and When" Speech Pipeline

A multi-stage audio-processing pipeline that takes **chaotic, multi-speaker, code-switched Indian audio** (community radio, panel discussions, agricultural phone-ins) and produces a **clean, speaker-attributed, summarized transcript report**.

This is a **systems-engineering + measurement** project, not a model-training project. It chains pretrained specialists — speech separation, speaker diarization, code-switched ASR, LLM post-processing — and, critically, **measures how errors propagate** from the acoustic front-end to the final text.

> **Status: mid-project.** The full pipeline runs end-to-end today. The acoustic front-end (preprocessing, diarization with hand-rolled clustering, separation) is implemented with real algorithms + metrics; ASR and LLM post-processing are structurally-valid **stubs** pending the second half. `ruff` + `mypy` clean, **21 tests pass**.

---

## What it does

```
raw multi-speaker audio (.wav)
        │
        ▼
 [1] Preprocessing      resample 16kHz mono, loudness-normalize, trim        [implemented]
        │
        ▼
 [2] Diarization        who spoke when (+ overlap, speaker count) via         [implemented]
                        hand-applied Spectral Clustering / GMM + timeline JSON
        │
        ▼
 [3] Separation         SepFormer/Conv-TasNet on OVERLAPPED regions only      [interface + metrics;
                                                                               real backend optional]
        │
        ▼
 [4] ASR + Alignment    faster-whisper / Tara → word-timestamped text         [stub — post-midsem]
        │
        ▼
 [5] LLM Post-Process   Google Gemini API (free tier) → clean, summarize      [stub — post-midsem]
        │
        ▼
 structured report: executive summary + keyword tags + speaker-separated dialogue
        + RunManifest (git SHA, config hash, seeds, per-stage latency + peak memory, notes)
```

The pipeline is **diarization-first by design** (not separation-first) — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the reasoning. The research payload is the **ablation study** (final WER vs. pipeline configuration × real/oracle diarization × separation mode), landing in the second half.

---

## Quickstart

Runs on one consumer PC. Two **free** credentials: a Hugging Face token (pyannote's gated-but-free licenses) and a Gemini API key (LLM stage, free tier). No paid services.

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. System deps: ffmpeg on PATH

# 3. Credentials — copy the template and fill in (both free)
cp .env.example .env
#   HF_TOKEN=...        https://huggingface.co/settings/tokens (+ accept pyannote licenses)
#   GEMINI_API_KEY=...  https://aistudio.google.com/apikey

# 4. Run the pipeline on one clip (raw audio -> report + manifest)
python -m src.pipeline --audio dhvani_eval/sequential/audio/clip_01.wav

# 5. Quality gates
ruff check src tests
mypy src
pytest
```

Outputs land in `outputs/<clip_id>/`: `report.json`, `report.md`, `timeline.json` (the Milestone-2 deliverable), and `manifest.json` (timing, memory, versions, notes).

---

## Repository layout

```
AI_Audio_Engineering_Project/
├── README.md                    ← you are here
├── CLAUDE.md                    ← guidance Claude Code reads automatically
├── PLAN.md                      ← phased build plan with acceptance criteria
├── MIDSEM_PRESENTATION_SPEC.md  ← slide-by-slide spec for the midsem deck
├── requirements.txt
├── pyproject.toml               ← ruff / mypy / pytest config
├── .env.example                 ← copy to .env (free HF_TOKEN + GEMINI_API_KEY)
├── config/
│   └── default.yaml             ← model registry + all behavioral toggles
├── docs/
│   ├── ARCHITECTURE.md          ← design, module contracts, data flow
│   ├── MODELS.md                ← concrete model picks, reasons, gotchas
│   ├── DATASETS.md              ← corpora, access, the ≤5 GB disk budget
│   ├── LOCAL_SETUP.md           ← install, hardware tiers, troubleshooting
│   └── EVALUATION.md            ← metrics, benchmark harness, ablation matrix
├── src/
│   ├── types.py                 ← dataclass contracts between stages
│   ├── config.py                ← OmegaConf load + .env secrets
│   ├── stage.py                 ← Stage protocol + timing/peak-memory instrumentation
│   ├── pipeline.py              ← orchestrator + CLI (python -m src.pipeline)
│   ├── preprocessing/           ← resample / normalize / trim
│   ├── diarization/             ← VAD, embeddings, Spectral/GMM clustering, RTTM, timeline JSON
│   ├── separation/              ← overlap-only separation (modes: none|overlap|everywhere)
│   ├── asr/                     ← ASR stub (post-midsem: faster-whisper / Tara + WhisperX)
│   ├── postprocessing/          ← LLM stub (post-midsem: Gemini report + guardrail)
│   └── evaluation/              ← DER, SI-SNRi/SDRi, WER/CER, script-safe normalizer
├── tests/                       ← 21 tests (unit + end-to-end smoke)
├── dhvani_eval/                 ← curated eval dataset + reproducible build/fetch scripts
├── indic_multispeaker_10/       ← raw source clips (IndicVoices-derived)
└── outputs/                     ← per-run reports + manifests (gitignored)
```

---

## Dataset (`dhvani_eval/`)

224 wav files (176 usable inputs + 48 separation reference stems), built reproducibly from real speech — see [`dhvani_eval/README.md`](dhvani_eval/README.md). Subsets: `sequential/` (multi-speaker + RTTM), `sources/` (isolated utterances), `overlap_mixtures/` (synthetic overlaps + ground-truth sources), `registers/` (English + code-switch), `vaani/` and `nirantar/` (diversity / transcribed-speech slices).

**Required corpora (proposal) — 3 of 4 in the eval set, all as ≤5 GB slices:**

| Corpus | Status |
| --- | --- |
| IndicVoices | ✅ used (base of dhvani_eval) |
| Project Vaani | ✅ used (`fetch_vaani.py`) |
| Nirantar | ✅ used (`fetch_nirantar.py`, official E2E-prefix slice) |
| AIR-RS-DB | ⏳ pending — IEEE DataPort subscription download → `ingest_air_rs_db.py` |

---

## Deliverables (course rubric)

1. **Pipeline source code** — unified, documented, raw audio → final report (`src/`).
2. **System benchmark notebook** — per-module latency + memory, plus the ablation study (post-midsem; instrumentation already in the `RunManifest`).

Per-milestone required outputs: M1 isolated per-speaker audio · M2 **timeline JSON** via self-implemented **Spectral Clustering / GMM** · M3 timestamped speaker-delineated transcript · M4 report (summary + keyword tags + speaker-separated dialogue).

See [`PLAN.md`](PLAN.md) for the phased plan and [`CLAUDE.md`](CLAUDE.md) for the engineering rules (incl. proposal-compulsory items).

---

## License / attribution

Uses AI4Bharat (IndicVoices, Nirantar, IndicConformer), IISc/ARTPARK (Vaani), TCS (AIR-RS-DB), pyannote, SpeechBrain, OpenAI Whisper, and Google Gemini artifacts — each under its own license. Check per-model/per-dataset terms before redistribution.
