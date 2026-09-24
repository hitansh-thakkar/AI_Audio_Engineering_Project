# Dhvani — "Who Spoke What and When" Speech Pipeline

> *Dhvani* (ध्वनि, "sound/voice") is a working codename — rename freely.

A multi-stage audio-processing pipeline that takes a **chaotic, multi-speaker, code-switched Indian audio broadcast** (community radio, panel discussions, agricultural phone-ins) and produces a **clean, speaker-attributed, summarized transcript report**.

This is a **systems-engineering + measurement** project, not a model-training project. It chains pretrained specialists — speech separation, speaker diarization, code-switched ASR, and LLM post-processing — and, critically, **measures how errors propagate** from the acoustic front-end to the final text.

---

## What it does

```
raw multi-speaker audio (.wav)
        │
        ▼
 [1] Preprocessing      resample 16kHz mono, loudness-normalize, optional denoise
        │
        ▼
 [2] Diarization        pyannote → who spoke when (+ overlap detection, speaker count)
        │
        ▼
 [3] Separation         SepFormer/Conv-TasNet on OVERLAPPED regions only
        │
        ▼
 [4] ASR + Alignment    faster-whisper (int8) large-v3 / Tara / IndicConformer → word-timestamped text
        │
        ▼
 [5] LLM Post-Process   local LLM via Ollama → clean, normalize code-switch, tag topics, summarize
        │
        ▼
 structured report: executive summary + keyword tags + speaker-separated dialogue
```

The pipeline is **diarization-first by design** (not separation-first). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the reasoning — in short, off-the-shelf separators are trained on clean 8 kHz two-speaker synthetic mixtures and inject artifacts into every downstream stage when run blind on real reverberant radio. We apply separation surgically, only where overlap actually exists.

## The research payload

The point of the project is the **ablation study**. Every run is configurable along two axes:

|                        | real diarization | oracle diarization |
| ---------------------- | ---------------- | ------------------ |
| **no separation**      | ✓                | ✓                  |
| **separate on overlap**| ✓                | ✓                  |
| **separate everywhere**| ✓                | ✓                  |

The headline result is *final WER vs. pipeline configuration* — a direct, quantified answer to "how much does each stage's error cost the final output?"

---

## Quickstart

**Runs fully locally on one consumer PC. Zero paid services** — no cloud inference, no per-token cost. The only credential is a **free** Hugging Face token (pyannote's licenses are click-through; *gated ≠ paid*). See [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) for the full install, the hardware-tier table, and VRAM/RAM/disk budgeting.

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. System deps
#   - ffmpeg on PATH (audio I/O)
#   - Ollama for the local LLM stage:  https://ollama.com/download
ollama pull qwen3:4b          # shipped default (see the tier table for bigger models)

# 3. Auth — ONE free token, no paid keys
export HF_TOKEN=...            # free HF token; accept the click-through licenses for
                              #   pyannote/speaker-diarization-community-1 AND pyannote/segmentation
#   (faster-whisper + Ollama weights are open and need no token.)

# 4. Run the pipeline on one clip (all inference on your machine)
python -m src.pipeline --audio data/samples/panel_clip.wav --config config/default.yaml

# 5. Run the benchmark / ablation suite
jupyter notebook notebooks/benchmark.ipynb
```

No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `SARVAM_API_KEY` / `PYANNOTEAI_API_KEY` are required or used on the default path. They map to **optional paid** backends only (see [`docs/MODELS.md`](docs/MODELS.md)); the zero-cost local stack never touches them.

## Repository layout

```
dhvani/
├── README.md              ← you are here
├── CLAUDE.md              ← guidance Claude Code reads automatically
├── PLAN.md                ← phased build plan with acceptance criteria
├── requirements.txt
├── config/
│   └── default.yaml       ← model registry + ablation toggles
├── docs/
│   ├── ARCHITECTURE.md    ← pipeline design, module contracts, data flow
│   ├── MODELS.md          ← concrete model picks, HF repos, reasons, gotchas
│   ├── LOCAL_SETUP.md     ← local install, hardware-tier table, VRAM/RAM/disk budget, troubleshooting
│   ├── DATASETS.md        ← corpora, access, the "AIR-RS-DB" caveat
│   └── EVALUATION.md      ← metrics, benchmark harness, ablation matrix
├── src/
│   ├── preprocessing/
│   ├── diarization/
│   ├── separation/
│   ├── asr/
│   ├── postprocessing/
│   ├── evaluation/
│   └── pipeline.py        ← orchestrator
├── notebooks/
│   └── benchmark.ipynb    ← Deliverable 2 (latency/memory + ablation)
├── data/                  ← sample clips + eval slices (not committed)
├── outputs/               ← generated transcripts + reports
└── tests/
```

## Deliverables (course rubric)

1. **Pipeline source code** — unified, documented, raw audio → final report. (`src/`)
2. **System benchmark notebook** — per-module latency + memory, plus the ablation study. (`notebooks/benchmark.ipynb`)

## Status & open questions

- ⚠️ **Confirm with instructor what "AIR-RS-DB (All India Radio Dataset)" is.** No canonical dataset exists by that name; the real AIR-derived resource is AI4Bharat's **Shrutilipi**. See [`docs/DATASETS.md`](docs/DATASETS.md).
- ⚠️ **Confirm the diarization-first reordering is acceptable.** If the rubric strictly grades the literal separate→diarize→transcribe order, keep it as the `baseline` config and add the reordered config + ablation to show *why* it underperforms. Either path satisfies the rubric.

## License / attribution

Uses AI4Bharat (IndicVoices, IndicConformer), IISc/ARTPARK (Vaani), pyannote, SpeechBrain, and OpenAI Whisper artifacts — each under its own license. Check per-model terms before redistribution.
