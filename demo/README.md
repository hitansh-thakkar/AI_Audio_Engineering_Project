# Dhvani — Demo Bundle

A self-contained walk-through of the **current** pipeline on four clips. Each clip
shows a different capability. `before/` holds the raw input you start with; `after/`
holds everything the pipeline produced.

## The pipeline (what happens to each clip)

```
raw audio (before/<clip>.wav)
   1. Preprocess   16 kHz mono, high-pass, loudness-normalize, trim
   2. Diarize      ECAPA embeddings -> hand-applied Spectral clustering -> who-spoke-when
                   + pyannote overlap detection (simultaneous speech)
   3. Separate     SepFormer on overlap regions -> clean per-speaker tracks
   4. ASR          IndicWav2Vec (AI4Bharat) -> native-script transcript per speaker
   5. Report       Gemini -> executive summary + keyword tags + cleaned dialogue
-> after/<clip>/   processed.wav, speakers/*.wav, separated/*.wav, timeline.json,
                   <clip>.rttm, transcript in report.json, report.md, manifest.json
```

## The four clips

| # | Clip | What it demonstrates | Speakers | Overlaps | Separated |
|---|------|----------------------|:--------:|:--------:|:---------:|
| 1 | `01_overlap_two_speakers` | Two speakers overlapping — overlap is detected and separated into clean per-speaker tracks. | 2 | 1 | 2 |
| 2 | `02_multispeaker_timeline` | A multi-speaker turn-taking clip — hand-applied spectral clustering builds the speaker timeline. | 3 | 0 | 0 |
| 3 | `03_realworld_conversation` | Real spontaneous conversational audio (background chatter) — robustness on messy input. | 2 | 1 | 2 |
| 4 | `04_codeswitch_hindi_english` | Hindi-English code-switched speech — localized Indic ASR transcribes to native script. | 3 | 0 | 0 |

## How to watch the demo

1. **Listen to the input:** `before/<clip>.wav` — the messy original.
2. **Listen to the result:** `after/<clip>/speakers/*.wav` — one clean track per speaker.
3. **See who spoke when:** open `after/<clip>/timeline.json` (also `<clip>.rttm`).
4. **Read the transcript + summary:** open `after/<clip>/report.md`.
5. **Check the instrumentation:** `after/<clip>/manifest.json` has per-stage wall-time,
   peak memory, model versions, seeds, config hash, and any graceful-degradation notes.

## Reproduce

```bash
# whole bundle
python demo/make_demo.py

# or a single clip through the same pipeline
python -m src.pipeline --audio demo/before/01_overlap_two_speakers.wav \
    --config config/live.yaml --out demo/after
```

Credentials (both free, in `.env`): `HF_TOKEN` (pyannote + IndicWav2Vec, gated-but-free)
and `GEMINI_API_KEY` (report stage). Without them the pipeline still runs and degrades
gracefully (per-speaker tracks without overlap split / a minimal report), noting it in the manifest.
