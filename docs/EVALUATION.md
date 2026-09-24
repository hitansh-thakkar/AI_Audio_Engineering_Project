# Evaluation & Benchmarking

This is **Deliverable 2** and the intellectual core (Milestone 5: error propagation). The grade lives here. Build the instrumentation into the pipeline from Phase 0 — do not bolt it on.

## Per-stage metrics

| Stage | Metrics | Tooling | Notes |
| --- | --- | --- | --- |
| Separation | **SI-SNRi**, **SDRi** | `torchmetrics` / `speechbrain` | Needs reference sources → synthetic mixtures. |
| Diarization | **DER**, **JER**, speaker-count error | `pyannote.metrics` | Report with no forgiveness collar for honesty. |
| ASR | **WER**, **CER** | `jiwer` | **Script-safe Indic normalizer only** (NFC + `indic-nlp-library`). Split by register: Hindi / English / code-switched. |
| LLM summary | **ROUGE-L**, **BERTScore**, factual-consistency rate | `rouge-score`, `bert-score` | `bert-score` downloads a **free** model (e.g. `roberta-large`, ~1.4 GB) from HF on first use — one-time, then cached, fully local. Consistency = fraction of report claims entailed by / overlapping the transcript, checked with the same local Ollama model. |
| System | **wall-clock latency**, **peak memory** per module | `RunManifest` | Required by the brief. Measure on a fixed clip + device. |

### The normalization rule (do not skip)

Whisper's default normalizer strips Devanagari matras and inflates WER. All ASR metrics use NFC + Indic normalization. Implement once in `src/evaluation/normalize.py` and use it everywhere. Every reported WER/CER is invalid otherwise.

## The ablation study (the money result)

Sweep two axes (× separation backend) and record final + intermediate metrics per cell:

| separation.mode \ diarization | real | oracle |
| --- | --- | --- |
| **none** | | |
| **overlap** (default) | | |
| **everywhere** | | |

What each comparison isolates:

- **real vs. oracle diarization** (same separation) → cost of diarization error to final WER.
- **none vs. overlap vs. everywhere** (same diarization) → whether separation *helps or hurts*, and whether blind separation degrades the whole clip (expected: `everywhere` worst on real audio — the WSJ0-2Mix domain-gap effect made visible).
- Intermediate metrics (DER, SI-SNRi) alongside final WER → **where** error enters and how it cascades.

### Headline plot

`final WER (per register) vs. pipeline configuration` — a grouped bar chart across the six cells. This single figure answers the brief's central question. Secondary plots: per-stage error contribution; latency/memory per module.

## Benchmark harness

- `config/ablation.yaml` enumerates the sweep. `src/evaluation/benchmark.py` runs every cell, writing one `RunManifest` per run into `outputs/runs/`.
- `notebooks/benchmark.ipynb` loads the manifests, builds the tables/plots, and renders the headline figure. It must run **top-to-bottom without manual edits** (grader reproducibility).
- Fix the eval set, device, and seeds; record them in every manifest.

## Reporting honesty

State realistic numbers, not aspirational ones:

- DER on chaotic real radio: **20–45%** is normal.
- Code-switched conversational WER: often **20–40%+** even with strong models; separation may *raise* it.
- Frame high error as a **finding about cascade fragility**, which is exactly the project's thesis — not as failure.

## Definition of done

- [ ] Every metric implemented and unit-tested on a tiny fixture.
- [ ] Latency + peak memory captured per module in the manifest.
- [ ] Full 6-cell ablation runs from one command.
- [ ] Notebook reproduces all tables + the headline plot from saved manifests.
- [ ] Every number traces to a manifest (git SHA + config hash).
