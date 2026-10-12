# Architecture

The design, the module contracts, and — importantly — *why* the pipeline is ordered the way it is.

## Design principle: diarization-first, separation-surgical

The project brief prescribes **separate → diarize → transcribe**. We invert it to **diarize → (separate overlaps) → transcribe**, for concrete, evidenced reasons:

1. **Off-the-shelf separators don't generalize to real audio.** SepFormer / Conv-TasNet public checkpoints are trained on **WSJ0-2Mix**: clean, 8 kHz, *exactly two* speakers, fully-overlapped *synthetic* mixtures. They score ~22 dB SI-SNRi on that matched test set, but the separation literature documents sharp degradation on real, reverberant, noisy audio. Real All India Radio is 16 kHz, reverberant, with a variable and unknown speaker count. Running blind separation up front therefore **injects artifacts into every downstream stage** and *raises* final WER.

2. **Separation front-ends assume a fixed speaker count** (usually 2). Radio panels have an unknown, variable number of speakers. Diarization is the component actually built to estimate speaker count.

3. **pyannote already does most of Milestone 2 end-to-end** — VAD, overlap detection, embedding, clustering. It also (in community-1) has an exclusive-speaker mode designed to reconcile with ASR timestamps. So diarization is the natural backbone; separation becomes a *targeted repair* for the specific regions where speakers actually overlap. **For Milestone-2 compliance we still perform the clustering step ourselves** — Spectral Clustering / GMM over pyannote embeddings (`diarization.clustering: spectral | gmm`) — rather than leaning on its built-in clusterer, which we keep only as a `builtin` comparison.

This ordering is also what makes the **error-propagation study** (Milestone 5 / Deliverable 2) clean: with diarization as the backbone, we can substitute *oracle* diarization and measure exactly how much downstream error is caused by diarization mistakes vs. everything else.

> If the instructor requires the literal brief order, keep it as the `baseline` config and add this reordering plus the ablation to demonstrate, with numbers, why it underperforms.

## Data flow

```
AudioClip
  → Preprocessing        → AudioClip (16kHz mono; optional high-pass, spectral denoise, LUFS)
  → Diarization          → DiarizationResult { turns[], overlap_regions[] }
                           ECAPA embeddings → hand-applied Spectral/GMM clustering (count via
                           cosine silhouette); overlap_regions from the pyannote detector when
                           enabled. Long clips: per-chunk labels reconciled into global speakers
                           by voiceprint (src/diarization/linking.py).
  → Separation (overlap) → list[SeparatedSource]: SepFormer runs on a context window around each
                           overlap; each output stream is attributed to a diarization speaker by
                           matching it to the mixture in that speaker's solo region, rescaled to
                           mixture level. The orchestrator stitches each source into a full-length
                           per-speaker track (original audio in solo turns).
  → ASR + Alignment      → Transcript { segments[] with words[] and speaker }   (stub, post-midsem)
  → LLM Post-Processing   → Report { summary, tags[], dialogue[] }   (Gemini API; stub, post-midsem)
  + RunManifest           (timing, memory, versions, seeds, config hash — accumulated across stages)

Deliverables written per run: speakers/<spk>.wav, separated/<src>.wav, timeline.json,
<clip>.rttm, report.json/md, manifest.json.
```

## Stage contract

Every stage implements the same protocol so stages are swappable and independently testable.

```python
from typing import Protocol
from dataclasses import dataclass

class Stage(Protocol):
    name: str          # e.g. "diarization"
    version: str       # model id + revision, for the manifest

    def load(self) -> None:
        """Lazily load weights/clients. No heavy work at import time."""

    def run(self, inp): ...
        """Pure-ish transform: input dataclass -> output dataclass.
        Must not raise on recoverable failures — log, degrade, record."""
```

The orchestrator wraps each `run()` in a timer + peak-memory tracker and appends a `StageResult` to the `RunManifest`. **The benchmark reads this manifest** — instrumentation lives in the code from Phase 0, not bolted on later.

## Core dataclasses (`src/types.py`)

```python
@dataclass
class AudioClip:
    samples: np.ndarray        # float32, mono
    sample_rate: int           # canonical 16000 after preprocessing
    source_path: str

@dataclass
class SpeakerTurn:
    start: float               # seconds
    end: float
    speaker: str               # "SPEAKER_00", ...

@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    overlap_regions: list[tuple[float, float]]   # (start, end) flagged as overlap
    num_speakers: int

@dataclass
class SeparatedSource:
    region: tuple[float, float]
    speaker: str               # assigned back to a diarization speaker
    samples: np.ndarray

@dataclass
class Word:
    text: str; start: float; end: float

@dataclass
class TranscriptSegment:
    start: float; end: float
    speaker: str
    text: str
    words: list[Word]
    lang: str | None           # detected/segment language, incl. "mixed"

@dataclass
class Transcript:
    segments: list[TranscriptSegment]

@dataclass
class Report:
    summary: str
    tags: list[str]
    dialogue: list[TranscriptSegment]   # speaker-separated, cleaned
    raw_transcript: Transcript

@dataclass
class StageResult:
    name: str; version: str
    wall_time_s: float
    peak_mem_mb: float
    notes: list[str]           # recoverable failures, degradations

@dataclass
class RunManifest:
    git_sha: str
    config_hash: str
    seeds: dict
    stages: list[StageResult]
    config_snapshot: dict
```

## Module responsibilities

| Module | Input | Output | Key model(s) |
| --- | --- | --- | --- |
| `preprocessing` | raw audio | `AudioClip` (16k mono, conditioned) | high-pass + spectral-gating denoise + BS.1770 LUFS (pyloudnorm); opt. Demucs/DNS |
| `diarization` | `AudioClip` | `DiarizationResult` | ECAPA-TDNN embeddings → hand-applied Spectral/GMM; pyannote community-1 as overlap detector (opt.) |
| `separation` | `AudioClip` + `DiarizationResult` | `list[SeparatedSource]` | SepFormer (config-driven: whamr/16 kHz) / Conv-TasNet (Asteroid, opt.); + solo-region attribution |
| `asr` | speaker turns (+ separated sources) | `Transcript` | faster-whisper (int8) large-v3 / Tara-CT2 / IndicConformer; WhisperX align |
| `postprocessing` | `Transcript` | `Report` | Google Gemini API, free tier (opt. local Ollama) |
| `evaluation` | manifests + refs | metrics, plots | jiwer, pyannote.metrics |

See `docs/MODELS.md` for exact model IDs, reasons, and gotchas.

## Configuration & toggles

All behavior is driven by `config/default.yaml`. The ablation-relevant switches:

- `diarization.source: real | oracle`
- `diarization.clustering: spectral | gmm` (**required** hand-applied unsupervised clustering of embeddings — proposal Milestone 2)
- `diarization.embedding: auto | mfcc` (offline fallback) `| ecapa` (SpeechBrain, recommended)
- `diarization.overlap_detector: none | pyannote` (gated-but-free; populates overlap_regions)
- `diarization.num_speakers / vad_top_db / window_seconds / window_hop` (counting + segmentation knobs)
- `separation.mode: none | overlap | everywhere`
- `separation.backend: sepformer | resepformer | whamr | wham | convtasnet`, with `separation.model`, `model_sample_rate`, `context_seconds`
- `preprocessing.highpass_hz / denoise (none|spectral) / normalize (peak|lufs|none) / target_lufs`
- `asr.backend: faster_whisper | indicconformer` (with `asr.model` selecting large-v3 / Tara-CT2 / distil / medium / small)
- `postprocessing.backend: gemini` (free API, default) `| ollama` (optional local) `| api` (optional paid)

The ablation runner sweeps the first two axes (× separation backend) and records final + intermediate metrics per cell.

## Failure & degradation policy

Errors must **flow, not halt** — the project measures propagation. Concretely:

- Diarization returns 1 speaker → skip separation, transcribe whole clip, note it.
- Separation output is low-energy/garbage → fall back to the un-separated overlapped segment, note it.
- ASR returns empty on a segment → keep an empty `TranscriptSegment`, note it.
- LLM unavailable (Gemini API error / rate-limit / no key, or Ollama not running) → emit a minimal report from the raw transcript, note it.

Every degradation is appended to the relevant `StageResult.notes` and surfaces in the manifest.

## Reproducibility

- Seed everything; log seeds in the manifest.
- Pin model revisions in config; record resolved versions per run.
- Record git SHA + a hash of the resolved config. Two runs with the same manifest inputs must produce the same outputs.
