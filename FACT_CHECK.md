# Dhvani — Repo Fact-Check Audit

Verification of every factual claim in the repo against measured reality.
Measured **2026-10-14** on the user's machine. Not committed.

Legend: ✅ accurate · ❌ inaccurate · ⚠️ misleading/partial

## Environment truth

| Fact | Value |
|---|---|
| Python | 3.12.5 (system install, **no venv**) |
| torch | `2.10.0+cpu` — **CPU-only build** |
| GPU | NVIDIA RTX 3050 Laptop, **4 GB VRAM**, driver 617.14 — *present but unusable by current torch* |
| Decision | **Run CPU-only.** CUDA torch reinstall (~2.5 GB wheels) is risky + marginal on 4 GB VRAM + eats the disk budget. GPU is an opt-in later. |
| Free disk | **11.7 GB** — the 10 GB budget is ~the whole free space; be frugal. |
| torchcodec | **Broken** (`libtorchcodec_image.dll` load fails) → pyannote built-in audio decoding fails; must feed in-memory `{"waveform","sample_rate"}` (overlap.py already does). |

## Quality gates (claims: "ruff + mypy clean", "N tests pass")

| Claim | Reality |
|---|---|
| "31 tests pass" (README) / "17 tests pass" (PROJECT_SUMMARY) | ❌ Both wrong & mutually contradictory. **Actual: 53 passed, 0 failed.** |
| "ruff clean" | ❌ **3 errors** (2 auto-fixable): import-sort + unused var in `src/ui/preprocess_service.py` / `preprocess_ui.py` (the newly-added UI). |
| "mypy clean" | ❌ **2 errors** in `src/ui/preprocess_service.py:246` (dict invariance on the ASR payload). |

Pre-existing `src/` was likely clean; the recently-added standalone preprocess UI broke both gates.

## Metric claims

| Claim (README) | Reality |
|---|---|
| "median SI-SNRi **+7.2 dB**" (SepFormer, 24 mixtures) | ✅ **7.20** median (n_scored=42 sources). Note: measured with **oracle** overlap regions from the mixture RTTMs, not the real diarization path. |
| "SDRi **+8.6 dB**" | ❌ **actual median 4.51** (`separation_eval` reports only `sdri_median`, no mean — not a bug, just a stale/wrong claim). |
| "Preliminary real DER ≈ 0.30 (MFCC)" (PROJECT_SUMMARY) | ✅ approx — MFCC DER mean 0.329 / median 0.344. |

### Measured diarization baseline (dhvani_eval/sequential, 10 clips, spectral)

| Embedding | DER median | DER mean | Speaker-count accuracy |
|---|---|---|---|
| `mfcc` (= `default.yaml: embedding: auto`) | 0.344 | 0.329 | **0.20** (2/10) |
| `ecapa` (= `live.yaml`) | **0.140** | 0.222 | **0.60** (6/10) |

Root-cause confirmed: the **MFCC default** roughly doubles DER and collapses speaker-count accuracy → the "pre-overlap→spk1 / post-overlap→spk2" temporal-block failure. ECAPA helps a lot but **still over-counts** on some clips (clip_01, clip_04: count_err=5) → the **silhouette k-estimator** also needs fixing.

## Implementation claims

| Claim | Reality |
|---|---|
| "faster-whisper / Tara is the ASR runtime" (README, docs, requirements) | ❌ **Not implemented** — 0 refs in `src`; package not even installed. Only `ASRStub` + `SarvamASR`. Default run ⇒ stub placeholder text. |
| "No paid services" (README:45) | ❌ `config/live.yaml` (`asr.backend: sarvam`, `saaras:v4`) + `src/ui/preprocess_service.run_asr` call **paid Sarvam**; `requirements.txt` lists `sarvamai` under a "fully local" header. (`sarvamai` not installed → currently fail-degrades.) |
| "ASR+LLM are structurally-valid stubs" | ✅ true (`ASRStub`, `PostProcessorStub`). |
| Missing installed deps | `faster_whisper`, `whisperx`, `sarvamai`, `google.genai`, `bert_score` all **not installed**. |

## Dataset claims — ✅ accurate

224 wav verified (sequential 10 + sources 30 + overlap_mixtures 72 + registers english 40 + codeswitch 24 + vaani 24 + nirantar 24), ~45 MB. 3/4 required corpora present; **AIR-RS-DB absent** (IEEE-gated) — matches README's "pending".

## Disk ledger (budget ≤ 10 GB incl. dataset)

| Item | Size |
|---|---|
| `dhvani_eval/` dataset | 0.045 GB |
| `models/` (sepformer + ecapa) | 0.193 GB |
| HF cache (`~/.cache/huggingface`) | 0.24 GB |
| **Total used** | **~0.48 GB** / 10 GB |

Headroom ~9.5 GB logical, but only **11.7 GB physical free** — never hold two heavy ASR weights at once.

## Bugs found & fixed during execution

- **Diarization clustering (C1):** MFCC default + raw cosine affinity cut the timeline on a *time* boundary → "pre-overlap=spk1, post-overlap=spk2". Fixed: default `embedding: ecapa` + `affinity: refined` (pruned kNN graph). Sequential **DER median 0.344 → ~0.10; speaker-count accuracy 0.20 → 0.80**.
- **Speaker over-counting on overlap clips:** blended overlap windows formed a phantom extra speaker (user-reported: mix_002 gave 3-4 vs true 2). Fixed two ways: (a) default `count_method: silhouette` (robust — overlap-set count accuracy 0.42→0.75, DER 0.40→0.21, negligible sequential cost) instead of eigengap (which over-counts short clips); (b) when pyannote overlap regions are known, overlap-contaminated windows are excluded from the speaker count and attached to the nearest speaker (`diarize._cluster`). mix_002 now → **2 speakers**.
- **Overlap detector (C2), real bug:** `overlap.py` called `.get_overlap()` on pyannote 4.0's `DiarizeOutput` (no such method) → always threw → **overlaps never detected on real audio → separation never fired** (a core cause of "overlap not separated"). Fixed by unwrapping `.speaker_diarization`. Verified: mix_002 now detects overlap 4.2–4.59s vs ground-truth 3.63–4.8s (was `[]`). Enabled `overlap_detector: pyannote` on `live.yaml`.

## Separation (C2) outcome

- Separator itself was never broken: SepFormer loads (`available: True`) and gives SI-SNRi +7.20 dB. The symptom was upstream (diarization + the dead overlap detector).
- Backend sweep on the **clean synthetic** overlap set: `sepformer-wsj02mix` (8 kHz) 7.20 dB ≫ `sepformer-whamr16k` (16 kHz) 2.81 dB — because whamr16k is trained for noisy/reverberant audio. **wsj02mix kept as default** (statistically best on available data); whamr16k documented as the toggle for noisy real uploads (no eval set can score it fairly — the known synthetic↔real gap).
- Tried a `merge_threshold` centroid-merge; it helped only artificial short-overlap clips but over-merged weak MFCC embeddings (0.12→0.62) and was a no-op on the representative set — **removed in the cleanup as dead flexibility**. Robust wins kept: ecapa + refined affinity + silhouette counting + overlap-window exclusion.
- **Known limitation:** very short (6–9 s) overlap clips still over-count by ~1 speaker (overlap-straddling windows); longer real audio clusters well. `num_speakers` override bypasses counting.

- **High-pass on the diarization path (live.yaml), real bug:** `highpass_hz: 70` distorted ECAPA embeddings and over-counted speakers (clip_01 3→6; SOS/lower-order filters over-count too, so it's inherent, not a filter-impl bug — same class as denoise-on-diar-path). Fixed: `highpass_hz: 0` on both configs; rumble is handled by `denoise_output` on the final tracks. Note: the first-half benchmark runs on raw audio so it had *hidden* this — with high-pass off, pipeline diarization matches the benchmark again (lufs/trim don't change the count).

## Corrective actions (all applied this session)

- [x] Diarization fix C1: `embedding: auto → ecapa`, `count_method: eigengap`, `affinity: refined`.
- [x] Fixed the dead pyannote overlap detector (`DiarizeOutput.speaker_diarization` unwrap); enabled on `live.yaml`.
- [x] Corrected the SDRi claim (8.6 → measured 4.51) in README.
- [x] Demoted Sarvam to optional paid toggle; `faster_whisper` is the free default on all paths; fixed `requirements.txt` header; `preprocess_service` now respects `cfg.asr.backend`.
- [x] Implemented real **faster-whisper** ASR (D1, validated) + optional IndicConformer; installed deps within budget.
- [x] Implemented real **Gemini** LLM post-processing (D2, validated) + Ollama option + no-new-facts guardrail; updated stale model id `gemini-2.5-flash → gemini-3.8-flash`.
- [x] Fixed the pre-existing ruff/mypy errors in the preprocess UI — repo now **ruff + mypy clean (36 files), 53 tests pass**.
- [x] Reconciled test-count (17/31 → 53), SDRi, and ASR/LLM-stub claims across README / PROJECT_SUMMARY / requirements.
- [x] Added `config/ablation.yaml` + runnable `notebooks/benchmark.ipynb` (System Benchmark deliverable).

## Final state (end of session)

- **Quality gates:** `ruff` clean, `mypy` clean (36 files), **53 tests pass** (exit 0). Tests stay fast/offline (stub ASR/LLM).
- **Disk ledger:** weights + dataset = **6.81 GB / 10 GB** budget (8.7 GB free) after adding faster-whisper-small + IndicWav2Vec-hindi.
- **ASR backend (proposal compliance):** proposal names IndicASR/IndicWav2Vec (localized, native script). `config/live.yaml` now defaults to **IndicWav2Vec** (`ai4bharat/indicwav2vec-hindi`, transformers CTC, native Devanagari) — gated-but-free (HF_TOKEN, accepted). Fixed the "mix_002 Stage-5 = 1 speaker" symptom: both speakers now transcribe (51/40 chars vs 3-6 with whisper-small). `default.yaml` keeps open faster-whisper (no gate) for cold runs. IndicConformer (multilingual, NeMo, **gate not accepted — 403**) wired + now passes HF_TOKEN; available once its licence is accepted.
- **Remaining (research payload, not blocking):** full real-ASR ablation matrix + money-plot across `config/ablation.yaml`; a literal `run.order: sep_first` reorder (the `separation.mode: everywhere` arm already stands in as the blind separation-front-end baseline); AIR-RS-DB 4th corpus (IEEE-gated, honestly pending).
