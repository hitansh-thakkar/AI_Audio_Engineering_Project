# Models

Concrete, current model picks per stage — with the reasoning, the alternatives, and the gotchas. Model IDs are targets for the config registry; pin revisions when you wire them up.

> Verify each repo/version at build time — the field moves fast. Treat this as the vetted starting point, not gospel.

---

## Stage 1 — Preprocessing / denoise (optional)

| Choice | ID | Role |
| --- | --- | --- |
| **Implemented (default-available)** | high-pass + spectral-gating denoise + BS.1770 LUFS | Dependency-light conditioning for mobile-mic / market-noise audio. `highpass_hz`, `denoise: spectral`, `normalize: lufs` (pyloudnorm). |
| Demucs (optional) | `htdemucs` (via `demucs` pkg) | Isolate the **vocal stem** from background music. |
| DNS denoiser (optional) | e.g. `speechbrain/sepformer-dns4` or a DNS-Challenge model | Speech enhancement / noise suppression. |

**Reasoning.** Demucs is a **music** source-separation model (vocals / drums / bass / other). Its only honest use here is stripping background music from a voice — it **cannot separate two speakers**. The brief's listing of Demucs as a speaker-separation front-end is a category error; we use it (if at all) purely as a denoiser. Keep denoise off by default and expose it as a toggle for the ablation.

---

## Stage 2 — Diarization

| Choice | ID | Why |
| --- | --- | --- |
| **Default (free, local)** | `pyannote/speaker-diarization-community-1` | Best open, self-hostable diarizer (pyannote.audio 4.0, Sept 2025). Runs **locally on CPU**. Free: needs only a free HF token + accepting two click-through licenses. Improves speaker counting/assignment over 3.1; has an exclusive-speaker mode built to reconcile with ASR timestamps. |
| ⚠️ Ceiling — **OPTIONAL PAID API** | `pyannote/speaker-diarization-precision-2` (pyannoteAI cloud API) | Higher accuracy + faster, but it is a **paid** hosted API (per-use credits, not local inference). **Not part of the free/default path.** Use only if you deliberately want to benchmark the accuracy gap and accept the cost. |
| Legacy baseline (free, local) | `pyannote/speaker-diarization-3.1` | The well-understood 3.1 pipeline, local, for comparison. |

**REQUIRED — hand-applied unsupervised clustering (proposal Milestone 2).** The proposal mandates clustering speaker embeddings yourself with **"specifically Spectral Clustering or Gaussian Mixture Models (GMMs)."**

**As implemented:** embeddings come from **SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`, open — no gated licence, `diarization.embedding: ecapa`; MFCC-stats is the offline fallback). These feed **our own Spectral/GMM clustering** (`diarization.clustering: spectral | gmm`) with cosine affinity and cosine-silhouette speaker counting. pyannote is used **only as an optional overlap detector** (`diarization.overlap_detector: pyannote`) to populate overlap regions — not as the clusterer. pyannote (or any framework) is only an *example* in the proposal ("a framework like pyannote.audio"); the **clustering method is the compulsory part**, and ECAPA→Spectral/GMM satisfies it. For long clips, per-chunk labels are reconciled into global speakers by clustering their voiceprints (`src/diarization/linking.py`).

**Gotchas.**
- **Gated but free.** Accept the click-through conditions on Hugging Face for *both* `speaker-diarization-community-1` **and** its `segmentation` dependency, then use a free `HF_TOKEN`. Gated ≠ paid. Missing the segmentation agreement → 401. Fail loud.
- Python **3.10+** required for pyannote.audio 4.0.
- Runs on CPU by default; send to GPU with `pipeline.to(torch.device("cuda"))`.
- Supports `num_speakers` / `min_speakers` / `max_speakers`.

**Realistic accuracy.** DER on hard conversational benchmarks: ~23% (MSDWild), ~27% (CALLHOME part 2), ~20% (DIHARD 3). Expect **20–45% DER on chaotic real radio** — report it as a finding, and use oracle diarization to isolate downstream error.

---

## Stage 3 — Separation (applied to overlapped regions)

| Choice | ID | Why |
| --- | --- | --- |
| **Default (large)** | `speechbrain/sepformer-wsj02mix` | Strong transformer separator, 22.4 dB SI-SNRi on WSJ0-2Mix. |
| Lighter variant | `speechbrain/resepformer-wsj02mix` | Smaller (18.6 dB) — useful for the latency/memory comparison. |
| Baseline | Conv-TasNet (SpeechBrain / Asteroid) | The 2019 baseline the brief names; lighter, for comparison. |

**Reasoning & hard limits (build these in as known limitations):**
- Checkpoints are **8 kHz, single-channel, exactly 2 speakers, trained on synthetic mixtures.** Resample 16 kHz → 8 kHz in, 8 kHz → 16 kHz out, and document the quality cost.
- **Domain gap:** trained on clean WSJ0-2Mix; real reverberant radio degrades output. This is *why* we apply separation only on overlap regions, not blind. The `everywhere` mode exists to demonstrate the degradation in the ablation.
- **>2 speakers overlapping:** out of the model's design. Document behavior; don't fake a solution.
- If time permits, fine-tune on **Libri2Mix** (freely available) — do **not** attempt WSJ0-2Mix training (needs LDC-licensed WSJ0).
- Metrics: **SI-SNRi, SDRi** against reference sources (use synthetic mixtures for eval).

---

## Stage 4 — ASR + alignment (fully local)

**Runtime: `faster-whisper` (CTranslate2) with `int8` quantization.** This is the key change that makes ASR run on a consumer PC: int8 CTranslate2 drops whisper-large-v3 to **~2–4 GB VRAM** (fits a 4 GB laptop GPU) and also runs on **CPU**. `faster-whisper` auto-downloads the CT2 conversion of standard sizes (`large-v3`, `medium`, `small`, `distil-large-v3`) from the free Systran HF repos; set `compute_type: int8` (CPU / low-VRAM) or `int8_float16` (GPU).

| Choice | ID / how to load | Why |
| --- | --- | --- |
| **Baseline (default runtime)** | `large-v3` via faster-whisper (`WhisperModel("large-v3", compute_type="int8")`) | Universal multilingual baseline; native Indic support. Runnable out of the box, no conversion, no token. |
| **Recommended (Hindi/Hinglish)** | `Trelis/tara` — **convert to CTranslate2** (`ct2-transformers-converter`) and point `asr.model` at the CT2 dir, *or* run via `transformers` with int8 | Frontier open-weights Hindi + code-switch model. SOTA on AI4Bharat Vistaar Hindi (~12.06 WER mean); ~8.37 WER on Hindi–English code-switch (CS-FLEURS). Built on whisper-large-v3, so it converts cleanly to CT2. Has a **mixed-code mode** (English in Latin + Hindi in Devanagari) — exactly the code-switch output the brief wants. |
| Fallbacks (weak hardware / tight disk) | `distil-large-v3`, `medium`, or `small` via faster-whisper | Much smaller + faster with modest accuracy loss. Use when VRAM/RAM/disk is tight (e.g. a 4 GB-VRAM laptop with a few GB of disk). |
| Other Indic langs (optional, heavy) | `ai4bharat/indic-conformer-600m-multilingual` | IndicConformer — all 22 official languages, Conformer + hybrid CTC-RNNT. Needs AI4Bharat NeMo (heavy install). Local + free, but optional; use for non-Hindi Indic audio and as a comparison. |
| Alignment | `whisperx` (m-bain/whisperx) | wav2vec2 forced alignment → word-level timestamps; wraps Whisper + pyannote. Local. |

All of the above are **open weights running on your machine** — no API keys, no per-token cost. (The former Sarvam Saaras-v3 API option has been removed; it is a paid cloud service and is off the local path.)

**Do not** default to IndicWav2Vec — it's the 2022, 9-language model, superseded by IndicConformer.

**Tara → CTranslate2 note.** No official prebuilt CT2 build of `Trelis/tara` is published as of this writing (unverified — check HF before assuming one exists). Convert it yourself with `ct2-transformers-converter --model Trelis/tara --quantization int8 --output_dir models/tara-ct2`, or run Tara through `transformers` with int8 as a slower fallback. The config default is therefore `large-v3` (works immediately); switch `asr.model` to your Tara CT2 dir once converted.

**Critical gotcha — normalization.** Whisper's default text normalizer **strips Devanagari matras and inflates Devanagari WER.** For all WER/CER, use a **script-safe Indic normalizer** (NFC + `indic-nlp-library`). Report WER **and** CER, split by register: pure-Hindi, pure-English, code-switched. Getting this wrong makes every ASR number wrong.

**Optional improvement:** LoRA fine-tuning of Whisper/Tara on an IndicVoices slice is a documented, single-GPU-feasible way to cut Hinglish WER. Out of scope for the baseline; note as future work.

---

## Stage 5 — LLM post-processing (default: Google Gemini API, free tier)

This stage only **cleans and summarizes text** that ASR already produced — it does not need a frontier model, but a capable one helps with Indic/code-switched cleanup. The default is the **Google Gemini API free tier**, which keeps the pipeline **zero-cost** while giving strong multilingual quality. It is a **cloud call**, so stage 5 is the one part of the pipeline that is not offline — a free API key, not a paid service. A fully-local Ollama backend stays available for offline runs; the paid `api` backend remains an optional escape hatch.

### Default — Gemini API (free tier)

| Choice | ID (`postprocessing.model`) | Why |
| --- | --- | --- |
| **Default** | `gemini-2.0-flash` | Fast, free-tier, strong multilingual incl. Hindi/Indic + code-switch. Ample for cleanup/summarization. |
| Alt (quality) | `gemini-2.5-flash` / `gemini-2.5-pro` | Higher quality if the free-tier limits allow; verify current free-tier availability and rate limits. |

- **Key:** free from <https://aistudio.google.com/apikey>; read from `GEMINI_API_KEY` (env), never commit. Client: `google-genai`.
- **Verify before wiring:** model tags and free-tier rate limits change — re-check AI Studio. Handle rate-limit / quota errors as a graceful degradation (fall back to a minimal report, note it in the manifest).
- Size is configurable (`postprocessing.model`) — do not hardcode it.

### Optional — fully-local Ollama (offline, zero cloud)

Prefer a 100%-offline run? Set `postprocessing.backend: ollama` and pick a model by hardware tier. Ollama tags were verified against the official library; re-check before pulling (tags change).

| Tier | Ollama model (tag) | Notes |
| --- | --- | --- |
| CPU-only / ≤6 GB VRAM | `qwen3:4b` **or** `phi4-mini` | Smallest sensible cleaners (~2.5 GB each). Runs on CPU or a 4 GB GPU. Good fit for a 4 GB-VRAM laptop. |
| 8 GB VRAM | `qwen3:8b` | ~5.2 GB. Apache-2.0, strong multilingual incl. Hindi. |
| 12–16 GB VRAM | `gemma3:12b` **or** `qwen3:14b` | Gemma 3 12B (~8.1 GB, good summarization/multilingual, Gemma license) or Qwen3 14B (~9.3 GB, Apache-2.0). |
| 24 GB+ VRAM | `gemma3:27b` **or** `qwen3:32b` | Largest local options (~17 GB / ~20 GB). **Sarvam-M only if it fits** (see caveat below). |

```bash
ollama pull qwen3:4b          # or the tag for your tier
ollama run qwen3:4b "hi"      # sanity check the local server
```

**Licenses (Ollama path):** Qwen3 = **Apache 2.0**; Gemma 3 = Gemma license (free, some use terms); Phi-4-mini = MIT. All open-weights, local, no per-token cost.

> ⚠️ **Sarvam-M on Ollama is unverified.** There is **no official `sarvam-m` tag in the Ollama library**. Only community uploads exist (e.g. `mashriram/sarvam-m`, a ~25 GB q8 build of the 24B model) — unofficial, unvetted, and only viable on a 24 GB+ setup. Treat it as optional/experimental, not a default. Prefer `qwen3:32b` / `gemma3:27b` at that tier.

**Do NOT use Airavata / OpenHathi** — 2023–24, Hindi-only, GPT-3.5-era. Poor fit for 2026 cleanup/summarization.

**Optional PAID escape hatch (`postprocessing.backend: api`).** A frontier or Sarvam cloud API can be wired behind the same interface if you *choose* to and accept the cost. It is off by default, requires its own API key, and is **not** part of the free stack. Do not make it a default value.

**Prompt requirements:**
- Force structured output: **executive summary + keyword tags + speaker-separated dialogue**.
- Normalize code-switching; repair noise-induced word dropouts *conservatively*.
- Extract core topics (e.g., market-rate disputes, weather updates).
- **Guardrail:** forbid introducing any fact not in the transcript. Add a factual-consistency check (entailment or keyword-overlap vs. source) and flag violations.

---

## Summary registry (for `config/default.yaml`)

```yaml
models:
  denoise:      spectral                                   # dependency-light; demucs/dns optional
  embedding:    speechbrain/spkrec-ecapa-voxceleb          # ECAPA speaker embeddings (open) for clustering
  overlap:      pyannote/speaker-diarization-community-1   # overlap detector only (gated-but-free); clustering is hand-applied
  separation:   speechbrain/sepformer-wsj02mix             # config-driven: whamr (noise) / 16 kHz / convtasnet
  asr:          large-v3                                   # faster-whisper CT2 (int8); Tara-CT2 for Hindi/Hinglish (post-midsem)
  align:        whisperx
  llm:          gemini-2.5-flash                           # Gemini API free tier (default); optional local Ollama (post-midsem)
```

The acoustic stack is open-weights running locally; the LLM stage calls the Gemini API free tier by default (optional local Ollama for offline). The two credentials are both free: an HF token for pyannote's gated (not paid) licenses, and a Gemini API key.
