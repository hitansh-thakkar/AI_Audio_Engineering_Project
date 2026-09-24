# Models

Concrete, current model picks per stage — with the reasoning, the alternatives, and the gotchas. Model IDs are targets for the config registry; pin revisions when you wire them up.

> Verify each repo/version at build time — the field moves fast. Treat this as the vetted starting point, not gospel.

---

## Stage 1 — Preprocessing / denoise (optional)

| Choice | ID | Role |
| --- | --- | --- |
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

## Stage 5 — LLM post-processing (fully local via Ollama)

This stage only **cleans and summarizes text** that ASR already produced — it does not need a frontier model. A **local 7–9B model via [Ollama](https://ollama.com)** (`http://localhost:11434`) is plenty, and it keeps the pipeline zero-cost and offline. All paid/frontier/Sarvam-API backends have been removed from the default path; the `api` backend remains only as an optional, clearly-labeled paid escape hatch.

**Pick the model by hardware tier.** Size is configurable (`postprocessing.model`) — do not hardcode it. Ollama tags below were verified against the official Ollama library; re-check before pulling (tags change).

| Tier | Ollama model (tag) | Notes |
| --- | --- | --- |
| CPU-only / ≤6 GB VRAM | `qwen3:4b` **or** `phi4-mini` | Smallest sensible cleaners (~2.5 GB each). Runs on CPU or a 4 GB GPU. **Shipped default is `qwen3:4b`** (tuned for a 4 GB-VRAM laptop with limited disk). |
| 8 GB VRAM *(reference default tier)* | `qwen3:8b` | ~5.2 GB. Apache-2.0, strong multilingual incl. Hindi. The recommended default if you have the VRAM/disk. |
| 12–16 GB VRAM | `gemma3:12b` **or** `qwen3:14b` | Gemma 3 12B (~8.1 GB, good summarization/multilingual, Gemma license) or Qwen3 14B (~9.3 GB, Apache-2.0). |
| 24 GB+ VRAM | `gemma3:27b` **or** `qwen3:32b` | Largest local options (~17 GB / ~20 GB). **Sarvam-M only if it fits** (see caveat below). |

**Pull + run:**
```bash
ollama pull qwen3:4b          # or the tag for your tier
ollama run qwen3:4b "hi"      # sanity check the local server
```

**Licenses:** Qwen3 = **Apache 2.0**; Gemma 3 = Gemma license (free, some use terms); Phi-4-mini = MIT. All open-weights, local, no per-token cost.

> ⚠️ **Sarvam-M on Ollama is unverified.** There is **no official `sarvam-m` tag in the Ollama library**. Only community uploads exist (e.g. `mashriram/sarvam-m`, a ~25 GB q8 build of the 24B model) — unofficial, unvetted, and only viable on a 24 GB+ setup. Treat it as optional/experimental, not a default. Prefer `qwen3:32b` / `gemma3:27b` at that tier.

**Do NOT use Airavata / OpenHathi** — 2023–24, Hindi-only, GPT-3.5-era. Poor fit for 2026 cleanup/summarization.

**Optional PAID escape hatch (`postprocessing.backend: api`).** A frontier or Sarvam cloud API can be wired behind the same interface if you *choose* to and accept the cost. It is off by default, requires its own API key, and is **not** part of the free/local stack. Do not make it a default value.

**Prompt requirements:**
- Force structured output: **executive summary + keyword tags + speaker-separated dialogue**.
- Normalize code-switching; repair noise-induced word dropouts *conservatively*.
- Extract core topics (e.g., market-rate disputes, weather updates).
- **Guardrail:** forbid introducing any fact not in the transcript. Add a factual-consistency check (entailment or keyword-overlap vs. source) and flag violations.

---

## Summary registry (for `config/default.yaml`)

```yaml
models:
  denoise:      htdemucs                                   # optional; off by default
  diarization:  pyannote/speaker-diarization-community-1   # free + local (precision-2 = optional paid API)
  separation:   speechbrain/sepformer-wsj02mix
  asr:          large-v3                                   # faster-whisper CT2 (int8); Tara-CT2 for Hindi/Hinglish
  align:        whisperx
  llm:          qwen3:4b                                   # local Ollama; scale per tier (qwen3:8b / gemma3:12b / qwen3:32b ...)
```

Everything above is open-weights running locally. The only credential is a free HF token for pyannote's gated (not paid) licenses.
