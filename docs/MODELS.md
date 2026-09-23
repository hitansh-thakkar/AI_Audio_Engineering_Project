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
| **Default** | `pyannote/speaker-diarization-community-1` | Best open, self-hostable diarizer (pyannote.audio 4.0, Sept 2025). Improves speaker counting/assignment over 3.1; has an exclusive-speaker mode built to reconcile with ASR timestamps. |
| Ceiling (optional) | `pyannote/speaker-diarization-precision-2` (pyannoteAI API) | One-line swap; higher accuracy + faster; free credits. Use to benchmark the gap. |
| Legacy baseline | `pyannote/speaker-diarization-3.1` | The well-understood 3.1 pipeline, for comparison. |

**Gotchas.**
- **Gated.** Accept conditions on Hugging Face for *both* `speaker-diarization-community-1` **and** its `segmentation` dependency. Missing the segmentation agreement → 401. Fail loud.
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

## Stage 4 — ASR + alignment

| Choice | ID | Why |
| --- | --- | --- |
| Baseline | `openai/whisper-large-v3` | Universal multilingual baseline; native Indic support; standard tooling. |
| **Recommended (Hindi/Hinglish)** | `Trelis/tara` | Frontier open-weights Hindi + code-switch model. SOTA on AI4Bharat Vistaar Hindi (~12.06 WER mean); ~8.37 WER on Hindi–English code-switch (CS-FLEURS). **Loads exactly like whisper-large-v3.** Has a **mixed-code mode** rendering English in Latin + Hindi in Devanagari — exactly the code-switch output the brief wants. |
| Other Indic langs | `ai4bharat/indic-conformer-600m-multilingual` | IndicConformer — all 22 official languages, Conformer + hybrid CTC-RNNT. Needs AI4Bharat NeMo. Use for non-Hindi Indic audio and as a comparison. |
| Commercial ceiling (optional) | Sarvam **Saaras-v3** API | ~12.32 Vistaar mean; benchmark against it. |
| Alignment | `whisperx` (m-bain/whisperx) | wav2vec2 forced alignment → word-level timestamps; wraps Whisper + pyannote. |

**Do not** default to IndicWav2Vec — it's the 2022, 9-language model, superseded by IndicConformer.

**Critical gotcha — normalization.** Whisper's default text normalizer **strips Devanagari matras and inflates Devanagari WER.** For all WER/CER, use a **script-safe Indic normalizer** (NFC + `indic-nlp-library`). Report WER **and** CER, split by register: pure-Hindi, pure-English, code-switched. Getting this wrong makes every ASR number wrong.

**Optional improvement:** LoRA fine-tuning of Whisper/Tara on an IndicVoices slice is a documented, single-GPU-feasible way to cut Hinglish WER. Out of scope for the baseline; note as future work.

---

## Stage 5 — LLM post-processing

| Choice | ID | Why |
| --- | --- | --- |
| **Recommended** | A frontier API (Anthropic / OpenAI / Google) | This stage operates on *text* — a strong general model does code-switch normalization + summarization best. |
| Open / Indic-tuned | **Sarvam-M** (open weights) | If you want an open, Indic-tuned model on-box. |
| API alternative | Sarvam **Saaras/M** API | Managed Indic option. |

**Do NOT use Airavata / OpenHathi** — 2023–24, Hindi-only, GPT-3.5-era. Poor fit for 2026 cleanup/summarization.

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
  diarization:  pyannote/speaker-diarization-community-1
  separation:   speechbrain/sepformer-wsj02mix
  asr:          Trelis/tara                                # or openai/whisper-large-v3
  align:        whisperx
  llm:          anthropic                                  # or sarvam_m
```
