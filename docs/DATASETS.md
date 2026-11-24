# Datasets

What to use, how to access it, and how to fit it on a normal disk.

**All four corpora below are REQUIRED by the proposal** — IndicVoices, Project Vaani, Nirantar, and AIR-RS-DB (*"you are **required** to construct … using **the following** open-source Indian speech corpora"*). They are compulsory **inputs**, not optional extras.

**You do NOT download them in full.** This is not a from-scratch training project, and the full corpora run from hundreds of GB to multiple TB — far beyond a normal laptop. Each is used only as **small per-purpose slices**: (1) **evaluation slices** of real multi-speaker / code-switched audio, (2) **synthetic mixtures** from single-speaker clips for separation references (SI-SNRi/SDRi), (3) optional **fine-tuning slices**, and (4) AIR-RS-DB's **read-vs-spontaneous** labels. **Stream** slices where the Hub allows it. **Never clone a full corpus.** (The committed `dhvani_eval/` set is already one such curated slice, built from IndicVoices.)

> **HARD DISK BUDGET: ≤ 5 GB total for all datasets combined.** Do not exceed it. Suggested allocation (adjust as needed, keep the sum ≤ 5 GB):
>
> | Item | Budget |
> | --- | --- |
> | AIR-RS-DB (full download, ~225 MB) | ~0.25 GB |
> | IndicVoices slice (stream, one language) | ~1.0 GB |
> | Project Vaani slice (stream) | ~1.0 GB |
> | Nirantar slice (stream) | ~0.75 GB |
> | `dhvani_eval/` (already built) | ~0.05 GB |
> | HF stream/cache headroom | ~2.0 GB |
>
> Streaming (`load_dataset(..., streaming=True)`) avoids full downloads, but the HF cache (`~/.cache/huggingface`) can still accumulate parquet shards — clear it between fetches (`huggingface-cli delete-cache`) and keep an eye on the total. Separate from this, the *model* weights have their own budget (see `docs/LOCAL_SETUP.md`).

---

## IndicVoices (AI4Bharat) — REQUIRED (primary source)

- **HF:** `ai4bharat/IndicVoices` (gated — accept terms)
- ~23.7K hours of read/extempore/conversational speech, 22 languages, 400+ districts; ~11.2K hours transcribed (Dec 2025 update). **~745 GB full — do not pull in full.** Stream, or download **one language, one version** incrementally (v1–v5) and stop once you have enough clips.
- **Use:** real spontaneous/conversational eval clips; source clips for synthetic mixtures; fine-tuning slices. Its annotations mark code-switched words, useful for Hinglish eval.

## Project Vaani (IISc / ARTPARK) — REQUIRED (diversity & noise)

- **HF:** `ARTPARK-IISc/Vaani` (gated — accept terms)
- ~31,255 hours, 156K speakers, 165 districts, ~106 languages; ~2,043 hours transcribed. Geocentric — rich accent/dialect variation. **Multi-TB — never clone.** Stream a small per-district slice only.
- Also: **`ARTPARK-IISc/Vaani-Noise-Event-Dataset`** — clips with inline noise-event timestamps/tags. Handy for noise-robustness testing.
- **Use:** accent/dialect diversity in eval; realistic noise conditions.

## Nirantar (AI4Bharat) — REQUIRED (transcribed speech)

- **HF / GitHub:** `AI4Bharat/Nirantar`
- ~3,240 hours human-transcribed, 22 languages, 208 districts. **Hundreds of GB — stream a slice, don't clone.**
- **Note:** Nirantar is really a *continual-learning* benchmark (incremental language/domain episodes). This pipeline does **no** continual learning, so its distinctive structure is irrelevant here — treat it as plain transcribed speech for fine-tuning/eval.

## AIR-RS-DB (All India Radio Read and Spontaneous Speech Data Base) — required

- **Where:** IEEE DataPort — <https://ieee-dataport.org/documents/air-rs-db-all-india-radio-read-and-spontaneous-speech-data-base>
- **Author:** Sunil Kumar Kopparapu (TCS Research). Accompanies the paper *"A Novel Scheme to classify Read and Spontaneous Speech"* (arXiv:2306.08012).
- **Contents:** 1,028 audio clips (224.57 MB), **16-bit PCM @ 16 kHz** (already our canonical format) + `air_rs_db_auto-annotation.csv` (67.6 KB) + `INFO.pdf`.
- **Annotation:** read vs. spontaneous labels generated from a readability score (threshold τ ≈ 1.75). CSV = `filename : duration : readability_score : {R|S}` — `R` = read (news readers), `S` = spontaneous (panel guests). This is exactly the read/spontaneous distinction the proposal calls for.
- **Access — gated, NOT free:** *"This dataset requires an IEEE DataPort Subscription to access."* You must log in with an IEEE DataPort account/subscription (often available via an institution) and download `air-rs-db.zip` + the CSV yourself; it cannot be pulled with an HF token or anonymously.
- **Action:** download it under your IEEE access, drop it under `data/air_rs_db/`, and we parse the CSV into read/spontaneous eval slices. (AI4Bharat's **Shrutilipi** remains a useful *free* AIR-derived read-speech complement, but it does not replace AIR-RS-DB's R/S labels.)

---

## Building the evaluation set

You need three things:

1. **Real multi-speaker clips (5–8):** hand-picked 30–120 s panel/conversation clips with a manually corrected reference transcript + speaker labels (RTTM). Enough for DER + WER, not for training.
2. **Synthetic mixture set (separation refs):** overlay pairs of single-speaker IndicVoices/Vaani utterances at controlled SNRs to get ground-truth per-speaker sources → enables SI-SNRi/SDRi. This is the only way to get separation references from real data.
3. **Code-switch eval slice:** utterances with Hindi–English switching (IndicVoices code-switch annotations, or a public CS set) → per-register WER.

Keep `data/` out of version control; document exact download + slice commands in a `data/README.md`.

### ASR set: `dhvani_eval/asr_whisper/`

A faster-whisper WER/CER set over the 14 Whisper-supported Indic languages, drawn from IndicVoices, Vaani and Nirantar (AIR-RS-DB has no transcripts). See `dhvani_eval/README.md`.

### Built set: `dhvani_eval/` (in this repo)

`dhvani_eval/` is the concrete eval set, generated **entirely from the real IndicVoices audio**
in `indic_multispeaker_10/` by a seeded, reproducible script (`python dhvani_eval/build_dataset.py`).
See `dhvani_eval/README.md`. It covers items **1 and 2** above:

- `sequential/` — 10 multi-speaker clips (no overlap) + reference RTTM/transcripts → DER + oracle diarization + per-language WER.
- `sources/` — 30 isolated single-speaker utterances (as/bn/hi) → clean per-language WER; mixture atoms.
- `overlap_mixtures/` — 24 synthetic 2-speaker **overlapping** mixtures, each with time-aligned ground-truth sources (`mix == s1 + s2`) → SI-SNRi/SDRi + the `separation.mode` ablation. Grid: overlap_ratio {0.25,0.5,0.75,1.0} × sir {0,5} dB.
- `registers/` — **real** pure-English (40) + Hindi–English code-switch (24) utterances, fetched from `commotion/Bilingual-Hindi-English-ASR-Lt-30` by `fetch_codeswitch.py` → the **pure-English** and **code-switch** WER registers.

**Item 3 (code-switch) — now covered with real speech, with one caveat.** The base corpus is monolingual
(as/bn/hi, no English), so `fetch_codeswitch.py` pulls real English + real Hindi from the Hub and
fills both missing registers. The code-switch clips are **inter-sentential** (two real monolingual
turns joined), not single-speaker intra-sentential code-mix. For genuine intra-sentential Hinglish,
accept the license on a gated MUCS code-mix set, point `MUCS_REPO` at it, and re-run. We do not
synthesize code-switch audio with TTS (it would put non-real audio into the eval set).

## Access checklist (all four datasets are required)

- [ ] Hugging Face account + `HF_TOKEN`.
- [ ] Accept terms for the three required HF corpora: `ai4bharat/IndicVoices`, `ARTPARK-IISc/Vaani`, `AI4Bharat/Nirantar` — plus diarization: `pyannote/speaker-diarization-community-1` and `pyannote/segmentation`.
- [ ] AIR-RS-DB: log in to IEEE DataPort (**subscription** — not free) and download `air-rs-db.zip` (~225 MB) + `air_rs_db_auto-annotation.csv` into `data/air_rs_db/`.
- [ ] **Storage plan — HARD CAP ≤ 5 GB** for all datasets combined. Full corpora are hundreds of GB to multiple TB; a normal laptop cannot hold them. **Stream slices, never clone.** Only AIR-RS-DB (~225 MB) is downloaded whole. Clear the HF cache (`huggingface-cli delete-cache`) between fetches so streamed shards don't push you over budget.
