# Datasets

What to use, how to access it, and the one dataset in the brief you should not trust as named.

**How these are used.** This is not a from-scratch training project. Corpora serve three purposes: (1) small **evaluation slices** of real multi-speaker / code-switched audio, (2) **synthetic mixtures** built from single-speaker clips to get separation references (SI-SNRi/SDRi), and (3) optional **fine-tuning slices**. Do not attempt to train on the full multi-thousand-hour corpora.

---

## IndicVoices (AI4Bharat) — primary

- **HF:** `ai4bharat/IndicVoices`
- ~23.7K hours of read/extempore/conversational speech, 22 languages, 400+ districts; ~11.2K hours transcribed (Dec 2025 update). ~745 GB full; download per-language incrementally (v1–v5).
- **Use:** real spontaneous/conversational eval clips; source clips for synthetic mixtures; fine-tuning slices. Its annotations mark code-switched words, useful for Hinglish eval.

## Project Vaani (IISc / ARTPARK) — diversity & noise

- **HF:** `ARTPARK-IISc/Vaani` (gated — accept terms)
- ~31,255 hours, 156K speakers, 165 districts, ~106 languages; ~2,043 hours transcribed. Geocentric — rich accent/dialect variation.
- Also: **`ARTPARK-IISc/Vaani-Noise-Event-Dataset`** — clips with inline noise-event timestamps/tags. Handy for noise-robustness testing.
- **Use:** accent/dialect diversity in eval; realistic noise conditions.

## Nirantar (AI4Bharat) — extra transcribed speech

- **HF / GitHub:** `AI4Bharat/Nirantar`
- ~3,240 hours human-transcribed, 22 languages, 208 districts.
- **Note:** Nirantar is really a *continual-learning* benchmark (incremental language/domain episodes). This pipeline does **no** continual learning, so its distinctive structure is irrelevant here — treat it as plain transcribed speech for fine-tuning/eval.

## ⚠️ "AIR-RS-DB (All India Radio Dataset)" — verify before use

The brief lists an "AIR-RS-DB" with "pre-annotated read-vs-spontaneous segments." **No canonical public dataset by that name with that structure could be found.** The brief's empty `[]` citation brackets cluster around exactly this item — a sign it may be a placeholder/hallucinated reference.

- **The real AIR-derived resource is AI4Bharat's Shrutilipi** — paired audio-text mined from All India Radio news bulletins (read speech).
- **Action:** confirm with the instructor which dataset is meant. If they can't point to a specific one, substitute **Shrutilipi** (for read speech) and use IndicVoices' conversational split for the spontaneous contrast.

---

## Building the evaluation set

You need three things:

1. **Real multi-speaker clips (5–8):** hand-picked 30–120 s panel/conversation clips with a manually corrected reference transcript + speaker labels (RTTM). Enough for DER + WER, not for training.
2. **Synthetic mixture set (separation refs):** overlay pairs of single-speaker IndicVoices/Vaani utterances at controlled SNRs to get ground-truth per-speaker sources → enables SI-SNRi/SDRi. This is the only way to get separation references from real data.
3. **Code-switch eval slice:** utterances with Hindi–English switching (IndicVoices code-switch annotations, or a public CS set) → per-register WER.

Keep `data/` out of version control; document exact download + slice commands in a `data/README.md`.

## Access checklist

- [ ] Hugging Face account + `HF_TOKEN`.
- [ ] Accept terms: `ARTPARK-IISc/Vaani`, `pyannote/speaker-diarization-community-1`, `pyannote/segmentation`.
- [ ] Confirm "AIR-RS-DB" with instructor (or default to Shrutilipi).
- [ ] Storage plan — full corpora are hundreds of GB; pull only the slices you need.
