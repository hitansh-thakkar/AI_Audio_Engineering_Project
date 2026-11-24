# dhvani_eval — benchmark evaluation dataset

The curated eval set for the Dhvani pipeline, built **entirely from the real IndicVoices
audio** in `../indic_multispeaker_10/` by `build_dataset.py`. No speech is invented; every
sample is a slice or a mix of the original recordings. The build is seeded (`SEED=42`) and
reproducible — re-running regenerates identical output.

```bash
python dhvani_eval/build_dataset.py      # from the repo root
```

## Layout

```
dhvani_eval/
├── dataset_info.json          provenance, seed, git SHA, languages, required_corpora_status
├── build_dataset.py           the seeded generator (edit knobs at the top)
├── fetch_codeswitch.py        real Hindi–English CS + English registers (needs HF_TOKEN)
├── fetch_vaani.py             real Project Vaani slice (needs HF_TOKEN + Vaani license)
├── fetch_nirantar.py          Nirantar ingester (official GitHub slice, or opt-in HF mirror)
├── ingest_air_rs_db.py        AIR-RS-DB ingester (run after your IEEE DataPort download)
├── sequential/                10 multi-speaker clips, NO overlap
│   ├── audio/clip_XX.wav       (16 kHz mono)
│   ├── rttm/clip_XX.rttm       reference for DER / oracle diarization
│   ├── transcripts/clip_XX.txt human-readable reference
│   └── manifest.jsonl
├── sources/                   30 isolated single-speaker utterances
│   ├── audio/<clip>_s<n>_<lang>.wav
│   └── manifest.jsonl          text + language + speaker provenance
├── overlap_mixtures/          24 synthetic 2-speaker OVERLAPPING mixtures
│   ├── mix/mix_XXX.wav         the mixture fed to separation
│   ├── s1/mix_XXX.wav          ground-truth source 1 (time-aligned, mix == s1 + s2)
│   ├── s2/mix_XXX.wav          ground-truth source 2
│   ├── rttm/mix_XXX.rttm       two overlapping SPEAKER lines
│   └── manifest.jsonl          overlap region, overlap_ratio, sir_db, source texts
├── registers/                 real English + code-switch (added by fetch_codeswitch.py)
│   ├── english/audio/en_XXX.wav     40 real English utterances  -> pure-English WER
│   ├── codeswitch/audio/cs_XXX.wav  24 bilingual Hindi+English clips -> code-switch WER
│   ├── codeswitch/rttm/cs_XXX.rttm  two-turn RTTM per clip
│   └── manifest.jsonl               register, text (both scripts), segments, source
├── vaani/                     real Project Vaani slice (added by fetch_vaani.py)
│   ├── audio/vaani_<lang>_XX.wav    24 clips (Hindi/Bengali/Assamese) -> accent/dialect diversity + noise
│   └── manifest.jsonl               language, district, state, gender, transcript (sparse), speaker_id
└── nirantar/                  real Nirantar slice (added by fetch_nirantar.py)
    ├── audio/<id>.wav               24 clips (official E2E-prefix; audio-only) -> transcribed-speech corpus
    └── manifest.jsonl               id, provenance, duration, source
    # air_rs_db/ appears here after ingest_air_rs_db.py (your IEEE DataPort download)
```

## Required corpora (proposal) — status

| Corpus | Status | How |
| --- | --- | --- |
| **IndicVoices** | ✅ used | base of `sequential/`, `sources/`, `overlap_mixtures/` |
| **Project Vaani** | ✅ used | `vaani/` slice (24 clips) — `fetch_vaani.py` |
| **Nirantar** | ✅ used | `nirantar/` slice (24 clips) — `fetch_nirantar.py` (official E2E object-storage prefix; full set ~198 GB, audio-only slice) |
| **AIR-RS-DB** | ⏳ pending | IEEE DataPort subscription download (manual) → `ingest_air_rs_db.py` |

All slices stay within a **≤ 5 GB** total dataset budget (stream, never clone).

## What each subset is for

| Subset | Stage / metric it feeds | Notes |
| --- | --- | --- |
| `sequential/` | Diarization **DER/JER**; **oracle** diarization (point `diarization.rttm_dir` here); per-language **WER/CER** | Clean sequential turns → an easy DER floor; good for isolating downstream error. |
| `sources/` | Clean per-language **WER/CER**; atoms for mixtures | 3 languages: Assamese / Bengali / Hindi. |
| `overlap_mixtures/` | **Separation** (the whole reason the stage exists); **SI-SNRi / SDRi**; `separation.mode` ablation | `s1`/`s2` are the references; `mix == s1 + s2` (−90 dB residual, PCM-16 floor). 2-speaker only — matches SepFormer's design and the documented >2-speaker limitation. |
| `registers/english/` | **pure-English WER/CER** | 40 real English utterances (the register the base set had zero audio for). |
| `registers/codeswitch/` | **code-switch WER/CER** | 24 bilingual clips, real Hindi + real English speech, transcript spans Devanagari + Latin. These are **inter-sentential** (constructed by joining two real monolingual utterances), not single-speaker intra-sentential code-mix. |

### Overlap-mixture grid
Balanced sweep, all cross-language speaker pairs:
`overlap_ratio ∈ {0.25, 0.50, 0.75, 1.00}` × `sir_db ∈ {0, 5}` × 3 pairs = **24 mixtures**.
This range lets the ablation show separation helping on heavy overlap and the WSJ0-2Mix
domain-gap hurting elsewhere.

## Using it in the pipeline

```yaml
# oracle diarization against these references
diarization:
  source: oracle
  rttm_dir: dhvani_eval/sequential/rttm/

# separation eval reads overlap_mixtures/{mix,s1,s2}
```

## Registers (English + code-switch) — status

The base corpus (`indic_multispeaker_10`) is monolingual (as/bn/hi) with **no English**, so the
pure-English and code-switch registers can't come from it. `fetch_codeswitch.py` fills them with
**real speech** from the Hub (`commotion/Bilingual-Hindi-English-ASR-Lt-30`):

- **pure-English** — 40 real English utterances. ✅ real.
- **code-switch** — 24 clips whose transcript spans Devanagari + Latin, built from a real Hindi
  utterance + a real English utterance. ✅ real speech, but **inter-sentential** (two monolingual
  turns joined), not a single speaker switching mid-sentence.

**Remaining optional upgrade — intra-sentential code-mix.** For genuine single-speaker Hinglish
(one speaker alternating mid-utterance), accept the license on a gated MUCS code-mix set
(e.g. <https://huggingface.co/datasets/sajalmadan0909/mucs-hindi-english-codemix-asr>), set
`MUCS_REPO` in `fetch_codeswitch.py`, and re-run — it will prefer that for the code-switch register.
We do **not** synthesize this with TTS (that would compromise the benchmark with non-real audio).

## Provenance & licenses

Derived from **AI4Bharat IndicVoices** (`ai4bharat/IndicVoices`, valid split) — see that
dataset's license/terms before redistribution. Mixtures and slices inherit those terms.
`dataset_info.json` records the source, seed, and the git SHA of the build.

## `asr_whisper/` — faster-whisper ASR test set (14 Indic languages)

Built by `python dhvani_eval/build_asr_whisper.py` (needs `HF_TOKEN`). The set covers the Indic
languages in Whisper's vocabulary: `hi bn ta te mr gu kn ml pa ur as sa ne sd`. Odia and the
other scheduled languages are absent from Whisper, so they are excluded. Every clip has a human
reference transcript, is 1.5–30 s long, and is 16 kHz mono PCM-16. Selection is deterministic:
the first N matching rows in stream order, at most 3 per speaker, deduplicated across corpora
by speaker + text.

| Corpus | Source | Per language |
| --- | --- | --- |
| IndicVoices | `ai4bharat/IndicVoices` `valid`, streamed | 20 × all 14 |
| Vaani | `ARTPARK-IISc/Vaani`, transcribed rows only | up to 20 × 13 (no Sanskrit config); transcripts are sparse, gaps are listed in `summary.json` |
| Nirantar | official `AUDIOS.tgz.aa` streamed, joined to the LIL manifests (`data/nirantar/LIL.tgz`) | 20 × all 14 |
| registers | the existing `registers/` (English 40, Hindi–English code-switch 24), copied into `audio/registers/` | `register != "monolingual"` |
| AIR-RS-DB | — | excluded: the corpus has no transcripts (only R/S labels) |

Manifest rows (`asr_whisper/manifest.jsonl`, audio paths relative to `dhvani_eval/`) contain
`id, corpus, language` (a Whisper code), `register, audio, duration, text` (the reference),
`text_raw` (the verbatim text, or Vaani's text with its `{gloss}`/`[tag]` markup), plus speaker
and district metadata. Score `text` with the project's Indic normalizer (NFC +
indic-nlp-library), never Whisper's default normalizer.

```python
from faster_whisper import WhisperModel
import soundfile as sf
m = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
# pass a float32 array: faster-whisper 1.2.1's file decoder breaks on PyAV 19 (metadata_errors kwarg)
y, _ = sf.read("dhvani_eval/" + row["audio"], dtype="float32")
segs, info = m.transcribe(y, language=row["language"], beam_size=5)
hyp = " ".join(s.text.strip() for s in segs)   # language=None to test Whisper's LID instead
```

**Caveats.** Every reference must be in its language's expected script (for example, Vaani's
Urdu is Devanagari, so it is rejected), and each row records its `script`. **Sindhi** references
are Devanagari (as shipped by IndicVoices and Nirantar), but Whisper writes Sindhi in Perso-Arabic,
so transliterate before scoring or report `sd` separately. Whisper has very little training data
for as/sa/sd/ne/pa, so expect very high WER there.
