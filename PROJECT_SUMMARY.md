# Slide 1 — Dhvani

## Who spoke what, and when?

Team: HTMN

Members:

- Hitansh Amrish Thakkar (2024A7PS0611G)
- Megh Brijesh Shah (2024A7PS0476G)
- Neev Lalit Rathod (2024A7PS0487G)
- Tanuj Dinesh Yadav (2024AAPS0740G)

---

# Slide 2 — What We Are Building

A system to process real Indian multi-speaker audio and produce:

- speaker-attributed timeline
- clean transcript
- summary/report of the conversation

Problem: real conversations are messy

- multiple speakers overlap
- Hindi-English and regional speech mix together
- background noise is common
- standard ASR systems struggle

---

# Slide 3 — Milestones

### Milestone 1: Blind Source Separation

- Separate overlapping speech into cleaner speaker streams
- Improve the audio before transcription

### Milestone 2: Acoustic Speaker Diarization

- Detect who is speaking and when
- Group similar voices using unsupervised clustering
- Produce timeline labels like Speaker A / Speaker B

### Regional and Code-Switched Transcription

- Convert diarized speech into text
- Handle Hindi-English and regional variations

### Semantic Post-Processing with LLM

- Clean the raw transcript
- Fix noisy text and format it properly
- Generate a summary and structured report

---

# Slide 4 — What We Have Built So Far

We built a working multi-stage pipeline:

Audio → Preprocess → Speaker Diarization → Overlap Separation → ASR → LLM Cleanup → Final Report

Current progress includes:

- audio preprocessing
- speaker detection and clustering
- overlap-aware separation logic
- timeline output with speaker labels
- evaluation framework and end-to-end execution

---

# Slide 5 — What Is Left

Next steps:

- strengthen multilingual ASR for Indian code-switched speech
- better LLM cleaning and structured summaries
- complete benchmarking and ablation study
