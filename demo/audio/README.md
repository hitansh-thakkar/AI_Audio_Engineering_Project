# Hindi demo clips

Two Hindi multi-speaker clips for live demos of `preprocess_ui.py`. Our corpora have no
all-Hindi multi-speaker recording, so each clip joins real single-speaker utterances from
our evaluation slices, with 0.7 s of silence between turns. Speaker identities were checked
with ECAPA similarity: same voice ≥ 0.70, different voices ≤ 0.30.

| File | Speakers | Turns (in order) |
|---|---|---|
| `hindi_demo_2_speakers.wav` (25.4 s) | 2 | vaani_hi_003 (A), iv_hi_013 (B), vaani_hi_004 (A), vaani_hi_005 (A) |
| `hindi_demo_3_speakers.wav` (29.2 s) | 3 | vaani_hi_016 (A), vaani_hi_019 (B), vaani_hi_001 (C) |

Expected UI result (verified 25 Nov 2026, Gemini at temperature 0):

- **2-speaker clip:** 2 speakers with every turn attributed correctly. The Sarvam
  transcript matches the reference apart from punctuation and one inflection
  (हुई → हुआ). The guardrail result is PASSED (1.0).
- **3-speaker clip:** 3 speakers with every turn attributed correctly. The transcript
  matches the reference apart from punctuation and nasalisation. The guardrail result is
  PASSED_WITH_WARNINGS (0.5): the summary's opening sentence ("This multi-speaker
  transcript…") is accurate, but the lexical guardrail cannot match it to any word in the
  transcript.

These clips are a demonstration, not an evaluation. They are stitched turns without
overlap, so they do not exercise the separation stage. Benchmark numbers come from
`outputs/reports/`.
