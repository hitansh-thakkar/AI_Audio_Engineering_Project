# Midsem Presentation Spec — for the pptx-generator agent

**Read this file completely before generating the deck.** It is the single source of truth for
*what goes in the midsem presentation*. You (the pptx agent) should turn each slide block below
into one slide. Pull real numbers/figures from the repo paths cited — **do not invent data**.

---

## 0. Deck metadata
- **Title:** Dhvani — "Who Spoke What and When"
- **Subtitle:** A multi-stage pipeline for chaotic, multi-speaker, code-switched Indian audio — Midsem Progress
- **Course framing:** Advanced Computational Speech Engineering. System-engineering + measurement project (we chain pretrained models; we do not train from scratch).
- **Suggested length:** 14–16 slides, 16:9.
- **Tone:** technical, honest about what is real vs. stubbed. No overclaiming.
- **Theme:** clean, high-contrast, one idea per slide. Monospace for code/paths. Accent color for the "diarization-first" idea and the compulsory-compliance ticks.
- **Repo facts to trust:** pipeline runs end-to-end today; 17 unit/integration tests pass; `ruff` + `mypy` clean. Preliminary real diarization DER ≈ 0.30 on the eval set.

---

## 1. Title slide
- Project title + subtitle (above).
- Team name: HTMN 
- Names and Roll numbers: 
  (Hitansh Amrish Thakkar(2024A7PS0611G)
  Megh Brijesh Shah(2024A7PS0476G) 
  Neev Lalit Rathod(2024A7PS0487G)
  Tanuj Dinesh Yadav(2024AAPS0740G)).
- One-line hook: *"Who spoke what, and when — turning overlapping, code-switched Indian radio into a clean, speaker-attributed report."*
- Speaker notes: state this is the midsem checkpoint covering Milestones 1 & 2 (the acoustic front-end) with the full pipeline already running end-to-end on stubs for the text stages.

## 2. The problem
- Real Indian audio = **chaotic**: multiple speakers overlapping, **code-switching** (Hindi–English / regional), low-cost mics, village/market noise.
- Off-the-shelf ASR collapses under these conditions.
- Goal: a **"Who Spoke What and When"** pipeline → clean, speaker-attributed, summarized transcript report.
- Visual: a "messy waveform + overlapping speaker bands" cartoon → arrow → tidy report card.
- Speaker notes: emphasize this is a *systems* problem (chaining specialists) not a *modeling* problem.

## 3. The 5 capabilities we must demonstrate (from the proposal)
Bullet list, each with a one-word status chip:
1. Isolate overlapping waveforms (deep source separation) — *front-end built, graceful*
2. Unsupervised clustering diarization, no speaker templates — **DONE (Spectral/GMM)**
3. Regional + code-switched ASR — *post-midsem (stubbed)*
4. LLM cleaning/summarizing of fragmented multilingual text — *post-midsem (stubbed)*
5. Manage the pipeline + measure **error propagation** — *harness built; ablation post-midsem*
- Speaker notes: capabilities 1,2,5 are the midsem focus; 3,4 are the final.

## 4. Architecture — diarization-first, separation-surgical
- Show the data-flow diagram (source: `README.md` "What it does" ASCII block and `docs/ARCHITECTURE.md`):
  `audio → preprocess → DIARIZE → separate(overlap only) → ASR+align → LLM → report`
- Key design decision to highlight: **we invert the proposal's separate→diarize order** to **diarize→(separate overlaps)→transcribe**.
- Why (one line each): off-the-shelf separators are 8 kHz / 2-speaker / synthetic → inject artifacts on real audio; diarization estimates the unknown speaker count; it also makes the error-propagation study clean (swap in oracle diarization).
- Note: the literal order is kept as a `run.order: sep_first` baseline for the final ablation — so we satisfy the rubric either way.
- Visual: the pipeline diagram with the DIARIZE box emphasized and a small "separate only here" bracket over overlap regions.

## 5. Proposal compliance — what is COMPULSORY vs optional
Two-column table (source of truth: `CLAUDE.md` rule 7, `PLAN.md`):
- **Compulsory (we honor exactly):**
  - All four datasets: IndicVoices, Vaani, Nirantar, AIR-RS-DB (as ≤5 GB slices)
  - Hand-applied **Spectral Clustering / GMM** on embeddings ("specifically")
  - Deliverable formats: Milestone-2 **timeline JSON**, final report structure
- **Optional (examples → we picked stronger):** Conv-TasNet/Demucs, pyannote, IndicWav2Vec, Airavata/OpenHathi → we use SepFormer, pyannote, faster-whisper/Tara, Gemini.
- Speaker notes: the proposal says "like/such as" for models (examples) but "required/specifically" for datasets + clustering — so our model swaps are compliant.

## 6. Dataset — what we built (`dhvani_eval/`)
- Built from real **IndicVoices** audio by a seeded, reproducible script (`dhvani_eval/build_dataset.py`), plus real English/code-switch from the Hub (`fetch_codeswitch.py`).
- Pull exact counts from `dhvani_eval/dataset_info.json` and the manifests. Current composition:
  - `sequential/` — 10 multi-speaker clips (+ RTTM + transcripts)
  - `sources/` — 30 isolated single-speaker utterances (as/bn/hi)
  - `overlap_mixtures/` — 24 synthetic 2-speaker overlaps + ground-truth sources (`mix == s1 + s2`)
  - `registers/` — 40 real English + 24 Hindi–English code-switch clips
  - `vaani/` — 24 real Project Vaani clips (as/bn/hi) with diversity metadata (district/state/gender)
  - `nirantar/` — 24 real Nirantar clips (official E2E-prefix slice)
  - **224 wav files total; 176 usable inputs + 48 reference stems.**
- Languages: Assamese, Bengali, Hindi, English + code-switch; Vaani adds accent/dialect diversity.
- Visual: a composition bar/pie (generate from `dataset_info.json` + manifest line counts).
- Speaker notes: everything is real speech or overlays of real speech — nothing synthesized by TTS.

## 7. Dataset — the four required corpora & disk budget
- All four are REQUIRED; we use **slices, never full downloads** (hundreds of GB–TB otherwise).
- Status table (source: `dhvani_eval/dataset_info.json` → `required_corpora_status`):
  - IndicVoices — ✅ used (base of dhvani_eval)
  - Project Vaani — ✅ used (24-clip slice via `fetch_vaani.py`)
  - Nirantar — ✅ used (24-clip official E2E-prefix slice via `fetch_nirantar.py`; full set ~198 GB)
  - AIR-RS-DB — ⬇️ pending (IEEE DataPort subscription; manual download → `ingest_air_rs_db.py`)
- **3 of 4 required corpora are physically in the eval set**; AIR-RS-DB is gated on IEEE access only.
- **Hard disk budget: ≤ 5 GB total** (actual `dhvani_eval/` ≈ 46 MB).
- Speaker notes: Nirantar has no official HF dataset (GitHub + E2E storage); we slice the official source within budget. AIR-RS-DB is a manual IEEE download; the ingester is plug-and-play.

## 8. Milestone 1 — Separation (status)
- Stage implemented behind a swappable interface; modes `none | overlap | everywhere`.
- Real SepFormer/Conv-TasNet (8 kHz, 2-speaker) plug in when SpeechBrain is installed; currently **graceful passthrough** with recorded notes (errors flow, not halt).
- Eval harness ready: **SI-SNRi / SDRi** implemented & unit-tested on the 24 overlap mixtures (ground-truth `s1`/`s2`).
- Visual: overlap-region bracket on a waveform; a "mix → [SepFormer] → s1,s2" box.
- Speaker notes: separation is applied *surgically* on overlap only; blind "everywhere" is kept for the ablation.

## 9. Milestone 2 — Diarization + the COMPULSORY clustering (the centerpiece of midsem)
- Pipeline: energy VAD → windowed segments → **speaker embeddings** → **hand-applied Spectral Clustering / GMM** (sklearn) → merged speaker turns. No templates.
- Speaker count estimated automatically (silhouette for spectral, BIC for GMM).
- **Timeline JSON** emitted (the required M2 deliverable) — show a snippet (`outputs/<clip>/timeline.json` shape: `{start, end, speaker}`).
- Oracle mode loads ground-truth RTTM for the error-propagation study.
- Code pointers (for credibility): `src/diarization/clustering.py`, `diarize.py`.
- Visual: scatter of 2-D embeddings colored by cluster (generate a small t-SNE/PCA from a clip's segment embeddings).

## 10. Preliminary results — diarization (REAL numbers)
- Setup: real diarization vs. oracle RTTM on the 10 `sequential/` clips; **no collar**; **MFCC-fallback embeddings** (pyannote not yet installed).
- Numbers (recompute to confirm; current run):
  - **mean DER ≈ 0.30**, median ≈ 0.295
  - speaker-count accuracy ≈ 2/10 (mixed: under-counts the 3-speaker clips, and the k≥2 floor over-splits single-speaker audio — see Risks)
- Framing: this sits squarely in the **20–45% DER expected on chaotic audio** (per `docs/MODELS.md`) — a *finding about cascade fragility*, not a failure.
- **Clear next lever:** swap MFCC → pyannote embeddings (expected large DER drop).
- Visual: per-clip DER bar chart (generate by running the diarization-eval snippet; see slide 16 "how to regenerate").
- Speaker notes: the value is that the measurement harness works and produces honest numbers now.

## 11. Preliminary results — separation & clustering sanity
- Clustering validated on synthetic 2-speaker embeddings: **adjusted Rand > 0.9** for both Spectral and GMM (`tests/test_clustering.py`).
- SI-SNRi/SDRi verified: perfect-estimate > mixture baseline (`tests/test_metrics.py`).
- DER metric verified: identical ref/hyp → 0.0, label-permutation invariant (Hungarian mapping).
- **End-to-end audit:** the full pipeline was run across every subset (real sequential, oracle mixture, Vaani, Nirantar) — all 5 stages execute, outputs (report/timeline/manifest) are well-formed, and the oracle-mixture run correctly surfaces 1 overlap region and triggers separation. All audit checks pass.
- Visual: a small green "17/17 tests pass + audit OK" badge + a table of the key asserted properties.

## 12. System benchmark — instrumentation (Deliverable 2 groundwork)
- Every stage is wrapped in a timer + peak-memory tracker; results land in a per-run `RunManifest` (git SHA, config hash, seeds, per-stage wall-time + peak MB, degradation notes).
- Show a real `manifest.json` stage table from an actual run (e.g. preprocessing ≈ 5.7 s, diarization ≈ 3.0 s on a ~15 s clip; memory via `tracemalloc`).
- Note the honest limitation: `tracemalloc` captures Python-heap peak, not native/torch — to be refined.
- Visual: the stage timing/memory table (pull from a fresh `outputs/<clip>/manifest.json`).

## 13. Engineering quality & reproducibility
- Typed dataclass contracts between stages (`src/types.py`); every stage swappable via `config/default.yaml` (no hardcoded model IDs).
- **Graceful degradation** everywhere (a failing/absent stage logs + passes through — because we *measure* error propagation).
- `ruff` + `mypy` clean; 17 tests; seeded runs; secrets only via `.env` (free `HF_TOKEN` + `GEMINI_API_KEY`).
- Fully local/free run (LLM stage uses the Gemini free tier; optional local Ollama retained for the "localized" requirement).
- Visual: a "contracts + config + manifest" triad diagram.

## 14. What's done vs. what's next (the 40/60 split)
- **Done (midsem, ~40%):** scaffold + contracts + instrumentation; preprocessing; diarization with compulsory Spectral/GMM clustering + timeline JSON; separation stage + metrics; curated dataset; end-to-end run on stubs; preliminary DER.
- **Next (final, ~60%):** code-switched ASR + WhisperX alignment (per-register WER); Gemini LLM report + guardrail; pyannote/SepFormer real backends; **the ablation study** (final WER vs. config × {real,oracle} diarization × separation modes) — the money plot; benchmark notebook; the AIR-RS-DB slice (IndicVoices/Vaani/Nirantar already in).
- **Course-deliverable status (stated plainly — do not overclaim):**
  - *Deliverable 1 — pipeline source code:* **partial.** Runs end-to-end, documented, typed; preprocessing + diarization (with the compulsory clustering) + separation interface are real, but **ASR and LLM stages are stubs** → the report is structurally valid, not yet a real transcript/summary. Completed in the final.
  - *Deliverable 2 — system benchmark notebook:* **not yet; groundwork only.** The per-stage latency + peak-memory **instrumentation is built into every run's `RunManifest`**, but there is **no `notebooks/benchmark.ipynb`, no `config/ablation.yaml`, and no ablation study yet** — all final-half work.
- Visual: a two-column roadmap; mark the ablation as the graded centerpiece; show the two deliverables with honest "partial / groundwork" chips.

## 15. Risks & honest caveats
- MFCC-fallback embeddings cap diarization quality → mitigate with pyannote.
- **Auto speaker-count floors at k≥2** (silhouette can't score k=1), so single-speaker clips (e.g. most Vaani/Nirantar/sources) are over-split into 2 — a known limitation to fix with a single-speaker guard + pyannote embeddings.
- Separation checkpoints are 8 kHz/2-speaker/synthetic → domain gap on real audio (apply on overlap only).
- AIR-RS-DB needs an IEEE DataPort subscription (manual download) — 3 of 4 corpora present meanwhile.
- Expected high DER/WER on chaotic audio → reported as a *finding*, not hidden.

## 16. (Appendix / speaker-notes only) How to regenerate the numbers & figures
Not a slide unless asked. For the pptx agent to produce real figures:
- **Tests / quality badge:** `python -m pytest -q` (expect 17 passed); `ruff check src tests`; `mypy src`.
- **End-to-end + manifest table (slide 12):** `python -m src.pipeline --audio dhvani_eval/sequential/audio/clip_01.wav --out outputs/` then read `outputs/clip_01/manifest.json` (stages[].wall_time_s, peak_mem_mb) and `timeline.json`.
- **DER bar chart (slide 10):** run the diarization-vs-oracle loop over `dhvani_eval/sequential/` (real Diarizer vs `read_rttm`), collect per-clip `der()["der"]`. (Current: mean ≈ 0.30.)
- **Dataset composition (slide 6):** counts from `dhvani_eval/dataset_info.json` + `wc -l` of each `manifest.jsonl`.
- **Embedding scatter (slide 9):** `embed_segments()` on one clip's windowed VAD segments → PCA/t-SNE to 2-D, color by `cluster_embeddings()` labels.
- Keep every figure traceable to a file/command; if a number can't be regenerated, omit it rather than guess.

---

### Hard rules for the pptx agent
1. **No invented metrics.** Only numbers you can reproduce from the repo (this file cites where).
2. Clearly label anything **stubbed** or **fallback** (ASR, LLM, MFCC embeddings, passthrough separation).
3. Keep the "diarization-first" and "compulsory Spectral/GMM clustering + timeline JSON" points prominent — they are the midsem's technical spine.
4. Leave a placeholder for team names/roll numbers on slide 1.
5. Prefer real screenshots of `manifest.json` / `timeline.json` over prose where possible.
