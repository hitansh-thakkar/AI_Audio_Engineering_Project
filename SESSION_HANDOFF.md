# Session Handoff — Dhvani pipeline

Context for the next Claude session to continue seamlessly. Written 2026-10-15.
Companion docs: `FACT_CHECK.md` (full measured audit), `C:\Users\Hitansh\.claude\plans\breezy-soaring-key.md` (the approved plan this session executed).

---

## 0. ⚠️ DO THIS FIRST — unpushed commit + diverged remote

- Local has **1 unpushed commit** `2046a82` ("Implement ASR + LLM stages, fix diarization accuracy, add demo bundle"), authored as the user (hitansh-thakkar), **no Claude co-authorship** (user requires this on all commits).
- `origin/main` **diverged** — it is **5 commits ahead** of our base (parallel work from another machine):
  `e83545c` change eav file from speaker to chunking · `4aff79c` updated ui with graphs · `67ad442` some ui changes · `054559d` update pyannote diarisation · `7b4a3b9` added subtitles.
- **`git push` was REJECTED** (non-fast-forward). Must integrate first.
- **Conflict-likely files** (changed on BOTH sides): `config/default.yaml`, `preprocess_ui.py`, `src/diarization/diarize.py`, `src/ui/preprocess.html`, `src/ui/preprocess_service.py`.
- Remote also ADDED: `src/ui/viz.py` (graphs), `tests/test_sarvam.py`; and touched `src/asr/sarvam.py`.
- **Plan:** `git fetch`; make a backup branch (`git branch backup/pre-merge HEAD`); `git rebase origin/main` (or merge). Resolve conflicts **keeping both**: the remote's UI graphs / subtitles / viz.py AND our diarization-accuracy + overlap-detector fix + ASR/LLM stages. Watch `054559d update pyannote diarisation` — it may overlap our `overlap.py`/`diarize.py` overlap fix; reconcile by hand. Then re-run the gate (below) + regenerate the demo, then push. No self co-authorship.
- Untracked `demo.zip` sits in the working tree (pre-existing from before this session; ignore or ask user).

---

## 1. What this session did (all in commit 2046a82, local)

Starting point: acoustic stages "done" but producing audibly wrong output; ASR + LLM were stubs. We fixed the acoustic stages, implemented the two stub stages for real, cleaned dead code, and built a demo. **ruff + mypy clean (37 src files), 53 tests pass** at session end.

**Diarization (src/diarization/):**
- Default embeddings MFCC→**ECAPA**; affinity raw-cosine→**refined** (pruned+symmetrised kNN graph); counting→**silhouette** (see §3 for why not eigengap).
- New: exclude pyannote-overlap windows from speaker counting, then attach them to the nearest speaker (`diarize._cluster`) — stops the blended overlap window from becoming a phantom speaker.
- Fixed dead overlap detector (see Bugs).

**ASR (src/asr/) — was a stub, now real via `make_asr` factory:**
- `indicwav2vec` → `ai4bharat/indicwav2vec-hindi` (transformers CTC, greedy decode, native Devanagari) — the proposal's localized Indic ASR, **live.yaml default**.
- `faster_whisper` → multilingual + native word timestamps, **default.yaml** (open, no gate, cold-run safe).
- `indicconformer` → multilingual Indic (gated + NeMo, **not usable yet**, see §5).
- `sarvam` → optional **PAID** toggle, off all default paths.

**LLM post-processing (src/postprocessing/) — was a stub, now real via `make_postprocessor` factory:**
- `gemini` (default, `google-genai`, model `gemini-3.8-flash`) + `ollama` (offline). Produces exec summary + keyword tags + cleaned speaker dialogue; **no-new-facts guardrail** + numeric-fact grounding check.

**Standalone preprocess UI (preprocess_ui.py + src/ui/preprocess_service.py + src/ui/preprocess.html):**
- Unified all stages on `live.yaml` (real overlap detection on), added the **LLM report stage (6th slide)**, made preprocessing config-first (removed the hardcoded recipe — it was denoising the diarization path, a bug).

**Deliverables & cleanup:**
- `notebooks/benchmark.ipynb` (latency/memory + DER/SI-SNRi + small ablation), `config/ablation.yaml`.
- `demo/` bundle: `before/` (4 raw inputs) + `after/<clip>/` (full pipeline outputs) + `README.md` + `make_demo.py`. 4 clips: overlap-separation, multi-speaker timeline, real conversation, code-switch.
- Removed dead code: `merge_threshold`/`_merge_similar`, the whole `experiments/` dir (diar_sweep/merge_sweep/make_audition scaffolding).
- `FACT_CHECK.md` audit; README/PROJECT_SUMMARY reconciled to measured reality.

---

## 2. Bugs found & fixed (with evidence)

1. **Dead overlap detector** — `overlap.py` called `.get_overlap()` on pyannote 4.0's `DiarizeOutput` (no such method) → always threw → overlaps never detected on real audio → separation never fired. Fixed by unwrapping `.speaker_diarization`.
2. **Diarization temporal split** ("pre-overlap=spk1, post-overlap=spk2") — weak MFCC embeddings + raw cosine affinity cut on a time boundary. Fixed via ECAPA + refined affinity. Measured: sequential DER median **0.344→~0.10**, speaker-count accuracy **0.20→0.80**.
3. **Speaker over-counting on overlap clips** — eigengap over-counts short clips (mix_002 gave 4). Fixed via silhouette counting (overlap-set count-acc **0.42→0.75**) + overlap-window exclusion. mix_002 now = 2 (true 2).
4. **70 Hz high-pass distorts ECAPA embeddings** — `live.yaml highpass_hz: 70` over-counted (clip_01 **3→6**); confirmed inherent (SOS/order-2/4 all over-count), not a filter-impl bug. Fixed: `highpass_hz: 0` on both configs; rumble handled by `denoise_output` on final tracks. NOTE: the benchmark runs on raw audio so it **hid** this — now consistent with high-pass off.
5. **Paid Sarvam on the default path** (golden-rule-0 violation) — demoted to optional toggle; `make_asr`/`make_postprocessor` factories; `preprocess_service` respects `cfg.asr.backend`.
6. **Stale Gemini model** `gemini-2.5-flash` (retired, 404) → `gemini-3.8-flash`.
7. **Factual-consistency false alarms** — was raw token-overlap (penalised abstractive/cross-lingual summaries); rewrote to ground only numeric facts (prices/counts), strips speaker labels. grounded→1.0, fabricated→0.0.

---

## 3. Key decisions (so they aren't re-litigated)

- **silhouette over eigengap** for counting: eigengap edged sequential (DER 0.082 vs 0.096) but over-counts short overlap clips (count-acc 0.42 vs 0.75). silhouette generalises better. Both remain config options (`diarization.count_method`).
- **No auto k=1 ("1 speaker") detection**: measured — silhouette of true-1-speaker clips overlaps true-2-speaker clips (nirantar 0.30 vs vaani_00 0.65), so no safe threshold. User accepted "2 clips is fine if acceptable." Use `num_speakers` override for known counts.
- **Separator stays `sepformer-wsj02mix` (8 kHz)**: beat 16 kHz `whamr16k` on the clean synthetic set (SI-SNRi 7.2 vs 2.8). `whamr16k`/MossFormer2 kept as documented toggles for noisy real audio (no eval set to score them fairly).
- **Diarize-first kept** (not the brief's separate-first); `separation.mode: everywhere` is the blind-front-end ablation arm.
- **IndicWav2Vec on live, faster-whisper on default**: live.yaml already needs HF_TOKEN (pyannote) so a gated Indic model is fine there; default.yaml stays open for cold runs.

---

## 4. Current config defaults (post-fixes)

`config/live.yaml` (the UI / real-upload path): preprocessing highpass **0**, denoise none, denoise_output spectral, normalize lufs, trim on · diarization ecapa + spectral + silhouette + refined + **overlap_detector pyannote** · separation sepformer-wsj02mix overlap · asr **indicwav2vec / ai4bharat/indicwav2vec-hindi / lang hi** · postprocessing gemini/gemini-3.8-flash.

`config/default.yaml` (offline/eval/cold): highpass 0 · diarization ecapa + spectral + silhouette + refined + **overlap_detector none** · asr **faster_whisper / small / auto** · postprocessing gemini. Tests override asr+postprocessing to `stub`.

---

## 5. Environment facts (this machine)

- Python 3.12.5, **torch 2.10.0+cpu** (CPU-only). GPU is an RTX 3050 Laptop 4 GB **but unusable** without a CUDA-torch reinstall (~2.5 GB, risky, marginal on 4 GB) — we chose CPU-only.
- **Disk budget 10 GB total** (incl. dataset). Currently ~6.8 GB used; ~8–11 GB free. Stay under 10.
- `torchcodec` is broken → pyannote must get in-memory `{"waveform","sample_rate"}` (already handled in `overlap.py`).
- `.env` has **HF_TOKEN + GEMINI_API_KEY + SARVAM_API_KEY** (all present).
- HF gated access: **`ai4bharat/indicwav2vec-hindi` = accessible** with the token; **`ai4bharat/indic-conformer-600m-multilingual` = 403 (gate NOT accepted)** → IndicConformer needs the user to click-accept its HF licence + a NeMo install before it works.
- Installed this session: `faster-whisper`, `google-genai`. Not installed: `whisperx` (unused — native word timestamps), `nemo`, `pyctcdecode`/`kenlm` (so IndicWav2Vec uses greedy CTC, no LM rescoring — KenLM won't build on Windows).

---

## 6. Known limitations / open items

- Short (6–9 s) overlap clips still over-count by ~1 (num_speakers override is the escape hatch).
- **IndicWav2Vec-hindi is Hindi-only**: English/Assamese/Bengali get phonetic Devanagari (what the proposal describes, but imperfect); no word timestamps (CTC → segment-level per speaker); greedy decode (no KenLM). For multilingual/English, switch `asr.backend: faster_whisper`.
- `preprocess_ui.py` has ~15 **pre-existing** mypy warnings (JSON/`parse_qs` typing in the stdlib HTTP handler). Out of the gate (`mypy src`), like `app.py`. Not fixed.
- Proposal items still open (research payload): full real-ASR ablation money-plot across `config/ablation.yaml`; AIR-RS-DB 4th corpus (IEEE-gated).
- **User-facing, pending the user's own listen** on the restarted preprocess UI: (a) nirantar clip "acceptability" with 2 speakers; (b) general ASR accuracy with IndicWav2Vec. They were actively testing clip-by-clip.

---

## 7. How to run / verify

```bash
# quality gate (must stay green)
ruff check src tests
mypy src
python -m pytest -q          # 53 pass; exit code is authoritative (summary line gets eaten in pipes)

# demo (regenerate after any pipeline change)
python demo/make_demo.py     # writes demo/before + demo/after + README

# standalone preprocess UI (the one the user tests) — RESTART after code changes
python preprocess_ui.py      # http://127.0.0.1:8765
# Streamlit UI
streamlit run app.py

# first-half benchmark
python -m src.evaluation.benchmark --out outputs/benchmark.json
```

Gotchas: set `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` when printing Devanagari on this Windows console. First IndicWav2Vec / model load spams tqdm "Loading weights" bars (disabled inside the backend via `hf_logging`). Gemini/pyannote/IndicWav2Vec need the `.env` tokens; all degrade gracefully without them.

---

## 8. Measured numbers (quick reference)

Diarization sequential: DER median ~0.10, count-acc 0.80 (ECAPA+silhouette+refined). Overlap set count-acc 0.75. Separation: SI-SNRi median **+7.2 dB**, SDRi **+4.5 dB** (README originally mis-claimed 8.6). Tests: **53** (docs had wrongly said 17 / 31). ASR on mix_002: both speakers transcribe (51/40 Devanagari chars vs 3–6 with whisper-small).
