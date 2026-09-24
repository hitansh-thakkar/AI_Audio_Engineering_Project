# CLAUDE.md

Guidance for Claude Code working in this repository. Read this first, then `PLAN.md`, then the `docs/`.

## What this project is

A "Who Spoke What and When" speech pipeline for chaotic, multi-speaker, code-switched Indian audio. It chains **pretrained** models — separation, diarization, ASR, LLM post-processing. **We do not train models from scratch.** The intellectual core is the **benchmark + ablation study** that measures error propagation across stages. Optimize your work for **measurability and reproducibility first, raw accuracy second.**

Read `docs/ARCHITECTURE.md` for the design and the reasons behind it before writing pipeline code.

## Golden rules (do not violate without flagging)

0. **Runs fully locally, no paid services.** The entire default pipeline runs on one consumer PC with zero paid APIs, zero cloud inference, and zero per-token cost. The *only* acceptable credential is a **free** Hugging Face token for pyannote's click-through gated licenses (gated ≠ paid). Every model is open-weights on the user's machine: pyannote (CPU), SpeechBrain, faster-whisper, and a local LLM via Ollama. Paid backends (frontier/Sarvam APIs, pyannote precision-2) may exist only as clearly-labeled *optional* toggles — never on the default path, never as a default value.
1. **Diarization runs before separation.** Separation is applied *only* to overlapped regions by default. Do not restore a blind separate-everything front-end except behind the `separation.mode: everywhere` toggle (it exists for the ablation, not as the default).
2. **Every stage is swappable behind an interface.** All models are loaded via a registry driven by `config/default.yaml`. Never hardcode a model ID inside pipeline logic — read it from config.
3. **Use an Indic-script-safe text normalizer for WER/CER.** Whisper's default normalizer strips Devanagari matras and inflates Devanagari WER. Use NFC + `indic-nlp-library` normalization. This is a correctness bug if you get it wrong — all reported numbers depend on it.
4. **The pipeline must degrade gracefully.** A failure in one stage (e.g., separation produces garbage, diarization finds 1 speaker) must not crash the run — log it, pass through, and record it in the run manifest. Error *propagation* is the thing we measure, so errors must flow, not halt.
5. **Do not fabricate content in the LLM stage.** The post-processing prompt must constrain the model to only reorganize/clean what is in the transcript. Summaries may not introduce facts absent from the input. Add an explicit instruction and, where feasible, a factual-consistency check.
6. **Determinism where possible.** Set and log all seeds. Record model versions, config hash, and git SHA in every run manifest so results are reproducible.

## Tech stack (all local, all open-weights)

- **Python 3.10+** (pyannote.audio 4.0 requires it), PyTorch, torchaudio.
- **speechbrain** (SepFormer/Conv-TasNet), **pyannote.audio** (diarization, runs on CPU).
- **faster-whisper** (CTranslate2, int8) is the ASR runtime — runs Whisper-large-v3 / Tara (converted to CT2) / distil/medium/small on GPU or CPU; **whisperx** for word alignment; **transformers** as the fallback path for Tara / IndicConformer.
- **jiwer** (WER/CER), **pyannote.metrics** (DER/JER), **indic-nlp-library** (normalization), **bert-score** (downloads a free model on first use).
- LLM stage runs **locally via Ollama** (`http://localhost:11434`); pick the model per hardware tier (see `docs/LOCAL_SETUP.md`). Paid frontier/Sarvam APIs are optional, off by default.
- Config: **OmegaConf / PyYAML**. Notebooks: Jupyter.

## Repo conventions

- **Module contract:** each stage lives in `src/<stage>/` and exposes a class implementing the interface in `docs/ARCHITECTURE.md` (`load()`, `run(input) -> output`, plus `name`/`version` metadata). Stages communicate through typed dataclasses (`Segment`, `SpeakerTurn`, `Transcript`), not loose dicts.
- **No side effects on import.** Model weights load lazily inside `load()`, never at module top level.
- **Config-first.** New knobs go in `config/default.yaml` with a sane default and a one-line comment, then get read in code.
- **Typed.** Use dataclasses / type hints throughout. Run `ruff` and `mypy` clean.
- **Tests.** Every stage gets a unit test with a tiny fixture clip in `tests/fixtures/`. Add a smoke test that runs the whole pipeline end-to-end on a 5-second clip.
- **Logging.** Use structured logging; every stage logs its wall-clock time and peak memory (the benchmark depends on this instrumentation existing in the code, not bolted on later).

## Commands

```bash
# install (Python deps)
pip install -r requirements.txt
# system deps: ffmpeg on PATH; Ollama installed + a model pulled (local LLM stage)
ollama pull qwen3:4b   # shipped default; see docs/LOCAL_SETUP.md for the tier table

# lint / type / test
ruff check src tests
mypy src
pytest -q

# run pipeline on one file
python -m src.pipeline --audio <path> --config config/default.yaml

# run one stage in isolation (debugging)
python -m src.diarization.run --audio <path>
python -m src.asr.run --audio <path> --model large-v3   # faster-whisper CT2 weights

# benchmark + ablation (Deliverable 2)
python -m src.evaluation.benchmark --suite config/ablation.yaml
```

## Build order

Follow `PLAN.md` phase by phase. Build **vertical slices**: get a trivial end-to-end run working (even with stub models) before making any single stage good. The orchestrator + data contracts + config system come first, then real models, then evaluation, then the ablation. Do not disappear into perfecting the separation model while the pipeline can't run end-to-end.

## Gated models / secrets

- `pyannote/speaker-diarization-community-1` and its `segmentation` dependency are **gated but free** — the user accepts the click-through conditions on Hugging Face and provides a free `HF_TOKEN`. Gated ≠ paid. Missing the *segmentation* agreement causes a 401; surface a clear error, don't retry silently.
- `HF_TOKEN` is the **only** credential on the default path. faster-whisper and Ollama weights are open and need no token.
- Optional paid backends (frontier/Sarvam APIs → `ANTHROPIC_API_KEY` etc.; pyannote precision-2 → `PYANNOTEAI_API_KEY`) are off by default. Do not read or require these keys on the default local path.
- Never commit tokens. Read all secrets from environment variables. Document any new env var in `README.md`.

## When you're unsure

- If a design choice conflicts with `docs/ARCHITECTURE.md`, follow the doc or flag the conflict — don't silently diverge.
- If a model download or API is unavailable in the environment, stub the stage behind its interface so the pipeline still runs end-to-end, and note it in the run manifest. Never invent fake model outputs in committed code paths.
