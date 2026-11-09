# Local Setup — Zero Paid Services

This pipeline runs on **one consumer PC** with **no paid APIs and no per-token cost**. The acoustic stack (preprocess → diarize → separate → ASR) is fully local and offline. The one cloud touch is the LLM post-processing stage (Stage 5), which by default calls the **Google Gemini API free tier** — a free key, not a paid service. You need **two free credentials**: a Hugging Face token (pyannote's click-through licenses; gated ≠ paid) and a Gemini API key. Want a 100%-offline run? Switch Stage 5 to the optional local Ollama backend (§2.3). This document covers install, the hardware-tier table, VRAM/RAM/disk budgeting, and troubleshooting.

---

## 1. What you install

| Component | What it's for | Cost |
| --- | --- | --- |
| **Python 3.10+** + `requirements.txt` | the pipeline itself | free |
| **ffmpeg** (on PATH) | audio decode/resample for Whisper & torchaudio | free |
| **Gemini API** (`google-genai`) | LLM post-processing (Stage 5), default | free tier — needs a free `GEMINI_API_KEY` (cloud call) |
| **Ollama** + one model *(optional)* | fully-offline LLM post-processing (Stage 5) | free, local — only if you opt out of Gemini |
| **SpeechBrain** weights | SepFormer separation (Stage 3) + **ECAPA-TDNN** speaker embeddings (Stage 2) | free, auto-downloaded (open, no token) |
| **pyloudnorm** + **scipy** | preprocessing: BS.1770 LUFS + high-pass (Stage 1) | free (pip) |
| **faster-whisper** weights | ASR runtime (Stage 4), int8 CTranslate2 | free, auto-downloaded |
| **pyannote** community-1 + segmentation | **overlap detection** (Stage 2) on real uploads | free — needs a free HF token + click-through licenses |
| **Streamlit** (`streamlit run app.py`) | drag-drop web UI → per-speaker audio + timeline | free, local |
| **bert-score** model | summary metric (Stage 6) | free, auto-downloaded once (~1.4 GB) |

The only keys on the default path are the free `HF_TOKEN` and `GEMINI_API_KEY`. No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `SARVAM_API_KEY` / `PYANNOTEAI_API_KEY` is required or used.

---

## 2. Install steps

### 2.1 Python environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 ffmpeg
- **Windows:** `winget install Gyan.FFmpeg` (or `choco install ffmpeg`), then reopen the shell so it's on PATH.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`
- Verify: `ffmpeg -version`.

### 2.3 LLM backend (Stage 5)

**Default — Gemini API (free tier):**
1. Get a free key at <https://aistudio.google.com/apikey> (Google account; no billing required for the free tier).
2. Export it:
   ```bash
   export GEMINI_API_KEY=...    # Windows PowerShell: $env:GEMINI_API_KEY="..."
   ```
3. The client (`google-genai`) is already in `requirements.txt`. `postprocessing.model` (`config/default.yaml`) selects the model (default `gemini-3.8-flash`). This stage makes a **cloud** call — the only non-local part of the pipeline.

**Optional — Ollama (fully offline, no cloud, no key):** set `postprocessing.backend: ollama` in the config, then:
1. Install from <https://ollama.com/download> (Windows/macOS/Linux). It runs a local server at `http://localhost:11434`.
2. Pull the model for your tier (see the table in §3):
   ```bash
   ollama pull qwen3:4b        # swap per your hardware
   ollama run qwen3:4b "ok?"   # sanity check
   ```
3. `postprocessing.host` (`config/default.yaml`) points at that server — no key, no account.

### 2.4 faster-whisper (local ASR)
Nothing to pre-install beyond `pip` — `faster-whisper` downloads the int8 CTranslate2 weights for `asr.model` on first run. To pre-warm:
```python
from faster_whisper import WhisperModel
WhisperModel("large-v3", device="cpu", compute_type="int8")   # caches the weights
```
For **Tara** (recommended Hindi/Hinglish), convert once to CTranslate2:
```bash
ct2-transformers-converter --model Trelis/tara --quantization int8 --output_dir models/tara-ct2
```
then set `asr.model: models/tara-ct2`. (No official prebuilt Tara-CT2 build is published — verify on HF before assuming one; otherwise convert as above or run Tara via `transformers` int8.)

### 2.5 Free Hugging Face token (diarization only)
1. Create a free account at <https://huggingface.co>, then a token at Settings → Access Tokens (read scope is enough).
2. Accept the **two** click-through licenses (both free):
   - `pyannote/speaker-diarization-community-1`
   - `pyannote/segmentation`  ← *missing this one causes a 401.*
3. Export it:
   ```bash
   export HF_TOKEN=hf_xxx      # Windows PowerShell: $env:HF_TOKEN="hf_xxx"
   ```

---

## 3. Hardware-tier table (LLM stage — Ollama backend only)

> The default **Gemini** backend runs in the cloud and needs **no local VRAM/RAM/disk** for the LLM. This table applies only if you opt into the fully-offline **Ollama** backend (§2.3).

The LLM stage only cleans and summarizes text, so a **7–9B local model is plenty**. Pick by VRAM. Model size is configurable via `postprocessing.model` — don't hardcode it. Tags below were verified against the official Ollama library; **re-check before pulling, tags change.**

| Tier | Ollama model | On-disk (approx) | License | Notes |
| --- | --- | --- | --- | --- |
| **CPU-only / ≤6 GB VRAM** | `qwen3:4b` or `phi4-mini` | ~2.5 GB | Apache-2.0 / MIT | Smallest sensible cleaners. **Recommended Ollama model = `qwen3:4b`.** |
| **8 GB VRAM** *(reference default tier)* | `qwen3:8b` | ~5.2 GB | Apache-2.0 | Strong multilingual incl. Hindi. Recommended if you have the VRAM/disk. |
| **12–16 GB VRAM** | `gemma3:12b` or `qwen3:14b` | ~8.1 / ~9.3 GB | Gemma / Apache-2.0 | Better summarization/multilingual. |
| **24 GB+ VRAM** | `gemma3:27b` or `qwen3:32b` | ~17 / ~20 GB | Gemma / Apache-2.0 | Largest local options. Sarvam-M **only if it fits** (unverified — see below). |

> ⚠️ **Sarvam-M is not in the official Ollama library.** Only unofficial community uploads exist (e.g. `mashriram/sarvam-m`, ~25 GB q8, 24B). Treat as experimental, 24 GB+ only; prefer `qwen3:32b` / `gemma3:27b`.

Ollama transparently offloads as many layers to the GPU as fit and runs the rest on CPU, so a model larger than your VRAM still runs (just slower) as long as it fits in RAM.

---

## 4. Your machine (this laptop) & budgeting

Detected: **NVIDIA RTX 3050 Laptop (4 GB VRAM), 16 GB RAM, Intel i5-12450H, ~17 GB free disk** (target: keep model downloads to ~5–7 GB).

That is the **CPU-only / ≤6 GB VRAM** tier. Recommended, budget-fitting stack:

| Stage | Pick | Disk | Runtime memory |
| --- | --- | --- | --- |
| Diarization | pyannote community-1 (CPU) | ~0.5 GB | CPU RAM; a few hundred MB |
| ASR | faster-whisper `large-v3`, `int8`, `device: cuda` | ~1.5 GB | ~2–4 GB VRAM (fits 4 GB); use `cpu` if OOM |
| LLM | **Gemini API (default)** | 0 (cloud) | 0 local — free-tier API call |
| LLM (if offline) | Ollama `qwen3:4b` | ~2.5 GB | ~3 GB (partial GPU offload + RAM) |
| Metrics | bert-score model | ~1.4 GB | one-time, cached |
| **Total (Gemini default)** | | **~3.4 GB** | comfortably within budget |
| **Total (offline Ollama)** | | **~5.9 GB** | fits your budget |

**If disk/VRAM is tight:**
- ASR: switch `asr.model` to `distil-large-v3` (~0.75 GB) or `small`, or set `asr.device: cpu`, `compute_type: int8`.
- LLM: `qwen3:4b` is already minimal; `phi4-mini` is a same-size alternative.
- Skip pre-warming models you won't run; delete cached weights you don't need (`~/.cache/huggingface`, `ollama rm <model>`).

**Rules of thumb:**
- VRAM ≈ int8 model size + ~1 GB activations. large-v3 int8 ≈ 2 GB weights.
- RAM: keep ~4 GB headroom; stages run sequentially, so peak is one heavy model at a time.
- Disk: HF cache lives in `~/.cache/huggingface`; Ollama models in `~/.ollama/models`.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `401` on diarization load | You didn't accept **both** licenses (community-1 **and** segmentation), or `HF_TOKEN` isn't exported. Accept both, re-export the token. |
| `401`/`PERMISSION_DENIED` on LLM stage (Gemini) | `GEMINI_API_KEY` missing/invalid. Create a free key at <https://aistudio.google.com/apikey> and re-export it. |
| `429`/quota exhausted on LLM stage (Gemini) | Free-tier rate/day limit hit. Wait for the window to reset, switch to a lighter `postprocessing.model`, or use the offline Ollama backend. |
| `Connection refused` on LLM stage (Ollama) | Ollama server not running. Start it (the desktop app, or `ollama serve`) and confirm `postprocessing.host` = `http://localhost:11434`. |
| `model 'qwen3:4b' not found` (Ollama) | Not pulled yet: `ollama pull qwen3:4b`. Verify the tag on <https://ollama.com/library>. |
| CUDA out of memory (ASR) | Drop to `compute_type: int8`, use a smaller `asr.model` (`distil-large-v3`/`small`), or set `asr.device: cpu`. |
| `ffmpeg not found` | Install ffmpeg and reopen the shell so PATH updates (§2.2). |
| Very slow LLM stage (Ollama) | Model too big for VRAM → heavy CPU offload. Use a smaller tier model, or accept the slower run. (Gemini default avoids this — it runs in the cloud.) |
| Devanagari WER looks absurdly high | You used Whisper's default normalizer. Must be the script-safe Indic normalizer (NFC + `indic-nlp-library`) — see `docs/EVALUATION.md`. |
| Out of disk mid-download | Free space or move caches. `ollama rm <model>` and `huggingface-cli delete-cache` reclaim space; prefer smaller tier models (§4). |

---

## 6. What is *not* here (by design)

No paid inference, ever, on the default path. The paid options that exist elsewhere in the docs — pyannote **precision-2** (cloud API) and the **`api` LLM backend** (frontier/Sarvam) — are optional accuracy ceilings you must opt into and pay for yourself. Leaving the config untouched keeps the run **100% free**; it is also fully local *except* the Stage-5 Gemini call. For a **100% offline** run, switch `postprocessing.backend` to `ollama` (§2.3).
