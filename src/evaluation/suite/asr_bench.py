"""ASR benchmark on dhvani_eval/asr_whisper (833 clips, 15 languages, 4 corpora).

Systems run locally here (all free; the externally-run Sarvam and faster-whisper systems are
imported by rescore_external.py instead of being re-run):
  * mms        — facebook/mms-1b-all (wav2vec2 CTC + per-language adapters), GPU fp16
  * indicw2v   — ai4bharat/indicwav2vec-hindi (proposal-named "IndicWav2Vec"), Hindi only
  * indicconf  — ai4bharat/indic-conformer-600m-multilingual, CTC and RNNT decoders
  * gemini:<m> — Gemini API free tier, stratified subset only (quota)

Language is given from the manifest (oracle language ID) for every system, matching the
external runs, so the comparison isolates recognition quality. Hypotheses are cached per
system in outputs/reports/data/hyps/<system>.jsonl, so an interrupted run resumes.

    python -m src.evaluation.suite.asr_bench --systems mms indicw2v
    python -m src.evaluation.suite.asr_bench --systems gemini:gemini-3.5-transcribe --subset
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from ...config import load_env
from .common import (
    DATA_OUT,
    EVAL_ROOT,
    ASRHyp,
    asr_manifest,
    breakdown,
    log,
    score_asr_rows,
    setup_logging,
    write_json,
    write_rows,
)

HYP_DIR = DATA_OUT / "hyps"
Transcriber = Callable[[np.ndarray, str], str]

# manifest language -> MMS adapter (ISO 639-3). Nepali is "npi"; Urdu adapter is Arabic script.
MMS_LANG = {
    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
    "te": "tel",
    "mr": "mar",
    "gu": "guj",
    "kn": "kan",
    "ml": "mal",
    "pa": "pan",
    "ur": "urd-script_arabic",
    "as": "asm",
    "sa": "hin",  # MMS has no Sanskrit adapter; Hindi shares Devanagari (fallback, flagged in report)
    "ne": "npi",
    "sd": "snd",
    "en": "eng",
}
LANG_NAME = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "as": "Assamese",
    "sa": "Sanskrit",
    "ne": "Nepali",
    "sd": "Sindhi",
    "en": "English",
}
# IndicConformer has no English head; those clips are recorded as unsupported (empty output).
INDICCONF_LANGS = set(MMS_LANG) - {"en"}


def stratified_subset(
    manifest: list[dict[str, Any]], per_lang: int = 6, seed: int = 42
) -> list[dict[str, Any]]:
    """Fixed, seeded subset: `per_lang` clips per language, spread across corpora, with
    code-switch clips forming 2 of the Hindi picks."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for lang in sorted({m["language"] for m in manifest}):
        pool = [m for m in manifest if m["language"] == lang]
        if lang == "hi":
            cs = [m for m in pool if m["register"] == "codeswitch"]
            mono = [m for m in pool if m["register"] != "codeswitch"]
            out += rng.sample(cs, 2) + rng.sample(mono, per_lang - 2)
            continue
        by_corpus: dict[str, list[dict[str, Any]]] = {}
        for m in pool:
            by_corpus.setdefault(m["corpus"], []).append(m)
        picks: list[dict[str, Any]] = []
        corpora = sorted(by_corpus)
        i = 0
        while len(picks) < min(per_lang, len(pool)):  # round-robin across corpora
            c = by_corpus[corpora[i % len(corpora)]]
            if c:
                picks.append(c.pop(rng.randrange(len(c))))
            i += 1
        out += picks
    return out


def _audio(m: dict[str, Any]) -> np.ndarray:
    y, sr = sf.read(EVAL_ROOT / m["audio"], dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y


# ------------------------------------------------------------------ backends
def mms_backend() -> Transcriber:
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = (
        Wav2Vec2ForCTC.from_pretrained(
            "facebook/mms-1b-all", torch_dtype=torch.float16, ignore_mismatched_sizes=True
        )
        .to("cuda")  # type: ignore[arg-type]  # transformers' decorated .to() stub rejects a device str
        .eval()
    )
    state = {"lang": ""}

    def run(y: np.ndarray, lang: str) -> str:
        code = MMS_LANG[lang.split("-")[0]]
        if state["lang"] != code:  # adapter swap is ~10 ms; only on language change
            proc.tokenizer.set_target_lang(code)
            model.load_adapter(code)
            model.half()
            state["lang"] = code
        x = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda", torch.float16)
        with torch.inference_mode():
            ids = model(x).logits.argmax(-1)[0]
        return str(proc.decode(ids))

    return run


def indicw2v_backend() -> Transcriber:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    load_env()
    rid = "ai4bharat/indicwav2vec-hindi"
    # The repo's default processor is the KenLM-backed one; greedy CTC keeps it comparable with
    # MMS (also greedy) and avoids the pyctcdecode/kenlm dependency.
    proc = Wav2Vec2Processor.from_pretrained(rid, token=os.environ.get("HF_TOKEN"))
    model = Wav2Vec2ForCTC.from_pretrained(rid, token=os.environ.get("HF_TOKEN"))
    model = model.to("cuda").eval()  # type: ignore[arg-type]  # same transformers .to() stub issue

    def run(y: np.ndarray, lang: str) -> str:
        x = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
        with torch.inference_mode():
            ids = model(x).logits.argmax(-1)[0]
        return str(proc.decode(ids))

    return run


def indicconf_backend(decoder: str) -> Transcriber:
    from transformers import AutoModel

    load_env()
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    def run(y: np.ndarray, lang: str) -> str:
        lang = lang.split("-")[0]
        if lang not in INDICCONF_LANGS:
            return ""  # unsupported language: scored as all-deletions, flagged in report
        with torch.inference_mode():
            return str(model(torch.from_numpy(y).unsqueeze(0), lang, decoder))

    return run


def gemini_backend(model_name: str) -> Transcriber:
    from google import genai
    from google.genai import errors, types

    load_env()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def run(y: np.ndarray, lang: str) -> str:
        buf = io.BytesIO()
        sf.write(buf, y, 16000, format="WAV", subtype="PCM_16")
        name = "Hindi-English code-switched" if lang.startswith("hi") else LANG_NAME[lang]
        # Same oracle-language policy as every other system. No translation, no cleanup:
        # verbatim, native script — anything else is scored as an error.
        prompt = (
            f"Transcribe this {name} speech verbatim in its native script. Output only "
            "the transcript text, with no translation, labels, or commentary. If there is "
            "no speech, output nothing."
        )
        for attempt in range(6):
            try:
                content = types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
                        types.Part.from_text(text=prompt),
                    ],
                )
                r = client.models.generate_content(
                    model=model_name,
                    contents=content,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return (r.text or "").strip()
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code in (429, 500, 503) and attempt < 5:  # free-tier rate limit / transient
                    time.sleep(min(60, 8 * 2**attempt))
                    continue
                raise
        return ""

    return run


def make_backend(system: str) -> Transcriber:
    if system == "mms":
        return mms_backend()
    if system == "indicw2v":
        return indicw2v_backend()
    if system.startswith("indicconf"):
        return indicconf_backend(system.split(":")[1] if ":" in system else "ctc")
    if system.startswith("gemini:"):
        return gemini_backend(system.split(":", 1)[1])
    raise ValueError(f"unknown system {system}")


# ------------------------------------------------------------------- runner
def _cache_path(system: str) -> Path:
    return HYP_DIR / (system.replace(":", "_").replace("/", "_") + ".jsonl")


def load_hyps(system: str) -> dict[str, ASRHyp]:
    p = _cache_path(system)
    if not p.exists():
        return {}
    out: dict[str, ASRHyp] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out[d["id"]] = ASRHyp(d["id"], d["hyp"], d.get("seconds"))
    return out


def transcribe(system: str, clips: list[dict[str, Any]]) -> dict[str, ASRHyp]:
    HYP_DIR.mkdir(parents=True, exist_ok=True)
    done = load_hyps(system)
    todo = [m for m in clips if m["id"] not in done]
    if system == "indicw2v":
        todo = [m for m in todo if m["language"] == "hi"]
    if not todo:
        return done
    fn = make_backend(system)
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    with _cache_path(system).open("a", encoding="utf-8") as f:
        for i, m in enumerate(todo):
            y = _audio(m)
            t0 = time.perf_counter()
            try:
                hyp = fn(y, m["language"])
            except Exception as exc:  # degrade gracefully: log, keep going (golden rule 4)
                # Not cached: a quota/transient failure must be retried on resume, not scored as
                # an empty transcript. Clips still missing at scoring time count as deletions.
                log.warning("%s failed on %s: %s", system, m["id"], str(exc)[:200])
                continue
            dt = time.perf_counter() - t0
            f.write(json.dumps({"id": m["id"], "hyp": hyp, "seconds": dt}, ensure_ascii=False) + "\n")
            f.flush()
            done[m["id"]] = ASRHyp(m["id"], hyp, dt)
            if i % 50 == 0:
                log.info("%s %d/%d", system, i, len(todo))
    if gpu:
        log.info("%s peak GPU %.0f MB", system, torch.cuda.max_memory_allocated() / 2**20)
    return done


def main() -> None:
    setup_logging()
    torch.manual_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--subset", action="store_true", help="stratified 6-per-language subset")
    args = ap.parse_args()
    manifest = asr_manifest()
    subset = stratified_subset(manifest)
    write_json("asr_subset_ids", [m["id"] for m in subset])
    clips = subset if args.subset else manifest
    for system in args.systems:
        hyps = transcribe(system, clips)
        scope = clips if system != "indicw2v" else [m for m in clips if m["language"] == "hi"]
        rows = score_asr_rows(system, hyps.values(), scope)
        name = "asr_" + system.replace(":", "_")
        write_rows(name, rows)
        write_json(name, breakdown(rows))
        o = breakdown(rows)["overall_excl_sd"]
        log.info("%s WER %.4f CER %.4f (n=%d)", system, o["wer"], o["cer"], o["clips"])


if __name__ == "__main__":
    main()
