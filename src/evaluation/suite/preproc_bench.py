"""Preprocessing ablation, measured downstream (a front-end is only "better" if the next stage
gets better): ASR WER/CER of a fixed recogniser on each preprocessing variant.

Clips: all Vaani (field recordings on phones — the noisy corpus) + 60 IndicVoices clips as a
cleaner control (seeded). Recogniser: MMS-1B-all with oracle language (the best fully-local
multilingual model in asr_bench; rerun asr_bench first so the "raw" row matches it).

    python -m src.evaluation.suite.preproc_bench
"""

from __future__ import annotations

import random
from collections.abc import Callable

import librosa
import numpy as np
import torch

from ...preprocessing.preprocess import highpass, lufs_normalize, spectral_denoise
from .asr_bench import _audio, mms_backend
from .common import (
    ASRHyp,
    asr_manifest,
    breakdown,
    log,
    score_asr_rows,
    setup_logging,
    write_json,
    write_rows,
)

Variant = Callable[[np.ndarray, int], np.ndarray]


def _peak(y: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y / m * 0.95 if m > 0 else y


def _noisereduce(y: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr

    return np.asarray(nr.reduce_noise(y=y, sr=sr, stationary=True), dtype=np.float32)


VARIANTS: dict[str, Variant] = {
    "raw": lambda y, sr: y,
    "peak + trim (current default)": lambda y, sr: _peak(librosa.effects.trim(y, top_db=30)[0]),
    "highpass 70 Hz": lambda y, sr: highpass(y, sr, 70.0),
    "LUFS -23": lambda y, sr: lufs_normalize(y, sr, -23.0),
    "spectral gate (project)": lambda y, sr: spectral_denoise(y, sr),
    "noisereduce (stationary)": _noisereduce,
    "highpass + noisereduce + LUFS": lambda y, sr: lufs_normalize(
        _noisereduce(highpass(y, sr, 70.0), sr), sr, -23.0
    ),
}


def main() -> None:
    setup_logging()
    torch.manual_seed(42)
    man = asr_manifest()
    rng = random.Random(42)
    iv = [m for m in man if m["corpus"] == "indicvoices"]
    clips = [m for m in man if m["corpus"] == "vaani"] + rng.sample(iv, 60)
    audio = {m["id"]: _audio(m) for m in clips}
    fn = mms_backend()
    summary: dict[str, dict[str, object]] = {}
    all_rows = []
    for name, var in VARIANTS.items():
        hyps = []
        for m in clips:
            y = np.asarray(var(audio[m["id"]], 16000), dtype=np.float32)
            hyps.append(ASRHyp(m["id"], fn(y, m["language"]) if y.size > 400 else ""))
        rows = score_asr_rows(f"mms | {name}", hyps, clips)
        b = breakdown(rows)
        summary[name] = {"overall_excl_sd": b["overall_excl_sd"], "by_corpus": b["by_corpus"]}
        all_rows += rows
        log.info(
            "%-32s WER %.4f  vaani %.4f  iv %.4f",
            name,
            b["overall_excl_sd"]["wer"],
            b["by_corpus"]["vaani"]["wer"],
            b["by_corpus"]["indicvoices"]["wer"],
        )
    write_rows("preproc_bench", all_rows)
    write_json("preproc_bench", summary)


if __name__ == "__main__":
    main()
