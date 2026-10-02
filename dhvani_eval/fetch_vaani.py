#!/usr/bin/env python3
"""Add a small REAL Project Vaani slice to dhvani_eval (required corpus #2).

Vaani (ARTPARK-IISc/Vaani) is the diversity/noise corpus: real accent/dialect variation
across Indian districts. It is multi-TB, so we STREAM and keep only a tiny slice — prefer
transcribed clips (for per-language WER), tag the rest (useful for diarization / noise
robustness). Audio is decoded via Audio(decode=False) + soundfile to avoid the torchcodec
dependency; Vaani is already 16 kHz mono.

Writes: dhvani_eval/vaani/audio/*.wav + dhvani_eval/vaani/manifest.jsonl
Disk-bounded (caps examples scanned); clear the HF cache afterwards (see docs/DATASETS.md).

Run:  python dhvani_eval/fetch_vaani.py        (needs HF_TOKEN in .env + Vaani license accepted)
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SR = 16_000
CONFIGS = ["Hindi", "Bengali", "Assamese"]   # match dhvani_eval's languages
N_PER = 8                                     # clips kept per config
MAX_SCAN_PER = 60                             # streaming cap per config (bounds disk/time)


def load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("[vaani] HF_TOKEN not set (.env). Accept the Vaani license on the Hub first.")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    import itertools

    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    out = ROOT / "vaani"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    def decode(ab: dict) -> "np.ndarray":
        y, sr = sf.read(io.BytesIO(ab["bytes"]))
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=SR)
        return y.astype("float32")

    for cfg in CONFIGS:
        try:
            ds = load_dataset("ARTPARK-IISc/Vaani", cfg, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:  # noqa: BLE001
            print(f"[vaani] skip {cfg}: {type(e).__name__} {str(e)[:120]}")
            continue

        kept, buffer = [], []
        for ex in itertools.islice(ds, 0, MAX_SCAN_PER):
            transcript = (ex.get("transcript") or "").strip()
            rec = (ex, transcript)
            if transcript:
                kept.append(rec)
            else:
                buffer.append(rec)
            if len(kept) >= N_PER:
                break
        # fill up to N_PER with untranscribed if needed (diversity/diarization use)
        for rec in buffer:
            if len(kept) >= N_PER:
                break
            kept.append(rec)

        for i, (ex, transcript) in enumerate(kept):
            y = decode(ex["audio"])
            cid = f"vaani_{cfg.lower()}_{i:02d}"
            rel = f"vaani/audio/{cid}.wav"
            sf.write(ROOT / rel, y, SR, subtype="PCM_16")
            manifest.append({
                "id": cid, "corpus": "vaani", "config": cfg,
                "language": ex.get("language"), "transcript": transcript,
                "transcribed": bool(transcript),
                "duration": round(len(y) / SR, 3), "audio": rel,
                "gender": ex.get("gender"), "state": ex.get("state"),
                "district": ex.get("district"), "speaker_id": ex.get("speakerID"),
                "source": "ARTPARK-IISc/Vaani",
            })
        print(f"[vaani] {cfg}: kept {len(kept)} ({sum(1 for _, t in kept if t)} transcribed)")

    if not manifest:
        sys.exit("[vaani] nothing collected — check license acceptance / network.")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[vaani] wrote dhvani_eval/vaani/ with {len(manifest)} clips across {len(CONFIGS)} languages")


if __name__ == "__main__":
    main()
