#!/usr/bin/env python3
"""Ingest AIR-RS-DB (required corpus #4) into dhvani_eval.

AIR-RS-DB is gated behind an IEEE DataPort *subscription* and cannot be fetched with an
API token, so YOU download it manually:
  https://ieee-dataport.org/documents/air-rs-db-all-india-radio-read-and-spontaneous-speech-data-base
Unzip so that this layout exists:
  data/air_rs_db/<audio files>.wav
  data/air_rs_db/air_rs_db_auto-annotation.csv     (fields: filename:duration:readability:R/S)

Then run:  python dhvani_eval/ingest_air_rs_db.py

It writes dhvani_eval/air_rs_db/audio/*.wav (16 kHz mono) + manifest.jsonl with the
read-vs-spontaneous (R/S) label — the style contrast reported in docs/EVALUATION.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SRC_DIR = REPO / "data" / "air_rs_db"
SR = 16_000


def _find_csv() -> Path | None:
    for name in ("air_rs_db_auto-annotation.csv", "air_rs_db_auto_annotation.csv"):
        p = SRC_DIR / name
        if p.exists():
            return p
    hits = list(SRC_DIR.rglob("*annotation*.csv")) if SRC_DIR.exists() else []
    return hits[0] if hits else None


def _parse_annotation(csv_path: Path) -> dict[str, dict]:
    """Return {filename_stem: {duration, readability, label}}. Fields are colon-separated."""
    labels: dict[str, dict] = {}
    for raw in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in (raw.split(":") if ":" in raw else raw.split(","))]
        if len(parts) < 4:
            continue
        fname, dur, read_score, rs = parts[0], parts[1], parts[2], parts[3].upper()
        if rs not in ("R", "S") or fname.lower() in ("filename", "file", "audio"):
            continue  # skip header / malformed
        labels[Path(fname).stem] = {
            "duration_csv": dur, "readability": read_score,
            "register": "read" if rs == "R" else "spontaneous", "rs": rs,
        }
    return labels


def main() -> None:
    if not SRC_DIR.exists():
        sys.exit(f"[air-rs-db] {SRC_DIR} not found. Download the dataset there first (see this file's header).")
    csv_path = _find_csv()
    if not csv_path:
        sys.exit(f"[air-rs-db] annotation CSV not found under {SRC_DIR}.")

    import librosa
    import numpy as np
    import soundfile as sf

    labels = _parse_annotation(csv_path)
    if not labels:
        sys.exit(f"[air-rs-db] no R/S rows parsed from {csv_path.name} — check its format.")

    wavs = {p.stem: p for p in SRC_DIR.rglob("*.wav")}
    out = ROOT / "air_rs_db"
    (out / "audio").mkdir(parents=True, exist_ok=True)

    manifest, n_r, n_s, missing = [], 0, 0, 0
    for stem, meta in sorted(labels.items()):
        src = wavs.get(stem)
        if src is None:
            missing += 1
            continue
        y, sr = sf.read(src)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=SR)
        rel = f"air_rs_db/audio/{stem}.wav"
        sf.write(out / "audio" / f"{stem}.wav", np.asarray(y, dtype="float32"), SR, subtype="PCM_16")
        manifest.append({
            "id": stem, "corpus": "air_rs_db", "register": meta["register"],
            "readability": meta["readability"], "duration": round(len(y) / SR, 3),
            "audio": rel, "source": "IEEE DataPort: AIR-RS-DB",
        })
        n_r += meta["rs"] == "R"
        n_s += meta["rs"] == "S"

    if not manifest:
        sys.exit("[air-rs-db] matched 0 audio files to annotations — check the unzip layout.")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[air-rs-db] wrote {len(manifest)} clips (read={n_r}, spontaneous={n_s}); "
          f"{missing} annotated files had no matching wav.")


if __name__ == "__main__":
    main()
