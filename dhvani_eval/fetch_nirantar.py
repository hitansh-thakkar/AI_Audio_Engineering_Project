#!/usr/bin/env python3
"""Add a small REAL Nirantar slice to dhvani_eval (required corpus #3).

Provenance: the official AI4Bharat Nirantar (github.com/AI4Bharat/Nirantar) hosts audio on
E2E object storage as a 5-part gzip-tar (`AUDIOS.tgz.aa..ae`), ~40 GB PER PART (~198 GB
total) — far beyond a laptop. There is NO official Hugging Face dataset (Hub mirrors are
unverified and were found non-loadable).

Default method `e2e-prefix`: Range-download only the FIRST chunk of part `aa`, gunzip the
prefix, and extract the leading complete FLAC files — a real, official-source slice that fits
the ≤5 GB budget. Audio is 16 kHz mono FLAC; we rewrite to 16 kHz PCM-16 wav.

Transcripts live in separate (small) manifest archives (LIL/DIL/LIDIL) on the same storage;
this slice is audio-only (fine for diarization/diversity). Fetch those separately if you need
Nirantar WER.

Run:  python dhvani_eval/fetch_nirantar.py                       # default: e2e-prefix slice
      python dhvani_eval/fetch_nirantar.py --mb 200 --n 24       # tune chunk size / clip count
      python dhvani_eval/fetch_nirantar.py --local               # ingest from data/nirantar/*.flac|wav
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SR = 16_000
E2E_URL = "https://objectstore.e2enetworks.net/nirantar/AUDIOS.tgz.aa"


def _write_slice(items: list[tuple[str, bytes]], n: int) -> list[dict]:
    """items: (name, flac_bytes). Decode, resample to 16k, write wav + manifest rows."""
    import librosa
    import numpy as np
    import soundfile as sf

    out = ROOT / "nirantar"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for name, data in items:
        if len(manifest) >= n:
            break
        try:
            y, sr = sf.read(io.BytesIO(data))
        except Exception:
            continue  # truncated / undecodable (e.g. the tail file)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=SR)
        stem = Path(name).stem
        rel = f"nirantar/audio/{stem}.wav"
        sf.write(out / "audio" / f"{stem}.wav", np.asarray(y, dtype="float32"), SR, subtype="PCM_16")
        manifest.append({
            "id": stem, "corpus": "nirantar", "provenance": "official-e2e-prefix",
            "text": "", "language": None, "duration": round(len(y) / SR, 3), "audio": rel,
            "source": "objectstore.e2enetworks.net/nirantar/AUDIOS.tgz (github.com/AI4Bharat/Nirantar)",
        })
    return manifest


def _fetch_e2e_prefix(mb: int, n: int) -> list[dict]:
    nbytes = mb * 1024 * 1024
    req = urllib.request.Request(E2E_URL, headers={"Range": f"bytes=0-{nbytes - 1}"})
    print(f"[nirantar] Range-downloading first {mb} MB of AUDIOS.tgz.aa ...")
    with urllib.request.urlopen(req, timeout=120) as r:
        compressed = r.read()

    # gunzip the prefix (truncated stream -> decompress what we can, ignore the incomplete tail)
    dobj = zlib.decompressobj(32 + 15)
    try:
        raw = dobj.decompress(compressed)
    except zlib.error:
        raw = dobj.unconsumed_tail and b"" or b""
    if not raw:
        sys.exit("[nirantar] could not decompress prefix — try a larger --mb.")

    items: list[tuple[str, bytes]] = []
    try:
        tar = tarfile.open(fileobj=io.BytesIO(raw), mode="r|")  # streaming; tolerates truncation
        for m in tar:
            if m.isfile() and m.name.lower().endswith((".flac", ".wav")):
                f = tar.extractfile(m)
                if f is not None:
                    items.append((m.name, f.read()))
            if len(items) > n + 8:  # a little headroom for undecodable tail entries
                break
    except Exception:
        pass  # truncation at the end of the prefix is expected
    if not items:
        sys.exit("[nirantar] no audio members extracted — increase --mb.")
    return _write_slice(items, n)


def _ingest_local(n: int) -> list[dict]:
    src = REPO / "data" / "nirantar"
    if not src.exists():
        sys.exit(f"[nirantar] {src} not found. Use the default e2e-prefix method, or download a slice there.")
    items = [(p.name, p.read_bytes()) for p in sorted(src.rglob("*.flac")) + sorted(src.rglob("*.wav"))]
    return _write_slice(items, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=200, help="MB of part aa to Range-download")
    ap.add_argument("--n", type=int, default=24, help="clips to keep")
    ap.add_argument("--local", action="store_true", help="ingest from data/nirantar/ instead")
    ap.add_argument("--out-dir", default=None, help="override output dir (testing)")
    args = ap.parse_args()

    manifest = _ingest_local(args.n) if args.local else _fetch_e2e_prefix(args.mb, args.n)
    if not manifest:
        sys.exit("[nirantar] nothing ingested.")
    (ROOT / "nirantar" / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    print(f"[nirantar] wrote dhvani_eval/nirantar/ with {len(manifest)} clips (official-e2e-prefix)")


if __name__ == "__main__":
    main()
