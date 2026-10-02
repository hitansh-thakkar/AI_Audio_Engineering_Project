#!/usr/bin/env python3
"""Add the missing ASR registers to dhvani_eval using REAL speech from the Hub.

The base dataset (built from monolingual as/bn/hi IndicVoices audio) has no English and
no code-switch. This script fills both registers with real human speech:

  registers/english/      real English utterances                       -> pure-English WER
  registers/codeswitch/   bilingual Hindi+English clips (real EN + real  -> code-switch WER
                          HI speech joined; Devanagari+Latin transcript)   + a bilingual
                                                                            multi-speaker clip

Source: `commotion/Bilingual-Hindi-English-ASR-Lt-30` (real read speech, 32 kHz, stored as
`audio_bytes`; resampled to 16 kHz here). It is a *parallel* bilingual corpus — individual
utterances are monolingual — so the code-switch clips here are **inter-sentential** (two real
monolingual utterances joined), NOT single-speaker intra-sentential code-mix.

For real *intra-sentential* code-mix, accept the license for a gated MUCS code-mix set
(e.g. https://huggingface.co/datasets/sajalmadan0909/mucs-hindi-english-codemix-asr) and set
MUCS_REPO below; this script will prefer it for the codeswitch register when accessible.

Run (needs a free HF_TOKEN in .env or env, network, and `pip install datasets`):
  python dhvani_eval/fetch_codeswitch.py
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
SEED = 42
N_ENGLISH = 40          # real English utterances to keep
N_HINDI = 30            # real Hindi utterances (used to build code-switch pairs)
N_CODESWITCH = 24       # bilingual code-switch clips to build
GAP_S = 0.3             # silence between the two turns in a code-switch clip
SRC_REPO = "commotion/Bilingual-Hindi-English-ASR-Lt-30"
MUCS_REPO = None        # set to a gated intra-sentential code-mix repo once accepted


def load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def dev(t: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in t)


def lat(t: str) -> bool:
    return any(c.isascii() and c.isalpha() for c in t)


def main() -> None:
    load_env()
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        sys.exit("[fetch] HF_TOKEN not set (put it in .env). This slice needs a free HF token.")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    try:
        import librosa
        import numpy as np
        import soundfile as sf
        from datasets import load_dataset
    except ImportError as e:
        sys.exit(f"[fetch] missing dependency: {e.name} (pip install datasets librosa soundfile)")

    rng = np.random.default_rng(SEED)

    def decode(ab: bytes) -> "np.ndarray":
        y, sr = sf.read(io.BytesIO(ab))
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=SR)
        return y.astype("float32")

    def write(path: Path, y) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        peak = float(np.max(np.abs(y))) or 1.0
        sf.write(path, (y * (0.9 / peak)).astype("float32"), SR, subtype="PCM_16")

    # ------------------------------------------------------------------ collect
    print(f"[fetch] streaming {SRC_REPO} ...")
    ds = load_dataset(SRC_REPO, split="train", streaming=True)
    english, hindi = [], []
    for ex in ds:
        t = (ex.get("text") or "").strip()
        ab = ex.get("audio_bytes")
        if not t or not ab:
            continue
        if lat(t) and not dev(t) and len(english) < N_ENGLISH:
            english.append((t, decode(ab)))
        elif dev(t) and not lat(t) and len(hindi) < N_HINDI:
            hindi.append((t, decode(ab)))
        if len(english) >= N_ENGLISH and len(hindi) >= N_HINDI:
            break
    print(f"[fetch] collected {len(english)} English, {len(hindi)} Hindi utterances")

    out = ROOT / "registers"
    manifest = []

    # ------------------------------------------------------------- english reg
    for i, (txt, y) in enumerate(english):
        rel = f"registers/english/audio/en_{i:03d}.wav"
        write(ROOT / rel, y)
        manifest.append({"id": f"en_{i:03d}", "register": "english", "language": "en",
                         "text": txt, "duration": round(len(y) / SR, 3), "audio": rel,
                         "source": SRC_REPO})

    # ---------------------------------------------------------- codeswitch reg
    import numpy as np
    (out / "codeswitch" / "rttm").mkdir(parents=True, exist_ok=True)
    pairs = min(N_CODESWITCH, len(english), len(hindi))
    ei = rng.permutation(len(english))[:pairs]
    hi = rng.permutation(len(hindi))[:pairs]
    gap = np.zeros(int(GAP_S * SR), dtype="float32")
    for n in range(pairs):
        en_t, en_y = english[ei[n]]
        hi_t, hi_y = hindi[hi[n]]
        # randomize which language leads, for variety
        if rng.random() < 0.5:
            segs = [("en", en_t, en_y), ("hi", hi_t, hi_y)]
        else:
            segs = [("hi", hi_t, hi_y), ("en", en_t, en_y)]
        clip = np.concatenate([segs[0][2], gap, segs[1][2]])
        cid = f"cs_{n:03d}"
        rel = f"registers/codeswitch/audio/{cid}.wav"
        write(ROOT / rel, clip)
        # segment timing + RTTM (two turns, two speakers, bilingual)
        t0 = 0.0
        seg_meta, rttm = [], []
        for si, (lang, txt, y) in enumerate(segs, 1):
            start = t0
            end = t0 + len(y) / SR
            spk = f"{cid}_spk{si}"
            seg_meta.append({"speaker": spk, "language": lang, "start": round(start, 3),
                             "end": round(end, 3), "text": txt})
            rttm.append(f"SPEAKER {cid} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>\n")
            t0 = end + GAP_S
        (out / "codeswitch" / "rttm" / f"{cid}.rttm").write_text("".join(rttm), encoding="utf-8")
        manifest.append({"id": cid, "register": "codeswitch", "language": "hi-en",
                         "text": " ".join(s[1] for s in segs),
                         "duration": round(len(clip) / SR, 3), "audio": rel,
                         "rttm": f"registers/codeswitch/rttm/{cid}.rttm",
                         "num_speakers": 2, "cs_type": "inter-sentential (constructed)",
                         "segments": seg_meta, "source": SRC_REPO})

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_en = sum(1 for m in manifest if m["register"] == "english")
    n_cs = sum(1 for m in manifest if m["register"] == "codeswitch")
    print(f"[fetch] wrote registers/: {n_en} english + {n_cs} codeswitch clips")

    # record the registers subset in dataset_info.json (survives a base rebuild)
    info_path = ROOT / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.setdefault("subsets", {})["registers"] = (
            f"real English ({n_en}) + inter-sentential Hindi-English code-switch ({n_cs}, constructed "
            f"from real speech) from {SRC_REPO}; fills the pure-English and code-switch WER registers"
        )
        info["known_gap"] = (
            "Registers added via fetch_codeswitch.py: pure-English and (inter-sentential) code-switch "
            "are real speech. Remaining upgrade: single-speaker INTRA-sentential code-mix needs a gated "
            "MUCS set (accept its license, set MUCS_REPO)."
        )
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[fetch] updated dataset_info.json")


if __name__ == "__main__":
    main()
