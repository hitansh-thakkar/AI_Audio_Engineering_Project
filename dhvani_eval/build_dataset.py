#!/usr/bin/env python3
"""Build the Dhvani benchmark eval dataset from the raw stitched clips.

Reads  : indic_multispeaker_10/   (real IndicVoices utterances, stitched sequentially)
Writes : dhvani_eval/             (this directory)

Everything here is derived from the SAME real audio — no new speech is invented.
The build is fully seeded and reproducible: re-running produces byte-identical output.

Subsets produced
----------------
sequential/        the 10 original multi-speaker clips (no overlap) + RTTM + transcripts
                   -> diarization (DER on clean turns) and per-language ASR baseline.
sources/           30 isolated single-speaker utterances (sliced at manifest timestamps,
                   silence-trimmed) -> clean per-language ASR refs + atoms for mixtures.
overlap_mixtures/  synthetic 2-speaker OVERLAPPING mixtures. Each mixture ships the two
                   time-aligned ground-truth isolated sources (mix == s1 + s2 exactly),
                   so SI-SNRi / SDRi have references. Overlap drives the separation stage
                   and the separation.mode ablation.

Run:  python dhvani_eval/build_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# Config (all knobs here; seeded for reproducibility)
# --------------------------------------------------------------------------- #
SEED = 42
SR = 16_000
TRIM_TOP_DB = 30           # silence trim threshold for isolated sources
PEAK = 0.9                 # mixture peak-normalization target (avoid clipping)
# Ablation grid for the synthetic overlaps: overlap fraction x speaker level ratio.
OVERLAP_RATIOS = [0.25, 0.50, 0.75, 1.00]   # fraction of the shorter source that overlaps
SIR_DB = [0.0, 5.0]                          # target source-to-source level ratio (dB)
PAIRS_PER_CELL = 3                           # mixtures per (ratio, SIR) cell

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RAW = REPO / "indic_multispeaker_10"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_mono(path: Path) -> np.ndarray:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)


def write_wav(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y.astype(np.float32), SR, subtype="PCM_16")


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y**2))) if y.size else 0.0


def rttm_line(uri: str, start: float, dur: float, spk: str) -> str:
    return f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"


# --------------------------------------------------------------------------- #
# Data model for an isolated source utterance
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    id: str
    clip: str
    speaker_global: str   # unique across the whole dataset
    source_speaker_id: str
    language: str
    text: str
    district: str
    duration: float
    path: str             # relative to dhvani_eval/


# --------------------------------------------------------------------------- #
# Stage 1 — sequential clips (copy + RTTM + transcripts + manifest)
# --------------------------------------------------------------------------- #
def build_sequential(clips: list[dict]) -> None:
    out = ROOT / "sequential"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "rttm").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in clips:
        cid = c["id"]
        shutil.copyfile(RAW / c["audio"], out / "audio" / f"{cid}.wav")
        # RTTM: one SPEAKER line per turn (these clips have no overlap by construction)
        rttm, lines = [], []
        for s in c["segments"]:
            spk = f"{cid}_{s['speaker']}"
            rttm.append(rttm_line(cid, s["start"], s["end"] - s["start"], spk))
            lines.append(f"[{s['start']:.2f}-{s['end']:.2f}] {spk} ({s['language']}): {s['text']}")
        (out / "rttm" / f"{cid}.rttm").write_text("".join(rttm), encoding="utf-8")
        (out / "transcripts" / f"{cid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({
            "id": cid, "audio": f"sequential/audio/{cid}.wav",
            "rttm": f"sequential/rttm/{cid}.rttm",
            "duration": c["duration"], "num_speakers": c["num_speakers"],
            "overlap_regions": [], "languages": c["languages"],
            "segments": c["segments"],
        })
    _write_jsonl(out / "manifest.jsonl", manifest)
    print(f"sequential/        {len(manifest)} clips")


# --------------------------------------------------------------------------- #
# Stage 2 — isolated single-speaker sources
# --------------------------------------------------------------------------- #
def build_sources(clips: list[dict]) -> list[Source]:
    out = ROOT / "sources"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    sources: list[Source] = []
    for c in clips:
        y = load_mono(RAW / c["audio"])
        for i, s in enumerate(c["segments"], 1):
            i0, i1 = int(s["start"] * SR), int(s["end"] * SR)
            seg = y[i0:i1]
            seg, _ = librosa.effects.trim(seg, top_db=TRIM_TOP_DB)
            sid = f"{c['id']}_s{i}_{s['language']}"
            rel = f"sources/audio/{sid}.wav"
            write_wav(ROOT / rel, seg)
            sources.append(Source(
                id=sid, clip=c["id"],
                speaker_global=f"{c['id']}_{s['speaker']}",
                source_speaker_id=s["source_speaker_id"], language=s["language"],
                text=s["text"], district=s.get("source_district", ""),
                duration=round(len(seg) / SR, 3), path=rel,
            ))
    _write_jsonl(out / "manifest.jsonl", [asdict(s) for s in sources])
    print(f"sources/           {len(sources)} isolated utterances")
    return sources


# --------------------------------------------------------------------------- #
# Stage 3 — synthetic overlapping mixtures (with ground-truth sources)
# --------------------------------------------------------------------------- #
def _place(buf_len: int, y: np.ndarray, offset: int) -> np.ndarray:
    buf = np.zeros(buf_len, dtype=np.float32)
    buf[offset:offset + len(y)] = y
    return buf


def build_mixtures(sources: list[Source], rng: np.random.Generator) -> None:
    out = ROOT / "overlap_mixtures"
    for sub in ("mix", "s1", "s2", "rttm"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # cache audio
    audio = {s.id: load_mono(ROOT / s.path) for s in sources}
    manifest, n = [], 0

    for ratio in OVERLAP_RATIOS:
        for sir in SIR_DB:
            made = 0
            # deterministic candidate order: all cross-speaker, prefer cross-language pairs
            idx = list(range(len(sources)))
            rng.shuffle(idx)
            pairs = []
            for a in idx:
                for b in idx:
                    if a >= b:
                        continue
                    sa, sb = sources[a], sources[b]
                    if sa.source_speaker_id == sb.source_speaker_id:
                        continue
                    pairs.append((a, b, sa.language != sb.language))
            # cross-language first, then shuffle within groups for variety
            pairs.sort(key=lambda p: not p[2])
            for a, b, _xlang in pairs:
                if made >= PAIRS_PER_CELL:
                    break
                sa, sb = sources[a], sources[b]
                ya, yb = audio[sa.id], audio[sb.id]
                la, lb = len(ya), len(yb)

                overlap_len = int(ratio * min(la, lb))
                off_b = max(0, la - overlap_len)
                buf_len = max(la, off_b + lb)

                # scale B to hit the target source-to-source ratio (SIR) vs A
                ga, gb = rms(ya), rms(yb)
                gain = (ga / gb / (10 ** (sir / 20))) if gb > 0 else 1.0
                s1 = _place(buf_len, ya, 0)
                s2 = _place(buf_len, yb * gain, off_b)
                mix = s1 + s2

                # peak-normalize mix; apply SAME scalar to sources so mix == s1 + s2
                peak = float(np.max(np.abs(mix))) or 1.0
                k = PEAK / peak
                mix, s1, s2 = mix * k, s1 * k, s2 * k

                mid = f"mix_{n:03d}"
                write_wav(out / "mix" / f"{mid}.wav", mix)
                write_wav(out / "s1" / f"{mid}.wav", s1)
                write_wav(out / "s2" / f"{mid}.wav", s2)

                ov_start, ov_end = off_b / SR, min(la, off_b + lb) / SR
                rttm = (
                    rttm_line(mid, 0.0, la / SR, "spk1")
                    + rttm_line(mid, off_b / SR, lb / SR, "spk2")
                )
                (out / "rttm" / f"{mid}.rttm").write_text(rttm, encoding="utf-8")

                residual = float(np.max(np.abs(mix - (s1 + s2))))
                manifest.append({
                    "id": mid,
                    "mix": f"overlap_mixtures/mix/{mid}.wav",
                    "sources": [f"overlap_mixtures/s1/{mid}.wav", f"overlap_mixtures/s2/{mid}.wav"],
                    "rttm": f"overlap_mixtures/rttm/{mid}.rttm",
                    "duration": round(buf_len / SR, 3),
                    "num_speakers": 2,
                    "overlap_ratio": ratio, "sir_db": sir,
                    "overlap_regions": [[round(ov_start, 3), round(ov_end, 3)]],
                    "mix_equals_sum_residual": round(residual, 8),
                    "segments": [
                        {"speaker": "spk1", "source_id": sa.id, "language": sa.language,
                         "start": 0.0, "end": round(la / SR, 3), "text": sa.text,
                         "source_speaker_id": sa.source_speaker_id},
                        {"speaker": "spk2", "source_id": sb.id, "language": sb.language,
                         "start": round(off_b / SR, 3), "end": round((off_b + lb) / SR, 3),
                         "text": sb.text, "source_speaker_id": sb.source_speaker_id},
                    ],
                })
                n += 1
                made += 1

    _write_jsonl(out / "manifest.jsonl", manifest)
    worst = max((m["mix_equals_sum_residual"] for m in manifest), default=0.0)
    print(f"overlap_mixtures/  {len(manifest)} mixtures  (max |mix-(s1+s2)| = {worst:.2e})")


# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)
    clips = [json.loads(l) for l in (RAW / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    build_sequential(clips)
    sources = build_sources(clips)
    build_mixtures(sources, rng)

    langs = sorted({s.language for s in sources})
    info = {
        "name": "dhvani_eval",
        "built_from": "indic_multispeaker_10 (ai4bharat/IndicVoices, valid split)",
        "seed": SEED, "sample_rate": SR, "git_sha": git_sha(),
        "subsets": {
            "sequential": "multi-speaker clips, no overlap (diarization + per-language ASR baseline)",
            "sources": "isolated single-speaker utterances (clean per-language ASR; mixture atoms)",
            "overlap_mixtures": "synthetic 2-speaker overlaps + ground-truth sources (separation, SI-SNRi/SDRi)",
        },
        "languages": langs,
        "known_gap": "No Hindi-English code-switch audio: source is monolingual (as/bn/hi). "
                     "Per-register WER is by language here; see fetch_codeswitch.py for a real CS slice.",
    }
    (ROOT / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset_info.json  languages={langs} git={info['git_sha'][:7]}")


if __name__ == "__main__":
    main()
