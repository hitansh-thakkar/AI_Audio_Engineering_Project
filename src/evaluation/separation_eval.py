"""Measure overlap-separation quality on the dhvani_eval/overlap_mixtures benchmark.

Stage-level (no preprocessing trim/normalize, no ASR/LLM) so the mixture fed to the
separator matches the clean s1/s2 references exactly. For every overlap region we take
the separated source assigned to each speaker and score it against that speaker's clean
reference, using the raw mixture as the no-separation baseline:

    SI-SNRi = SI-SNR(separated | ref) - SI-SNR(mixture | ref)     (higher = better)

Reports per-region and median across the set. This is the number we iterate on.

    python -m src.evaluation.separation_eval [--root dhvani_eval/overlap_mixtures] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from ..diarization.rttm import compute_overlaps, read_rttm
from ..separation import Separator
from ..types import AudioClip, DiarizationResult
from .metrics import sdri, si_snri


def _sep_cfg(backend: str, model: str, model_sr: int) -> DictConfig:
    return OmegaConf.create(dict(
        mode="overlap", backend=backend, model=model, model_sample_rate=model_sr,
        max_speakers_supported=2, context_seconds=3.0))


def _scale_align(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale est to best match ref's amplitude (neural separators output arbitrary scale);
    required before raw SDR, which is not scale-invariant. SI-SNR needs no such step."""
    est = est.astype(np.float64)
    ref = ref.astype(np.float64)
    alpha = float(np.dot(est, ref) / (np.dot(est, est) + eps))
    return alpha * est


def _speaker_ref_paths(manifest_row: dict, root: Path) -> dict[str, Path]:
    """Map diarization speaker label (spk1, spk2, ...) -> clean reference wav."""
    sources = manifest_row["sources"]
    mapping: dict[str, Path] = {}
    for seg in manifest_row["segments"]:
        spk = seg["speaker"]
        if spk not in mapping:
            mapping[spk] = root.parent / sources[len(mapping)]
    return mapping


def evaluate(root: Path, backend: str, model: str, model_sr: int,
             limit: int | None = None) -> dict:
    manifest = [json.loads(line) for line in
                (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        manifest = manifest[:limit]

    sep = Separator(_sep_cfg(backend, model, model_sr))
    sep.load()

    si_vals: list[float] = []
    sdr_vals: list[float] = []
    rows: list[dict] = []
    for row in manifest:
        mix, sr = sf.read(root.parent / row["mix"])
        mix = mix.astype(np.float32)
        clip = AudioClip(samples=mix, sample_rate=sr, source_path=str(row["mix"]))
        turns = read_rttm(root / "rttm" / f"{row['id']}.rttm")
        overlaps = compute_overlaps(turns)
        diar = DiarizationResult(turns=turns, overlap_regions=overlaps,
                                 num_speakers=len({t.speaker for t in turns}))
        notes: list[str] = []
        separated = sep.run(clip, diar, notes)

        refs = _speaker_ref_paths(row, root)
        ref_wave = {spk: sf.read(p)[0].astype(np.float32) for spk, p in refs.items()}

        for src in separated:
            if src.speaker not in ref_wave:
                continue
            r0, r1 = src.region
            i0, i1 = int(r0 * sr), int(r1 * sr)
            est = src.samples
            ref = ref_wave[src.speaker][i0:i1]
            mixseg = mix[i0:i1]
            n = min(len(est), len(ref), len(mixseg))
            if n < sr // 10:          # skip sub-100 ms slivers
                continue
            si = si_snri(est[:n], ref[:n], mixseg[:n])
            sd = sdri(_scale_align(est[:n], ref[:n]), ref[:n], mixseg[:n])
            si_vals.append(si)
            sdr_vals.append(sd)
            rows.append({"id": row["id"], "speaker": src.speaker,
                         "region": [r0, r1], "si_snri": si, "sdri": sd})

    summary = {
        "backend": backend, "model": model, "model_sr": model_sr,
        "available": sep._available, "load_error": sep._load_error,
        "n_scored": len(si_vals),
        "si_snri_median": round(statistics.median(si_vals), 4) if si_vals else None,
        "si_snri_mean": round(statistics.fmean(si_vals), 4) if si_vals else None,
        "sdri_median": round(statistics.median(sdr_vals), 4) if sdr_vals else None,
        "rows": rows,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="overlap separation benchmark (SI-SNRi)")
    ap.add_argument("--root", default="dhvani_eval/overlap_mixtures")
    ap.add_argument("--backend", default="sepformer")
    ap.add_argument("--model", default="speechbrain/sepformer-wsj02mix")
    ap.add_argument("--model-sr", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    summary = evaluate(Path(args.root), args.backend, args.model, args.model_sr, args.limit)
    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
