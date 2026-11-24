"""Separation benchmark on dhvani_eval/overlap_mixtures (24 synthetic 2-speaker mixtures with
clean sources; overlap ratio 0.25-1.0, SIR 0/5 dB).

Two conditions per separator, both with oracle diarization so only separation is measured:
  * full   — separate the whole mixture; permutation-invariant (PIT) SI-SNRi / SDRi vs the
             clean sources. This is the standard WSJ0-2mix/Libri2Mix protocol.
  * region — what the pipeline actually does (separation.mode: overlap): separate the overlap
             region + context, keep the overlap slice, and score (a) with PIT and (b) with the
             pipeline's own stream→speaker assignment. (a)-(b) isolates assignment errors.

    python -m src.evaluation.suite.sep_bench
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from ...diarization.rttm import compute_overlaps, read_rttm
from ...separation import Separator
from ...types import AudioClip, DiarizationResult
from ..metrics import pit_si_snri, sdri, si_snri
from ..separation_eval import _speaker_ref_paths
from .common import EVAL_ROOT, log, read_jsonl, setup_logging, write_json, write_rows

SEPARATORS: list[tuple[str, str, str, int]] = [
    # (label, backend, model id, model sample rate)
    ("SepFormer WSJ0-2mix (8k)", "sepformer", "speechbrain/sepformer-wsj02mix", 8000),
    ("RE-SepFormer WSJ0-2mix (8k)", "resepformer", "speechbrain/resepformer-wsj02mix", 8000),
    ("SepFormer Libri2Mix (8k)", "sepformer", "speechbrain/sepformer-libri2mix", 8000),
    ("SepFormer WHAMR! (16k)", "whamr", "speechbrain/sepformer-whamr16k", 16000),
]


def _cfg(backend: str, model: str, sr: int, device: str):
    return OmegaConf.create(
        dict(
            mode="overlap",
            backend=backend,
            model=model,
            model_sample_rate=sr,
            max_speakers_supported=2,
            context_seconds=3.0,
            device=device,
        )
    )


def _summ(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"median": None, "mean": None}
    return {"median": round(statistics.median(vals), 3), "mean": round(statistics.fmean(vals), 3)}


def run_one(
    label: str, backend: str, model: str, model_sr: int, device: str, manifest: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = EVAL_ROOT / "overlap_mixtures"
    sep = Separator(_cfg(backend, model, model_sr, device))
    notes: list[str] = []
    if not sep._ensure_model(notes):
        return {"label": label, "error": sep._load_error}, []
    rows: list[dict[str, Any]] = []
    proc_s = audio_s = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for m in manifest:
        mix, sr = sf.read(EVAL_ROOT / m["mix"], dtype="float32")
        refs = {spk: sf.read(p, dtype="float32")[0] for spk, p in _speaker_ref_paths(m, root).items()}
        ref_list = list(refs.values())
        # -- full-mixture condition
        t0 = time.perf_counter()
        streams = sep._separate_region(mix, sr)
        proc_s += time.perf_counter() - t0
        audio_s += len(mix) / sr
        n = min(len(mix), *(len(s) for s in streams), *(len(r) for r in ref_list))
        vals, perm = pit_si_snri([s[:n] for s in streams], [r[:n] for r in ref_list], mix[:n])
        sd = [sdri(streams[p][:n], ref_list[i][:n], mix[:n]) for i, p in enumerate(perm)]
        rows.append(
            {
                "separator": label,
                "id": m["id"],
                "condition": "full",
                "overlap_ratio": m["overlap_ratio"],
                "sir_db": m["sir_db"],
                "si_snri": round(float(np.mean(vals)), 3),
                "sdri": round(float(np.mean(sd)), 3),
                "assigned_si_snri": None,
                "assignment_correct": None,
            }
        )
        # -- pipeline region condition (oracle diarization)
        turns = read_rttm(root / "rttm" / f"{m['id']}.rttm")
        diar = DiarizationResult(
            turns=turns, overlap_regions=compute_overlaps(turns), num_speakers=len({t.speaker for t in turns})
        )
        clip = AudioClip(samples=mix, sample_rate=sr, source_path=str(m["mix"]))
        out = sep.run(clip, diar, [])
        for region in diar.overlap_regions:
            i0, i1 = int(region[0] * sr), int(region[1] * sr)
            srcs = [s for s in out if s.region == region]
            if len(srcs) < 2 or i1 - i0 < sr // 10:
                continue
            k = min(i1 - i0, *(len(s.samples) for s in srcs))
            mseg = mix[i0 : i0 + k]
            rseg = {spk: r[i0 : i0 + k] for spk, r in refs.items()}
            pv, _ = pit_si_snri([s.samples[:k] for s in srcs], list(rseg.values()), mseg)
            assigned = [si_snri(s.samples[:k], rseg[s.speaker], mseg) for s in srcs if s.speaker in rseg]
            correct = len(assigned) == len(rseg) and abs(np.mean(assigned) - np.mean(pv)) < 1e-6
            rows.append(
                {
                    "separator": label,
                    "id": m["id"],
                    "condition": "region",
                    "overlap_ratio": m["overlap_ratio"],
                    "sir_db": m["sir_db"],
                    "si_snri": round(float(np.mean(pv)), 3),
                    "sdri": None,
                    "assigned_si_snri": round(float(np.mean(assigned)), 3) if assigned else None,
                    "assignment_correct": bool(correct),
                }
            )
    full = [r for r in rows if r["condition"] == "full"]
    reg = [r for r in rows if r["condition"] == "region"]
    summary = {
        "label": label,
        "model": model,
        "model_sr": model_sr,
        "device": device,
        "full_si_snri": _summ([r["si_snri"] for r in full]),
        "full_sdri": _summ([r["sdri"] for r in full]),
        "region_pit_si_snri": _summ([r["si_snri"] for r in reg]),
        "region_assigned_si_snri": _summ(
            [r["assigned_si_snri"] for r in reg if r["assigned_si_snri"] is not None]
        ),
        "assignment_accuracy": round(sum(bool(r["assignment_correct"]) for r in reg) / len(reg), 3)
        if reg
        else None,
        "full_by_overlap_ratio": {
            str(o): _summ([r["si_snri"] for r in full if r["overlap_ratio"] == o])
            for o in sorted({r["overlap_ratio"] for r in full})
        },
        "rtf": round(proc_s / audio_s, 4),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
    }
    del sep
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary, rows


def main() -> None:
    setup_logging()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = read_jsonl(EVAL_ROOT / "overlap_mixtures" / "manifest.jsonl")
    summaries, all_rows = [], []
    for spec in SEPARATORS:
        s, rows = run_one(*spec, device=device, manifest=manifest)
        log.info(
            "%s",
            {
                k: s.get(k)
                for k in (
                    "label",
                    "full_si_snri",
                    "region_pit_si_snri",
                    "assignment_accuracy",
                    "rtf",
                    "error",
                )
            },
        )
        summaries.append(s)
        all_rows += rows
    write_rows("sep_bench", all_rows)
    write_json("sep_bench", summaries)


if __name__ == "__main__":
    main()
