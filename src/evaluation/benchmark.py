"""One-command first-half benchmark (the proposal's required System Benchmark).

Reports the numbers every optimization in this project is judged by:
- Milestone 2 (diarization): median DER + speaker-count accuracy on a labelled set.
- Milestone 1 (separation): median SI-SNRi / SDRi on the overlap-mixture set.
- Per-stage latency + peak memory on a representative clip (proposal requirement).

    python -m src.evaluation.benchmark                 # full first-half report
    python -m src.evaluation.benchmark --diar-limit 3 --sep-limit 3 --out outputs/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from ..diarization import Diarizer
from ..diarization.rttm import read_rttm
from ..types import AudioClip
from .metrics import der
from .separation_eval import evaluate as sep_evaluate


def _diar_cfg(embedding: str, clustering: str) -> DictConfig:
    return OmegaConf.create(dict(
        source="real", backend="community1", model="x", clustering=clustering,
        overlap_detector="none", emit_timeline_json=True, embedding=embedding,
        num_speakers=None, min_speakers=None, max_speakers=None, rttm_dir=None))


def benchmark_diarization(diar_root: Path, embedding: str, clustering: str,
                          limit: int | None = None) -> dict:
    rttms = sorted((diar_root / "rttm").glob("*.rttm"))
    if limit:
        rttms = rttms[:limit]
    cfg, run_cfg = _diar_cfg(embedding, clustering), OmegaConf.create(dict(seed=42))
    ders: list[float] = []
    count_err: list[int] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for r in rttms:
        wav = diar_root / "audio" / f"{r.stem}.wav"
        if not wav.exists():
            continue
        y, sr = sf.read(wav)
        clip = AudioClip(samples=y.astype(np.float32), sample_rate=sr, source_path=str(wav))
        ref = read_rttm(r)
        hyp = Diarizer(cfg, run_cfg).run(clip)
        d = der(ref, hyp.turns)["der"]
        ce = abs(hyp.num_speakers - len({t.speaker for t in ref}))
        ders.append(d)
        count_err.append(ce)
        rows.append({"clip": r.stem, "der": round(d, 4), "count_error": ce})
    return {
        "embedding": embedding, "clustering": clustering, "n_clips": len(ders),
        "der_median": round(statistics.median(ders), 4) if ders else None,
        "der_mean": round(statistics.fmean(ders), 4) if ders else None,
        "speaker_count_accuracy": (round(sum(e == 0 for e in count_err) / len(count_err), 3)
                                   if count_err else None),
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "rows": rows,
    }


def benchmark_separation(sep_root: Path, limit: int | None = None) -> dict:
    s = sep_evaluate(sep_root, "sepformer", "speechbrain/sepformer-wsj02mix", 8000, limit)
    return {k: s[k] for k in ("n_scored", "si_snri_median", "si_snri_mean", "sdri_median",
                              "available", "load_error")}


def benchmark_latency(sample_audio: Path, config: str, out_root: Path) -> list[dict]:
    from ..pipeline import run_pipeline
    manifest = run_pipeline(str(sample_audio), config=config, out=str(out_root))
    return [{"stage": s.name, "wall_time_s": s.wall_time_s, "peak_mem_mb": s.peak_mem_mb}
            for s in manifest.stages]


def run_benchmark(diar_root: Path, sep_root: Path, embedding: str, clustering: str,
                  diar_limit: int | None, sep_limit: int | None,
                  sample_audio: Path | None, config: str, out_root: Path) -> dict:
    report: dict = {}
    if diar_limit != 0:
        report["diarization"] = benchmark_diarization(diar_root, embedding, clustering, diar_limit)
    if sep_limit != 0:
        report["separation"] = benchmark_separation(sep_root, sep_limit)
    if sample_audio is not None:
        report["latency"] = benchmark_latency(sample_audio, config, out_root)
    return report


def _summary(report: dict) -> dict:
    return {k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v)
            for k, v in report.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Dhvani first-half benchmark (DER + SI-SNRi + latency)")
    ap.add_argument("--diar-set", default="dhvani_eval/sequential")
    ap.add_argument("--sep-set", default="dhvani_eval/overlap_mixtures")
    ap.add_argument("--embedding", default="ecapa")
    ap.add_argument("--clustering", default="spectral")
    ap.add_argument("--diar-limit", type=int, default=None, help="0 to skip diarization")
    ap.add_argument("--sep-limit", type=int, default=None, help="0 to skip separation")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    diar_root = Path(args.diar_set)
    sample = next((diar_root / "audio").glob("*.wav"), None)
    out_root = Path(args.out).parent if args.out else Path("outputs") / "benchmark"
    report = run_benchmark(diar_root, Path(args.sep_set), args.embedding, args.clustering,
                           args.diar_limit, args.sep_limit, sample, args.config, out_root)
    print(json.dumps(_summary(report), ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
