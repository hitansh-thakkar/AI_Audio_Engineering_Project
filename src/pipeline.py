"""Orchestrator: raw audio -> Report, accumulating a RunManifest (timing, memory, notes,
versions, seeds, config hash, git SHA). Stages run in order, threading typed dataclasses.

Default order is diarization-first (our design); `run.order: sep_first` is the literal
proposal baseline, deferred to the post-midsem ablation (noted, not silently altered).

CLI:  python -m src.pipeline --audio <path> [--config config/default.yaml] [--out outputs/]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .asr import ASRStub
from .config import config_hash, git_sha, load_config
from .diarization import Diarizer
from .postprocessing import PostProcessorStub
from .preprocessing import Preprocessor
from .separation import Separator
from .stage import measure, set_seeds
from .types import Report, RunManifest, StageResult


def _report_to_dict(report: Report) -> dict:
    return {
        "summary": report.summary,
        "tags": report.tags,
        "dialogue": [
            {"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text, "lang": s.lang}
            for s in report.dialogue
        ],
    }


def _report_to_markdown(report: Report, clip_id: str) -> str:
    lines = [f"# Report — {clip_id}", "", "## Executive summary", report.summary, ""]
    if report.tags:
        lines += ["## Keyword tags", ", ".join(report.tags), ""]
    lines += ["## Speaker-separated dialogue", ""]
    for s in report.dialogue:
        lines.append(f"- **{s.speaker}** [{s.start:.2f}–{s.end:.2f}]: {s.text}")
    return "\n".join(lines) + "\n"


def run_pipeline(audio: str, config: str = "config/default.yaml", out: str | None = None) -> RunManifest:
    cfg = load_config(config)
    seeds = set_seeds(int(cfg.run.seed))
    stages: list[StageResult] = []

    out_root = Path(out or str(cfg.run.output_dir))
    clip_id = Path(audio).stem
    out_dir = out_root / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(cfg.run.order) == "sep_first":
        # Documented divergence: keep our diarization-first design; the literal baseline
        # is a post-midsem ablation config, not silently executed here.
        pass

    # --- Stage 1: preprocessing
    pre = Preprocessor(cfg.preprocessing)
    with measure(pre.name, pre.version, stages) as notes:
        pre.load()
        clip = pre.run(audio, notes)

    # --- Stage 2: diarization (backbone)
    diar = Diarizer(cfg.diarization, cfg.run)
    with measure(diar.name, diar.version, stages) as notes:
        diar.load()
        diarization = diar.run(clip, notes)
    if bool(cfg.diarization.emit_timeline_json):
        (out_dir / "timeline.json").write_text(
            json.dumps(diarization.to_timeline_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Stage 3: separation (surgical)
    sep = Separator(cfg.separation)
    with measure(sep.name, sep.version, stages) as notes:
        sep.load()
        separated = sep.run(clip, diarization.overlap_regions, notes)

    # --- Stage 4: ASR + alignment (stub pre-midsem)
    asr = ASRStub(cfg.asr)
    with measure(asr.name, asr.version, stages) as notes:
        asr.load()
        transcript = asr.run(clip, diarization, separated, notes)

    # --- Stage 5: LLM post-processing (stub pre-midsem)
    post = PostProcessorStub(cfg.postprocessing)
    with measure(post.name, post.version, stages) as notes:
        post.load()
        report = post.run(transcript, notes)

    # --- outputs
    (out_dir / "report.json").write_text(
        json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_report_to_markdown(report, clip_id), encoding="utf-8")

    manifest = RunManifest(
        git_sha=git_sha(), config_hash=config_hash(cfg), seeds=seeds, stages=stages,
        config_snapshot=json.loads(json.dumps(_oc_to_container(cfg))),
        artifacts={
            "report_json": str(out_dir / "report.json"),
            "report_md": str(out_dir / "report.md"),
            "timeline_json": str(out_dir / "timeline.json"),
            "num_speakers": diarization.num_speakers,
            "num_overlap_regions": len(diarization.overlap_regions),
            "num_separated_sources": len(separated),
            "num_transcript_segments": len(transcript.segments),
        },
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _oc_to_container(cfg) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dhvani pipeline: raw audio -> report")
    ap.add_argument("--audio", required=True, help="path to an input .wav")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default=None, help="output dir (default: config run.output_dir)")
    args = ap.parse_args()

    manifest = run_pipeline(args.audio, args.config, args.out)
    print(f"git={manifest.git_sha[:7]} config={manifest.config_hash}")
    for s in manifest.stages:
        note = f"  [{'; '.join(s.notes)}]" if s.notes else ""
        print(f"  {s.name:<15} {s.wall_time_s:>7.3f}s  {s.peak_mem_mb:>8.2f} MB{note}")
    print("artifacts:", json.dumps(manifest.artifacts, ensure_ascii=False))


if __name__ == "__main__":
    main()
