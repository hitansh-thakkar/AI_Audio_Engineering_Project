"""Orchestrator: raw audio -> Report, accumulating a RunManifest (timing, memory, notes,
versions, seeds, config hash, git SHA). Stages run in order, threading typed dataclasses.

Default order is diarization-first (our design); `run.order: sep_first` is the literal
proposal baseline, deferred to the post-midsem ablation (noted, not silently altered).

CLI:  python -m src.pipeline --audio <path> [--config config/default.yaml] [--out outputs/]
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from .asr import ASRStub
from .config import config_hash, git_sha, load_config
from .diarization import Diarizer
from .diarization.rttm import write_rttm
from .postprocessing import PostProcessorStub
from .preprocessing import Preprocessor, chunk_audio
from .separation import Separator
from .separation.stitch import build_speaker_tracks
from .stage import measure, set_seeds
from .types import (
    DiarizationResult,
    Report,
    RunManifest,
    SpeakerTurn,
    StageResult,
    Transcript,
    TranscriptSegment,
)


def _write_speaker_tracks(clip, diarization, separated, out_dir: Path) -> list[str]:
    """Write one full-length WAV per speaker: original audio in their solo turns, their
    assigned separated source inside overlap regions (mix fallback if unassigned)."""
    speaker_dir = out_dir / "speakers"
    speaker_dir.mkdir(exist_ok=True)
    tracks = build_speaker_tracks(
        clip, diarization.turns, diarization.overlap_regions, separated)
    paths: list[str] = []
    for speaker in sorted(tracks):
        path = speaker_dir / f"{speaker}.wav"
        sf.write(path, tracks[speaker], clip.sample_rate)
        paths.append(str(path))
    return paths


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


def _timed(fn):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak / (1024 * 1024)


def _run_chunked(chunks, chunk_seconds, diar, sep, asr, stages):
    """Process a long clip window-by-window: offset each chunk's timestamps and accumulate
    one StageResult per stage (summed time, peak memory, notes). Speaker labels are per-chunk
    — cross-chunk speaker linking is future work (noted in the manifest)."""
    names = ("diarization", "separation", "asr")
    acc_t: dict[str, float] = {n: 0.0 for n in names}
    acc_m: dict[str, float] = {n: 0.0 for n in names}
    acc_notes: dict[str, list[str]] = {n: [] for n in names}
    all_turns: list[SpeakerTurn] = []
    all_overlaps: list[tuple[float, float]] = []
    all_segments: list[TranscriptSegment] = []
    all_separated = []

    for idx, ch in enumerate(chunks):
        off = idx * chunk_seconds
        dnotes: list[str] = []
        dres, dt, dm = _timed(lambda ch=ch, n=dnotes: diar.run(ch, n))
        snotes: list[str] = []
        separated, st, sm = _timed(lambda ch=ch, d=dres, n=snotes: sep.run(ch, d, n))
        anotes: list[str] = []
        tr, at, am = _timed(lambda ch=ch, d=dres, s=separated, n=anotes: asr.run(ch, d, s, n))

        for name, dur, mem, ns in (("diarization", dt, dm, dnotes), ("separation", st, sm, snotes),
                                   ("asr", at, am, anotes)):
            acc_t[name] += dur
            acc_m[name] = max(acc_m[name], mem)
            acc_notes[name] += [f"chunk{idx}: {x}" for x in ns]

        all_turns += [SpeakerTurn(t.start + off, t.end + off, t.speaker) for t in dres.turns]
        all_overlaps += [(s + off, e + off) for (s, e) in dres.overlap_regions]
        all_separated += separated
        all_segments += [TranscriptSegment(s.start + off, s.end + off, s.speaker, s.text, s.words, s.lang)
                         for s in tr.segments]

    note = f"processed {len(chunks)} chunks of <= {chunk_seconds:g}s (per-chunk speaker labels)"
    versions = {"diarization": diar.version, "separation": sep.version, "asr": asr.version}
    for name in names:
        stages.append(StageResult(name=name, version=versions[name],
                                  wall_time_s=round(acc_t[name], 4), peak_mem_mb=round(acc_m[name], 2),
                                  notes=acc_notes[name] + [note]))
    diarization = DiarizationResult(
        turns=sorted(all_turns, key=lambda t: t.start), overlap_regions=all_overlaps,
        num_speakers=len({t.speaker for t in all_turns}))
    return diarization, all_separated, Transcript(segments=all_segments)


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
    processed_audio_path = out_dir / "processed.wav"
    sf.write(processed_audio_path, clip.samples, clip.sample_rate)

    # --- long files are processed in <= chunk_seconds windows (one window for short clips)
    chunk_seconds = float(cfg.preprocessing.chunk_seconds)
    chunks = chunk_audio(clip, chunk_seconds)

    diar = Diarizer(cfg.diarization, cfg.run)
    diar.load()
    sep = Separator(cfg.separation)
    sep.load()
    asr = ASRStub(cfg.asr)
    asr.load()

    if len(chunks) == 1:
        # --- Stage 2: diarization (backbone)
        with measure(diar.name, diar.version, stages) as notes:
            diarization = diar.run(clip, notes)
        # --- Stage 3: separation (surgical): context-window separate + attribute to speakers
        with measure(sep.name, sep.version, stages) as notes:
            separated = sep.run(clip, diarization, notes)
        # --- Stage 4: ASR + alignment (stub pre-midsem)
        with measure(asr.name, asr.version, stages) as notes:
            transcript = asr.run(clip, diarization, separated, notes)
    else:
        diarization, separated, transcript = _run_chunked(
            chunks, chunk_seconds, diar, sep, asr, stages)

    if bool(cfg.diarization.emit_timeline_json):
        (out_dir / "timeline.json").write_text(
            json.dumps(diarization.to_timeline_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # RTTM: the standard NIST diarization format, alongside the timeline JSON
    rttm_path = out_dir / f"{clip_id}.rttm"
    write_rttm(diarization.turns, clip_id, rttm_path)
    speaker_audio_paths = _write_speaker_tracks(clip, diarization, separated, out_dir)

    # write any separated sources (from overlap/everywhere separation above) to disk
    separated_audio_paths: list[str] = []
    if separated:
        separated_dir = out_dir / "separated"
        separated_dir.mkdir(exist_ok=True)
        for index, source in enumerate(separated):
            source_path = separated_dir / f"source_{index:03d}_{source.speaker}.wav"
            sf.write(source_path, source.samples, clip.sample_rate)
            separated_audio_paths.append(str(source_path))

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
            "processed_audio": str(processed_audio_path),
            "speaker_audio": speaker_audio_paths,
            "separated_audio": separated_audio_paths,
            "report_json": str(out_dir / "report.json"),
            "report_md": str(out_dir / "report.md"),
            "timeline_json": str(out_dir / "timeline.json"),
            "rttm": str(rttm_path),
            "num_speakers": diarization.num_speakers,
            "overlap_regions": [
                {"start_s": round(start, 3), "end_s": round(end, 3)}
                for start, end in diarization.overlap_regions
            ],
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
