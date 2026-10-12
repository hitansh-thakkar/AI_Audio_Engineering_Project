"""Stage 2 — diarization (the backbone): who spoke when.

- source=oracle: load ground-truth RTTM (for the error-propagation ablation).
- source=real: energy VAD -> windowed segments -> embeddings -> REQUIRED hand-applied
  Spectral/GMM clustering -> merge into speaker turns. No prior speaker templates.

Emits a DiarizationResult (+ a timeline JSON written by the orchestrator).
Degrades gracefully: no speech -> 1 speaker over the whole clip, noted.
"""
from __future__ import annotations

from pathlib import Path

from ..types import AudioClip, DiarizationResult, SpeakerTurn
from .clustering import cluster_embeddings
from .embeddings import embed_segments
from .overlap import OverlapDetector
from .rttm import compute_overlaps, read_rttm
from .vad import energy_vad, window_segments


class Diarizer:
    name = "diarization"

    def __init__(self, cfg, run_cfg):
        self.cfg = cfg
        self.run_cfg = run_cfg
        self.version = f"{cfg.source}-{cfg.clustering}-{cfg.backend}"

    def load(self) -> None:
        pass

    def run(self, clip: AudioClip, notes: list[str] | None = None) -> DiarizationResult:
        notes = [] if notes is None else notes
        if str(self.cfg.source) == "oracle":
            return self._oracle(clip, notes)
        return self._real(clip, notes)

    # ------------------------------------------------------------------ oracle
    def _oracle(self, clip: AudioClip, notes: list[str]) -> DiarizationResult:
        if not self.cfg.rttm_dir:
            notes.append("oracle requested but diarization.rttm_dir is null; falling back to real")
            return self._real(clip, notes)
        rttm_path = Path(str(self.cfg.rttm_dir)) / f"{clip.clip_id}.rttm"
        if not rttm_path.exists():
            notes.append(f"oracle RTTM not found at {rttm_path}; falling back to real")
            return self._real(clip, notes)
        turns = read_rttm(rttm_path)
        speakers = sorted({t.speaker for t in turns})
        return DiarizationResult(
            turns=turns, overlap_regions=compute_overlaps(turns), num_speakers=len(speakers),
        )

    # -------------------------------------------------------------------- real
    def _real(self, clip: AudioClip, notes: list[str]) -> DiarizationResult:
        sr = clip.sample_rate
        top_db = int(getattr(self.cfg, "vad_top_db", 30))
        win = float(getattr(self.cfg, "window_seconds", 1.5))
        hop = float(getattr(self.cfg, "window_hop", 0.75))
        regions = energy_vad(clip.samples, sr, top_db=top_db)
        segments = window_segments(regions, win=win, hop=hop)
        if len(segments) <= 1:
            notes.append("<=1 speech segment detected; treating clip as a single speaker")
            end = clip.duration
            return DiarizationResult(
                turns=[SpeakerTurn(0.0, end, "SPEAKER_00")], overlap_regions=[], num_speakers=1,
            )

        emb = embed_segments(clip.samples, sr, segments, str(self.cfg.embedding), notes)
        labels = cluster_embeddings(
            emb, method=str(self.cfg.clustering),
            num_speakers=self.cfg.num_speakers,
            min_speakers=self.cfg.min_speakers, max_speakers=self.cfg.max_speakers,
            seed=int(self.run_cfg.seed), notes=notes,
        )

        # segment -> labelled turn, then merge adjacent same-speaker (gap < 0.5s)
        raw = [SpeakerTurn(s, e, f"SPEAKER_{labels[i]:02d}") for i, (s, e) in enumerate(segments)]
        raw.sort(key=lambda t: t.start)
        merged: list[SpeakerTurn] = []
        for t in raw:
            if merged and merged[-1].speaker == t.speaker and t.start - merged[-1].end < 0.5:
                merged[-1] = SpeakerTurn(merged[-1].start, max(merged[-1].end, t.end), t.speaker)
            else:
                merged.append(t)

        num = len(set(labels.tolist()))

        # Speaker attribution above is from the compulsory hand-applied clustering. Overlap
        # regions (simultaneous speech) come from a dedicated detector when enabled; single-
        # stream VAD cannot find them. Degrades to [] with a clear note if unavailable.
        detector = OverlapDetector(self.cfg, self.run_cfg)
        overlap_regions = detector.detect(clip, notes) if detector.enabled else []
        if not overlap_regions:
            inferred = compute_overlaps(merged)
            if inferred:
                overlap_regions = inferred
                notes.append("overlap regions inferred from overlapping diarized turns")
        if not detector.enabled:
            notes.append("overlap detection off (diarization.overlap_detector=none); "
                         "overlap_regions left empty")
        return DiarizationResult(turns=merged, overlap_regions=overlap_regions, num_speakers=num)
