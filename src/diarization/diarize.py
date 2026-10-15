"""Stage 2 — diarization (the backbone): who spoke when.

- source=oracle: load ground-truth RTTM (for the error-propagation ablation).
- source=real:
    * backend=precision2 (default): upload to pyannote.ai cloud, poll result.
      Requires PYANNOTE_API_KEY in environment.  Falls back to hand-rolled
      clustering when the key is absent or the API call fails.
    * backend=community1 / legacy31 / anything else: energy VAD → windowed
      segments → REQUIRED hand-applied Spectral/GMM clustering → merge turns.

Emits a DiarizationResult (+ a timeline JSON written by the orchestrator).
Degrades gracefully: no speech -> 1 speaker over the whole clip, noted.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import load_env
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

    # ---------------------------------------------------- precision-2 cloud API
    def _precision2(self, clip: AudioClip, notes: list[str]) -> DiarizationResult | None:
        """Try pyannote.ai precision-2.  Returns None on any failure so the caller
        can transparently fall back to the hand-rolled path."""
        load_env()
        key_env = str(getattr(self.cfg, "api_key_env", "PYANNOTE_API_KEY"))
        api_key = os.environ.get(key_env)
        if not api_key:
            notes.append(
                f"pyannote.ai backend requested but {key_env} is not set; "
                "falling back to hand-rolled clustering"
            )
            return None
        try:
            from .pyannote_api import PyannoteAPIBackend
            model = str(getattr(self.cfg, "model", "precision-2"))
            backend = PyannoteAPIBackend(
                api_key=api_key,
                model=model,
                num_speakers=self.cfg.num_speakers,
                min_speakers=self.cfg.min_speakers,
                max_speakers=self.cfg.max_speakers,
            )
            return backend.diarize(
                clip.samples,
                clip.sample_rate,
                notes,
                audio_path=getattr(clip, "source_path", None),
            )
        except Exception as exc:
            detail = str(exc).replace("\n", " ")[:240]
            notes.append(
                f"pyannote.ai API error ({type(exc).__name__}: {detail}); "
                "falling back to hand-rolled clustering"
            )
            return None

    # -------------------------------------------------------------------- real
    def _real(self, clip: AudioClip, notes: list[str]) -> DiarizationResult:
        # --- try cloud backend first ---
        if str(getattr(self.cfg, "backend", "community1")) == "precision2":
            result = self._precision2(clip, notes)
            if result is not None:
                return result
            # fall through to hand-rolled path

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
        # regions (simultaneous speech) come from the pyannote detector when enabled; otherwise
        # they are inferred from overlapping diarized turns (fast, no gated model).
        detector = OverlapDetector(self.cfg, self.run_cfg)
        overlap_regions = detector.detect(clip, notes) if detector.enabled else []
        if not overlap_regions:
            overlap_regions = compute_overlaps(merged)
            if overlap_regions:
                notes.append("overlap regions inferred from overlapping diarized turns")
            else:
                notes.append("no overlapping speech detected")
        return DiarizationResult(turns=merged, overlap_regions=overlap_regions, num_speakers=num)
