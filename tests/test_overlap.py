"""Overlap detection must convert a pyannote annotation into regions, and must degrade
gracefully (no crash, clear note, empty regions) when the gated model isn't available."""
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from src.diarization.overlap import OverlapDetector, overlap_regions_from_annotation
from src.types import AudioClip


def _cfg(detector):
    return OmegaConf.create(dict(overlap_detector=detector,
                                 model="pyannote/speaker-diarization-community-1"))


def _clip():
    return AudioClip(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000,
                     source_path="x.wav")


class _FakeAnnotation:
    def get_overlap(self):
        return [SimpleNamespace(start=1.2345, end=2.6789),
                SimpleNamespace(start=4.0, end=4.5)]


def test_extraction_rounds_and_lists_regions():
    assert overlap_regions_from_annotation(_FakeAnnotation()) == [(1.234, 2.679), (4.0, 4.5)]


def test_disabled_detector_returns_no_regions_without_loading():
    notes: list[str] = []
    assert OverlapDetector(_cfg("none"), None).detect(_clip(), notes) == []
    assert notes == []                     # disabled: must not even try to load


def test_enabled_but_unloadable_degrades_with_note(monkeypatch):
    det = OverlapDetector(_cfg("pyannote"), None)
    monkeypatch.setattr(det, "_load", lambda notes: notes.append("load failed") or False)
    notes: list[str] = []
    assert det.detect(_clip(), notes) == []
    assert any("load failed" in n for n in notes)
