"""The first-half benchmark must produce DER / count-accuracy for diarization and be
selectable per-section (so CI can run a fast, offline slice)."""
from pathlib import Path

from src.evaluation.benchmark import run_benchmark

REPO = Path(__file__).resolve().parent.parent
DIAR = REPO / "dhvani_eval" / "sequential"


def test_diarization_section_reports_der_and_count_accuracy(tmp_path):
    report = run_benchmark(
        diar_root=DIAR, sep_root=REPO / "dhvani_eval" / "overlap_mixtures",
        embedding="mfcc", clustering="spectral",
        diar_limit=2, sep_limit=0, sample_audio=None,     # diarization only (fast, offline)
        config="config/default.yaml", out_root=tmp_path)
    assert "separation" not in report and "latency" not in report
    diar = report["diarization"]
    assert diar["n_clips"] == 2
    assert 0.0 <= diar["der_median"] <= 1.0
    assert 0.0 <= diar["speaker_count_accuracy"] <= 1.0
