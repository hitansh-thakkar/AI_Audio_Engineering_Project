"""Guard the diarization accuracy we worked to achieve: real-mode DER on labeled clips must
stay under a sane bound. Offline (MFCC embeddings) so it runs without downloading ECAPA."""
from pathlib import Path

from src.evaluation.benchmark import benchmark_diarization

DIAR = Path(__file__).resolve().parent.parent / "dhvani_eval" / "sequential"


def test_real_mode_der_stays_under_bound():
    r = benchmark_diarization(DIAR, embedding="mfcc", clustering="spectral", limit=4)
    assert r["n_clips"] == 4
    assert r["der_median"] is not None and r["der_median"] < 0.45
