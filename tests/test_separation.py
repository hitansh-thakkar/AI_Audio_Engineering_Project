"""Separated sources must be rescaled to mixture level so they don't clip when written
to 16-bit PCM (SepFormer emits arbitrary, often large, scale)."""
import numpy as np

from src.separation.separate import _scale_to_reference


def test_scale_recovers_reference_level():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(8000).astype(np.float32)
    est = 16.0 * ref                      # neural separator: right shape, 16x too loud
    scaled = _scale_to_reference(est, ref)
    assert np.allclose(scaled, ref, atol=1e-4)


def test_scale_prevents_clipping_against_normal_mix():
    t = np.arange(8000) / 16000
    ref = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # mixture-level, peak 0.3
    est = 20.0 * ref
    scaled = _scale_to_reference(est, ref)
    assert np.max(np.abs(scaled)) <= 1.0          # safe for PCM_16
    assert np.max(np.abs(est)) > 1.0              # the raw estimate would have clipped


def test_scale_handles_silent_estimate():
    ref = np.ones(100, dtype=np.float32)
    assert np.array_equal(_scale_to_reference(np.zeros(100, dtype=np.float32), ref),
                          np.zeros(100, dtype=np.float32))
