"""Separated sources must be rescaled to mixture level so they don't clip when written
to 16-bit PCM (SepFormer emits arbitrary, often large, scale)."""
import numpy as np
from omegaconf import OmegaConf

from src.separation.separate import Separator, _scale_to_reference


def _sep_cfg(backend, model="x"):
    return OmegaConf.create(dict(mode="overlap", backend=backend, model=model,
                                 model_sample_rate=8000, max_speakers_supported=2,
                                 context_seconds=3.0))


def test_unknown_backend_degrades_with_note():
    notes: list[str] = []
    assert Separator(_sep_cfg("bogus"))._ensure_model(notes) is False
    assert any("not wired" in n for n in notes)


def test_convtasnet_without_asteroid_degrades_with_note():
    notes: list[str] = []
    ok = Separator(_sep_cfg("convtasnet", "mpariente/ConvTasNet_Libri2Mix_sepclean_16k"))._ensure_model(notes)
    # asteroid isn't a hard dependency; absent it must degrade cleanly, not crash
    if not ok:
        assert any("asteroid" in n.lower() or "convtasnet" in n.lower() for n in notes)


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
