import numpy as np
from omegaconf import OmegaConf

from src.preprocessing import Preprocessor
from src.types import AudioClip


def _cfg(**over):
    base = dict(target_sample_rate=16000, mono=True, normalize="peak",
                denoise="none", trim_silence=True, trim_top_db=30, chunk_seconds=120)
    base.update(over)
    return OmegaConf.create(base)


def test_canonicalizes_to_16k_mono(sequential_clip):
    pre = Preprocessor(_cfg())
    clip = pre.run(str(sequential_clip))
    assert isinstance(clip, AudioClip)
    assert clip.sample_rate == 16000
    assert clip.samples.ndim == 1
    assert clip.samples.dtype == np.float32
    assert clip.duration > 0


def test_peak_normalization_bounds():
    # stereo, 8 kHz, loud -> should become mono 16 k, peak ~0.9
    sr = 8000
    y = np.stack([np.sin(np.linspace(0, 100, sr)) * 3.0,
                  np.sin(np.linspace(0, 100, sr)) * 3.0], axis=1).astype(np.float32)
    clip = AudioClip(samples=y, sample_rate=sr, source_path="synthetic.wav")
    out = Preprocessor(_cfg(trim_silence=False)).run(clip)
    assert out.sample_rate == 16000
    assert out.samples.ndim == 1
    assert np.max(np.abs(out.samples)) <= 0.9 + 1e-3
