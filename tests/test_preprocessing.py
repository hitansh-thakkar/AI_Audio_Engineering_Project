import numpy as np
from omegaconf import OmegaConf

from src.preprocessing import Preprocessor, chunk_audio
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


def _clip(seconds, sr=16000):
    return AudioClip(samples=np.zeros(int(seconds * sr), dtype=np.float32),
                     sample_rate=sr, source_path="x.wav")


def test_chunk_splits_long_clip():
    chunks = chunk_audio(_clip(12.0), chunk_seconds=5.0)
    assert len(chunks) == 3                      # 5 + 5 + 2
    assert abs(chunks[0].duration - 5.0) < 1e-6
    assert abs(chunks[1].duration - 5.0) < 1e-6
    assert abs(chunks[2].duration - 2.0) < 1e-6
    # chunks reconstruct the original length exactly
    assert sum(len(c.samples) for c in chunks) == int(12.0 * 16000)


def test_chunk_noop_for_short_clip():
    chunks = chunk_audio(_clip(8.0), chunk_seconds=120.0)
    assert len(chunks) == 1 and chunks[0].duration == 8.0


def test_chunk_disabled():
    assert len(chunk_audio(_clip(300.0), chunk_seconds=0)) == 1
