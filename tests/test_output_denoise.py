"""Denoise belongs on the OUTPUT per-speaker tracks, not on the signal that feeds
diarization embeddings (denoising there over-counts speakers). So _write_speaker_tracks
applies it only when asked, leaving the diarization/separation path untouched."""
import numpy as np
import soundfile as sf

from src.pipeline import _write_speaker_tracks
from src.types import AudioClip, DiarizationResult, SpeakerTurn

SR = 16000


def _noisy_clip():
    rng = np.random.default_rng(0)
    t = np.arange(3 * SR) / SR
    samples = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(3 * SR)).astype(np.float32)
    clip = AudioClip(samples, SR, "c")
    diar = DiarizationResult(turns=[SpeakerTurn(0.0, 3.0, "SPEAKER_00")],
                             overlap_regions=[], num_speakers=1)
    return clip, diar


def test_output_denoise_changes_track_only_when_enabled(tmp_path):
    clip, diar = _noisy_clip()
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = _write_speaker_tracks(clip, diar, [], tmp_path / "off", denoise_output=False)
    on = _write_speaker_tracks(clip, diar, [], tmp_path / "on", denoise_output=True)
    off_t = sf.read(off[0])[0].astype(np.float32)
    on_t = sf.read(on[0])[0].astype(np.float32)
    assert np.allclose(off_t, clip.samples, atol=1e-3)   # off: original audio passes through
    assert not np.allclose(on_t, off_t, atol=1e-3)       # on: denoise actually altered the output
