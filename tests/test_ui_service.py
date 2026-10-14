"""The UI service layer must transcode uploads, run the pipeline, and shape per-speaker
results the way the Streamlit app expects."""
import soundfile as sf
from omegaconf import OmegaConf

from src.config import load_config
from src.ui.service import envelope, run_on_file, speaker_color, transcode_to_wav


def _stub_asr_config(tmp_path):
    """Default config with the ASR stub so the UI test stays fast + offline."""
    cfg = load_config("config/default.yaml")
    cfg.asr.backend = "stub"
    cfg.postprocessing.backend = "stub"
    path = tmp_path / "ui_stub.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def test_transcode_produces_16k_mono_wav(sequential_clip, tmp_path):
    out = transcode_to_wav(sequential_clip, tmp_path / "in.wav")
    y, sr = sf.read(out)
    assert sr == 16000
    assert y.ndim == 1               # mono


def test_envelope_is_normalized_and_bounded(sequential_clip, tmp_path):
    wav = transcode_to_wav(sequential_clip, tmp_path / "in.wav")
    env = envelope(wav, buckets=200)
    assert len(env) <= 200
    assert env.min() >= 0.0 and env.max() <= 1.0 + 1e-6


def test_speaker_colors_are_stable_and_distinct():
    assert speaker_color(0) != speaker_color(1)
    assert speaker_color(0) == speaker_color(0)


def test_run_on_file_returns_per_speaker_results(sequential_clip, tmp_path):
    result = run_on_file(sequential_clip, _stub_asr_config(tmp_path), tmp_path)
    assert result.num_speakers >= 1
    assert len(result.speakers) == result.num_speakers
    for spk in result.speakers:
        assert spk.color.startswith("#")
        assert (tmp_path / sequential_clip.stem / "speakers" / f"{spk.name}.wav").exists()
        assert spk.speaking_seconds >= 0.0
    assert result.duration > 0.0
