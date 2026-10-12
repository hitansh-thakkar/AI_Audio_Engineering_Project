import soundfile as sf
from omegaconf import OmegaConf

from src.diarization import Diarizer
from src.diarization.rttm import read_rttm
from src.evaluation.metrics import der
from src.types import AudioClip


def _diar_cfg(source, rttm_dir=None):
    return OmegaConf.create(dict(
        source=source, backend="community1", model="x", clustering="spectral",
        emit_timeline_json=True, embedding="auto",
        num_speakers=None, min_speakers=None, max_speakers=None, rttm_dir=rttm_dir,
    ))


def _run_cfg():
    return OmegaConf.create(dict(seed=42))


def test_oracle_matches_reference(mixture_clip, mixture_rttm_dir):
    y, sr = sf.read(mixture_clip)
    clip = AudioClip(samples=y.astype("float32"), sample_rate=sr, source_path=str(mixture_clip))
    diar = Diarizer(_diar_cfg("oracle", str(mixture_rttm_dir)), _run_cfg())
    result = diar.run(clip)

    assert result.num_speakers == 2
    assert len(result.overlap_regions) >= 1          # mixtures are overlapped by construction
    ref = read_rttm(mixture_rttm_dir / "mix_000.rttm")
    assert der(ref, result.turns)["der"] == 0.0       # oracle == ground truth


def test_oracle_emits_timeline_json(mixture_clip, mixture_rttm_dir):
    y, sr = sf.read(mixture_clip)
    clip = AudioClip(samples=y.astype("float32"), sample_rate=sr, source_path=str(mixture_clip))
    result = Diarizer(_diar_cfg("oracle", str(mixture_rttm_dir)), _run_cfg()).run(clip)
    timeline = result.to_timeline_json()
    assert timeline and all({"start", "end", "speaker"} <= set(row) for row in timeline)


def test_real_mode_runs_and_clusters(sequential_clip):
    y, sr = sf.read(sequential_clip)
    clip = AudioClip(samples=y.astype("float32"), sample_rate=sr, source_path=str(sequential_clip))
    result = Diarizer(_diar_cfg("real"), _run_cfg()).run(clip)
    assert result.num_speakers >= 1
    assert len(result.turns) >= 1


def test_real_mode_honors_vad_and_window_config(monkeypatch, sequential_clip):
    import src.diarization.diarize as D
    seen: dict = {}
    real_vad, real_win = D.energy_vad, D.window_segments

    def spy_vad(samples, sr, top_db=30):
        seen["top_db"] = top_db
        return real_vad(samples, sr, top_db=top_db)

    def spy_win(regions, win=1.5, hop=0.75):
        seen["win"], seen["hop"] = win, hop
        return real_win(regions, win=win, hop=hop)

    monkeypatch.setattr(D, "energy_vad", spy_vad)
    monkeypatch.setattr(D, "window_segments", spy_win)
    cfg = _diar_cfg("real")
    cfg.vad_top_db, cfg.window_seconds, cfg.window_hop = 25, 2.0, 1.0
    y, sr = sf.read(sequential_clip)
    D.Diarizer(cfg, _run_cfg()).run(
        AudioClip(samples=y.astype("float32"), sample_rate=sr, source_path=str(sequential_clip)))
    assert seen == {"top_db": 25, "win": 2.0, "hop": 1.0}
