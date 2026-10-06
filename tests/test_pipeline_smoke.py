"""Phase-0 acceptance: end-to-end run produces a Report + RunManifest with all five stages."""
import json

from omegaconf import OmegaConf

from src.config import load_config
from src.pipeline import run_pipeline


def test_end_to_end_smoke(sequential_clip, tmp_path):
    manifest = run_pipeline(str(sequential_clip), config="config/default.yaml", out=str(tmp_path))

    # all five stages timed
    names = [s.name for s in manifest.stages]
    assert names == ["preprocessing", "diarization", "separation", "asr", "postprocessing"]
    for s in manifest.stages:
        assert s.wall_time_s >= 0.0
        assert s.peak_mem_mb >= 0.0

    assert manifest.git_sha
    assert manifest.config_hash

    out_dir = tmp_path / sequential_clip.stem
    for fname in ("processed.wav", "report.json", "report.md", "timeline.json", "manifest.json"):
        assert (out_dir / fname).exists(), f"missing {fname}"
    speaker_paths = manifest.artifacts["speaker_audio"]
    assert len(speaker_paths) == manifest.artifacts["num_speakers"]
    assert all((out_dir / "speakers" / path.split("speakers\\")[-1]).exists() for path in speaker_paths)

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert "summary" in report and "dialogue" in report
    assert len(report["dialogue"]) >= 1


def test_chunking_triggers_end_to_end(sequential_clip, tmp_path):
    # force a tiny window so a ~14 s clip is processed as multiple chunks
    cfg = load_config("config/default.yaml")
    cfg.preprocessing.chunk_seconds = 5
    cfg_path = tmp_path / "chunked.yaml"
    OmegaConf.save(cfg, cfg_path)

    manifest = run_pipeline(str(sequential_clip), config=str(cfg_path), out=str(tmp_path))
    names = [s.name for s in manifest.stages]
    assert names == ["preprocessing", "diarization", "separation", "asr", "postprocessing"]
    # the chunked stages record how many windows were processed
    diar = next(s for s in manifest.stages if s.name == "diarization")
    assert any("chunks of" in n for n in diar.notes)
    assert (tmp_path / sequential_clip.stem / "timeline.json").exists()
