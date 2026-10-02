"""Phase-0 acceptance: end-to-end run produces a Report + RunManifest with all five stages."""
import json

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
    for fname in ("report.json", "report.md", "timeline.json", "manifest.json"):
        assert (out_dir / fname).exists(), f"missing {fname}"

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert "summary" in report and "dialogue" in report
    assert len(report["dialogue"]) >= 1
