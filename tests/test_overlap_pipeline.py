"""End-to-end: the full pipeline on an overlap mixture must emit clean, full-length,
non-clipping per-speaker WAVs whose overlap region is separated (closer to each speaker's
clean reference than the mix). Skipped if the SepFormer weights aren't present locally."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.evaluation.metrics import si_snri
from src.pipeline import run_pipeline

REPO = Path(__file__).resolve().parent.parent
MIX_ROOT = REPO / "dhvani_eval" / "overlap_mixtures"
WEIGHTS = REPO / "models" / "sepformer-wsj02mix" / "masknet.ckpt"

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="SepFormer weights not downloaded; skipping heavy e2e")


def _row(mix_id: str) -> dict:
    for line in (MIX_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line)["id"] == mix_id:
            return json.loads(line)
    raise AssertionError(f"{mix_id} not in manifest")


def test_overlap_pipeline_emits_clean_per_speaker_tracks(tmp_path):
    mix_id = "mix_002"
    manifest = run_pipeline(str(MIX_ROOT / "mix" / f"{mix_id}.wav"),
                            config=str(REPO / "config" / "overlap_demo.yaml"), out=str(tmp_path))
    speaker_paths = manifest.artifacts["speaker_audio"]
    assert len(speaker_paths) == 2

    row = _row(mix_id)
    sr = 16000
    mix = sf.read(MIX_ROOT / "mix" / f"{mix_id}.wav")[0].astype(np.float32)
    refs = {"spk1": sf.read(REPO / "dhvani_eval" / row["sources"][0])[0].astype(np.float32),
            "spk2": sf.read(REPO / "dhvani_eval" / row["sources"][1])[0].astype(np.float32)}
    r0, r1 = row["overlap_regions"][0]
    i0, i1 = int(r0 * sr), int(r1 * sr)

    improvements = []
    for spk in ("spk1", "spk2"):
        track = sf.read(tmp_path / mix_id / "speakers" / f"{spk}.wav")[0].astype(np.float32)
        assert len(track) == len(mix)                       # full-length, timeline preserved
        assert np.max(np.abs(track)) <= 1.0                 # no PCM clipping
        assert not np.allclose(track[i0:i1], mix[i0:i1])    # overlap really was separated
        improvements.append(si_snri(track[i0:i1], refs[spk][i0:i1], mix[i0:i1]))

    # separation must help on average (both speakers positive on this balanced mixture)
    assert float(np.mean(improvements)) > 1.0
