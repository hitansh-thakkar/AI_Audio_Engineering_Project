from pathlib import Path

import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "dhvani_eval"


@pytest.fixture
def repo_root() -> Path:
    return REPO


@pytest.fixture
def sequential_clip() -> Path:
    return EVAL / "sequential" / "audio" / "clip_01.wav"


@pytest.fixture
def mixture_clip() -> Path:
    return EVAL / "overlap_mixtures" / "mix" / "mix_000.wav"


@pytest.fixture
def mixture_rttm_dir() -> Path:
    return EVAL / "overlap_mixtures" / "rttm"


def load_wav(path: Path):
    y, sr = sf.read(path)
    return y, sr
