"""Separated window-streams must be attributed to the right speaker by matching each
stream against the mixture in that speaker's SOLO region (where the mix IS that speaker)."""
import numpy as np

from src.separation.assign import attribute_streams
from src.types import DiarizationResult, SpeakerTurn

SR = 16000


def _tone(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _window_scene():
    """0-5 s window. SPEAKER_00 (220 Hz) active 0-3 s, SPEAKER_01 (440 Hz) active 2-5 s,
    overlap 2-3 s. Mixture = sum; in each solo region the mix equals that one speaker."""
    n = 5 * SR
    spk0 = np.zeros(n, np.float32)
    spk0[: 3 * SR] = _tone(220, 3 * SR)
    spk1 = np.zeros(n, np.float32)
    spk1[2 * SR:] = _tone(440, 3 * SR)
    mix = spk0 + spk1
    diar = DiarizationResult(
        turns=[SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(2.0, 5.0, "SPEAKER_01")],
        overlap_regions=[(2.0, 3.0)], num_speakers=2)
    return (0.0, 5.0), diar, mix, spk0, spk1


def test_streams_attributed_by_solo_region_match():
    window, diar, mix, spk0, spk1 = _window_scene()
    # streams in the "wrong" order: stream 0 is the 440 Hz (speaker 1) voice
    mapping = attribute_streams([spk1, spk0], window, diar, mix, SR)
    assert mapping == {0: "SPEAKER_01", 1: "SPEAKER_00"}


def test_empty_streams_map_to_nothing():
    window, diar, mix, _, _ = _window_scene()
    assert attribute_streams([], window, diar, mix, SR) == {}
