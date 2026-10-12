"""Across chunk boundaries, the same voice must collapse to one global speaker label
(and distinct voices stay distinct) — otherwise long broadcasts mislabel everyone."""
import numpy as np

from src.diarization.linking import link_speakers

SR = 16000


def _tone(freq, seconds=1.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_same_voice_in_two_chunks_gets_one_label():
    mapping = link_speakers(
        {"c0_SPEAKER_00": _tone(220), "c1_SPEAKER_00": _tone(220),
         "c1_SPEAKER_01": _tone(880)},
        SR, embedding="mfcc", clustering="spectral", seed=42)
    assert mapping["c0_SPEAKER_00"] == mapping["c1_SPEAKER_00"]   # same 220 Hz voice -> one id
    assert mapping["c1_SPEAKER_01"] != mapping["c0_SPEAKER_00"]   # 880 Hz voice -> different id


def test_single_speaker_maps_to_one_label():
    mapping = link_speakers({"c0_SPEAKER_00": _tone(300)}, SR, "mfcc", "spectral", 42)
    assert set(mapping.values()) == {"SPEAKER_00"}
