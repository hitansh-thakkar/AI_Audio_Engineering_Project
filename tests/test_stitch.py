"""Per-speaker tracks must be full-length, carry the original audio in solo turns, and
carry the speaker's SEPARATED source (not the mix) inside overlap regions."""
import numpy as np

from src.separation.stitch import build_speaker_tracks
from src.types import AudioClip, SeparatedSource, SpeakerTurn

SR = 16000


def _tone(freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _clip_and_turns():
    a = _tone(220, 2.0)
    mix = _tone(220, 1.0) + _tone(440, 1.0)   # overlap 2-3 s
    b = _tone(440, 2.0)
    clip = AudioClip(samples=np.concatenate([a, mix, b]).astype(np.float32),
                     sample_rate=SR, source_path="synthetic.wav")
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(2.0, 5.0, "SPEAKER_01")]
    overlaps = [(2.0, 3.0)]
    separated = [
        SeparatedSource(region=(2.0, 3.0), speaker="SPEAKER_00", samples=_tone(220, 1.0)),
        SeparatedSource(region=(2.0, 3.0), speaker="SPEAKER_01", samples=_tone(440, 1.0)),
    ]
    return clip, turns, overlaps, separated


def test_tracks_are_full_length_and_per_speaker():
    clip, turns, overlaps, separated = _clip_and_turns()
    tracks = build_speaker_tracks(clip, turns, overlaps, separated)
    assert set(tracks) == {"SPEAKER_00", "SPEAKER_01"}
    for t in tracks.values():
        assert len(t) == len(clip.samples)


def test_solo_region_uses_original_audio():
    clip, turns, overlaps, separated = _clip_and_turns()
    tracks = build_speaker_tracks(clip, turns, overlaps, separated)
    # speaker 0 solo = [0,2) s
    assert np.allclose(tracks["SPEAKER_00"][: 2 * SR], clip.samples[: 2 * SR])
    # speaker 1 is silent before its turn starts (2 s)
    assert np.allclose(tracks["SPEAKER_01"][: 2 * SR], 0.0)


def test_overlap_region_uses_separated_source_not_mix():
    clip, turns, overlaps, separated = _clip_and_turns()
    tracks = build_speaker_tracks(clip, turns, overlaps, separated)
    ov = slice(2 * SR, 3 * SR)
    mix = clip.samples[ov]
    assert np.allclose(tracks["SPEAKER_00"][ov], _tone(220, 1.0))
    assert np.allclose(tracks["SPEAKER_01"][ov], _tone(440, 1.0))
    assert not np.allclose(tracks["SPEAKER_00"][ov], mix)   # the whole point: not the mix


def test_unassigned_overlap_falls_back_to_mix():
    clip, turns, overlaps, _ = _clip_and_turns()
    tracks = build_speaker_tracks(clip, turns, overlaps, separated=[])
    ov = slice(2 * SR, 3 * SR)
    # with no separated sources, both speakers' overlap region passes through the mix
    assert np.allclose(tracks["SPEAKER_00"][ov], clip.samples[ov])
