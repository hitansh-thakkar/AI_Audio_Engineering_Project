from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from src.asr.sarvam import SarvamASR
from src.types import AudioClip, DiarizationResult, SpeakerTurn


class _FakeSpeechToText:
    def __init__(self):
        self.calls = 0

    def transcribe(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            transcript="hello",
            timestamps=SimpleNamespace(
                words=["hello"],
                start_time_seconds=[0.1],
                end_time_seconds=[0.4],
            ),
        )


class _FakeClient:
    def __init__(self):
        self.speech_to_text = _FakeSpeechToText()


def test_sarvam_transcribes_diarization_chunks_and_offsets_timestamps(tmp_path):
    cfg = OmegaConf.create({"model": "saaras:v4"})
    asr = SarvamASR(cfg)
    asr._client = _FakeClient()
    clip = AudioClip(np.ones(4 * 16000, dtype=np.float32), 16000, str(tmp_path / "clip.wav"))
    diarization = DiarizationResult(
        turns=[SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(2.0, 3.0, "SPEAKER_01")],
        num_speakers=2,
    )

    transcript = asr.run(clip, diarization)

    assert asr._client.speech_to_text.calls == 2
    assert [(segment.speaker, segment.start, segment.end) for segment in transcript.segments] == [
        ("SPEAKER_00", 0.1, 0.4),
        ("SPEAKER_01", 2.1, 2.4),
    ]