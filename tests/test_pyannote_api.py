"""Unit tests for the pyannote.ai cloud API client."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.diarization.pyannote_api import (
    PyannoteAPIBackend,
    _find_overlapping_regions,
    _parse_output,
    _poll_job,
    _submit_job,
    _upload_audio,
    verify_api_key,
)
from src.types import SpeakerTurn


def test_verify_api_key_success():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text='{"status":"OK"}')
        assert verify_api_key("valid_key") is True


def test_verify_api_key_failure():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=401, text="Unauthorized")
        with pytest.raises(RuntimeError, match="pyannote.ai API key verification failed"):
            verify_api_key("bad_key")


def test_upload_audio_201_created():
    """Verify that HTTP 201 from /media/input is accepted as success (per pyannote docs)."""
    with patch("requests.post") as mock_post, patch("requests.put") as mock_put:
        mock_post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"url": "https://s3.amazonaws.com/presigned-put"},
        )
        mock_put.return_value = MagicMock(status_code=200)

        samples = np.zeros(16000, dtype=np.float32)
        media_uri = _upload_audio(samples, 16000, "fake_key")

        assert media_uri.startswith("media://dhvani-")
        mock_post.assert_called_once()
        mock_put.assert_called_once()


def test_upload_audio_raw_file_bytes(tmp_path):
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(b"RIFFdummydata")

    with patch("requests.post") as mock_post, patch("requests.put") as mock_put:
        mock_post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"url": "https://s3.amazonaws.com/presigned-put"},
        )
        mock_put.return_value = MagicMock(status_code=204)

        media_uri = _upload_audio(wav_file, 16000, "fake_key")
        assert media_uri.startswith("media://dhvani-")
        mock_put.assert_called_once()
        assert mock_put.call_args[1]["data"] == b"RIFFdummydata"


def test_submit_job():
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"jobId": "test-job-123", "status": "created"},
        )
        job_id = _submit_job("media://dhvani-abc", "fake_key", model="precision-2")
        assert job_id == "test-job-123"


def test_poll_job_success():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "jobId": "test-job-123",
                "status": "succeeded",
                "output": {"diarization": []},
            },
        )
        res = _poll_job("test-job-123", "fake_key")
        assert res["status"] == "succeeded"


def test_poll_job_failed():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "jobId": "test-job-123",
                "status": "failed",
                "output": {"error": "Audio corrupted"},
            },
        )
        with pytest.raises(RuntimeError, match="Audio corrupted"):
            _poll_job("test-job-123", "fake_key")


def test_find_overlapping_regions():
    turns = [
        SpeakerTurn(start=1.0, end=5.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=7.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=8.0, end=10.0, speaker="SPEAKER_00"),
    ]
    overlaps = _find_overlapping_regions(turns)
    assert len(overlaps) == 1
    assert overlaps[0] == (3.0, 5.0)


def test_parse_output():
    api_output = {
        "diarization": [
            {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_00"},
        ]
    }
    result = _parse_output(api_output)
    assert result.num_speakers == 2
    assert len(result.turns) == 3
    assert len(result.overlap_regions) == 1
    assert result.overlap_regions[0] == (2.0, 2.5)


def test_backend_diarize_full_flow():
    backend = PyannoteAPIBackend(api_key="mock_key", model="precision-2")

    with patch("src.diarization.pyannote_api._upload_audio") as mock_upload, \
         patch("src.diarization.pyannote_api._submit_job") as mock_submit, \
         patch("src.diarization.pyannote_api._poll_job") as mock_poll:

        mock_upload.return_value = "media://dhvani-test"
        mock_submit.return_value = "job-xyz"
        mock_poll.return_value = {
            "status": "succeeded",
            "output": {
                "diarization": [
                    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                    {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
                ]
            },
        }

        notes = []
        samples = np.zeros(32000, dtype=np.float32)
        res = backend.diarize(samples, 16000, notes=notes)

        assert res.num_speakers == 2
        assert len(res.turns) == 2
        assert len(res.overlap_regions) == 1
        assert any("pyannote.ai" in n for n in notes)
