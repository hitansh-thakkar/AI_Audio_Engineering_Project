"""pyannote.ai cloud diarization backend — precision-2 model.

Implements the official pyannote.ai REST API workflow:
1. POST /v1/media/input
   Request a pre-signed S3 PUT URL for a temporary ``media://<object-key>`` identifier.
   Returns HTTP 201 Created with {"url": "<presigned-s3-put-url>"}.
2. PUT <presigned-url>
   Upload raw audio bytes (16-bit PCM WAV / audio file) to the pre-signed S3 location.
   Returns HTTP 200 OK.
3. POST /v1/diarize
   Submit a diarization job pointing to the ``media://<object-key>`` URI.
   Returns HTTP 200 OK with {"jobId": "<uuid>", "status": "created"}.
4. GET /v1/jobs/{jobId}
   Poll job status with back-off until terminal state ('succeeded', 'failed', 'canceled').
   Returns HTTP 200 OK with {"status": "succeeded", "output": {"diarization": [...]}}.
5. Parse output["diarization"] into:
   - SpeakerTurn list (chronological)
   - overlap_regions: intervals where >=2 distinct speakers speak simultaneously.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np

from ..types import DiarizationResult, SpeakerTurn

log = logging.getLogger(__name__)

_BASE = "https://api.pyannote.ai/v1"
_DEFAULT_MODEL = "precision-2"

# Polling configuration: start at 2 s, increase by 1.5x, cap at 15 s, timeout at 10 min.
_POLL_INITIAL = 2.0
_POLL_FACTOR = 1.5
_POLL_MAX = 15.0
_POLL_TIMEOUT = 600.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono samples as 16-bit PCM WAV in memory."""
    import wave

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def verify_api_key(api_key: str) -> bool:
    """Verify pyannote.ai API key with GET /v1/test.

    Returns True if valid, raises RuntimeError otherwise.
    """
    import requests

    resp = requests.get(
        f"{_BASE}/test",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if resp.status_code == 200:
        return True
    raise RuntimeError(
        f"pyannote.ai API key verification failed ({resp.status_code}): {resp.text[:300]}"
    )


# ---------------------------------------------------------------------------
# Step 1 + 2: Upload to pyannote.ai media storage
# ---------------------------------------------------------------------------

def _upload_audio(
    audio: np.ndarray | bytes | str | Path,
    sample_rate: int,
    api_key: str,
) -> str:
    """Upload audio bytes to pyannote.ai temporary storage.

    Returns the ``media://<object-key>`` URI for subsequent API calls.
    """
    import requests

    object_key = f"dhvani-{uuid.uuid4().hex[:12]}"
    media_uri = f"media://{object_key}"

    # 1. Request pre-signed PUT URL (HTTP 201 Created as per official docs)
    resp = requests.post(
        f"{_BASE}/media/input",
        json={"url": media_uri},
        headers=_headers(api_key),
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"media/input failed {resp.status_code}: {resp.text[:400]}"
        )
    presigned_url: str = resp.json().get("url", "")
    if not presigned_url:
        raise RuntimeError("media/input returned no URL in payload")

    # 2. Extract or convert audio to bytes
    if isinstance(audio, (str, Path)):
        p = Path(audio)
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {p}")
        raw_bytes = p.read_bytes()
    elif isinstance(audio, bytes):
        raw_bytes = audio
    elif isinstance(audio, np.ndarray):
        raw_bytes = _wav_bytes(audio, sample_rate)
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    # 3. Upload to pre-signed S3 URL via PUT
    put_resp = requests.put(
        presigned_url,
        data=raw_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=120,
    )
    if put_resp.status_code not in (200, 201, 204):
        raise RuntimeError(
            f"media PUT upload failed {put_resp.status_code}: {put_resp.text[:400]}"
        )

    log.debug("Uploaded %d bytes as %s", len(raw_bytes), media_uri)
    return media_uri


# ---------------------------------------------------------------------------
# Step 3: Submit Diarization Job
# ---------------------------------------------------------------------------

def _submit_job(
    media_uri: str,
    api_key: str,
    model: str = _DEFAULT_MODEL,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    turn_level_confidence: bool = False,
    exclusive: bool = False,
) -> str:
    """POST /v1/diarize and return the jobId."""
    import requests

    body: dict[str, Any] = {
        "url": media_uri,
        "model": model,
    }
    if num_speakers is not None:
        body["numSpeakers"] = int(num_speakers)
    elif min_speakers is not None or max_speakers is not None:
        if min_speakers is not None:
            body["minSpeakers"] = int(min_speakers)
        if max_speakers is not None:
            body["maxSpeakers"] = int(max_speakers)
    if turn_level_confidence:
        body["turnLevelConfidence"] = True
    if exclusive:
        body["exclusive"] = True

    resp = requests.post(
        f"{_BASE}/diarize",
        json=body,
        headers=_headers(api_key),
        timeout=30,
    )
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"POST /diarize failed {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    job_id: str = data["jobId"]
    log.debug("Diarization job submitted: %s (status=%s)", job_id, data.get("status"))
    return job_id


# ---------------------------------------------------------------------------
# Step 4: Poll Job Until Completion
# ---------------------------------------------------------------------------

def _poll_job(
    job_id: str,
    api_key: str,
    timeout: float = _POLL_TIMEOUT,
) -> dict[str, Any]:
    """Poll GET /v1/jobs/{jobId} with back-off.

    Returns the full response dict when status is 'succeeded'.
    Raises RuntimeError on 'failed', 'canceled', or timeout.
    """
    import requests

    wait = _POLL_INITIAL
    deadline = time.monotonic() + timeout
    hdr = {"Authorization": f"Bearer {api_key}"}

    while True:
        resp = requests.get(
            f"{_BASE}/jobs/{job_id}",
            headers=hdr,
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"GET /jobs/{job_id} failed {resp.status_code}: {resp.text[:400]}"
            )

        data: dict[str, Any] = resp.json()
        status: str = data.get("status", "unknown")

        if status == "succeeded":
            log.debug("Job %s succeeded", job_id)
            return data

        if status in ("failed", "canceled"):
            out = data.get("output") or {}
            err_msg = out.get("error") or data.get("error") or data.get("message") or "no details"
            raise RuntimeError(
                f"Diarization job {job_id} ended with status '{status}': {err_msg}"
            )

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Diarization job {job_id} timed out after {timeout:.0f}s (last status: {status})"
            )

        log.debug("Job %s status=%s; retrying in %.1fs", job_id, status, wait)
        time.sleep(wait)
        wait = min(wait * _POLL_FACTOR, _POLL_MAX)


# ---------------------------------------------------------------------------
# Step 5: Parse Output -> SpeakerTurn + Overlap Regions
# ---------------------------------------------------------------------------

def _find_overlapping_regions(turns: list[SpeakerTurn]) -> list[tuple[float, float]]:
    """Identify all intervals where >=2 different speakers overlap in time."""
    raw_overlaps: list[tuple[float, float]] = []
    n = len(turns)
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = turns[i], turns[j]
            if t1.speaker != t2.speaker:
                o_start = max(t1.start, t2.start)
                o_end = min(t1.end, t2.end)
                if o_end - o_start > 0.05:  # >= 50ms overlap
                    raw_overlaps.append((o_start, o_end))

    if not raw_overlaps:
        return []

    # Merge overlapping or touching overlap intervals
    raw_overlaps.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    curr_s, curr_e = raw_overlaps[0]
    for s, e in raw_overlaps[1:]:
        if s <= curr_e:
            curr_e = max(curr_e, e)
        else:
            merged.append((round(curr_s, 3), round(curr_e, 3)))
            curr_s, curr_e = s, e
    merged.append((round(curr_s, 3), round(curr_e, 3)))
    return merged


def _parse_output(output: dict[str, Any]) -> DiarizationResult:
    """Convert the API output dict into a DiarizationResult."""
    raw_segs: list[dict[str, Any]] = output.get("diarization", [])
    if not raw_segs:
        return DiarizationResult(
            turns=[],
            overlap_regions=[],
            num_speakers=0,
        )

    turns: list[SpeakerTurn] = []
    for seg in raw_segs:
        spk: str = str(seg.get("speaker", "SPEAKER_00"))
        start: float = float(seg.get("start", 0.0))
        end: float = float(seg.get("end", 0.0))
        if end > start:
            turns.append(SpeakerTurn(
                start=round(start, 3),
                end=round(end, 3),
                speaker=spk,
            ))

    turns.sort(key=lambda t: t.start)
    speakers = sorted({t.speaker for t in turns})

    # Detect multi-speaker overlaps
    overlap_regions = _find_overlapping_regions(turns)

    return DiarizationResult(
        turns=turns,
        overlap_regions=overlap_regions,
        num_speakers=len(speakers),
    )


# ---------------------------------------------------------------------------
# Public Entrypoint
# ---------------------------------------------------------------------------

class PyannoteAPIBackend:
    """pyannote.ai cloud speaker diarization client."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        turn_level_confidence: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty")
        self._api_key = api_key
        self._model = model
        self._num_speakers = num_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._turn_level_confidence = turn_level_confidence

    def test(self) -> bool:
        """Verify API credentials."""
        return verify_api_key(self._api_key)

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        notes: list[str] | None = None,
        audio_path: str | Path | None = None,
    ) -> DiarizationResult:
        """Upload audio -> submit diarization job -> poll results -> parse output."""
        if notes is None:
            notes = []

        dur = len(samples) / sample_rate if sample_rate else 0.0
        notes.append(f"pyannote.ai backend: uploading audio ({dur:.1f}s)")

        # Upload audio (using existing file on disk if available, else in-memory WAV)
        source = audio_path if (audio_path and Path(audio_path).is_file()) else samples
        media_uri = _upload_audio(source, sample_rate, self._api_key)
        notes.append(f"pyannote.ai media uploaded: {media_uri}")

        # Submit diarization job
        job_id = _submit_job(
            media_uri,
            self._api_key,
            model=self._model,
            num_speakers=self._num_speakers,
            min_speakers=self._min_speakers,
            max_speakers=self._max_speakers,
            turn_level_confidence=self._turn_level_confidence,
        )
        notes.append(f"pyannote.ai job submitted: {job_id}; polling …")

        # Poll until complete
        result_data = _poll_job(job_id, self._api_key)
        output: dict[str, Any] = result_data.get("output", {})

        if output.get("warning"):
            notes.append(f"pyannote.ai warning: {output['warning']}")

        result = _parse_output(output)
        notes.append(
            f"pyannote.ai {self._model}: {result.num_speakers} speakers, "
            f"{len(result.turns)} turns, "
            f"{len(result.overlap_regions)} overlap regions"
        )
        return result
