"""Run the standalone Dhvani Stage 1 web UI.

Usage: python preprocess_ui.py
Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import re
import uuid
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

from src.ui.preprocess_service import (
    make_upload_dir,
    process_upload,
    run_asr,
    run_diarization,
    run_llm,
    run_separation,
)

REPO = Path(__file__).resolve().parent
HTML_PATH = REPO / "src" / "ui" / "preprocess.html"
OUTPUT_ROOT = Path(os.environ.get("DHVANI_PREPROCESS_OUTPUT", make_upload_dir().parent / "results"))
MAX_UPLOAD_BYTES = 250 * 1024 * 1024


def _multipart_fields(content_type: str, body: bytes) -> dict[str, bytes | str]:
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    fields: dict[str, bytes | str] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        fields[name] = payload if part.get_filename() else payload.decode("utf-8")
    return fields


class Handler(BaseHTTPRequestHandler):
    server_version = "DhvaniPreprocess/1.0"

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, value: dict[str, object]) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(value).encode())

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", HTML_PATH.read_bytes())
            return
        if parsed.path == "/api/download":
            values = parse_qs(parsed.query).get("job", [])
            if not values or not re.fullmatch(r"[0-9a-f]{32}", values[0]):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid download id"})
                return
            path = (OUTPUT_ROOT / values[0] / "processed.wav").resolve()
            if not path.is_file() or OUTPUT_ROOT.resolve() not in path.parents:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Processed file not found"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{path.parent.name}_preprocessed.wav"')
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            self.wfile.write(path.read_bytes())
            return
        if parsed.path == "/api/file":
            values = parse_qs(parsed.query)
            job = values.get("job", [""])[0]
            kind = values.get("kind", [""])[0]
            name = values.get("name", [""])[0]
            if kind not in {"speakers", "separated"} or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid audio file request"})
                return
            path = (OUTPUT_ROOT / job / kind / name).resolve()
            if not path.is_file() or OUTPUT_ROOT.resolve() not in path.parents:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Audio file not found"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            self.wfile.write(path.read_bytes())
            return
        if parsed.path == "/api/waveform":
            values = parse_qs(parsed.query)
            job = values.get("job", [""])[0]
            kind = values.get("kind", [""])[0]
            name = values.get("name", [""])[0]
            valid_name = re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            valid_kind = kind in {"processed", "speakers", "separated"}
            if not valid_kind or not valid_name or not re.fullmatch(r"[0-9a-f]{32}", job):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid waveform request"})
                return
            relative = Path("processed.wav") if kind == "processed" else Path(kind) / name
            path = (OUTPUT_ROOT / job / relative).resolve()
            if not path.is_file() or OUTPUT_ROOT.resolve() not in path.parents:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Audio file not found"})
                return
            samples, _ = sf.read(str(path), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            buckets = 260
            step = max(1, len(samples) // buckets)
            values = [float(np.max(np.abs(samples[i:i + step]))) for i in range(0, len(samples), step)]
            values = values[:buckets]
            peak = max(values, default=0.0)
            normalized = [round(value / peak, 4) for value in values] if peak else values
            self._json(HTTPStatus.OK, {"samples": normalized, "peak": round(peak, 4)})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route in {"/api/diarize", "/api/separate", "/api/asr", "/api/llm"}:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length))
                job_id = str(payload["job_id"])
                if route == "/api/diarize":
                    result = run_diarization(job_id, OUTPUT_ROOT)
                elif route == "/api/separate":
                    result = run_separation(job_id, OUTPUT_ROOT)
                elif route == "/api/asr":
                    result = run_asr(job_id, OUTPUT_ROOT)
                else:
                    result = run_llm(job_id, OUTPUT_ROOT)
                self._json(HTTPStatus.OK, result)
            except Exception as exc:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY,
                           {"error": f"Stage failed: {type(exc).__name__}: {exc}"})
            return
        if route != "/api/process":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                       {"error": "Upload must be between 1 byte and 250 MB"})
            return
        fields = _multipart_fields(self.headers.get("Content-Type", ""), self.rfile.read(length))
        upload = fields.get("audio")
        if not isinstance(upload, bytes) or not upload:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Choose an audio file first"})
            return
        filename = str(fields.get("filename", "audio"))
        suffix = Path(filename).suffix.lower() or ".audio"
        source = make_upload_dir() / f"{uuid.uuid4().hex}{suffix}"
        source.write_bytes(upload)
        try:
            result = process_upload(
                source,
                OUTPUT_ROOT,
            )
            result["download_url"] = f"/api/download?job={result['job_id']}"
            result["waveform_url"] = (
                f"/api/waveform?job={result['job_id']}&kind=processed&name=processed.wav"
            )
            self._json(HTTPStatus.OK, result)
        except Exception as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY,
                       {"error": f"Could not process this file: {type(exc).__name__}: {exc}"})
        finally:
            source.unlink(missing_ok=True)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[preprocess-ui] {format % args}")


if __name__ == "__main__":
    port = int(os.environ.get("DHVANI_UI_PORT", "8765"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dhvani Stage 1 UI running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping preprocessing UI")
    finally:
        server.server_close()
