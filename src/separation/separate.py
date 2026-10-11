"""Stage 3 — separation (surgical): recover per-speaker audio in overlapped regions.

mode: none      -> skip (return []).
      overlap   -> run only on diarization overlap regions (default).
      everywhere-> run on the whole clip (ablation; expected to hurt on real audio).

Real backends (SepFormer / Conv-TasNet via SpeechBrain) are 8 kHz, fixed-2-speaker; they
plug in behind `load()` when installed, resampling 16k<->8k. When unavailable, the stage
degrades gracefully: it records the region un-separated and notes it — errors flow, not halt.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from ..config import load_env
from ..types import AudioClip, DiarizationResult, SeparatedSource
from .assign import attribute_streams


def _scale_to_reference(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rescale a separated source to the amplitude it has inside the mixture.

    Neural separators (SepFormer) emit arbitrary, often large scale (peaks >> 1.0); written
    to 16-bit PCM that clips and destroys the audio. Projecting the estimate onto the mixture
    recovers the true per-source level, keeps the track loudness-coherent with its solo
    regions, and is SI-SNR-invariant so separation quality is unchanged.
    """
    e = est.astype(np.float64)
    r = ref.astype(np.float64)
    denom = float(np.dot(e, e))
    if denom < eps:
        return est
    alpha = float(np.dot(e, r) / denom)
    return (alpha * e).astype(np.float32)


class Separator:
    name = "separation"

    def __init__(self, cfg):
        self.cfg = cfg
        self.version = f"{cfg.mode}-{cfg.backend}"
        self._model = None
        self._available = False
        self._load_error: str | None = None

    def load(self) -> None:
        """Defer model loading until an overlap region actually needs separation."""
        return

    def _ensure_model(self, notes: list[str]) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        backend = str(self.cfg.backend)
        if backend != "sepformer":
            self._load_error = f"separator backend '{backend}' is not wired"
            notes.append(self._load_error)
            return False
        try:
            import inspect

            import huggingface_hub
            import torchaudio

            load_env()
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
            if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
                original_hf_hub_download = huggingface_hub.hf_hub_download

                def hf_hub_download_compat(*args, use_auth_token=None, **kwargs):
                    if use_auth_token is not None:
                        kwargs["token"] = use_auth_token
                    return original_hf_hub_download(*args, **kwargs)

                huggingface_hub.hf_hub_download = hf_hub_download_compat
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.utils.fetching import LocalStrategy

            model_dir = Path("models") / "sepformer-wsj02mix"
            model_dir.mkdir(parents=True, exist_ok=True)
            # Let speechbrain fetch the model's own custom.py (don't pass pymodule_file /
            # pre-create an empty one — that breaks from_hparams on speechbrain 1.1.x).
            self._model = SepformerSeparation.from_hparams(
                source=str(self.cfg.model),
                savedir=str(model_dir),
                run_opts={"device": "cpu"},
                local_strategy=LocalStrategy.COPY,
            )
            self._available = True
            return True
        except Exception as exc:
            self._load_error = f"could not load {backend} model '{self.cfg.model}': {exc}"
            notes.append(self._load_error)
            return False

    def _context_window(self, diar: DiarizationResult, region: tuple[float, float],
                        ctx: float, duration: float, notes: list[str]) -> tuple[float, float]:
        """Widen an overlap region to the span of its active speakers' turns (+ ctx margin),
        clamped to the clip. Context lets the separator anchor each voice on its solo audio.
        wsj02mix is 2-speaker; a window with >2 active speakers is noted as a degradation."""
        r0, r1 = region
        active = [t for t in diar.turns if t.start < r1 and t.end > r0]
        w0 = min([t.start for t in active], default=r0)
        w1 = max([t.end for t in active], default=r1)
        w0 = max(0.0, min(w0, r0) - ctx)
        w1 = min(duration, max(w1, r1) + ctx)
        n_spk = len({t.speaker for t in diar.turns if t.start < w1 and t.end > w0})
        if n_spk > int(self.cfg.max_speakers_supported):
            notes.append(f"context window {(round(w0, 3), round(w1, 3))} spans {n_spk} speakers "
                         f"but backend supports {self.cfg.max_speakers_supported}; separation may degrade")
        return w0, w1

    def run(self, clip: AudioClip, diarization: DiarizationResult,
            notes: list[str] | None = None) -> list[SeparatedSource]:
        notes = [] if notes is None else notes
        mode = str(self.cfg.mode)
        if mode == "none":
            return []

        if mode == "everywhere":
            regions = [(0.0, clip.duration)]
        else:  # overlap
            regions = list(diarization.overlap_regions)
            if not regions:
                notes.append("separation.mode=overlap but no overlap regions; nothing to separate")
                return []

        model_ok = self._ensure_model(notes)
        if not model_ok:
            notes.append(
                "real separator unavailable; "
                "returning un-separated regions as passthrough sources"
            )

        out: list[SeparatedSource] = []
        sr = clip.sample_rate
        ctx = float(getattr(self.cfg, "context_seconds", 3.0))
        for (start, end) in regions:
            if mode == "everywhere":
                w0, w1 = 0.0, clip.duration
            else:
                w0, w1 = self._context_window(diarization, (start, end), ctx, clip.duration, notes)
            wi0, wi1 = int(w0 * sr), int(w1 * sr)
            window = clip.samples[wi0:wi1]
            if model_ok and self._model is not None:  # pragma: no cover (needs weights)
                streams = self._separate_region(window, sr)
            else:
                streams = [window]  # passthrough

            mapping = attribute_streams(streams, (w0, w1), diarization, window, sr, notes)
            oi0, oi1 = int(round((start - w0) * sr)), int(round((end - w0) * sr))
            mix_overlap = window[oi0:oi1]
            for k, stream in enumerate(streams):
                speaker = mapping.get(k, f"SRC_{k}")
                samples = _scale_to_reference(stream[oi0:oi1], mix_overlap)
                out.append(SeparatedSource(region=(start, end), speaker=speaker, samples=samples))
        return out

    def _separate_region(self, seg: np.ndarray, sr: int) -> list[np.ndarray]:  # pragma: no cover
        assert self._model is not None  # run() only calls this once the model is loaded
        model_sr = int(self.cfg.model_sample_rate)
        model_input = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=model_sr)
        import torch

        estimated = self._model.separate_batch(torch.from_numpy(model_input).unsqueeze(0))

        estimated = estimated.detach().cpu().numpy()
        if estimated.ndim == 3:
            estimated = estimated[0]
        if estimated.ndim != 2:
            raise ValueError(f"unexpected separator output shape: {estimated.shape}")
        return [
            librosa.resample(source.astype(np.float32), orig_sr=model_sr, target_sr=sr)
            for source in estimated.T
        ]
