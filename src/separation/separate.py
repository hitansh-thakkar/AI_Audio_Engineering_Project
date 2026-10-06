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
from ..types import AudioClip, SeparatedSource


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
            custom_module = model_dir / "custom.py"
            custom_module.touch(exist_ok=True)
            self._model = SepformerSeparation.from_hparams(
                source=str(self.cfg.model),
                pymodule_file="custom.py",
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

    def run(self, clip: AudioClip, overlap_regions: list[tuple[float, float]],
            notes: list[str] | None = None) -> list[SeparatedSource]:
        notes = [] if notes is None else notes
        mode = str(self.cfg.mode)
        if mode == "none":
            return []

        if mode == "everywhere":
            regions = [(0.0, clip.duration)]
        else:  # overlap
            regions = list(overlap_regions)
            if not regions:
                notes.append("separation.mode=overlap but no overlap regions; nothing to separate")
                return []

        if not self._ensure_model(notes):
            notes.append(
                "real separator unavailable; "
                "returning un-separated regions as passthrough sources"
            )

        out: list[SeparatedSource] = []
        sr = clip.sample_rate
        for (start, end) in regions:
            i0, i1 = int(start * sr), int(end * sr)
            seg = clip.samples[i0:i1]
            if self._available and self._model is not None:  # pragma: no cover (needs weights)
                sources = self._separate_region(seg, sr)
            else:
                sources = [seg]  # passthrough
            for k, src in enumerate(sources):
                out.append(SeparatedSource(region=(start, end), speaker=f"SRC_{k}", samples=src))
        return out

    def _separate_region(self, seg: np.ndarray, sr: int) -> list[np.ndarray]:  # pragma: no cover
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
