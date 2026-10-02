"""Config loading (OmegaConf) + secret resolution from .env / environment.

Secrets are NEVER hardcoded — they are read from env vars named in the config
(e.g. postprocessing.api_key_env). The only credentials on the default path are the
free HF_TOKEN (diarization) and GEMINI_API_KEY (LLM stage).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(repo_root: Path = REPO_ROOT) -> None:
    """Load .env into os.environ if present (without overriding existing vars)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root / ".env", override=False)
    except Exception:
        # Minimal fallback parser if python-dotenv is unavailable.
        env = repo_root / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def load_config(path: str | Path) -> DictConfig:
    load_env()
    cfg = OmegaConf.load(Path(path))
    assert isinstance(cfg, DictConfig)
    return cfg


def config_hash(cfg: DictConfig) -> str:
    blob = OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_secret(name: str) -> str | None:
    """Fetch a credential from the environment (after load_env). Never logged."""
    return os.environ.get(name)
