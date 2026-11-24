#!/usr/bin/env python3
"""Build `dhvani_eval/asr_whisper/`: a faster-whisper ASR test set over the Indic languages
Whisper supports, drawn from every corpus the project uses.

Languages (Whisper/ISO-639-1 codes): hi bn ta te mr gu kn ml pa ur as sa ne sd
(Odia and the other scheduled languages are absent from Whisper's vocabulary.)

Every clip has a human reference transcript; untranscribed audio is excluded (WER/CER only).

Sources
  indicvoices  ai4bharat/IndicVoices, `valid` split, streamed            (all 14 languages)
  vaani        ARTPARK-IISc/Vaani, transcribed rows only, streamed        (13; no Sanskrit config)
  nirantar     official E2E AUDIOS.tgz.aa streamed through gunzip+untar, labelled by joining
               on the LIL episodic manifests (LIL.tgz, ~179 MB, cached under data/)
  registers    existing dhvani_eval/registers (English + Hindi-English code-switch), referenced
               in place as extra registers; filter on `register` to exclude them
  air_rs_db    has no transcripts (read/spontaneous labels only) -> recorded as a gap

Selection is deterministic (first N matching rows in stream order, capped per speaker); no
randomness. Dataset revisions, params and the git SHA go into asr_whisper/summary.json.

Output layout (paths in the manifest are relative to dhvani_eval/, like the other subsets):
  asr_whisper/audio/<corpus>/<id>.wav     16 kHz mono PCM-16
  asr_whisper/parts/<corpus>.jsonl        per-corpus partial manifests (re-run one corpus alone)
  asr_whisper/manifest.jsonl              merged manifest
  asr_whisper/summary.json                counts per corpus x language, gaps, provenance

Run (needs HF_TOKEN in .env + IndicVoices/Vaani licences accepted):
  python dhvani_eval/build_asr_whisper.py
  python dhvani_eval/build_asr_whisper.py --corpora vaani --n 20   # rebuild one part, re-merge
"""

from __future__ import annotations

import argparse
import functools
import io
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "asr_whisper"
SR = 16_000

# Whisper code -> (IndicVoices config, Vaani config | None)
LANGS: dict[str, tuple[str, str | None]] = {
    "hi": ("hindi", "Hindi"),
    "bn": ("bengali", "Bengali"),
    "ta": ("tamil", "Tamil"),
    "te": ("telugu", "Telugu"),
    "mr": ("marathi", "Marathi"),
    "gu": ("gujarati", "Gujarati"),
    "kn": ("kannada", "Kannada"),
    "ml": ("malayalam", "Malayalam"),
    "pa": ("punjabi", "Punjabi"),
    "ur": ("urdu", "Urdu"),
    "as": ("assamese", "Assamese"),
    "sa": ("sanskrit", None),  # Vaani has no Sanskrit config
    "ne": ("nepali", "Nepali"),
    "sd": ("sindhi", "Sindhi"),
}
CORPORA = ("indicvoices", "vaani", "nirantar", "registers")
# Script each reference must be written in (first word of the Unicode char name). Rows in any
# other script are rejected: e.g. some Vaani Urdu is Devanagari, which Whisper never emits for ur.
# NOTE sd: IndicVoices/Nirantar write Sindhi in Devanagari, but Whisper emits Perso-Arabic, so
# raw sd WER is a script mismatch, not an acoustic error (flagged in summary.json).
SCRIPT: dict[str, str] = {
    "hi": "DEVANAGARI", "mr": "DEVANAGARI", "ne": "DEVANAGARI", "sa": "DEVANAGARI", "sd": "DEVANAGARI",
    "bn": "BENGALI", "as": "BENGALI", "ta": "TAMIL", "te": "TELUGU", "gu": "GUJARATI",
    "kn": "KANNADA", "ml": "MALAYALAM", "pa": "GURMUKHI", "ur": "ARABIC",
}  # fmt: skip
NIRANTAR_AUDIO_URL = "https://objectstore.e2enetworks.net/nirantar/AUDIOS.tgz.aa"
NIRANTAR_LIL_URL = "https://objectstore.e2enetworks.net/nirantar/LIL.tgz"
NIRANTAR_LIL_CACHE = REPO / "data" / "nirantar" / "LIL.tgz"  # data/ is gitignored

log = logging.getLogger("asr_whisper")


@dataclass(frozen=True)
class Params:
    n: int
    min_dur: float
    max_dur: float
    per_speaker: int
    iv_scan: int
    vaani_scan: int
    nirantar_mb: int


@dataclass
class Record:
    id: str
    corpus: str
    language: str  # Whisper code; pass straight to WhisperModel.transcribe(language=...)
    register: str  # "monolingual" | "english" | "codeswitch"
    audio: str  # relative to dhvani_eval/
    duration: float
    text: str  # reference for WER/CER (still run the project's Indic normalizer on it)
    text_raw: str  # as shipped by the corpus (verbatim / with markup)
    speaker_id: str | None = None
    gender: str | None = None
    scenario: str | None = None
    district: str | None = None
    state: str | None = None
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def load_env() -> None:
    """Populate os.environ from .env without overriding existing values (same as fetch_vaani)."""
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def decode_bytes(data: bytes) -> npt.NDArray[np.float32]:
    """Decode wav/flac bytes to 16 kHz mono float32 (soundfile avoids the torchcodec dep)."""
    import librosa
    import soundfile as sf

    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.asarray(y, dtype=np.float32)


def write_wav(rel: str, y: npt.NDArray[np.float32]) -> None:
    import soundfile as sf

    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR, subtype="PCM_16")


_MARKUP = re.compile(r"\{[^}]*\}|\[[^\]]*\]|<[^>]*>")


def clean_vaani(text: str) -> str:
    """Drop Vaani inline markup: `{english gloss}`, `[tag]`, `<noise>`; collapse whitespace."""
    return re.sub(r"\s+", " ", _MARKUP.sub(" ", text)).strip()


def dedup_key(lang: str, speaker: str | None, text: str) -> str:
    # IndicVoices and Nirantar come from the same AI4Bharat collection, so the same utterance
    # can appear in both under different file names; speaker + normalized text catches that.
    return f"{lang}|{speaker or ''}|{unicodedata.normalize('NFC', text).strip()}"


class Quota:
    """Per-language counters with a per-(language, speaker) cap for speaker diversity."""

    def __init__(self, langs: Iterable[str], n: int, per_speaker: int) -> None:
        self.n, self.per_speaker = n, per_speaker
        self.count: Counter[str] = Counter({lang: 0 for lang in langs})
        self.spk: Counter[tuple[str, str]] = Counter()

    def full(self, lang: str) -> bool:
        return self.count[lang] >= self.n

    def all_full(self) -> bool:
        return all(self.full(lang) for lang in self.count)

    def speaker_ok(self, lang: str, speaker: str | None) -> bool:
        return speaker is None or self.spk[(lang, speaker)] < self.per_speaker

    def take(self, lang: str, speaker: str | None) -> None:
        self.count[lang] += 1
        if speaker is not None:
            self.spk[(lang, speaker)] += 1


def dur_ok(d: float, p: Params) -> bool:
    return p.min_dur <= d <= p.max_dur


def dominant_script(text: str) -> str:
    """Most frequent script among letters, from the Unicode name ("DEVANAGARI LETTER KA")."""
    names = Counter(unicodedata.name(ch, "UNKNOWN").split()[0] for ch in text if ch.isalpha())
    return names.most_common(1)[0][0] if names else "NONE"


def text_ok(lang: str, text: str) -> bool:
    return bool(text) and dominant_script(text) == SCRIPT[lang]


# --------------------------------------------------------------------------- IndicVoices


RETRIES = 3


@dataclass
class LangResult:
    recs: list[Record]
    keys: set[str]  # dedup keys of `recs`; merged into the global `seen` only on success
    scanned: int
    # (relative path, samples) held in memory until the attempt succeeds, so a failed attempt
    # can never overwrite a wav that a carried-over row from the previous build points to.
    audio: list[tuple[str, npt.NDArray[np.float32]]] = field(default_factory=list)


def with_retries(corpus: str, lang: str, attempt_fn: Callable[[], LangResult]) -> LangResult | None:
    """Run one language's scan. On a network error, reset huggingface_hub's shared HTTP client
    and restart the language. Without the reset, one dropped connection leaves the global client
    closed, and every later language fails instantly with "client has been closed"."""
    from huggingface_hub import close_session

    for attempt in range(1, RETRIES + 1):
        try:
            return attempt_fn()
        except Exception as e:  # noqa: BLE001 - Hub/auth/stream failure: retry, then record gap
            log.warning(
                "%s %s: attempt %d/%d failed: %s: %s",
                corpus,
                lang,
                attempt,
                RETRIES,
                type(e).__name__,
                str(e)[:160],
            )
            close_session()
            time.sleep(30 * 2 ** (attempt - 1))  # 30/60/120 s: outlast short connectivity drops
    return None


def _scan_indicvoices(p: Params, seen: set[str], lang: str, cfg: str) -> LangResult:
    from datasets import Audio, load_dataset

    ds = load_dataset("ai4bharat/IndicVoices", cfg, split="valid", streaming=True)
    ds = ds.cast_column("audio_filepath", Audio(decode=False))
    out = LangResult([], set(), 0)
    quota = Quota([lang], p.n, p.per_speaker)
    # Rows rejected only by the speaker cap, used to top up if the stream is speaker-sorted.
    overflow: list[tuple[dict[str, Any], str, npt.NDArray[np.float32]]] = []

    def emit(ex: dict[str, Any], text: str, y: npt.NDArray[np.float32]) -> None:
        spk = ex.get("speaker_id")
        cid = f"iv_{lang}_{quota.count[lang]:03d}"
        rel = f"asr_whisper/audio/indicvoices/{cid}.wav"
        out.audio.append((rel, y))  # written by fetch_hf only if the attempt succeeds
        out.recs.append(
            Record(
                id=cid,
                corpus="indicvoices",
                language=lang,
                register="monolingual",
                audio=rel,
                duration=round(len(y) / SR, 3),
                text=text,
                text_raw=(ex.get("verbatim") or text).strip(),
                speaker_id=spk,
                gender=ex.get("gender"),
                scenario=ex.get("scenario"),
                district=ex.get("district"),
                state=ex.get("state"),
                source="ai4bharat/IndicVoices (valid)",
                extra={
                    "source_file": Path(ex["audio_filepath"].get("path") or cid).stem,
                    "task_name": ex.get("task_name"),
                },
            )
        )
        out.keys.add(dedup_key(lang, spk, text))
        quota.take(lang, spk)

    for ex in itertools.islice(ds, p.iv_scan):
        out.scanned += 1
        text = (ex.get("text") or "").strip()
        spk = ex.get("speaker_id")
        key = dedup_key(lang, spk, text)
        if not text_ok(lang, text) or not dur_ok(float(ex.get("duration") or 0.0), p):
            continue
        if key in seen or key in out.keys:
            continue
        try:
            y = decode_bytes(ex["audio_filepath"]["bytes"])
        except Exception:  # noqa: BLE001 - undecodable row: skip it, don't abort
            continue
        if quota.speaker_ok(lang, spk):
            emit(ex, text, y)
        elif len(overflow) < p.n:
            overflow.append((ex, text, y))
        if quota.full(lang):
            break
    for ex, text, y in overflow:  # top-up, relaxing the speaker cap
        if quota.full(lang):
            break
        if dedup_key(lang, ex.get("speaker_id"), text) not in out.keys:
            emit(ex, text, y)
    return out


def _scan_vaani(p: Params, seen: set[str], lang: str, cfg: str) -> LangResult:
    from datasets import Audio, load_dataset

    ds = load_dataset("ARTPARK-IISc/Vaani", cfg, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = LangResult([], set(), 0)
    quota = Quota([lang], p.n, p.per_speaker)
    for ex in itertools.islice(ds, p.vaani_scan):
        out.scanned += 1
        raw = (ex.get("transcript") or "").strip()
        text = clean_vaani(raw)
        spk = ex.get("speakerID")
        key = dedup_key(lang, spk, text)
        if not text_ok(lang, text) or not dur_ok(float(ex.get("duration") or 0.0), p):
            continue
        if not quota.speaker_ok(lang, spk) or key in seen or key in out.keys:
            continue
        try:
            y = decode_bytes(ex["audio"]["bytes"])
        except Exception:  # noqa: BLE001 - undecodable row: skip it
            continue
        cid = f"vaani_{lang}_{quota.count[lang]:03d}"
        rel = f"asr_whisper/audio/vaani/{cid}.wav"
        out.audio.append((rel, y))  # written by fetch_hf only if the attempt succeeds
        out.recs.append(
            Record(
                id=cid,
                corpus="vaani",
                language=lang,
                register="monolingual",
                audio=rel,
                duration=round(len(y) / SR, 3),
                text=text,
                text_raw=raw,
                speaker_id=spk,
                gender=ex.get("gender"),
                district=ex.get("district"),
                state=ex.get("state"),
                source="ARTPARK-IISc/Vaani",
                extra={"has_markup": text != raw, "pincode": ex.get("pincode")},
            )
        )
        out.keys.add(key)
        quota.take(lang, spk)
        if quota.full(lang):
            break
    return out


def fetch_hf(
    corpus: str, p: Params, seen: set[str], previous: list[Record]
) -> tuple[list[Record], dict[str, str]]:
    """Per-language streamed scan of IndicVoices or Vaani, with retries. A language whose
    retries are exhausted keeps its rows from `previous` (the last build) instead of being wiped."""
    scan = _scan_indicvoices if corpus == "indicvoices" else _scan_vaani
    recs: list[Record] = []
    gaps: dict[str, str] = {}
    for lang, (iv_cfg, vaani_cfg) in LANGS.items():
        cfg = iv_cfg if corpus == "indicvoices" else vaani_cfg
        if cfg is None:
            gaps[lang] = f"{corpus} has no config for {lang}"
            continue
        t0 = time.perf_counter()
        res = with_retries(corpus, lang, functools.partial(scan, p, seen, lang, cfg))
        if res is None:
            kept = [r for r in previous if r.language == lang]
            recs += kept
            seen |= {dedup_key(lang, r.speaker_id, r.text) for r in kept}
            gaps[lang] = f"fetch failed after {RETRIES} attempts; kept {len(kept)} rows from previous build"
            continue
        for rel, y in res.audio:
            write_wav(rel, y)
        recs += res.recs
        seen |= res.keys
        if len(res.recs) < p.n:
            gaps[lang] = f"only {len(res.recs)}/{p.n} usable in first {res.scanned} rows"
        log.info(
            "%s %s: kept %d (scanned %d, %.1fs)",
            corpus,
            lang,
            len(res.recs),
            res.scanned,
            time.perf_counter() - t0,
        )
    return recs, gaps


# --------------------------------------------------------------------------- Nirantar


class _CountingReader(io.RawIOBase):
    """Wraps the HTTP response, counting compressed bytes so the scan can be capped."""

    def __init__(self, raw: BinaryIO, cap: int) -> None:
        self.raw, self.cap, self.read_bytes = raw, cap, 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:  # Any: buffer protocol object (memoryview/bytearray)
        if self.read_bytes >= self.cap:
            return 0  # behave as EOF; tarfile/gzip then stop on a truncated stream
        chunk = self.raw.read(min(len(b), self.cap - self.read_bytes))
        n = len(chunk)
        b[:n] = chunk
        self.read_bytes += n
        return n


def load_nirantar_index(p: Params) -> dict[str, dict[str, Any]]:
    """stem -> manifest row, for target languages and durations only."""
    if not NIRANTAR_LIL_CACHE.exists():
        NIRANTAR_LIL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        log.info("nirantar: downloading LIL manifests (~179 MB) -> %s", NIRANTAR_LIL_CACHE)
        urllib.request.urlretrieve(NIRANTAR_LIL_URL, NIRANTAR_LIL_CACHE)
    idx: dict[str, dict[str, Any]] = {}
    with tarfile.open(NIRANTAR_LIL_CACHE) as t:
        for m in t:
            f = t.extractfile(m) if m.isfile() and m.name.endswith(".json") else None
            if f is None:
                continue
            for line in f.read().decode("utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if (
                    r.get("lang") in LANGS
                    and dur_ok(float(r.get("duration") or 0.0), p)
                    and text_ok(r["lang"], r.get("text") or "")
                ):
                    r["_episode_file"] = m.name
                    idx[Path(r["audio_filepath"]).stem] = r
    log.info(
        "nirantar: %d candidate rows indexed (%s)", len(idx), dict(Counter(r["lang"] for r in idx.values()))
    )
    return idx


def fetch_nirantar(p: Params, seen: set[str]) -> tuple[list[Record], dict[str, str]]:
    t0 = time.perf_counter()
    idx = load_nirantar_index(p)
    quota = Quota(LANGS, p.n, p.per_speaker)
    recs: list[Record] = []
    members = 0
    reader = _CountingReader(urllib.request.urlopen(NIRANTAR_AUDIO_URL, timeout=120), p.nirantar_mb * 2**20)
    try:
        # Streaming mode ("r|gz") never seeks, so it reads the 40 GB part sequentially and we
        # stop as soon as every quota is met or the byte cap hits; nothing touches disk.
        with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 20), mode="r|gz") as tar:
            for m in tar:
                members += 1
                if members % 5000 == 0:
                    log.info(
                        "nirantar: %d members, %d MB read, kept %s",
                        members,
                        reader.read_bytes >> 20,
                        dict(quota.count),
                    )
                row = idx.get(Path(m.name).stem) if m.isfile() else None
                if row is None:
                    continue
                lang, spk, text = row["lang"], row.get("speaker_id"), row["text"].strip()
                if quota.full(lang) or not quota.speaker_ok(lang, spk) or dedup_key(lang, spk, text) in seen:
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                try:
                    y = decode_bytes(f.read())
                except Exception:  # noqa: BLE001 - corrupt member: skip
                    continue
                cid = f"nir_{lang}_{quota.count[lang]:03d}"
                rel = f"asr_whisper/audio/nirantar/{cid}.wav"
                write_wav(rel, y)
                recs.append(
                    Record(
                        id=cid,
                        corpus="nirantar",
                        language=lang,
                        register="monolingual",
                        audio=rel,
                        duration=round(len(y) / SR, 3),
                        text=text,
                        text_raw=(row.get("verbatim") or text).strip(),
                        speaker_id=spk,
                        gender=row.get("gender"),
                        scenario=row.get("scenario"),
                        district=row.get("district"),
                        state=row.get("state"),
                        source="objectstore.e2enetworks.net/nirantar (github.com/AI4Bharat/Nirantar)",
                        extra={"source_file": Path(m.name).stem, "episode_file": row["_episode_file"]},
                    )
                )
                seen.add(dedup_key(lang, spk, text))
                quota.take(lang, spk)
                if quota.all_full():
                    break
    except (tarfile.TarError, EOFError, OSError) as e:
        log.info("nirantar: stream ended (%s: %s) — expected when the byte cap is hit", type(e).__name__, e)
    gaps = {
        lang: f"only {c}/{p.n} within first {reader.read_bytes >> 20} MB of AUDIOS.tgz.aa"
        for lang, c in quota.count.items()
        if c < p.n
    }
    log.info(
        "nirantar: kept %d from %d members, %d MB read (%.1fs)",
        len(recs),
        members,
        reader.read_bytes >> 20,
        time.perf_counter() - t0,
    )
    return recs, gaps


# --------------------------------------------------------------------------- registers


def collect_registers() -> tuple[list[Record], dict[str, str]]:
    path = ROOT / "registers" / "manifest.jsonl"
    if not path.exists():
        return [], {"registers": "dhvani_eval/registers missing — run fetch_codeswitch.py"}
    recs: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        reg = r["register"]
        # Copy into asr_whisper/ so the dataset folder is self-contained (no paths outside it).
        rel = f"asr_whisper/audio/registers/{Path(r['audio']).name}"
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r["audio"], ROOT / rel)
        # Code-switch clips span two languages; Whisper needs one hint, so tag "hi" (the base
        # language) and leave auto-detect as the alternative you can test (language=None).
        recs.append(
            Record(
                id=r["id"],
                corpus="registers",
                language="en" if reg == "english" else "hi",
                register=reg,
                audio=rel,
                duration=r["duration"],
                text=r["text"],
                text_raw=r["text"],
                source=r.get("source", ""),
                extra={k: v for k, v in r.items() if k in ("segments", "rttm")},
            )
        )
    return recs, {}


# --------------------------------------------------------------------------- main


def dataset_revision(repo: str) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha or "unknown"
    except Exception:  # noqa: BLE001 - provenance is best-effort, never blocks the build
        return "unknown"


def read_part(parts: Path, corpus: str) -> list[Record]:
    path = parts / f"{corpus}.jsonl"
    if not path.exists():
        return []
    return [Record(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, recs: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", default=",".join(CORPORA), help=f"subset of {','.join(CORPORA)}")
    ap.add_argument("--n", type=int, default=20, help="utterances per language per corpus")
    ap.add_argument("--min-dur", type=float, default=1.5, help="seconds; shorter clips give unstable WER")
    ap.add_argument("--max-dur", type=float, default=30.0, help="seconds; Whisper's single-window length")
    ap.add_argument("--per-speaker", type=int, default=3, help="max clips per speaker per language")
    ap.add_argument("--iv-scan", type=int, default=1500, help="IndicVoices rows scanned per language")
    ap.add_argument("--vaani-scan", type=int, default=1500, help="Vaani rows scanned per language")
    ap.add_argument("--nirantar-mb", type=int, default=4000, help="compressed MB of AUDIOS.tgz.aa to stream")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP range request otherwise
    load_env()

    chosen = [c.strip() for c in args.corpora.split(",") if c.strip()]
    if bad := set(chosen) - set(CORPORA):
        sys.exit(f"unknown corpora: {sorted(bad)}")
    if {"indicvoices", "vaani"} & set(chosen) and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set (.env); IndicVoices and Vaani are gated.")
    p = Params(
        args.n, args.min_dur, args.max_dur, args.per_speaker, args.iv_scan, args.vaani_scan, args.nirantar_mb
    )

    parts = OUT / "parts"
    gaps_path = OUT / "gaps.json"
    all_gaps: dict[str, dict[str, str]] = (
        json.loads(gaps_path.read_text(encoding="utf-8")) if gaps_path.exists() else {}
    )
    # Dedup across corpora: seed `seen` from parts we are NOT rebuilding.
    seen: set[str] = set()
    for c in CORPORA:
        if c not in chosen and (parts / f"{c}.jsonl").exists():
            for line in (parts / f"{c}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                seen.add(dedup_key(r["language"], r.get("speaker_id"), r["text"]))

    fetchers = {
        "indicvoices": lambda: fetch_hf("indicvoices", p, seen, read_part(parts, "indicvoices")),
        "vaani": lambda: fetch_hf("vaani", p, seen, read_part(parts, "vaani")),
        "nirantar": lambda: fetch_nirantar(p, seen),
        "registers": collect_registers,
    }
    for c in CORPORA:  # fixed order keeps dedup (first corpus wins) deterministic
        if c not in chosen:
            continue
        recs, gaps = fetchers[c]()
        write_jsonl(parts / f"{c}.jsonl", recs)
        all_gaps[c] = gaps
        # Only after a completed fetch: delete wavs from an earlier run that this run didn't write.
        keep = {(ROOT / r.audio).resolve() for r in recs}
        for w in (OUT / "audio" / c).glob("*.wav"):
            if w.resolve() not in keep:
                w.unlink()

    merged: list[dict[str, Any]] = []
    for c in CORPORA:
        if (parts / f"{c}.jsonl").exists():
            merged += [json.loads(x) for x in (parts / f"{c}.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in merged:  # scoring code can branch on this (e.g. transliterate sd hypotheses)
        r["script"] = dominant_script(r["text"])
    (OUT / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8"
    )
    gaps_path.write_text(json.dumps(all_gaps, ensure_ascii=False, indent=2), encoding="utf-8")

    table: dict[str, dict[str, int]] = defaultdict(dict)
    for r in merged:
        table[r["language"]][r["corpus"]] = table[r["language"]].get(r["corpus"], 0) + 1
    summary = {
        "name": "dhvani_eval/asr_whisper",
        "purpose": (
            "faster-whisper ASR WER/CER over Whisper-supported Indic languages (+ en / code-switch registers)"
        ),
        "languages": list(LANGS),
        "params": asdict(p),
        "git_sha": git_sha(),
        "sample_rate": SR,
        "total_clips": len(merged),
        "total_hours": round(sum(r["duration"] for r in merged) / 3600, 3),
        "counts_language_x_corpus": dict(sorted(table.items())),
        "dataset_revisions": {
            "ai4bharat/IndicVoices": dataset_revision("ai4bharat/IndicVoices"),
            "ARTPARK-IISc/Vaani": dataset_revision("ARTPARK-IISc/Vaani"),
            "nirantar": f"{NIRANTAR_AUDIO_URL} + {NIRANTAR_LIL_URL}",
        },
        "gaps": all_gaps,
        "known_issues": {
            "sd": "references are Devanagari; Whisper outputs Sindhi in Perso-Arabic -> transliterate "
            "hypotheses (or references) before WER/CER, or report sd separately",
            "low_resource": "Whisper's training data for as/sa/sd/ne/pa is tiny; expect very high WER",
        },
        "excluded": {
            "air_rs_db": "no transcripts (only read/spontaneous labels) — cannot produce WER/CER",
            "dhvani_eval/nirantar": "existing slice is 0.4-0.6 s chunks, too short for ASR; re-fetched here",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %d clips (%.2f h) -> %s", len(merged), summary["total_hours"], OUT)


if __name__ == "__main__":
    main()
