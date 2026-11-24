"""Shared helpers for the tool-comparison suite (Deliverable 2).

Every ASR system — run locally or imported from an external run — is scored by
`score_asr_rows`, so all WER/CER numbers in the reports share one normalizer and one pooling
rule. Results land in outputs/reports/data/ as CSV (per-clip) + JSON (aggregates).
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..metrics import EditCounts, char_edits, pooled, word_edits
from ..normalize import indic_normalizer_available, normalize_text, script_share

REPO = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO / "dhvani_eval"
ASR_ROOT = EVAL_ROOT / "asr_whisper"
DATA_OUT = REPO / "outputs" / "reports" / "data"

log = logging.getLogger("suite")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def asr_manifest() -> list[dict[str, Any]]:
    return read_jsonl(ASR_ROOT / "manifest.jsonl")


@dataclass(frozen=True)
class ASRHyp:
    clip_id: str
    hypothesis: str
    seconds: float | None = None  # wall-clock decode time, when the system recorded it


def _lang_for_norm(lang: str) -> str | None:
    # code-switch clips are labelled "hi" in asr_whisper (or "hi-en" in registers); Hindi
    # normalization is a no-op on the Latin half, so "hi" is correct for both halves.
    return "hi" if lang.startswith("hi") else lang


def score_asr_rows(
    system: str, hyps: Iterable[ASRHyp], manifest: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Score hypotheses against the manifest references. Missing clips count as empty output
    (all deletions) — a system must not look better by silently skipping hard clips."""
    if not indic_normalizer_available():
        raise RuntimeError("indic-nlp-library is required for reported WER/CER numbers")
    manifest = asr_manifest() if manifest is None else manifest
    by_id = {h.clip_id: h for h in hyps}
    rows: list[dict[str, Any]] = []
    for m in manifest:
        h = by_id.get(m["id"])
        hyp = h.hypothesis if h else ""
        lang = _lang_for_norm(m["language"])
        w = word_edits(m["text"], hyp, lang)
        c = char_edits(m["text"], hyp, lang)
        wn = word_edits(m["text"], hyp, lang, fold_nukta=True)
        share = None if m["register"] == "codeswitch" else script_share(hyp, m["language"])
        rows.append(
            {
                "system": system,
                "id": m["id"],
                "corpus": m["corpus"],
                "language": m["language"],
                "register": m["register"],
                "duration": m["duration"],
                "missing": h is None,
                "reference": m["text"],
                "hypothesis": hyp,
                "ref_norm": normalize_text(m["text"], lang),
                "hyp_norm": normalize_text(hyp, lang),
                **{
                    f"w_{k}": getattr(w, k)
                    for k in ("hits", "substitutions", "deletions", "insertions", "ref_units")
                },
                **{
                    f"c_{k}": getattr(c, k)
                    for k in ("hits", "substitutions", "deletions", "insertions", "ref_units")
                },
                "wn_errors": wn.errors,
                "wn_ref_units": wn.ref_units,
                "wer": round(w.rate, 4),
                "cer": round(c.rate, 4),
                "script_share": None if share is None else round(share, 4),
                "seconds": h.seconds if h else None,
            }
        )
    return rows


def _counts(rows: list[dict[str, Any]], prefix: str) -> EditCounts:
    return pooled(
        [
            EditCounts(
                r[f"{prefix}_hits"],
                r[f"{prefix}_substitutions"],
                r[f"{prefix}_deletions"],
                r[f"{prefix}_insertions"],
                r[f"{prefix}_ref_units"],
            )
            for r in rows
        ]
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled WER/CER (+ nukta-folded WER, script-ID failure rate) for one group of rows."""
    if not rows:
        return {"clips": 0}
    w, c = _counts(rows, "w"), _counts(rows, "c")
    wn_ref = sum(r["wn_ref_units"] for r in rows)
    shares = [r["script_share"] for r in rows if r["script_share"] is not None]
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    audio = sum(r["duration"] for r in rows)
    return {
        "clips": len(rows),
        "missing": sum(r["missing"] for r in rows),
        "ref_words": w.ref_units,
        "wer": round(w.rate, 4),
        "cer": round(c.rate, 4),
        "wer_nukta_folded": round(sum(r["wn_errors"] for r in rows) / wn_ref, 4) if wn_ref else None,
        "substitutions": w.substitutions,
        "deletions": w.deletions,
        "insertions": w.insertions,
        "exact_match": round(sum(r["ref_norm"] == r["hyp_norm"] for r in rows) / len(rows), 4),
        # a clip "fails script ID" when < 50 % of its output letters are in the expected script
        "wrong_script_rate": round(sum(s < 0.5 for s in shares) / len(shares), 4) if shares else None,
        "rtf": round(sum(secs) / audio, 4) if secs and audio else None,
    }


def breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline excludes Sindhi: references are Devanagari, systems often emit Perso-Arabic,
    so literal-script WER there measures transliteration, not recognition."""
    out: dict[str, Any] = {
        "overall_excl_sd": aggregate([r for r in rows if r["language"] != "sd"]),
        "overall_all": aggregate(rows),
    }
    for key in ("corpus", "language", "register"):
        out[f"by_{key}"] = {
            v: aggregate([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})
        }
    return out


def write_rows(name: str, rows: list[dict[str, Any]]) -> Path:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    path = DATA_OUT / f"{name}.csv"
    if rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return path


def write_json(name: str, obj: Any) -> Path:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    path = DATA_OUT / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
