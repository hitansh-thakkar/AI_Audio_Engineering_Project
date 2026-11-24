"""LLM post-processing benchmark (stage 5). Reference-free, because there are no gold summaries:
every metric checks the output against the *input transcript* it was given.

Inputs: 10 sequential clips (3 speakers, as/bn/hi) + 24 code-switch clips (hi/en), using their
reference segment texts as the transcript (oracle ASR), so LLM behaviour is measured in isolation.

Metrics per backend:
  * llm_ok          — the model produced a parseable report (not the heuristic fallback)
  * rewrite_cer     — pooled CER of cleaned dialogue vs input text. The stage may only
                      reorganise/clean (golden rule 5); a large value means it rewrote speech.
  * lang_acc        — per-segment language label matches the reference language
  * translation_rate— share of non-English segments that received an English translation
  * grounding       — LaBSE (cross-lingual) cosine of each summary sentence to its best-matching
                      input segment; we report the mean and the share of sentences < 0.30
  * novel_numbers   — digits in summary/tags that never occur in the input (fabrication proxy)
  * guardrail       — the project's own check_factual_consistency status
  * latency_s

    python -m src.evaluation.suite.llm_bench --backends gemini:gemini-3.8-flash ollama:airavata-tulu
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from ...postprocessing.postprocess import create_postprocessor
from ...types import Transcript, TranscriptSegment
from ..metrics import char_edits, pooled
from .common import DATA_OUT, EVAL_ROOT, log, read_jsonl, setup_logging, write_json, write_rows

LANG_WORDS = {"hi": "hindi", "en": "english", "bn": "bengali", "as": "assamese"}


Item = tuple[str, Transcript, list[str | None]]  # (clip id, oracle transcript, reference langs)


def inputs() -> list[Item]:
    out: list[Item] = []
    rows = [m for m in read_jsonl(EVAL_ROOT / "sequential" / "manifest.jsonl")]
    rows += [
        m for m in read_jsonl(EVAL_ROOT / "registers" / "manifest.jsonl") if m.get("register") == "codeswitch"
    ]
    for m in rows:
        segs = [
            TranscriptSegment(
                float(s["start"]), float(s["end"]), str(s["speaker"]), str(s["text"]), lang=None
            )
            for s in m["segments"]
        ]
        # the reference language stays out of the LLM input; it is kept alongside for scoring
        out.append((m["id"], Transcript(segs), [s.get("language") for s in m["segments"]]))
    return out


_SPEAKER_LABEL = re.compile(r"\b(?:speaker|spk)[\s_]*\d+", re.IGNORECASE)


def _numbers(text: str) -> set[int]:
    r"""Numbers as values, so Devanagari/Bengali digits match ASCII ones (re's \d is Unicode).
    "Speaker 2" labels are structure, not content claims, and are excluded."""
    return {int(n) for n in re.findall(r"\d+", _SPEAKER_LABEL.sub(" ", text))}


_LABSE: Any = None


def _labse() -> Any:
    global _LABSE
    if _LABSE is None:
        from sentence_transformers import SentenceTransformer

        _LABSE = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    return _LABSE


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if len(s.strip()) > 3]


def grounding(summary: str, transcript: Transcript) -> list[float]:
    sents = _sentences(summary)
    src = [s.text for s in transcript.segments if s.text.strip()]
    if not sents or not src:
        return []
    m = _labse()
    a = m.encode(sents, normalize_embeddings=True)
    b = m.encode(src, normalize_embeddings=True)
    return [float(x) for x in (a @ b.T).max(axis=1)]


def doc_grounding(rows: list[dict[str, Any]], items: dict[str, Transcript]) -> dict[str, float | None]:
    """Document-level LaBSE grounding with a calibrated null.

    Sentence-vs-segment cosine barely separates a summary from a *mismatched* transcript here
    (AUC 0.73), because one summary sentence fuses several turns. Whole-summary vs whole-transcript
    does (AUC 0.94), so that is the headline. The null is every mismatched (summary, transcript)
    pair of the same backend; a summary is "grounded" if its own cosine exceeds the null's p95.
    retrieval_top1 = share of summaries whose nearest transcript is their own (specificity)."""
    ok = [r for r in rows if r["llm_ok"] and str(r["summary"]).strip()]
    if len(ok) < 3:
        return {"doc_grounding": None, "doc_grounded_rate": None, "retrieval_top1": None}
    m = _labse()
    s = m.encode([str(r["summary"]) for r in ok], normalize_embeddings=True)
    t = m.encode([" ".join(x.text for x in items[r["id"]].segments) for r in ok], normalize_embeddings=True)
    c = s @ t.T
    diag = np.diag(c)
    thr = float(np.quantile(c[~np.eye(len(ok), dtype=bool)], 0.95))
    for r, d in zip(ok, diag, strict=True):
        r["doc_grounding"] = round(float(d), 4)
    return {
        "doc_grounding": round(float(np.median(diag)), 4),
        "doc_grounded_rate": round(float((diag > thr).mean()), 4),
        "null_p95": round(thr, 4),
        "retrieval_top1": round(float((c.argmax(1) == np.arange(len(ok))).mean()), 4),
    }


def _cfg(backend: str, model: str):
    return OmegaConf.create(
        dict(
            backend=backend,
            model=model,
            api_key_env="GEMINI_API_KEY",
            host="http://localhost:11434",
            structured_output=True,
            factual_consistency_check=True,
            forbid_new_facts=True,
            max_retries=4,
            timeout_s=240,
        )
    )


def run_backend(
    spec: str, items: list[Item]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    backend, model = spec.split(":", 1)
    pp = create_postprocessor(_cfg(backend, model))
    pp.load()
    # Per-row cache so a long local/free-tier run resumes. Only successful LLM reports are cached:
    # a fallback (quota, 503, timeout) is retried next run rather than frozen in as a result.
    cache = DATA_OUT / "hyps" / ("llm_" + spec.replace(":", "_").replace("/", "_") + ".jsonl")
    cache.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["id"]] = d
    rows: list[dict[str, Any]] = []
    for cid, tr, ref_langs in items:
        if cid in done:
            row = done[cid]
            # derived metrics are recomputed so metric fixes apply to cached LLM outputs too
            src_n = _numbers(" ".join(s.text for s in tr.segments))
            row["novel_numbers"] = len(_numbers(f"{row['summary']} {row['tags'].replace('|', ' ')}") - src_n)
            rows.append(row)
            continue
        inp_text = [s.text for s in tr.segments]
        notes: list[str] = []
        t0 = time.perf_counter()
        rep = pp.run(
            Transcript([TranscriptSegment(s.start, s.end, s.speaker, s.text) for s in tr.segments]), notes
        )
        dt = time.perf_counter() - t0
        ok = not any(
            ("fall" in n.lower() and "back" in n.lower()) or "unavailable" in n.lower() for n in notes
        )
        edits = pooled([char_edits(a, d.text, "hi") for a, d in zip(inp_text, rep.dialogue, strict=False)])
        lang_hits = [
            (LANG_WORDS.get(str(r or "")[:2], "?") in (d.lang or "").lower())
            for r, d in zip(ref_langs, rep.dialogue, strict=False)
            if r
        ]
        non_en = [d for r, d in zip(ref_langs, rep.dialogue, strict=False) if r and r != "en"]
        g = grounding(rep.summary, tr) if ok else []
        src_digits = _numbers(" ".join(inp_text))
        out_digits = _numbers(rep.summary + " " + " ".join(map(str, rep.tags)))
        rows.append(
            {
                "backend": spec,
                "id": cid,
                "llm_ok": ok,
                "latency_s": round(dt, 2),
                "rewrite_char_errors": edits.errors,
                "rewrite_ref_chars": edits.ref_units,
                "lang_hits": sum(lang_hits),
                "lang_total": len(lang_hits),
                "translated": sum(bool(d.translation) for d in non_en),
                "non_en": len(non_en),
                "summary_sentences": len(_sentences(rep.summary)),
                "grounding_mean": round(float(np.mean(g)), 4) if g else None,
                "grounding_min": round(float(np.min(g)), 4) if g else None,
                "ungrounded_sentences": sum(x < 0.30 for x in g),
                "novel_numbers": len(out_digits - src_digits),
                "n_tags": len(rep.tags),
                "guardrail": rep.guardrail_status,
                "summary": rep.summary,
                "tags": "|".join(map(str, rep.tags)),
                "notes": " || ".join(notes)[:400],
            }
        )
        if ok:
            with cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        log.info("%s %s ok=%s %.1fs", spec, cid, ok, dt)
    ok_rows = [r for r in rows if r["llm_ok"]]
    sents = sum(r["summary_sentences"] for r in ok_rows)

    def ratio(a: str, b: str) -> float | None:
        den = sum(r[b] for r in ok_rows)
        return round(sum(r[a] for r in ok_rows) / den, 4) if den else None

    summary = {
        "backend": spec,
        "inputs": len(rows),
        "llm_ok_rate": round(len(ok_rows) / len(rows), 4),
        "rewrite_cer": ratio("rewrite_char_errors", "rewrite_ref_chars"),
        "lang_acc": ratio("lang_hits", "lang_total"),
        "translation_rate": ratio("translated", "non_en"),
        **doc_grounding(rows, {cid: tr for cid, tr, _ in items}),
        # sentence-level LaBSE kept only as a diagnostic: poorly calibrated (see doc_grounding)
        "sentence_ungrounded_rate_diag": round(sum(r["ungrounded_sentences"] for r in ok_rows) / sents, 4)
        if sents
        else None,
        "novel_numbers_per_report": round(statistics.fmean([r["novel_numbers"] for r in ok_rows]), 3)
        if ok_rows
        else None,
        "guardrail_pass_rate": round(sum(r["guardrail"] == "PASSED" for r in ok_rows) / len(ok_rows), 4)
        if ok_rows
        else None,
        "latency_median_s": round(statistics.median([r["latency_s"] for r in rows]), 2),
    }
    return summary, rows


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", required=True)
    args = ap.parse_args()
    items = inputs()
    summaries, all_rows = [], []
    for spec in args.backends:
        s, rows = run_backend(spec, items)
        log.info("%s", s)
        summaries.append(s)
        all_rows += rows
        name = "llm_" + spec.replace(":", "_").replace("/", "_").replace(".", "-")
        write_rows(name, rows)
        write_json(name, s)


if __name__ == "__main__":
    main()
