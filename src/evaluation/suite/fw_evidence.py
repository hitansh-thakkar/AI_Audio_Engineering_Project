"""Evidence for *why* faster-whisper scores the way it does on dhvani_eval (report 2, section 6).

Reads the teammate's results zip, the manifest and Whisper's tokenizer (no model is run), and
writes outputs/reports/data/fw_evidence.json, which build_reports.py turns into tables:

  * tokens per character by language under Whisper's byte-level BPE, and how many reference
    transcripts cannot fit in one decode window (~224 new tokens);
  * truncation fingerprints in the teammate's output (U+FFFD = a multi-byte character cut in
    half, longest segment length) and how much of the WER comes from capped clips;
  * reference-variant sensitivity: scoring against `text` (standard spelling) vs `text_raw`
    (verbatim pronunciation), to rule out the references as the cause.

    python -m src.evaluation.suite.fw_evidence
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..metrics import EditCounts, char_edits, pooled, word_edits
from .common import asr_manifest, log, setup_logging, write_json
from .rescore_external import FW_ZIP

# faster-whisper's default per-window budget: max_length 448 // 2 new tokens (generate_segments)
WINDOW_TOKENS = 224
CAP_NEAR = 215  # a segment this long almost certainly stopped at the budget, not at end-of-text


def _tokenizer() -> Any:
    from tokenizers import Tokenizer

    snaps = sorted(
        Path.home().glob(
            ".cache/huggingface/hub/models--Systran--faster-whisper-*/snapshots/*/tokenizer.json"
        )
    )
    if not snaps:
        raise FileNotFoundError("no cached faster-whisper tokenizer.json (run any fw: backend once)")
    return Tokenizer.from_file(str(snaps[-1]))  # all Whisper v3/v2 checkpoints share this BPE


def _loop(h: str) -> bool:
    w = h.split()
    return len(w) >= 8 and max(w.count(x) for x in set(w)) >= max(4, len(w) // 3)


def main() -> None:
    setup_logging()
    tok = _tokenizer()
    man = {m["id"]: m for m in asr_manifest() if m["language"] != "sd"}  # Sindhi refs are Arabic

    def ntok(s: str) -> int:
        return len(tok.encode(" " + s).ids)

    # 1. tokenizer cost per language, on the *reference* text (what a perfect system must emit)
    per_lang: dict[str, dict[str, float]] = defaultdict(
        lambda: {"chars": 0, "tokens": 0, "clips": 0, "over": 0}
    )
    for m in man.values():
        d = per_lang[m["language"]]
        n = ntok(m["text"])
        d["chars"] += len(m["text"])
        d["tokens"] += n
        d["clips"] += 1
        d["over"] += n > WINDOW_TOKENS
    tok_rows = {
        lang: {
            "tokens per character": d["tokens"] / d["chars"],
            "clips": d["clips"],
            f"refs > {WINDOW_TOKENS} tokens": d["over"],
            "share": d["over"] / d["clips"],
        }
        for lang, d in sorted(per_lang.items())
    }

    # 2. truncation fingerprints in the teammate's run
    with zipfile.ZipFile(FW_ZIP) as z:
        recs = [
            json.loads(x)
            for x in z.read("results/full/transcripts.jsonl").decode("utf-8").splitlines()
            if x.strip()
        ]
    recs = [r for r in recs if r["id"] in man]
    groups: dict[str, list[EditCounts]] = defaultdict(list)
    fffd = 0
    max_seg = 0
    for r in recs:
        m = man[r["id"]]
        h = str(r["original"].get("text", ""))
        segs = r["original"].get("whisper_segments", {}).get("main", [])
        seg_tok = [len(tok.encode(s["text"]).ids) for s in segs]
        max_seg = max([max_seg, *seg_tok])
        fffd += "�" in h
        capped = any(n >= CAP_NEAR for n in seg_tok) or "�" in h
        if capped:
            g = "hit the cap, repetition loop" if _loop(h) else "hit the cap, no loop"
        else:
            g = "reference too long, not capped" if ntok(m["text"]) > WINDOW_TOKENS else "fits in one window"
        groups[g].append(word_edits(m["text"], h, m["language"]))
    total = sum(e.errors for v in groups.values() for e in v)
    cap_rows = {
        g: {
            "clips": len(v),
            "WER": pooled(v).rate,
            "share of all word errors": sum(e.errors for e in v) / total,
        }
        for g, v in sorted(groups.items())
    }

    # 3. reference variant sensitivity (teammate run, full set)
    rv: dict[str, dict[str, list[EditCounts]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        m = man[r["id"]]
        h = str(r["original"].get("text", ""))
        for key in ("text", "text_raw"):
            ref = m.get(key) or m["text"]
            rv["all"][f"{key} WER"].append(word_edits(ref, h, m["language"]))
            rv["all"][f"{key} CER"].append(char_edits(ref, h, m["language"]))
    ref_rows = {k: pooled(v).rate for k, v in rv["all"].items()}

    out = {
        "window_tokens": WINDOW_TOKENS,
        "tokens_by_language": tok_rows,
        "fffd_outputs": fffd,
        "clips": len(recs),
        "longest_segment_tokens": max_seg,
        "cap_groups": cap_rows,
        "reference_variant": ref_rows,
    }
    p = write_json("fw_evidence", out)
    log.info("wrote %s (U+FFFD in %d/%d outputs, longest segment %d tokens)", p, fffd, len(recs), max_seg)


if __name__ == "__main__":
    main()
