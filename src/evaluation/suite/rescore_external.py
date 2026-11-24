"""Import externally-produced results (not re-run here) and re-score them uniformly.

Two external runs on dhvani_eval/asr_whisper (833 clips):
  * "result zip sarvam.zip"      — pyannoteAI precision-2 diarization → Conv-TasNet (Libri2Mix)
                                   guided stems on overlaps → Sarvam STT.
  * "results-faster-whisper.zip" — pyannote diarization → SepFormer on overlaps → faster-whisper
                                   (model size unrecorded), CUDA fp16, beam 5. Two conditions:
                                   "original" (raw audio) and "processed" (separation applied).

Their transcripts and diarization outputs are taken AS-IS. Only the scoring is redone, with the
project normalizer + pooled counts, so the numbers are comparable with locally-run systems.
Each external run's *self-reported* headline is stored alongside for transparency.

    python -m src.evaluation.suite.rescore_external
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from ...types import SpeakerTurn
from ..metrics import der, jer
from .common import (
    REPO,
    ASRHyp,
    asr_manifest,
    breakdown,
    log,
    read_jsonl,
    score_asr_rows,
    setup_logging,
    write_json,
    write_rows,
)

SARVAM_ZIP = REPO / "result zip sarvam.zip"
FW_ZIP = REPO / "results-faster-whisper.zip"


def _parse_rttm(text: str) -> dict[str, list[SpeakerTurn]]:
    out: dict[str, list[SpeakerTurn]] = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s, d = float(p[3]), float(p[4])
            out.setdefault(p[1], []).append(SpeakerTurn(s, s + d, p[7]))
    return out


def reference_turns() -> dict[str, list[SpeakerTurn]]:
    """Reference speaker turns for the 24 code-switch clips (the only asr_whisper clips with
    timed turns), read from the project's own RTTMs — not from either external zip."""
    refs: dict[str, list[SpeakerTurn]] = {}
    for m in read_jsonl(REPO / "dhvani_eval" / "registers" / "manifest.jsonl"):
        if m.get("register") == "codeswitch" and m.get("rttm"):
            p = REPO / "dhvani_eval" / m["rttm"]
            refs.update(_parse_rttm(p.read_text(encoding="utf-8")))
    return refs


def _der_block(
    system: str, hyps: dict[str, list[SpeakerTurn]], refs: dict[str, list[SpeakerTurn]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cid, ref in sorted(refs.items()):
        hyp = hyps.get(cid, [])
        for collar in (0.0, 0.25):
            d = der(ref, hyp, collar=collar)
            rows.append(
                {
                    "system": system,
                    "id": cid,
                    "collar": collar,
                    **d,
                    "jer": jer(ref, hyp, collar=collar),
                    "hyp_speakers": len({t.speaker for t in hyp}),
                    "ref_speakers": len({t.speaker for t in ref}),
                }
            )
    summary: dict[str, Any] = {}
    for collar in (0.0, 0.25):
        sel = [r for r in rows if r["collar"] == collar]
        tot = sum(r["total"] for r in sel)
        summary[f"collar_{collar}"] = {
            "clips": len(sel),
            # pooled: total error seconds / total reference speaker-seconds
            "der": round(sum(r["der"] * r["total"] for r in sel) / tot, 4),
            "missed": round(sum(r["missed_s"] for r in sel) / tot, 4),
            "false_alarm": round(sum(r["false_alarm_s"] for r in sel) / tot, 4),
            "confusion": round(sum(r["confusion_s"] for r in sel) / tot, 4),
            "jer_mean": round(sum(r["jer"] for r in sel) / len(sel), 4),
            "speaker_count_acc": round(
                sum(r["hyp_speakers"] == r["ref_speakers"] for r in sel) / len(sel), 4
            ),
        }
    return {"rows": rows, "summary": summary}


def sarvam(manifest: list[dict[str, Any]], refs: dict[str, list[SpeakerTurn]]) -> dict[str, Any]:
    with zipfile.ZipFile(SARVAM_ZIP) as z:
        clips = list(csv.DictReader(io.StringIO(z.read("clips.csv").decode("utf-8-sig"))))
        reported = json.loads(z.read("summary.json"))
        diar: dict[str, list[SpeakerTurn]] = {}
        for cid in refs:
            name = f"cache/{cid}.diarize.json"
            if name in z.namelist():
                diar[cid] = [
                    SpeakerTurn(float(t["start"]), float(t["end"]), str(t["speaker"]))
                    for t in json.loads(z.read(name))
                ]
    man_text = {m["id"]: m["text"] for m in manifest}
    ref_mismatch = sum(c["text"] != man_text.get(c["id"]) for c in clips)
    rows = score_asr_rows("sarvam (external)", [ASRHyp(c["id"], c["transcript"]) for c in clips], manifest)
    write_rows("asr_external_sarvam", rows)
    d = _der_block("pyannoteAI precision-2 (external, via sarvam run)", diar, refs)
    write_rows("diar_external_precision2", d["rows"])
    return {
        "asr": breakdown(rows),
        "der": d["summary"],
        "reference_text_mismatches": ref_mismatch,
        "self_reported": {k: reported[k] for k in ("wer", "cer", "exact_match")}
        | {"excluding_sindhi_wer": reported["excluding_sindhi"]["wer"]},
    }


def faster_whisper(manifest: list[dict[str, Any]], refs: dict[str, list[SpeakerTurn]]) -> dict[str, Any]:
    with zipfile.ZipFile(FW_ZIP) as z:
        recs = [
            json.loads(x)
            for x in z.read("results/full/transcripts.jsonl").decode("utf-8").splitlines()
            if x.strip()
        ]
        metrics_csv = list(
            csv.DictReader(io.StringIO(z.read("results/full/metrics.csv").decode("utf-8-sig")))
        )
        pred = _parse_rttm(z.read("results/full/predicted.rttm").decode("utf-8"))
        reported = json.loads(z.read("results/full/metrics.json"))
    secs = {r["id"]: float(r["seconds_asr_original"]) for r in metrics_csv}
    man_text = {m["id"]: m["text"] for m in manifest}
    out: dict[str, Any] = {
        "reference_text_mismatches": sum(r["reference_text"] != man_text.get(r["id"]) for r in recs),
        "self_reported_keys": sorted(reported.keys())[:20],
    }
    for cond in ("original", "processed"):
        hyps = [
            ASRHyp(r["id"], str(r[cond].get("text", "")), secs.get(r["id"]) if cond == "original" else None)
            for r in recs
        ]
        rows = score_asr_rows(f"faster-whisper {cond} (external)", hyps, manifest)
        write_rows(f"asr_external_fw_{cond}", rows)
        out[cond] = breakdown(rows)
    d = _der_block("pyannote (external, via faster-whisper run)", pred, refs)
    write_rows("diar_external_fw_pyannote", d["rows"])
    out["der"] = d["summary"]
    return out


def main() -> None:
    setup_logging()
    manifest = asr_manifest()
    refs = reference_turns()
    log.info("manifest=%d clips, reference-turn clips=%d", len(manifest), len(refs))
    result = {"sarvam": sarvam(manifest, refs), "faster_whisper": faster_whisper(manifest, refs)}
    path = write_json("external_rescored", result)
    for sysname, blk in (
        ("sarvam", result["sarvam"]["asr"]),
        ("fw original", result["faster_whisper"]["original"]),
        ("fw processed", result["faster_whisper"]["processed"]),
    ):
        o = blk["overall_excl_sd"]
        log.info(
            "%-13s WER %.4f CER %.4f nuktaWER %.4f wrong-script %.3f",
            sysname,
            o["wer"],
            o["cer"],
            o["wer_nukta_folded"],
            o["wrong_script_rate"] or 0,
        )
    log.info("wrote %s", Path(path).relative_to(REPO))


if __name__ == "__main__":
    main()
