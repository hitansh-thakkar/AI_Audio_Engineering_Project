"""Diarization benchmark: the proposal-required hand-applied clustering (spectral | GMM) over
speaker embeddings (MFCC-stats | ECAPA), on three reference sets with timed turns:

  * codeswitch  — 24 constructed 2-speaker clips (registers/codeswitch); the only set the
                  external runs (pyannote community, pyannoteAI precision-2) were scored on.
  * sequential  — 10 clips, 3 speakers, no overlap, multilingual.
  * mixtures    — 24 synthetic 2-speaker overlap mixtures (overlap ratio 0.25-1.0).

Each config runs with K estimated (the real pipeline) and K given (oracle count), so speaker-
count errors are separated from clustering errors. DER via pyannote.metrics, overlap scored,
collars 0 and 0.25 s, pooled by reference speaker-seconds.

Also tests two hypotheses for the high external DER on codeswitch:
  H1 reference turns contain silence (constructed clips use whole-utterance spans) →
     re-score every system against VAD-tightened references;
  H2 the two "speakers" sound alike → ECAPA cosine between the two halves.

    python -m src.evaluation.suite.diar_bench
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from ...diarization import Diarizer
from ...diarization.embeddings import embed_segments
from ...diarization.rttm import read_rttm
from ...diarization.vad import energy_vad
from ...types import AudioClip, SpeakerTurn
from ..metrics import der, jer
from .common import EVAL_ROOT, log, read_jsonl, setup_logging, write_json, write_rows

EMBEDDINGS = ("mfcc", "ecapa")
CLUSTERINGS = ("spectral", "gmm")
COLLARS = (0.0, 0.25)


def datasets() -> dict[str, list[dict[str, Any]]]:
    cs = [
        {"id": m["id"], "audio": m["audio"], "rttm": m["rttm"]}
        for m in read_jsonl(EVAL_ROOT / "registers" / "manifest.jsonl")
        if m.get("register") == "codeswitch" and m.get("rttm")
    ]
    seq = [
        {"id": m["id"], "audio": m["audio"], "rttm": m["rttm"]}
        for m in read_jsonl(EVAL_ROOT / "sequential" / "manifest.jsonl")
    ]
    mix = [
        {"id": m["id"], "audio": m["mix"], "rttm": m["rttm"]}
        for m in read_jsonl(EVAL_ROOT / "overlap_mixtures" / "manifest.jsonl")
    ]
    return {"codeswitch": cs, "sequential": seq, "mixtures": mix}


def _load(m: dict[str, Any]) -> AudioClip:
    y, sr = sf.read(EVAL_ROOT / m["audio"], dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return AudioClip(samples=y, sample_rate=sr, source_path=str(EVAL_ROOT / m["audio"]))


def _cfg(embedding: str, clustering: str, k: int | None):
    return OmegaConf.create(
        dict(
            source="real",
            backend="community1",
            model="x",
            clustering=clustering,
            overlap_detector="none",
            emit_timeline_json=False,
            embedding=embedding,
            vad_top_db=30,
            window_seconds=1.5,
            window_hop=0.75,
            num_speakers=k,
            min_speakers=None,
            max_speakers=None,
            rttm_dir=None,
        )
    )


def vad_tight(ref: list[SpeakerTurn], clip: AudioClip, top_db: int = 30) -> list[SpeakerTurn]:
    """Intersect reference turns with energy-VAD speech regions (H1)."""
    speech = energy_vad(clip.samples, clip.sample_rate, top_db=top_db)
    out: list[SpeakerTurn] = []
    for t in ref:
        for s, e in speech:
            a, b = max(t.start, s), min(t.end, e)
            if b - a > 0.01:
                out.append(SpeakerTurn(a, b, t.speaker))
    return out


def _pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tot = sum(r["total"] for r in rows)
    if not rows or not tot:
        return {"clips": len(rows)}
    return {
        "clips": len(rows),
        "der": round(sum(r["der"] * r["total"] for r in rows) / tot, 4),
        "missed": round(sum(r["missed_s"] for r in rows) / tot, 4),
        "false_alarm": round(sum(r["false_alarm_s"] for r in rows) / tot, 4),
        "confusion": round(sum(r["confusion_s"] for r in rows) / tot, 4),
        "jer_mean": round(float(np.mean([r["jer"] for r in rows])), 4),
        "speaker_count_acc": round(float(np.mean([r["hyp_speakers"] == r["ref_speakers"] for r in rows])), 4),
    }


def _score(
    system: str, dataset: str, cid: str, ref: list[SpeakerTurn], hyp: list[SpeakerTurn], ref_kind: str
) -> list[dict[str, Any]]:
    return [
        {
            "system": system,
            "dataset": dataset,
            "id": cid,
            "reference": ref_kind,
            "collar": c,
            **der(ref, hyp, collar=c),
            "jer": jer(ref, hyp, collar=c),
            "hyp_speakers": len({t.speaker for t in hyp}),
            "ref_speakers": len({t.speaker for t in ref}),
        }
        for c in COLLARS
    ]


def _external(name: str) -> dict[str, list[SpeakerTurn]]:
    """Hypothesis turns of the external runs are not stored as turns in our CSVs; re-read zips."""
    import json
    import zipfile

    from .rescore_external import FW_ZIP, SARVAM_ZIP, _parse_rttm

    if name == "fw":
        with zipfile.ZipFile(FW_ZIP) as z:
            return _parse_rttm(z.read("results/full/predicted.rttm").decode("utf-8"))
    out: dict[str, list[SpeakerTurn]] = {}
    with zipfile.ZipFile(SARVAM_ZIP) as z:
        for n in z.namelist():
            if n.startswith("cache/") and n.endswith(".diarize.json"):
                cid = n.split("/")[1].removesuffix(".diarize.json")
                out[cid] = [
                    SpeakerTurn(float(t["start"]), float(t["end"]), str(t["speaker"]))
                    for t in json.loads(z.read(n))
                ]
    return out


def main() -> None:
    setup_logging()
    sets = datasets()
    run_cfg = OmegaConf.create(dict(seed=42))
    clips = {(d, m["id"]): (_load(m), read_rttm(EVAL_ROOT / m["rttm"])) for d, ms in sets.items() for m in ms}
    rows: list[dict[str, Any]] = []
    timing: dict[str, float] = {}

    for emb in EMBEDDINGS:
        for clu in CLUSTERINGS:
            for kmode in ("estimated", "oracle"):
                system = f"{emb}+{clu} (K {kmode})"
                t0 = time.perf_counter()
                audio_s = 0.0
                for (d, cid), (clip, ref) in clips.items():
                    k = len({t.speaker for t in ref}) if kmode == "oracle" else None
                    hyp = Diarizer(_cfg(emb, clu, k), run_cfg).run(clip, []).turns
                    audio_s += clip.duration
                    rows += _score(system, d, cid, ref, hyp, "original")
                    if d == "codeswitch":
                        rows += _score(system, d, cid, vad_tight(ref, clip), hyp, "vad_tight")
                timing[system] = round((time.perf_counter() - t0) / audio_s, 4)
                log.info("%s done rtf=%.3f", system, timing[system])

    # external systems, codeswitch only, both reference variants (H1)
    for name, label in (
        ("fw", "pyannote community (external, FW run)"),
        ("sarvam", "pyannoteAI precision-2 (external, Sarvam run)"),
    ):
        hyps = _external(name)
        for (d, cid), (clip, ref) in clips.items():
            if d == "codeswitch":
                hyp = hyps.get(cid, [])
                rows += _score(label, d, cid, ref, hyp, "original")
                rows += _score(label, d, cid, vad_tight(ref, clip), hyp, "vad_tight")

    # H1 magnitude: share of reference speaker-time that energy-VAD calls non-speech
    cs = [(clip, ref) for (d, _), (clip, ref) in clips.items() if d == "codeswitch"]
    ref_s = sum(t.end - t.start for _, ref in cs for t in ref)
    tight_s = sum(t.end - t.start for clip, ref in cs for t in vad_tight(ref, clip))
    # H2: ECAPA cosine between speaker halves, vs same-clip-same-speaker baseline unavailable →
    # compare against cross-clip pairs of the sequential set (genuinely different speakers).
    sims = []
    for clip, ref in cs:
        spk = sorted({t.speaker for t in ref})
        if len(spk) == 2:
            segs = [next((t.start, t.end) for t in ref if t.speaker == s) for s in spk]
            e = embed_segments(clip.samples, clip.sample_rate, segs, "ecapa")
            sims.append(float(e[0] @ e[1]))
    diff = []
    for clip, ref in [v for (d, _), v in clips.items() if d == "sequential"]:
        spk = sorted({t.speaker for t in ref})
        segs = [next((t.start, t.end) for t in ref if t.speaker == s) for s in spk]
        e = embed_segments(clip.samples, clip.sample_rate, segs, "ecapa")
        diff += [float(e[i] @ e[j]) for i in range(len(e)) for j in range(i + 1, len(e))]

    summary: dict[str, Any] = {"rtf": timing, "systems": {}}
    keys = sorted({(r["system"], r["dataset"], r["reference"], r["collar"]) for r in rows})
    for sysname, d, refk, c in keys:
        sel = [
            r
            for r in rows
            if (r["system"], r["dataset"], r["reference"], r["collar"]) == (sysname, d, refk, c)
        ]
        summary["systems"].setdefault(sysname, {}).setdefault(d, {})[f"{refk}@{c}"] = _pooled(sel)
    summary["h1_reference_silence_share"] = round(1 - tight_s / ref_s, 4)
    summary["h2_ecapa_cos_codeswitch_halves"] = {"median": round(float(np.median(sims)), 3), "n": len(sims)}
    summary["h2_ecapa_cos_sequential_diff_speakers"] = {
        "median": round(float(np.median(diff)), 3),
        "n": len(diff),
    }
    write_rows("diar_bench", rows)
    write_json("diar_bench", summary)
    log.info(
        "H1 silence share %.3f | H2 cs cos %s vs diff-speaker cos %s",
        summary["h1_reference_silence_share"],
        summary["h2_ecapa_cos_codeswitch_halves"],
        summary["h2_ecapa_cos_sequential_diff_speakers"],
    )


if __name__ == "__main__":
    main()
