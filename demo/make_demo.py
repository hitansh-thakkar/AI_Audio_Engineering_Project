"""Build the Dhvani demo bundle: run the CURRENT pipeline (config/live.yaml) on 4 clips and
lay the results out as demo/before/<clip>.wav (raw input) + demo/after/<clip>/ (outputs).

    python demo/make_demo.py

Each clip showcases a different capability (overlap separation, multi-speaker timeline,
real-world conversational audio, code-switching). Uses the real stack: ECAPA + spectral
clustering + pyannote overlap detection + SepFormer + IndicWav2Vec ASR + Gemini report.
Needs HF_TOKEN (pyannote + IndicWav2Vec, gated-but-free) and GEMINI_API_KEY (both in .env).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline import run_pipeline  # noqa: E402

DEMO = REPO / "demo"
CONFIG = str(REPO / "config" / "live.yaml")

# friendly demo name -> (source clip, what it demonstrates)
CLIPS = {
    "01_overlap_two_speakers": (
        "dhvani_eval/overlap_mixtures/mix/mix_002.wav",
        "Two speakers overlapping — overlap is detected and separated into clean per-speaker tracks.",
    ),
    "02_multispeaker_timeline": (
        "dhvani_eval/sequential/audio/clip_01.wav",
        "A multi-speaker turn-taking clip — hand-applied spectral clustering builds the speaker timeline.",
    ),
    "03_realworld_conversation": (
        "dhvani_eval/nirantar/audio/1125899906944868_chunk_1.wav",
        "Real spontaneous conversational audio (background chatter) — robustness on messy input.",
    ),
    "04_codeswitch_hindi_english": (
        "dhvani_eval/registers/codeswitch/audio/cs_000.wav",
        "Hindi-English code-switched speech — localized Indic ASR transcribes to native script.",
    ),
}


def main() -> None:
    before = DEMO / "before"
    after = DEMO / "after"
    for d in (before, after):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (src, what) in CLIPS.items():
        src_path = REPO / src
        if not src_path.exists():
            print(f"skip {name}: missing {src}")
            continue
        raw = before / f"{name}{src_path.suffix}"
        shutil.copy(src_path, raw)
        print(f"running pipeline on {name} ...")
        manifest = run_pipeline(str(raw), config=CONFIG, out=str(after))
        art = manifest.artifacts
        report = {}
        rj = after / name / "report.json"
        if rj.exists():
            report = json.loads(rj.read_text(encoding="utf-8"))
        rows.append({
            "name": name, "what": what,
            "speakers": art.get("num_speakers"),
            "overlaps": art.get("num_overlap_regions"),
            "separated": art.get("num_separated_sources"),
            "tags": report.get("tags", []),
        })

    _write_readme(rows)
    print(f"\nDemo ready at {DEMO}")


def _write_readme(rows: list[dict]) -> None:
    lines = [
        "# Dhvani — Demo Bundle",
        "",
        "A self-contained walk-through of the **current** pipeline on four clips. Each clip",
        "shows a different capability. `before/` holds the raw input you start with; `after/`",
        "holds everything the pipeline produced.",
        "",
        "## The pipeline (what happens to each clip)",
        "",
        "```",
        "raw audio (before/<clip>.wav)",
        "   1. Preprocess   16 kHz mono, high-pass, loudness-normalize, trim",
        "   2. Diarize      ECAPA embeddings -> hand-applied Spectral clustering -> who-spoke-when",
        "                   + pyannote overlap detection (simultaneous speech)",
        "   3. Separate     SepFormer on overlap regions -> clean per-speaker tracks",
        "   4. ASR          IndicWav2Vec (AI4Bharat) -> native-script transcript per speaker",
        "   5. Report       Gemini -> executive summary + keyword tags + cleaned dialogue",
        "-> after/<clip>/   processed.wav, speakers/*.wav, separated/*.wav, timeline.json,",
        "                   <clip>.rttm, transcript in report.json, report.md, manifest.json",
        "```",
        "",
        "## The four clips",
        "",
        "| # | Clip | What it demonstrates | Speakers | Overlaps | Separated |",
        "|---|------|----------------------|:--------:|:--------:|:---------:|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | `{r['name']}` | {r['what']} | {r['speakers']} | "
                     f"{r['overlaps']} | {r['separated']} |")
    lines += [
        "",
        "## How to watch the demo",
        "",
        "1. **Listen to the input:** `before/<clip>.wav` — the messy original.",
        "2. **Listen to the result:** `after/<clip>/speakers/*.wav` — one clean track per speaker.",
        "3. **See who spoke when:** open `after/<clip>/timeline.json` (also `<clip>.rttm`).",
        "4. **Read the transcript + summary:** open `after/<clip>/report.md`.",
        "5. **Check the instrumentation:** `after/<clip>/manifest.json` has per-stage wall-time,",
        "   peak memory, model versions, seeds, config hash, and any graceful-degradation notes.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# whole bundle",
        "python demo/make_demo.py",
        "",
        "# or a single clip through the same pipeline",
        "python -m src.pipeline --audio demo/before/01_overlap_two_speakers.wav \\",
        "    --config config/live.yaml --out demo/after",
        "```",
        "",
        "Credentials (both free, in `.env`): `HF_TOKEN` (pyannote + IndicWav2Vec, gated-but-free)",
        "and `GEMINI_API_KEY` (report stage). Without them the pipeline still runs and degrades",
        "gracefully (per-speaker tracks without overlap split / a minimal report), noting it in the manifest.",
    ]
    (DEMO / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
