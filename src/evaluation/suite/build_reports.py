"""Build the charts (PNG) and computed tables (markdown) used by the two reports.

Reads only the benchmark outputs in outputs/reports/data/ — no model is run here, so the
reports are reproducible from the committed CSV/JSON. Every number is recomputed as a pooled
ratio (sum of edits / sum of reference units), never as a mean of per-clip rates.

    python -m src.evaluation.suite.build_reports
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import DATA_OUT  # noqa: E402

OUT = DATA_OUT.parent
CHARTS = OUT / "charts"
LANG = {
    "as": "Assamese", "bn": "Bengali", "en": "English", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "pa": "Punjabi",
    "sa": "Sanskrit", "sd": "Sindhi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}  # fmt: skip
# reference categorical palette, fixed order (dataviz references/palette.md, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INT_COLS = {"clips", "errors", "ins", "peak GPU MB"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ASR = {
    "Sarvam (external)": "asr_external_sarvam",
    "faster-whisper, model unrecorded (external)": "asr_external_fw_original",
    "faster-whisper + separation (external)": "asr_external_fw_processed",
    "MMS-1B-all (local)": "asr_mms",
    "IndicWav2Vec-Hindi (local)": "asr_indicw2v",
    "Gemini 3.5 Flash-Lite (free API)": "asr_gemini_gemini-3.5-flash-lite",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
            "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
            "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.grid": True,
            "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
            "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
            "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        }
    )  # fmt: skip


def _save(fig: Any, name: str) -> str:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS / f"{name}.png", dpi=150)
    plt.close(fig)
    return f"charts/{name}.png"


def _md(df: pd.DataFrame, floatfmt: str = "{:.1%}") -> str:
    def f(v: Any, col: str) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            if col in INT_COLS:  # counts survive iterrows() as floats; print them as integers
                return str(int(v))
            return floatfmt.format(v)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append(
            "| " + " | ".join([str(idx)] + [f(v, c) for v, c in zip(row.tolist(), cols, strict=True)]) + " |"
        )
    return "\n".join(lines)


def _pooled(g: pd.DataFrame) -> pd.Series:
    w = (g.w_substitutions + g.w_deletions + g.w_insertions).sum()
    c = (g.c_substitutions + g.c_deletions + g.c_insertions).sum()
    return pd.Series(
        {
            "WER": w / g.w_ref_units.sum(),
            "CER": c / g.c_ref_units.sum(),
            "sub": g.w_substitutions.sum() / g.w_ref_units.sum(),
            "del": g.w_deletions.sum() / g.w_ref_units.sum(),
            "ins": g.w_insertions.sum() / g.w_ref_units.sum(),
            "clips": len(g),
        }
    )


def _loop(t: str) -> bool:
    """Repetition loop: >= 8 words and some word trigram occurring >= 4 times."""
    w = str(t).split()
    if len(w) < 8:
        return False
    return collections.Counter(zip(w, w[1:], w[2:], strict=False)).most_common(1)[0][1] >= 4


def load_asr() -> pd.DataFrame:
    frames = []
    for name, f in ASR.items():
        p = DATA_OUT / f"{f}.csv"
        if p.exists():
            frames.append(pd.read_csv(p).assign(system=name))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ ASR
def asr_section(t: dict[str, str]) -> None:
    df = load_asr()
    df = df[df.language != "sd"]  # Sindhi excluded from headlines: references are Arabic-script
    full = ["Sarvam (external)", "faster-whisper, model unrecorded (external)",
            "faster-whisper + separation (external)", "MMS-1B-all (local)"]  # fmt: skip
    ov = df[df.system.isin(full)].groupby("system").apply(_pooled).loc[full]
    t["asr_overall"] = _md(ov.assign(clips=ov.clips.astype(int)))

    sub_ids = set(df[df.system.str.startswith("Gemini")].id)
    sub = df[df.id.isin(sub_ids)].groupby("system").apply(_pooled)
    sub = sub[sub.clips == len(sub_ids)].sort_values("WER")
    t["asr_subset"] = _md(sub.assign(clips=sub.clips.astype(int)))

    hi_ids = set(df[df.system.str.startswith("IndicWav2Vec")].id)
    hi = df[df.id.isin(hi_ids)].groupby("system").apply(_pooled)
    hi = hi[hi.clips == len(hi_ids)].sort_values("WER")
    t["asr_hindi"] = _md(hi.assign(clips=hi.clips.astype(int)))

    # per-language, full-set systems
    pl = df[df.system.isin(full)].groupby(["language", "system"]).apply(_pooled)
    wer = pl.WER.unstack()[full]
    wer.index = [LANG[i] for i in wer.index]
    wer.index.name = "Language"
    t["asr_lang_wer"] = _md(wer)
    sv = pl.xs("Sarvam (external)", level="system").copy()
    sv.index = [LANG[i] for i in sv.index]
    sv.index.name = "Language"
    other = wer.drop(columns="Sarvam (external)")
    sv["best other system"] = other.idxmin(axis=1)
    sv["its WER"] = other.min(axis=1)
    sv = sv.sort_values("WER")
    t["asr_best_lang"] = _md(
        sv[["clips", "WER", "CER", "sub", "del", "ins", "best other system", "its WER"]].assign(
            clips=sv.clips.astype(int)
        )
    )
    by_c = df[df.system == "Sarvam (external)"].groupby("corpus").apply(_pooled)
    t["asr_best_corpus"] = _md(by_c[["clips", "WER", "CER"]].assign(clips=by_c.clips.astype(int)))
    by_r = df[df.system == "Sarvam (external)"].groupby("register").apply(_pooled)
    t["asr_best_register"] = _md(by_r[["clips", "WER", "CER"]].assign(clips=by_r.clips.astype(int)))

    # chart: per-language WER for the chosen system (single series, sorted)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(sv.index, sv.WER, color=C[0], height=0.62)
    for y, (w, n) in enumerate(zip(sv.WER, sv.clips, strict=True)):
        ax.text(w + 0.006, y, f"{w:.1%}  (n={int(n)})", va="center", fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Word error rate (pooled)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, sv.WER.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("Sarvam ASR — WER by language (782 clips, Sindhi excluded)")
    _save(fig, "best_asr_wer_by_language")

    # chart: system x language heatmap (sequential single hue)
    fig, ax = plt.subplots(figsize=(8, 6))
    order = wer["Sarvam (external)"].sort_values().index
    m = wer.loc[order].clip(upper=1.5)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])
    im = ax.imshow(m.values, cmap=cmap, aspect="auto", vmin=0, vmax=1.5)
    ax.set_xticks(range(len(m.columns)), ["Sarvam", "FW (unrec.)", "FW + sep", "MMS"])
    ax.set_yticks(range(len(m.index)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = wer.loc[order].values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.7 else INK)  # fmt: skip
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="WER (clipped at 150%)", shrink=0.8)
    ax.set_title("ASR WER by language and system (full 782-clip set)")
    _save(fig, "cmp_asr_heatmap")

    # chart: overall WER on the 84-clip common subset (all systems that ran on it)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(sub.index, sub.WER, color=C[0], height=0.6)
    for y, w in enumerate(sub.WER):
        ax.text(w + 0.01, y, f"{w:.1%}", va="center", fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, sub.WER.max() * 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"ASR WER on the common stratified subset ({len(sub_ids)} clips)")
    _save(fig, "cmp_asr_subset")


def fw_diagnosis(t: dict[str, str]) -> None:
    df = load_asr()
    fw = df[(df.system == "faster-whisper, model unrecorded (external)") & (df.language != "sd")].copy()
    sv = df[(df.system == "Sarvam (external)") & (df.language != "sd")].set_index("id")
    fw["errors"] = fw.w_substitutions + fw.w_deletions + fw.w_insertions
    fw["cause"] = np.where(
        fw.hypothesis.fillna("").map(_loop),
        "repetition loop",
        np.where(fw.script_share.fillna(1.0) < 0.5, "wrong script", "same-script recognition error"),
    )
    g = fw.groupby("cause").agg(clips=("id", "size"), errors=("errors", "sum"), ins=("w_insertions", "sum"))
    g["share of all word errors"] = g.errors / g.errors.sum()
    t["fw_causes"] = _md(g.assign(clips=g.clips.astype(int), errors=g.errors.astype(int),
                                  ins=g.ins.astype(int)), "{:.1%}")  # fmt: skip
    clean = fw[fw.cause == "same-script recognition error"].id
    both = pd.DataFrame(
        {
            "faster-whisper": _pooled(fw[fw.id.isin(clean)]),
            "Sarvam": _pooled(sv.loc[clean].reset_index()),
        }
    ).T
    t["fw_clean"] = _md(both[["WER", "CER", "clips"]].assign(clips=both.clips.astype(int)))
    ws = fw[fw.cause == "wrong script"]
    t["fw_wrong_script_lang"] = ", ".join(f"{LANG[k]} {v}" for k, v in ws.language.value_counts().items())

    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    left = 0.0
    for i, (cause, row) in enumerate(g.sort_values("errors", ascending=False).iterrows()):
        share = float(row["share of all word errors"])
        ax.barh([0], [share], left=left, color=C[i], height=0.5, edgecolor=SURF, linewidth=2)
        if share >= 0.12:  # label inside; a narrow segment is labelled below the bar instead
            ax.text(left + share / 2, 0, f"{cause}\n{share:.0%}", ha="center", va="center",
                    fontsize=9, color="white")  # fmt: skip
        else:
            ax.text(left + share, -0.32, f"{cause} {share:.0%}", ha="right", va="top",
                    fontsize=9, color=INK2)  # fmt: skip
        left += share
    ax.set_ylim(-0.6, 0.35)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(False)
    ax.set_title("Where faster-whisper's word errors come from (782 clips)")
    _save(fig, "cmp_fw_error_causes")


# ------------------------------------------------------------ separation
def sep_section(t: dict[str, str]) -> None:
    sj = json.loads((DATA_OUT / "sep_bench.json").read_text(encoding="utf-8"))
    tab = pd.DataFrame(
        {
            s["label"]: {
                "full-utterance SI-SNRi (dB)": s["full_si_snri"]["median"],
                "overlap-region SI-SNRi (dB)": s["region_pit_si_snri"]["median"],
                "assigned-speaker SI-SNRi (dB)": s["region_assigned_si_snri"]["median"],
                "speaker assignment correct": s["assignment_accuracy"],
                "RTF (GPU)": s["rtf"],
                "peak GPU MB": s["peak_gpu_mb"],
            }
            for s in sj
        }
    ).T.sort_values("overlap-region SI-SNRi (dB)", ascending=False)
    tab.index.name = "Separator (median over 24 mixtures)"
    t["sep_table"] = _md(tab, "{:.2f}")
    ov = pd.DataFrame(
        {s["label"]: {k: v["median"] for k, v in s["full_by_overlap_ratio"].items()} for s in sj}
    ).T
    ov = ov.loc[tab.index]
    ov.index.name = "Separator — full SI-SNRi by overlap ratio"
    t["sep_by_overlap"] = _md(ov, "{:.2f}")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(tab))
    conds = ["full-utterance SI-SNRi (dB)", "overlap-region SI-SNRi (dB)"]
    h = 0.36
    for i, c in enumerate(conds):
        ax.barh(y + (i - 0.5) * h, tab[c], height=h - 0.04, color=C[i], label=c.replace(" (dB)", ""))
        for yy, v in zip(y, tab[c], strict=True):
            ax.text(v + 0.2, yy + (i - 0.5) * h, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, tab.index)
    ax.invert_yaxis()
    ax.set_xlabel("Median SI-SNR improvement (dB)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Separation quality (24 mixtures)")
    _save(fig, "cmp_separation")


# ----------------------------------------------------------- diarization
def diar_section(t: dict[str, str]) -> None:
    s = json.loads((DATA_OUT / "diar_bench.json").read_text(encoding="utf-8"))
    rows = []
    for sysn, ds in s["systems"].items():
        for dname, v in ds.items():
            x = v.get("original@0.0")
            if x:
                rows.append({"system": sysn, "dataset": dname, **x})
    d = pd.DataFrame(rows)
    piv = d.pivot(index="system", columns="dataset", values="der")
    piv = piv.sort_values("codeswitch")
    piv.index.name = "System (collar 0, overlap scored)"
    t["diar_der"] = _md(piv[[c for c in ["sequential", "mixtures", "codeswitch"] if c in piv]])
    cnt = d.pivot(index="system", columns="dataset", values="speaker_count_acc").loc[piv.index]
    cnt.index.name = "System"
    t["diar_count"] = _md(cnt[[c for c in ["sequential", "mixtures", "codeswitch"] if c in cnt]])
    vt = {k: v.get("codeswitch", {}).get("vad_tight@0.0", {}).get("der") for k, v in s["systems"].items()}
    vtd = pd.DataFrame(
        {"DER vs original reference": piv["codeswitch"], "DER vs VAD-tightened reference": pd.Series(vt)}
    ).loc[piv.index]
    vtd.index.name = "System (code-switch set)"
    t["diar_vadtight"] = _md(vtd)
    best = "ecapa+spectral (K estimated)"
    comp = d[d.system == best].set_index("dataset")[["missed", "false_alarm", "confusion", "jer_mean"]]
    comp.index.name = "Dataset"
    t["diar_best_components"] = _md(comp)
    t["diar_rtf"] = _md(pd.DataFrame({"RTF (CPU)": pd.Series(s["rtf"])}).sort_values("RTF (CPU)"), "{:.3f}")
    t["h1"] = f"{s['h1_reference_silence_share']:.1%}"
    t["h2_cs"] = f"{s['h2_ecapa_cos_codeswitch_halves']['median']:.3f}"
    t["h2_seq"] = f"{s['h2_ecapa_cos_sequential_diff_speakers']['median']:.3f}"

    est = [i for i in piv.index if "K oracle" not in i]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    sets = [c for c in ["sequential", "mixtures", "codeswitch"] if c in piv]
    y = np.arange(len(est))
    h = 0.8 / len(sets)
    for i, c in enumerate(sets):
        vals = piv.loc[est, c]
        ax.barh(y + (i - (len(sets) - 1) / 2) * h, vals, height=h - 0.03, color=C[i], label=c)
    ax.set_yticks(y, est)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("DER (collar 0; missing bars = system not run on that set)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_title("Diarization error rate, speaker count estimated", pad=26)
    _save(fig, "cmp_diarization")

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    left = np.zeros(len(comp))
    for i, col in enumerate(["missed", "false_alarm", "confusion"]):
        ax.barh(comp.index, comp[col], left=left, color=C[i], label=col.replace("_", " "),
                edgecolor=SURF, linewidth=2, height=0.55)  # fmt: skip
        left += comp[col].to_numpy()
    for yy, v in enumerate(left):
        ax.text(v + 0.01, yy, f"DER {v:.1%}", va="center", fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, left.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    ax.set_title("ECAPA + spectral clustering — DER components", pad=24)
    _save(fig, "best_diar_components")


# --------------------------------------------------------- preprocessing
def preproc_section(t: dict[str, str]) -> None:
    p = DATA_OUT / "preproc_bench.json"
    if not p.exists():
        t["preproc"] = "_not run_"
        t["preproc_choice"] = "_pending_"
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    rows = {
        k: {
            "WER (all)": v["overall_excl_sd"]["wer"],
            "WER Vaani (noisy)": v["by_corpus"]["vaani"]["wer"],
            "WER IndicVoices": v["by_corpus"]["indicvoices"]["wer"],
            "CER (all)": v["overall_excl_sd"]["cer"],
        }
        for k, v in s.items()
    }
    d = pd.DataFrame(rows).T
    d["ΔWER vs raw (pp)"] = (d["WER (all)"] - d.loc["raw", "WER (all)"]) * 100
    d.index.name = "Front-end"
    t["preproc"] = _md(d.drop(columns="ΔWER vs raw (pp)")) + "\n\n" + _md(d[["ΔWER vs raw (pp)"]], "{:+.2f}")
    dd = d.drop(index="raw")["ΔWER vs raw (pp)"].sort_values()
    # On 258 clips a sub-0.5 pp move is within run-to-run noise, so "best" is only claimed for a
    # real gain; otherwise the current default is kept (trim is needed downstream for chunking).
    if dd.iloc[0] <= -0.5:
        t["preproc_choice"] = f"{dd.index[0]} ({dd.iloc[0]:+.2f} pp WER vs raw audio)"
    else:
        cur = "peak + trim (current default)"
        t["preproc_choice"] = (
            f"keep peak normalisation + silence trim, no denoising ({dd[cur]:+.2f} pp WER vs raw; "
            "no front-end beats raw audio by ≥0.5 pp)"
        )
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(dd.index, dd.values, color=[C[2] if v < 0 else C[1] for v in dd.values], height=0.6)
    for yy, v in enumerate(dd.values):
        ax.text(v + (0.1 if v >= 0 else -0.1), yy, f"{v:+.2f} pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK2)  # fmt: skip
    ax.axvline(0, color=INK2, linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Change in WER vs raw audio (percentage points; negative = better)")
    lim = max(abs(dd.values).max() * 1.5, 1)
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_title("Preprocessing, judged by downstream ASR (MMS)")
    _save(fig, "cmp_preprocessing")


# ------------------------------------------------------------------- LLM
def llm_section(t: dict[str, str]) -> None:
    rows = {}
    for p in sorted(DATA_OUT.glob("llm_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("llm_ok_rate", 0) == 0:
            continue  # quota-exhausted run: nothing was measured
        rows[s["backend"]] = {
            k: s.get(k)
            for k in [
                "llm_ok_rate", "rewrite_cer", "lang_acc", "translation_rate", "doc_grounding",
                "doc_grounded_rate", "retrieval_top1", "novel_numbers_per_report",
                "guardrail_pass_rate", "latency_median_s",
            ]
        }  # fmt: skip
    d = pd.DataFrame(rows).T.astype(float)
    d.index.name = "Backend"
    t["llm"] = _md(d, "{:.3f}")
    best = d.loc[[i for i in d.index if i.startswith("gemini")]].T
    best.index.name = "Metric"
    t["llm_best"] = _md(best, "{:.3f}")


def free_asr_section(t: dict[str, str]) -> None:
    """Per-language table for the best free/local ASR (MMS), with where another system beats it."""
    df = load_asr()
    df = df[df.language != "sd"]
    pl = df.groupby(["language", "system"]).apply(_pooled)
    mms = pl.xs("MMS-1B-all (local)", level="system").copy()
    rows = []
    for lang, r in mms.iterrows():
        others = pl.xs(lang, level="language")
        # compare only on identical clip sets (IndicWav2Vec covers all Hindi; Gemini is a subset)
        free = others[others.clips == r.clips].drop(index="MMS-1B-all (local)", errors="ignore")
        free = free[~free.index.str.contains("external")]
        win = free[free.WER < r.WER]
        rows.append(
            {
                "Language": LANG[str(lang)],
                "clips": r.clips,
                "WER": r.WER,
                "CER": r.CER,
                "free system that beats it": ", ".join(f"{i} {w:.1%}" for i, w in win.WER.items()) or "none",
            }
        )
    d = pd.DataFrame(rows).set_index("Language").sort_values("WER")
    t["asr_mms_lang"] = _md(d)


def render(t: dict[str, str]) -> None:
    """Fill {{key}} placeholders in the report templates with the computed tables."""
    for tmpl in sorted((Path(__file__).parent / "templates").glob("*.md")):
        s = tmpl.read_text(encoding="utf-8")
        for k, v in t.items():
            s = s.replace("{{" + k + "}}", v)
        (OUT / tmpl.name).write_text(s, encoding="utf-8")
        print(f"rendered {OUT / tmpl.name}")


def main() -> None:
    _style()
    t: dict[str, str] = {}
    sections = (asr_section, free_asr_section, fw_diagnosis, sep_section, diar_section,
                preproc_section, llm_section)  # fmt: skip
    for fn in sections:
        try:
            fn(t)
        except FileNotFoundError as exc:  # a benchmark not yet run: leave its section empty
            t[fn.__name__] = f"_missing input: {exc}_"
    out = OUT / "data" / "tables.md"
    out.write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in t.items()), encoding="utf-8")
    print(f"wrote {out} and {len(list(CHARTS.glob('*.png')))} charts")
    render(t)


if __name__ == "__main__":
    main()
