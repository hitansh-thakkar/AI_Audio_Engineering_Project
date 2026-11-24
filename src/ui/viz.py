"""Matplotlib visualizations for the Dhvani preprocessing UI.

Each function renders one chart and saves it as a PNG to `out_path`.
All functions use the non-interactive Agg backend so they work headlessly
(no display required, safe on Windows servers).

Charts produced:
  energy_vad_chart    — RMS energy envelope with VAD speech regions shaded
  embedding_heatmap   — cosine-similarity matrix of per-segment embeddings
  speaker_gantt       — horizontal Gantt bar chart of speaker turns + overlaps
  separation_snr_chart — SI-SNRi bar chart per separated source
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── palette that matches the UI's warm paper theme ────────────────────────────
_PAPER  = "#f2eee5"
_PANEL  = "#fffdf8"
_INK    = "#172225"
_MUTED  = "#6c7776"
_CORAL  = "#ee765f"
_MINT   = "#56a88e"
_YELLOW = "#d29b3f"
_LINE   = "#d8d1c3"

_SPEAKER_COLORS = [_MINT, _CORAL, _YELLOW, "#668cc4", "#9b77b7", "#c66f91"]

_FONT = {"family": "DejaVu Sans", "size": 10}


def _fig(w: float = 9, h: float = 3) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=_PAPER)
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(_LINE)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color(_INK)
    return fig, ax


def energy_vad_chart(
    samples: np.ndarray,
    sr: int,
    vad_regions: list[tuple[float, float]],
    out_path: str | Path,
    frame_seconds: float = 0.02,
) -> None:
    """RMS energy envelope with speech (VAD) regions shaded in mint.

    Args:
        samples:      mono float32 waveform
        sr:           sample rate
        vad_regions:  list of (start_s, end_s) from energy_vad()
        out_path:     destination PNG path
        frame_seconds: RMS window size in seconds
    """
    frame_len = max(1, int(frame_seconds * sr))
    # pad so we get whole frames
    pad = (-len(samples)) % frame_len
    padded = np.pad(samples, (0, pad))
    frames = padded.reshape(-1, frame_len)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    times = np.arange(len(rms)) * frame_seconds

    fig, ax = _fig(w=10, h=2.8)
    # shade VAD regions first (behind the line)
    for start, end in vad_regions:
        ax.axvspan(start, end, alpha=0.25, color=_MINT, linewidth=0, label="_nolegend_")
    ax.plot(times, rms, color=_CORAL, linewidth=0.9, alpha=0.9)
    ax.fill_between(times, rms, alpha=0.18, color=_CORAL)

    # legend patches
    speech_patch = mpatches.Patch(color=_MINT, alpha=0.5, label="Speech (VAD)")
    energy_patch = mpatches.Patch(color=_CORAL, alpha=0.7, label="RMS energy")
    ax.legend(handles=[speech_patch, energy_patch], loc="upper right",
              fontsize=8, facecolor=_PANEL, edgecolor=_LINE, labelcolor=_MUTED)

    ax.set_xlabel("Time (s)", fontdict=_FONT)
    ax.set_ylabel("RMS", fontdict=_FONT)
    ax.set_title("Energy VAD — speech regions shaded", fontdict={**_FONT, "size": 11, "weight": "bold"})
    ax.set_xlim(0, max(times[-1], 0.1))
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    fig.tight_layout(pad=0.8)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_PAPER)
    plt.close(fig)


def embedding_heatmap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
) -> None:
    """Cosine-similarity heatmap of per-segment embeddings, sorted by speaker label.

    Args:
        embeddings: (n_segments, dim) float32 array
        labels:     (n_segments,)    int array — speaker cluster per segment
        out_path:   destination PNG path
    """
    n = len(embeddings)
    if n == 0:
        return

    # sort segments by label so same-speaker blocks cluster on the diagonal
    order = np.argsort(labels)
    emb_sorted = embeddings[order]
    labels_sorted = labels[order]

    sim = cosine_similarity(emb_sorted)
    # clip to [0,1] — negative cosine is meaningless for speaker embeddings
    sim = np.clip(sim, 0.0, 1.0)

    fig, ax = _fig(w=6, h=5)
    im = ax.imshow(sim, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, colors=_MUTED)
    cbar.set_label("Cosine similarity", fontsize=8, color=_MUTED)

    # draw speaker boundary lines
    unique, counts = np.unique(labels_sorted, return_counts=True)
    boundaries = np.cumsum(counts)[:-1] - 0.5
    for b in boundaries:
        ax.axhline(b, color=_LINE, linewidth=1.2)
        ax.axvline(b, color=_LINE, linewidth=1.2)

    # speaker tick labels at the center of each block
    centers = (np.cumsum(counts) - counts / 2 - 0.5).astype(float)
    spk_names = [f"SPK {u:02d}" for u in unique]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(spk_names, fontsize=7, color=_MUTED, rotation=30)
    ax.set_yticklabels(spk_names, fontsize=7, color=_MUTED)

    ax.set_title("Embedding similarity matrix\n(diagonal = same speaker)", fontdict={**_FONT, "size": 11, "weight": "bold"})
    fig.tight_layout(pad=0.8)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_PAPER)
    plt.close(fig)


def speaker_gantt(
    turns: list[dict],
    overlap_regions: list[dict],
    duration_s: float,
    out_path: str | Path,
) -> None:
    """Horizontal Gantt chart of speaker turns with overlap intervals highlighted.

    Args:
        turns:           list of {"speaker": str, "start_s": float, "end_s": float}
        overlap_regions: list of {"start_s": float, "end_s": float}
        duration_s:      total clip duration
        out_path:        destination PNG path
    """
    if not turns:
        return

    speakers = sorted({t["speaker"] for t in turns})
    spk_index = {spk: i for i, spk in enumerate(speakers)}
    n_spk = len(speakers)
    height = max(2.8, n_spk * 0.7 + 1.0)

    fig, ax = _fig(w=10, h=height)

    bar_h = 0.55
    for turn in turns:
        spk = turn["speaker"]
        y = spk_index[spk]
        start = turn["start_s"]
        width = max(turn["end_s"] - start, 0.01)
        color = _SPEAKER_COLORS[y % len(_SPEAKER_COLORS)]
        ax.barh(y, width, left=start, height=bar_h, color=color,
                alpha=0.82, linewidth=0, zorder=2)

    # overlap bands (drawn over the speaker bars)
    for ov in overlap_regions:
        start = ov["start_s"]
        width = max(ov["end_s"] - start, 0.01)
        ax.barh(-0.55, width, left=start, height=0.38, color=_CORAL,
                alpha=0.75, linewidth=0, zorder=3)

    # y-axis: speakers + overlap row
    ytick_pos = list(range(n_spk)) + [-0.55]
    ytick_labels = speakers + ["OVERLAP"]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    # color each speaker label
    for tick, label in zip(ax.get_yticklabels(), ytick_labels):
        if label == "OVERLAP":
            tick.set_color(_CORAL)
        else:
            idx = spk_index[label]
            tick.set_color(_SPEAKER_COLORS[idx % len(_SPEAKER_COLORS)])

    ax.set_xlim(0, duration_s)
    ax.set_ylim(-1.1, n_spk - 0.3)
    ax.set_xlabel("Time (s)", fontdict=_FONT)
    ax.set_title("Speaker timeline (Gantt)", fontdict={**_FONT, "size": 11, "weight": "bold"})
    ax.grid(axis="x", color=_LINE, linewidth=0.6, alpha=0.7, zorder=0)

    # legend
    handles = [mpatches.Patch(color=_SPEAKER_COLORS[i % len(_SPEAKER_COLORS)],
                               alpha=0.82, label=spk)
               for i, spk in enumerate(speakers)]
    if overlap_regions:
        handles.append(mpatches.Patch(color=_CORAL, alpha=0.75, label="Overlap"))
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              facecolor=_PANEL, edgecolor=_LINE, labelcolor=_MUTED)

    fig.tight_layout(pad=0.8)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_PAPER)
    plt.close(fig)


def separation_snr_chart(
    sources: list[dict],
    out_path: str | Path,
    clip_samples: np.ndarray | None = None,
    sr: int = 16000,
) -> None:
    """Waveform comparison panel for each separated source.

    Shows the amplitude envelope of every separated source side-by-side so you
    can visually compare the per-speaker waveforms. If clip_samples is provided
    the original mixture is shown at the top for reference.

    Args:
        sources:      list of {"speaker": str, "samples": np.ndarray}
        out_path:     destination PNG path
        clip_samples: original mixture samples (optional reference row)
        sr:           sample rate
    """
    n = len(sources)
    if n == 0:
        return

    rows = ([{"speaker": "MIX (original)", "samples": clip_samples}]
            if clip_samples is not None else []) + sources
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.9 * n_rows),
                             facecolor=_PAPER, sharex=False)
    if n_rows == 1:
        axes = [axes]

    buckets = 300
    for idx, (ax, row) in enumerate(zip(axes, rows)):
        samps = row["samples"]
        if samps is None or len(samps) == 0:
            ax.set_visible(False)
            continue
        samps = np.asarray(samps, dtype=np.float32)
        step = max(1, len(samps) // buckets)
        env = np.array([np.abs(samps[i: i + step]).max()
                        for i in range(0, len(samps), step)])[:buckets]
        t = np.linspace(0, len(samps) / sr, len(env))

        is_mix = row["speaker"].startswith("MIX")
        color = _MUTED if is_mix else _SPEAKER_COLORS[
            (idx - (1 if clip_samples is not None else 0)) % len(_SPEAKER_COLORS)
        ]
        ax.set_facecolor(_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(_LINE)
        ax.plot(t, env, color=color, linewidth=0.9, alpha=0.9)
        ax.fill_between(t, env, alpha=0.18, color=color)
        ax.set_ylabel(row["speaker"], fontsize=7, color=color, rotation=0,
                      ha="right", va="center", labelpad=60)
        ax.set_ylim(bottom=0)
        ax.tick_params(colors=_MUTED, labelsize=7)
        ax.margins(x=0)
        if idx < n_rows - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Time (s)", fontdict=_FONT)
    fig.suptitle("Separated sources — amplitude envelopes", fontsize=11,
                 fontweight="bold", color=_INK, y=1.01)
    fig.tight_layout(pad=0.6)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_PAPER)
    plt.close(fig)
