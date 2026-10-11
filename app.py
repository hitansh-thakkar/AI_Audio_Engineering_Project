"""Dhvani — Streamlit UI for speaker separation.

    streamlit run app.py

Upload any audio → the pipeline splits it into one clean track per speaker, separating
overlapping speech where it's detected. Presentation only; all logic lives in src/ui/service
and src/pipeline (both unit-tested).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.ui.service import ProcessResult, envelope, run_on_file, transcode_to_wav

REPO = Path(__file__).resolve().parent
LIVE_CFG = REPO / "config" / "live.yaml"
SCRATCH = Path(tempfile.gettempdir()) / "dhvani_ui"

st.set_page_config(page_title="Dhvani — Speaker Separation", page_icon="🎙", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .dhv-title { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
    .dhv-hero { padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid #222a38; margin-bottom: 1.4rem; }
    .dhv-title { font-size: 2.6rem; font-weight: 700; margin: 0; line-height: 1.05; }
    .dhv-sub { color: #9AA3B2; font-size: 1.02rem; margin-top: 0.35rem; max-width: 60ch; }
    .dhv-card { background: #161B26; border: 1px solid #222a38; border-left-width: 4px;
                border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.9rem; }
    .dhv-card h4 { font-family: 'Space Grotesk', sans-serif; margin: 0 0 0.1rem 0; font-size: 1.15rem; }
    .dhv-meta { color: #9AA3B2; font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── hero: the act of separation — one wave splitting into distinct colored voices ──────────
st.markdown(
    """
    <div class="dhv-hero">
      <div style="display:flex; align-items:center; gap:1.1rem;">
        <svg width="112" height="64" viewBox="0 0 112 64" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M2 32 Q 14 32 20 32" stroke="#6B7280" stroke-width="2.4" fill="none"/>
          <path d="M20 32 Q 34 8 48 10 Q 72 13 110 8" stroke="#4EA8DE" stroke-width="2.6" fill="none"/>
          <path d="M20 32 Q 36 32 52 32 Q 80 32 110 32" stroke="#F4A259" stroke-width="2.6" fill="none"/>
          <path d="M20 32 Q 34 56 48 54 Q 72 51 110 56" stroke="#8AC926" stroke-width="2.6" fill="none"/>
          <circle cx="20" cy="32" r="3.4" fill="#2EC4B6"/>
        </svg>
        <div>
          <p class="dhv-title">Dhvani</p>
          <p class="dhv-sub">Pull a chaotic, multi-speaker recording apart into one clean
          audio track per voice — overlapping speech separated where it happens.</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def waveform(path: str, color: str):
    env = envelope(path, buckets=360)
    df = pd.DataFrame({"t": range(len(env)), "a": env})
    chart = (
        alt.Chart(df).mark_area(opacity=0.85, color=color, interpolate="monotone")
        .encode(
            x=alt.X("t", axis=None),
            y=alt.Y("a", axis=None, scale=alt.Scale(domain=[0, 1])),
        )
        .properties(height=58)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def timeline_chart(result: ProcessResult) -> None:
    overlap_regions = getattr(result, "overlap_regions", [])
    rows = [
        {"speaker": row["speaker"], "start": row["start_s"], "end": row["end_s"]}
        for row in result.timeline
    ]
    rows += [
        {"speaker": "OVERLAP", "start": region["start_s"], "end": region["end_s"]}
        for region in overlap_regions
    ]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    domain = [speaker.name for speaker in result.speakers] + ["OVERLAP"]
    range_ = [speaker.color for speaker in result.speakers] + ["#111827"]
    chart = (
        alt.Chart(frame)
        .mark_bar(height=20, cornerRadius=3)
        .encode(
            x=alt.X("start:Q", title="Time (seconds)"),
            x2="end:Q",
            y=alt.Y("speaker:N", title=None, sort=domain),
            color=alt.Color("speaker:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
            tooltip=["speaker:N", "start:Q", "end:Q"],
        )
        .properties(height=max(100, 42 * len(domain)))
    )
    st.altair_chart(chart, use_container_width=True)


def show_results(result: ProcessResult, source_audio: str) -> None:
    overlap_regions = getattr(result, "overlap_regions", [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Speakers", result.num_speakers)
    c2.metric("Overlap regions", result.num_overlap_regions)
    c3.metric("Separated sources", result.num_separated_sources)
    c4.metric("Duration", f"{result.duration:.1f}s")

    overlap_warning = next((n for n in result.notes if "overlap detection" in n and
                            ("unavailable" in n or "disabled" in n or "failed" in n)), None)
    if result.num_overlap_regions:
        st.success(f"Detected and separated overlap in {result.num_overlap_regions} region(s).")
    elif overlap_warning:
        st.warning(
            "Speakers were split, but automatic overlap detection failed for this file. "
            "The detailed reason is available under Run notes."
        )
    else:
        st.info("No overlapping speech detected — speakers were already sequential.")

    st.markdown("##### Original")
    st.audio(source_audio)

    st.markdown("##### Speaker timeline")
    timeline_chart(result)
    timeline_payload = {
        "clip_id": result.clip_id,
        "duration_s": result.duration,
        "turns": result.timeline,
        "overlap_regions": overlap_regions,
        "num_speakers": result.num_speakers,
    }
    st.download_button(
        "Download timeline JSON",
        json.dumps(timeline_payload, ensure_ascii=False, indent=2),
        file_name=f"{result.clip_id}_timeline.json",
        mime="application/json",
        key="download_timeline",
    )

    st.markdown("##### Separated voices")
    for spk in result.speakers:
        st.markdown(
            f'<div class="dhv-card" style="border-left-color:{spk.color};">'
            f'<h4 style="color:{spk.color};">{spk.name}</h4>'
            f'<span class="dhv-meta">{spk.speaking_seconds:.1f}s of speech</span></div>',
            unsafe_allow_html=True)
        waveform(spk.path, spk.color)
        st.audio(spk.path)
        with open(spk.path, "rb") as fh:
            st.download_button(f"Download {spk.name}.wav", fh.read(),
                               file_name=f"{result.clip_id}_{spk.name}.wav",
                               mime="audio/wav", key=f"dl_{spk.name}")

    if result.timeline:
        with st.expander("Timeline (who spoke when)"):
            st.dataframe(pd.DataFrame(result.timeline), use_container_width=True, hide_index=True)
    if result.notes:
        with st.expander("Run notes"):
            for n in result.notes:
                st.markdown(f"- {n}")


upload = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "flac", "ogg", "m4a", "mpeg"],
    help="Dhvani automatically detects speakers and overlapping speech.",
)
n_spk = st.number_input(
    "Number of speakers (0 = detect automatically)", min_value=0, max_value=10, value=0, step=1,
    help="If you know how many people are talking, set it — it's the biggest accuracy boost.")
if upload and st.button("Analyze audio", type="primary"):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    raw = SCRATCH / upload.name
    raw.write_bytes(upload.getbuffer())
    stem = "".join(c if c.isalnum() else "_" for c in Path(upload.name).stem) or "upload"
    with st.spinner("Detecting speakers and overlap, then separating where needed…"):
        wav = transcode_to_wav(raw, SCRATCH / "in" / f"{stem}.wav")
        result = run_on_file(wav, LIVE_CFG, SCRATCH / "out", num_speakers=int(n_spk) or None)
    st.session_state["result"] = (result, str(wav))

if "result" in st.session_state:
    st.divider()
    res, src = st.session_state["result"]
    show_results(res, src)
