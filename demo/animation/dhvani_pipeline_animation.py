"""
Dhvani Pipeline Animation — 3Blue1Brown-style Manim video
==========================================================
Install (once):
    pip install -r demo/animation/requirements.txt      # needs ffmpeg on PATH

Render (from the repo root):
    manim -pqh demo/animation/dhvani_pipeline_animation.py DhvaniPipeline   # 1080p60
    manim -pql demo/animation/dhvani_pipeline_animation.py DhvaniPipeline   # quick 480p preview

Produces a ~1-minute-30 animation walking through every stage of the
"Who Spoke What and When" speech pipeline. All metrics shown on screen are the
ones in report/project_report.tex (from outputs/reports/*.md); keep them in sync.
"""

from __future__ import annotations

import numpy as np
from manim import *  # noqa: F403 — Manim's documented usage; scenes use dozens of its names

# ── Colour palette (3B1B-ish deep blues, teals, golds) ───────────
BG = "#1a1a2e"
BLUE_DARK = "#16213e"
BLUE_MID = "#0f3460"
TEAL = "#00b4d8"
GOLD = "#e2b714"
PINK = "#e94560"
GREEN = "#06d6a0"
PURPLE = "#9b5de5"
ORANGE = "#ff6700"
WHITE_T = "#e0e0e0"

# Devanagari needs a font that has the glyphs; Pango's default renders them as boxes on Windows.
# Nirmala UI ships with Windows 10/11; on Linux/macOS use e.g. "Noto Sans Devanagari".
INDIC_FONT = "Nirmala UI"

# Speaker palette
SPK_COLORS = [TEAL, GOLD, PINK, GREEN, PURPLE, ORANGE]


# ════════════════════════════════════════════════════════════════════
#  Helper: draw a waveform (synthetic sine-composite)
# ════════════════════════════════════════════════════════════════════
def make_waveform(
    width: float = 10,
    height: float = 1.5,
    n_points: int = 800,
    seed: int = 42,
    color: str = TEAL,
    noisy: bool = False,
) -> VMobject:
    """Return a VMobject containing the waveform path."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_points)

    # Base signal — superposition of sines mimicking speech envelope
    y = (
        0.6 * np.sin(2.3 * t)
        + 0.35 * np.sin(5.7 * t)
        + 0.2 * np.sin(11.3 * t + 1)
        + 0.15 * np.sin(23 * t + 0.5)
    )
    # Speech-like amplitude envelope (bursts and pauses)
    envelope = np.clip(
        0.5 + 0.6 * np.sin(0.4 * t) + 0.3 * np.sin(1.1 * t + 2),
        0.05,
        1,
    )
    y *= envelope

    if noisy:
        y += rng.normal(0, 0.15, n_points)

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9) * height / 2

    xs = np.linspace(-width / 2, width / 2, n_points)
    points = [np.array([x, yi, 0]) for x, yi in zip(xs, y)]
    wave = VMobject(stroke_color=color, stroke_width=2.0)
    wave.set_points_smoothly(points)
    return wave


def make_bars(
    n_bars: int = 50,
    width: float = 10,
    max_h: float = 1.2,
    seed: int = 7,
    color: str = TEAL,
) -> VGroup:
    rng = np.random.default_rng(seed)
    bar_w = width / n_bars * 0.75
    bars = VGroup()
    for i in range(n_bars):
        h = rng.uniform(0.1, 1.0) * max_h
        envelope = 0.5 + 0.5 * np.sin(i / n_bars * np.pi)
        h *= envelope
        bar = Rectangle(
            width=bar_w,
            height=h,
            fill_color=color,
            fill_opacity=0.8,
            stroke_width=0,
        )
        bar.move_to(
            np.array(
                [
                    -width / 2 + (i + 0.5) * width / n_bars,
                    h / 2,
                    0,
                ]
            )
        )
        bars.add(bar)
    return bars


# ════════════════════════════════════════════════════════════════════
#  Helper: speaker segment timeline
# ════════════════════════════════════════════════════════════════════
def make_timeline(
    segments: list[tuple[float, float, int]],
    total_dur: float = 10.0,
    width: float = 10,
    row_h: float = 0.35,
    n_speakers: int = 3,
) -> VGroup:
    """segments: [(start, end, speaker_idx), ...]"""
    group = VGroup()
    # Background track
    bg = Rectangle(
        width=width,
        height=row_h * n_speakers + 0.3,
        fill_color=BLUE_DARK,
        fill_opacity=0.6,
        stroke_color=WHITE_T,
        stroke_width=0.5,
    )
    group.add(bg)

    for start, end, spk in segments:
        x0 = -width / 2 + (start / total_dur) * width
        x1 = -width / 2 + (end / total_dur) * width
        seg_w = x1 - x0
        y = (n_speakers / 2 - 0.5 - spk) * row_h
        rect = Rectangle(
            width=seg_w,
            height=row_h * 0.8,
            fill_color=SPK_COLORS[spk % len(SPK_COLORS)],
            fill_opacity=0.85,
            stroke_width=0,
        )
        rect.move_to(np.array([(x0 + x1) / 2, y, 0]))
        group.add(rect)
    return group


# ════════════════════════════════════════════════════════════════════
#  Helper: Spectrogram-style grid
# ════════════════════════════════════════════════════════════════════
def make_spectrogram(
    rows: int = 20,
    cols: int = 50,
    width: float = 8,
    height: float = 2.5,
    seed: int = 99,
) -> VGroup:
    rng = np.random.default_rng(seed)
    cell_w = width / cols
    cell_h = height / rows
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            # Energy falls off at high frequencies
            energy = rng.uniform(0, 1) * max(0, 1 - r / rows + 0.2)
            energy *= 0.5 + 0.5 * np.sin(c / cols * np.pi)  # speech envelope
            cell = Rectangle(
                width=cell_w * 0.95,
                height=cell_h * 0.95,
                fill_color=interpolate_color(ManimColor(BLUE_DARK), ManimColor(GOLD), min(energy, 1.0)),
                fill_opacity=0.9,
                stroke_width=0,
            )
            cell.move_to(
                np.array(
                    [
                        -width / 2 + (c + 0.5) * cell_w,
                        -height / 2 + (r + 0.5) * cell_h,
                        0,
                    ]
                )
            )
            group.add(cell)
    return group


# ════════════════════════════════════════════════════════════════════
#  Helper: Neural-net / node graph
# ════════════════════════════════════════════════════════════════════
def make_nn_layer(
    n_nodes: int,
    x: float,
    spread: float = 2.5,
    radius: float = 0.15,
    color: str = TEAL,
) -> list[Circle]:
    nodes = []
    for i in range(n_nodes):
        y = (i - (n_nodes - 1) / 2) * (spread / max(n_nodes - 1, 1))
        node = Circle(
            radius=radius,
            fill_color=color,
            fill_opacity=0.8,
            stroke_color=WHITE_T,
            stroke_width=1,
        )
        node.move_to(np.array([x, y, 0]))
        nodes.append(node)
    return nodes


def make_nn(
    layers: list[int],
    x_start: float = -3,
    x_end: float = 3,
    colors: list[str] | None = None,
) -> VGroup:
    if colors is None:
        colors = [TEAL, BLUE_MID, GOLD, GREEN, PINK]
    group = VGroup()
    all_layers: list[list[Circle]] = []
    xs = np.linspace(x_start, x_end, len(layers))
    for i, (n, x) in enumerate(zip(layers, xs)):
        col = colors[i % len(colors)]
        nodes = make_nn_layer(n, x, color=col)
        all_layers.append(nodes)
        for nd in nodes:
            group.add(nd)
    # Edges
    for l_idx in range(len(all_layers) - 1):
        for n1 in all_layers[l_idx]:
            for n2 in all_layers[l_idx + 1]:
                edge = Line(
                    n1.get_center(),
                    n2.get_center(),
                    stroke_color=WHITE_T,
                    stroke_opacity=0.15,
                    stroke_width=1,
                )
                group.add(edge)
    return group


# ════════════════════════════════════════════════════════════════════
#  MAIN SCENE
# ════════════════════════════════════════════════════════════════════
class DhvaniPipeline(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG

        self.play_title()  # Title
        self.play_raw_audio()  # Stage 1 — raw audio + waveform
        self.play_preprocessing()  # Stage 2 — preprocessing
        self.play_vad_diarization()  # Stage 3 — VAD + diarization
        self.play_overlap_separation()  # Stage 4 — overlap detection + clustering + separation
        self.play_asr()  # Stage 5 — ASR
        self.play_llm_postprocessing()  # Stage 6 — LLM post-processing
        self.play_pipeline_overview()  # Full pipeline overview
        self.play_end_card()  # End card

    # ── SCENE 1: Title ────────────────────────────────────────────
    def play_title(self) -> None:
        title = Text(
            "Dhvani",
            font_size=80,
            color=TEAL,
            weight=BOLD,
        )
        subtitle = Text(
            '"Who Spoke What and When"',
            font_size=36,
            color=GOLD,
        ).next_to(title, DOWN, buff=0.4)
        tagline = Text(
            "A Multi-Speaker Speech Pipeline for Indian Audio",
            font_size=22,
            color=WHITE_T,
        ).next_to(subtitle, DOWN, buff=0.5)

        # Decorative waveform behind title
        bg_wave = make_waveform(width=14, height=3, color=BLUE_MID, seed=10)
        bg_wave.set_opacity(0.2)

        self.play(
            Create(bg_wave, run_time=2),
            Write(title, run_time=1.5),
        )
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=0.8)
        self.play(FadeIn(tagline, shift=UP * 0.2), run_time=0.6)
        self.wait(1.5)
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(tagline),
            FadeOut(bg_wave),
            run_time=0.8,
        )

    # ── SCENE 2: Raw Audio + Waveform ─────────────────────────────
    def play_raw_audio(self) -> None:
        header = Text("Stage 1 — Raw Audio Input", font_size=38, color=TEAL)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Microphone icon (stylized)
        mic_body = RoundedRectangle(
            width=0.6,
            height=1.2,
            corner_radius=0.3,
            fill_color=GOLD,
            fill_opacity=0.9,
            stroke_width=0,
        )
        mic_stand = Line(
            mic_body.get_bottom(),
            mic_body.get_bottom() + DOWN * 0.5,
            stroke_color=GOLD,
            stroke_width=4,
        )
        mic_base = Line(
            mic_stand.get_end() + LEFT * 0.4,
            mic_stand.get_end() + RIGHT * 0.4,
            stroke_color=GOLD,
            stroke_width=4,
        )
        mic = VGroup(mic_body, mic_stand, mic_base).move_to(LEFT * 4.5 + UP * 0.3)

        self.play(FadeIn(mic, scale=0.5), run_time=0.6)

        # Noisy waveform
        noisy_wave = make_waveform(width=7, height=2, seed=42, color=PINK, noisy=True)
        noisy_wave.move_to(RIGHT * 0.5)
        label = Text(
            "Multi-speaker, noisy, code-switched audio",
            font_size=18,
            color=WHITE_T,
        ).next_to(noisy_wave, DOWN, buff=0.4)

        # Sound-wave arcs emanating from mic
        arcs = VGroup()
        for i in range(3):
            arc = Arc(
                radius=0.5 + i * 0.35,
                start_angle=PI / 3,
                angle=-2 * PI / 3,
                stroke_color=GOLD,
                stroke_width=2 - i * 0.5,
                stroke_opacity=0.7 - i * 0.2,
            ).next_to(mic, RIGHT, buff=0.1 + i * 0.15)
            arcs.add(arc)

        self.play(
            LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.3),
            run_time=1,
        )
        self.play(Create(noisy_wave, run_time=2.5))
        self.play(FadeIn(label), run_time=0.5)

        # Highlight noise
        noise_box = SurroundingRectangle(noisy_wave, color=PINK, buff=0.2, stroke_width=1.5)
        noise_label = Text("Background noise + overlapping speakers", font_size=16, color=PINK)
        noise_label.next_to(noise_box, UP, buff=0.2)
        self.play(Create(noise_box), FadeIn(noise_label), run_time=0.8)
        self.wait(1)

        self.play(
            *[FadeOut(m) for m in [header, mic, arcs, noisy_wave, label, noise_box, noise_label]],
            run_time=0.6,
        )

    # ── SCENE 3: Preprocessing ────────────────────────────────────
    def play_preprocessing(self) -> None:
        header = Text("Stage 2 — Preprocessing", font_size=38, color=TEAL)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Noisy → Clean waveform transformation
        noisy = make_waveform(width=4.5, height=1.8, seed=42, color=PINK, noisy=True)
        noisy.move_to(LEFT * 3.5)

        clean = make_waveform(width=4.5, height=1.5, seed=42, color=GREEN)
        clean.move_to(RIGHT * 3.5)

        arrow = Arrow(
            LEFT * 0.8,
            RIGHT * 0.8,
            stroke_color=GOLD,
            buff=0.1,
        )

        # Processing steps
        steps = (
            VGroup(
                Text("16 kHz Mono", font_size=16, color=WHITE_T),
                Text("Peak Normalize", font_size=16, color=WHITE_T),
                Text("Trim Silence", font_size=16, color=WHITE_T),
            )
            .arrange(DOWN, buff=0.15)
            .next_to(arrow, DOWN, buff=0.3)
        )

        label_before = Text("Raw Audio", font_size=18, color=PINK)
        label_before.next_to(noisy, DOWN, buff=0.3)
        label_after = Text("Clean 16 kHz", font_size=18, color=GREEN)
        label_after.next_to(clean, DOWN, buff=0.3)

        # Gear icon drawn from primitives (an emoji glyph renders as a box on fonts without it)
        gear_teeth = VGroup(
            *[
                Rectangle(
                    width=0.14,
                    height=0.2,
                    fill_color=GOLD,
                    fill_opacity=0.9,
                    stroke_width=0,
                )
                .shift(UP * 0.33)
                .rotate(k * TAU / 8, about_point=ORIGIN)
                for k in range(8)
            ]
        )
        gear = Circle(
            radius=0.28,
            fill_color=GOLD,
            fill_opacity=0.3,
            stroke_color=GOLD,
            stroke_width=2,
        )
        hub = Circle(radius=0.08, fill_color=GOLD, fill_opacity=0.9, stroke_width=0)
        gear_group = VGroup(gear_teeth, gear, hub).next_to(arrow, UP, buff=0.2)

        self.play(Create(noisy, run_time=1.5))
        self.play(FadeIn(label_before), run_time=0.4)
        self.play(
            GrowArrow(arrow),
            FadeIn(gear_group),
            run_time=0.6,
        )
        self.play(
            LaggedStart(*[FadeIn(s, shift=DOWN * 0.2) for s in steps], lag_ratio=0.2),
            run_time=0.8,
        )

        # Transform waveform
        self.play(
            Rotating(gear_group, angle=2 * PI, run_time=1.5),
            TransformFromCopy(noisy, clean, run_time=1.5),
        )
        self.play(FadeIn(label_after), run_time=0.4)
        self.wait(0.8)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.6,
        )

    # ── SCENE 4: VAD + Diarization ────────────────────────────────
    def play_vad_diarization(self) -> None:
        header = Text("Stage 3 — Voice Activity Detection & Diarization", font_size=34, color=TEAL)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Clean waveform
        wave = make_waveform(width=11, height=1.5, seed=42, color=GREEN)
        wave.move_to(UP * 1.5)
        self.play(Create(wave, run_time=1.5))

        # VAD regions (highlighted boxes)
        vad_label = Text("VAD — Voice Activity Detection", font_size=20, color=GOLD)
        vad_label.next_to(wave, DOWN, buff=0.3)
        self.play(FadeIn(vad_label), run_time=0.5)

        # Show speech regions lighting up
        vad_regions = VGroup()
        region_specs = [
            (-4.5, -3.0),
            (-2.5, -0.8),
            (-0.3, 1.5),
            (1.8, 3.2),
            (3.5, 5.0),
        ]
        for x0, x1 in region_specs:
            rect = Rectangle(
                width=x1 - x0,
                height=1.8,
                fill_color=GOLD,
                fill_opacity=0.15,
                stroke_color=GOLD,
                stroke_width=1,
            )
            rect.move_to(np.array([(x0 + x1) / 2, 1.5, 0]))
            vad_regions.add(rect)

        self.play(
            LaggedStart(
                *[FadeIn(r, run_time=0.4) for r in vad_regions],
                lag_ratio=0.15,
            ),
        )
        self.wait(0.5)

        # Transition to speaker-colored timeline
        diar_label = Text(
            'Speaker Diarization — "Who spoke when?"',
            font_size=22,
            color=TEAL,
        )
        diar_label.move_to(DOWN * 0.5)
        self.play(
            FadeOut(vad_label),
            FadeIn(diar_label),
            run_time=0.5,
        )

        # Speaker timeline
        segments = [
            (0.0, 2.0, 0),
            (2.0, 4.5, 1),
            (4.3, 6.0, 0),
            (5.8, 7.5, 2),
            (7.5, 8.5, 1),
            (8.5, 10.0, 0),
        ]
        timeline = make_timeline(segments, total_dur=10, width=11, n_speakers=3)
        timeline.move_to(DOWN * 2)

        # Speaker labels (left of the 11-unit-wide track, inside the 14.2-unit frame)
        spk_labels = VGroup()
        for i in range(3):
            lbl = Text(
                f"Speaker {i + 1}",
                font_size=14,
                color=SPK_COLORS[i],
            )
            y = (3 / 2 - 0.5 - i) * 0.35
            lbl.move_to(np.array([0, y - 2, 0]))
            lbl.next_to(timeline, LEFT, buff=0.15).set_y(y - 2)
            spk_labels.add(lbl)

        self.play(
            FadeIn(timeline, shift=DOWN * 0.3),
            LaggedStart(
                *[FadeIn(lb) for lb in spk_labels],
                lag_ratio=0.2,
            ),
            run_time=1.2,
        )

        # Animated scanning line
        scan_line = Line(
            UP * 0.5 + LEFT * 5.5,
            DOWN * 3.2 + LEFT * 5.5,
            stroke_color=WHITE_T,
            stroke_width=1.5,
            stroke_opacity=0.8,
        )
        self.add(scan_line)
        self.play(
            scan_line.animate.shift(RIGHT * 11),
            run_time=2.5,
            rate_func=linear,
        )
        self.wait(0.5)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.6,
        )

    # ── SCENE 5: Overlap + Spectral Clustering ────────────────────
    def play_overlap_separation(self) -> None:
        header = Text(
            "Stage 4 — Overlap Detection & Spectral Clustering",
            font_size=32,
            color=TEAL,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Timeline with overlap
        segments = [
            (0.0, 3.5, 0),
            (2.5, 5.5, 1),  # overlaps with spk 0
            (5.5, 8.0, 0),
            (7.0, 10.0, 2),  # overlaps with spk 0
        ]
        timeline = make_timeline(segments, total_dur=10, width=11, n_speakers=3)
        timeline.move_to(UP * 1.8)

        self.play(FadeIn(timeline), run_time=0.8)

        # Highlight overlap zones: x = -5.5 + t * 1.1 for the 11-wide, 10-s track
        overlap1 = Rectangle(
            width=1.1,
            height=1.5,
            fill_color=PINK,
            fill_opacity=0.3,
            stroke_color=PINK,
            stroke_width=2,
        )
        overlap1.move_to(np.array([-2.2, 1.8, 0]))  # 2.5-3.5 s
        ov1_label = Text("OVERLAP", font_size=12, color=PINK, weight=BOLD)
        ov1_label.next_to(overlap1, UP, buff=0.1)

        overlap2 = Rectangle(
            width=1.1,
            height=1.5,
            fill_color=PINK,
            fill_opacity=0.3,
            stroke_color=PINK,
            stroke_width=2,
        )
        overlap2.move_to(np.array([2.75, 1.8, 0]))  # 7.0-8.0 s
        ov2_label = Text("OVERLAP", font_size=12, color=PINK, weight=BOLD)
        ov2_label.next_to(overlap2, UP, buff=0.1)

        self.play(
            FadeIn(overlap1),
            FadeIn(ov1_label),
            FadeIn(overlap2),
            FadeIn(ov2_label),
            run_time=0.8,
        )
        self.wait(0.5)

        # Spectral Clustering visualization
        sc_title = Text("Spectral Clustering", font_size=24, color=GOLD)
        sc_title.move_to(DOWN * 0.3)
        self.play(Write(sc_title), run_time=0.6)

        # Scatter plot of embeddings
        rng = np.random.default_rng(42)
        dots = VGroup()
        for centre, n, spread, spk in [
            ((-3, -2.2), 15, 0.4, 0),
            ((0, -2.2), 15, 0.4, 1),
            ((3, -2.2), 12, 0.35, 2),
        ]:
            for _ in range(n):
                pos = rng.normal(0, spread, 2) + np.array(centre)
                dots.add(Dot(np.array([pos[0], pos[1], 0]), radius=0.06, color=SPK_COLORS[spk]))

        emb_label = Text("Speaker Embedding Space", font_size=16, color=WHITE_T)
        emb_label.move_to(DOWN * 0.9)

        # Initially all dots are white (unclustered)
        unclustered = dots.copy()
        for d in unclustered:
            d.set_color(WHITE_T)

        self.play(
            FadeIn(emb_label),
            LaggedStart(*[FadeIn(d, scale=0.3) for d in unclustered], lag_ratio=0.03),
            run_time=1.2,
        )
        self.wait(0.3)

        # Cluster them (transform white → colored)
        cluster_label = Text("Clustering speakers...", font_size=16, color=GOLD)
        cluster_label.next_to(emb_label, DOWN, buff=0.3)
        self.play(FadeIn(cluster_label), run_time=0.4)

        self.play(
            Transform(unclustered, dots),
            run_time=1.5,
        )

        # Draw cluster circles
        circles = VGroup()
        for cx, cy, col in [(-3, -2.2, SPK_COLORS[0]), (0, -2.2, SPK_COLORS[1]), (3, -2.2, SPK_COLORS[2])]:
            circ = Circle(
                radius=0.9,
                stroke_color=col,
                stroke_width=2,
                fill_opacity=0,
            ).move_to(np.array([cx, cy, 0]))
            circles.add(circ)

        self.play(
            LaggedStart(*[Create(c) for c in circles], lag_ratio=0.2),
            run_time=0.8,
        )
        self.wait(0.8)

        # SepFormer separation
        sep_title = Text(
            "SepFormer — Separating overlapped speech",
            font_size=22,
            color=GREEN,
        )
        sep_title.move_to(DOWN * 0.3)

        self.play(
            FadeOut(sc_title),
            FadeOut(emb_label),
            FadeOut(cluster_label),
            FadeOut(unclustered),
            FadeOut(circles),
            FadeIn(sep_title),
            run_time=0.6,
        )

        # Overlapped waveform → two separated waveforms
        mixed = make_waveform(width=4, height=1.2, seed=100, color=PINK)
        mixed.move_to(LEFT * 3.5 + DOWN * 2)
        mix_label = Text("Mixed Audio", font_size=14, color=PINK)
        mix_label.next_to(mixed, DOWN, buff=0.2)

        sep1 = make_waveform(width=4, height=0.8, seed=42, color=SPK_COLORS[0])
        sep1.move_to(RIGHT * 3 + DOWN * 1.5)
        sep1_label = Text("Speaker 1", font_size=14, color=SPK_COLORS[0])
        sep1_label.next_to(sep1, DOWN, buff=0.15)

        sep2 = make_waveform(width=4, height=0.8, seed=77, color=SPK_COLORS[1])
        sep2.move_to(RIGHT * 3 + DOWN * 2.8)
        sep2_label = Text("Speaker 2", font_size=14, color=SPK_COLORS[1])
        sep2_label.next_to(sep2, DOWN, buff=0.15)

        sep_arrow = Arrow(
            LEFT * 1.2 + DOWN * 2,
            RIGHT * 0.8 + DOWN * 2,
            stroke_color=GREEN,
        )

        self.play(Create(mixed), FadeIn(mix_label), run_time=0.8)
        self.play(GrowArrow(sep_arrow), run_time=0.5)
        self.play(
            Create(sep1),
            FadeIn(sep1_label),
            Create(sep2),
            FadeIn(sep2_label),
            run_time=1.2,
        )

        # Separation metric (SepFormer-Libri2Mix, overlap-only; report Table / Results)
        metric = Text(
            "SI-SNRi on overlaps: +15.8 dB  |  correct speaker assignment: 75%",
            font_size=16,
            color=GOLD,
        )
        metric.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(metric, shift=UP * 0.2), run_time=0.5)
        self.wait(1)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.6,
        )

    # ── SCENE 6: ASR (Speech to Text) ─────────────────────────────
    def play_asr(self) -> None:
        header = Text("Stage 5 — Automatic Speech Recognition", font_size=34, color=TEAL)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Per-speaker waveforms on the left
        waves = VGroup()
        wave_labels = VGroup()
        for i in range(3):
            w = make_waveform(width=3.5, height=0.7, seed=42 + i * 20, color=SPK_COLORS[i])
            w.move_to(LEFT * 4 + UP * (1.5 - i * 1.4))
            lbl = Text(f"Speaker {i + 1}", font_size=14, color=SPK_COLORS[i])
            lbl.next_to(w, LEFT, buff=0.3)
            waves.add(w)
            wave_labels.add(lbl)

        self.play(
            LaggedStart(*[Create(w) for w in waves], lag_ratio=0.2),
            LaggedStart(*[FadeIn(lb) for lb in wave_labels], lag_ratio=0.2),
            run_time=1.5,
        )

        # ASR model box — the benchmarked systems (report Table 1)
        asr_box = RoundedRectangle(
            width=2.5,
            height=2.8,
            corner_radius=0.2,
            fill_color=BLUE_MID,
            fill_opacity=0.5,
            stroke_color=TEAL,
            stroke_width=2,
        ).move_to(ORIGIN)
        asr_text = Text("Indic ASR\nMMS-1B-all", font_size=18, color=WHITE_T)
        asr_text.move_to(asr_box).shift(UP * 0.3)
        asr_sub = Text("Sarvam · faster-whisper\nIndicWav2Vec", font_size=12, color=GOLD)
        asr_sub.next_to(asr_text, DOWN, buff=0.3)

        self.play(
            FadeIn(asr_box, scale=0.8),
            Write(asr_text),
            FadeIn(asr_sub),
            run_time=0.8,
        )

        # Arrows from waves to ASR
        arrows_in = VGroup()
        for w in waves:
            arr = Arrow(
                w.get_right() + RIGHT * 0.1,
                asr_box.get_left() + LEFT * 0.05,
                stroke_color=WHITE_T,
                stroke_width=1.5,
                buff=0.1,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_in.add(arr)
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_in], lag_ratio=0.15),
            run_time=0.8,
        )

        # Text output on the right (typewriter effect)
        transcript_lines = [
            ('"नमस्ते, आज हम बात करेंगे..."', SPK_COLORS[0]),
            ('"Yes, the community radio..."', SPK_COLORS[1]),
            ('"किसानों के लिए यह scheme..."', SPK_COLORS[2]),
        ]
        texts = VGroup()
        arrows_out = VGroup()
        for i, (line, col) in enumerate(transcript_lines):
            txt = Text(line, font_size=15, color=col, font=INDIC_FONT)
            txt.move_to(RIGHT * 4.3 + UP * (1.5 - i * 1.4))
            texts.add(txt)

            arr = Arrow(
                asr_box.get_right() + RIGHT * 0.05,
                txt.get_left() + LEFT * 0.1,
                stroke_color=WHITE_T,
                stroke_width=1.5,
                buff=0.1,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_out.add(arr)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_out], lag_ratio=0.15),
            run_time=0.6,
        )

        # Typewriter effect for each transcript
        for txt in texts:
            self.play(Write(txt, run_time=1))

        ts_note = Text(
            "Transcribed per speaker turn  •  best WER: Sarvam 24.9%, MMS 56.6% (local)",
            font_size=14,
            color=GOLD,
        )
        ts_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(ts_note), run_time=0.5)
        self.wait(0.8)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.6,
        )

    # ── SCENE 7: LLM Post-Processing ─────────────────────────────
    def play_llm_postprocessing(self) -> None:
        header = Text("Stage 6 — LLM Post-Processing", font_size=38, color=TEAL)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Raw transcript on left
        raw_title = Text("Raw Transcript", font_size=18, color=PINK)
        raw_title.move_to(LEFT * 4.5 + UP * 2)

        raw_lines = VGroup(
            Text("um... so the किसान uh...", font_size=13, color=WHITE_T, font=INDIC_FONT),
            Text("scheme, the PM-KISAN is...", font_size=13, color=WHITE_T),
            Text("haan haan... [inaudible]...", font_size=13, color=WHITE_T),
            Text("very im- important for...", font_size=13, color=WHITE_T),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        raw_lines.next_to(raw_title, DOWN, buff=0.3)

        raw_box = SurroundingRectangle(
            VGroup(raw_title, raw_lines),
            color=PINK,
            buff=0.3,
            stroke_width=1.5,
            corner_radius=0.1,
        )

        self.play(
            FadeIn(raw_box),
            Write(raw_title),
            LaggedStart(*[FadeIn(lb) for lb in raw_lines], lag_ratio=0.15),
            run_time=1.2,
        )

        # Gemini model in center (the model behind the report's LLM numbers)
        gemini_box = RoundedRectangle(
            width=2.8,
            height=1.8,
            corner_radius=0.2,
            fill_color=BLUE_MID,
            fill_opacity=0.5,
            stroke_color=GOLD,
            stroke_width=2,
        )
        gemini_text = Text("Google Gemini\n3.5 Flash-Lite", font_size=16, color=GOLD)
        gemini_text.move_to(gemini_box)
        gemini_group = VGroup(gemini_box, gemini_text)

        arr_in = Arrow(
            raw_box.get_right(),
            gemini_box.get_left(),
            stroke_color=WHITE_T,
            buff=0.15,
        )

        self.play(
            GrowArrow(arr_in),
            FadeIn(gemini_group, scale=0.8),
            run_time=0.8,
        )

        # Processing effect
        glow = gemini_box.copy()
        glow.set_stroke(GOLD, width=4, opacity=0.6)
        self.play(
            ShowPassingFlash(glow, time_width=0.4, run_time=1.5),
        )

        # Clean output on right (plain bullets: emoji glyphs render as boxes without an emoji font)
        clean_title = Text("Clean Report", font_size=18, color=GREEN)
        clean_title.move_to(RIGHT * 4.5 + UP * 2)

        output_items = VGroup(
            Text("• Executive Summary", font_size=14, color=GREEN),
            Text("• Keywords: agriculture,", font_size=13, color=WHITE_T),
            Text("    PM-KISAN, rural dev", font_size=13, color=WHITE_T),
            Text("• Speaker-attributed", font_size=14, color=TEAL),
            Text("    dialogue transcript", font_size=13, color=WHITE_T),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        # Pango strips leading spaces, so indent the continuation lines (2 and 4) explicitly
        for idx in (2, 4):
            output_items[idx].shift(RIGHT * 0.2)
        output_items.next_to(clean_title, DOWN, buff=0.3)

        clean_box = SurroundingRectangle(
            VGroup(clean_title, output_items),
            color=GREEN,
            buff=0.3,
            stroke_width=1.5,
            corner_radius=0.1,
        )

        arr_out = Arrow(
            gemini_box.get_right(),
            clean_box.get_left(),
            stroke_color=WHITE_T,
            buff=0.15,
        )

        self.play(
            GrowArrow(arr_out),
            FadeIn(clean_box),
            Write(clean_title),
            run_time=0.6,
        )
        self.play(
            LaggedStart(*[FadeIn(item, shift=RIGHT * 0.2) for item in output_items], lag_ratio=0.15),
            run_time=1,
        )

        guard = Text(
            "Grounded prompt: no facts absent from the transcript  •  0 invented numbers",
            font_size=14,
            color=GOLD,
        )
        guard.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(guard), run_time=0.5)
        self.wait(1)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.6,
        )

    # ── SCENE 8: Full Pipeline Overview ───────────────────────────
    def play_pipeline_overview(self) -> None:
        header = Text("The Complete Dhvani Pipeline", font_size=38, color=TEAL)
        header.to_edge(UP, buff=0.4)
        self.play(Write(header), run_time=0.8)

        # Pipeline boxes
        stage_data = [
            ("1. Preprocessing", "Resample\n16kHz Mono\nNormalize", TEAL),
            ("2. VAD", "Voice Activity\nDetection", GOLD),
            ("3. Diarization", "ECAPA +\nSpectral\nClustering", PURPLE),
            ("4. Separation", "SepFormer\nOverlaps\nonly", PINK),
            ("5. ASR", "MMS-1B-all\nSarvam\nfaster-whisper", GREEN),
            ("6. LLM", "Gemini 3.5\nFlash-Lite\nReport Gen", ORANGE),
        ]

        boxes = VGroup()
        for title, desc, color in stage_data:
            box = RoundedRectangle(
                width=1.8,
                height=2.2,
                corner_radius=0.15,
                fill_color=color,
                fill_opacity=0.15,
                stroke_color=color,
                stroke_width=2,
            )
            t = Text(title, font_size=13, color=color, weight=BOLD)
            d = Text(desc, font_size=10, color=WHITE_T)
            t.move_to(box.get_top() + DOWN * 0.35)
            d.move_to(box.get_center() + DOWN * 0.15)
            boxes.add(VGroup(box, t, d))

        boxes.arrange(RIGHT, buff=0.25)
        boxes.move_to(DOWN * 0.3)

        # Animate boxes one by one
        self.play(
            LaggedStart(
                *[FadeIn(b, shift=UP * 0.5, scale=0.8) for b in boxes],
                lag_ratio=0.15,
            ),
            run_time=2,
        )

        # Connecting arrows
        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arr = Arrow(
                boxes[i].get_right(),
                boxes[i + 1].get_left(),
                stroke_color=WHITE_T,
                stroke_width=1.5,
                buff=0.08,
                max_tip_length_to_length_ratio=0.3,
            )
            arrows.add(arr)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
            run_time=1,
        )

        # Input and Output labels; the six boxes span ~12 units, so the labels sit above
        # the first/last box instead of beside them (beside would leave the 14.2-unit frame)
        input_label = Text("Raw Audio (.wav)", font_size=14, color=PINK)
        input_label.next_to(boxes[0], UP, buff=0.35)
        input_arr = Arrow(
            input_label.get_bottom(),
            boxes[0].get_top(),
            stroke_color=PINK,
            buff=0.05,
        )

        output_label = Text("Report + Manifest", font_size=14, color=GREEN)
        output_label.next_to(boxes[-1], UP, buff=0.35)
        output_arr = Arrow(
            boxes[-1].get_top(),
            output_label.get_bottom(),
            stroke_color=GREEN,
            buff=0.05,
        )

        self.play(
            FadeIn(input_label),
            GrowArrow(input_arr),
            FadeIn(output_label),
            GrowArrow(output_arr),
            run_time=0.8,
        )

        # Animated data flow pulse
        for _ in range(2):
            pulse = Dot(
                radius=0.12,
                color=GOLD,
                fill_opacity=0.9,
            )
            path_points = [boxes[0].get_left()] + [b.get_center() for b in boxes] + [boxes[-1].get_right()]
            path = VMobject()
            path.set_points_as_corners([np.array([p[0], boxes.get_y(), 0]) for p in path_points])
            pulse.move_to(path.get_start())
            self.play(
                MoveAlongPath(pulse, path, run_time=2, rate_func=linear),
            )
            self.remove(pulse)

        # Metrics banner (report Tables 1-2 and Results)
        metrics = Text(
            "DER 25.4% (13.2% code-switch)  •  SI-SNRi +15.8 dB  •  WER 24.9% Sarvam / 56.6% MMS",
            font_size=16,
            color=GOLD,
        )
        metrics.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(metrics, shift=UP * 0.2), run_time=0.5)
        self.wait(1.5)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.8,
        )

    # ── SCENE 9: End Card ─────────────────────────────────────────
    def play_end_card(self) -> None:
        # Neural network background
        nn = make_nn([4, 6, 8, 6, 4], x_start=-4, x_end=4)
        nn.set_opacity(0.15)
        self.play(FadeIn(nn, run_time=1.5))

        title = Text("Dhvani", font_size=72, color=TEAL, weight=BOLD)
        sub = Text(
            "Who Spoke What and When",
            font_size=28,
            color=GOLD,
        )
        sub.next_to(title, DOWN, buff=0.4)

        team = Text(
            "Team HTMN  •  CS F407 Artificial Intelligence  •  BITS Pilani, Goa",
            font_size=18,
            color=WHITE_T,
        )
        team.next_to(sub, DOWN, buff=0.6)

        tech = Text(
            "Spectral Clustering • SepFormer • MMS / Sarvam ASR • Gemini",
            font_size=16,
            color=ManimColor(TEAL),
        )
        tech.next_to(team, DOWN, buff=0.3)

        self.play(
            Write(title, run_time=1),
            FadeIn(sub, shift=UP * 0.3, run_time=0.8),
        )
        self.play(
            FadeIn(team, shift=UP * 0.2),
            FadeIn(tech, shift=UP * 0.2),
            run_time=0.8,
        )

        # Final decorative waveform sweep
        final_wave = make_waveform(width=14, height=1, color=TEAL, seed=99)
        final_wave.to_edge(DOWN, buff=0.5)
        final_wave.set_opacity(0.4)
        self.play(Create(final_wave, run_time=2))
        self.wait(2)

        # Fade all out
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
        self.wait(0.5)


# ════════════════════════════════════════════════════════════════════
#  MODULAR INDIVIDUAL SCENES (Render separately for slides / clips)
# ════════════════════════════════════════════════════════════════════
class Scene1_Title(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_title()


class Scene2_RawAudio(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_raw_audio()


class Scene3_Preprocessing(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_preprocessing()


class Scene4_VADDiarization(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_vad_diarization()


class Scene5_OverlapSpectralClustering(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_overlap_separation()


class Scene6_ASR(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_asr()


class Scene7_LLMPostProcessing(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_llm_postprocessing()


class Scene8_PipelineOverview(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_pipeline_overview()


class Scene9_EndCard(DhvaniPipeline):
    def construct(self) -> None:
        self.camera.background_color = BG
        self.play_end_card()
