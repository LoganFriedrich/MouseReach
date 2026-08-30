"""
DLC Model Comparison Viewer
============================

Side-by-side comparison of two DLC models on the same video.

Two view modes:
  - **Merged View**: single video with both model overlays on top (different colours).
  - **Split View**: video shown twice side-by-side; Model A on the left, Model B on
    the right, using identical point colours so differences are easy to spot.

Load a video, then load two H5 files from different DLC models.
Both overlays render simultaneously so you can compare tracking in real time.

This is a standalone tool — not part of the processing pipeline.
"""

from pathlib import Path
from typing import Optional, List, Tuple

import napari
import numpy as np
import pandas as pd
from napari.layers import Image, Points
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QCheckBox, QRadioButton, QButtonGroup, QScrollArea,
    QFileDialog,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont


# ── Colour palettes ────────────────────────────────────────────────
# Warm tones — per-bodypart distinct mode (Model A in merged view)
MODEL_A_COLORS = [
    [1.0, 0.2, 0.2],   # red
    [1.0, 0.5, 0.0],   # orange
    [0.9, 0.9, 0.0],   # yellow
    [1.0, 0.3, 0.6],   # pink
    [0.8, 0.4, 0.0],   # brown-orange
    [1.0, 0.7, 0.3],   # gold
    [0.9, 0.1, 0.4],   # crimson
    [1.0, 0.6, 0.6],   # salmon
    [0.8, 0.8, 0.0],   # olive-yellow
    [1.0, 0.4, 0.2],   # tangerine
]

# Cool tones — per-bodypart distinct mode (Model B in merged view)
MODEL_B_COLORS = [
    [0.2, 0.4, 1.0],   # blue
    [0.0, 0.8, 0.4],   # green
    [0.5, 0.2, 1.0],   # purple
    [0.0, 0.9, 0.9],   # cyan
    [0.3, 0.6, 1.0],   # sky blue
    [0.0, 1.0, 0.6],   # spring green
    [0.6, 0.0, 0.8],   # violet
    [0.2, 0.8, 0.8],   # teal
    [0.4, 0.4, 1.0],   # periwinkle
    [0.0, 0.6, 0.3],   # forest green
]

# Shared palette for split view — identical on both sides
SPLIT_COLORS = [
    [0.0, 1.0, 0.0],   # green
    [1.0, 1.0, 0.0],   # yellow
    [0.0, 1.0, 1.0],   # cyan
    [1.0, 0.5, 0.0],   # orange
    [1.0, 0.0, 1.0],   # magenta
    [0.5, 1.0, 0.5],   # light green
    [1.0, 0.8, 0.4],   # gold
    [0.4, 0.8, 1.0],   # sky
    [1.0, 0.4, 0.4],   # salmon
    [0.6, 0.4, 1.0],   # lavender
]

# Uniform mode — all points one colour per model
MODEL_A_UNIFORM = [1.0, 0.2, 0.2]   # red
MODEL_B_UNIFORM = [0.3, 0.5, 1.0]   # blue

LIKELIHOOD_THRESHOLD = 0.5  # legacy default; per-bp table below overrides for known anatomical points

# Per-bodypart likelihood thresholds. SA / box reference points need to be
# tightly tracked for downstream geometry (ruler/pillar/slit), so we apply
# a stricter threshold to them. Anything not listed falls back to 0.5.
LK_THRESHOLD_PER_BP = {
    'SABL': 0.8, 'SABR': 0.8, 'SATL': 0.8, 'SATR': 0.8,
    'BOXL': 0.8, 'BOXR': 0.8,
}


def _lk_threshold(bp: str) -> float:
    return LK_THRESHOLD_PER_BP.get(bp, LIKELIHOOD_THRESHOLD)


def _load_dlc_h5(h5_path: Path) -> Optional[pd.DataFrame]:
    """Load a DLC H5 file and flatten multi-level columns to bodypart_coord."""
    try:
        df = pd.read_hdf(str(h5_path))
        # Flatten hierarchical columns: (scorer, bodypart, coord) -> bodypart_coord
        df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
        return df
    except Exception as e:
        print(f"[DLC Compare] Failed to load {h5_path}: {e}")
        return None


def _get_bodyparts(df: pd.DataFrame) -> List[str]:
    """Extract sorted unique bodypart names from flattened DLC dataframe."""
    bps = []
    for col in df.columns:
        if col.endswith('_x'):
            bps.append(col[:-2])
    return sorted(set(bps))


def _build_points_array(df: pd.DataFrame, bodyparts: List[str],
                        palette: List[List[float]], alpha: float = 0.7,
                        uniform_color: Optional[List[float]] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build napari-ready points/colours/labels arrays.

    Returns ``(points (N, 3), colors (N, 4) RGBA, bps (N,) str)``.

    Below the per-bodypart likelihood threshold the point still renders, but
    at alpha ~0.05 so it reads as "the model would discard this." Above
    threshold, alpha follows a quadratic ramp from 0.10*alpha to 1.0*alpha
    so a borderline-confident point looks visibly fainter than a rock-solid
    one. The caller's ``alpha`` argument still scales the *upper* end so
    Model A vs Model B can be differentiated in merged view.
    """
    if uniform_color is not None:
        bp_colors = {bp: uniform_color for bp in bodyparts}
    else:
        bp_colors = {bp: palette[i % len(palette)] for i, bp in enumerate(bodyparts)}

    points = []
    colors = []
    bps = []

    n_frames = len(df)
    for frame_idx in range(n_frames):
        row = df.iloc[frame_idx]
        for bp in bodyparts:
            x = row.get(f'{bp}_x', np.nan)
            y = row.get(f'{bp}_y', np.nan)
            if np.isnan(x) or np.isnan(y):
                continue
            lk = float(row.get(f'{bp}_likelihood', 0))
            threshold = _lk_threshold(bp)
            if lk < threshold:
                pt_alpha = 0.05
            else:
                norm = (lk - threshold) / max(1e-6, 1.0 - threshold)
                pt_alpha = (0.10 + 0.90 * (norm ** 2)) * alpha
            points.append([frame_idx, y, x])
            colors.append(bp_colors[bp] + [pt_alpha])
            bps.append(bp)

    if not points:
        return (
            np.empty((0, 3)),
            np.empty((0, 4)),
            np.array([], dtype=object),
        )

    return np.array(points), np.array(colors), np.array(bps, dtype=object)


class DLCCompareWidget(QWidget):
    """Side-by-side DLC model comparison viewer."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Data
        self.video_path: Optional[Path] = None
        self.h5_path_a: Optional[Path] = None
        self.h5_path_b: Optional[Path] = None
        self.df_a: Optional[pd.DataFrame] = None
        self.df_b: Optional[pd.DataFrame] = None

        # Layers — merged mode
        self.video_layer: Optional[Image] = None
        self.points_layer_a: Optional[Points] = None
        self.points_layer_b: Optional[Points] = None

        # Layers — split mode (second video + separate point layers)
        self.video_layer_b: Optional[Image] = None
        self.points_layer_split_a: Optional[Points] = None
        self.points_layer_split_b: Optional[Points] = None

        # View mode: 'merged' or 'split'
        self._view_mode = 'merged'

        # Colour mode (merged only): True = uniform, False = distinct per-bodypart
        self._uniform_color_mode = False

        # Playback
        self.is_playing = False
        self.playback_speed = 1
        self.n_frames = 0
        self.fps = 60.0
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        self.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout()
        layout.setSpacing(6)
        inner.setLayout(layout)

        # ── Instructions ──
        instr = QLabel(
            "Space=play/pause  S=stop  Left/Right=frame step"
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(instr)

        # ── View mode selection ──
        mode_group = QGroupBox("View Mode")
        mode_layout = QHBoxLayout()
        self._mode_btn_group = QButtonGroup(self)

        self._merged_radio = QRadioButton("Merged View")
        self._merged_radio.setToolTip("Both models overlaid on one video")
        self._merged_radio.setChecked(True)
        self._mode_btn_group.addButton(self._merged_radio)
        mode_layout.addWidget(self._merged_radio)

        self._split_radio = QRadioButton("Split View")
        self._split_radio.setToolTip(
            "Video shown twice side-by-side: Model A left, Model B right"
        )
        self._mode_btn_group.addButton(self._split_radio)
        mode_layout.addWidget(self._split_radio)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        self._mode_btn_group.buttonClicked.connect(self._on_view_mode_changed)

        # ── File loading ──
        file_group = QGroupBox("Load Files")
        file_layout = QVBoxLayout()

        # Video
        video_row = QHBoxLayout()
        self.video_btn = QPushButton("Load Video...")
        self.video_btn.clicked.connect(self._pick_video)
        video_row.addWidget(self.video_btn)
        self.video_label = QLabel("No video loaded")
        self.video_label.setStyleSheet("color: #888;")
        video_row.addWidget(self.video_label, 1)
        file_layout.addLayout(video_row)

        # Model A
        h5a_row = QHBoxLayout()
        self.h5a_btn = QPushButton("Load Model A (.h5)...")
        self.h5a_btn.clicked.connect(lambda: self._pick_h5('a'))
        self.h5a_btn.setStyleSheet("background-color: #4a2020;")
        h5a_row.addWidget(self.h5a_btn)
        self.h5a_label = QLabel("—")
        self.h5a_label.setStyleSheet("color: #ff6666;")
        h5a_row.addWidget(self.h5a_label, 1)
        file_layout.addLayout(h5a_row)

        # Model B
        h5b_row = QHBoxLayout()
        self.h5b_btn = QPushButton("Load Model B (.h5)...")
        self.h5b_btn.clicked.connect(lambda: self._pick_h5('b'))
        self.h5b_btn.setStyleSheet("background-color: #202040;")
        h5b_row.addWidget(self.h5b_btn)
        self.h5b_label = QLabel("—")
        self.h5b_label.setStyleSheet("color: #6688ff;")
        h5b_row.addWidget(self.h5b_label, 1)
        file_layout.addLayout(h5b_row)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # ── Playback ──
        play_group = QGroupBox("Playback")
        play_layout = QVBoxLayout()

        self.frame_label = QLabel("Frame: — / —")
        self.frame_label.setFont(QFont("Consolas", 10))
        play_layout.addWidget(self.frame_label)

        self.time_label = QLabel("Time: —")
        self.time_label.setFont(QFont("Consolas", 10))
        play_layout.addWidget(self.time_label)

        # Play / Stop
        transport_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        transport_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        transport_row.addWidget(self.stop_btn)
        play_layout.addLayout(transport_row)

        # Speed buttons
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        for mult in (1, 2, 4, 8, 16):
            btn = QPushButton(f"{mult}x")
            btn.setCheckable(True)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, m=mult: self._set_speed_direct(m))
            speed_row.addWidget(btn)
            if mult == 1:
                btn.setChecked(True)
                self._active_speed_btn = btn
            setattr(self, f'_speed_btn_{mult}', btn)
        play_layout.addLayout(speed_row)

        # Visibility toggles
        vis_row = QHBoxLayout()
        self.show_a_cb = QCheckBox("Show Model A")
        self.show_a_cb.setChecked(True)
        self.show_a_cb.setStyleSheet("color: #ff6666;")
        self.show_a_cb.toggled.connect(self._toggle_visibility_a)
        vis_row.addWidget(self.show_a_cb)

        self.show_b_cb = QCheckBox("Show Model B")
        self.show_b_cb.setChecked(True)
        self.show_b_cb.setStyleSheet("color: #6688ff;")
        self.show_b_cb.toggled.connect(self._toggle_visibility_b)
        vis_row.addWidget(self.show_b_cb)
        play_layout.addLayout(vis_row)

        # Colour mode toggle (merged view only)
        color_row = QHBoxLayout()
        self.color_mode_btn = QPushButton("Colour Mode: Distinct (per bodypart)")
        self.color_mode_btn.clicked.connect(self._toggle_color_mode)
        color_row.addWidget(self.color_mode_btn)
        play_layout.addLayout(color_row)

        play_group.setLayout(play_layout)
        layout.addWidget(play_group)

        layout.addStretch()

        # Connect to napari frame slider
        self.viewer.dims.events.current_step.connect(self._on_frame_change)

    # ── View mode ───────────────────────────────────────────────────

    def _on_view_mode_changed(self, btn):
        new_mode = 'split' if btn is self._split_radio else 'merged'
        if new_mode == self._view_mode:
            return
        self._view_mode = new_mode

        # Show/hide merged-only controls
        is_merged = new_mode == 'merged'
        self.color_mode_btn.setVisible(is_merged)
        self.show_a_cb.setVisible(is_merged)
        self.show_b_cb.setVisible(is_merged)

        # Reload everything in the new mode
        self._clear_all_layers()
        if self.video_path is not None:
            self._load_video(self.video_path)
        self._refresh_overlays()

    # ── File loading ────────────────────────────────────────────────

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return
        self._load_video(Path(path))

    def _load_video(self, path: Path):
        from mousereach.lazy_video import LazyVideoArray

        # Remove old video layers
        for attr in ('video_layer', 'video_layer_b'):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)

        try:
            lazy_a = LazyVideoArray(path)
            self.n_frames = lazy_a.n_frames
            self.fps = lazy_a.fps
            self.video_path = path

            if self._view_mode == 'split':
                # Two copies of the video for grid display
                self.video_layer = self.viewer.add_image(
                    lazy_a, name=f"[Model A] {path.stem}",
                    metadata={'path': str(path), 'dlc_compare': True},
                )
                lazy_b = LazyVideoArray(path)
                self.video_layer_b = self.viewer.add_image(
                    lazy_b, name=f"[Model B] {path.stem}",
                    metadata={'path': str(path), 'dlc_compare': True},
                )
                # Enable napari grid mode: 1 row, 2 columns
                self.viewer.grid.enabled = True
                self.viewer.grid.shape = (1, 2)
            else:
                # Single video (merged)
                self.video_layer = self.viewer.add_image(
                    lazy_a, name=f"Video: {path.stem}",
                    metadata={'path': str(path), 'dlc_compare': True},
                )
                self.viewer.grid.enabled = False

            self.video_label.setText(path.name)
            self.video_label.setStyleSheet("color: #8f8;")
            self._on_frame_change()
            mode_str = self._view_mode
            print(f"[DLC Compare] Loaded video ({mode_str}): {path.name} "
                  f"({self.n_frames} frames, {self.fps:.0f} fps)")
        except Exception as e:
            self.video_label.setText(f"ERROR: {e}")
            self.video_label.setStyleSheet("color: #f88;")

    def _pick_h5(self, which: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select DLC H5 file (Model {'A' if which == 'a' else 'B'})", "",
            "HDF5 Files (*.h5);;All Files (*)"
        )
        if not path:
            return
        self._load_h5(Path(path), which)

    def _load_h5(self, path: Path, which: str):
        df = _load_dlc_h5(path)
        if df is None:
            label = self.h5a_label if which == 'a' else self.h5b_label
            label.setText(f"ERROR loading {path.name}")
            return

        if which == 'a':
            self.h5_path_a = path
            self.df_a = df
            self.h5a_label.setText(path.name)
            self.h5a_label.setToolTip(str(path))
            print(f"[DLC Compare] Model A loaded: {path.name} ({len(df)} frames)")
        else:
            self.h5_path_b = path
            self.df_b = df
            self.h5b_label.setText(path.name)
            self.h5b_label.setToolTip(str(path))
            print(f"[DLC Compare] Model B loaded: {path.name} ({len(df)} frames)")

        self._refresh_overlays()

    # ── Layer management ────────────────────────────────────────────

    def _clear_all_layers(self):
        """Remove every layer this widget owns."""
        layer_attrs = (
            'points_layer_a', 'points_layer_b',
            'points_layer_split_a', 'points_layer_split_b',
            'video_layer', 'video_layer_b',
        )
        for attr in layer_attrs:
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)
        self.viewer.grid.enabled = False

    # ── Overlay rendering ───────────────────────────────────────────

    def _refresh_overlays(self):
        """Rebuild point layers for the active view mode."""
        if self._view_mode == 'split':
            self._refresh_overlays_split()
        else:
            self._refresh_overlays_merged()

    def _refresh_overlays_merged(self):
        """Merged view: two point layers on one video, different colours."""
        # Remove old point layers
        for attr in ('points_layer_a', 'points_layer_b',
                     'points_layer_split_a', 'points_layer_split_b'):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)

        uniform_a = MODEL_A_UNIFORM if self._uniform_color_mode else None
        uniform_b = MODEL_B_UNIFORM if self._uniform_color_mode else None

        # Model A: larger points, reduced opacity (sits underneath)
        if self.df_a is not None:
            bps = _get_bodyparts(self.df_a)
            pts, cols, pt_bps = _build_points_array(
                self.df_a, bps, MODEL_A_COLORS, alpha=0.66,
                uniform_color=uniform_a,
            )
            if len(pts) > 0:
                self.points_layer_a = self.viewer.add_points(
                    pts, name='Model A', size=4, face_color=cols,
                    visible=self.show_a_cb.isChecked(),
                    features={'bp': pt_bps},
                )

        # Model B: smaller points, full opacity (layers on top)
        if self.df_b is not None:
            bps = _get_bodyparts(self.df_b)
            pts, cols, pt_bps = _build_points_array(
                self.df_b, bps, MODEL_B_COLORS, alpha=1.0,
                uniform_color=uniform_b,
            )
            if len(pts) > 0:
                self.points_layer_b = self.viewer.add_points(
                    pts, name='Model B', size=3, face_color=cols,
                    visible=self.show_b_cb.isChecked(),
                    features={'bp': pt_bps},
                )

    def _refresh_overlays_split(self):
        """Split view: identical colours on each side, points bound to their video."""
        # Remove old point layers
        for attr in ('points_layer_a', 'points_layer_b',
                     'points_layer_split_a', 'points_layer_split_b'):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)

        # Both models use the same colour palette so you see differences by position
        if self.df_a is not None:
            bps = _get_bodyparts(self.df_a)
            pts, cols = _build_points_array(self.df_a, bps, SPLIT_COLORS)
            if len(pts) > 0:
                self.points_layer_split_a = self.viewer.add_points(
                    pts, name='Model A (left)', size=8, face_color=cols,
                )
                # Bind to first video layer so grid keeps it on the left
                if self.video_layer is not None:
                    self.points_layer_split_a.translate = \
                        self.video_layer.translate

        if self.df_b is not None:
            bps = _get_bodyparts(self.df_b)
            pts, cols = _build_points_array(self.df_b, bps, SPLIT_COLORS)
            if len(pts) > 0:
                self.points_layer_split_b = self.viewer.add_points(
                    pts, name='Model B (right)', size=8, face_color=cols,
                )
                # Bind to second video layer so grid keeps it on the right
                if self.video_layer_b is not None:
                    self.points_layer_split_b.translate = \
                        self.video_layer_b.translate

    # ── Colour mode toggle (merged only) ────────────────────────────

    def _toggle_color_mode(self):
        self._uniform_color_mode = not self._uniform_color_mode
        if self._uniform_color_mode:
            self.color_mode_btn.setText("Colour Mode: Uniform (A=red, B=blue)")
        else:
            self.color_mode_btn.setText("Colour Mode: Distinct (per bodypart)")
        self._refresh_overlays()

    # ── Visibility toggles (merged only) ────────────────────────────

    def _toggle_visibility_a(self, checked):
        if self.points_layer_a is not None:
            self.points_layer_a.visible = checked

    def _toggle_visibility_b(self, checked):
        if self.points_layer_b is not None:
            self.points_layer_b.visible = checked

    # ── Playback ────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.play_btn.setText("Play")
        else:
            self.is_playing = True
            interval = max(1, int(1000 / (self.fps * self.playback_speed)))
            self.playback_timer.start(interval)
            self.play_btn.setText("Pause")

    def _stop(self):
        """Stop playback and rewind to frame 0."""
        self.is_playing = False
        self.playback_timer.stop()
        self.play_btn.setText("Play")
        if self.n_frames > 0:
            self.viewer.dims.set_current_step(0, 0)

    def _playback_step(self):
        if self.n_frames == 0:
            return
        current = self.viewer.dims.current_step[0]
        next_frame = current + self.playback_speed
        if next_frame >= self.n_frames:
            next_frame = 0
        self.viewer.dims.set_current_step(0, next_frame)

    def _set_speed_direct(self, multiplier: int):
        """Set playback speed from button click."""
        self.playback_speed = multiplier
        for mult in (1, 2, 4, 8, 16):
            btn = getattr(self, f'_speed_btn_{mult}', None)
            if btn is not None:
                btn.setChecked(mult == multiplier)
        if self.is_playing:
            interval = max(1, int(1000 / (self.fps * self.playback_speed)))
            self.playback_timer.setInterval(interval)

    # ── Frame change handler ────────────────────────────────────────

    def _on_frame_change(self, event=None):
        if self.n_frames == 0:
            return
        frame_idx = self.viewer.dims.current_step[0]
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.frame_label.setText(f"Frame: {frame_idx} / {self.n_frames}")
        self.time_label.setText(f"Time: {mins}:{secs:05.2f}")

    # ── Keybindings ─────────────────────────────────────────────────

    def _setup_keybindings(self):
        """Register keyboard shortcuts with napari. Called after widget is docked."""

        @self.viewer.bind_key('Space', overwrite=True)
        def _toggle_play(viewer):
            self._toggle_play()

        @self.viewer.bind_key('s', overwrite=True)
        def _stop(viewer):
            self._stop()


# ── Standalone launcher ─────────────────────────────────────────────

def main():
    """Launch DLC Model Comparison as a standalone napari window."""
    import os
    import argparse

    os.environ['NAPARI_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    os.environ['NAPARI_DISABLE_PLUGINS'] = '1'

    parser = argparse.ArgumentParser(description="DLC Model Comparison Viewer")
    parser.add_argument('video', nargs='?', type=Path,
                        help="Video file to load")
    parser.add_argument('--model-a', '-a', type=Path,
                        help="DLC H5 file for Model A")
    parser.add_argument('--model-b', '-b', type=Path,
                        help="DLC H5 file for Model B")
    parser.add_argument('--split', action='store_true',
                        help="Start in split view mode")
    args = parser.parse_args()

    print("DLC Model Comparison Viewer")
    print("Loading napari...", flush=True)

    viewer = napari.Viewer(title="DLC Model Comparison")
    widget = DLCCompareWidget(viewer)
    viewer.window.add_dock_widget(widget, name="DLC Compare", area="right")
    widget._setup_keybindings()

    # Set initial view mode from CLI
    if args.split:
        widget._split_radio.setChecked(True)
        widget._on_view_mode_changed(widget._split_radio)

    # Auto-load files if provided
    if args.video and args.video.exists():
        QTimer.singleShot(300, lambda: widget._load_video(args.video))

        if args.model_a and args.model_a.exists():
            QTimer.singleShot(600, lambda: widget._load_h5(args.model_a, 'a'))
        if args.model_b and args.model_b.exists():
            QTimer.singleShot(900, lambda: widget._load_h5(args.model_b, 'b'))

    napari.run()


if __name__ == "__main__":
    main()
