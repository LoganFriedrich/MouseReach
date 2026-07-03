"""
Causal Review Widget -- results-checker / triage-answerer for MouseReach.

Walks the segments of a video and presents an editable question panel
for each segment, collecting algo-vs-human review records that the
active-learning loop can bulk-read.

Design:
  - Reuses GroundTruthWidget's video loading, DLC overlay, pillar
    overlay, and playback machinery via composition (not subclassing).
  - Walks segments sequentially; per-segment question panels are
    editable Yes/No toggles with inline correction widgets.
  - Saves to a NEW file type (``*_causal_review.json``) plus a
    corpus-level index.

Entry points:
  - napari plugin contribution (``mousereach.causal_review_widget``)
  - Tab inside UnifiedReviewWidget
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QFrame, QComboBox, QSpinBox,
    QTextEdit, QRadioButton, QButtonGroup, QSizePolicy,
    QFileDialog, QProgressBar, QCheckBox, QMessageBox,
)
from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QFont, QColor

import napari
from napari.utils.notifications import show_info, show_warning, show_error

from .causal_review_io import (
    build_segment_record,
    collect_provenance,
    save_causal_review,
    update_corpus_index,
    _get_username,
    _get_timestamp,
)


# ---------------------------------------------------------------------------
# Outcome choices (matches the cascade label set)
# ---------------------------------------------------------------------------
OUTCOME_CHOICES = [
    "untouched",
    "displaced_sa",
    "displaced_outside",
    "retrieved",
    "abnormal",
]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class CausalReviewWidget(QWidget):
    """Per-segment causal-reach review tool.

    Loads a video's algo outputs (segments, reaches, outcomes,
    assignments) and walks through each segment presenting yes/no
    questions about the algo's causal attribution. Human corrections
    are saved as ``*_causal_review.json``.
    """

    data_saved = Signal(Path)

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Video state
        self.video_path: Optional[Path] = None
        self.video_layer = None
        self.n_frames = 0
        self.fps = 60.0
        self.scale_factor = 1.0
        self.frame_offset = 0
        self.frame_window_end = 0

        # DLC overlay state
        self.dlc_df = None
        self.points_layer = None
        self.pillar_shapes_layer = None

        # Algo data
        self._segments_data: Dict[str, Any] = {}
        self._reaches_data: Dict[str, Any] = {}
        self._outcomes_data: Dict[str, Any] = {}
        self._assignments_data: Dict[str, Any] = {}
        self._video_stem: str = ""

        # Segment walking state
        self._segment_list: List[Dict[str, Any]] = []
        self._current_seg_idx: int = 0

        # Per-segment review records (accumulated across the walk)
        self._review_records: Dict[int, Dict[str, Any]] = {}

        # Playback
        self.is_playing = False
        self.playback_speed = 1.0
        self.playback_direction = 1
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        self._build_ui()

    # ===================================================================
    # UI construction
    # ===================================================================

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(4)
        self.setLayout(main_layout)

        # --- Header: video selector ---
        header = QGroupBox("Video")
        header_layout = QVBoxLayout()
        header.setLayout(header_layout)

        title = QLabel("<b>Causal Review Tool</b>")
        title.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(title)

        vid_row = QHBoxLayout()
        browse_btn = QPushButton("Load Video...")
        browse_btn.clicked.connect(self._browse_video)
        vid_row.addWidget(browse_btn)
        self._video_label = QLabel("No video loaded")
        self._video_label.setWordWrap(True)
        self._video_label.setStyleSheet("color: #888;")
        vid_row.addWidget(self._video_label, stretch=1)
        header_layout.addLayout(vid_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        header_layout.addWidget(self._progress)

        main_layout.addWidget(header)

        # --- Navigation bar ---
        nav = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()
        nav.setLayout(nav_layout)

        # Frame display
        frame_row = QHBoxLayout()
        self._frame_label = QLabel("Frame: -- / --")
        self._frame_label.setFont(QFont("", 11, QFont.Bold))
        frame_row.addWidget(self._frame_label)
        frame_row.addStretch()
        self._time_label = QLabel("Time: --:--")
        frame_row.addWidget(self._time_label)
        nav_layout.addLayout(frame_row)

        # Playback buttons
        play_row = QHBoxLayout()
        self._play_rev_btn = QPushButton("Rev")
        self._play_rev_btn.clicked.connect(self._play_reverse)
        self._play_rev_btn.setEnabled(False)
        self._play_rev_btn.setMaximumWidth(40)
        play_row.addWidget(self._play_rev_btn)
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._play_forward)
        self._play_btn.setEnabled(False)
        self._play_btn.setMaximumWidth(40)
        play_row.addWidget(self._play_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop_play)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setMaximumWidth(40)
        play_row.addWidget(self._stop_btn)
        play_row.addStretch()
        # Speed buttons
        play_row.addWidget(QLabel("Speed:"))
        self._speed_buttons = {}
        for speed in [0.25, 0.5, 1, 2, 4, 8]:
            label = f"{speed}x" if speed < 1 else f"{int(speed)}x"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMaximumWidth(35)
            btn.clicked.connect(lambda _, s=speed: self._set_speed(s))
            self._speed_buttons[speed] = btn
            play_row.addWidget(btn)
        self._speed_buttons[1].setChecked(True)
        nav_layout.addLayout(play_row)

        # Frame step buttons
        step_row = QHBoxLayout()
        for delta, label in [(-100, "<<"), (-10, "<"), (-1, "-1"), (1, "+1"), (10, ">"), (100, ">>")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, d=delta: self._jump_frames(d))
            btn.setMaximumWidth(35)
            step_row.addWidget(btn)
        nav_layout.addLayout(step_row)

        # Segment navigation
        seg_nav_row = QHBoxLayout()
        self._prev_seg_btn = QPushButton("<< Prev Segment")
        self._prev_seg_btn.clicked.connect(self._prev_segment)
        self._prev_seg_btn.setEnabled(False)
        seg_nav_row.addWidget(self._prev_seg_btn)
        self._seg_label = QLabel("Segment: -- / --")
        self._seg_label.setAlignment(Qt.AlignCenter)
        seg_nav_row.addWidget(self._seg_label, stretch=1)
        self._next_seg_btn = QPushButton("Next Segment >>")
        self._next_seg_btn.clicked.connect(self._next_segment)
        self._next_seg_btn.setEnabled(False)
        seg_nav_row.addWidget(self._next_seg_btn)
        nav_layout.addLayout(seg_nav_row)

        # Load whole segment button
        whole_seg_row = QHBoxLayout()
        self._load_whole_seg_btn = QPushButton("Load Whole Segment")
        self._load_whole_seg_btn.setToolTip(
            "Load the full segment's frame range (default view shows "
            "only the causal reach neighborhood)"
        )
        self._load_whole_seg_btn.clicked.connect(self._load_whole_segment)
        self._load_whole_seg_btn.setEnabled(False)
        whole_seg_row.addWidget(self._load_whole_seg_btn)
        whole_seg_row.addStretch()
        nav_layout.addLayout(whole_seg_row)

        main_layout.addWidget(nav)

        # --- Scrollable question panel ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._questions_container = QWidget()
        self._questions_layout = QVBoxLayout()
        self._questions_layout.setSpacing(6)
        self._questions_container.setLayout(self._questions_layout)
        scroll.setWidget(self._questions_container)
        main_layout.addWidget(scroll, stretch=1)

        # --- Footer: save + status ---
        footer = QFrame()
        footer_layout = QHBoxLayout()
        footer.setLayout(footer_layout)

        self._save_btn = QPushButton("Save Review")
        self._save_btn.setStyleSheet("font-weight: bold; background: #2d5016; color: white;")
        self._save_btn.setToolTip("Save all reviewed segments to *_causal_review.json")
        self._save_btn.clicked.connect(self._save_review)
        self._save_btn.setEnabled(False)
        footer_layout.addWidget(self._save_btn)

        self._save_advance_btn = QPushButton("Save Segment + Next")
        self._save_advance_btn.setToolTip(
            "Record this segment's answers and advance to the next"
        )
        self._save_advance_btn.clicked.connect(self._save_segment_and_advance)
        self._save_advance_btn.setEnabled(False)
        footer_layout.addWidget(self._save_advance_btn)

        footer_layout.addStretch()

        self._status_label = QLabel("Load a video to begin")
        self._status_label.setStyleSheet("color: #888;")
        footer_layout.addWidget(self._status_label)

        main_layout.addWidget(footer)

        # Connect frame change
        self.viewer.dims.events.current_step.connect(self._on_frame_change)

    # ===================================================================
    # Video loading (reuses GT widget patterns)
    # ===================================================================

    def _browse_video(self):
        try:
            from mousereach.config import Paths
            default_dir = str(Paths.PROCESSING_ROOT) if Paths.PROCESSING_ROOT else ""
        except Exception:
            default_dir = ""

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", default_dir,
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        if path:
            self._load_video(Path(path))

    def _load_video(self, video_path: Path):
        """Load video frames, DLC overlays, and algo data."""
        import cv2

        self.video_path = video_path
        self._video_label.setText(f"Loading: {video_path.name}")
        self._progress.setVisible(True)
        self._progress.setValue(0)

        try:
            # Resolve preview
            video_stem = video_path.stem.replace("_preview", "")
            if "DLC" in video_stem:
                video_stem = video_stem.split("DLC")[0].rstrip("_")
            self._video_stem = video_stem

            preview_path = video_path.parent / f"{video_stem}_preview.mp4"
            if "_preview" not in video_path.stem and preview_path.exists():
                actual_video = preview_path
                self.scale_factor = 0.75
            else:
                actual_video = video_path
                self.scale_factor = 1.0

            # Read frames
            cap = cv2.VideoCapture(str(actual_video))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            self.n_frames = total_frames
            self.frame_offset = 0
            self.frame_window_end = total_frames - 1

            frames = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if i % 500 == 0:
                    self._progress.setValue(int(60 * i / max(total_frames, 1)))
                    from qtpy.QtWidgets import QApplication
                    QApplication.processEvents()
            cap.release()
            self.n_frames = len(frames)
            self.frame_window_end = self.n_frames - 1
            self._progress.setValue(65)

            # Remove old video layer
            if self.video_layer is not None:
                try:
                    self.viewer.layers.remove(self.video_layer)
                except (ValueError, AttributeError):
                    pass

            self.video_layer = self.viewer.add_image(
                np.stack(frames), name=video_path.stem, rgb=True
            )
            self._progress.setValue(75)

            # Load DLC data
            self._load_dlc_data()
            self._progress.setValue(80)
            self._add_dlc_points_layer()
            self._progress.setValue(85)
            self._add_pillar_shapes_layer()
            self._progress.setValue(88)

            # Load algo data
            self._load_algo_data()
            self._progress.setValue(92)

            # Build segment list
            self._build_segment_list()
            self._progress.setValue(95)

            # Enable controls
            self._enable_controls(True)

            # Show first segment
            self._current_seg_idx = 0
            self._show_current_segment()

            self._video_label.setText(
                f"Loaded: {video_path.name} ({self.n_frames} frames, "
                f"{len(self._segment_list)} segments)"
            )
            self._progress.setValue(100)

        except Exception as e:
            show_error(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._progress.setVisible(False)

    def _load_dlc_data(self):
        """Load DLC H5 for overlays."""
        if not self.video_path:
            return
        video_stem = self._video_stem
        for pattern_name in [f"{video_stem}*.h5"]:
            for h5_path in self.video_path.parent.glob(pattern_name):
                try:
                    import pandas as pd
                    df = pd.read_hdf(h5_path)
                    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
                    self.dlc_df = df
                    return
                except Exception:
                    pass
        self.dlc_df = None

    def _add_dlc_points_layer(self):
        """Add DLC tracking points overlay (copied from GroundTruthWidget pattern)."""
        if self.dlc_df is None:
            return

        bodyparts = sorted({col[:-2] for col in self.dlc_df.columns if col.endswith('_x')})
        colors_base = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5]
        ]
        bp_colors = {bp: colors_base[i % len(colors_base)] for i, bp in enumerate(bodyparts)}

        points_data = []
        point_colors = []
        point_bps = []
        scale = self.scale_factor

        for frame_idx in range(len(self.dlc_df)):
            for bp in bodyparts:
                x = self.dlc_df.iloc[frame_idx].get(f'{bp}_x', np.nan)
                y = self.dlc_df.iloc[frame_idx].get(f'{bp}_y', np.nan)
                lk = float(self.dlc_df.iloc[frame_idx].get(f'{bp}_likelihood', 0))
                if np.isnan(x) or np.isnan(y):
                    continue
                if lk < 0.5:
                    alpha = 0.05
                else:
                    norm = (lk - 0.5) / 0.5
                    alpha = 0.10 + 0.90 * (norm ** 2)
                points_data.append([frame_idx, y * scale, x * scale])
                point_colors.append(bp_colors[bp] + [alpha])
                point_bps.append(bp)

        if points_data:
            if self.points_layer is not None:
                try:
                    self.viewer.layers.remove(self.points_layer)
                except Exception:
                    pass
            self.points_layer = self.viewer.add_points(
                np.array(points_data), name='DLC Points', size=3,
                face_color=np.array(point_colors),
                features={'bp': point_bps},
                text={'string': '{bp}', 'size': 7, 'color': 'white',
                      'translation': [0, -7, 0]},
            )

    def _add_pillar_shapes_layer(self):
        """Add pillar circle overlay."""
        if self.dlc_df is None:
            return
        if self.pillar_shapes_layer is not None:
            try:
                self.viewer.layers.remove(self.pillar_shapes_layer)
            except Exception:
                pass
        self.pillar_shapes_layer = self.viewer.add_shapes(
            name='Pillar', edge_color='red',
            face_color=[0, 0, 0, 0], edge_width=3,
        )
        if not hasattr(self, '_pillar_callback_connected'):
            self.viewer.dims.events.current_step.connect(self._update_pillar_circle)
            self._pillar_callback_connected = True
        self._update_pillar_circle()

    def _update_pillar_circle(self, event=None):
        """Update pillar circle for current frame."""
        if self.pillar_shapes_layer is None or self.dlc_df is None:
            return
        frame_idx = int(self.viewer.dims.current_step[0])
        if frame_idx < 0 or frame_idx >= len(self.dlc_df):
            return
        row = self.dlc_df.iloc[frame_idx]
        scale = self.scale_factor

        sabl_lk = row.get('SABL_likelihood', 0)
        sabr_lk = row.get('SABR_likelihood', 0)
        if sabl_lk < 0.5 or sabr_lk < 0.5:
            self.pillar_shapes_layer.data = []
            return
        sabl_x = row.get('SABL_x', np.nan) * scale
        sabl_y = row.get('SABL_y', np.nan) * scale
        sabr_x = row.get('SABR_x', np.nan) * scale
        sabr_y = row.get('SABR_y', np.nan) * scale

        if np.isnan([sabl_x, sabl_y, sabr_x, sabr_y]).any():
            self.pillar_shapes_layer.data = []
            return

        ruler = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)
        if ruler < 1:
            self.pillar_shapes_layer.data = []
            return

        mid_x = (sabl_x + sabr_x) / 2
        mid_y = (sabl_y + sabr_y) / 2
        pillar_x = mid_x
        pillar_y = mid_y - (0.944 * ruler)
        pillar_r = 0.138 * ruler

        # Draw circle as ellipse
        n_pts = 40
        theta = np.linspace(0, 2 * np.pi, n_pts)
        circle = np.column_stack([
            np.full(n_pts, frame_idx),
            pillar_y + pillar_r * np.sin(theta),
            pillar_x + pillar_r * np.cos(theta),
        ])
        self.pillar_shapes_layer.data = [circle]

    # ===================================================================
    # Algo data loading
    # ===================================================================

    def _load_algo_data(self):
        """Load segments, reaches, outcomes, and assignments JSONs."""
        if not self.video_path:
            return
        parent = self.video_path.parent
        stem = self._video_stem

        def _try_load(suffix: str) -> Dict[str, Any]:
            path = parent / f"{stem}{suffix}"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            return {}

        self._segments_data = _try_load("_segments.json")
        if not self._segments_data:
            self._segments_data = _try_load("_segmentation.json")
        self._reaches_data = _try_load("_reaches.json")
        self._outcomes_data = _try_load("_pellet_outcomes.json")
        self._assignments_data = _try_load("_reach_assignments.json")

    def _build_segment_list(self):
        """Build the unified segment list from segments + outcomes + assignments."""
        self._segment_list = []

        # Extract boundaries
        boundaries = []
        if "boundaries" in self._segments_data:
            for b in self._segments_data["boundaries"]:
                if isinstance(b, dict):
                    boundaries.append(int(b.get("frame", b.get("index", 0))))
                else:
                    boundaries.append(int(b))
        elif "segments" in self._segments_data:
            for s in self._segments_data["segments"]:
                sf = s.get("start_frame")
                ef = s.get("end_frame")
                if sf is not None and ef is not None:
                    if not boundaries:
                        boundaries.append(int(sf))
                    boundaries.append(int(ef) + 1)

        if len(boundaries) < 2:
            self._status_label.setText("No segments found in algo data")
            return

        # Build segment entries
        n_segments = len(boundaries) - 1

        # Index outcomes by segment_num
        outcomes_by_seg: Dict[int, Dict] = {}
        for seg in self._outcomes_data.get("segments", []):
            sn = seg.get("segment_num")
            if sn is not None:
                outcomes_by_seg[int(sn)] = seg

        # Index reaches by segment_num from assignments
        assignments_by_seg: Dict[int, List[Dict]] = {}
        for r in self._assignments_data.get("reaches", []):
            sn = r.get("segment_num")
            if sn is not None:
                assignments_by_seg.setdefault(int(sn), []).append(r)

        # Also get raw reaches (from reach detector) grouped by segment
        raw_reaches_by_seg: Dict[int, List[Dict]] = {}
        # Flat format (v8+)
        if isinstance(self._reaches_data.get("reaches"), list):
            for r in self._reaches_data["reaches"]:
                # Determine which segment this reach belongs to by frame
                mid = (int(r.get("start_frame", 0)) + int(r.get("end_frame", 0))) // 2
                for seg_idx in range(n_segments):
                    if boundaries[seg_idx] <= mid < boundaries[seg_idx + 1]:
                        raw_reaches_by_seg.setdefault(seg_idx + 1, []).append(r)
                        break
        # Nested format
        elif isinstance(self._reaches_data.get("segments"), list):
            for seg in self._reaches_data["segments"]:
                sn = seg.get("segment_num")
                if sn is not None:
                    raw_reaches_by_seg[int(sn)] = seg.get("reaches", [])

        for seg_idx in range(n_segments):
            seg_num = seg_idx + 1
            start_frame = boundaries[seg_idx]
            end_frame = boundaries[seg_idx + 1] - 1

            # Pellet number: segment 1 = NO PELLET (boundary), last = NO PELLET
            # segment 2 = pellet 1, segment 3 = pellet 2, ...
            is_boundary = (seg_num == 1 or seg_num == n_segments)
            pellet_num = None if is_boundary else seg_num - 1

            outcome_info = outcomes_by_seg.get(seg_num, {})
            outcome = outcome_info.get("outcome")
            interaction_frame = outcome_info.get("interaction_frame")
            flagged = bool(outcome_info.get("flagged_for_review", False))

            # Find the causal reach from assignments
            causal_reach = None
            assigned_reaches = assignments_by_seg.get(seg_num, [])
            for ar in assigned_reaches:
                if ar.get("is_causal"):
                    causal_reach = {
                        "start": int(ar["start_frame"]),
                        "end": int(ar["end_frame"]),
                        "reach_id": ar.get("reach_id"),
                        "label": ar.get("label"),
                    }
                    break

            # All reaches in this segment (for the reach picker)
            all_reaches = []
            for r in raw_reaches_by_seg.get(seg_num, []):
                all_reaches.append({
                    "start": int(r.get("start_frame", 0)),
                    "end": int(r.get("end_frame", 0)),
                    "reach_id": r.get("reach_id"),
                })

            self._segment_list.append({
                "segment_num": seg_num,
                "pellet_num": pellet_num,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "is_boundary": is_boundary,
                "outcome": outcome,
                "interaction_frame": interaction_frame,
                "flagged": flagged,
                "causal_reach": causal_reach,
                "all_reaches": all_reaches,
            })

    # ===================================================================
    # Segment display
    # ===================================================================

    def _show_current_segment(self):
        """Update the question panel for the current segment."""
        if not self._segment_list:
            return

        seg = self._segment_list[self._current_seg_idx]
        total = len(self._segment_list)

        self._seg_label.setText(
            f"Segment: {self._current_seg_idx + 1} / {total}  "
            f"(seg #{seg['segment_num']}, "
            f"frames {seg['start_frame']}-{seg['end_frame']})"
        )

        # Update nav button state
        self._prev_seg_btn.setEnabled(self._current_seg_idx > 0)
        self._next_seg_btn.setEnabled(self._current_seg_idx < total - 1)

        # Navigate viewer to the segment's frame range
        self._navigate_to_segment(seg)

        # Build question panel
        self._build_question_panel(seg)

        # Pre-populate from saved record if re-visiting
        if seg["segment_num"] in self._review_records:
            self._restore_answers(seg["segment_num"])

    def _navigate_to_segment(self, seg: Dict[str, Any]):
        """Jump the viewer to show the segment's relevant frames."""
        # For touched segments with a causal reach, center on the reach
        # with prev/next reach context
        if seg.get("causal_reach") and not seg["is_boundary"]:
            cr = seg["causal_reach"]
            # Show from 30 frames before reach start to 30 after reach end
            target = max(0, cr["start"] - 30)
        else:
            target = seg["start_frame"]

        target = max(0, min(self.n_frames - 1, target))
        try:
            self.viewer.dims.set_current_step(0, target)
        except Exception:
            pass

    def _build_question_panel(self, seg: Dict[str, Any]):
        """Build the editable question panel for one segment."""
        # Clear existing questions
        while self._questions_layout.count():
            child = self._questions_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # --- Segment info banner ---
        banner = QFrame()
        banner.setStyleSheet(
            "QFrame { background: #1a3a5c; border: 1px solid #2a5a8c; padding: 6px; }"
        )
        banner_layout = QVBoxLayout()
        banner_layout.setContentsMargins(6, 4, 6, 4)
        banner.setLayout(banner_layout)

        seg_info = (
            f"<b>Segment {seg['segment_num']}</b> "
            f"(frames {seg['start_frame']}-{seg['end_frame']})"
        )
        if seg["is_boundary"]:
            seg_info += "  --  <i>BOUNDARY (no pellet)</i>"
        elif seg["pellet_num"] is not None:
            seg_info += f"  --  Pellet #{seg['pellet_num']}"

        if seg.get("outcome"):
            seg_info += f"  --  Algo outcome: <b>{seg['outcome']}</b>"

        if seg.get("flagged"):
            seg_info += "  --  <span style='color: #ffa500;'>TRIAGED</span>"

        banner_layout.addWidget(QLabel(seg_info))

        if seg.get("causal_reach"):
            cr = seg["causal_reach"]
            cr_info = (
                f"Algo causal reach: frames {cr['start']}-{cr['end']}"
                f"  (label: {cr.get('label', '?')})"
            )
            cr_label = QLabel(cr_info)
            cr_label.setStyleSheet("color: #8ac; font-size: 11px;")
            banner_layout.addWidget(cr_label)

        self._questions_layout.addWidget(banner)

        # --- Boundary segments: no causal questions ---
        if seg["is_boundary"]:
            no_q = QLabel(
                "<i>Boundary segment (no pellet) -- no causal questions. "
                "Advance to next segment.</i>"
            )
            no_q.setWordWrap(True)
            no_q.setStyleSheet("color: #888; padding: 8px;")
            self._questions_layout.addWidget(no_q)
            self._questions_layout.addStretch()
            return

        # Store question widget references for answer collection
        self._q_widgets: Dict[str, Any] = {}

        is_touched = seg.get("outcome") in (
            "displaced_sa", "displaced_outside", "retrieved", "abnormal",
            "abnormal_exception",
        )

        # --- Q1: Is this pellet #N? ---
        q1 = self._make_yes_no_question(
            f"Q1: Is this pellet #{seg['pellet_num']}?",
            "is_pellet",
            correction_widget_factory=lambda: self._make_pellet_correction(seg),
        )
        self._questions_layout.addWidget(q1)

        if is_touched:
            # --- Q2: Is the causal reach correct? ---
            cr = seg.get("causal_reach")
            if cr:
                q2_text = (
                    f"Q2: Is the reach starting at frame {cr['start']} "
                    f"the causal reach?"
                )
            else:
                q2_text = "Q2: Is the causal reach correctly identified? (none detected)"
            q2 = self._make_yes_no_question(
                q2_text, "is_causal",
                correction_widget_factory=lambda: self._make_reach_picker(seg),
            )
            self._questions_layout.addWidget(q2)

            # --- Q3: End frame correct? ---
            if cr:
                q3 = self._make_yes_no_question(
                    f"Q3: Did that reach end at frame {cr['end']}?",
                    "end_correct",
                    correction_widget_factory=lambda: self._make_frame_setter("Correct end frame:", cr["end"]),
                )
                self._questions_layout.addWidget(q3)

            # --- Q4: Outcome correct? ---
            q4 = self._make_yes_no_question(
                f"Q4: Did that reach cause outcome '{seg.get('outcome', '?')}'?",
                "outcome_correct",
                correction_widget_factory=lambda: self._make_outcome_picker(seg),
            )
            self._questions_layout.addWidget(q4)

        else:
            # Untouched segment
            q_miss = self._make_yes_no_question(
                "Q2: Are all reaches misses (pellet on pillar at segment end)?",
                "all_miss",
                correction_widget_factory=lambda: self._make_untouched_correction(seg),
            )
            self._questions_layout.addWidget(q_miss)

        # --- Notes ---
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout()
        notes_group.setLayout(notes_layout)
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText("Optional free-text notes for this segment...")
        self._notes_edit.setMaximumHeight(60)
        notes_layout.addWidget(self._notes_edit)
        self._questions_layout.addWidget(notes_group)

        self._questions_layout.addStretch()

    # ===================================================================
    # Question widgets
    # ===================================================================

    def _make_yes_no_question(
        self,
        question_text: str,
        key: str,
        correction_widget_factory=None,
    ) -> QGroupBox:
        """Create a Yes/No toggle question with optional inline correction."""
        group = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        group.setLayout(layout)

        q_label = QLabel(question_text)
        q_label.setWordWrap(True)
        q_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(q_label)

        btn_row = QHBoxLayout()
        yes_btn = QPushButton("Yes")
        yes_btn.setCheckable(True)
        yes_btn.setMaximumWidth(60)
        yes_btn.setStyleSheet(
            "QPushButton:checked { background: #28a745; color: white; font-weight: bold; }"
        )
        no_btn = QPushButton("No")
        no_btn.setCheckable(True)
        no_btn.setMaximumWidth(60)
        no_btn.setStyleSheet(
            "QPushButton:checked { background: #dc3545; color: white; font-weight: bold; }"
        )

        btn_group = QButtonGroup(group)
        btn_group.setExclusive(True)
        btn_group.addButton(yes_btn, 1)
        btn_group.addButton(no_btn, 0)

        btn_row.addWidget(yes_btn)
        btn_row.addWidget(no_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Correction widget (hidden by default, shown on No)
        correction_container = QWidget()
        correction_container.setVisible(False)
        correction_layout = QVBoxLayout()
        correction_layout.setContentsMargins(10, 4, 4, 4)
        correction_container.setLayout(correction_layout)

        if correction_widget_factory is not None:
            correction_content = correction_widget_factory()
            if correction_content is not None:
                correction_layout.addWidget(correction_content)

        layout.addWidget(correction_container)

        # Toggle correction visibility based on Yes/No
        def on_toggle(btn_id):
            correction_container.setVisible(btn_id == 0)  # Show on No

        btn_group.idClicked.connect(on_toggle)

        # Store references
        self._q_widgets[key] = {
            "btn_group": btn_group,
            "yes_btn": yes_btn,
            "no_btn": no_btn,
            "correction_container": correction_container,
        }

        return group

    def _make_pellet_correction(self, seg: Dict) -> QWidget:
        """Inline correction for wrong pellet number."""
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)

        layout.addWidget(QLabel("Correction options:"))

        # Phantom marker
        phantom_check = QCheckBox("Mark as phantom (no real pellet)")
        layout.addWidget(phantom_check)
        self._q_widgets.setdefault("_pellet_correction", {})["phantom"] = phantom_check

        # Renumber
        renum_row = QHBoxLayout()
        renum_row.addWidget(QLabel("Correct pellet #:"))
        renum_spin = QSpinBox()
        renum_spin.setRange(0, 30)
        renum_spin.setValue(seg.get("pellet_num") or 0)
        renum_spin.setMaximumWidth(60)
        renum_row.addWidget(renum_spin)
        renum_row.addStretch()
        layout.addLayout(renum_row)
        self._q_widgets.setdefault("_pellet_correction", {})["renumber"] = renum_spin

        # Merge/split stubs
        merge_label = QLabel(
            "<i>Merge/split boundary editing: TODO -- use the "
            "Boundaries tab in the Review Tool for now.</i>"
        )
        merge_label.setWordWrap(True)
        merge_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(merge_label)

        return w

    def _make_reach_picker(self, seg: Dict) -> QWidget:
        """Inline reach picker: choose from detected reaches or draw a new one."""
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)

        layout.addWidget(QLabel("Pick the correct causal reach:"))

        # List detected reaches
        all_reaches = seg.get("all_reaches", [])
        reach_group = QButtonGroup(w)
        reach_group.setExclusive(True)

        for i, r in enumerate(all_reaches):
            rb = QRadioButton(
                f"Reach {r.get('reach_id', i)}: "
                f"frames {r['start']}-{r['end']}"
            )
            rb.setProperty("reach_data", r)
            reach_group.addButton(rb, i)
            layout.addWidget(rb)

        # Option: undetected reach (manual entry)
        manual_rb = QRadioButton("Undetected reach (enter frames manually):")
        reach_group.addButton(manual_rb, len(all_reaches))
        layout.addWidget(manual_rb)

        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Start:"))
        start_spin = QSpinBox()
        start_spin.setRange(0, self.n_frames - 1)
        start_spin.setValue(seg["start_frame"])
        start_spin.setMaximumWidth(80)
        manual_row.addWidget(start_spin)
        manual_row.addWidget(QLabel("End:"))
        end_spin = QSpinBox()
        end_spin.setRange(0, self.n_frames - 1)
        end_spin.setValue(seg["end_frame"])
        end_spin.setMaximumWidth(80)
        manual_row.addWidget(end_spin)
        manual_row.addStretch()
        layout.addLayout(manual_row)

        # Jump button for each reach
        jump_row = QHBoxLayout()
        jump_btn = QPushButton("Jump to selected reach")
        jump_btn.clicked.connect(
            lambda: self._jump_to_selected_reach(reach_group, all_reaches, start_spin)
        )
        jump_row.addWidget(jump_btn)
        jump_row.addStretch()
        layout.addLayout(jump_row)

        self._q_widgets["_reach_picker"] = {
            "reach_group": reach_group,
            "all_reaches": all_reaches,
            "start_spin": start_spin,
            "end_spin": end_spin,
        }

        return w

    def _jump_to_selected_reach(self, group, reaches, start_spin):
        """Jump viewer to the selected reach's start frame."""
        checked_id = group.checkedId()
        if checked_id < 0:
            return
        if checked_id < len(reaches):
            frame = reaches[checked_id]["start"]
        else:
            frame = start_spin.value()
        frame = max(0, min(self.n_frames - 1, frame))
        try:
            self.viewer.dims.set_current_step(0, frame)
        except Exception:
            pass

    def _make_frame_setter(self, label_text: str, default: int) -> QWidget:
        """Inline frame setter for correcting an end frame."""
        w = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)

        layout.addWidget(QLabel(label_text))
        spin = QSpinBox()
        spin.setRange(0, self.n_frames - 1)
        spin.setValue(default)
        spin.setMaximumWidth(80)
        layout.addWidget(spin)

        use_current_btn = QPushButton("Use current frame")
        use_current_btn.clicked.connect(
            lambda: spin.setValue(int(self.viewer.dims.current_step[0]))
        )
        layout.addWidget(use_current_btn)
        layout.addStretch()

        self._q_widgets["_end_frame_spin"] = spin
        return w

    def _make_outcome_picker(self, seg: Dict) -> QWidget:
        """Inline outcome picker dropdown."""
        w = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)

        layout.addWidget(QLabel("Correct outcome:"))
        combo = QComboBox()
        combo.addItems(OUTCOME_CHOICES)
        # Pre-select current algo outcome if it matches
        current = seg.get("outcome", "")
        if current in OUTCOME_CHOICES:
            combo.setCurrentText(current)
        layout.addWidget(combo)
        layout.addStretch()

        self._q_widgets["_outcome_combo"] = combo
        return w

    def _make_untouched_correction(self, seg: Dict) -> QWidget:
        """Correction widget for untouched-was-wrong: identify the causal reach + outcome."""
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)

        layout.addWidget(QLabel(
            "This segment was called untouched but it's not. "
            "Identify the causal reach and outcome:"
        ))

        # Reach picker (reuse)
        reach_widget = self._make_reach_picker(seg)
        layout.addWidget(reach_widget)

        # Outcome picker
        outcome_row = QHBoxLayout()
        outcome_row.addWidget(QLabel("Correct outcome:"))
        combo = QComboBox()
        combo.addItems([o for o in OUTCOME_CHOICES if o != "untouched"])
        outcome_row.addWidget(combo)
        outcome_row.addStretch()
        layout.addLayout(outcome_row)

        self._q_widgets["_untouched_outcome_combo"] = combo

        return w

    # ===================================================================
    # Answer collection
    # ===================================================================

    def _collect_answers(self) -> Optional[Dict[str, Any]]:
        """Collect current segment's answers from the question widgets.

        Returns None if the segment has no questions (boundary segment).
        """
        if not self._segment_list:
            return None

        seg = self._segment_list[self._current_seg_idx]
        if seg["is_boundary"]:
            # Boundary segments auto-agree (no questions)
            return build_segment_record(
                segment_num=seg["segment_num"],
                pellet_num=None,
                algo_outcome=seg.get("outcome"),
                algo_causal_reach=seg.get("causal_reach"),
                algo_interaction_frame=seg.get("interaction_frame"),
                human_outcome=seg.get("outcome"),
                human_causal_reach=None,
                is_phantom=False,
                agreed=True,
                answers={"is_boundary": True},
                notes="",
            )

        answers: Dict[str, Any] = {}
        human_outcome = seg.get("outcome")
        human_causal_reach = seg.get("causal_reach")
        is_phantom = False
        agreed = True

        # Q1: Is pellet correct?
        q1_ans = self._get_toggle_answer("is_pellet")
        answers["is_pellet"] = q1_ans
        if q1_ans is False:
            agreed = False
            pc = self._q_widgets.get("_pellet_correction", {})
            phantom_check = pc.get("phantom")
            renum_spin = pc.get("renumber")
            if phantom_check and phantom_check.isChecked():
                is_phantom = True
            # If renumbered, we note it but don't change pellet_num in the
            # segment list (the review record captures the correction)

        is_touched = seg.get("outcome") in (
            "displaced_sa", "displaced_outside", "retrieved", "abnormal",
            "abnormal_exception",
        )

        if is_touched:
            # Q2: causal reach correct?
            q2_ans = self._get_toggle_answer("is_causal")
            answers["is_causal"] = q2_ans
            if q2_ans is False:
                agreed = False
                rp = self._q_widgets.get("_reach_picker", {})
                if rp:
                    checked_id = rp["reach_group"].checkedId()
                    reaches = rp["all_reaches"]
                    if 0 <= checked_id < len(reaches):
                        r = reaches[checked_id]
                        human_causal_reach = {"start": r["start"], "end": r["end"]}
                    elif checked_id == len(reaches):
                        human_causal_reach = {
                            "start": rp["start_spin"].value(),
                            "end": rp["end_spin"].value(),
                        }

            # Q3: end frame correct?
            q3_ans = self._get_toggle_answer("end_correct")
            answers["end_correct"] = q3_ans
            if q3_ans is False:
                agreed = False
                spin = self._q_widgets.get("_end_frame_spin")
                if spin and human_causal_reach:
                    human_causal_reach = dict(human_causal_reach)
                    human_causal_reach["end"] = spin.value()

            # Q4: outcome correct?
            q4_ans = self._get_toggle_answer("outcome_correct")
            answers["outcome_correct"] = q4_ans
            if q4_ans is False:
                agreed = False
                combo = self._q_widgets.get("_outcome_combo")
                if combo:
                    human_outcome = combo.currentText()
        else:
            # Untouched segment
            q_miss_ans = self._get_toggle_answer("all_miss")
            answers["all_miss"] = q_miss_ans
            if q_miss_ans is False:
                agreed = False
                # Reach picker + outcome for untouched-was-wrong
                rp = self._q_widgets.get("_reach_picker", {})
                if rp:
                    checked_id = rp["reach_group"].checkedId()
                    reaches = rp["all_reaches"]
                    if 0 <= checked_id < len(reaches):
                        r = reaches[checked_id]
                        human_causal_reach = {"start": r["start"], "end": r["end"]}
                    elif checked_id == len(reaches):
                        human_causal_reach = {
                            "start": rp["start_spin"].value(),
                            "end": rp["end_spin"].value(),
                        }
                combo = self._q_widgets.get("_untouched_outcome_combo")
                if combo:
                    human_outcome = combo.currentText()

        notes = self._notes_edit.toPlainText().strip() if hasattr(self, '_notes_edit') else ""

        return build_segment_record(
            segment_num=seg["segment_num"],
            pellet_num=seg["pellet_num"],
            algo_outcome=seg.get("outcome"),
            algo_causal_reach=seg.get("causal_reach"),
            algo_interaction_frame=seg.get("interaction_frame"),
            human_outcome=human_outcome,
            human_causal_reach=human_causal_reach,
            is_phantom=is_phantom,
            agreed=agreed,
            answers=answers,
            notes=notes,
        )

    def _get_toggle_answer(self, key: str) -> Optional[bool]:
        """Get Yes/No/unanswered from a toggle question."""
        q = self._q_widgets.get(key)
        if not q:
            return None
        checked = q["btn_group"].checkedId()
        if checked < 0:
            return None  # unanswered
        return checked == 1  # 1 = Yes, 0 = No

    def _restore_answers(self, segment_num: int):
        """Pre-populate question widgets from a saved review record."""
        rec = self._review_records.get(segment_num)
        if not rec:
            return
        answers = rec.get("answers", {})

        for key in ["is_pellet", "is_causal", "end_correct", "outcome_correct", "all_miss"]:
            val = answers.get(key)
            if val is None:
                continue
            q = self._q_widgets.get(key)
            if not q:
                continue
            if val:
                q["yes_btn"].setChecked(True)
            else:
                q["no_btn"].setChecked(True)
            # Trigger correction visibility
            q["correction_container"].setVisible(not val)

        # Restore notes
        notes = rec.get("notes", "")
        if hasattr(self, '_notes_edit') and notes:
            self._notes_edit.setPlainText(notes)

    # ===================================================================
    # Save
    # ===================================================================

    def _save_segment_and_advance(self):
        """Save current segment's answers and advance."""
        record = self._collect_answers()
        if record is not None:
            seg_num = record["segment_num"]
            self._review_records[seg_num] = record
            show_info(f"Segment {seg_num} recorded")

        if self._current_seg_idx < len(self._segment_list) - 1:
            self._current_seg_idx += 1
            self._show_current_segment()
        else:
            show_info("Last segment reached. Use 'Save Review' to write the file.")

    def _save_review(self):
        """Save all accumulated review records to the per-video file + index."""
        if not self.video_path or not self._video_stem:
            show_warning("No video loaded")
            return

        # Collect current segment's answers if not already saved
        record = self._collect_answers()
        if record is not None:
            self._review_records[record["segment_num"]] = record

        if not self._review_records:
            show_warning("No segments reviewed yet")
            return

        # Build the segment list in order
        segments = []
        for seg in self._segment_list:
            sn = seg["segment_num"]
            if sn in self._review_records:
                segments.append(self._review_records[sn])
            else:
                # Unreviewed segments get a placeholder
                segments.append(build_segment_record(
                    segment_num=sn,
                    pellet_num=seg["pellet_num"],
                    algo_outcome=seg.get("outcome"),
                    algo_causal_reach=seg.get("causal_reach"),
                    algo_interaction_frame=seg.get("interaction_frame"),
                    human_outcome=None,
                    human_causal_reach=None,
                    is_phantom=False,
                    agreed=False,
                    answers={"reviewed": False},
                    notes="",
                ))

        provenance = collect_provenance(self.video_path.parent, self._video_stem)
        reviewer = _get_username()
        timestamp = _get_timestamp()

        # Save per-video file
        out_path = save_causal_review(
            video_stem=self._video_stem,
            output_dir=self.video_path.parent,
            segments=segments,
            provenance=provenance,
            reviewer=reviewer,
        )

        # Update corpus index
        try:
            update_corpus_index(
                video_stem=self._video_stem,
                review_file_path=out_path,
                segments=[s for s in segments if s.get("answers", {}).get("reviewed", True) is not False],
                reviewer=reviewer,
                reviewed_at=timestamp,
            )
        except Exception as e:
            print(f"Warning: could not update corpus index: {e}")

        n_reviewed = sum(
            1 for s in segments
            if s.get("answers", {}).get("reviewed", True) is not False
        )
        show_info(
            f"Saved causal review: {n_reviewed}/{len(segments)} segments "
            f"-> {out_path.name}"
        )
        self._status_label.setText(f"Saved: {out_path.name}")
        self.data_saved.emit(out_path)

    # ===================================================================
    # Navigation
    # ===================================================================

    def _prev_segment(self):
        # Save current before leaving
        record = self._collect_answers()
        if record is not None:
            self._review_records[record["segment_num"]] = record
        if self._current_seg_idx > 0:
            self._current_seg_idx -= 1
            self._show_current_segment()

    def _next_segment(self):
        record = self._collect_answers()
        if record is not None:
            self._review_records[record["segment_num"]] = record
        if self._current_seg_idx < len(self._segment_list) - 1:
            self._current_seg_idx += 1
            self._show_current_segment()

    def _load_whole_segment(self):
        """Jump viewer to show the entire current segment."""
        if not self._segment_list:
            return
        seg = self._segment_list[self._current_seg_idx]
        try:
            self.viewer.dims.set_current_step(0, seg["start_frame"])
        except Exception:
            pass

    # ===================================================================
    # Playback (reused from GT widget pattern)
    # ===================================================================

    def _play_forward(self):
        self.playback_direction = 1
        self._start_playback()

    def _play_reverse(self):
        self.playback_direction = -1
        self._start_playback()

    def _stop_play(self):
        self.is_playing = False
        self.playback_timer.stop()
        self._play_btn.setText("Play")
        self._play_rev_btn.setText("Rev")

    def _start_playback(self):
        if self.is_playing:
            self._stop_play()
            return
        self.is_playing = True
        interval = max(1, int(1000 / (self.fps * self.playback_speed)))
        self.playback_timer.start(interval)
        if self.playback_direction == 1:
            self._play_btn.setText("||")
        else:
            self._play_rev_btn.setText("||")

    def _playback_step(self):
        current = self.viewer.dims.current_step[0]
        skip = max(1, int(self.playback_speed))
        new_frame = current + (skip * self.playback_direction)
        if 0 <= new_frame < self.n_frames:
            self.viewer.dims.set_current_step(0, new_frame)
        else:
            self._stop_play()

    def _set_speed(self, speed: float):
        self.playback_speed = speed
        for s, btn in self._speed_buttons.items():
            btn.setChecked(s == speed)
        if self.is_playing:
            interval = max(1, int(1000 / (self.fps * self.playback_speed)))
            self.playback_timer.stop()
            self.playback_timer.start(interval)

    def _jump_frames(self, delta: int):
        if self.n_frames == 0:
            return
        current = self.viewer.dims.current_step[0]
        new_frame = max(0, min(self.n_frames - 1, current + delta))
        self.viewer.dims.set_current_step(0, new_frame)

    def _on_frame_change(self, event=None):
        if self.n_frames == 0:
            return
        frame_idx = self.viewer.dims.current_step[0]
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self._frame_label.setText(f"Frame: {int(frame_idx)} / {self.n_frames}")
        self._time_label.setText(f"Time: {mins}:{secs:05.2f}")

    def _enable_controls(self, enabled: bool):
        self._play_btn.setEnabled(enabled)
        self._play_rev_btn.setEnabled(enabled)
        self._stop_btn.setEnabled(enabled)
        self._prev_seg_btn.setEnabled(enabled)
        self._next_seg_btn.setEnabled(enabled)
        self._load_whole_seg_btn.setEnabled(enabled)
        self._save_btn.setEnabled(enabled)
        self._save_advance_btn.setEnabled(enabled)
