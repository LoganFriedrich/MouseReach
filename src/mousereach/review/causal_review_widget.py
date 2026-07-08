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
    load_causal_review,
    update_corpus_index,
    flag_session,
    is_session_flagged,
    session_key,
    find_gt,
    has_gt,
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


class _LazyVideo:
    """Array-like view of a video that decodes frames on demand.

    Presents ``shape (n_frames, H, W, 3)`` uint8 to napari. ``__getitem__`` over
    the time axis reads (and caches) individual frames directly -- NO dask graph,
    so slicing is O(1) instead of O(n_frames) (a 37k-frame dask stack makes
    per-frame slicing pathologically slow). Sequential reads skip the per-frame
    seek so Play runs at decode speed; a bounded cache serves re-visits instantly.
    """

    def __init__(self, path, n_frames, h, w, max_cache=1500):
        import threading
        self._path = str(path)
        self.shape = (int(n_frames), int(h), int(w), 3)
        self.dtype = np.dtype(np.uint8)
        self.ndim = 4
        self._h, self._w = int(h), int(w)
        self._cache: Dict[int, Any] = {}
        self._order: List[int] = []
        self._max = int(max_cache)
        self._lock = threading.Lock()
        self._tl = threading.local()

    def __len__(self):
        return self.shape[0]

    def _frame(self, i: int):
        import cv2
        i = int(i)
        if i < 0:
            i += self.shape[0]
        with self._lock:
            a = self._cache.get(i)
        if a is not None:
            return a
        cap = getattr(self._tl, "cap", None)
        pos = getattr(self._tl, "pos", -10)
        if cap is None:
            cap = self._tl.cap = cv2.VideoCapture(self._path)
            pos = -10
        if i != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # seek only when NOT sequential
        ok, frame = cap.read()
        self._tl.pos = i
        rgb = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok
               else np.zeros((self._h, self._w, 3), np.uint8))
        with self._lock:
            self._cache[i] = rgb
            self._order.append(i)
            if len(self._order) > self._max:
                self._cache.pop(self._order.pop(0), None)
        return rgb

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t = key[0]
        rest = key[1:]
        if isinstance(t, (int, np.integer)):
            frame = self._frame(int(t))          # (H, W, 3)
            return frame[rest] if rest else frame
        if isinstance(t, slice):
            idxs = range(*t.indices(self.shape[0]))
        else:
            idxs = [int(x) for x in np.atleast_1d(t)]
        stacked = np.stack([self._frame(int(i)) for i in idxs], axis=0)
        return stacked[(slice(None),) + rest] if rest else stacked

    def __array__(self, dtype=None):
        # Last-resort full materialization; napari shouldn't call this for
        # display, but keep it correct just in case.
        arr = np.stack([self._frame(i) for i in range(self.shape[0])], axis=0)
        return arr.astype(dtype) if dtype is not None else arr


class _NotesTextEdit(QTextEdit):
    """QTextEdit that consumes its own keystrokes, so typing notes does not leak
    through to napari's single-key shortcuts (which otherwise fire mid-word)."""

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        event.accept()


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
        self._setup_keybindings()

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

        self._save_next_video_btn = QPushButton("Save Review + Next Video")
        self._save_next_video_btn.setStyleSheet("background: #16405a; color: white;")
        self._save_next_video_btn.setToolTip(
            "Save this video's review and load the next video in the Pending queue."
        )
        self._save_next_video_btn.clicked.connect(self._save_and_next_video)
        self._save_next_video_btn.setEnabled(False)
        footer_layout.addWidget(self._save_next_video_btn)

        self._flag_session_btn = QPushButton("Flag Session (needs review)")
        self._flag_session_btn.setStyleSheet("background: #5a3a16; color: white;")
        self._flag_session_btn.setToolTip(
            "Mark this mouse+day session (every P# of this date) as "
            "must-be-human-reviewed -- e.g. a cage artifact affecting every "
            "video shot that day."
        )
        self._flag_session_btn.clicked.connect(self._flag_session)
        self._flag_session_btn.setEnabled(False)
        footer_layout.addWidget(self._flag_session_btn)

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

    def load_from_manifest(self, manifest: Dict[str, Any], bundle_dir: Path) -> None:
        """Load a copy-free review bundle.

        The bundle dir holds only the four algo JSONs + manifest.json; the mp4
        and DLC pose are loaded from the manifest's canonical Y: paths (they live
        in different dirs and are never copied). See mousereach.review.staging
        for how bundles are produced.
        """
        self._manifest = dict(manifest)
        self._bundle_dir = Path(bundle_dir)
        self._load_video(Path(manifest["canonical_video_path"]))

    # frames of context padded around a reach when windowing a segment
    WINDOW_PAD = 45

    def _load_video(self, video_path: Path):
        """One-time setup: a lazy full-video layer + overlays + algo data.

        The video is added ONCE as a decode-on-demand (dask) image layer, so its
        extent is stable and we NEVER swap image layers on navigation. Repeated
        layer add/remove or in-place data swaps churn the vispy/OpenGL renderer
        and crash it here (OpenGL fault / evented-model stack overflow). With a
        static full-video layer, navigation just moves the playhead to the
        segment's reach -- nothing to churn.
        """
        import cv2

        self.video_path = video_path
        self._video_label.setText(f"Loading: {video_path.name}")
        self._progress.setVisible(True)
        self._progress.setValue(0)

        try:
            # Resolve preview / scale
            video_stem = video_path.stem.replace("_preview", "")
            if "DLC" in video_stem:
                video_stem = video_stem.split("DLC")[0].rstrip("_")
            self._video_stem = video_stem

            preview_path = video_path.parent / f"{video_stem}_preview.mp4"
            if "_preview" not in video_path.stem and preview_path.exists():
                self._actual_video = preview_path
                self.scale_factor = 0.75
            else:
                self._actual_video = video_path
                self.scale_factor = 1.0

            cap = cv2.VideoCapture(str(self._actual_video))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            self.frame_offset = 0
            self.frame_window_end = self.n_frames - 1
            self._progress.setValue(20)

            # Lazy, decode-on-demand video layer -- added ONCE, never swapped.
            lazy = self._make_lazy_video(self._actual_video, self.n_frames, h, w)
            if self.video_layer is not None and self.video_layer in self.viewer.layers:
                try:
                    self.viewer.layers.remove(self.video_layer)
                except Exception:
                    pass
            self.video_layer = self.viewer.add_image(
                lazy, name=video_path.stem, rgb=True)
            self._progress.setValue(40)

            # DLC overlay (all frames, once) + pillar overlay (per-frame callback)
            self._load_dlc_data()
            self._add_dlc_points_layer()
            self._add_pillar_shapes_layer()
            self._progress.setValue(70)

            # Algo data + segment list
            self._load_algo_data()
            # Restore any prior manual re-segmentation before building segments.
            self._manual_boundaries = self._peek_manual_boundaries()
            self._build_segment_list()
            self._progress.setValue(85)

            self._enable_controls(True)
            # Resume: load any prior review, restore answers, skip to first
            # segment that is unscored or whose algo output changed since review.
            self._load_saved_review()
            self._show_current_segment()

            self._video_label.setText(
                f"Loaded: {video_path.name}  --  {self.n_frames} frames, "
                f"{len(self._segment_list)} segments (lazy)"
            )
            self._progress.setValue(100)

        except Exception as e:
            show_error(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._progress.setVisible(False)

    def _make_lazy_video(self, path: Path, n_frames: int, h: int, w: int):
        """Return a decode-on-demand array-like for the whole video.

        Uses a custom __getitem__ array (_LazyVideo) rather than a dask stack:
        dask's stack-of-37k-delayed graph makes per-frame slicing O(n_frames)
        and freezes scrubbing/playback. _LazyVideo slices in O(1) with a frame
        cache + sequential-read fast path.
        """
        video = _LazyVideo(path, n_frames, h, w)
        self._frame_cache = video._cache
        return video

    def _relevant_reach(self, seg: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """The reach the reviewer should open ON for a segment.

        Priority (this is the whole point of the tool -- always land on the
        evaluated element, never on a segment boundary or arbitrary frame):
          1. YOUR corrected causal reach, if you've reviewed this segment and set
             one -- once you correct it, THAT is the evaluated element.
          2. the algo's causal reach (touched)
          3. the LAST reach in the segment (miss / triaged)
        None only if the segment genuinely has no reaches.
        """
        rec = self._review_records.get(seg["segment_num"]) if getattr(self, "_review_records", None) else None
        if rec:
            hc = (rec.get("human") or {}).get("causal_reach")
            if hc and hc.get("start") is not None and hc.get("end") is not None:
                return {"start": int(hc["start"]), "end": int(hc["end"])}
        cr = seg.get("causal_reach")
        if cr:
            return {"start": int(cr["start"]), "end": int(cr["end"])}
        reaches = seg.get("all_reaches") or []
        if reaches:
            last = max(reaches, key=lambda r: r.get("start", 0))
            return {"start": int(last["start"]), "end": int(last["end"])}
        return None

    def _compute_segment_window(self, seg: Dict[str, Any]) -> Tuple[int, int]:
        """Absolute frame range to frame for a segment: the RELEVANT REACH plus a
        few frames each side, clamped to the segment. Reach-centric always --
        touched -> causal reach; miss/triaged -> last reach.
        """
        pad = self.WINDOW_PAD
        sf, ef = int(seg["start_frame"]), int(seg["end_frame"])
        reach = self._relevant_reach(seg)
        if reach is None:
            # No reaches at all -> don't force watching the whole segment. Frame
            # the segment-CHANGING move (the tray cycle at the segment end): open
            # ~30 frames before it (the full video stays scrubbable to watch it).
            lead = 30
            return (max(sf, ef - lead - pad), min(self.n_frames - 1, ef + 15))
        return (max(sf, min(reach["start"] - pad, ef)),
                max(sf, min(reach["end"] + pad, ef)))

    def _load_frame_window(self, frame_range: Tuple[int, int]):
        """Frame the current segment: record the window bounds so the playhead
        and playback stay near the relevant reach. No layer manipulation -- the
        lazy video layer is static, so navigation can't churn/crash the renderer.
        The actual playhead jump happens in _navigate_to_segment.
        """
        fr_start = max(0, int(frame_range[0]))
        fr_end = min(self.n_frames - 1, int(frame_range[1]))
        if fr_end < fr_start:
            fr_end = fr_start
        self.frame_offset = fr_start
        self.frame_window_end = fr_end

    def _load_dlc_data(self):
        """Load DLC H5 for overlays.

        In manifest (bundle) mode the pose h5 is loaded from the manifest's
        canonical path (it lives in a different dir than the mp4); otherwise we
        glob the video's parent dir.
        """
        if not self.video_path:
            return
        import pandas as pd
        candidates = []
        manifest = getattr(self, "_manifest", None)
        if manifest and manifest.get("canonical_dlc_h5_path"):
            candidates.append(Path(manifest["canonical_dlc_h5_path"]))
        candidates.extend(self.video_path.parent.glob(f"{self._video_stem}*.h5"))
        for h5_path in candidates:
            try:
                df = pd.read_hdf(h5_path)
                df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
                self.dlc_df = df
                return
            except Exception:
                pass
        self.dlc_df = None

    def _add_dlc_points_layer(self):
        """Add the DLC tracking-points overlay for the WHOLE video (built once).

        Vectorized per bodypart so all ~37k frames build in well under a second.
        napari only renders the points at the current frame, so a full-video
        points layer is cheap to display and never needs rebuilding on nav.
        """
        if self.dlc_df is None:
            return
        df = self.dlc_df
        n = len(df)
        bodyparts = sorted({col[:-2] for col in df.columns if col.endswith('_x')})
        colors_base = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5]
        ]
        scale = self.scale_factor
        frames_all = np.arange(n)

        pts_chunks, col_chunks, bp_list = [], [], []
        for i, bp in enumerate(bodyparts):
            xcol, ycol, lcol = f'{bp}_x', f'{bp}_y', f'{bp}_likelihood'
            if xcol not in df.columns or ycol not in df.columns:
                continue
            xs = df[xcol].to_numpy(dtype=float)
            ys = df[ycol].to_numpy(dtype=float)
            lks = df[lcol].to_numpy(dtype=float) if lcol in df.columns else np.ones(n)
            valid = ~(np.isnan(xs) | np.isnan(ys))
            if not valid.any():
                continue
            fv = frames_all[valid]
            xv = xs[valid] * scale
            yv = ys[valid] * scale
            lv = np.clip(lks[valid], 0.0, 1.0)
            alpha = np.where(lv < 0.5, 0.05,
                             0.10 + 0.90 * (((lv - 0.5) / 0.5) ** 2))
            base = np.array(colors_base[i % len(colors_base)], dtype=float)
            m = len(fv)
            pts_chunks.append(np.column_stack([fv, yv, xv]))
            col_chunks.append(np.column_stack([np.tile(base, (m, 1)), alpha]))
            bp_list.extend([bp] * m)

        if not pts_chunks:
            self.points_layer = None
            return
        points_data = np.vstack(pts_chunks)
        point_colors = np.vstack(col_chunks)

        if self.points_layer is not None and self.points_layer in self.viewer.layers:
            try:
                self.viewer.layers.remove(self.points_layer)
            except Exception:
                pass
        self.points_layer = self.viewer.add_points(
            points_data, name='DLC Points', size=3,
            face_color=point_colors,
            features={'bp': bp_list},
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
        if getattr(self, '_suspend_pillar', False):
            return
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
        """Load segments, reaches, outcomes, and assignments JSONs.

        In manifest (bundle) mode the JSONs live in the bundle dir, not next to
        the canonical mp4.
        """
        if not self.video_path:
            return
        bundle_dir = getattr(self, "_bundle_dir", None)
        parent = Path(bundle_dir) if bundle_dir else self.video_path.parent
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

    def _review_dir(self) -> Path:
        """Where to SAVE the review: the bundle dir in manifest mode (so it
        travels with the bundle Pending->Reviewed), else next to the video.
        Loading checks both locations."""
        bd = getattr(self, "_bundle_dir", None)
        return Path(bd) if bd else self.video_path.parent

    def _algo_dir(self) -> Path:
        """Where the algo output JSONs live (the bundle dir in manifest mode)."""
        bd = getattr(self, "_bundle_dir", None)
        return Path(bd) if bd else self.video_path.parent

    def _load_saved_review(self):
        """Load a prior review (if any) so scored segments restore their answers
        and the walk RESUMES at the first unscored / algo-changed segment.

        A segment counts as already done only if it was ACTUALLY reviewed AND the
        algo's outcome for it is unchanged since. If the algo output changed, it's
        worth re-checking -- so it's treated as needing review again.
        """
        self._review_records = {}
        # Check next to the video (canonical) first, then the bundle dir, so a
        # review saved under either location is always found.
        candidates = []
        bd = getattr(self, "_bundle_dir", None)
        if bd:
            candidates.append(Path(bd))
        if self.video_path.parent not in candidates:
            candidates.append(self.video_path.parent)
        by_seg = {}
        for d in candidates:
            try:
                _, bs = load_causal_review(self._video_stem, d)
            except Exception:
                bs = {}
            if bs:
                by_seg = bs
                break
        # Keep only ACTUALLY-reviewed segments; the save writes reviewed:False
        # placeholder rows for un-scored segments, which must not count as done.
        self._review_records = {
            sn: rec for sn, rec in by_seg.items()
            if (rec.get("answers") or {}).get("reviewed") is not False
        }

        def _done(seg):
            rec = self._review_records.get(seg["segment_num"])
            if not rec:
                return False
            saved_algo = (rec.get("algo") or {}).get("outcome")
            return saved_algo == seg.get("outcome")  # algo unchanged -> still done

        idx = 0
        for i, seg in enumerate(self._segment_list):
            if not _done(seg):
                idx = i
                break
        self._current_seg_idx = idx

        if self._review_records:
            n_done = sum(1 for seg in self._segment_list if _done(seg))
            n_changed = sum(
                1 for seg in self._segment_list
                if seg["segment_num"] in self._review_records and not _done(seg))
            first = (self._segment_list[idx]["segment_num"]
                     if self._segment_list else "?")
            msg = f"Resumed: {n_done}/{len(self._segment_list)} already scored"
            if n_changed:
                msg += f"; {n_changed} to re-check (algo changed)"
            msg += f"  --  opening segment {first}"
            self._status_label.setText(msg)

    def _build_segment_list(self):
        """Build the unified segment list from segments + outcomes + assignments."""
        self._segment_list = []

        # Manual re-segmentation override: when the reviewer re-cut the video via
        # the segmentation editor (Q1 = No), use THEIR boundaries and score every
        # segment fresh (the algo's per-old-segment outcomes no longer apply).
        manual = getattr(self, "_manual_boundaries", None)

        # Extract boundaries
        boundaries = []
        if manual:
            boundaries = sorted({int(x) for x in manual})
        elif "boundaries" in self._segments_data:
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

        # Raw reaches (from the reach detector) grouped by segment. Flatten both
        # storage formats to one list first.
        raw_reaches_by_seg: Dict[int, List[Dict]] = {}
        _all_raw: List[Dict] = []
        if isinstance(self._reaches_data.get("reaches"), list):
            _all_raw = list(self._reaches_data["reaches"])            # flat (v8+)
        elif isinstance(self._reaches_data.get("segments"), list):
            for _s in self._reaches_data["segments"]:                # nested
                _all_raw.extend(_s.get("reaches", []))

        if manual or isinstance(self._reaches_data.get("reaches"), list):
            # Assign each reach to the segment whose window contains its midpoint.
            # REQUIRED after manual re-segmentation: a reach's ORIGINAL segment_num
            # no longer maps to the reviewer's new cuts, so the Q2 candidates must
            # be re-derived by frame -- otherwise the picker offers reaches from
            # other windows entirely (the nested format grouped them by old
            # segment_num). The flat format already worked this way.
            for r in _all_raw:
                mid = (int(r.get("start_frame", 0)) + int(r.get("end_frame", 0))) // 2
                for seg_idx in range(n_segments):
                    if boundaries[seg_idx] <= mid < boundaries[seg_idx + 1]:
                        raw_reaches_by_seg.setdefault(seg_idx + 1, []).append(r)
                        break
        elif isinstance(self._reaches_data.get("segments"), list):
            # Nested format on the algo's own boundaries: trust the detector's
            # per-segment grouping.
            for seg in self._reaches_data["segments"]:
                sn = seg.get("segment_num")
                if sn is not None:
                    raw_reaches_by_seg[int(sn)] = seg.get("reaches", [])

        for seg_idx in range(n_segments):
            seg_num = seg_idx + 1
            start_frame = boundaries[seg_idx]
            end_frame = boundaries[seg_idx + 1] - 1

            # The segmenter emits one segment per pellet presentation and drops
            # the pre/post no-pellet brackets, so EVERY segment is a real pellet
            # segment (the validated pipeline scores all N as pellet outcomes)
            # and gets the full review flow. In the segmenter's numbering,
            # segment N == pellet N. (If the segmenter is later fixed to emit the
            # two bracket segments, revisit this.)
            is_boundary = False
            pellet_num = seg_num

            if manual:
                # Fresh segments from the reviewer's cuts -- no algo outcome maps.
                outcome = None
                interaction_frame = None
                flagged = False
                causal_reach = None
            else:
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

            # Two independent triage axes (mutually exclusive in practice):
            #   outcome_uncertain -> algo-3 could not call the OUTCOME (== "triaged").
            #     The human must make the outcome call.  ("OUTCOME CALL" tag)
            #   reach_uncertain   -> algo-3 committed a touched outcome but algo-4
            #     pinned NO causal reach. The outcome is algo-confirmed; the human
            #     must identify the reach.  ("TRIAGED (reach)" tag)
            _TOUCHED = ("retrieved", "displaced_sa", "displaced_outside")
            outcome_uncertain = (outcome == "triaged") or flagged
            reach_uncertain = (outcome in _TOUCHED) and (causal_reach is None)

            self._segment_list.append({
                "segment_num": seg_num,
                "pellet_num": pellet_num,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "is_boundary": is_boundary,
                "outcome": outcome,
                "interaction_frame": interaction_frame,
                "flagged": flagged,
                "outcome_uncertain": outcome_uncertain,
                "reach_uncertain": reach_uncertain,
                "manual": bool(manual),
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

        # Load ONLY this segment's reach window (fast; no full-video decode)
        self._load_frame_window(self._compute_segment_window(seg))

        # Navigate playhead within the loaded window
        self._navigate_to_segment(seg)

        # Build question panel
        self._build_question_panel(seg)

        # Pre-populate from saved record if re-visiting
        if seg["segment_num"] in self._review_records:
            self._restore_answers(seg["segment_num"])

    def _navigate_to_segment(self, seg: Dict[str, Any]):
        """Open the playhead ON the relevant reach -- the causal reach for a
        touched segment, the last reach for a miss/triaged segment -- NOT on a
        segment-relative frame."""
        reach = self._relevant_reach(seg)
        if reach is not None:
            target = reach["start"]
        else:
            # no reaches -> open ~30 frames before the segment-changing move (end)
            target = int(seg["end_frame"]) - 30
        target = max(self.frame_offset, min(self.frame_window_end, target))
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

        if seg.get("outcome_uncertain"):
            seg_info += "  --  <span style='color: #ffa500;'>OUTCOME CALL</span>"
        if seg.get("reach_uncertain"):
            seg_info += "  --  <span style='color: #ffcc00;'>TRIAGED (reach)</span>"
        if seg.get("manual"):
            seg_info += "  --  <span style='color: #4af;'>MANUAL SEG</span>"

        banner_layout.addWidget(QLabel(seg_info))

        # If this segment was already reviewed, say so (answers restored below).
        prev = self._review_records.get(seg["segment_num"])
        if prev:
            agreed = (prev.get("human") or {}).get("agreed")
            tag = "agreed with algo" if agreed else "corrected"
            note = QLabel(
                f"[OK] Previously scored ({tag}) -- your answers are loaded; "
                f"edit and re-save to update.")
            note.setStyleSheet("color: #6c6; font-size: 11px;")
            note.setWordWrap(True)
            banner_layout.addWidget(note)

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
            elif seg.get("reach_uncertain"):
                q2_text = (
                    f"Q2: Algo determined the outcome is '{seg.get('outcome')}' but did NOT "
                    f"identify the causal reach (TRIAGED -- reach uncertain). Answer NO and "
                    f"pick the causal reach below (or use the ignore-window if no reach caused it)."
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

        # --- Ignore windows (abnormal non-reach events) -- on ANY non-boundary
        # segment, independent of the reach questions above ---
        self._questions_layout.addWidget(self._make_abnormal_ranges_widget(seg))

        # --- Notes ---
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout()
        notes_group.setLayout(notes_layout)
        self._notes_edit = _NotesTextEdit()
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

        # Deep correction: RE-SEGMENT the whole video. When the pellet/segment is
        # wrong because the SEGMENTATION is wrong (e.g. the segmenter fell back to
        # uniform slicing on bad tracking), phantom/renumber can't help -- the
        # reviewer has to re-cut the video. This opens a full segmentation editor.
        reseg_btn = QPushButton("Re-segment this video (open segmentation editor)")
        reseg_btn.setStyleSheet("font-weight: bold;")
        reseg_btn.clicked.connect(self._open_resegmentation_editor)
        layout.addWidget(reseg_btn)
        hint = QLabel(
            "<i>Use this when the segmentation itself is wrong. You'll mark the "
            "real segment cuts on the video, then reload and score them.</i>")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(hint)

        return w

    # ===================================================================
    # Manual re-segmentation (deep correction, opened from Q1 = No)
    # ===================================================================

    def _current_boundaries(self):
        """Active segment-cut frames from the algo segmentation data."""
        sd = getattr(self, "_segments_data", {}) or {}
        if "boundaries" in sd:
            out = []
            for b in sd["boundaries"]:
                out.append(int(b.get("frame", b.get("index", 0)))
                           if isinstance(b, dict) else int(b))
            return sorted(set(out))
        bs = []
        for s in sd.get("segments", []):
            if s.get("start_frame") is not None:
                bs.append(int(s["start_frame"]))
            if s.get("end_frame") is not None:
                bs.append(int(s["end_frame"]) + 1)
        return sorted(set(bs))

    def _peek_manual_boundaries(self):
        """Restore a prior manual re-segmentation (saved in the review doc) so the
        segments rebuild from the reviewer's cuts on reopen; else None."""
        candidates = []
        bd = getattr(self, "_bundle_dir", None)
        if bd:
            candidates.append(Path(bd))
        if self.video_path and self.video_path.parent not in candidates:
            candidates.append(self.video_path.parent)
        for d in candidates:
            try:
                doc, _ = load_causal_review(self._video_stem, d)
            except Exception:
                doc = None
            b = (doc or {}).get("manual_segmentation", {}).get("boundaries")
            if b:
                return sorted({int(x) for x in b})
        return None

    def _reseg_cur_frame(self):
        try:
            return int(self.viewer.dims.current_step[0])
        except Exception:
            return 0

    def _open_resegmentation_editor(self):
        """Full-video segmentation editor: scrub the video, drop/remove segment
        cuts at the current frame, then Apply to rebuild + re-score the segments
        from those cuts. Used when the SEGMENTATION itself is wrong."""
        seed = getattr(self, "_manual_boundaries", None) or self._current_boundaries()
        self._reseg_boundaries = sorted({int(x) for x in seed})

        while self._questions_layout.count():
            child = self._questions_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        banner = QLabel(
            "<b>Segmentation editor</b> -- the cuts below define the segments. "
            "Scrub to the START of each real pellet segment and 'Add cut'. "
            "'Apply' rebuilds and re-scores from your cuts.")
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel { background:#3a2a1a; border:1px solid #6a4a2a; padding:6px; }")
        self._questions_layout.addWidget(banner)

        self._reseg_summary = QLabel()
        self._reseg_summary.setWordWrap(True)
        self._questions_layout.addWidget(self._reseg_summary)

        def _row(*btns):
            r = QHBoxLayout()
            for b in btns:
                r.addWidget(b)
            r.addStretch()
            holder = QWidget()
            holder.setLayout(r)
            self._questions_layout.addWidget(holder)

        add_btn = QPushButton("Add cut @ current frame")
        add_btn.clicked.connect(self._reseg_add_cut)
        rm_btn = QPushButton("Remove nearest cut")
        rm_btn.clicked.connect(self._reseg_remove_nearest)
        _row(add_btn, rm_btn)
        clear_btn = QPushButton("Clear all cuts")
        clear_btn.clicked.connect(self._reseg_clear)
        reset_btn = QPushButton("Reset to algo cuts")
        reset_btn.clicked.connect(self._reseg_reset)
        _row(clear_btn, reset_btn)
        apply_btn = QPushButton("Apply & reload segments")
        apply_btn.setStyleSheet("font-weight:bold; background:#2a5a2a;")
        apply_btn.clicked.connect(self._reseg_apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._reseg_cancel)
        _row(apply_btn, cancel_btn)

        self._questions_layout.addStretch()
        self._reseg_refresh()

    def _reseg_refresh(self):
        b = self._reseg_boundaries
        n_seg = max(0, len(b) - 1)
        cuts = ", ".join(str(x) for x in b) if b else "(none)"
        self._reseg_summary.setText(
            f"<b>{len(b)} cuts -> {n_seg} segments.</b>  Cuts: {cuts}")

    def _reseg_add_cut(self):
        self._reseg_boundaries = sorted(
            set(self._reseg_boundaries) | {self._reseg_cur_frame()})
        self._reseg_refresh()

    def _reseg_remove_nearest(self):
        if not self._reseg_boundaries:
            return
        f = self._reseg_cur_frame()
        nearest = min(self._reseg_boundaries, key=lambda b: abs(b - f))
        self._reseg_boundaries = [b for b in self._reseg_boundaries if b != nearest]
        self._reseg_refresh()

    def _reseg_clear(self):
        self._reseg_boundaries = []
        self._reseg_refresh()

    def _reseg_reset(self):
        self._reseg_boundaries = sorted({int(x) for x in self._current_boundaries()})
        self._reseg_refresh()

    def _reseg_cancel(self):
        self._show_current_segment()

    def _reseg_apply(self):
        b = sorted({int(x) for x in getattr(self, "_reseg_boundaries", [])})
        if len(b) < 2:
            show_warning("Need at least 2 cuts (a start and an end) to define a segment.")
            return
        pend = self._pending_dir()
        if pend is None or not self._video_stem:
            show_warning("Re-segmentation needs a bundle loaded from the review queue.")
            return
        # RE-RUN mousereach on the reviewer's boundaries (reach -> outcome ->
        # assignment) instead of blanking. Unchanged segments keep the algo's
        # verdict, every segment shows an algo call, and fixing one boundary no
        # longer forces a full re-score. This runs the real pipeline (loads
        # models), so it takes ~1 min and the window may appear to pause.
        self._status_label.setText(
            "Re-running mousereach on your boundaries (reach + outcome + "
            "assignment; ~1 min -- the window may pause)...")
        try:
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()   # flush the status before the blocking run
        except Exception:
            pass
        try:
            from mousereach.review.staging import stage_video
            stage_video(self._video_stem, pending_dir=pend, overwrite=True,
                        verbose=False, boundaries_override=b)
        except Exception as e:
            import traceback
            traceback.print_exc()
            show_warning(f"Re-segmentation pipeline failed: {e}")
            self._status_label.setText("Re-segmentation failed -- see console.")
            return
        # Reload the freshly re-scored bundle (real algo verdicts per segment).
        self._manual_boundaries = None
        self._review_records = {}
        self._load_algo_data()
        self._build_segment_list()
        self._current_seg_idx = 0
        self._show_current_segment()
        show_info(
            f"Re-segmented + re-scored into {len(self._segment_list)} segments -- "
            f"unchanged ranges keep the algo's verdict; correct only what's wrong.")

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

        # (Abnormal / non-reach events are marked with the standalone "Ignore
        # windows" section on the segment -- not a reach-picker option -- so they
        # can be recorded even when every reach is a genuine miss.)

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

    def _make_abnormal_ranges_widget(self, seg: Dict) -> QWidget:
        """Ignore-window picker: mark frame ranges where a NON-reach event moved
        the pellet (tail bump, artifact). Available on every non-boundary segment
        ALONGSIDE the normal reach scoring -- reaches keep their own miss/causal
        labels. The windows exclude that stretch from causal attribution +
        causal-reach kinematics; if no reach is affirmed causal but a window is
        set, the segment's outcome becomes abnormal_exception (a human-only,
        non-evaluable class)."""
        box = QGroupBox("Ignore windows (abnormal non-reach events)")
        layout = QVBoxLayout()
        box.setLayout(layout)

        info = QLabel(
            "Mark frame ranges with non-reach weirdness (e.g. tail/bump moved the "
            "pellet) to exclude from causal attribution + kinematics. Reaches still "
            "get their own miss/causal scoring.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #9ab; font-size: 11px;")
        layout.addWidget(info)

        summary = QLabel("Ignore windows: (none)")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        ranges: List[Dict[str, Any]] = []

        def _refresh():
            if ranges:
                summary.setText("Ignore windows: " + ", ".join(
                    f"{r['start_frame']}-{r['end_frame']}" for r in ranges))
            else:
                summary.setText("Ignore windows: (none)")

        row = QHBoxLayout()
        row.addWidget(QLabel("Start:"))
        start_spin = QSpinBox()
        start_spin.setRange(0, self.n_frames - 1)
        start_spin.setValue(seg["start_frame"])
        start_spin.setMaximumWidth(80)
        row.addWidget(start_spin)
        start_cur = QPushButton("Use current")
        start_cur.clicked.connect(
            lambda: start_spin.setValue(int(self.viewer.dims.current_step[0])))
        row.addWidget(start_cur)
        row.addWidget(QLabel("End:"))
        end_spin = QSpinBox()
        end_spin.setRange(0, self.n_frames - 1)
        end_spin.setValue(seg["end_frame"])
        end_spin.setMaximumWidth(80)
        row.addWidget(end_spin)
        end_cur = QPushButton("Use current")
        end_cur.clicked.connect(
            lambda: end_spin.setValue(int(self.viewer.dims.current_step[0])))
        row.addWidget(end_cur)
        row.addStretch()
        layout.addLayout(row)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add ignore window")
        clear_btn = QPushButton("Clear windows")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        def _add():
            s, e = int(start_spin.value()), int(end_spin.value())
            if e < s:
                s, e = e, s
            ranges.append({"start_frame": s, "end_frame": e, "reason": ""})
            _refresh()

        def _clear():
            ranges.clear()
            _refresh()

        add_btn.clicked.connect(_add)
        clear_btn.clicked.connect(_clear)

        self._q_widgets["_abnormal_ranges"] = {"ranges": ranges, "refresh": _refresh}
        return box

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
                    # (abnormal / non-reach events -> Ignore-windows section)

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
            # Untouched / outcome-uncertain segment.
            q_miss_ans = self._get_toggle_answer("all_miss")
            answers["all_miss"] = q_miss_ans
            if q_miss_ans is True:
                # "all reaches are misses" == the segment is untouched. Record it
                # explicitly so an algo 'triaged' (outcome-uncertain) segment the
                # reviewer resolves as all-miss becomes 'untouched' -- not left
                # 'triaged', which otherwise surfaces as an unresolved
                # triaged -> triaged even though the human DID make the call.
                if seg.get("outcome") == "triaged":
                    agreed = False   # the human resolved what the algo could not
                human_outcome = "untouched"
            elif q_miss_ans is False:
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
                    # (abnormal / non-reach events -> Ignore-windows section)
                combo = self._q_widgets.get("_untouched_outcome_combo")
                if combo:
                    human_outcome = combo.currentText()

        # Ignore windows: frame ranges where a NON-reach event moved the pellet.
        # Recorded alongside the normal reach scoring (misses stay misses). If no
        # reach is affirmed causal but a window is set, the pellet's displacement
        # was not reach-caused -> the segment outcome is abnormal_exception (a
        # human-only, non-evaluable class, excluded from causal-reach kinematics).
        # The windows also pin WHERE the weirdness is for frame-level exclusion
        # downstream.
        abn = self._q_widgets.get("_abnormal_ranges")
        abnormal_ranges = list(abn["ranges"]) if abn else []
        if abnormal_ranges:
            answers["abnormal_ranges"] = abnormal_ranges
            # No human-affirmed causal reach? (untouched w/ all-miss, or touched
            # w/ the algo reach rejected and no replacement chosen.)
            no_causal = human_causal_reach is None
            if answers.get("is_causal") is False:
                no_causal = (human_causal_reach is None
                             or human_causal_reach == seg.get("causal_reach"))
            if no_causal:
                human_outcome = "abnormal_exception"
                human_causal_reach = None
                if seg.get("outcome") not in ("abnormal", "abnormal_exception"):
                    agreed = False

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
        """Pre-populate the question panel from a saved review record, including
        the corrections, so a revisited segment shows your work as you left it."""
        rec = self._review_records.get(segment_num)
        if not rec:
            return
        answers = rec.get("answers", {}) or {}
        human = rec.get("human", {}) or {}
        hc = human.get("causal_reach") or {}
        ho = human.get("outcome")

        for key in ["is_pellet", "is_causal", "end_correct", "outcome_correct", "all_miss"]:
            val = answers.get(key)
            if val is None:
                continue
            q = self._q_widgets.get(key)
            if not q:
                continue
            (q["yes_btn"] if val else q["no_btn"]).setChecked(True)
            q["correction_container"].setVisible(not val)

        # Restore the specific corrections made on a "no" answer.
        if answers.get("outcome_correct") is False and ho:
            combo = self._q_widgets.get("_outcome_combo")
            if combo:
                i = combo.findText(ho)
                if i >= 0:
                    combo.setCurrentIndex(i)
        if answers.get("all_miss") is False and ho:
            combo = self._q_widgets.get("_untouched_outcome_combo")
            if combo:
                i = combo.findText(ho)
                if i >= 0:
                    combo.setCurrentIndex(i)
        if answers.get("end_correct") is False and hc.get("end") is not None:
            spin = self._q_widgets.get("_end_frame_spin")
            if spin:
                spin.setValue(int(hc["end"]))
        if (answers.get("is_causal") is False or answers.get("all_miss") is False) and hc:
            rp = self._q_widgets.get("_reach_picker", {})
            if rp and "start_spin" in rp and hc.get("start") is not None:
                try:
                    rp["start_spin"].setValue(int(hc["start"]))
                    rp["end_spin"].setValue(int(hc["end"]))
                except Exception:
                    pass

        # Restore the ignore windows (abnormal non-reach events).
        abn = self._q_widgets.get("_abnormal_ranges")
        saved_ranges = answers.get("abnormal_ranges")
        if abn is not None and saved_ranges:
            abn["ranges"].clear()
            abn["ranges"].extend(saved_ranges)
            abn["refresh"]()

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
            self._status_label.setText(
                f"[OK] Recorded segment {seg_num}  "
                f"({len(self._review_records)} reviewed so far)")

        if self._current_seg_idx < len(self._segment_list) - 1:
            self._current_seg_idx += 1
            self._show_current_segment()
        else:
            show_info("Last segment -- saving review and loading next video.")
            self._save_and_next_video()

    # ===================================================================
    # Review queue (Pending folder) navigation
    # ===================================================================

    def _pending_dir(self) -> Optional[Path]:
        """The review queue root -- the bundle's parent (the Pending folder)."""
        bd = getattr(self, "_bundle_dir", None)
        return Path(bd).parent if bd else None

    def _review_root(self) -> Path:
        """Root of the review corpus (Model40_Review), where corpus-level files
        like flagged_sessions.json live -- the parent of Pending."""
        pd = self._pending_dir()
        return pd.parent if pd is not None else self.video_path.parent

    def _flag_session(self):
        """Flag this video's whole mouse+day session for mandatory human review
        (e.g. a cage artifact that day affecting every P# of this mouse)."""
        if not self._video_stem:
            return
        reason = (self._notes_edit.toPlainText().strip()
                  if hasattr(self, "_notes_edit") else "")
        try:
            key = flag_session(self._video_stem, self._review_root(), reason=reason)
            show_info(f"Flagged session {key} -- all its videos need human review.")
            self._status_label.setText(
                f"[FLAG] Session {key} flagged for mandatory human review "
                f"(every P# of this mouse+day).")
        except Exception as e:
            show_error(f"Could not flag session: {e}")

    def _list_queue(self) -> List[Path]:
        """Sorted per-video bundle dirs in the Pending queue (a bundle is a dir
        containing a manifest.json)."""
        pd = self._pending_dir()
        if pd is None or not pd.exists():
            return []
        return sorted(d for d in pd.iterdir()
                      if d.is_dir() and (d / "manifest.json").exists())

    def _bundle_reviewed(self, bundle_dir: Path) -> bool:
        """True if this bundle already has a COMPLETE review (every segment
        real-reviewed), in the bundle or next to its canonical video."""
        stem = bundle_dir.name
        dirs = [bundle_dir]
        try:
            man = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            cvp = man.get("canonical_video_path")
            if cvp:
                dirs.append(Path(cvp).parent)
        except Exception:
            pass
        for d in dirs:
            try:
                _, by_seg = load_causal_review(stem, d)
            except Exception:
                by_seg = {}
            if by_seg and all((r.get("answers") or {}).get("reviewed") is not False
                              for r in by_seg.values()):
                return True
        return False

    def _bundle_needs_review(self, bundle_dir: Path, root: Path) -> bool:
        """A bundle needs review iff it is NOT flagged, NOT already ground-truthed
        (GT IS the answer -- it stands in for a review), and NOT already reviewed."""
        stem = bundle_dir.name
        if is_session_flagged(stem, root):
            return False
        if has_gt(stem, extra_dirs=[bundle_dir]):
            return False  # already ground-truthed -> GT is the correct answer
        if self._bundle_reviewed(bundle_dir):
            return False
        return True

    def _segmentation_failed(self, b: Path) -> bool:
        """True if the segmenter FAILED on this video (overall_confidence 0 / bad
        reference quality -> uniform-fallback boundaries -- e.g. an over-long
        recording whose trailing junk frames tank DLC reference tracking). These
        need manual re-segmentation, not normal outcome review, so they are held
        OUT of the auto-review queue (pending a proper failed-seg triage flow)."""
        try:
            sd = json.loads((b / f"{b.name}_segments.json").read_text(encoding="utf-8"))
        except Exception:
            return False
        c = sd.get("overall_confidence")
        if c is not None and c <= 0.0:
            return True
        anoms = sd.get("anomalies") or []
        return any("reference quality" in str(a).lower() for a in anoms)

    def _needs_review_pool(self, pending_dir: Path, root: Path,
                           exclude: Optional[Path] = None) -> List[Path]:
        """All Pending bundles that still need review (not flagged, not GT'd, not
        done, not a failed-segmentation video), optionally excluding one bundle."""
        ex = str(Path(exclude)) if exclude else None
        pool = []
        for b in Path(pending_dir).iterdir():
            if not (b.is_dir() and (b / "manifest.json").exists()):
                continue
            if ex and str(b) == ex:
                continue
            if self._segmentation_failed(b):
                continue   # failed-seg -> manual re-seg queue, not normal review
            if self._bundle_needs_review(b, root):
                pool.append(b)
        return pool

    def load_pending_queue(self, pending_dir: Path):
        """Enter the review queue on a RANDOM video that still needs review
        (skips flagged sessions and already-complete videos). Random sampling
        keeps the reviewed set unbiased across cohorts/days."""
        import random
        pending_dir = Path(pending_dir)
        root = pending_dir.parent
        pool = self._needs_review_pool(pending_dir, root)
        if not pool:
            show_info("Review queue: nothing left to review (all flagged or complete).")
            return
        b = random.choice(pool)
        manifest = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
        self.load_from_manifest(manifest, b)
        self._status_label.setText(
            self._status_label.text() + f"   [random pick -- {len(pool)} left to review]")

    def _load_next_video(self):
        """Load a RANDOM next video from the pool that still NEEDS review,
        skipping flagged sessions, already-complete videos, and the current one.
        Random sampling keeps the reviewed set unbiased across cohorts/days."""
        import random
        pending = self._pending_dir()
        if pending is None or not pending.exists():
            show_info("No review queue found (not loaded from a Pending bundle).")
            return
        root = self._review_root()
        pool = self._needs_review_pool(pending, root,
                                       exclude=getattr(self, "_bundle_dir", None))
        if not pool:
            msg = "Queue complete -- nothing left to review."
            show_info(msg)
            self._status_label.setText(msg)
            return
        nxt = random.choice(pool)
        try:
            manifest = json.loads((nxt / "manifest.json").read_text(encoding="utf-8"))
            self.load_from_manifest(manifest, nxt)
            self._status_label.setText(
                self._status_label.text()
                + f"   [random pick -- {len(pool)} left to review]")
        except Exception as e:
            show_error(f"Could not load next video ({nxt.name}): {e}")

    def _save_and_next_video(self):
        """Save this video's review, then load the next video in the queue."""
        self._save_review()
        self._load_next_video()

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

        # Provenance from the algo JSONs (bundle dir in manifest mode); the
        # review file itself saves next to the video (_review_dir).
        provenance = collect_provenance(self._algo_dir(), self._video_stem)
        reviewer = _get_username()
        timestamp = _get_timestamp()

        # Save per-video file
        mb = getattr(self, "_manual_boundaries", None)
        manual_seg = ({"boundaries": list(mb), "by": reviewer, "at": timestamp,
                       "n_segments": max(0, len(mb) - 1)} if mb else None)
        out_path = save_causal_review(
            video_stem=self._video_stem,
            output_dir=self._review_dir(),
            segments=segments,
            provenance=provenance,
            reviewer=reviewer,
            manual_segmentation=manual_seg,
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
        """Load the entire current segment's frames (wider than the default
        reach window) so the reviewer can hunt outside the reach."""
        if not self._segment_list:
            return
        seg = self._segment_list[self._current_seg_idx]
        self._load_frame_window((seg["start_frame"], seg["end_frame"]))
        try:
            self.viewer.dims.set_current_step(0, seg["start_frame"])
        except Exception:
            pass

    # ===================================================================
    # Playback (reused from GT widget pattern)
    # ===================================================================

    def _setup_keybindings(self):
        """Keyboard shortcuts, bound THROUGH napari (viewer.bind_key) so they
        actually fire in a docked widget AND are suppressed while a text field
        (notes) is focused -- which also stops napari's own single-key shortcuts
        from hijacking the arrows/space during playback."""
        v = self.viewer

        @v.bind_key('Space', overwrite=True)
        def _toggle_play(viewer):
            if self.is_playing:
                self._stop_play()
            else:
                self._play_forward()

        @v.bind_key('b', overwrite=True)
        def _toggle_reverse(viewer):
            if self.is_playing:
                self._stop_play()
            else:
                self._play_reverse()

        for key, delta in [('Left', -1), ('Right', 1), ('Shift-Left', -10),
                           ('Shift-Right', 10), ('Control-Left', -100),
                           ('Control-Right', 100)]:
            v.bind_key(key, (lambda d: (lambda viewer: self._jump_frames(d)))(delta),
                       overwrite=True)

        # Speed keys 1..6 -> 0.25, 0.5, 1, 2, 4, 8
        for key, spd in zip('123456', [0.25, 0.5, 1, 2, 4, 8]):
            v.bind_key(key, (lambda s: (lambda viewer: self._set_speed(s)))(spd),
                       overwrite=True)

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
        self._save_next_video_btn.setEnabled(enabled)
        self._flag_session_btn.setEnabled(enabled)
