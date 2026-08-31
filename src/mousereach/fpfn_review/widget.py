"""
FP/FN Reach Review Widget
=========================

Loads a per-video JSON manifest of detector TP / FP / FN reach events (see
``SCHEMA.md`` in this folder for the format) and lets you scrub a video while
the widget surfaces where each error happens.

Display rules (read-only):
  * Three side-by-side colored squares appear in the upper-left of the
    canvas, each only visible while the playhead is inside that event's
    window:
      - **FP** (red, leftmost) — visible during the detector reach window.
      - **TP** (green, middle) — visible during the detector reach window.
      - **FN** (blue, rightmost) — visible during the GT reach window
        (the algo didn't fire, so only GT defines a span).
  * The status panel on the right of the widget mirrors the active event
    with its frame range and category. If multiple kinds overlap on the
    same frame, the panel surfaces FP > FN > TP, but every square still
    shows independently.
  * Each kind has a checkbox to toggle its layer on/off; all default ON.
  * A single unified table on the right lists every event (FP / TP / FN)
    sorted chronologically by anchor frame. Rows are color-coded by kind
    (red / green / blue) so you can scan them at a glance. Each row has
    a Jump button; double-clicking a row also jumps to its anchor frame.
    Prev/Next FP and Prev/Next FN buttons skip between same-kind events.

The widget never writes anything to the manifest or to the video data. It is
purely a read-only visualization.
"""

from __future__ import annotations

import getpass
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import napari
import numpy as np
from napari.layers import Image, Points, Shapes
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# Canvas marker placement: small squares in the BOTTOM-LEFT of the frame, laid
# out side-by-side so FP / TP / FN can all be visible if they ever overlap.
# Coordinates are (y, x) in pixel space. Exact y is computed at rebuild time
# from the loaded video's height; x positions are fixed near the left edge.
MARKER_SIZE = 18         # 18 px square (half of the original 36)
MARKER_GAP = 3           # gap between adjacent squares
MARKER_PAD = 4           # margin from the canvas edge

# Default video height/width used when no video is loaded yet (e.g. user
# loaded a manifest first). The shapes won't be visible without a video
# anyway; this just keeps the add_shapes call from blowing up on Nones.
_DEFAULT_VIDEO_H = 720
_DEFAULT_VIDEO_W = 1280

# GT roots per manifest corpus come from configuration, never from code:
# which corpora exist and where their ground truth lives is a per-lab fact.
# Add to ~/.mousereach/config.json:
#   "fpfn_gt_roots": {"<corpus label>": "<dir holding *_unified_ground_truth.json>"}
# The widget then auto-resolves the GT path from manifest.corpus +
# manifest.video_id on manifest load; the manual "Load GT" button covers
# everything else.
def _gt_roots() -> Dict[str, str]:
    """Load the corpus -> GT-directory map from ~/.mousereach/config.json."""
    try:
        cfg = Path.home() / ".mousereach" / "config.json"
        with open(cfg, "r", encoding="utf-8") as f:
            roots = json.load(f).get("fpfn_gt_roots", {})
        return roots if isinstance(roots, dict) else {}
    except Exception:
        return {}

# Colors as RGBA (napari Shapes wants 0-1 floats).
# FP / TP / FN now fire only for true FALSE_POSITIVE / TP / FALSE_NEGATIVE
# topology rows — not for the algo/GT halves of TOLERANCE_ERROR, MERGED,
# FRAGMENTED, or COMPLEX components (those get their own markers below).
FP_FACE = [0.85, 0.10, 0.10, 0.55]
FP_EDGE = [1.00, 0.00, 0.00, 1.00]
TP_FACE = [0.10, 0.65, 0.20, 0.45]
TP_EDGE = [0.10, 0.85, 0.20, 1.00]
FN_FACE = [0.15, 0.40, 0.95, 0.55]
FN_EDGE = [0.20, 0.55, 1.00, 1.00]
GT_FACE = [0.95, 0.95, 0.95, 0.55]
GT_EDGE = [1.00, 1.00, 1.00, 1.00]
# Topology-specific squares — colors mirror the table TOPOLOGY_COLOR map.
TOL_FACE = [1.00, 0.80, 0.33, 0.55]        # #ffcc55 amber
TOL_EDGE = [1.00, 0.85, 0.40, 1.00]
MERGED_MARK_FACE = [1.00, 0.60, 0.20, 0.55]  # #ff9933 orange
MERGED_MARK_EDGE = [1.00, 0.65, 0.25, 1.00]
FRAGMENTED_MARK_FACE = [0.80, 0.40, 1.00, 0.55]  # #cc66ff purple
FRAGMENTED_MARK_EDGE = [0.85, 0.45, 1.00, 1.00]
COMPLEX_FACE = [1.00, 0.40, 0.80, 0.55]    # #ff66cc magenta
COMPLEX_EDGE = [1.00, 0.45, 0.85, 1.00]


def _gt_now_iso() -> str:
    """Timestamp for `last_modified_at` / `*_determined_at` fields."""
    return datetime.now().isoformat(timespec="microseconds")


def _gt_username() -> str:
    """Username for `last_modified_by` / `*_determined_by` fields."""
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _rect_corners(y_top: float, y_bot: float, x_left: float, x_right: float) -> np.ndarray:
    """Return a 4x2 (y, x) corner array for a napari rectangle shape."""
    return np.array(
        [
            [y_top, x_left],
            [y_top, x_right],
            [y_bot, x_right],
            [y_bot, x_left],
        ],
        dtype=float,
    )


class FPFNReviewWidget(QWidget):
    """Napari widget: visualize detector FP/FN errors on a video."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Data
        self.manifest_path: Optional[Path] = None
        self.manifest: Optional[Dict[str, Any]] = None
        # _events_raw holds every event from the manifest. self.events (and
        # the kind-split lists below) hold the post-filter view that the rest
        # of the widget consumes; when hide_filtered is True, events flagged
        # kinematically_excluded or outside_gt_segmentation are dropped.
        self._events_raw: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.fp_events: List[Dict[str, Any]] = []
        self.fn_events: List[Dict[str, Any]] = []
        self.tp_events: List[Dict[str, Any]] = []
        self.hide_filtered: bool = True
        self.video_path: Optional[Path] = None
        self.n_frames: int = 0
        self.fps: float = 60.0

        # Layers
        self.video_layer: Optional[Image] = None
        self.fp_shapes_layer: Optional[Shapes] = None
        self.tp_shapes_layer: Optional[Shapes] = None
        self.fn_shapes_layer: Optional[Shapes] = None
        self.gt_shapes_layer: Optional[Shapes] = None  # any-GT-active marker
        self.tol_shapes_layer: Optional[Shapes] = None         # TOLERANCE_ERROR
        self.merged_shapes_layer: Optional[Shapes] = None      # MERGED
        self.fragmented_shapes_layer: Optional[Shapes] = None  # FRAGMENTED
        self.complex_shapes_layer: Optional[Shapes] = None     # COMPLEX
        self.dlc_points_layer: Optional[Points] = None

        # DLC overlay
        self.dlc_h5_path: Optional[Path] = None

        # Ground-truth editing state
        self.gt_path: Optional[Path] = None
        self.gt_data: Optional[Dict[str, Any]] = None  # full JSON, round-tripped
        self.gt_reaches: List[Dict[str, Any]] = []     # alias to gt_data["reaches"]["reaches"]
        self.gt_dirty: bool = False                    # any unsaved edits since last load/save
        self._gt_edits_saved_since_load: bool = False  # for stale-indicator
        # Cache the last reach_id the user selected. Some Qt versions clear
        # the table's selectionModel when focus moves to an action button,
        # which would make _gt_selected_reach return None right when the user
        # clicks Set Start / Set End. Caching the id at selection-change time
        # keeps the handlers working past that focus shift.
        self._gt_last_selected_rid: Optional[int] = None

        # Cached frame ranges aligned with shape indices in each layer.
        # FP/TP use the detector reach window; FN uses the GT reach window
        # (since FN means the algo didn't fire there — only GT has a span).
        # GT ranges (for the any-GT-active white marker) come from
        # self.gt_reaches live so they reflect edits immediately, with a
        # fall-back to manifest event GT ranges when no GT file is loaded.
        self._fp_ranges: List[tuple] = []
        self._tp_ranges: List[tuple] = []
        self._fn_ranges: List[tuple] = []
        self._gt_ranges: List[tuple] = []
        self._tol_ranges: List[tuple] = []
        self._merged_ranges: List[tuple] = []
        self._fragmented_ranges: List[tuple] = []
        self._complex_ranges: List[tuple] = []

        # Playback state (matches DLC Compare pattern)
        self.is_playing = False
        self.playback_speed = 1
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        self._build_ui()
        self.viewer.dims.events.current_step.connect(self._on_frame_change)

    # ── UI build ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
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
            "Load a manifest JSON (see SCHEMA.md), then a video. "
            "Red box appears on the canvas while the playhead is inside an FP window."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(instr)

        # ── File loading ──
        file_group = QGroupBox("Load")
        file_layout = QVBoxLayout()

        manifest_row = QHBoxLayout()
        self.manifest_btn = QPushButton("Load Manifest (.json)…")
        self.manifest_btn.clicked.connect(self._pick_manifest)
        manifest_row.addWidget(self.manifest_btn)
        self.manifest_label = QLabel("No manifest loaded")
        self.manifest_label.setStyleSheet("color: #888;")
        manifest_row.addWidget(self.manifest_label, 1)
        file_layout.addLayout(manifest_row)

        video_row = QHBoxLayout()
        self.video_btn = QPushButton("Load Video…")
        self.video_btn.clicked.connect(self._pick_video)
        video_row.addWidget(self.video_btn)
        self.video_label = QLabel("No video loaded")
        self.video_label.setStyleSheet("color: #888;")
        video_row.addWidget(self.video_label, 1)
        file_layout.addLayout(video_row)

        dlc_row = QHBoxLayout()
        self.dlc_btn = QPushButton("Load DLC (.h5)…")
        self.dlc_btn.clicked.connect(self._pick_dlc)
        dlc_row.addWidget(self.dlc_btn)
        self.dlc_label = QLabel("No DLC loaded")
        self.dlc_label.setStyleSheet("color: #888;")
        dlc_row.addWidget(self.dlc_label, 1)
        file_layout.addLayout(dlc_row)

        gt_row = QHBoxLayout()
        self.gt_btn = QPushButton("Load GT (.json)…")
        self.gt_btn.clicked.connect(self._pick_gt)
        gt_row.addWidget(self.gt_btn)
        self.gt_label = QLabel("No GT loaded")
        self.gt_label.setStyleSheet("color: #888;")
        gt_row.addWidget(self.gt_label, 1)
        file_layout.addLayout(gt_row)

        # Stale indicator: shown after a successful Save. Hidden by default.
        self.gt_stale_label = QLabel(
            "GT edited - manifest now stale; re-score and regenerate it from the updated GT."
        )
        self.gt_stale_label.setStyleSheet(
            "color: #fa8; font-style: italic; padding: 2px 0 0 4px;"
        )
        self.gt_stale_label.setVisible(False)
        file_layout.addWidget(self.gt_stale_label)

        # Clears the loaded manifest/video/DLC/GT from the widget only.
        # Files on disk are not touched.
        self.clear_all_btn = QPushButton("Clear loaded files (widget only)")
        self.clear_all_btn.setToolTip(
            "Resets the widget so you can load a new manifest/video/DLC/GT "
            "from scratch. Does NOT delete or modify any files on disk."
        )
        self.clear_all_btn.clicked.connect(self._clear_all)
        file_layout.addWidget(self.clear_all_btn)

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

        transport_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        transport_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        transport_row.addWidget(self.stop_btn)
        play_layout.addLayout(transport_row)

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
            setattr(self, f"_speed_btn_{mult}", btn)
        play_layout.addLayout(speed_row)

        play_group.setLayout(play_layout)
        layout.addWidget(play_group)

        # ── Header info ──
        self.header_label = QLabel("—")
        self.header_label.setWordWrap(True)
        self.header_label.setFont(QFont("Consolas", 10))
        self.header_label.setStyleSheet("padding: 4px; background: #1a1a1a;")
        layout.addWidget(self.header_label)

        # Per-video topology breakdown. Same counting convention as the
        # manifest generator: per-event for TP/FALSE_POSITIVE/FALSE_NEGATIVE/
        # COMPLEX, per-component for MERGED/FRAGMENTED, per-pair for
        # TOLERANCE_ERROR. Reflects the current filter (hide_filtered) since
        # it counts from self.events.
        self.topology_table = QTableWidget(len(self.TOPOLOGY_ORDER), 2)
        self.topology_table.setHorizontalHeaderLabels(["Type", "Count"])
        self.topology_table.verticalHeader().setVisible(False)
        self.topology_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.topology_table.setSelectionMode(QTableWidget.NoSelection)
        self.topology_table.setFocusPolicy(Qt.NoFocus)
        topo_header = self.topology_table.horizontalHeader()
        topo_header.setSectionResizeMode(0, QHeaderView.Stretch)
        topo_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for row_idx, topo in enumerate(self.TOPOLOGY_ORDER):
            name_item = QTableWidgetItem(topo)
            color = self.TOPOLOGY_COLOR.get(topo)
            if color is not None:
                name_item.setForeground(color)
            self.topology_table.setItem(row_idx, 0, name_item)
            count_item = QTableWidgetItem("0")
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.topology_table.setItem(row_idx, 1, count_item)
        # Fit-to-content height: header + N rows, no scrollbar.
        row_h = self.topology_table.verticalHeader().defaultSectionSize()
        header_h = self.topology_table.horizontalHeader().height()
        self.topology_table.setFixedHeight(
            header_h + row_h * len(self.TOPOLOGY_ORDER) + 4
        )
        layout.addWidget(self.topology_table)

        # ── Current-event status (canvas-marker equivalent in panel form) ──
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(36)
        self.status_label.setStyleSheet(
            "padding: 6px; font-size: 13px; font-weight: bold; "
            "background: #181818; color: #888;"
        )
        layout.addWidget(self.status_label)

        # (Marker visibility toggles removed 2026-05-21 — each marker now
        # fires whenever its frame range contains the playhead, no opt-out.)

        # ── Unified events table ──
        events_group = QGroupBox(
            "Events  —  FP (red) · FN (blue) · TP (green, GT-matched)"
        )
        events_layout = QVBoxLayout()

        nav_row = QHBoxLayout()
        self.prev_fp_btn = QPushButton("◀ Prev FP")
        self.prev_fp_btn.clicked.connect(lambda: self._step_event(self.fp_events, -1, "detector"))
        self.next_fp_btn = QPushButton("Next FP ▶")
        self.next_fp_btn.clicked.connect(lambda: self._step_event(self.fp_events, +1, "detector"))
        nav_row.addWidget(self.prev_fp_btn)
        nav_row.addWidget(self.next_fp_btn)

        self.prev_fn_btn = QPushButton("◀ Prev FN")
        self.prev_fn_btn.clicked.connect(lambda: self._step_event(self.fn_events, -1, "gt"))
        self.next_fn_btn = QPushButton("Next FN ▶")
        self.next_fn_btn.clicked.connect(lambda: self._step_event(self.fn_events, +1, "gt"))
        nav_row.addWidget(self.prev_fn_btn)
        nav_row.addWidget(self.next_fn_btn)
        nav_row.addStretch()
        events_layout.addLayout(nav_row)

        # Hide events flagged kinematically_excluded or outside_gt_segmentation
        # — mirrors the post-match filter the metrics scripts apply.
        self.hide_filtered_cb = QCheckBox("Hide filtered-out reaches")
        self.hide_filtered_cb.setChecked(self.hide_filtered)
        self.hide_filtered_cb.toggled.connect(self._on_hide_filtered_toggled)
        events_layout.addWidget(self.hide_filtered_cb)

        # Columns: Reach Type | Start | End | Category | Jump
        # Reach Type is the topology label (new 2026-05-20):
        # TP / TOLERANCE_ERROR / MERGED / FRAGMENTED / FALSE_POSITIVE /
        # FALSE_NEGATIVE / COMPLEX. Empty for older manifests without the field.
        # Kind column was removed as redundant with Reach Type.
        self.events_table = QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(
            ["Reach Type", "Start", "End", "Category", ""]
        )
        self.events_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.events_table.setSelectionMode(QTableWidget.SingleSelection)
        # Sorting intentionally disabled — rows are pre-sorted by anchor
        # frame in Python so MERGED/FRAGMENTED components stay grouped.
        # Click-to-sort by any column would split anchor rows from their
        # ↓ continuations (which have blank Start/End).
        self.events_table.setSortingEnabled(False)
        header = self.events_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Reach Type
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Start
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # End
        header.setSectionResizeMode(3, QHeaderView.Stretch)           # Category
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Jump
        self.events_table.itemDoubleClicked.connect(
            self._on_events_table_double_clicked
        )
        events_layout.addWidget(self.events_table)
        events_group.setLayout(events_layout)
        layout.addWidget(events_group)

        # ── Ground-truth reaches (editable) ──
        gt_group = QGroupBox("Ground Truth Reaches (editable)")
        gt_layout = QVBoxLayout()

        # Top row: Add Reach + Save
        gt_top_row = QHBoxLayout()
        self.gt_add_btn = QPushButton("+ Add Reach Here")
        self.gt_add_btn.setToolTip(
            "Create a new GT reach starting at the current playhead frame."
        )
        self.gt_add_btn.clicked.connect(self._gt_add_reach_here)
        gt_top_row.addWidget(self.gt_add_btn)

        self.gt_save_btn = QPushButton("Save GT (atomic)")
        self.gt_save_btn.setStyleSheet("background-color: #2a4a20;")
        self.gt_save_btn.setToolTip(
            "Atomically write the edited GT back to the file it was loaded from."
        )
        self.gt_save_btn.clicked.connect(self._gt_save)
        gt_top_row.addWidget(self.gt_save_btn)
        gt_top_row.addStretch()
        gt_layout.addLayout(gt_top_row)

        # Reach table: Reach# | Start | Set S | End | Set E | Status | Jump
        # Per-row Set S / Set E buttons let us avoid relying on table selection
        # state for actions — each button captures the reach_id at row-render
        # time via a closure, so clicking it always acts on the right reach.
        self.gt_table = QTableWidget(0, 7)
        self.gt_table.setHorizontalHeaderLabels(
            ["Reach #", "Start", "Set S", "End", "Set E", "Status", ""]
        )
        self.gt_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.gt_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.gt_table.setSelectionMode(QTableWidget.SingleSelection)
        self.gt_table.setSortingEnabled(True)
        gt_header = self.gt_table.horizontalHeader()
        gt_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Reach #
        gt_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Start
        gt_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Set S
        gt_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # End
        gt_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Set E
        gt_header.setSectionResizeMode(5, QHeaderView.Stretch)           # Status
        gt_header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Jump
        self.gt_table.itemDoubleClicked.connect(self._on_gt_table_double_clicked)
        self.gt_table.itemSelectionChanged.connect(self._gt_update_selected_label)
        gt_layout.addWidget(self.gt_table)

        # Selected-reach action panel
        self.gt_selected_label = QLabel("Selected: (none)")
        self.gt_selected_label.setStyleSheet("color: #ccc; padding-top: 4px;")
        gt_layout.addWidget(self.gt_selected_label)

        gt_action_row = QHBoxLayout()
        self.gt_set_start_btn = QPushButton("Set Start (S)")
        self.gt_set_start_btn.setToolTip("Set selected reach's start_frame to current playhead.")
        self.gt_set_start_btn.clicked.connect(self._gt_set_selected_start)
        gt_action_row.addWidget(self.gt_set_start_btn)

        self.gt_set_end_btn = QPushButton("Set End (E)")
        self.gt_set_end_btn.setToolTip("Set selected reach's end_frame to current playhead.")
        self.gt_set_end_btn.clicked.connect(self._gt_set_selected_end)
        gt_action_row.addWidget(self.gt_set_end_btn)

        self.gt_exclude_btn = QPushButton("Toggle Exclude")
        self.gt_exclude_btn.setToolTip(
            "Mark selected reach as excluded from analysis (soft-delete). "
            "Downstream loaders skip excluded reaches by default."
        )
        self.gt_exclude_btn.clicked.connect(self._gt_toggle_selected_exclude)
        gt_action_row.addWidget(self.gt_exclude_btn)

        self.gt_comment_btn = QPushButton("💬 Comment")
        self.gt_comment_btn.setToolTip("Add or edit a comment on the selected reach.")
        self.gt_comment_btn.clicked.connect(self._gt_edit_selected_comment)
        gt_action_row.addWidget(self.gt_comment_btn)
        gt_action_row.addStretch()
        gt_layout.addLayout(gt_action_row)

        gt_group.setLayout(gt_layout)
        layout.addWidget(gt_group)

        layout.addStretch()

    # ── Manifest loading ────────────────────────────────────────────────

    def _pick_manifest(self) -> None:
        # Default to this module's folder so a recently used manifest folder is one click away.
        start_dir = str(Path(__file__).parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Manifest", start_dir, "JSON (*.json);;All Files (*)"
        )
        if not path:
            return
        self._load_manifest(Path(path))

    def _load_manifest(self, path: Path) -> None:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            self.manifest_label.setText(f"ERROR: {e}")
            self.manifest_label.setStyleSheet("color: #f88;")
            return

        if "events" not in data or not isinstance(data["events"], list):
            self.manifest_label.setText("ERROR: manifest missing 'events' list")
            self.manifest_label.setStyleSheet("color: #f88;")
            return

        self.manifest_path = path
        self.manifest = data
        self._events_raw = data["events"]
        self._apply_event_filter()

        self.n_frames = int(data.get("n_frames") or 0)
        self.fps = float(data.get("fps") or 60.0)

        self.manifest_label.setText(path.name)
        self.manifest_label.setStyleSheet("color: #8f8;")
        self._refresh_header()
        self._populate_tables()
        # Defer Shapes-layer creation until a video is loaded — adding 2D
        # shapes BEFORE the 3D image confuses napari's dims reconciliation
        # (on some versions it raises with the new Image layer as the
        # exception arg, leaving the user with a cryptic error).
        if self.video_layer is not None:
            self._rebuild_marker_layers()

        # If the manifest names a resolvable video path, auto-load it.
        vp = data.get("video_path")
        if vp:
            vp_path = Path(vp)
            if vp_path.exists():
                self._load_video(vp_path)
            else:
                print(f"[FP/FN Review] manifest video_path not resolvable: {vp}")

        # Auto-resolve the canonical GT file for this manifest and load it.
        self._gt_auto_load_for_manifest()

        print(
            f"[FP/FN Review] Manifest loaded: {path.name}  "
            f"({len(self.fp_events)} FP, {len(self.fn_events)} FN, "
            f"{len(self.tp_events)} TP)"
        )

    def _apply_event_filter(self) -> None:
        # Rebuild self.events (and the kind-split lists) from _events_raw,
        # dropping flagged events when hide_filtered is on. Same flag check
        # the metrics scripts apply for headline FP/FN/TP totals.
        if self.hide_filtered:
            self.events = [
                e for e in self._events_raw
                if not (
                    e.get("kinematically_excluded")
                    or e.get("outside_gt_segmentation")
                )
            ]
        else:
            self.events = list(self._events_raw)
        self.fp_events = sorted(
            (e for e in self.events if e.get("kind") == "FP"),
            key=lambda e: (e.get("detector") or {}).get("start") or 0,
        )
        self.fn_events = sorted(
            (e for e in self.events if e.get("kind") == "FN"),
            key=lambda e: (e.get("gt") or {}).get("start") or 0,
        )
        self.tp_events = sorted(
            (e for e in self.events if e.get("kind") == "TP"),
            key=lambda e: (e.get("detector") or {}).get("start") or 0,
        )

    def _refresh_topology_summary(self) -> None:
        """Recompute per-video topology counts from self.events.

        Mirrors the manifest generator's convention:
          * TP / FALSE_POSITIVE / FALSE_NEGATIVE / COMPLEX: per-event count
          * MERGED / FRAGMENTED: per-component count (collapse multi-event
            components to one)
          * TOLERANCE_ERROR: per-pair count (2 events per pair, // 2)
        """
        counts = {t: 0 for t in self.TOPOLOGY_ORDER}
        merged_cids: set = set()
        frag_cids: set = set()
        tol_events = 0
        for ev in self.events:
            topo = ev.get("topology")
            if topo == "MERGED":
                merged_cids.add(ev.get("component_id"))
            elif topo == "FRAGMENTED":
                frag_cids.add(ev.get("component_id"))
            elif topo == "TOLERANCE_ERROR":
                tol_events += 1
            elif topo in counts:
                counts[topo] += 1
        counts["MERGED"] = len(merged_cids)
        counts["FRAGMENTED"] = len(frag_cids)
        counts["TOLERANCE_ERROR"] = tol_events // 2
        for row_idx, topo in enumerate(self.TOPOLOGY_ORDER):
            item = self.topology_table.item(row_idx, 1)
            if item is not None:
                item.setText(str(counts[topo]))

    def _on_hide_filtered_toggled(self, checked: bool) -> None:
        self.hide_filtered = bool(checked)
        if self.manifest is None:
            return
        self._apply_event_filter()
        self._refresh_header()
        self._populate_tables()
        if self.video_layer is not None:
            self._rebuild_marker_layers()

    def _refresh_header(self) -> None:
        if self.manifest is None:
            self.header_label.setText("—")
            return
        m = self.manifest
        bits = [
            f"video: {m.get('video_id', '?')}",
            f"detector: {m.get('detector_version', '?')}",
            f"corpus: {m.get('corpus', '?')}",
            f"match: {m.get('matching_criterion', '?')}",
            f"frames: {m.get('n_frames', '?')}",
            f"FP={len(self.fp_events)}  FN={len(self.fn_events)}  TP={len(self.tp_events)}",
        ]
        self.header_label.setText("\n".join(bits))

    # ── Video loading ───────────────────────────────────────────────────

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not path:
            return
        self._load_video(Path(path))

    def _load_video(self, path: Path) -> None:
        # Tear down any existing video, DLC overlay, and marker layers from a
        # previous video, since they were built against the old image's dims.
        for attr in (
            "fp_shapes_layer",
            "tp_shapes_layer",
            "fn_shapes_layer",
            "dlc_points_layer",
            "video_layer",
        ):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)
        # Drop the previous DLC path/label so the user explicitly re-picks one
        # for the new video (DLC points are video-specific).
        self.dlc_h5_path = None
        self.dlc_label.setText("No DLC loaded")
        self.dlc_label.setStyleSheet("color: #888;")

        # Phase 1: actually load the video. A failure here is the only thing
        # that should put the label into a red ERROR state.
        try:
            from mousereach.lazy_video import LazyVideoArray

            lazy = LazyVideoArray(path)
            self.n_frames = lazy.n_frames
            self.fps = lazy.fps
            self.video_path = path
            self.video_layer = self.viewer.add_image(
                lazy,
                name=f"Video: {path.stem}",
                metadata={"path": str(path), "fpfn_review": True},
            )
            self.video_label.setText(path.name)
            self.video_label.setStyleSheet("color: #8f8;")
            print(
                f"[FP/FN Review] Video loaded: {path.name} "
                f"({self.n_frames} frames, {self.fps:.0f} fps)"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Use repr() so an exception whose __str__ returns a napari Layer
            # repr can't masquerade as a working load. The full traceback is
            # already in the terminal via print_exc above.
            self.video_label.setText(f"ERROR: {type(e).__name__}: {e!r}")
            self.video_label.setStyleSheet("color: #f88;")
            return

        # Phase 2: build / refresh the marker overlays. If anything here
        # throws, the video is still loaded — surface a warning in the
        # console but leave the video status green.
        try:
            self._rebuild_marker_layers()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(
                f"[FP/FN Review] WARNING: marker rebuild failed: "
                f"{type(e).__name__}: {e!r}"
            )
        try:
            self._on_frame_change()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(
                f"[FP/FN Review] WARNING: frame-change refresh failed: "
                f"{type(e).__name__}: {e!r}"
            )

    # ── DLC overlay ─────────────────────────────────────────────────────

    def _pick_dlc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DLC H5 file",
            "",
            "HDF5 Files (*.h5);;All Files (*)",
        )
        if not path:
            return
        self._load_dlc(Path(path))

    def _load_dlc(self, path: Path) -> None:
        """Load a DLC H5 and overlay its points on the viewer.

        Reuses the H5 reader, bodypart parser, and point-array builder from
        the DLC Compare widget so the rendering style (per-bodypart colors,
        likelihood-modulated alpha) matches across tools.
        """
        # Drop any prior DLC layer
        if self.dlc_points_layer is not None:
            try:
                self.viewer.layers.remove(self.dlc_points_layer)
            except Exception:
                pass
            self.dlc_points_layer = None

        try:
            from mousereach.dlc_compare.widget import (
                _build_points_array,
                _get_bodyparts,
                _load_dlc_h5,
                SPLIT_COLORS,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.dlc_label.setText(f"ERROR import: {type(e).__name__}")
            self.dlc_label.setStyleSheet("color: #f88;")
            return

        try:
            df = _load_dlc_h5(path)
            if df is None:
                self.dlc_label.setText(f"ERROR loading {path.name}")
                self.dlc_label.setStyleSheet("color: #f88;")
                return

            bodyparts = _get_bodyparts(df)
            points, colors, bps = _build_points_array(
                df, bodyparts, SPLIT_COLORS, alpha=0.7
            )
            if len(points) == 0:
                self.dlc_label.setText(f"{path.name} (no points above threshold)")
                self.dlc_label.setStyleSheet("color: #fa8;")
                self.dlc_h5_path = path
                return

            self.dlc_h5_path = path
            self.dlc_points_layer = self.viewer.add_points(
                points,
                face_color=colors,
                size=6,
                name=f"DLC: {path.stem}",
                features={"bp": bps},
            )
            self.dlc_points_layer.visible = True
            self.dlc_label.setText(path.name)
            self.dlc_label.setToolTip(str(path))
            self.dlc_label.setStyleSheet("color: #8f8;")
            print(
                f"[FP/FN Review] DLC loaded: {path.name} "
                f"({len(bodyparts)} bodyparts, {len(points)} points)"
            )
            # Keep the FP/TP/FN squares on top so DLC overlay doesn't bury them.
            self._move_marker_layers_to_top()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.dlc_label.setText(f"ERROR: {type(e).__name__}: {e!r}")
            self.dlc_label.setStyleSheet("color: #f88;")

    def _refresh_dlc_visibility(self, _checked: bool = False) -> None:
        # No-op kept for back-compat. DLC layer is always visible now that
        # the visibility toggles have been removed.
        if self.dlc_points_layer is not None:
            self.dlc_points_layer.visible = True

    # ── Ground-truth editing ───────────────────────────────────────────

    def _gt_auto_load_for_manifest(self) -> None:
        """Resolve the canonical GT path from manifest corpus + video_id."""
        self._gt_close()
        if self.manifest is None:
            return
        corpus = self.manifest.get("corpus")
        vid = self.manifest.get("video_id")
        if not corpus or not vid:
            self.gt_label.setText("no auto-GT: manifest missing corpus/video_id")
            self.gt_label.setStyleSheet("color: #fa8;")
            return
        root = _gt_roots().get(corpus)
        if root is None:
            self.gt_label.setText(
                f"no auto-GT: corpus '{corpus}' not configured (use Load GT to pick)"
            )
            self.gt_label.setStyleSheet("color: #fa8;")
            return
        candidate = Path(root) / f"{vid}_unified_ground_truth.json"
        if not candidate.exists():
            self.gt_label.setText(
                f"no auto-GT: file not found ({candidate})"
            )
            self.gt_label.setStyleSheet("color: #fa8;")
            return
        self._gt_load(candidate)

    def _pick_gt(self) -> None:
        # Default start dir matches whichever configured GT root makes sense
        # for the currently-loaded manifest. Falls back to no preset.
        start_dir = ""
        if self.manifest is not None:
            corpus = self.manifest.get("corpus")
            roots = _gt_roots()
            if corpus in roots:
                start_dir = roots[corpus]
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GT JSON", start_dir,
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        self._gt_load(Path(path))

    def _gt_load(self, path: Path) -> None:
        """Load a unified GT JSON, set state, populate the table."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gt_label.setText(f"ERROR: {type(e).__name__}: {e!r}")
            self.gt_label.setStyleSheet("color: #f88;")
            return

        # Tolerate both fully-unified and partial schemas, but warn.
        if "reaches" not in data or not isinstance(data["reaches"], dict):
            self.gt_label.setText("ERROR: unified GT missing 'reaches' object")
            self.gt_label.setStyleSheet("color: #f88;")
            return
        if "reaches" not in data["reaches"]:
            # Older schema: reaches block exists but no inner reaches list
            data["reaches"]["reaches"] = []
        if not isinstance(data["reaches"]["reaches"], list):
            self.gt_label.setText("ERROR: reaches.reaches is not a list")
            self.gt_label.setStyleSheet("color: #f88;")
            return

        self.gt_path = path
        self.gt_data = data
        self.gt_reaches = data["reaches"]["reaches"]
        self.gt_dirty = False
        self._gt_edits_saved_since_load = False
        self.gt_stale_label.setVisible(False)

        self.gt_label.setText(f"{path.name}  ({len(self.gt_reaches)} reaches)")
        self.gt_label.setToolTip(str(path))
        self.gt_label.setStyleSheet("color: #8f8;")
        self._gt_populate_table()
        print(
            f"[FP/FN Review] GT loaded: {path}  "
            f"({len(self.gt_reaches)} reaches)"
        )

    def _gt_close(self) -> None:
        """Clear all GT state without saving."""
        self.gt_path = None
        self.gt_data = None
        self.gt_reaches = []
        self.gt_dirty = False
        self._gt_edits_saved_since_load = False
        self.gt_stale_label.setVisible(False)
        self.gt_label.setText("No GT loaded")
        self.gt_label.setStyleSheet("color: #888;")
        self.gt_label.setToolTip("")
        self.gt_table.setRowCount(0)
        self.gt_selected_label.setText("Selected: (none)")

    def _clear_all(self) -> None:
        """Reset the widget so a new manifest/video/DLC/GT can be loaded.

        Does not touch any files on disk. If there are unsaved GT edits,
        prompts the user before discarding them.
        """
        if self.gt_dirty:
            resp = QMessageBox.question(
                self,
                "Discard unsaved GT edits?",
                "There are unsaved edits to the loaded GT. Clearing the "
                "widget will discard them.\n\n"
                "Files on disk are NOT modified — this only resets the "
                "widget. Continue?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if resp != QMessageBox.Discard:
                return

        # Drop GT state via the existing close path.
        self._gt_close()

        # Remove napari layers (video, DLC, all marker overlays).
        for attr in (
            "video_layer",
            "dlc_points_layer",
            "fp_shapes_layer",
            "tp_shapes_layer",
            "fn_shapes_layer",
            "gt_shapes_layer",
            "tol_shapes_layer",
            "merged_shapes_layer",
            "fragmented_shapes_layer",
            "complex_shapes_layer",
        ):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)

        # Reset manifest state.
        self.manifest_path = None
        self.manifest = None
        self._events_raw = []
        self.events = []
        self.fp_events = []
        self.fn_events = []
        self.tp_events = []

        # Reset video / DLC state.
        self.video_path = None
        self.dlc_h5_path = None
        self.n_frames = 0
        self.fps = 60.0

        # Reset the file-row labels to their initial "nothing loaded" state.
        self.manifest_label.setText("No manifest loaded")
        self.manifest_label.setStyleSheet("color: #888;")
        self.manifest_label.setToolTip("")
        self.video_label.setText("No video loaded")
        self.video_label.setStyleSheet("color: #888;")
        self.video_label.setToolTip("")
        self.dlc_label.setText("No DLC loaded")
        self.dlc_label.setStyleSheet("color: #888;")
        self.dlc_label.setToolTip("")

        # Clear the events table, reset the topology summary, refresh header.
        self.events_table.setRowCount(0)
        self._refresh_topology_summary()
        self._refresh_header()

        print("[FP/FN Review] Widget cleared (no files modified).")

    def _gt_save(self) -> None:
        """Atomically write the GT back to the loaded path.

        Round-trips the original JSON so non-reach fields (segmentation
        block, outcomes, completion_status, schema_version, …) are
        preserved. Only mutates ``reaches.reaches`` and updates
        ``last_modified_at`` / ``last_modified_by``.
        """
        if self.gt_data is None or self.gt_path is None:
            return
        try:
            self.gt_data["reaches"]["reaches"] = self.gt_reaches
            self.gt_data["last_modified_at"] = _gt_now_iso()
            self.gt_data["last_modified_by"] = _gt_username()
            tmp = self.gt_path.with_suffix(self.gt_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.gt_data, f, indent=2)
            os.replace(tmp, self.gt_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gt_label.setText(f"SAVE ERROR: {type(e).__name__}: {e!r}")
            self.gt_label.setStyleSheet("color: #f88;")
            return
        self.gt_dirty = False
        self._gt_edits_saved_since_load = True
        self.gt_stale_label.setVisible(True)
        self.gt_label.setText(
            f"{self.gt_path.name}  ({len(self.gt_reaches)} reaches, saved)"
        )
        self.gt_label.setStyleSheet("color: #8f8;")
        print(f"[FP/FN Review] GT saved: {self.gt_path}")

    def _gt_mark_dirty(self) -> None:
        self.gt_dirty = True
        if self.gt_path is not None:
            self.gt_label.setText(
                f"{self.gt_path.name}  ({len(self.gt_reaches)} reaches, UNSAVED *)"
            )
            self.gt_label.setStyleSheet("color: #fc8;")

    # ── GT reach operations ────────────────────────────────────────────

    def _gt_current_frame(self) -> Optional[int]:
        try:
            return int(self.viewer.dims.current_step[0])
        except Exception:
            return None

    def _gt_flash(self, msg: str) -> None:
        """Surface a short status message via the selected-reach label + stdout.

        Used when GT operations succeed or no-op so the user gets feedback
        instead of silent failure. Also prints to the terminal so the message
        survives subsequent selection-change re-renders of the label.
        """
        try:
            self.gt_selected_label.setText(msg)
        except Exception:
            pass
        print(f"[FP/FN Review] {msg}")

    def _gt_segment_for_frame(self, frame: int) -> int:
        """Locate the segment index containing the given frame from gt_data.

        Defaults to 1 if segmentation info isn't present. Segments are stored
        under ``segmentation.boundaries`` as a list of {start_frame, end_frame}
        or similar; fall back to 1 on any structural mismatch.
        """
        if self.gt_data is None:
            return 1
        seg_block = self.gt_data.get("segmentation") or {}
        boundaries = seg_block.get("boundaries") or []
        for i, b in enumerate(boundaries):
            s = b.get("start_frame") if isinstance(b, dict) else None
            e = b.get("end_frame") if isinstance(b, dict) else None
            if s is None or e is None:
                continue
            if s <= frame <= e:
                return i + 1
        return 1

    def _gt_add_reach_here(self) -> None:
        """Create a new GT reach at the current playhead, schema-complete."""
        if self.gt_data is None:
            return
        cur = self._gt_current_frame()
        if cur is None:
            return
        new_id = (
            max((r.get("reach_id", 0) for r in self.gt_reaches), default=0) + 1
        )
        now = _gt_now_iso()
        user = _gt_username()
        new_reach = {
            "reach_id": new_id,
            "segment_num": self._gt_segment_for_frame(cur),
            "start_frame": cur,
            "start_determined": True,
            "start_determined_by": user,
            "start_determined_at": now,
            "end_frame": cur,  # placeholder; user sets via Set End (E)
            "end_determined": False,
            "end_determined_by": None,
            "end_determined_at": None,
            "apex_frame": None,  # human-added reaches leave apex null per convention
            "exclude_from_analysis": False,
            "exclude_reason": None,
            "comment": None,
        }
        self.gt_reaches.append(new_reach)
        # Keep the list sorted by start_frame so downstream consumers can
        # rely on chronological order (and so the table reads that way).
        self.gt_reaches.sort(key=lambda r: r.get("start_frame", 0))
        self._gt_mark_dirty()
        self._gt_populate_table()
        # Select the new reach row so S/E act on it immediately.
        self._gt_select_reach_by_id(new_id)

    def _gt_selected_reach(self) -> Optional[Dict[str, Any]]:
        idx = self._gt_selected_index()
        if idx is None:
            return None
        return self.gt_reaches[idx]

    def _gt_selected_index(self) -> Optional[int]:
        """Return index into self.gt_reaches for the currently-selected row.

        Prefers the live selectionModel; falls back to the cached
        ``_gt_last_selected_rid`` if the table's selection has been cleared
        (some Qt versions drop it when an action button steals focus).
        """
        rid: Optional[int] = None
        try:
            rows = self.gt_table.selectionModel().selectedRows()
            if rows:
                rid_item = self.gt_table.item(rows[0].row(), 0)
                if rid_item is not None:
                    try:
                        rid = int(rid_item.data(Qt.DisplayRole))
                    except (TypeError, ValueError):
                        rid = None
        except Exception:
            rid = None
        if rid is None:
            rid = self._gt_last_selected_rid
        if rid is None:
            return None
        for i, r in enumerate(self.gt_reaches):
            if r.get("reach_id") == rid:
                return i
        return None

    def _gt_select_reach_by_id(self, reach_id: int) -> None:
        """Programmatic selection — find and highlight the row with this id."""
        for row in range(self.gt_table.rowCount()):
            item = self.gt_table.item(row, 0)
            if item is None:
                continue
            try:
                if int(item.data(Qt.DisplayRole)) == reach_id:
                    self.gt_table.selectRow(row)
                    return
            except (TypeError, ValueError):
                continue

    def _gt_set_selected_start(self) -> None:
        if self.gt_data is None:
            self._gt_flash("No GT loaded - click 'Load GT (.json)...' first.")
            return
        reach = self._gt_selected_reach()
        if reach is None:
            self._gt_flash("No GT reach selected - click a row in the GT table first.")
            return
        cur = self._gt_current_frame()
        if cur is None:
            self._gt_flash("No video frame available.")
            return
        reach["start_frame"] = cur
        reach["start_determined"] = True
        reach["start_determined_by"] = _gt_username()
        reach["start_determined_at"] = _gt_now_iso()
        # Re-sort because start moved; remember reach_id so we can re-select.
        rid = reach.get("reach_id")
        self.gt_reaches.sort(key=lambda r: r.get("start_frame", 0))
        self._gt_mark_dirty()
        self._gt_populate_table()
        if rid is not None:
            self._gt_select_reach_by_id(rid)
        self._gt_flash(f"Reach #{rid} start set to frame {cur}")

    def _gt_set_selected_end(self) -> None:
        if self.gt_data is None:
            self._gt_flash("No GT loaded - click 'Load GT (.json)...' first.")
            return
        reach = self._gt_selected_reach()
        if reach is None:
            self._gt_flash("No GT reach selected - click a row in the GT table first.")
            return
        cur = self._gt_current_frame()
        if cur is None:
            self._gt_flash("No video frame available.")
            return
        reach["end_frame"] = cur
        reach["end_determined"] = True
        reach["end_determined_by"] = _gt_username()
        reach["end_determined_at"] = _gt_now_iso()
        self._gt_mark_dirty()
        rid = reach.get("reach_id")
        self._gt_populate_table()
        if rid is not None:
            self._gt_select_reach_by_id(rid)
        self._gt_flash(f"Reach #{rid} end set to frame {cur}")

    def _gt_toggle_selected_exclude(self) -> None:
        reach = self._gt_selected_reach()
        if reach is None:
            return
        if reach.get("exclude_from_analysis"):
            reach["exclude_from_analysis"] = False
            reach["exclude_reason"] = None
        else:
            from qtpy.QtWidgets import QInputDialog
            reason, ok = QInputDialog.getText(
                self,
                "Exclude reach",
                f"Reason for excluding reach #{reach.get('reach_id')}:",
                text="Not a valid reach",
            )
            if not ok:
                return
            reach["exclude_from_analysis"] = True
            reach["exclude_reason"] = reason or "No reason given"
        self._gt_mark_dirty()
        rid = reach.get("reach_id")
        self._gt_populate_table()
        if rid is not None:
            self._gt_select_reach_by_id(rid)

    def _gt_edit_selected_comment(self) -> None:
        reach = self._gt_selected_reach()
        if reach is None:
            return
        from qtpy.QtWidgets import QInputDialog
        current = reach.get("comment") or ""
        comment, ok = QInputDialog.getMultiLineText(
            self,
            "Reach comment",
            f"Comment for reach #{reach.get('reach_id')}:",
            text=current,
        )
        if not ok:
            return
        reach["comment"] = comment.strip() or None
        self._gt_mark_dirty()
        rid = reach.get("reach_id")
        self._gt_populate_table()
        if rid is not None:
            self._gt_select_reach_by_id(rid)

    # ── Per-row GT action handlers ─────────────────────────────────────

    def _gt_find_reach(self, rid: int) -> Optional[Dict[str, Any]]:
        for r in self.gt_reaches:
            if r.get("reach_id") == rid:
                return r
        return None

    def _gt_apply_start_for_rid(self, rid: int) -> None:
        """Set start_frame of the reach with this rid to current playhead.

        Used by per-row Set S buttons; bypasses table-selection lookup so
        focus changes can't make the action no-op.
        """
        if self.gt_data is None:
            self._gt_flash("No GT loaded.")
            return
        cur = self._gt_current_frame()
        if cur is None:
            self._gt_flash("No video frame available.")
            return
        reach = self._gt_find_reach(rid)
        if reach is None:
            self._gt_flash(f"Reach #{rid} not found in GT.")
            return
        reach["start_frame"] = cur
        reach["start_determined"] = True
        reach["start_determined_by"] = _gt_username()
        reach["start_determined_at"] = _gt_now_iso()
        self.gt_reaches.sort(key=lambda r: r.get("start_frame", 0))
        self._gt_mark_dirty()
        self._gt_populate_table()
        self._gt_select_reach_by_id(rid)
        self._gt_flash(f"Reach #{rid} start set to frame {cur}")

    def _gt_apply_end_for_rid(self, rid: int) -> None:
        """Set end_frame of the reach with this rid to current playhead.

        Used by per-row Set E buttons; bypasses table-selection lookup.
        """
        if self.gt_data is None:
            self._gt_flash("No GT loaded.")
            return
        cur = self._gt_current_frame()
        if cur is None:
            self._gt_flash("No video frame available.")
            return
        reach = self._gt_find_reach(rid)
        if reach is None:
            self._gt_flash(f"Reach #{rid} not found in GT.")
            return
        reach["end_frame"] = cur
        reach["end_determined"] = True
        reach["end_determined_by"] = _gt_username()
        reach["end_determined_at"] = _gt_now_iso()
        self._gt_mark_dirty()
        self._gt_populate_table()
        self._gt_select_reach_by_id(rid)
        self._gt_flash(f"Reach #{rid} end set to frame {cur}")

    # ── GT table rendering ─────────────────────────────────────────────

    def _gt_status_text(self, r: Dict[str, Any]) -> str:
        if r.get("exclude_from_analysis"):
            return f"🚫 EXCL: {r.get('exclude_reason') or '?'}"
        s = "✓" if r.get("start_determined") else "·"
        e = "✓" if r.get("end_determined") else "·"
        suffix = ""
        if r.get("comment"):
            c = r["comment"]
            suffix = f"  💬 {c[:40]}{'…' if len(c) > 40 else ''}"
        return f"{s} start / {e} end{suffix}"

    def _gt_populate_table(self) -> None:
        # Columns: Reach # | Start | Set S | End | Set E | Status | Jump
        self.gt_table.setSortingEnabled(False)
        self.gt_table.setRowCount(len(self.gt_reaches))
        for row_idx, r in enumerate(self.gt_reaches):
            rid = r.get("reach_id", row_idx)
            self._set_int_cell(self.gt_table, row_idx, 0, rid)
            self._set_int_cell(self.gt_table, row_idx, 1, r.get("start_frame"))

            # Set S button — captures rid by closure so it acts on this reach
            # regardless of table selection state.
            set_s_btn = QPushButton("Set S")
            set_s_btn.setMaximumWidth(56)
            set_s_btn.setToolTip(
                "Set this reach's start_frame to current playhead"
            )
            set_s_btn.clicked.connect(
                lambda _, x=rid: self._gt_apply_start_for_rid(x)
            )
            self.gt_table.setCellWidget(row_idx, 2, set_s_btn)

            self._set_int_cell(self.gt_table, row_idx, 3, r.get("end_frame"))

            set_e_btn = QPushButton("Set E")
            set_e_btn.setMaximumWidth(56)
            set_e_btn.setToolTip(
                "Set this reach's end_frame to current playhead"
            )
            set_e_btn.clicked.connect(
                lambda _, x=rid: self._gt_apply_end_for_rid(x)
            )
            self.gt_table.setCellWidget(row_idx, 4, set_e_btn)

            self.gt_table.setItem(
                row_idx, 5, QTableWidgetItem(self._gt_status_text(r))
            )
            if r.get("exclude_from_analysis"):
                # Gray-out only the TEXT cells (0, 1, 3, 5); button cells are
                # widgets, not items, so they don't have foreground colors.
                for c in (0, 1, 3, 5):
                    item = self.gt_table.item(row_idx, c)
                    if item is not None:
                        item.setForeground(QColor("#888"))

            jump_btn = QPushButton("Jump")
            jump_btn.setMaximumWidth(56)
            start_frame = r.get("start_frame")
            jump_btn.clicked.connect(
                lambda _, s=start_frame: self._jump_to_frame(s)
            )
            self.gt_table.setCellWidget(row_idx, 6, jump_btn)

        self.gt_table.setSortingEnabled(True)
        self.gt_table.sortItems(1, Qt.AscendingOrder)
        self._gt_update_selected_label()
        # Keep the white GT-active marker in sync with edits. Recomputes
        # _gt_ranges from the current gt_reaches (respects exclude flag),
        # then re-evaluates marker visibility against the current playhead.
        self._refresh_gt_ranges()
        self._refresh_marker_visibility()

    def _gt_update_selected_label(self) -> None:
        # Read selection live (don't use the cache yet — this is what populates
        # it). If something IS selected, cache the reach_id for later resilience.
        rid: Optional[int] = None
        try:
            rows = self.gt_table.selectionModel().selectedRows()
            if rows:
                rid_item = self.gt_table.item(rows[0].row(), 0)
                if rid_item is not None:
                    try:
                        rid = int(rid_item.data(Qt.DisplayRole))
                    except (TypeError, ValueError):
                        rid = None
        except Exception:
            rid = None

        if rid is not None:
            self._gt_last_selected_rid = rid

        reach = None
        if rid is not None:
            for r in self.gt_reaches:
                if r.get("reach_id") == rid:
                    reach = r
                    break
        if reach is None:
            # If the live selection vanished but we have a cached rid that still
            # resolves to a real reach, surface that so the user can still see
            # which row their buttons will act on.
            cached = self._gt_last_selected_rid
            if cached is not None:
                for r in self.gt_reaches:
                    if r.get("reach_id") == cached:
                        reach = r
                        break
        if reach is None:
            self.gt_selected_label.setText("Selected: (none)")
            return
        rid = reach.get("reach_id", "?")
        s = reach.get("start_frame")
        e = reach.get("end_frame")
        extra = ""
        if reach.get("exclude_from_analysis"):
            extra = "  (EXCLUDED)"
        self.gt_selected_label.setText(
            f"Selected: #{rid} (frames {s}-{e}){extra}"
        )

    def _on_gt_table_double_clicked(self, item: QTableWidgetItem) -> None:
        row = item.row()
        try:
            start = int(self.gt_table.item(row, 1).data(Qt.DisplayRole))
        except (AttributeError, TypeError, ValueError):
            return
        self._jump_to_frame(start)

    # ── Tables ──────────────────────────────────────────────────────────

    # Colors used for row text in the unified events table.
    KIND_COLOR = {
        "FP": QColor("#ff6666"),  # red
        "FN": QColor("#6699ff"),  # blue
        "TP": QColor("#66ff88"),  # green
    }

    # Topology colors (added 2026-05-20 alongside the new event_types).
    # Used to color rows by topology when present in the manifest.
    # Falls back to KIND_COLOR for legacy manifests.
    TOPOLOGY_COLOR = {
        "TP":              QColor("#66ff88"),  # green
        "TOLERANCE_ERROR": QColor("#ffcc55"),  # amber
        "MERGED":          QColor("#ff9933"),  # orange
        "FRAGMENTED":      QColor("#cc66ff"),  # purple
        "FALSE_POSITIVE":  QColor("#ff6666"),  # red
        "FALSE_NEGATIVE":  QColor("#6699ff"),  # blue
        "COMPLEX":         QColor("#ff66cc"),  # magenta
    }
    # Row order for the per-video topology summary table.
    # COMPLEX intentionally omitted: matcher-aware manifests do not emit it
    # (N:M components decompose into per-event labels). The TOPOLOGY_COLOR
    # entry above is kept in case a legacy manifest still carries it on
    # individual rows.
    TOPOLOGY_ORDER = (
        "TP",
        "TOLERANCE_ERROR",
        "MERGED",
        "FRAGMENTED",
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
    )

    def _populate_tables(self) -> None:
        """Populate the unified events table.

        Combines FP / FN / TP into one chronological view sorted by anchor
        frame (FP/TP anchor = detector.start, FN anchor = gt.start). Rows
        are color-coded by topology when present (the new 2026-05-20 field),
        otherwise by legacy kind.

        Columns: Kind | Topology | Start | End | Category | Jump

        For non-trivial topologies (MERGED / FRAGMENTED / TOLERANCE_ERROR /
        COMPLEX) the Kind column is suppressed — legacy "FP"/"FN" labels
        aren't meaningful for those — and the Category column shows the
        paired side from the same connected component (GT range for algo
        rows, algo range for GT rows) with deltas the same way TPs do.
        """
        rows = []
        for ev in self.fp_events:
            det = ev.get("detector") or {}
            rows.append(
                {
                    "ev": ev,  # retained for component-companion lookup
                    "kind": "FP",
                    "topology": ev.get("topology"),
                    "topology_sub": ev.get("topology_sub"),
                    "component_id": ev.get("component_id"),
                    "start": det.get("start"),
                    "end": det.get("end"),
                    "category": ev.get("category") or "",
                    "anchor": det.get("start") or 0,
                }
            )
        for ev in self.fn_events:
            gt = ev.get("gt") or {}
            rows.append(
                {
                    "ev": ev,
                    "kind": "FN",
                    "topology": ev.get("topology"),
                    "topology_sub": ev.get("topology_sub"),
                    "component_id": ev.get("component_id"),
                    "start": gt.get("start"),
                    "end": gt.get("end"),
                    "category": ev.get("category") or "",
                    "anchor": gt.get("start") or 0,
                }
            )
        for ev in self.tp_events:
            det = ev.get("detector") or {}
            gt = ev.get("gt") or {}
            # Manifest stores start_delta and span_delta on TPs. The user
            # asked for end_delta, so compute it from the frame pairs.
            algo_end = det.get("end")
            gt_end = gt.get("end")
            end_delta = (
                algo_end - gt_end
                if (algo_end is not None and gt_end is not None)
                else None
            )
            start_delta = ev.get("start_delta")
            parts = []
            if gt.get("start") is not None and gt_end is not None:
                parts.append(f"GT: {gt.get('start')}-{gt_end}")
            if start_delta is not None:
                parts.append(f"Start Δ: {start_delta:+d}")
            if end_delta is not None:
                parts.append(f"End Δ: {end_delta:+d}")
            detail = "   ".join(parts)
            rows.append(
                {
                    "ev": ev,
                    "kind": "TP",
                    "topology": ev.get("topology"),
                    "topology_sub": ev.get("topology_sub"),
                    "component_id": ev.get("component_id"),
                    "start": det.get("start"),
                    "end": det.get("end"),
                    "category": detail,  # GT range + deltas, rendered bold/white
                    "anchor": det.get("start") or 0,
                    "_show_detail_bold": True,
                }
            )

        # Build component_id -> [row, ...] lookup for sibling resolution on
        # rows whose topology spans multiple manifest entries (MERGED,
        # FRAGMENTED, TOLERANCE_ERROR, COMPLEX).
        components_by_cid: Dict[Any, List[Dict[str, Any]]] = {}
        for r in rows:
            cid = r.get("component_id")
            if cid is None:
                continue
            components_by_cid.setdefault(cid, []).append(r)

        # For each multi-row-topology row, build a bold-white category text
        # describing the paired side(s) + deltas the same way TPs do.
        TOPOLOGY_PAIRED = {"TOLERANCE_ERROR", "MERGED", "FRAGMENTED", "COMPLEX"}
        for r in rows:
            topology = r.get("topology")
            if topology not in TOPOLOGY_PAIRED:
                continue
            r["category"] = self._build_companion_category_text(
                r, components_by_cid
            )
            r["_show_detail_bold"] = True

        # Collapse TOLERANCE_ERROR pairs to a single row. Keep the algo-side
        # row so Start/End columns hold the algo's frames (in the topology
        # yellow color), and the Category column carries
        # "GT: <gs>-<ge>   Start Δ: ...   End Δ: ..."
        # — the ground truth this algo output was matched against. The
        # GT-side row is redundant in that case. Drop it only when an
        # algo-side sibling exists; if missing, keep the GT row so we
        # don't silently lose the event.
        deduped: List[Dict[str, Any]] = []
        for r in rows:
            if r.get("topology") == "TOLERANCE_ERROR":
                ev = r.get("ev") or {}
                if not ev.get("detector"):  # this row is GT-side
                    cid = r.get("component_id")
                    siblings = components_by_cid.get(cid, [])
                    has_algo_side = any(
                        (s.get("ev") or {}).get("detector") for s in siblings
                    )
                    if has_algo_side:
                        continue  # drop redundant GT-side row
            deduped.append(r)
        rows = deduped

        # Collapse MERGED components to one row per merged GT. The first GT
        # row in each component carries the algo Start/End and a "MERGED"
        # Reach Type label; subsequent GT rows in the same component show
        # a single down-arrow (↓) in Reach Type and blank Start/End to
        # signal "same algo output as the row above". Category on every
        # row shows just "GT: <gs>-<ge>" — no deltas (user asked to drop
        # them for MERGED). Algo-side row is dropped since its frames are
        # already replicated on the first GT row.
        merged_rows: List[Dict[str, Any]] = []
        merged_first_seen: set = set()
        for r in rows:
            if r.get("topology") != "MERGED":
                merged_rows.append(r)
                continue
            ev = r.get("ev") or {}
            if ev.get("detector"):
                # Algo-side MERGED row — drop
                continue
            cid = r.get("component_id")
            siblings = components_by_cid.get(cid, [])
            algo_sibs = [
                s for s in siblings if (s.get("ev") or {}).get("detector")
            ]
            if not algo_sibs:
                # No algo to anchor against; keep the GT row as-is
                merged_rows.append(r)
                continue
            algo_det = (algo_sibs[0]["ev"] or {}).get("detector") or {}
            algo_s = algo_det.get("start")
            algo_e = algo_det.get("end")
            gt = ev.get("gt") or {}
            gt_s = gt.get("start")
            gt_e = gt.get("end")

            is_first = cid not in merged_first_seen
            if is_first:
                merged_first_seen.add(cid)
                r["start"] = algo_s
                r["end"] = algo_e
            else:
                r["start"] = None  # blank in display
                r["end"] = None
                r["_is_continuation"] = True

            # Use algo_s as the anchor so all rows of one MERGED component
            # sort together; Python sort is stable so within-component
            # order is preserved (smallest gt_start first since fn_events
            # are pre-sorted in _load_manifest).
            r["anchor"] = (
                algo_s if isinstance(algo_s, int)
                else (gt_s if isinstance(gt_s, int) else 0)
            )
            r["category"] = f"GT: {gt_s}-{gt_e}"
            r["_show_detail_bold"] = True
            merged_rows.append(r)
        rows = merged_rows

        # Collapse FRAGMENTED components — mirror of MERGED but inverted:
        # 1 GT split into N algo pieces. We keep all N algo-side rows and
        # drop the GT-side row. First algo row carries the GT's Start/End
        # in WHITE (because Start/End holds the ground-truth frames, not
        # the algo output for this topology). Subsequent algo rows are
        # ↓ continuations with blank Start/End. Category on every row
        # shows "Algo: <a_s>-<a_e>" — no deltas. Algo-side rows are
        # already sorted by det.start (fp_events are pre-sorted in
        # _load_manifest), so the first algo encountered per component
        # is the earliest piece.
        fragmented_rows: List[Dict[str, Any]] = []
        fragmented_first_seen: set = set()
        for r in rows:
            if r.get("topology") != "FRAGMENTED":
                fragmented_rows.append(r)
                continue
            ev = r.get("ev") or {}
            if not ev.get("detector"):
                # GT-side FRAGMENTED row — drop (its info is on each
                # algo row's Start/End via the lookup below).
                continue
            cid = r.get("component_id")
            siblings = components_by_cid.get(cid, [])
            gt_sibs = [
                s for s in siblings if (s.get("ev") or {}).get("gt")
            ]
            if not gt_sibs:
                fragmented_rows.append(r)
                continue
            gt = (gt_sibs[0]["ev"] or {}).get("gt") or {}
            gt_s = gt.get("start")
            gt_e = gt.get("end")
            det = ev.get("detector") or {}
            det_s = det.get("start")
            det_e = det.get("end")

            is_first = cid not in fragmented_first_seen
            if is_first:
                fragmented_first_seen.add(cid)
                r["start"] = gt_s
                r["end"] = gt_e
                # Flag tells the renderer to color Start/End white (GT
                # color) and the Category cell purple (algo color) —
                # opposite of the default algo-in-Start/End layout.
                r["_gt_in_start_end"] = True
            else:
                r["start"] = None
                r["end"] = None
                r["_is_continuation"] = True
                r["_gt_in_start_end"] = True  # so renderer keeps category purple

            r["anchor"] = (
                gt_s if isinstance(gt_s, int)
                else (det_s if isinstance(det_s, int) else 0)
            )
            r["category"] = f"Algo: {det_s}-{det_e}"
            # Algo info is in Category for this topology; bold but the
            # color stays at the topology color (purple), not white. The
            # renderer handles this via _gt_in_start_end.
            r["_show_detail_bold"] = True
            fragmented_rows.append(r)
        rows = fragmented_rows

        rows.sort(key=lambda r: r["anchor"])

        # Topology sub-labels that are redundant with the visual rendering.
        # TOLERANCE_ERROR: (start_off) etc. is obvious from the deltas.
        # MERGED: (N_gt) is obvious from the down-arrow continuation rows
        # rendered below.
        TOPOLOGY_SUB_REDUNDANT = {"TOLERANCE_ERROR", "MERGED"}

        self.events_table.setSortingEnabled(False)
        self.events_table.setRowCount(len(rows))
        for row_idx, d in enumerate(rows):
            topology = d.get("topology")
            # Prefer topology color; fall back to kind color for older
            # manifests without the topology field.
            color = self.TOPOLOGY_COLOR.get(topology) if topology else None
            if color is None:
                color = self.KIND_COLOR.get(d["kind"], QColor("#cccccc"))

            # Col 0: Reach Type. Continuation rows of a multi-row component
            # (MERGED ↓, FRAGMENTED ↓) render as a single down-arrow
            # signalling "same component as the row above"; everything
            # else uses the topology label (with topology_sub when not
            # redundant).
            is_continuation = d.get("_is_continuation")
            if is_continuation:
                topology_text = "↓"
            else:
                topology_text = topology or ""
                sub = d.get("topology_sub")
                if (
                    topology
                    and sub
                    and topology not in TOPOLOGY_SUB_REDUNDANT
                ):
                    topology_text = f"{topology} ({sub})"
            top_item = QTableWidgetItem(topology_text)
            top_item.setForeground(color)
            tf = top_item.font()
            tf.setBold(True)
            top_item.setFont(tf)
            cid = d.get("component_id")
            if cid is not None:
                top_item.setToolTip(f"component_id: {cid}")
            self.events_table.setItem(row_idx, 0, top_item)

            # Col 1/2: Start, End. Blank for ↓ continuation rows.
            if d.get("start") is None:
                self.events_table.setItem(row_idx, 1, QTableWidgetItem(""))
            else:
                self._set_int_cell(self.events_table, row_idx, 1, d["start"])
            if d.get("end") is None:
                self.events_table.setItem(row_idx, 2, QTableWidgetItem(""))
            else:
                self._set_int_cell(self.events_table, row_idx, 2, d["end"])

            # Col 3: Category
            self.events_table.setItem(
                row_idx, 3, QTableWidgetItem(str(d["category"]))
            )

            # Color rule: algo content uses the topology color, GT content
            # uses white. For most topologies the algo lives in Start/End
            # and the GT lives in Category — so Start/End get the topology
            # color and Category goes bold-white. FRAGMENTED inverts this:
            # the GT is in Start/End (since it's the "real" reach being
            # fragmented) and algo pieces are in Category, so Start/End
            # render white and Category keeps the topology color.
            gt_in_start_end = d.get("_gt_in_start_end")
            for c in (1, 2, 3):
                item = self.events_table.item(row_idx, c)
                if item is None:
                    continue
                if gt_in_start_end:
                    if c in (1, 2):
                        item.setForeground(QColor("#ffffff"))
                    else:
                        item.setForeground(color)
                else:
                    item.setForeground(color)

            # Bold-white Category for rows that carry GT pairing detail
            # (TPs, MERGED, TOLERANCE_ERROR, COMPLEX rows where the
            # Category holds the GT side). Skip when _gt_in_start_end
            # because in that layout Category holds algo info, not GT.
            if d.get("_show_detail_bold") and not gt_in_start_end:
                cat_item = self.events_table.item(row_idx, 3)
                if cat_item is not None:
                    cat_item.setForeground(QColor("#ffffff"))
                    f = cat_item.font()
                    f.setBold(True)
                    cat_item.setFont(f)
            elif d.get("_show_detail_bold") and gt_in_start_end:
                # FRAGMENTED-style row: bold Category but keep its color
                # at the topology color (purple).
                cat_item = self.events_table.item(row_idx, 3)
                if cat_item is not None:
                    f = cat_item.font()
                    f.setBold(True)
                    cat_item.setFont(f)

            # Col 4: Jump
            jump_btn = QPushButton("Jump")
            anchor = d["anchor"]
            jump_btn.clicked.connect(lambda _, s=anchor: self._jump_to_frame(s))
            self.events_table.setCellWidget(row_idx, 4, jump_btn)
        # Leave sortingEnabled at False (the value set before the populate
        # loop). Rows are already in the right Python-side anchor order —
        # components stay grouped. Any column-based sort would scatter the
        # MERGED ↓ and FRAGMENTED ↓ continuation rows away from their
        # anchors because those rows have blank Start/End cells that Qt
        # sorts to the top.

        self._refresh_topology_summary()

    def _build_companion_category_text(
        self,
        row: Dict[str, Any],
        components_by_cid: Dict[Any, List[Dict[str, Any]]],
    ) -> str:
        """Render the paired-side description for a multi-row topology.

        For algo-side rows (have detector, no gt) the companion(s) are the
        GT sides in the same component. For GT-side rows the companion is
        the algo side. When exactly one companion exists we render Start
        Delta / End Delta the same way TPs do; when multiple companions
        exist (MERGED on algo side, FRAGMENTED on GT side, COMPLEX) we
        list each companion's frame range without deltas.
        """
        ev = row.get("ev") or {}
        cid = row.get("component_id")
        if cid is None:
            return row.get("category") or ""
        siblings = [
            s for s in components_by_cid.get(cid, [])
            if s is not row
        ]
        det = ev.get("detector") or {}
        gt = ev.get("gt") or {}
        is_algo_side = bool(det)

        def _frame_range(side: Dict[str, Any]) -> str:
            return f"{side.get('start')}-{side.get('end')}"

        def _delta_pair(a: Optional[int], b: Optional[int]) -> Optional[int]:
            if isinstance(a, int) and isinstance(b, int):
                return a - b
            return None

        if is_algo_side:
            gt_sibs = [s for s in siblings if (s.get("ev") or {}).get("gt")]
            if not gt_sibs:
                return row.get("category") or ""
            if len(gt_sibs) == 1:
                companion_gt = (gt_sibs[0]["ev"] or {}).get("gt") or {}
                parts = [f"GT: {_frame_range(companion_gt)}"]
                sd = _delta_pair(det.get("start"), companion_gt.get("start"))
                ed = _delta_pair(det.get("end"), companion_gt.get("end"))
                if sd is not None:
                    parts.append(f"Start Δ: {sd:+d}")
                if ed is not None:
                    parts.append(f"End Δ: {ed:+d}")
                return "   ".join(parts)
            ranges = ", ".join(
                _frame_range((s["ev"] or {}).get("gt") or {})
                for s in gt_sibs
            )
            return f"GT ({len(gt_sibs)}): {ranges}"
        else:
            algo_sibs = [
                s for s in siblings if (s.get("ev") or {}).get("detector")
            ]
            if not algo_sibs:
                return row.get("category") or ""
            if len(algo_sibs) == 1:
                companion_det = (
                    (algo_sibs[0]["ev"] or {}).get("detector") or {}
                )
                parts = [f"Algo: {_frame_range(companion_det)}"]
                sd = _delta_pair(
                    companion_det.get("start"), gt.get("start")
                )
                ed = _delta_pair(companion_det.get("end"), gt.get("end"))
                if sd is not None:
                    parts.append(f"Start Δ: {sd:+d}")
                if ed is not None:
                    parts.append(f"End Δ: {ed:+d}")
                return "   ".join(parts)
            ranges = ", ".join(
                _frame_range((s["ev"] or {}).get("detector") or {})
                for s in algo_sibs
            )
            return f"Algo ({len(algo_sibs)}): {ranges}"

    @staticmethod
    def _set_int_cell(table: QTableWidget, row: int, col: int, value: Any) -> None:
        """Populate a cell with an int value while keeping numeric sort order."""
        item = QTableWidgetItem()
        try:
            v = int(value)
            item.setData(Qt.DisplayRole, v)
        except (TypeError, ValueError):
            item.setData(Qt.DisplayRole, str(value))
        table.setItem(row, col, item)

    def _on_events_table_double_clicked(self, item: QTableWidgetItem) -> None:
        row = item.row()
        # Column 2 is "Start" (after the Topology column was inserted at 1);
        # that's the anchor for jumping regardless of kind.
        try:
            start = int(self.events_table.item(row, 2).data(Qt.DisplayRole))
        except (AttributeError, TypeError, ValueError):
            return
        self._jump_to_frame(start)

    # ── Marker layers ───────────────────────────────────────────────────

    def _rebuild_marker_layers(self) -> None:
        """Recreate napari Shapes layers for FP / TP / FN markers.

        Each kind gets exactly ONE rectangle at its canvas position. The
        whole layer's ``visible`` attribute is toggled per frame in
        ``_on_frame_change`` based on whether the current frame is inside
        any event of that kind. (Earlier versions used a per-shape
        ``shown`` array but that doesn't trigger a redraw on some napari
        versions, leaving the squares stuck on the canvas.)
        """
        # Remove any previous instances
        for attr in (
            "fp_shapes_layer",
            "tp_shapes_layer",
            "fn_shapes_layer",
            "gt_shapes_layer",
            "tol_shapes_layer",
            "merged_shapes_layer",
            "fragmented_shapes_layer",
            "complex_shapes_layer",
        ):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._fp_ranges = []
        self._tp_ranges = []
        self._fn_ranges = []
        self._gt_ranges = []
        self._tol_ranges = []
        self._merged_ranges = []
        self._fragmented_ranges = []
        self._complex_ranges = []

        if not self.events:
            return

        # Determine video height for bottom-left anchoring. Fall back to a
        # sensible default if the video isn't loaded yet.
        height = _DEFAULT_VIDEO_H
        if self.video_layer is not None:
            try:
                shape = self.video_layer.data.shape
                if len(shape) >= 3:
                    height = int(shape[1])  # (T, H, W) or (T, H, W, C)
            except Exception:
                pass

        # Bottom-left placement: anchor the bottom of each square MARKER_PAD
        # px above the canvas bottom, and start the leftmost square MARKER_PAD
        # px from the left edge.
        y_bot = height - MARKER_PAD
        y_top = y_bot - MARKER_SIZE
        # Layout: FP, TP, FN, GT, TOL, MERGED, FRAGMENTED, COMPLEX (left to right)
        fp_xl = MARKER_PAD
        fp_xr = fp_xl + MARKER_SIZE
        tp_xl = fp_xr + MARKER_GAP
        tp_xr = tp_xl + MARKER_SIZE
        fn_xl = tp_xr + MARKER_GAP
        fn_xr = fn_xl + MARKER_SIZE
        gt_xl = fn_xr + MARKER_GAP
        gt_xr = gt_xl + MARKER_SIZE
        tol_xl = gt_xr + MARKER_GAP
        tol_xr = tol_xl + MARKER_SIZE
        merged_xl = tol_xr + MARKER_GAP
        merged_xr = merged_xl + MARKER_SIZE
        fragmented_xl = merged_xr + MARKER_GAP
        fragmented_xr = fragmented_xl + MARKER_SIZE
        complex_xl = fragmented_xr + MARKER_GAP
        complex_xr = complex_xl + MARKER_SIZE

        # Topology-scoped event filters. An event is "true FP" only when
        # topology == FALSE_POSITIVE (or topology absent for legacy
        # manifests, fall back to kind). Same logic for TP and FN. This
        # keeps FP/FN markers from firing on the algo/GT halves of
        # TOLERANCE_ERROR / MERGED / FRAGMENTED / COMPLEX components —
        # those get their own topology-specific markers below.
        def _is_true_kind(ev: Dict[str, Any], kind: str, topo: str) -> bool:
            t = ev.get("topology")
            if t is None:
                return ev.get("kind") == kind
            return t == topo

        true_fp_events = [
            e for e in self.fp_events
            if _is_true_kind(e, "FP", "FALSE_POSITIVE")
        ]
        true_tp_events = [
            e for e in self.tp_events
            if _is_true_kind(e, "TP", "TP")
        ]
        true_fn_events = [
            e for e in self.fn_events
            if _is_true_kind(e, "FN", "FALSE_NEGATIVE")
        ]

        if true_fp_events:
            self.fp_shapes_layer = self.viewer.add_shapes(
                [_rect_corners(y_top, y_bot, fp_xl, fp_xr)],
                shape_type="rectangle",
                face_color=[FP_FACE],
                edge_color=[FP_EDGE],
                edge_width=2,
                opacity=1.0,
                name="FP markers",
            )
            self.fp_shapes_layer.visible = False
            self._fp_ranges = [
                (
                    (e.get("detector") or {}).get("start"),
                    (e.get("detector") or {}).get("end"),
                )
                for e in true_fp_events
            ]

        if true_tp_events:
            self.tp_shapes_layer = self.viewer.add_shapes(
                [_rect_corners(y_top, y_bot, tp_xl, tp_xr)],
                shape_type="rectangle",
                face_color=[TP_FACE],
                edge_color=[TP_EDGE],
                edge_width=2,
                opacity=1.0,
                name="TP markers",
            )
            self.tp_shapes_layer.visible = False
            self._tp_ranges = [
                (
                    (e.get("detector") or {}).get("start"),
                    (e.get("detector") or {}).get("end"),
                )
                for e in true_tp_events
            ]

        if true_fn_events:
            self.fn_shapes_layer = self.viewer.add_shapes(
                [_rect_corners(y_top, y_bot, fn_xl, fn_xr)],
                shape_type="rectangle",
                face_color=[FN_FACE],
                edge_color=[FN_EDGE],
                edge_width=2,
                opacity=1.0,
                name="FN markers",
            )
            self.fn_shapes_layer.visible = False
            self._fn_ranges = [
                (
                    (e.get("gt") or {}).get("start"),
                    (e.get("gt") or {}).get("end"),
                )
                for e in true_fn_events
            ]

        # Always create the GT-active layer so we can show GT coverage even
        # when there are no FN/TP events (e.g. clean video, or when GT has
        # been edited after manifest generation). Visibility is driven by
        # whether any GT reach is active at the current frame.
        self.gt_shapes_layer = self.viewer.add_shapes(
            [_rect_corners(y_top, y_bot, gt_xl, gt_xr)],
            shape_type="rectangle",
            face_color=[GT_FACE],
            edge_color=[GT_EDGE],
            edge_width=2,
            opacity=1.0,
            name="GT markers",
        )
        self.gt_shapes_layer.visible = False
        self._refresh_gt_ranges()

        # Topology-specific marker layers. Each marker fires only while the
        # algo is actually emitting output for an event of that topology —
        # i.e. the playhead must be inside an algo (detector) window. GT
        # windows are intentionally NOT unioned in: the marker represents
        # "the algo is making this kind of error right now". A TOLERANCE_
        # ERROR whose GT begins at 7645 but whose algo begins at 7648
        # shouldn't pop the marker at frame 7647.
        def _topology_ranges(target: str) -> List[tuple]:
            out: List[tuple] = []
            for ev in self.events:
                if ev.get("topology") != target:
                    continue
                det = ev.get("detector") or {}
                if (
                    det.get("start") is not None
                    and det.get("end") is not None
                ):
                    out.append((det["start"], det["end"]))
            return out

        topo_spec = [
            ("tol_shapes_layer", "_tol_ranges", "TOLERANCE_ERROR",
             tol_xl, tol_xr, TOL_FACE, TOL_EDGE, "TOL markers"),
            ("merged_shapes_layer", "_merged_ranges", "MERGED",
             merged_xl, merged_xr, MERGED_MARK_FACE, MERGED_MARK_EDGE,
             "MERGED markers"),
            ("fragmented_shapes_layer", "_fragmented_ranges", "FRAGMENTED",
             fragmented_xl, fragmented_xr,
             FRAGMENTED_MARK_FACE, FRAGMENTED_MARK_EDGE,
             "FRAGMENTED markers"),
            ("complex_shapes_layer", "_complex_ranges", "COMPLEX",
             complex_xl, complex_xr, COMPLEX_FACE, COMPLEX_EDGE,
             "COMPLEX markers"),
        ]
        for (
            layer_attr, ranges_attr, target_topo,
            xl, xr, face, edge, name,
        ) in topo_spec:
            ranges = _topology_ranges(target_topo)
            if not ranges:
                continue
            layer = self.viewer.add_shapes(
                [_rect_corners(y_top, y_bot, xl, xr)],
                shape_type="rectangle",
                face_color=[face],
                edge_color=[edge],
                edge_width=2,
                opacity=1.0,
                name=name,
            )
            layer.visible = False
            setattr(self, layer_attr, layer)
            setattr(self, ranges_attr, ranges)

        self._move_marker_layers_to_top()
        # Trigger a first paint pass against the current playhead.
        self._on_frame_change()

    def _refresh_gt_ranges(self) -> None:
        """Recompute the white-marker windows from current GT state.

        Prefers self.gt_reaches (live, reflects edits + exclude flag). Falls
        back to manifest event GT spans when no GT file is loaded. Excluded
        reaches are filtered out either way.
        """
        ranges: List[tuple] = []
        if self.gt_reaches:
            for r in self.gt_reaches:
                if r.get("exclude_from_analysis"):
                    continue
                s = r.get("start_frame")
                e = r.get("end_frame")
                if s is None or e is None:
                    continue
                ranges.append((s, e))
        else:
            seen: set = set()
            for ev in self.events:
                gt = ev.get("gt")
                if not gt:
                    continue
                s = gt.get("start")
                e = gt.get("end")
                if s is None or e is None:
                    continue
                key = (s, e)
                if key in seen:
                    continue
                seen.add(key)
                ranges.append((s, e))
        self._gt_ranges = ranges

    def _move_marker_layers_to_top(self) -> None:
        """Ensure marker layers are above the video so the rectangles aren't hidden."""
        for layer in (
            self.fp_shapes_layer,
            self.tp_shapes_layer,
            self.fn_shapes_layer,
            self.gt_shapes_layer,
            self.tol_shapes_layer,
            self.merged_shapes_layer,
            self.fragmented_shapes_layer,
            self.complex_shapes_layer,
        ):
            if layer is None:
                continue
            try:
                idx = self.viewer.layers.index(layer)
                self.viewer.layers.move(idx, -1)
            except Exception:
                pass

    def _refresh_marker_visibility(self) -> None:
        # Per-frame visibility is recomputed in _on_frame_change, so a
        # checkbox change just triggers a fresh evaluation against the
        # current playhead frame.
        self._on_frame_change()

    # ── Playback ───────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.play_btn.setText("Play")
        else:
            if self.n_frames == 0:
                return
            self.is_playing = True
            interval = max(1, int(1000 / (self.fps * self.playback_speed)))
            self.playback_timer.start(interval)
            self.play_btn.setText("Pause")

    def _stop(self) -> None:
        """Stop playback and rewind to frame 0."""
        self.is_playing = False
        self.playback_timer.stop()
        self.play_btn.setText("Play")
        if self.n_frames > 0:
            self.viewer.dims.set_current_step(0, 0)

    def _playback_step(self) -> None:
        if self.n_frames == 0:
            return
        current = self.viewer.dims.current_step[0]
        next_frame = current + self.playback_speed
        if next_frame >= self.n_frames:
            next_frame = 0
        self.viewer.dims.set_current_step(0, next_frame)

    def _set_speed_direct(self, multiplier: int) -> None:
        self.playback_speed = multiplier
        for mult in (1, 2, 4, 8, 16):
            btn = getattr(self, f"_speed_btn_{mult}", None)
            if btn is not None:
                btn.setChecked(mult == multiplier)
        if self.is_playing:
            interval = max(1, int(1000 / (self.fps * self.playback_speed)))
            self.playback_timer.setInterval(interval)

    def _setup_keybindings(self) -> None:
        """Register Space=play/pause, S=set GT start, E=set GT end.

        Called after the widget is docked. S/E mirror the GT review tool's
        muscle memory; the Stop button still works by click but no longer
        has a keyboard shortcut in this widget.
        """

        @self.viewer.bind_key("Space", overwrite=True)
        def _kb_toggle_play(viewer):
            self._toggle_play()

        @self.viewer.bind_key("s", overwrite=True)
        def _kb_set_start(viewer):
            self._gt_set_selected_start()

        @self.viewer.bind_key("e", overwrite=True)
        def _kb_set_end(viewer):
            self._gt_set_selected_end()

    # ── Frame-change driver ────────────────────────────────────────────

    def _on_frame_change(self, _event: Any = None) -> None:
        # Update Frame / Time labels (always, regardless of markers)
        try:
            cur = int(self.viewer.dims.current_step[0])
        except Exception:
            cur = 0
        if self.n_frames > 0:
            self.frame_label.setText(f"Frame: {cur} / {self.n_frames}")
            time_sec = cur / max(self.fps, 1e-6)
            mins = int(time_sec // 60)
            secs = time_sec % 60
            self.time_label.setText(f"Time: {mins}:{secs:05.2f}")
        else:
            self.frame_label.setText(f"Frame: {cur} / —")
            self.time_label.setText("Time: —")

        # Bail early on marker work if nothing to show
        if (
            self.fp_shapes_layer is None
            and self.tp_shapes_layer is None
            and self.fn_shapes_layer is None
            and self.gt_shapes_layer is None
            and self.tol_shapes_layer is None
            and self.merged_shapes_layer is None
            and self.fragmented_shapes_layer is None
            and self.complex_shapes_layer is None
        ):
            self._update_status_label(None, None, None)
            return

        def _find_active(ranges, events):
            """Return the first event whose [start, end] window contains cur."""
            for (s, e), ev in zip(ranges, events):
                if s is None or e is None:
                    continue
                if s <= cur <= e:
                    return ev
            return None

        active_fp = _find_active(self._fp_ranges, self.fp_events)
        active_tp = _find_active(self._tp_ranges, self.tp_events)
        active_fn = _find_active(self._fn_ranges, self.fn_events)
        # GT marker only needs to know IF any range contains cur; doesn't
        # carry per-event data, so a simpler scan is enough.
        gt_active = any(
            s is not None and e is not None and s <= cur <= e
            for (s, e) in self._gt_ranges
        )
        # Boolean "any range contains cur" for each topology marker layer.
        def _any_in(ranges: List[tuple]) -> bool:
            return any(
                s is not None and e is not None and s <= cur <= e
                for (s, e) in ranges
            )
        tol_active = _any_in(self._tol_ranges)
        merged_active = _any_in(self._merged_ranges)
        fragmented_active = _any_in(self._fragmented_ranges)
        complex_active = _any_in(self._complex_ranges)

        # Only assign .visible when the value actually changes. Setting it
        # to the same value still triggers napari's events.visible(), which
        # cascades into a vispy reorder — and if this handler ran during a
        # mid-flight layer insertion (e.g. add_points firing dims.range
        # which re-enters _on_frame_change), the not-yet-registered layer
        # blows up with KeyError in layer_to_visual.
        def _toggle(layer, should_show):
            if layer is None:
                return
            try:
                if bool(layer.visible) != bool(should_show):
                    layer.visible = should_show
            except Exception:
                # napari can still throw during re-entrant redraws; never
                # let the frame-change handler crash the load.
                pass

        _toggle(self.fp_shapes_layer, active_fp is not None)
        _toggle(self.tp_shapes_layer, active_tp is not None)
        _toggle(self.fn_shapes_layer, active_fn is not None)
        _toggle(self.gt_shapes_layer, gt_active)
        _toggle(self.tol_shapes_layer, tol_active)
        _toggle(self.merged_shapes_layer, merged_active)
        _toggle(self.fragmented_shapes_layer, fragmented_active)
        _toggle(self.complex_shapes_layer, complex_active)

        self._update_status_label(active_fp, active_tp, active_fn)

    def _update_status_label(
        self,
        active_fp: Optional[Dict[str, Any]],
        active_tp: Optional[Dict[str, Any]],
        active_fn: Optional[Dict[str, Any]],
    ) -> None:
        # Priority: FP > FN > TP for which one fills the status panel
        # (errors are more interesting than matches when both happen)
        if active_fp is not None:
            det = active_fp.get("detector") or {}
            cat = active_fp.get("category") or ""
            self.status_label.setText(
                f"FP  ({det.get('start', '?')}–{det.get('end', '?')})  {cat}"
            )
            self.status_label.setStyleSheet(
                "padding: 6px; font-size: 13px; font-weight: bold; "
                "background: #3a1010; color: #ff8080;"
            )
        elif active_fn is not None:
            gt = active_fn.get("gt") or {}
            cat = active_fn.get("category") or ""
            self.status_label.setText(
                f"FN  GT ({gt.get('start', '?')}–{gt.get('end', '?')})  {cat}"
            )
            self.status_label.setStyleSheet(
                "padding: 6px; font-size: 13px; font-weight: bold; "
                "background: #10203a; color: #88aaff;"
            )
        elif active_tp is not None:
            det = active_tp.get("detector") or {}
            self.status_label.setText(
                f"TP  ({det.get('start', '?')}–{det.get('end', '?')})"
            )
            self.status_label.setStyleSheet(
                "padding: 6px; font-size: 13px; font-weight: bold; "
                "background: #103018; color: #88ffaa;"
            )
        else:
            self.status_label.setText("")
            self.status_label.setStyleSheet(
                "padding: 6px; font-size: 13px; font-weight: bold; "
                "background: #181818; color: #666;"
            )

    # ── Navigation ─────────────────────────────────────────────────────

    def _jump_to_frame(self, frame: Any) -> None:
        try:
            f = int(frame)
        except (TypeError, ValueError):
            return
        if self.n_frames:
            f = max(0, min(f, self.n_frames - 1))
        try:
            self.viewer.dims.set_current_step(0, f)
        except Exception:
            pass

    def _step_event(
        self, events: List[Dict[str, Any]], direction: int, source: str
    ) -> None:
        """Jump to next/prev event of a given kind based on its anchor frame.

        ``source`` is ``"detector"`` for FP (use detector.start) or ``"gt"``
        for FN (use gt.start).
        """
        if not events:
            return
        try:
            cur = int(self.viewer.dims.current_step[0])
        except Exception:
            cur = 0

        anchors = []
        for e in events:
            slot = e.get(source) or {}
            s = slot.get("start")
            if s is not None:
                anchors.append(int(s))
        if not anchors:
            return
        anchors.sort()

        target: Optional[int] = None
        if direction > 0:
            for a in anchors:
                if a > cur:
                    target = a
                    break
            if target is None:
                target = anchors[0]  # wrap
        else:
            for a in reversed(anchors):
                if a < cur:
                    target = a
                    break
            if target is None:
                target = anchors[-1]  # wrap
        self._jump_to_frame(target)


def main() -> None:  # pragma: no cover - manual entry-point
    """Standalone launcher: open a napari viewer with the widget docked."""
    viewer = napari.Viewer()
    widget = FPFNReviewWidget(viewer)
    viewer.window.add_dock_widget(widget, name="FP/FN Reach Review", area="right")
    # After docking, like the launcher does -- Space/S/E are dead code otherwise.
    widget._setup_keybindings()
    napari.run()


if __name__ == "__main__":  # pragma: no cover
    main()
