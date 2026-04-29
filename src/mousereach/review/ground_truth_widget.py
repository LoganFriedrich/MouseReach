"""
MouseReach Ground Truth Tool
============================

Create ground truth annotations for all three data types:
- Segment boundaries
- Reaches (start/end frames)
- Outcomes

Key design:
- NO VERIFY BUTTONS - when you SET a value, that IS the ground truth
- Algorithm outputs as starting point - correct them to create GT
- Single unified GT file format

This is NOT for quick review/accept workflow. Use the Review Tool for that.
This is for creating detailed ground truth for algorithm evaluation.

Usage:
    mousereach-unified-review video.mp4
    # Or via Plugins -> MouseReach -> GT Tool
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json
import cv2

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QScrollArea,
    QDialog, QTextBrowser, QComboBox, QSpinBox, QFrame,
    QSizePolicy, QCheckBox, QButtonGroup, QRadioButton,
    QMessageBox, QSplitter
)
from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QFont, QColor

import napari
from napari.utils.notifications import show_info, show_warning, show_error

from .unified_gt import (
    UnifiedGroundTruth, BoundaryGT, ReachGT, OutcomeGT,
    load_or_create_unified_gt, save_unified_gt, get_username, get_timestamp
)


HELP_TEXT = """
<h2>MouseReach Ground Truth Tool</h2>

<h3>Overview</h3>
<p>Create ground truth by setting values for all three annotation types:</p>
<ul>
<li><b>Segment Boundaries</b> - 21 frames marking trial transitions</li>
<li><b>Reaches</b> - Start/end frames for each reaching movement</li>
<li><b>Outcomes</b> - Pellet outcome classification per segment</li>
</ul>

<h3>Key Concepts</h3>
<ul>
<li><b>No Verify Buttons</b> - When you SET a value, that IS the ground truth</li>
<li><b>Algorithm as Starting Point</b> - Loads algorithm outputs, you correct them</li>
<li><b>Single GT File</b> - All annotations saved to unified ground truth file</li>
</ul>

<h3>Workflow</h3>
<ol>
<li>Jump to an item to see the current frame</li>
<li>If correct, move on (no action needed)</li>
<li>If incorrect, SET the correct value - that becomes GT</li>
<li>Save when done</li>
</ol>

<h3>Keyboard Shortcuts</h3>
<table>
<tr><td>Space</td><td>Play/Pause</td></tr>
<tr><td>Left/Right</td><td>-1/+1 frame</td></tr>
<tr><td>Shift+Left/Right</td><td>-10/+10 frames</td></tr>
<tr><td>Ctrl+Left/Right</td><td>-100/+100 frames</td></tr>
<tr><td>J/K</td><td>Next/Prev hand-visible frame</td></tr>
<tr><td>V</td><td>Set boundary frame</td></tr>
<tr><td>S</td><td>Set reach start frame</td></tr>
<tr><td>E</td><td>Set reach end frame</td></tr>
<tr><td>I</td><td>Set interaction frame</td></tr>
<tr><td>O</td><td>Set outcome known frame</td></tr>
<tr><td>Delete</td><td>Delete selected reach</td></tr>
<tr><td>Ctrl+S</td><td>Save progress</td></tr>
</table>
"""


class GroundTruthWidget(QWidget):
    """
    Ground Truth Tool - create GT by setting values (no verification buttons).

    Layout:
    - Sticky video player at top
    - Scrollable content below with three sections:
      1. Segment Boundaries (Jump + Set)
      2. Reaches (Jump + Set start/end)
      3. Outcomes (dropdown + Set frames)

    When you SET a value, that IS the ground truth. No separate verify step.
    """

    # Signals
    data_saved = Signal(Path)

    def __init__(self, napari_viewer: napari.Viewer, review_mode: bool = False):
        """
        Args:
            napari_viewer: The napari viewer instance
            review_mode: If True, operates as Review Tool (saves to algo files).
                        If False, operates as GT Tool (saves to unified GT file).
        """
        super().__init__()
        self.viewer = napari_viewer
        self.review_mode = review_mode

        # Video state
        self.video_path: Optional[Path] = None
        self.video_layer = None
        self.n_frames = 0
        self.fps = 60.0
        self.scale_factor = 1.0

        # Data
        self.gt: Optional[UnifiedGroundTruth] = None
        self.dlc_data = None  # For J/K hand-visible navigation
        self.dlc_df = None  # Full DLC dataframe for points overlay

        # Visualization layers
        self.points_layer = None  # DLC points overlay
        self.pillar_shapes_layer = None  # Pillar circle overlay

        # v4.0.0 screenshot defaults (overridden by CLI --screenshot-dir)
        self.screenshot_dir = None
        self._screenshot_segment = None

        # UI tracking - selected items for keyboard shortcuts
        self._current_boundary_idx = 0
        self._current_reach_idx = 0
        self._current_outcome_idx = 0
        self._selected_boundary: Optional[BoundaryGT] = None
        self._selected_reach: Optional[ReachGT] = None
        self._selected_outcome: Optional[OutcomeGT] = None

        # Playback
        self.is_playing = False
        self.playback_speed = 1.0
        self.playback_direction = 1
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        # Dropdown data
        self._dropdown_items: List[Tuple[str, Path, str]] = []

        # Build UI
        self._build_ui()
        self._setup_keybindings()
        self._populate_video_dropdown()

    def _show_help(self):
        """Show help dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Unified Review v2 - Help")
        dialog.setMinimumSize(550, 500)
        layout = QVBoxLayout()
        dialog.setLayout(layout)

        text = QTextBrowser()
        text.setHtml(HELP_TEXT)
        layout.addWidget(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec_()

    def _build_ui(self):
        """Build the main UI with sticky header and scrollable content."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        # === STICKY HEADER (Video + Navigation) ===
        self._build_sticky_header(main_layout)

        # === SCROLLABLE CONTENT ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(10)
        scroll_content.setLayout(scroll_layout)

        # Section 1: Segment Boundaries
        self._build_boundaries_section(scroll_layout)

        # Section 2: Reaches
        self._build_reaches_section(scroll_layout)

        # Section 3: Outcomes
        self._build_outcomes_section(scroll_layout)

        # Add stretch at bottom
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=1)

        # === FOOTER (Save buttons + Status) ===
        self._build_footer(main_layout)

        # Connect frame change
        self.viewer.dims.events.current_step.connect(self._on_frame_change)

    def _build_sticky_header(self, parent_layout: QVBoxLayout):
        """Build the sticky header with video selector and navigation."""
        header = QFrame()
        header.setFrameStyle(QFrame.StyledPanel)
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(5, 5, 5, 5)
        header.setLayout(header_layout)

        # Title row
        title_row = QHBoxLayout()
        tool_name = "Review Tool" if self.review_mode else "Ground Truth Tool"
        title = QLabel(f"<b>MouseReach {tool_name}</b>")
        title.setStyleSheet("font-size: 14px;")
        title_row.addWidget(title)
        title_row.addStretch()

        help_btn = QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.setToolTip("Help")
        help_btn.clicked.connect(self._show_help)
        title_row.addWidget(help_btn)
        header_layout.addLayout(title_row)

        # Video selector row
        video_row = QHBoxLayout()
        self.video_combo = QComboBox()
        self.video_combo.setPlaceholderText("Select video...")
        self.video_combo.currentIndexChanged.connect(self._on_video_selected)
        video_row.addWidget(self.video_combo, stretch=1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._populate_video_dropdown)
        video_row.addWidget(refresh_btn)
        header_layout.addLayout(video_row)

        # Video status
        self.video_label = QLabel("No video loaded")
        self.video_label.setStyleSheet("color: #888;")
        header_layout.addWidget(self.video_label)

        # Progress bar (hidden by default)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        header_layout.addWidget(self.progress)

        # Frame display
        frame_row = QHBoxLayout()
        self.frame_label = QLabel("Frame: -- / --")
        self.frame_label.setFont(QFont("", 12, QFont.Bold))
        frame_row.addWidget(self.frame_label)
        frame_row.addStretch()
        self.time_label = QLabel("Time: --:--")
        frame_row.addWidget(self.time_label)
        header_layout.addLayout(frame_row)

        # Playback controls
        play_row = QHBoxLayout()

        self.play_rev_btn = QPushButton("Rev")
        self.play_rev_btn.clicked.connect(self._play_reverse)
        self.play_rev_btn.setEnabled(False)
        self.play_rev_btn.setMaximumWidth(40)
        play_row.addWidget(self.play_rev_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._play_forward)
        self.play_btn.setEnabled(False)
        self.play_btn.setMaximumWidth(40)
        play_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_play)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMaximumWidth(40)
        play_row.addWidget(self.stop_btn)

        play_row.addStretch()

        # Speed buttons
        play_row.addWidget(QLabel("Speed:"))
        self.speed_buttons = {}
        for speed in [0.25, 0.5, 1, 2, 4, 8, 16]:
            if speed < 1:
                label = f"{speed}x"
            else:
                label = f"{int(speed)}x"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMaximumWidth(35)
            btn.clicked.connect(lambda _, s=speed: self._set_speed(s))
            self.speed_buttons[speed] = btn
            play_row.addWidget(btn)
        self.speed_buttons[1].setChecked(True)
        header_layout.addLayout(play_row)

        # Frame step row: segment nav + frame jumps
        step_row = QHBoxLayout()

        # Segment navigation
        prev_seg_btn = QPushButton("<<< -seg")
        prev_seg_btn.clicked.connect(self._jump_to_prev_segment)
        prev_seg_btn.setMaximumWidth(60)
        step_row.addWidget(prev_seg_btn)

        # Frame jump buttons: -100, -10, -1, +1, +10, +100
        for delta, label in [(-100, "<<"), (-10, "<"), (-1, "◀"), (1, "▶"), (10, ">"), (100, ">>")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, d=delta: self._jump_frames(d))
            btn.setMaximumWidth(30 if abs(delta) == 1 else 35)
            step_row.addWidget(btn)

        next_seg_btn = QPushButton("+seg >>>")
        next_seg_btn.clicked.connect(self._jump_to_next_segment)
        next_seg_btn.setMaximumWidth(60)
        step_row.addWidget(next_seg_btn)

        header_layout.addLayout(step_row)

        # Go to frame row
        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("Go to:"))
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(0, 999999)
        self.goto_spin.setMaximumWidth(80)
        goto_row.addWidget(self.goto_spin)

        goto_btn = QPushButton("Go")
        goto_btn.clicked.connect(self._goto_frame)
        goto_btn.setMaximumWidth(30)
        goto_row.addWidget(goto_btn)

        goto_row.addStretch()

        # J/K hand visible navigation
        goto_row.addWidget(QLabel("Hand visible:"))
        prev_hand_btn = QPushButton("< K")
        prev_hand_btn.clicked.connect(self._jump_to_prev_hand)
        prev_hand_btn.setMaximumWidth(35)
        prev_hand_btn.setToolTip("Jump to previous frame where hand is visible (K)")
        goto_row.addWidget(prev_hand_btn)

        next_hand_btn = QPushButton("J >")
        next_hand_btn.clicked.connect(self._jump_to_next_hand)
        next_hand_btn.setMaximumWidth(35)
        next_hand_btn.setToolTip("Jump to next frame where hand is visible (J)")
        goto_row.addWidget(next_hand_btn)

        # v4.0.0: screenshot button. Captures the napari canvas at the
        # current frame and opens a Save As dialog rooted at the
        # screenshot_dir provided by the launcher (or cwd otherwise).
        # Default filename uses the absolute video frame.
        screenshot_btn = QPushButton("Screenshot")
        screenshot_btn.clicked.connect(self._save_screenshot)
        screenshot_btn.setToolTip("Capture canvas; auto-name with absolute frame")
        goto_row.addWidget(screenshot_btn)

        header_layout.addLayout(goto_row)

        # Event navigation row: reach, segment bounds, interaction
        event_row = QHBoxLayout()

        # Segment bounds
        event_row.addWidget(QLabel("Seg:"))
        seg_start_btn = QPushButton("|< Start")
        seg_start_btn.clicked.connect(self._jump_to_segment_start)
        seg_start_btn.setMaximumWidth(55)
        seg_start_btn.setToolTip("Jump to current segment start frame")
        event_row.addWidget(seg_start_btn)

        seg_end_btn = QPushButton("End >|")
        seg_end_btn.clicked.connect(self._jump_to_segment_end)
        seg_end_btn.setMaximumWidth(55)
        seg_end_btn.setToolTip("Jump to current segment end frame")
        event_row.addWidget(seg_end_btn)

        event_row.addWidget(QLabel("  "))  # spacer

        # Reach navigation
        event_row.addWidget(QLabel("Reach:"))
        prev_reach_btn = QPushButton("< Prev")
        prev_reach_btn.clicked.connect(self._jump_to_prev_reach)
        prev_reach_btn.setMaximumWidth(50)
        prev_reach_btn.setToolTip("Jump to previous reach start")
        event_row.addWidget(prev_reach_btn)

        reach_start_btn = QPushButton("Start")
        reach_start_btn.clicked.connect(self._jump_to_reach_start)
        reach_start_btn.setMaximumWidth(45)
        reach_start_btn.setToolTip("Jump to current reach start frame")
        event_row.addWidget(reach_start_btn)

        reach_end_btn = QPushButton("End")
        reach_end_btn.clicked.connect(self._jump_to_reach_end)
        reach_end_btn.setMaximumWidth(40)
        reach_end_btn.setToolTip("Jump to current reach end frame")
        event_row.addWidget(reach_end_btn)

        next_reach_btn = QPushButton("Next >")
        next_reach_btn.clicked.connect(self._jump_to_next_reach)
        next_reach_btn.setMaximumWidth(50)
        next_reach_btn.setToolTip("Jump to next reach start")
        event_row.addWidget(next_reach_btn)

        event_row.addWidget(QLabel("  "))  # spacer

        # Interaction jump
        interaction_btn = QPushButton("Interaction (I)")
        interaction_btn.clicked.connect(self._jump_to_interaction)
        interaction_btn.setMaximumWidth(90)
        interaction_btn.setToolTip("Jump to pellet interaction frame (I)")
        event_row.addWidget(interaction_btn)

        event_row.addStretch()
        header_layout.addLayout(event_row)

        # Layer visibility toggles
        vis_row = QHBoxLayout()
        vis_row.addWidget(QLabel("Show:"))

        self.show_dlc_check = QCheckBox("DLC Points")
        self.show_dlc_check.setChecked(True)
        self.show_dlc_check.stateChanged.connect(
            lambda state: self._toggle_dlc_visibility(state == Qt.Checked)
        )
        vis_row.addWidget(self.show_dlc_check)

        self.show_pillar_check = QCheckBox("Pillar Circle")
        self.show_pillar_check.setChecked(True)
        self.show_pillar_check.stateChanged.connect(
            lambda state: self._toggle_pillar_visibility(state == Qt.Checked)
        )
        vis_row.addWidget(self.show_pillar_check)

        vis_row.addStretch()

        # Prominent "Add Reach" button - always visible in header
        self.add_reach_here_btn = QPushButton("+ Add Reach Here (A)")
        self.add_reach_here_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.add_reach_here_btn.setToolTip("Add a new reach starting at current frame (shortcut: A)")
        self.add_reach_here_btn.clicked.connect(self._add_reach)
        vis_row.addWidget(self.add_reach_here_btn)

        header_layout.addLayout(vis_row)

        parent_layout.addWidget(header)

    def _build_boundaries_section(self, parent_layout: QVBoxLayout):
        """Build the segment boundaries section."""
        section = QGroupBox("SEGMENT BOUNDARIES")
        section.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        layout = QVBoxLayout()
        section.setLayout(layout)

        # Progress label
        self.boundaries_progress = QLabel("0/0 determined")
        self.boundaries_progress.setStyleSheet("color: #888;")
        layout.addWidget(self.boundaries_progress)

        # Exhaustive status row
        exhaustive_row = QHBoxLayout()
        self.boundaries_exhaustive_label = QLabel("Not marked exhaustive")
        self.boundaries_exhaustive_label.setStyleSheet("color: #888; font-style: italic;")
        exhaustive_row.addWidget(self.boundaries_exhaustive_label)
        exhaustive_row.addStretch()
        self.boundaries_exhaustive_btn = QPushButton("Mark Exhaustive")
        self.boundaries_exhaustive_btn.setMaximumWidth(140)
        self.boundaries_exhaustive_btn.clicked.connect(lambda: self._toggle_exhaustive("segmentation"))
        exhaustive_row.addWidget(self.boundaries_exhaustive_btn)
        layout.addLayout(exhaustive_row)

        # Boundaries list container
        self.boundaries_container = QVBoxLayout()
        layout.addLayout(self.boundaries_container)

        # Filter checkbox
        self.boundaries_filter = QCheckBox("Show only incomplete")
        self.boundaries_filter.stateChanged.connect(self._refresh_boundaries_list)
        layout.addWidget(self.boundaries_filter)

        parent_layout.addWidget(section)

    def _build_reaches_section(self, parent_layout: QVBoxLayout):
        """Build the reaches section with split verification."""
        section = QGroupBox("REACHES")
        section.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        layout = QVBoxLayout()
        section.setLayout(layout)

        # Progress label
        self.reaches_progress = QLabel("0/0 fully determined")
        self.reaches_progress.setStyleSheet("color: #888;")
        layout.addWidget(self.reaches_progress)

        # Exhaustive status row
        exhaustive_row = QHBoxLayout()
        self.reaches_exhaustive_label = QLabel("Not marked exhaustive")
        self.reaches_exhaustive_label.setStyleSheet("color: #888; font-style: italic;")
        exhaustive_row.addWidget(self.reaches_exhaustive_label)
        exhaustive_row.addStretch()
        self.reaches_exhaustive_btn = QPushButton("Mark Exhaustive")
        self.reaches_exhaustive_btn.setMaximumWidth(140)
        self.reaches_exhaustive_btn.clicked.connect(lambda: self._toggle_exhaustive("reaches"))
        exhaustive_row.addWidget(self.reaches_exhaustive_btn)
        layout.addLayout(exhaustive_row)

        # Reaches list container
        self.reaches_container = QVBoxLayout()
        layout.addLayout(self.reaches_container)

        # Filter and add reach buttons
        filter_row = QHBoxLayout()
        self.reaches_filter = QCheckBox("Show only incomplete")
        self.reaches_filter.stateChanged.connect(self._refresh_reaches_list)
        filter_row.addWidget(self.reaches_filter)

        filter_row.addStretch()

        add_reach_btn = QPushButton("+ Add Reach")
        add_reach_btn.clicked.connect(self._add_reach)
        filter_row.addWidget(add_reach_btn)

        layout.addLayout(filter_row)

        parent_layout.addWidget(section)

    def _build_outcomes_section(self, parent_layout: QVBoxLayout):
        """Build the outcomes section."""
        section = QGroupBox("OUTCOMES")
        section.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        layout = QVBoxLayout()
        section.setLayout(layout)

        # Progress label
        self.outcomes_progress = QLabel("0/0 determined")
        self.outcomes_progress.setStyleSheet("color: #888;")
        layout.addWidget(self.outcomes_progress)

        # Exhaustive status row
        exhaustive_row = QHBoxLayout()
        self.outcomes_exhaustive_label = QLabel("Not marked exhaustive")
        self.outcomes_exhaustive_label.setStyleSheet("color: #888; font-style: italic;")
        exhaustive_row.addWidget(self.outcomes_exhaustive_label)
        exhaustive_row.addStretch()
        self.outcomes_exhaustive_btn = QPushButton("Mark Exhaustive")
        self.outcomes_exhaustive_btn.setMaximumWidth(140)
        self.outcomes_exhaustive_btn.clicked.connect(lambda: self._toggle_exhaustive("outcomes"))
        exhaustive_row.addWidget(self.outcomes_exhaustive_btn)
        layout.addLayout(exhaustive_row)

        # Outcomes list container
        self.outcomes_container = QVBoxLayout()
        layout.addLayout(self.outcomes_container)

        # Filter checkbox
        self.outcomes_filter = QCheckBox("Show only incomplete")
        self.outcomes_filter.stateChanged.connect(self._refresh_outcomes_list)
        layout.addWidget(self.outcomes_filter)

        parent_layout.addWidget(section)

    def _build_footer(self, parent_layout: QVBoxLayout):
        """Build the footer with save buttons and status."""
        footer = QFrame()
        footer.setFrameStyle(QFrame.StyledPanel)
        footer_layout = QVBoxLayout()
        footer_layout.setContentsMargins(5, 5, 5, 5)
        footer.setLayout(footer_layout)

        # Save buttons
        save_row = QHBoxLayout()

        if self.review_mode:
            # Review Tool: single "Save & Continue" button that saves to algo files
            save_btn = QPushButton("✓ Save & Continue")
            save_btn.setToolTip("Save corrections to algorithm output files")
            save_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px 16px;
                }
                QPushButton:hover { background-color: #45a049; }
            """)
            save_btn.clicked.connect(self._save_to_algo_files)
            save_row.addWidget(save_btn)
        else:
            # GT Tool: Save Progress + Save as GT
            save_progress_btn = QPushButton("Save Progress")
            save_progress_btn.setToolTip("Save current state (can resume later)")
            save_progress_btn.clicked.connect(self._save_progress)
            save_row.addWidget(save_progress_btn)

            save_row.addStretch()

            save_gt_btn = QPushButton("Save as Ground Truth")
            save_gt_btn.setToolTip("Save as finalized ground truth file")
            save_gt_btn.setStyleSheet("background-color: #2d5016;")
            save_gt_btn.clicked.connect(self._save_ground_truth)
            save_row.addWidget(save_gt_btn)

        footer_layout.addLayout(save_row)

        # Status label
        self.status_label = QLabel("Select a video to begin")
        self.status_label.setStyleSheet("color: #888;")
        footer_layout.addWidget(self.status_label)

        # v4.0.0: live frame info panel -- updates as you scrub. Content is
        # context-aware via self.info_panel_mode (set by CLI; defaults to
        # 'outcome' when launched with --segment + --algo-dir pointing at
        # pellet outcomes, otherwise 'general').
        self.info_panel_mode = getattr(self, 'info_panel_mode', 'general')
        self.frame_info_label = QLabel("")
        self.frame_info_label.setStyleSheet(
            "color: #ddd; background-color: #222; padding: 6px; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 11px;"
        )
        self.frame_info_label.setWordWrap(True)
        footer_layout.addWidget(self.frame_info_label)

        parent_layout.addWidget(footer)

    # =========================================================================
    # Video Loading
    # =========================================================================

    def _populate_video_dropdown(self):
        """Populate dropdown with videos needing review."""
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        self._dropdown_items = []

        try:
            from mousereach.index import PipelineIndex
            from mousereach.config import PROCESSING_ROOT

            index = PipelineIndex()
            index.load()

            # Get all videos needing any kind of review
            seg_videos = set(index.get_needs_seg_review())
            reach_videos = set(index.get_needs_reach_review())
            outcome_videos = set(index.get_needs_outcome_review())
            all_videos = seg_videos | reach_videos | outcome_videos

            if all_videos:
                self.video_combo.addItem(f"-- {len(all_videos)} videos need review --")
                self.video_combo.model().item(0).setEnabled(False)

                for video_id in sorted(all_videos)[:30]:
                    video_path = self._find_video_path(video_id, PROCESSING_ROOT)
                    if video_path:
                        # Build status string
                        status = []
                        if video_id in seg_videos:
                            status.append("B")  # Boundaries
                        if video_id in reach_videos:
                            status.append("R")  # Reaches
                        if video_id in outcome_videos:
                            status.append("O")  # Outcomes
                        status_str = ",".join(status)

                        self._dropdown_items.append((video_id, video_path))
                        self.video_combo.addItem(f"  {video_id} [{status_str}]")

            # Browse option
            self.video_combo.addItem("--------")
            self.video_combo.model().item(self.video_combo.count() - 1).setEnabled(False)
            self._dropdown_items.append(("BROWSE", None))
            self.video_combo.addItem("Browse for video...")

            self.status_label.setText(f"{len(all_videos)} videos need review")

        except Exception as e:
            self._dropdown_items.append(("BROWSE", None))
            self.video_combo.addItem("Browse for video...")
            self.status_label.setText(f"Index unavailable: {e}")

        self.video_combo.blockSignals(False)

    def _find_video_path(self, video_id: str, processing_root: Path) -> Optional[Path]:
        """Find video file for a video ID.

        Prioritizes Processing/ folder since that's where algorithm output JSON files live.
        """
        # First check Processing/ folder specifically (where JSON outputs are)
        processing_dir = processing_root / "Processing"
        if processing_dir.exists():
            for mp4 in processing_dir.glob(f"{video_id}*.mp4"):
                if "_preview" not in mp4.name:
                    return mp4

        # Fall back to recursive search in other locations
        for mp4 in processing_root.glob(f"**/{video_id}*.mp4"):
            if "_preview" not in mp4.name:
                return mp4
        return None

    def _on_video_selected(self, index: int):
        """Handle video selection from dropdown."""
        if index < 0:
            return

        item_text = self.video_combo.itemText(index).strip()

        if item_text.startswith("--") or not item_text:
            return

        for video_id, video_path in self._dropdown_items:
            if video_id == "BROWSE" and "Browse" in item_text:
                self._browse_for_video()
                return
            elif video_id in item_text and video_path:
                self._load_video(video_path)
                return

    def _browse_for_video(self):
        """Open file dialog to select video."""
        from mousereach.config import Paths

        default_dir = str(Paths.PROCESSING_ROOT) if Paths.PROCESSING_ROOT.exists() else ""

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", default_dir, "Video Files (*.mp4 *.avi)"
        )
        if path:
            self._load_video(Path(path))

    def _load_video(self, video_path: Path, frame_range: Optional[Tuple[int, int]] = None):
        """Load video and initialize data.

        If ``frame_range=(start, end)`` is given, only those frames are read
        from disk and added as the image layer. The image layer is translated
        so its world coordinates still match absolute frame indices, so all
        downstream logic that uses absolute frame indices keeps working.
        DLC points and shapes are filtered to the same window.
        """
        self.video_path = video_path
        self.video_label.setText(f"Loading: {video_path.name}")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        try:
            # Use preview if exists
            video_stem = video_path.stem.replace("_preview", "")
            preview_path = video_path.parent / f"{video_stem}_preview.mp4"

            if "_preview" not in video_path.stem and preview_path.exists():
                actual_video = preview_path
                self.scale_factor = 0.75
            else:
                actual_video = video_path
                self.scale_factor = 1.0

            # Load video frames
            cap = cv2.VideoCapture(str(actual_video))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

            # Resolve frame range: clamp to [0, total_frames-1].
            if frame_range is not None:
                fr_start = max(0, int(frame_range[0]))
                fr_end = min(total_frames - 1, int(frame_range[1]))
                if fr_end < fr_start:
                    fr_end = fr_start
            else:
                fr_start = 0
                fr_end = total_frames - 1
            self.frame_offset = fr_start  # world-coord offset of frame 0 in the slice
            self.frame_window_end = fr_end
            n_to_read = fr_end - fr_start + 1
            self.n_frames = total_frames  # absolute total; absolute frame indices remain valid

            # Seek to start frame, then read sequentially.
            if fr_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr_start)

            frames = []
            for i in range(n_to_read):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if i % 500 == 0:
                    self.progress.setValue(int(70 * i / max(n_to_read, 1)))
                    from qtpy.QtWidgets import QApplication
                    QApplication.processEvents()

            cap.release()
            self.progress.setValue(75)

            # Remove old video layer
            if self.video_layer is not None:
                try:
                    self.viewer.layers.remove(self.video_layer)
                except:
                    pass

            # Add video to viewer with time-axis translate so absolute frame
            # indices map to world coords.
            self.video_layer = self.viewer.add_image(
                np.stack(frames),
                name=video_path.stem,
                rgb=True,
                translate=(self.frame_offset, 0.0, 0.0),
            )
            self.progress.setValue(85)

            # Load GT data
            self.gt = load_or_create_unified_gt(video_path)
            self.progress.setValue(90)

            # Load DLC data for J/K navigation and overlays
            self._load_dlc_data()
            self.progress.setValue(92)

            # Add DLC points overlay
            self._add_dlc_points_layer()
            self.progress.setValue(94)

            # Add pillar circle overlay
            self._add_pillar_shapes_layer()
            self.progress.setValue(96)

            # Refresh UI
            self._refresh_all_lists()
            self._enable_nav_controls(True)

            # Constrain navigation to the loaded window when one is set,
            # so the user cannot scrub into frames that were not loaded.
            if getattr(self, 'frame_offset', 0) > 0 or self.frame_window_end < self.n_frames - 1:
                self.goto_spin.setRange(self.frame_offset, self.frame_window_end)
                # Constrain the napari time-axis dim to the loaded window AND
                # explicitly position the playhead at the start frame.
                try:
                    self.viewer.dims.set_range(0, (self.frame_offset, self.frame_window_end + 1, 1))
                except Exception:
                    pass
                try:
                    self.viewer.dims.set_point(0, self.frame_offset)
                except Exception:
                    try:
                        cs = list(self.viewer.dims.current_step)
                        cs[0] = self.frame_offset
                        self.viewer.dims.current_step = tuple(cs)
                    except Exception:
                        pass
                self.video_label.setText(
                    f"Loaded: {video_path.name}  "
                    f"window=[{self.frame_offset}, {self.frame_window_end}]  "
                    f"({self.frame_window_end - self.frame_offset + 1} frames; "
                    f"video has {self.n_frames})"
                )
            else:
                self.goto_spin.setRange(0, self.n_frames - 1)
                self.video_label.setText(f"Loaded: {video_path.name} ({self.n_frames} frames)")
            # v4.0.0: explicitly refresh per-frame overlays after dims have
            # been positioned, in case the pillar/info panel were computed
            # on a stale (pre-translate) current_step.
            try:
                self._update_pillar_circle()
            except Exception:
                pass
            try:
                self._update_frame_info_panel(
                    self._slider_to_abs(int(self.viewer.dims.current_step[0]))
                )
            except Exception:
                pass
            self.progress.setValue(100)

            status = self.gt.completion_status
            self.status_label.setText(
                f"Progress: {status.boundaries_determined}/{status.boundaries_total} boundaries, "
                f"{status.reaches_determined}/{status.reaches_total} reaches, "
                f"{status.outcomes_determined}/{status.outcomes_total} outcomes"
            )

        except Exception as e:
            show_error(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.progress.setVisible(False)

    def _load_dlc_data(self):
        """Load DLC data for hand-visible frame navigation and points overlay."""
        if not self.video_path:
            return

        video_stem = self.video_path.stem.replace("_preview", "")

        # Find H5 file
        h5_patterns = [
            self.video_path.parent / f"{video_stem}*.h5",
            self.video_path.parent / f"{video_stem.split('DLC')[0]}*.h5",
        ]

        for pattern in h5_patterns:
            for h5_path in self.video_path.parent.glob(pattern.name):
                try:
                    import pandas as pd
                    self.dlc_data = pd.read_hdf(h5_path)
                    # Also create flattened column names for points overlay
                    df = pd.read_hdf(h5_path)
                    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
                    self.dlc_df = df
                    return
                except:
                    pass

        self.dlc_data = None
        self.dlc_df = None

    def _add_dlc_points_layer(self):
        """Add DLC tracking points as overlay."""
        if self.dlc_df is None:
            return

        # Find bodyparts
        bodyparts = []
        for col in self.dlc_df.columns:
            if col.endswith('_x'):
                bodyparts.append(col[:-2])
        bodyparts = sorted(set(bodyparts))

        # Assign colors
        colors_base = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5]
        ]
        bp_colors = {bp: colors_base[i % len(colors_base)] for i, bp in enumerate(bodyparts)}

        # Collect all points -- restricted to the loaded frame window if set.
        points_data = []
        point_colors = []
        point_bps = []  # per-point bodypart name, used for text labels

        win_start = getattr(self, 'frame_offset', 0)
        win_end = getattr(self, 'frame_window_end', len(self.dlc_df) - 1)

        for frame_idx in range(win_start, min(win_end + 1, len(self.dlc_df))):
            for bp in bodyparts:
                x = self.dlc_df.iloc[frame_idx].get(f'{bp}_x', np.nan)
                y = self.dlc_df.iloc[frame_idx].get(f'{bp}_y', np.nan)
                likelihood = self.dlc_df.iloc[frame_idx].get(f'{bp}_likelihood', 0)

                if not np.isnan(x) and not np.isnan(y):
                    # Scale coordinates to match downsampled video
                    scale = getattr(self, 'scale_factor', 1.0)
                    # v4.0.0+: hard truncation below the algo's likelihood
                    # threshold (so "visible dot == algo would use it"),
                    # squared gradient above the threshold so "barely above"
                    # stays visibly faint and "strongly above" looks solid.
                    threshold = self._get_lk_threshold(bp)
                    lk = float(likelihood)
                    if lk < threshold:
                        alpha = 0.05
                    else:
                        norm = (lk - threshold) / max(1e-6, 1.0 - threshold)
                        alpha = 0.10 + 0.90 * (norm ** 2)
                    points_data.append([frame_idx, y * scale, x * scale])
                    point_colors.append(bp_colors[bp] + [alpha])
                    point_bps.append(bp)

        if points_data:
            # Remove old layer
            if self.points_layer is not None:
                try:
                    self.viewer.layers.remove(self.points_layer)
                except:
                    pass

            self.points_layer = self.viewer.add_points(
                np.array(points_data),
                name='DLC Points',
                size=3,
                face_color=np.array(point_colors),
                features={'bp': point_bps},
                text={'string': '{bp}', 'size': 7, 'color': 'white',
                      'translation': [0, -7, 0]},
            )

    def _add_pillar_shapes_layer(self):
        """Add pillar circle overlay based on geometric computation from SA endpoints."""
        if self.dlc_df is None:
            return

        # Remove old layer
        if self.pillar_shapes_layer is not None:
            try:
                self.viewer.layers.remove(self.pillar_shapes_layer)
            except:
                pass

        # Create empty shapes layer. v4.0.0: bumped edge_color to bright red
        # and edge_width to 3 so the pillar is impossible to miss while we
        # debug visibility issues.
        self.pillar_shapes_layer = self.viewer.add_shapes(
            name='Pillar',
            edge_color='red',
            face_color=[0, 0, 0, 0],  # Transparent fill
            edge_width=3,
        )

        # Connect to frame change to update circle (only once)
        # Use a flag to avoid duplicate connections
        if not hasattr(self, '_pillar_callback_connected'):
            self.viewer.dims.events.current_step.connect(self._update_pillar_circle)
            self._pillar_callback_connected = True
        self._update_pillar_circle()

    def _update_pillar_circle(self, event=None):
        """Update pillar circle for current frame based on SA geometry."""
        if self.pillar_shapes_layer is None or self.dlc_df is None:
            return

        # current_step is the napari slider position (0..N-1 within the
        # loaded slice). Convert to an absolute video frame so dlc_df
        # (which holds the entire video's DLC tracking) is indexed
        # correctly.
        slider_val = int(self.viewer.dims.current_step[0])
        frame_idx = self._slider_to_abs(slider_val)
        if frame_idx < 0 or frame_idx >= len(self.dlc_df):
            return

        row = self.dlc_df.iloc[frame_idx]
        scale = getattr(self, 'scale_factor', 1.0)

        # Get SA corners (SABL = SA Bottom Left, SABR = SA Bottom Right)
        # These define the "ruler" - the bottom edge of the staging area.
        # v4.0.0: also gate on SA likelihood, not just NaN, otherwise low-
        # confidence (mistracked) SA points produce a pillar at the wrong
        # location. This is what caused the "pillar slides in" effect at
        # the first few frames of a loaded segment.
        sabl_lk = row.get('SABL_likelihood', 0)
        sabr_lk = row.get('SABR_likelihood', 0)
        if sabl_lk < 0.5 or sabr_lk < 0.5:
            self.pillar_shapes_layer.data = []
            return
        sabl_x = row.get('SABL_x', np.nan) * scale
        sabl_y = row.get('SABL_y', np.nan) * scale
        sabr_x = row.get('SABR_x', np.nan) * scale
        sabr_y = row.get('SABR_y', np.nan) * scale

        # Clear if SA corners not available
        if np.isnan([sabl_x, sabl_y, sabr_x, sabr_y]).any():
            self.pillar_shapes_layer.data = []
            return

        # Compute ruler distance (SA width)
        ruler = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)
        if ruler < 1:
            self.pillar_shapes_layer.data = []
            return

        # Compute midpoint of SA bottom edge
        mid_x = (sabl_x + sabr_x) / 2
        mid_y = (sabl_y + sabr_y) / 2

        # Geometric pillar position: 0.944 ruler units perpendicular from SA midpoint
        pillar_x = mid_x
        pillar_y = mid_y - (0.944 * ruler)

        # Pillar radius (4.125mm diameter pillar / ruler = ~9mm)
        # Note: scale already applied to SA coords, so ruler is already scaled
        ruler_mm = 9.0
        pillar_diameter_mm = 4.125
        pillar_radius_ruler_units = (pillar_diameter_mm / ruler_mm) / 2.0
        pillar_radius = pillar_radius_ruler_units * ruler

        # Create circle as polygon
        angles = np.linspace(0, 2 * np.pi, 32)
        circle_y = pillar_y + pillar_radius * np.sin(angles)
        circle_x = pillar_x + pillar_radius * np.cos(angles)
        circle_data = np.column_stack([circle_y, circle_x])

        self.pillar_shapes_layer.data = [circle_data]

    def _toggle_dlc_visibility(self, visible: bool):
        """Toggle DLC points layer visibility."""
        if self.points_layer is not None and self.points_layer in self.viewer.layers:
            self.points_layer.visible = visible

    def _toggle_pillar_visibility(self, visible: bool):
        """Toggle pillar circle layer visibility."""
        if self.pillar_shapes_layer is not None and self.pillar_shapes_layer in self.viewer.layers:
            self.pillar_shapes_layer.visible = visible

    # =========================================================================
    # List Refresh
    # =========================================================================

    def _refresh_all_lists(self):
        """Refresh all three sections."""
        self._refresh_boundaries_list()
        self._refresh_reaches_list()
        self._refresh_outcomes_list()
        self._refresh_exhaustive_status()

    def _refresh_boundaries_list(self):
        """Refresh the boundaries list."""
        # Clear existing
        while self.boundaries_container.count():
            item = self.boundaries_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.gt:
            return

        show_undetermined_only = self.boundaries_filter.isChecked()
        determined_count = 0
        total = len(self.gt.boundaries)

        for b in self.gt.boundaries:
            if b.determined:
                determined_count += 1
                if show_undetermined_only:
                    continue

            row = self._create_boundary_row(b)
            self.boundaries_container.addWidget(row)

        self.boundaries_progress.setText(f"{determined_count}/{total} determined")
        color = "#4a9" if determined_count == total else "#f80" if determined_count > 0 else "#888"
        self.boundaries_progress.setStyleSheet(f"color: {color};")

    def _create_boundary_row(self, boundary: BoundaryGT) -> QWidget:
        """Create a row widget for a boundary.

        GT Tool: No verify button. Jump to see, Set to change.
        When you Set a frame, that IS the ground truth.
        """
        row = QFrame()
        row.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 3, 5, 3)
        row.setLayout(layout)

        # Index label
        idx_label = QLabel(f"B{boundary.index + 1}:")
        idx_label.setFixedWidth(35)
        layout.addWidget(idx_label)

        # Frame value (green if determined)
        frame_label = QLabel(f"Frame {boundary.frame}")
        if boundary.determined:
            frame_label.setStyleSheet("color: #4a9;")
        layout.addWidget(frame_label)

        layout.addStretch()

        # Jump button
        jump_btn = QPushButton("Jump")
        jump_btn.setMaximumWidth(50)
        jump_btn.clicked.connect(lambda _, b=boundary: self._select_and_jump_boundary(b))
        layout.addWidget(jump_btn)

        # Set Frame button (sets current frame as this boundary)
        set_btn = QPushButton("Set Here")
        set_btn.setMaximumWidth(65)
        set_btn.setToolTip("Set current frame as this boundary (V key)")
        set_btn.clicked.connect(lambda _, b=boundary: self._set_boundary_frame(b))
        layout.addWidget(set_btn)

        # Comment button
        if boundary.comment:
            comment_btn = QPushButton("💬")
            comment_btn.setStyleSheet("background-color: #369; color: white;")
            comment_btn.setToolTip(f"Comment: {boundary.comment}")
        else:
            comment_btn = QPushButton("💬")
            comment_btn.setStyleSheet("")
            comment_btn.setToolTip("Add a comment")
        comment_btn.setMaximumWidth(28)
        comment_btn.clicked.connect(lambda _, b=boundary: self._edit_boundary_comment(b))
        layout.addWidget(comment_btn)

        # Style: green if determined
        if boundary.determined:
            row.setStyleSheet("QFrame { background-color: #1a3010; }")

        return row

    def _refresh_reaches_list(self):
        """Refresh the reaches list, showing all segments including those with no reaches."""
        while self.reaches_container.count():
            item = self.reaches_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.gt:
            return

        show_undetermined_only = self.reaches_filter.isChecked()
        determined_count = 0
        total = len(self.gt.reaches)

        # Group reaches by segment
        reaches_by_segment = {}
        for r in self.gt.reaches:
            if r.segment_num not in reaches_by_segment:
                reaches_by_segment[r.segment_num] = []
            reaches_by_segment[r.segment_num].append(r)

        # Determine expected segment count from boundaries
        n_segments = max(len(self.gt.boundaries) - 1, 0) if self.gt.boundaries else 0

        # Show reaches grouped by segment, including empty segments
        for seg_num in range(1, n_segments + 1):
            segment_reaches = reaches_by_segment.get(seg_num, [])

            if not segment_reaches:
                # Show placeholder for segments with no reaches
                if not show_undetermined_only:
                    empty_label = QLabel(f"Seg {seg_num}: No reaches detected")
                    empty_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
                    self.reaches_container.addWidget(empty_label)
            else:
                for r in segment_reaches:
                    if r.fully_determined:
                        determined_count += 1
                        if show_undetermined_only:
                            continue

                    row = self._create_reach_row(r)
                    self.reaches_container.addWidget(row)

        self.reaches_progress.setText(f"{determined_count}/{total} fully determined")
        color = "#4a9" if determined_count == total else "#f80" if determined_count > 0 else "#888"
        self.reaches_progress.setStyleSheet(f"color: {color};")

    def _create_reach_row(self, reach: ReachGT) -> QWidget:
        """Create a row widget for a reach.

        GT Tool: No verify buttons. Jump to see, Set to change.
        When you Set start/end, that IS the ground truth.
        """
        row = QFrame()
        row.setFrameStyle(QFrame.StyledPanel)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(3)
        row.setLayout(main_layout)

        # Header row: Reach ID + Exclude + Delete
        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>Reach #{reach.reach_id}</b> (Seg {reach.segment_num})"))
        header.addStretch()

        # Exclude toggle button
        if reach.exclude_from_analysis:
            excl_btn = QPushButton("Include")
            excl_btn.setStyleSheet("background-color: #f80; color: white;")
            excl_btn.setToolTip(f"Currently excluded: {reach.exclude_reason or 'No reason'}")
        else:
            excl_btn = QPushButton("Exclude")
            excl_btn.setStyleSheet("")
            excl_btn.setToolTip("Mark this reach as excluded from analysis")
        excl_btn.setMaximumWidth(55)
        excl_btn.clicked.connect(lambda _, r=reach: self._toggle_reach_exclude(r))
        header.addWidget(excl_btn)

        # Comment button
        if reach.comment:
            comment_btn = QPushButton("💬")
            comment_btn.setStyleSheet("background-color: #369; color: white;")
            comment_btn.setToolTip(f"Comment: {reach.comment[:50]}..." if len(reach.comment or '') > 50 else f"Comment: {reach.comment}")
        else:
            comment_btn = QPushButton("💬")
            comment_btn.setStyleSheet("")
            comment_btn.setToolTip("Add a comment")
        comment_btn.setMaximumWidth(28)
        comment_btn.clicked.connect(lambda _, r=reach: self._edit_reach_comment(r))
        header.addWidget(comment_btn)

        # Delete button
        del_btn = QPushButton("X")
        del_btn.setMaximumWidth(25)
        del_btn.setStyleSheet("color: #f55;")
        del_btn.clicked.connect(lambda _, r=reach: self._delete_reach(r))
        header.addWidget(del_btn)
        main_layout.addLayout(header)

        # Start row: frame + Jump + Set
        start_row = QHBoxLayout()
        start_label = QLabel(f"  Start: {reach.start_frame}")
        if reach.start_determined:
            start_label.setStyleSheet("color: #4a9;")
        start_label.setMinimumWidth(150)
        start_row.addWidget(start_label)

        jump_start_btn = QPushButton("Jump")
        jump_start_btn.setMaximumWidth(45)
        jump_start_btn.setToolTip("Jump to start frame")
        jump_start_btn.clicked.connect(lambda _, r=reach: self._select_and_jump_reach(r, "start"))
        start_row.addWidget(jump_start_btn)

        set_start_btn = QPushButton("Set (S)")
        set_start_btn.setMaximumWidth(55)
        set_start_btn.setToolTip("Set current frame as start")
        set_start_btn.clicked.connect(lambda _, r=reach: self._set_reach_start(r))
        start_row.addWidget(set_start_btn)

        start_row.addStretch()
        main_layout.addLayout(start_row)

        # End row: frame + Jump + Set
        end_row = QHBoxLayout()
        end_label = QLabel(f"  End: {reach.end_frame}")
        if reach.end_determined:
            end_label.setStyleSheet("color: #4a9;")
        end_label.setMinimumWidth(150)
        end_row.addWidget(end_label)

        jump_end_btn = QPushButton("Jump")
        jump_end_btn.setMaximumWidth(45)
        jump_end_btn.setToolTip("Jump to end frame")
        jump_end_btn.clicked.connect(lambda _, r=reach: self._select_and_jump_reach(r, "end"))
        end_row.addWidget(jump_end_btn)

        set_end_btn = QPushButton("Set (E)")
        set_end_btn.setMaximumWidth(55)
        set_end_btn.setToolTip("Set current frame as end")
        set_end_btn.clicked.connect(lambda _, r=reach: self._set_reach_end(r))
        end_row.addWidget(set_end_btn)

        end_row.addStretch()
        main_layout.addLayout(end_row)

        # Optional: Exclude indicator
        if reach.exclude_from_analysis:
            exclude_row = QHBoxLayout()
            exclude_label = QLabel(f"  ⚠ EXCLUDED: {reach.exclude_reason or 'No reason'}")
            exclude_label.setStyleSheet("color: #f80;")
            exclude_row.addWidget(exclude_label)
            main_layout.addLayout(exclude_row)

        # Optional: Comment display
        if reach.comment:
            comment_row = QHBoxLayout()
            comment_label = QLabel(f"  💬 {reach.comment}")
            comment_label.setStyleSheet("color: #6af; font-style: italic;")
            comment_label.setWordWrap(True)
            comment_row.addWidget(comment_label)
            main_layout.addLayout(comment_row)

        # Style: green if either start or end has been determined
        if reach.exclude_from_analysis:
            row.setStyleSheet("QFrame { background-color: #2a2010; }")
        elif reach.start_determined or reach.end_determined:
            row.setStyleSheet("QFrame { background-color: #1a3010; }")

        return row

    def _set_boundary_frame(self, boundary: BoundaryGT):
        """Set current frame as this boundary. This IS ground truth - no verification needed."""
        current = int(self.viewer.dims.current_step[0])
        boundary.frame = current
        boundary.determined = True
        boundary.determined_by = get_username()
        boundary.determined_at = get_timestamp()
        self._refresh_boundaries_list()
        self._update_status()
        show_info(f"Boundary {boundary.index + 1} set to frame {current}")

    def _set_reach_start(self, reach: ReachGT):
        """Set current frame as reach start. This IS ground truth."""
        current = int(self.viewer.dims.current_step[0])
        reach.start_frame = current
        reach.start_determined = True
        reach.start_determined_by = get_username()
        reach.start_determined_at = get_timestamp()
        # Re-sort to maintain temporal order after start frame change
        self.gt.reaches.sort(key=lambda r: r.start_frame)
        self._refresh_reaches_list()
        self._update_status()
        show_info(f"Reach #{reach.reach_id} start set to frame {current}")

    def _set_reach_end(self, reach: ReachGT):
        """Set current frame as reach end. This IS ground truth."""
        current = int(self.viewer.dims.current_step[0])
        reach.end_frame = current
        reach.end_determined = True
        reach.end_determined_by = get_username()
        reach.end_determined_at = get_timestamp()
        self._refresh_reaches_list()
        self._update_status()
        show_info(f"Reach #{reach.reach_id} end set to frame {current}")

    def _accept_reach(self, reach: ReachGT):
        """Accept reach as-is (determine both start and end at once)."""
        reach.start_determined = True
        reach.start_determined_by = get_username()
        reach.start_determined_at = get_timestamp()
        reach.end_determined = True
        reach.end_determined_by = get_username()
        reach.end_determined_at = get_timestamp()
        self._refresh_reaches_list()
        self._update_status()
        show_info(f"Reach #{reach.reach_id} accepted (both start and end determined)")

    def _refresh_outcomes_list(self):
        """Refresh the outcomes list."""
        while self.outcomes_container.count():
            item = self.outcomes_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.gt:
            return

        show_undetermined_only = self.outcomes_filter.isChecked()
        determined_count = 0
        total = len(self.gt.outcomes)

        for o in self.gt.outcomes:
            if o.determined:
                determined_count += 1
                if show_undetermined_only:
                    continue

            row = self._create_outcome_row(o)
            self.outcomes_container.addWidget(row)

        self.outcomes_progress.setText(f"{determined_count}/{total} determined")
        color = "#4a9" if determined_count == total else "#f80" if determined_count > 0 else "#888"
        self.outcomes_progress.setStyleSheet(f"color: {color};")

    def _create_outcome_row(self, outcome: OutcomeGT) -> QWidget:
        """Create a row widget for an outcome.

        GT Tool: No verify buttons. Set outcome type, interaction frame, outcome known frame.
        When you set values, that IS the ground truth.
        """
        row = QFrame()
        row.setFrameStyle(QFrame.StyledPanel)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(3)
        row.setLayout(main_layout)

        # Check if this is a placeholder (no algorithm data)
        is_placeholder = not outcome.determined and not outcome.outcome

        # Row 1: Segment info + outcome selector
        header_row = QHBoxLayout()

        # Segment number (highlight placeholder segments)
        if is_placeholder:
            seg_label = QLabel(f"<b>Seg {outcome.segment_num}</b> <i>(no algo data)</i>")
            seg_label.setStyleSheet("color: #888;")
        else:
            seg_label = QLabel(f"<b>Seg {outcome.segment_num}</b>")
        seg_label.setFixedWidth(130 if is_placeholder else 55)
        header_row.addWidget(seg_label)

        # Outcome selector (setting this IS the GT)
        outcome_combo = QComboBox()
        outcome_combo.addItems(["retrieved", "displaced_sa", "displaced_outside", "untouched", "unknown"])
        current_idx = outcome_combo.findText(outcome.outcome) if outcome.outcome else outcome_combo.findText("unknown")
        if current_idx >= 0:
            outcome_combo.setCurrentIndex(current_idx)
        outcome_combo.currentTextChanged.connect(lambda text, o=outcome: self._set_outcome_type(o, text))
        outcome_combo.setMaximumWidth(130)
        header_row.addWidget(outcome_combo)

        # Show determined indicator
        if outcome.determined:
            det_label = QLabel("(determined)")
            det_label.setStyleSheet("color: #4a9;")
            header_row.addWidget(det_label)
        elif is_placeholder:
            # No algorithm data
            no_data_label = QLabel("(set values below)")
            no_data_label.setStyleSheet("color: #f80;")
            header_row.addWidget(no_data_label)
        else:
            undetermined_label = QLabel("(undetermined)")
            undetermined_label.setStyleSheet("color: #888;")
            header_row.addWidget(undetermined_label)


        # Comment button
        if outcome.comment:
            comment_btn = QPushButton("💬")
            comment_btn.setStyleSheet("background-color: #369; color: white;")
            comment_btn.setToolTip(f"Comment: {outcome.comment[:50]}..." if len(outcome.comment or '') > 50 else f"Comment: {outcome.comment}")
        else:
            comment_btn = QPushButton("💬")
            comment_btn.setStyleSheet("")
            comment_btn.setToolTip("Add a comment")
        comment_btn.setMaximumWidth(28)
        comment_btn.clicked.connect(lambda _, o=outcome: self._edit_outcome_comment(o))
        header_row.addWidget(comment_btn)

        header_row.addStretch()
        main_layout.addLayout(header_row)

        # Row 2: Interaction frame + Outcome known frame
        frame_row = QHBoxLayout()

        # Interaction frame
        frame_row.addWidget(QLabel("Interact:"))
        interact_frame = outcome.interaction_frame
        interact_label = QLabel(str(interact_frame) if interact_frame else "--")
        interact_label.setMinimumWidth(50)
        frame_row.addWidget(interact_label)

        jump_interact_btn = QPushButton("→")
        jump_interact_btn.setMaximumWidth(25)
        jump_interact_btn.setToolTip("Jump to interaction frame")
        if interact_frame:
            jump_interact_btn.clicked.connect(lambda _, o=outcome: self._select_and_jump_outcome(o, "interaction"))
        else:
            jump_interact_btn.setEnabled(False)
        frame_row.addWidget(jump_interact_btn)

        set_interact_btn = QPushButton("Set (I)")
        set_interact_btn.setMaximumWidth(50)
        set_interact_btn.setToolTip("Set current frame as interaction frame")
        set_interact_btn.clicked.connect(lambda _, o=outcome: self._set_interaction_frame(o))
        frame_row.addWidget(set_interact_btn)

        frame_row.addWidget(QLabel(" | Known:"))

        # Outcome known frame
        known_frame = outcome.outcome_known_frame
        known_label = QLabel(str(known_frame) if known_frame else "--")
        known_label.setMinimumWidth(50)
        frame_row.addWidget(known_label)

        jump_known_btn = QPushButton("→")
        jump_known_btn.setMaximumWidth(25)
        jump_known_btn.setToolTip("Jump to outcome known frame")
        if known_frame:
            jump_known_btn.clicked.connect(lambda _, o=outcome: self._select_and_jump_outcome(o, "known"))
        else:
            jump_known_btn.setEnabled(False)
        frame_row.addWidget(jump_known_btn)

        set_known_btn = QPushButton("Set (K)")
        set_known_btn.setMaximumWidth(50)
        set_known_btn.setToolTip("Set current frame as outcome known frame")
        set_known_btn.clicked.connect(lambda _, o=outcome: self._set_outcome_known_frame(o))
        frame_row.addWidget(set_known_btn)

        frame_row.addStretch()
        main_layout.addLayout(frame_row)

        # Optional: Comment display
        if outcome.comment:
            comment_row = QHBoxLayout()
            comment_label = QLabel(f"💬 {outcome.comment}")
            comment_label.setStyleSheet("color: #6af; font-style: italic;")
            comment_label.setWordWrap(True)
            comment_row.addWidget(comment_label)
            main_layout.addLayout(comment_row)

        # Style: green if determined
        if outcome.determined:
            row.setStyleSheet("QFrame { background-color: #1a3010; }")

        return row

    def _set_outcome_type(self, outcome: OutcomeGT, new_type: str):
        """Change the outcome type. Setting this IS the ground truth."""
        if outcome.outcome != new_type:
            outcome.outcome = new_type
            outcome.determined = True
            outcome.determined_by = get_username()
            outcome.determined_at = get_timestamp()
            self._refresh_outcomes_list()
            self._update_status()

    def _set_interaction_frame(self, outcome: OutcomeGT):
        """Set current frame as interaction frame. This IS the ground truth."""
        current = int(self.viewer.dims.current_step[0])
        outcome.interaction_frame = current
        outcome.determined = True
        outcome.determined_by = get_username()
        outcome.determined_at = get_timestamp()
        self._refresh_outcomes_list()
        self._update_status()
        show_info(f"Seg {outcome.segment_num} interaction frame set to {current}")

    def _set_outcome_known_frame(self, outcome: OutcomeGT):
        """Set current frame as outcome known frame. This IS the ground truth."""
        current = int(self.viewer.dims.current_step[0])
        outcome.outcome_known_frame = current
        outcome.determined = True
        outcome.determined_by = get_username()
        outcome.determined_at = get_timestamp()
        self._refresh_outcomes_list()
        self._update_status()
        show_info(f"Seg {outcome.segment_num} outcome known frame set to {current}")

    # =========================================================================
    # Verification Actions
    # =========================================================================

    def _determine_boundary(self, boundary: BoundaryGT):
        """Mark a boundary as determined."""
        boundary.determined = True
        boundary.determined_by = get_username()
        boundary.determined_at = get_timestamp()
        self._refresh_boundaries_list()
        self._update_status()
        show_info(f"Boundary {boundary.index + 1} determined")

    def _determine_reach_start(self, reach: ReachGT):
        """Mark reach start as determined."""
        reach.start_determined = True
        reach.start_determined_by = get_username()
        reach.start_determined_at = get_timestamp()
        self._refresh_reaches_list()
        self._update_status()
        show_info(f"Reach #{reach.reach_id} start determined")

    def _determine_reach_end(self, reach: ReachGT):
        """Mark reach end as determined."""
        if not reach.start_determined:
            show_warning("Must determine start first")
            return

        reach.end_determined = True
        reach.end_determined_by = get_username()
        reach.end_determined_at = get_timestamp()
        self._refresh_reaches_list()
        self._update_status()
        show_info(f"Reach #{reach.reach_id} fully determined")

    def _determine_outcome(self, outcome: OutcomeGT):
        """Mark an outcome as determined."""
        outcome.determined = True
        outcome.determined_by = get_username()
        outcome.determined_at = get_timestamp()
        self._refresh_outcomes_list()
        self._update_status()
        show_info(f"Segment {outcome.segment_num} outcome determined")

    def _add_reach(self):
        """Add a new reach at current frame."""
        if not self.gt:
            show_warning("No video loaded")
            return

        current_frame = int(self.viewer.dims.current_step[0])

        # Find which segment we're in
        segment_num = 1
        for b in self.gt.boundaries:
            if current_frame >= b.frame:
                segment_num = b.index + 1
            else:
                break

        # Generate new reach ID
        new_id = max((r.reach_id for r in self.gt.reaches), default=0) + 1

        username = get_username()
        timestamp = get_timestamp()
        new_reach = ReachGT(
            reach_id=new_id,
            segment_num=segment_num,
            start_frame=current_frame,
            end_frame=current_frame + 30,  # Default 30 frames duration
            start_determined=True,
            start_determined_by=username,
            start_determined_at=timestamp,
            end_determined=True,
            end_determined_by=username,
            end_determined_at=timestamp,
        )

        self.gt.reaches.append(new_reach)

        # Sort reaches by start frame so new reach appears in correct position
        self.gt.reaches.sort(key=lambda r: r.start_frame)

        # Select the new reach for immediate editing
        self._selected_reach = new_reach

        self._refresh_reaches_list()
        show_info(f"Added reach #{new_id} at frame {current_frame} - use S/E to set start/end")

    def _delete_reach(self, reach: ReachGT):
        """Delete a reach."""
        if not self.gt:
            return

        reply = QMessageBox.question(
            self, "Delete Reach",
            f"Delete reach #{reach.reach_id}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.gt.reaches = [r for r in self.gt.reaches if r.reach_id != reach.reach_id]
            self._refresh_reaches_list()
            show_info(f"Deleted reach #{reach.reach_id}")

    def _toggle_reach_exclude(self, reach: ReachGT):
        """Toggle exclude status for a reach."""
        if not self.gt:
            return

        if reach.exclude_from_analysis:
            # Currently excluded, include it
            reach.exclude_from_analysis = False
            reach.exclude_reason = None
            show_info(f"Reach #{reach.reach_id} included in analysis")
        else:
            # Prompt for exclude reason
            from qtpy.QtWidgets import QInputDialog
            reason, ok = QInputDialog.getText(
                self, "Exclude Reach",
                f"Reason for excluding reach #{reach.reach_id}:",
                text="Not a valid reach"
            )
            if ok:
                reach.exclude_from_analysis = True
                reach.exclude_reason = reason or "No reason given"
                show_info(f"Reach #{reach.reach_id} excluded: {reach.exclude_reason}")

        self._refresh_reaches_list()

    def _toggle_outcome_flag(self, outcome: OutcomeGT):
        """Flag toggling removed in v2 - flagging belongs in algorithm output files."""
        pass

    def _toggle_exhaustive(self, component: str):
        """Toggle exhaustive status for a component (segmentation, reaches, outcomes)."""
        if not self.gt:
            return

        # Get current state
        if component == "segmentation":
            currently_exhaustive = self.gt.segmentation_exhaustive
        elif component == "reaches":
            currently_exhaustive = self.gt.reaches_exhaustive
        elif component == "outcomes":
            currently_exhaustive = self.gt.outcomes_exhaustive
        else:
            return

        if currently_exhaustive:
            # Removing exhaustive - simple confirmation
            reply = QMessageBox.question(
                self, "Remove Exhaustive",
                f"Remove exhaustive declaration for {component}?\n\n"
                f"This means the {component} list may be incomplete.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            if component == "segmentation":
                self.gt.segmentation_exhaustive = False
                self.gt.segmentation_exhaustive_determined_by = None
                self.gt.segmentation_exhaustive_determined_at = None
            elif component == "reaches":
                self.gt.reaches_exhaustive = False
                self.gt.reaches_exhaustive_determined_by = None
                self.gt.reaches_exhaustive_determined_at = None
            elif component == "outcomes":
                self.gt.outcomes_exhaustive = False
                self.gt.outcomes_exhaustive_determined_by = None
                self.gt.outcomes_exhaustive_determined_at = None
            show_info(f"{component.capitalize()} no longer marked exhaustive")
        else:
            # Setting exhaustive - strong confirmation
            reply = QMessageBox.warning(
                self, "Mark Exhaustive",
                f"Mark {component} as EXHAUSTIVE?\n\n"
                f"This declares: 'I have determined ALL {component} in this video. "
                f"There are no others besides what is listed here.'\n\n"
                f"This enables full precision + recall evaluation against algorithms.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            username = get_username()
            timestamp = get_timestamp()
            if component == "segmentation":
                self.gt.segmentation_exhaustive = True
                self.gt.segmentation_exhaustive_determined_by = username
                self.gt.segmentation_exhaustive_determined_at = timestamp
            elif component == "reaches":
                self.gt.reaches_exhaustive = True
                self.gt.reaches_exhaustive_determined_by = username
                self.gt.reaches_exhaustive_determined_at = timestamp
            elif component == "outcomes":
                self.gt.outcomes_exhaustive = True
                self.gt.outcomes_exhaustive_determined_by = username
                self.gt.outcomes_exhaustive_determined_at = timestamp
            show_info(f"{component.capitalize()} marked exhaustive by {username}")

        self._refresh_exhaustive_status()

    def _refresh_exhaustive_status(self):
        """Update exhaustive status labels and buttons for all components."""
        if not self.gt:
            return

        for component, label, btn in [
            ("segmentation", self.boundaries_exhaustive_label, self.boundaries_exhaustive_btn),
            ("reaches", self.reaches_exhaustive_label, self.reaches_exhaustive_btn),
            ("outcomes", self.outcomes_exhaustive_label, self.outcomes_exhaustive_btn),
        ]:
            if component == "segmentation":
                is_exhaustive = self.gt.segmentation_exhaustive
                determined_by = self.gt.segmentation_exhaustive_determined_by
            elif component == "reaches":
                is_exhaustive = self.gt.reaches_exhaustive
                determined_by = self.gt.reaches_exhaustive_determined_by
            else:
                is_exhaustive = self.gt.outcomes_exhaustive
                determined_by = self.gt.outcomes_exhaustive_determined_by

            if is_exhaustive:
                label.setText(f"EXHAUSTIVE (by {determined_by or 'unknown'})")
                label.setStyleSheet("color: #4a9; font-weight: bold;")
                btn.setText("Remove Exhaustive")
                btn.setStyleSheet("background-color: #4a9; color: white;")
            else:
                label.setText("Not marked exhaustive")
                label.setStyleSheet("color: #888; font-style: italic;")
                btn.setText("Mark Exhaustive")
                btn.setStyleSheet("")

    def _edit_reach_comment(self, reach: ReachGT):
        """Edit comment for a reach."""
        if not self.gt:
            return

        from qtpy.QtWidgets import QInputDialog
        current = reach.comment or ""
        comment, ok = QInputDialog.getMultiLineText(
            self, "Reach Comment",
            f"Comment for reach #{reach.reach_id}:\n(Describe what's happening, algorithm challenges, etc.)",
            text=current
        )
        if ok:
            reach.comment = comment.strip() if comment.strip() else None
            self._refresh_reaches_list()
            if reach.comment:
                show_info(f"Comment saved for reach #{reach.reach_id}")
            else:
                show_info(f"Comment removed from reach #{reach.reach_id}")

    def _edit_outcome_comment(self, outcome: OutcomeGT):
        """Edit comment for an outcome."""
        if not self.gt:
            return

        from qtpy.QtWidgets import QInputDialog
        current = outcome.comment or ""
        comment, ok = QInputDialog.getMultiLineText(
            self, "Outcome Comment",
            f"Comment for segment {outcome.segment_num} outcome:\n(Describe what's happening, algorithm challenges, etc.)",
            text=current
        )
        if ok:
            outcome.comment = comment.strip() if comment.strip() else None
            self._refresh_outcomes_list()
            if outcome.comment:
                show_info(f"Comment saved for segment {outcome.segment_num} outcome")
            else:
                show_info(f"Comment removed from segment {outcome.segment_num} outcome")

    def _edit_boundary_comment(self, boundary: BoundaryGT):
        """Edit comment for a boundary."""
        if not self.gt:
            return

        from qtpy.QtWidgets import QInputDialog
        current = boundary.comment or ""
        comment, ok = QInputDialog.getMultiLineText(
            self, "Boundary Comment",
            f"Comment for boundary {boundary.index}:\n(Describe what's happening, algorithm challenges, etc.)",
            text=current
        )
        if ok:
            boundary.comment = comment.strip() if comment.strip() else None
            self._refresh_boundaries_list()
            if boundary.comment:
                show_info(f"Comment saved for boundary {boundary.index}")
            else:
                show_info(f"Comment removed from boundary {boundary.index}")

    def _update_status(self):
        """Update the status label with current progress."""
        if not self.gt:
            return

        self.gt.update_completion_status()
        status = self.gt.completion_status

        self.status_label.setText(
            f"Progress: {status.boundaries_determined}/{status.boundaries_total} boundaries, "
            f"{status.reaches_determined}/{status.reaches_total} reaches, "
            f"{status.outcomes_determined}/{status.outcomes_total} outcomes"
        )

    # =========================================================================
    # Save Functions
    # =========================================================================

    def _save_progress(self):
        """Save current progress to unified GT file."""
        if not self.gt or not self.video_path:
            return

        path = save_unified_gt(self.gt, self.video_path)
        show_info(f"Progress saved to {path.name}")
        self.data_saved.emit(path)

    def _save_ground_truth(self):
        """Save as finalized ground truth.

        GT files only contain items the human explicitly interacted with.
        Algorithm-seeded items the human never touched are NOT saved.
        This makes GT files true training data - only human-provided examples.
        """
        if not self.gt or not self.video_path:
            return

        from mousereach.review.unified_gt import UnifiedGroundTruth

        # Create filtered copy with only human-interacted items
        filtered_gt = UnifiedGroundTruth(
            video_name=self.gt.video_name,
            schema_version=self.gt.schema_version,
            created_by=self.gt.created_by,
            created_at=self.gt.created_at,
            anomalies=self.gt.anomalies,
            anomaly_annotations=self.gt.anomaly_annotations,
        )

        # Filter boundaries: keep if human determined the value
        filtered_gt.boundaries = [
            b for b in self.gt.boundaries
            if b.determined
        ]

        # Filter reaches: keep if human determined start OR end
        filtered_gt.reaches = [
            r for r in self.gt.reaches
            if r.start_determined or r.end_determined
        ]

        # Filter outcomes: keep if human determined the value
        filtered_gt.outcomes = [
            o for o in self.gt.outcomes
            if o.determined
        ]

        # Save the filtered GT
        path = save_unified_gt(filtered_gt, self.video_path)

        # Report what was saved
        n_boundaries = len(filtered_gt.boundaries)
        n_reaches = len(filtered_gt.reaches)
        n_outcomes = len(filtered_gt.outcomes)
        show_info(f"Ground truth saved: {n_boundaries} boundaries, {n_reaches} reaches, {n_outcomes} outcomes")
        self.data_saved.emit(path)

    def _save_to_algo_files(self):
        """Save corrections back to algorithm output files (Review Tool mode)."""
        if not self.gt or not self.video_path:
            return

        from datetime import datetime
        import os

        username = get_username()
        timestamp = get_timestamp()

        # Get video stem for finding algo files
        video_stem = self.video_path.stem
        if 'DLC_' in video_stem:
            video_stem = video_stem.split('DLC_')[0]
        parent = self.video_path.parent

        saved_files = []

        # === Save segments ===
        segments_path = parent / f"{video_stem}_segments.json"
        if segments_path.exists():
            try:
                with open(segments_path, 'r') as f:
                    seg_data = json.load(f)

                # Update boundaries from GT data
                original_boundaries = seg_data.get("boundaries", [])
                new_boundaries = [b.frame for b in self.gt.boundaries]

                # Track per-boundary corrections
                boundary_corrections = {}
                changes = []
                for i, b in enumerate(self.gt.boundaries):
                    orig_frame = original_boundaries[i] if i < len(original_boundaries) else None
                    was_corrected = orig_frame is not None and b.frame != orig_frame
                    if was_corrected:
                        boundary_corrections[str(i)] = {
                            "human_corrected": True,
                            "original_frame": orig_frame,
                            "corrected_by": username,
                            "corrected_at": timestamp
                        }
                        changes.append({
                            "index": i,
                            "original": orig_frame,
                            "corrected": b.frame,
                            "delta": b.frame - orig_frame
                        })
                    else:
                        boundary_corrections[str(i)] = {
                            "human_corrected": False,
                            "original_frame": orig_frame,
                            "corrected_by": None,
                            "corrected_at": None
                        }

                seg_data.update({
                    "boundaries": new_boundaries,
                    "n_boundaries": len(new_boundaries),
                    "boundary_corrections": boundary_corrections,
                    "validation_status": "validated",
                    "validation_record": {
                        "validated_by": username,
                        "validated_at": timestamp,
                        "changes_made": changes,
                        "total_items": len(new_boundaries),
                        "items_changed": len(changes)
                    }
                })

                with open(segments_path, 'w') as f:
                    json.dump(seg_data, f, indent=2)
                saved_files.append("segments")
            except Exception as e:
                show_warning(f"Failed to save segments: {e}")

        # === Save reaches ===
        reaches_path = parent / f"{video_stem}_reaches.json"
        if reaches_path.exists():
            try:
                with open(reaches_path, 'r') as f:
                    reach_data = json.load(f)

                # Build lookup of original algo reaches for correction detection
                algo_reaches_by_id = {}
                for seg in reach_data.get("segments", []):
                    for ar in seg.get("reaches", []):
                        algo_reaches_by_id[ar.get("reach_id")] = ar

                updated_reaches = []
                for r in self.gt.reaches:
                    algo_r = algo_reaches_by_id.get(r.reach_id, {})
                    start_corrected = algo_r.get("start_frame") is not None and r.start_frame != algo_r.get("start_frame")
                    end_corrected = algo_r.get("end_frame") is not None and r.end_frame != algo_r.get("end_frame")
                    reach_dict = {
                        "reach_id": r.reach_id,
                        "segment_num": r.segment_num,
                        "start_frame": r.start_frame,
                        "end_frame": r.end_frame,
                        "human_corrected": start_corrected or end_corrected,
                    }
                    if start_corrected:
                        reach_dict["original_start_frame"] = algo_r.get("start_frame")
                    if end_corrected:
                        reach_dict["original_end_frame"] = algo_r.get("end_frame")
                    if r.exclude_from_analysis:
                        reach_dict["exclude_from_analysis"] = True
                        reach_dict["exclude_reason"] = r.exclude_reason
                    updated_reaches.append(reach_dict)

                reach_data["reaches"] = updated_reaches
                reach_data["validation_status"] = "validated"
                reach_data["validated_by"] = username
                reach_data["validated_at"] = timestamp

                with open(reaches_path, 'w') as f:
                    json.dump(reach_data, f, indent=2)
                saved_files.append("reaches")
            except Exception as e:
                show_warning(f"Failed to save reaches: {e}")

        # === Save outcomes ===
        outcomes_path = parent / f"{video_stem}_pellet_outcomes.json"
        if outcomes_path.exists():
            try:
                with open(outcomes_path, 'r') as f:
                    outcome_data = json.load(f)

                # Update outcomes from GT data
                for o in self.gt.outcomes:
                    if not o.determined:
                        continue  # Skip undetermined outcomes

                    # Find matching segment in algo data
                    for seg in outcome_data.get("segments", []):
                        if seg.get("segment_num") == o.segment_num:
                            algo_outcome = seg.get("outcome", "")
                            was_corrected = algo_outcome and o.outcome != algo_outcome
                            if was_corrected:
                                seg["human_corrected"] = True
                                seg["original_outcome"] = algo_outcome
                            seg["outcome"] = o.outcome
                            if o.interaction_frame is not None:
                                seg["interaction_frame"] = o.interaction_frame
                            if o.outcome_known_frame is not None:
                                seg["outcome_known_frame"] = o.outcome_known_frame
                            break

                outcome_data["validation_status"] = "validated"
                outcome_data["validated_by"] = username
                outcome_data["validated_at"] = timestamp

                with open(outcomes_path, 'w') as f:
                    json.dump(outcome_data, f, indent=2)
                saved_files.append("outcomes")
            except Exception as e:
                show_warning(f"Failed to save outcomes: {e}")

        if saved_files:
            show_info(f"Saved corrections to: {', '.join(saved_files)}")
            self.data_saved.emit(self.video_path)

            # Trigger database update check (async/background)
            self._maybe_update_database()
        else:
            show_warning("No algorithm files found to update")

    def _maybe_update_database(self):
        """
        Check if unified database needs updating after save.

        This is a lightweight check - only triggers rebuild if files changed.
        Runs in background to not block the UI.
        """
        try:
            from mousereach.analysis.data import trigger_database_update
            import threading

            def update_bg():
                try:
                    result = trigger_database_update(force=False)
                    if result:
                        print(f"Database updated: {result}")
                except Exception as e:
                    print(f"Database update skipped: {e}")

            # Run in background thread
            thread = threading.Thread(target=update_bg, daemon=True)
            thread.start()
        except ImportError:
            pass  # Analysis module not available

    # =========================================================================
    # Selection + Navigation
    # =========================================================================

    def _select_and_jump_boundary(self, boundary: BoundaryGT):
        """Select boundary and jump to its frame."""
        self._selected_boundary = boundary
        self._selected_reach = None
        self._selected_outcome = None
        self._jump_to_frame(boundary.frame)

    def _select_and_jump_reach(self, reach: ReachGT, which: str = "start"):
        """Select reach and jump to start or end frame."""
        self._selected_reach = reach
        self._selected_boundary = None
        self._selected_outcome = None
        frame = reach.start_frame if which == "start" else reach.end_frame
        self._jump_to_frame(frame)

    def _select_and_jump_outcome(self, outcome: OutcomeGT, which: str = "interaction"):
        """Select outcome and jump to interaction or outcome_known frame."""
        self._selected_outcome = outcome
        self._selected_boundary = None
        self._selected_reach = None
        if which == "interaction" and outcome.interaction_frame:
            self._jump_to_frame(outcome.interaction_frame)
        elif which == "known" and outcome.outcome_known_frame:
            self._jump_to_frame(outcome.outcome_known_frame)

    # =========================================================================
    # Navigation
    # =========================================================================

    # Per-bodypart likelihood thresholds used by the algorithm. The DLC
    # point overlay uses these to mirror what each algo actually sees:
    # below threshold -> nearly invisible (algo would discard); above
    # threshold -> opacity scales with confidence so the user can tell
    # "barely above threshold" from "rock-solid."
    _LK_THRESHOLD_PER_BP = {
        # SA reference points -- outcome detector requires high lk for
        # geometry (ruler/pillar). Stricter than other points.
        'SABL': 0.8, 'SABR': 0.8, 'SATL': 0.8, 'SATR': 0.8,
        # Box edges (slit) -- treat as SA-like reference geometry.
        'BOXL': 0.8, 'BOXR': 0.8,
        # Pellet/Pillar/paw/face all use 0.5 by default (matches outcome
        # and reach detector defaults).
    }

    def _get_lk_threshold(self, bp: str) -> float:
        """v4.0.0+: per-bodypart likelihood threshold used by the overlay
        to decide whether the algo would consider a point real."""
        return self._LK_THRESHOLD_PER_BP.get(bp, 0.5)

    def _save_screenshot(self):
        """v4.0.0: capture the napari canvas at the current frame and save
        it. Default save path = self.screenshot_dir (set by the launcher
        from --screenshot-dir or auto-derived). Default filename includes
        the absolute frame number, the video stem, and the segment number
        if known. The user can adjust the path via the Save As dialog."""
        try:
            from qtpy.QtWidgets import QFileDialog
            slider_val = int(self.viewer.dims.current_step[0])
            abs_frame = self._slider_to_abs(slider_val)
            video_stem = (self.video_path.stem.replace('_preview', '')
                          if self.video_path is not None else 'video')
            seg_tag = f"_seg{self._screenshot_segment}" if getattr(self, '_screenshot_segment', None) else ''
            default_name = f"f{abs_frame}_{video_stem}{seg_tag}.png"
            default_dir = getattr(self, 'screenshot_dir', None)
            from pathlib import Path as _P
            if default_dir is None:
                default_dir = _P.cwd()
            else:
                default_dir = _P(default_dir)
            default_dir.mkdir(parents=True, exist_ok=True)
            default_path = str(default_dir / default_name)
            path, _ = QFileDialog.getSaveFileName(
                self, "Save screenshot", default_path,
                "PNG image (*.png);;All files (*.*)"
            )
            if not path:
                return
            arr = self.viewer.screenshot(canvas_only=True, flash=False)
            try:
                import imageio.v3 as iio
                iio.imwrite(path, arr)
            except Exception:
                # Fallback to imageio v2 API
                import imageio
                imageio.imwrite(path, arr)
            print(f"[mousereach-gt] saved screenshot: {path}")
        except Exception as e:
            print(f"[mousereach-gt] screenshot failed: {e}")

    def _abs_to_slider(self, abs_frame: int) -> int:
        """Convert an absolute video frame to napari's slider position.
        When a slice is loaded, the slider shows array indices 0..N-1
        within the slice; absolute frames must be offset down by
        frame_offset to land on the correct slice index."""
        return int(abs_frame) - int(getattr(self, 'frame_offset', 0))

    def _slider_to_abs(self, slider_val: int) -> int:
        """Convert napari's slider position to an absolute video frame."""
        return int(slider_val) + int(getattr(self, 'frame_offset', 0))

    def _slider_max(self) -> int:
        """Largest valid slider position for the currently loaded slice."""
        offset = int(getattr(self, 'frame_offset', 0))
        end = int(getattr(self, 'frame_window_end', self.n_frames - 1))
        return max(0, end - offset)

    def _jump_frames(self, delta: int):
        """Jump by delta frames (delta is in absolute frames, same as slider since they shift together)."""
        if self.n_frames == 0:
            return
        current = self.viewer.dims.current_step[0]
        new_slider = max(0, min(self._slider_max(), current + delta))
        self.viewer.dims.set_current_step(0, new_slider)

    def _jump_to_frame(self, frame: int):
        """Jump to a specific ABSOLUTE video frame."""
        if self.n_frames == 0:
            return
        slider_val = self._abs_to_slider(frame)
        slider_val = max(0, min(self._slider_max(), slider_val))
        self.viewer.dims.set_current_step(0, slider_val)

    def _goto_frame(self):
        """Go to frame from spinbox."""
        self._jump_to_frame(self.goto_spin.value())

    def _jump_to_prev_segment(self):
        """Jump to the previous segment boundary."""
        if not self.gt or not self.gt.boundaries:
            return
        current = self.viewer.dims.current_step[0]
        # Find the boundary just before current frame
        for b in reversed(self.gt.boundaries):
            if b.frame < current - 5:  # Small buffer to avoid getting stuck
                self._jump_to_frame(b.frame)
                return
        # If none found, go to first boundary
        self._jump_to_frame(self.gt.boundaries[0].frame)

    def _jump_to_next_segment(self):
        """Jump to the next segment boundary."""
        if not self.gt or not self.gt.boundaries:
            return
        current = self.viewer.dims.current_step[0]
        # Find the first boundary after current frame
        for b in self.gt.boundaries:
            if b.frame > current + 5:  # Small buffer to avoid getting stuck
                self._jump_to_frame(b.frame)
                return
        # If none found, go to last boundary
        self._jump_to_frame(self.gt.boundaries[-1].frame)

    def _on_frame_change(self, event=None):
        """Update display when frame changes."""
        if self.n_frames == 0:
            return

        slider_val = self.viewer.dims.current_step[0]
        abs_frame = self._slider_to_abs(slider_val)
        time_sec = abs_frame / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60

        # Show ABSOLUTE frame in the label so the user is never confused
        # about whether a frame number is slice-relative or video-absolute.
        self.frame_label.setText(f"Frame: {abs_frame} / {self.n_frames}")
        self.time_label.setText(f"Time: {mins}:{secs:05.2f}")

        self._update_frame_info_panel(abs_frame)

    def _update_frame_info_panel(self, abs_frame: int):
        """v4.0.0: live per-frame info panel. Content is mode-dependent.

        Takes an ABSOLUTE video frame index (callers should convert from
        slider via self._slider_to_abs if needed)."""
        if not hasattr(self, 'frame_info_label') or self.frame_info_label is None:
            return
        if self.dlc_df is None or abs_frame < 0 or abs_frame >= len(self.dlc_df):
            self.frame_info_label.setText("")
            return
        row = self.dlc_df.iloc[abs_frame]
        mode = getattr(self, 'info_panel_mode', 'general')
        lines = [f"abs frame: {abs_frame}"]

        def fmt_pt(name):
            x = row.get(f'{name}_x', float('nan'))
            y = row.get(f'{name}_y', float('nan'))
            lk = row.get(f'{name}_likelihood', 0.0)
            try:
                if x != x:  # NaN check without numpy
                    return f"{name}: lk={lk:.2f} (no pos)"
                return f"{name}: ({x:.0f}, {y:.0f})  lk={lk:.2f}"
            except Exception:
                return f"{name}: (err)"

        if mode == 'outcome':
            lines.append(fmt_pt('Pellet'))
            lines.append(fmt_pt('Pillar'))
            # slit_y from BOXL/BOXR
            try:
                slit_y = float((row.get('BOXL_y', 0) + row.get('BOXR_y', 0)) / 2)
                lines.append(f"slit_y: {slit_y:.0f}")
            except Exception:
                pass
            # SA confidences + computed pillar position so the user can
            # tell if the pillar OVERLAY actually matches the computed
            # geometric pillar, frame-by-frame.
            try:
                sabl_lk = float(row.get('SABL_likelihood', 0))
                sabr_lk = float(row.get('SABR_likelihood', 0))
                lines.append(f"SA_lk: BL={sabl_lk:.2f} BR={sabr_lk:.2f}")
                if sabl_lk >= 0.5 and sabr_lk >= 0.5:
                    sabl_x = float(row.get('SABL_x', float('nan')))
                    sabl_y = float(row.get('SABL_y', float('nan')))
                    sabr_x = float(row.get('SABR_x', float('nan')))
                    sabr_y = float(row.get('SABR_y', float('nan')))
                    if all(v == v for v in (sabl_x, sabl_y, sabr_x, sabr_y)):
                        ruler = ((sabr_x - sabl_x) ** 2 + (sabr_y - sabl_y) ** 2) ** 0.5
                        mid_x = (sabl_x + sabr_x) / 2
                        mid_y = (sabl_y + sabr_y) / 2
                        pillar_x_geom = mid_x
                        pillar_y_geom = mid_y - 0.944 * ruler
                        lines.append(
                            f"pillar(geom): ({pillar_x_geom:.0f}, {pillar_y_geom:.0f}) ruler={ruler:.1f}"
                        )
            except Exception:
                pass
            # Diagnostic: how many shapes does the pillar layer currently
            # contain? If geom shows a position but n_shapes == 0, the
            # update isn't reaching the layer. If n_shapes >= 1 but the
            # circle still isn't visible, it's a render-side issue.
            try:
                n_shapes = (len(self.pillar_shapes_layer.data)
                            if self.pillar_shapes_layer is not None else -1)
                visible = (self.pillar_shapes_layer.visible
                           if self.pillar_shapes_layer is not None else False)
                lines.append(f"pillar_layer: n={n_shapes}, visible={visible}")
            except Exception:
                pass
        elif mode == 'reach':
            for p in ('RightHand', 'RHLeft', 'RHOut', 'RHRight'):
                lines.append(fmt_pt(p))
            lines.append(fmt_pt('Pellet'))
        elif mode == 'segmentation':
            for p in ('SABL', 'SABR', 'SATL', 'SATR'):
                lines.append(fmt_pt(p))
        else:
            lines.append(fmt_pt('Pellet'))
            lines.append(fmt_pt('Pillar'))
            for p in ('RightHand', 'RHLeft'):
                lines.append(fmt_pt(p))
        self.frame_info_label.setText("    ".join(lines))

    def _jump_to_next_hand(self):
        """Jump to next frame where hand is visible (DLC likelihood > 0.5)."""
        if self.dlc_data is None:
            show_warning("No DLC data loaded")
            return

        current = self.viewer.dims.current_step[0]

        # Look for RH columns
        rh_cols = [c for c in self.dlc_data.columns if 'RH' in str(c) and 'likelihood' in str(c)]

        if not rh_cols:
            return

        # Find next frame with high likelihood
        for frame in range(current + 1, self.n_frames):
            max_like = max(self.dlc_data.iloc[frame][col] for col in rh_cols)
            if max_like > 0.5:
                self._jump_to_frame(frame)
                return

    def _jump_to_prev_hand(self):
        """Jump to previous frame where hand is visible."""
        if self.dlc_data is None:
            show_warning("No DLC data loaded")
            return

        current = self.viewer.dims.current_step[0]

        rh_cols = [c for c in self.dlc_data.columns if 'RH' in str(c) and 'likelihood' in str(c)]

        if not rh_cols:
            return

        for frame in range(current - 1, -1, -1):
            max_like = max(self.dlc_data.iloc[frame][col] for col in rh_cols)
            if max_like > 0.5:
                self._jump_to_frame(frame)
                return

    def _get_current_segment(self):
        """Get the segment containing the current frame."""
        if not self.gt or not self.gt.boundaries:
            return None, None
        current = self.viewer.dims.current_step[0]
        # Find segment bounds (boundaries are sorted)
        start_frame = 0
        end_frame = self.n_frames - 1
        for i, b in enumerate(self.gt.boundaries):
            if b.frame <= current:
                start_frame = b.frame
            if b.frame > current:
                end_frame = b.frame
                break
        return start_frame, end_frame

    def _jump_to_segment_start(self):
        """Jump to start of current segment."""
        start, _ = self._get_current_segment()
        if start is not None:
            self._jump_to_frame(start)

    def _jump_to_segment_end(self):
        """Jump to end of current segment."""
        _, end = self._get_current_segment()
        if end is not None:
            self._jump_to_frame(end)

    def _get_current_reach(self):
        """Get the reach containing or nearest to current frame."""
        if not self.gt or not self.gt.reaches:
            return None, None
        current = self.viewer.dims.current_step[0]
        # Find reach containing current frame or nearest
        for i, r in enumerate(self.gt.reaches):
            if r.start_frame <= current <= r.end_frame:
                return i, r
        # Find nearest reach
        for i, r in enumerate(self.gt.reaches):
            if r.start_frame > current:
                return i, r
        # Return last reach if past all
        return len(self.gt.reaches) - 1, self.gt.reaches[-1]

    def _jump_to_reach_start(self):
        """Jump to start of current/nearest reach."""
        idx, reach = self._get_current_reach()
        if reach:
            self._jump_to_frame(reach.start_frame)

    def _jump_to_reach_end(self):
        """Jump to end of current/nearest reach."""
        idx, reach = self._get_current_reach()
        if reach:
            self._jump_to_frame(reach.end_frame)

    def _jump_to_prev_reach(self):
        """Jump to previous reach start."""
        if not self.gt or not self.gt.reaches:
            return
        current = self.viewer.dims.current_step[0]
        # Find reach before current frame
        for r in reversed(self.gt.reaches):
            if r.start_frame < current - 5:
                self._jump_to_frame(r.start_frame)
                return
        # Go to first reach
        self._jump_to_frame(self.gt.reaches[0].start_frame)

    def _jump_to_next_reach(self):
        """Jump to next reach start."""
        if not self.gt or not self.gt.reaches:
            return
        current = self.viewer.dims.current_step[0]
        # Find reach after current frame
        for r in self.gt.reaches:
            if r.start_frame > current + 5:
                self._jump_to_frame(r.start_frame)
                return
        # Go to last reach
        self._jump_to_frame(self.gt.reaches[-1].start_frame)

    def _jump_to_interaction(self):
        """Jump to pellet interaction frame for current segment."""
        if not self.gt or not self.gt.outcomes:
            show_warning("No outcome data loaded")
            return
        current = self.viewer.dims.current_step[0]
        # Find the outcome for current segment
        seg_start, seg_end = self._get_current_segment()
        for outcome in self.gt.outcomes:
            # Check if outcome is in current segment
            if hasattr(outcome, 'interaction_frame') and outcome.interaction_frame:
                if seg_start <= outcome.interaction_frame <= seg_end:
                    self._jump_to_frame(outcome.interaction_frame)
                    return
            # Fallback: if outcome has segment_index, match by position
            if hasattr(outcome, 'segment_index'):
                # Count which segment we're in
                seg_idx = 0
                for b in self.gt.boundaries:
                    if b.frame < current:
                        seg_idx += 1
                if outcome.segment_index == seg_idx and hasattr(outcome, 'interaction_frame'):
                    if outcome.interaction_frame:
                        self._jump_to_frame(outcome.interaction_frame)
                        return

    # =========================================================================
    # Playback
    # =========================================================================

    def _set_speed(self, speed: float):
        """Set playback speed."""
        self.playback_speed = speed
        for s, btn in self.speed_buttons.items():
            btn.setChecked(s == speed)
        if self.is_playing:
            self._update_playback_timer()

    def _update_playback_timer(self):
        """Update timer interval."""
        self.playback_timer.stop()
        if self.playback_speed >= 1:
            interval = max(1, int(1000 / self.fps))
        else:
            interval = max(1, int(1000 / (self.fps * self.playback_speed)))
        self.playback_timer.start(interval)

    def _play_forward(self):
        """Start forward playback."""
        self.playback_direction = 1
        self._start_playback()

    def _play_reverse(self):
        """Start reverse playback."""
        self.playback_direction = -1
        self._start_playback()

    def _stop_play(self):
        """Stop playback."""
        self.is_playing = False
        self.playback_timer.stop()
        self.play_btn.setText("Play")
        self.play_rev_btn.setText("Rev")

    def _start_playback(self):
        """Start playback."""
        if self.is_playing:
            self._stop_play()
            return

        self.is_playing = True
        self._update_playback_timer()

        if self.playback_direction == 1:
            self.play_btn.setText("||")
        else:
            self.play_rev_btn.setText("||")

    def _playback_step(self):
        """Advance one playback frame."""
        current = self.viewer.dims.current_step[0]
        skip = int(self.playback_speed) if self.playback_speed >= 1 else 1
        new_frame = current + (skip * self.playback_direction)

        if 0 <= new_frame < self.n_frames:
            self.viewer.dims.set_current_step(0, new_frame)
        else:
            self._stop_play()

    def _enable_nav_controls(self, enabled: bool):
        """Enable/disable navigation controls."""
        self.play_btn.setEnabled(enabled)
        self.play_rev_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)

    # =========================================================================
    # Keybindings
    # =========================================================================

    def _setup_keybindings(self):
        """Set up keyboard shortcuts."""
        @self.viewer.bind_key('Left', overwrite=True)
        def left_key(viewer):
            self._jump_frames(-1)

        @self.viewer.bind_key('Right', overwrite=True)
        def right_key(viewer):
            self._jump_frames(1)

        @self.viewer.bind_key('Shift-Left', overwrite=True)
        def shift_left(viewer):
            self._jump_frames(-10)

        @self.viewer.bind_key('Shift-Right', overwrite=True)
        def shift_right(viewer):
            self._jump_frames(10)

        @self.viewer.bind_key('Control-Left', overwrite=True)
        def ctrl_left(viewer):
            self._jump_frames(-100)

        @self.viewer.bind_key('Control-Right', overwrite=True)
        def ctrl_right(viewer):
            self._jump_frames(100)

        @self.viewer.bind_key('Space', overwrite=True)
        def toggle_play(viewer):
            if self.is_playing:
                self._stop_play()
            else:
                self._play_forward()

        @self.viewer.bind_key('j', overwrite=True)
        def next_hand(viewer):
            self._jump_to_next_hand()

        @self.viewer.bind_key('k', overwrite=True)
        def prev_hand(viewer):
            self._jump_to_prev_hand()

        @self.viewer.bind_key('Control-s', overwrite=True)
        def save(viewer):
            self._save_progress()

        @self.viewer.bind_key('i', overwrite=True)
        def set_interaction(viewer):
            """Set interaction frame on selected outcome."""
            if self._selected_outcome:
                self._set_interaction_frame(self._selected_outcome)
            else:
                show_warning("No outcome selected. Click Jump on an outcome first.")

        @self.viewer.bind_key('o', overwrite=True)
        def set_outcome_known(viewer):
            """Set outcome known frame on selected outcome (O for outcome)."""
            if self._selected_outcome:
                self._set_outcome_known_frame(self._selected_outcome)
            else:
                show_warning("No outcome selected. Click Jump on an outcome first.")

        @self.viewer.bind_key('s', overwrite=True)
        def set_reach_start(viewer):
            """Set reach start frame on selected reach."""
            if self._selected_reach:
                self._set_reach_start(self._selected_reach)
            else:
                show_warning("No reach selected. Click Jump on a reach first.")

        @self.viewer.bind_key('e', overwrite=True)
        def set_reach_end(viewer):
            """Set reach end frame on selected reach."""
            if self._selected_reach:
                self._set_reach_end(self._selected_reach)
            else:
                show_warning("No reach selected. Click Jump on a reach first.")

        @self.viewer.bind_key('v', overwrite=True)
        def set_boundary_frame(viewer):
            """Set current frame as boundary (GT Tool)."""
            if self._selected_boundary:
                self._set_boundary_frame(self._selected_boundary)
            else:
                show_warning("No boundary selected. Click Jump on a boundary first.")

        @self.viewer.bind_key('Delete', overwrite=True)
        def delete_selected(viewer):
            """Delete selected reach."""
            if self._selected_reach:
                self._delete_reach(self._selected_reach)

        @self.viewer.bind_key('a', overwrite=True)
        def add_reach_here(viewer):
            """Add new reach starting at current frame."""
            self._add_reach()

        # Navigation keys
        @self.viewer.bind_key('n', overwrite=True)
        def next_reach(viewer):
            """Jump to next reach."""
            self._jump_to_next_reach()

        @self.viewer.bind_key('p', overwrite=True)
        def prev_reach(viewer):
            """Jump to previous reach."""
            self._jump_to_prev_reach()

        @self.viewer.bind_key('[', overwrite=True)
        def seg_start(viewer):
            """Jump to segment start."""
            self._jump_to_segment_start()

        @self.viewer.bind_key(']', overwrite=True)
        def seg_end(viewer):
            """Jump to segment end."""
            self._jump_to_segment_end()

        @self.viewer.bind_key('Shift-i', overwrite=True)
        def jump_interaction(viewer):
            """Jump to interaction frame."""
            self._jump_to_interaction()

        # Speed keys 1-5
        @self.viewer.bind_key('1', overwrite=True)
        def speed_1(viewer):
            self._set_speed(0.25)

        @self.viewer.bind_key('2', overwrite=True)
        def speed_2(viewer):
            self._set_speed(0.5)

        @self.viewer.bind_key('3', overwrite=True)
        def speed_3(viewer):
            self._set_speed(1)

        @self.viewer.bind_key('4', overwrite=True)
        def speed_4(viewer):
            self._set_speed(2)

        @self.viewer.bind_key('5', overwrite=True)
        def speed_5(viewer):
            self._set_speed(4)

        @self.viewer.bind_key('6', overwrite=True)
        def speed_6(viewer):
            self._set_speed(8)

        @self.viewer.bind_key('7', overwrite=True)
        def speed_7(viewer):
            self._set_speed(16)


def _resolve_decision_window(video_path, segment_num, algo_dir):
    """
    Look up the v4.0.0+ decision_window for (video, segment) from the
    pellet_outcomes JSON. Returns (start, end, outcome, rule) or None.
    """
    import json as _json
    if algo_dir is None:
        # Default: same dir as video.
        algo_dir = video_path.parent
    else:
        algo_dir = Path(algo_dir)
    video_stem = video_path.stem.replace("_preview", "")
    outcome_json = algo_dir / f"{video_stem}_pellet_outcomes.json"
    if not outcome_json.exists():
        return None
    try:
        data = _json.loads(outcome_json.read_text(encoding='utf-8'))
    except Exception:
        return None
    for seg in data.get('segments', []):
        if seg.get('segment_num') == segment_num:
            ws = seg.get('decision_window_start')
            we = seg.get('decision_window_end')
            outc = seg.get('outcome')
            rule = seg.get('decision_rule') or seg.get('flag_reason')
            if ws is not None and we is not None:
                return int(ws), int(we), outc, rule
            return None
    return None


def _apply_segment_jump(widget, video_path, segment_num, algo_dir):
    """After video is loaded, jump to decision_window_start for the given segment."""
    if segment_num is None:
        return
    info = _resolve_decision_window(video_path, segment_num, algo_dir)
    if info is None:
        print(f"[mousereach-gt] No decision_window found for {video_path.name} seg {segment_num}; jumping to segment start")
        if widget.gt and widget.gt.boundaries and segment_num <= len(widget.gt.boundaries):
            target = widget.gt.boundaries[segment_num - 1].frame
            widget._jump_to_frame(target)
        return
    ws, we, outc, rule = info
    print(f"[mousereach-gt] Segment {segment_num}: algo says '{outc}'")
    print(f"  decision_window: frames [{ws}, {we}]  ({we - ws + 1} frames)")
    if rule:
        print(f"  rule: {rule}")
    widget._jump_to_frame(ws)


def _resolve_screenshot_dir(explicit, algo_dir, video_path, segment_num):
    """
    Decide where the Screenshot button saves by default.

    Priority:
      1. --screenshot-dir flag (explicit), if provided
      2. Auto-derive from algo_dir's parent Improvement_Snapshots structure
         when algo-dir + segment + video are all available:
         <snapshots>/screenshots/case_{video_id}_seg{N}/
      3. None -> caller falls back to cwd
    """
    from pathlib import Path as _P
    if explicit is not None:
        return _P(explicit)
    if algo_dir is None or video_path is None or segment_num is None:
        return None
    # Look for an Improvement_Snapshots parent above algo_dir (heuristic);
    # otherwise fall back to algo_dir/screenshots.
    algo_dir = _P(algo_dir)
    candidate_root = None
    # Walk up looking for "outcome_*" snapshot dir
    for parent in [algo_dir, *algo_dir.parents]:
        if parent.name.startswith('outcome_v'):
            candidate_root = parent
            break
    if candidate_root is None:
        candidate_root = algo_dir
    video_id = _P(video_path).stem.replace('_preview', '')
    return candidate_root / 'screenshots' / f'case_{video_id}_seg{segment_num}'


def _detect_panel_mode(algo_dir, explicit_mode):
    """
    Decide what the live info panel should show. If user passed --mode
    explicitly, honor it. Otherwise, auto-detect from algo_dir contents.
    """
    if explicit_mode:
        return explicit_mode
    if algo_dir is None:
        return 'general'
    try:
        algo_dir = Path(algo_dir)
        if any(algo_dir.glob('*_pellet_outcomes.json')):
            return 'outcome'
        if any(algo_dir.glob('*_reaches.json')):
            return 'reach'
        if any(algo_dir.glob('*_segments.json')):
            return 'segmentation'
    except Exception:
        pass
    return 'general'


def _resolve_segment_window(video_path, segment_num, algo_dir, pre_pad=0, post_pad=0):
    """
    Pre-load helper: return the (start, end) window the GT tool should
    actually load video frames for, given a segment number. Returns None
    if no decision_window is available (caller should load full video).
    Optional pre_pad/post_pad extends the window with extra context frames.

    v4.0.0+ refinement: when the GT interaction frame falls OUTSIDE the
    algo's decision_window (e.g., algo picked the wrong causal reach),
    expand the loaded window to include GT's interaction frame plus a
    margin. This ensures the visible event in GT is always loadable.
    """
    info = _resolve_decision_window(video_path, segment_num, algo_dir)
    if info is None:
        return None
    ws, we, _, _ = info
    final_start = max(0, int(ws) - int(pre_pad))
    final_end = int(we) + int(post_pad)

    # Hard rule: if GT has an interaction frame for this segment, and it
    # falls outside the current loaded window, expand the window.
    try:
        import json as _json
        video_id = video_path.stem.replace('_preview', '')
        # GT files might live in algo_dir or a sibling 'gt' dir; try a few
        # canonical locations.
        candidates = []
        if algo_dir is not None:
            candidates.append(Path(algo_dir).parent / 'gt' / f'{video_id}_unified_ground_truth.json')
            candidates.append(Path(algo_dir) / f'{video_id}_unified_ground_truth.json')
        candidates.append(video_path.parent / f'{video_id}_unified_ground_truth.json')
        gt_path = next((p for p in candidates if p.exists()), None)
        if gt_path is not None:
            gt = _json.loads(gt_path.read_text(encoding='utf-8'))
            gt_segs = gt.get('outcomes', {}).get('segments', [])
            gt_seg = next((s for s in gt_segs if s.get('segment_num') == segment_num), None)
            if gt_seg is not None:
                ifr = gt_seg.get('interaction_frame')
                if ifr is not None:
                    margin = 200  # frames either side of GT interaction
                    if ifr - margin < final_start:
                        final_start = max(0, int(ifr) - margin)
                        print(f'[mousereach-gt] expanding load window backward to GT interaction {ifr}')
                    if ifr + margin > final_end:
                        final_end = int(ifr) + margin
                        print(f'[mousereach-gt] expanding load window forward to GT interaction {ifr}')
    except Exception:
        pass
    return (final_start, final_end)


def main():
    """Launch the Ground Truth Tool (default mode)."""
    import argparse

    parser = argparse.ArgumentParser(description="MouseReach Ground Truth Tool")
    parser.add_argument('video', nargs='?', type=Path, help="Video file to load")
    parser.add_argument('--segment', type=int, default=None,
                        help="Jump to this segment's decision_window after load")
    parser.add_argument('--frame', type=int, default=None,
                        help="Jump to this absolute frame after load")
    parser.add_argument('--algo-dir', type=Path, default=None,
                        help="Directory containing the *_pellet_outcomes.json "
                             "to read decision_window from. Defaults to the video's directory.")
    parser.add_argument('--mode', choices=['outcome', 'reach', 'segmentation', 'general'],
                        default=None,
                        help="Info panel mode (auto-detected from --algo-dir if omitted)")
    parser.add_argument('--screenshot-dir', type=Path, default=None,
                        help="Default save dir for the Screenshot button. "
                             "Auto-derived from --algo-dir + --segment if omitted.")
    parser.add_argument('--pre-pad', type=int, default=0,
                        help="Extra frames to load BEFORE the decision window (default 0)")
    parser.add_argument('--post-pad', type=int, default=0,
                        help="Extra frames to load AFTER the decision window (default 0)")
    args = parser.parse_args()

    viewer = napari.Viewer(title="MouseReach Ground Truth Tool")
    widget = GroundTruthWidget(viewer, review_mode=False)
    viewer.window.add_dock_widget(widget, name="Ground Truth Tool", area="right")

    if args.video:
        # Resolve frame_range so we only LOAD the decision-window frames.
        frame_range = None
        if args.segment is not None:
            frame_range = _resolve_segment_window(args.video, args.segment, args.algo_dir,
                                                   pre_pad=args.pre_pad, post_pad=args.post_pad)
        # Set info_panel_mode for the live panel (auto-detected from algo-dir
        # contents when --segment + --algo-dir are given).
        widget.info_panel_mode = _detect_panel_mode(args.algo_dir, args.mode)
        widget.screenshot_dir = _resolve_screenshot_dir(
            args.screenshot_dir, args.algo_dir, args.video, args.segment
        )
        widget._screenshot_segment = args.segment
        widget._load_video(args.video, frame_range=frame_range)
        if args.segment is not None:
            _apply_segment_jump(widget, args.video, args.segment, args.algo_dir)
        elif args.frame is not None:
            widget._jump_to_frame(args.frame)

    napari.run()


def main_review():
    """Launch the Review Tool (saves to algo files, not GT files)."""
    import argparse

    parser = argparse.ArgumentParser(description="MouseReach Review Tool")
    parser.add_argument('video', nargs='?', type=Path, help="Video file to load")
    parser.add_argument('--segment', type=int, default=None,
                        help="Jump to this segment's decision_window after load")
    parser.add_argument('--frame', type=int, default=None,
                        help="Jump to this absolute frame after load")
    parser.add_argument('--algo-dir', type=Path, default=None,
                        help="Directory containing the *_pellet_outcomes.json")
    parser.add_argument('--mode', choices=['outcome', 'reach', 'segmentation', 'general'],
                        default=None,
                        help="Info panel mode (auto-detected from --algo-dir if omitted)")
    parser.add_argument('--screenshot-dir', type=Path, default=None,
                        help="Default save dir for the Screenshot button. "
                             "Auto-derived from --algo-dir + --segment if omitted.")
    parser.add_argument('--pre-pad', type=int, default=0,
                        help="Extra frames to load BEFORE the decision window (default 0)")
    parser.add_argument('--post-pad', type=int, default=0,
                        help="Extra frames to load AFTER the decision window (default 0)")
    args = parser.parse_args()

    viewer = napari.Viewer(title="MouseReach Review Tool")
    widget = GroundTruthWidget(viewer, review_mode=True)
    viewer.window.add_dock_widget(widget, name="Review Tool", area="right")

    if args.video:
        # Resolve frame_range so we only LOAD the decision-window frames.
        frame_range = None
        if args.segment is not None:
            frame_range = _resolve_segment_window(args.video, args.segment, args.algo_dir,
                                                   pre_pad=args.pre_pad, post_pad=args.post_pad)
        widget.info_panel_mode = _detect_panel_mode(args.algo_dir, args.mode)
        widget.screenshot_dir = _resolve_screenshot_dir(
            args.screenshot_dir, args.algo_dir, args.video, args.segment
        )
        widget._screenshot_segment = args.segment
        widget._load_video(args.video, frame_range=frame_range)
        if args.segment is not None:
            _apply_segment_jump(widget, args.video, args.segment, args.algo_dir)
        elif args.frame is not None:
            widget._jump_to_frame(args.frame)

    napari.run()


if __name__ == "__main__":
    main()
