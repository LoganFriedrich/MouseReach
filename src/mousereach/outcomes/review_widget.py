"""
MouseReach Pellet Outcome Annotator
===============================

Napari widget for reviewing and correcting pellet outcome detection.

Workflow:
1. Load video (auto-loads DLC, segments, algorithm outcomes)
2. Navigate through segments
3. Mark interaction frame (first pellet touch)
4. Mark outcome known frame (when outcome determinable)
5. Classify outcome: retrieved, displaced_sa, displaced_outside, untouched
6. Flag uncertain cases for review
7. Save validated or ground truth

Outcomes:
- retrieved: Mouse grabbed and ate the pellet
- displaced_sa: Pellet knocked into scoring area
- displaced_outside: Pellet knocked outside scoring area
- untouched: Pellet still on pillar at segment end
- no_pellet: No pellet visible (unusual, flagged automatically)

Install as plugin:
    pip install -e .
    # Then: Plugins → MouseReach Pellet Outcomes → Pellet Outcome Annotator

Or run standalone:
    python -m mousereach_pellet_outcomes._napari_widget
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import json
import os
from datetime import datetime

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QListWidgetItem, QCheckBox, QLineEdit, QSlider, QButtonGroup,
    QRadioButton, QScrollArea
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont

import napari
from napari.utils.notifications import show_info, show_warning, show_error
import pandas as pd
import cv2

from mousereach.review import ComparisonPanel, create_outcome_comparison
from mousereach.review.save_panel import SimpleSavePanel


# Outcome types and colors
OUTCOMES = ['retrieved', 'displaced_sa', 'displaced_outside', 'untouched']
OUTCOME_LABELS = {
    'retrieved': 'Retrieved (R)',
    'displaced_sa': 'Displaced into SA (D)',
    'displaced_outside': 'Displaced outside (O)',
    'untouched': 'Untouched/Missed (U)',
}
OUTCOME_COLORS = {
    'retrieved': '#4CAF50',      # Green
    'displaced_sa': '#FF9800',   # Orange
    'displaced_outside': '#FF5722',  # Deep orange
    'untouched': '#2196F3',      # Blue
    'no_pellet': '#9E9E9E',      # Gray
    'uncertain': '#9C27B0',      # Purple
}


def get_username():
    """Get current username."""
    try:
        return os.getlogin()
    except (OSError, AttributeError):
        return os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))


class PelletOutcomeAnnotatorWidget(QWidget):
    """
    Widget for annotating/correcting pellet outcome detection.
    """

    def __init__(self, napari_viewer: napari.Viewer, embedded_mode: bool = False):
        super().__init__()
        self.viewer = napari_viewer
        self.embedded_mode = embedded_mode  # When True, hides video load and nav controls

        # Data
        self.video_path = None
        self.dlc_path = None
        self.segments_path = None
        self.outcomes_path = None

        self.dlc_df = None
        self.boundaries = []
        self.outcomes_data = None
        self.original_outcomes_json = None

        # State
        self.n_frames = 0
        self.fps = 60.0
        self.current_segment = 1
        self.is_playing = False
        self.playback_speed = 1.0  # 0.25, 0.5, 1, 2, 4, 8, 16
        self.playback_direction = 1  # 1 = forward, -1 = backward
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        # Layers
        self.video_layer = None
        self.points_layer = None
        self.pillar_shapes_layer = None

        # Cache
        self.cached_video_path = None  # Track last loaded video to avoid reloading

        self._build_ui()
        if not embedded_mode:
            self._setup_keybindings()
    
    def _build_ui(self):
        # Main layout for the widget
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll)
        
        # Inner widget that holds all content
        inner_widget = QWidget()
        scroll.setWidget(inner_widget)
        layout = QVBoxLayout()
        inner_widget.setLayout(layout)
        
        # === Instructions (hidden in embedded mode) ===
        if not self.embedded_mode:
            instructions = QLabel(
                "Space=play, Shift+R=reverse, N/P=next/prev segment\n"
                "I=set interaction, K=set outcome known\n"
                "R/D/O/U=retrieved/displaced SA/outside/untouched\n\n"
                "Cyan circle = Geometric pillar position (algorithm's reference)"
            )
            instructions.setWordWrap(True)
            layout.addWidget(instructions)

        # === File Loading (hidden in embedded mode) ===
        if not self.embedded_mode:
            file_group = QGroupBox("1. Load Video")
            file_layout = QVBoxLayout()
            file_group.setLayout(file_layout)

            self.video_label = QLabel("No video loaded")
            self.video_label.setWordWrap(True)
            file_layout.addWidget(self.video_label)

            # File status indicators
            self.file_status_label = QLabel("")
            self.file_status_label.setWordWrap(True)
            self.file_status_label.setStyleSheet("font-size: 10px; padding: 2px;")
            file_layout.addWidget(self.file_status_label)

            self.load_btn = QPushButton("Select Video...")
            self.load_btn.clicked.connect(self._load_video)
            file_layout.addWidget(self.load_btn)

            self.reload_btn = QPushButton("Reload Data (Keep Video)")
            self.reload_btn.clicked.connect(self._reload_data_only)
            self.reload_btn.setEnabled(False)
            file_layout.addWidget(self.reload_btn)

            self.progress = QProgressBar()
            self.progress.setVisible(False)
            file_layout.addWidget(self.progress)

            # Pillar visibility toggle
            self.show_pillar_check = QCheckBox("Show pillar circle")
            self.show_pillar_check.setChecked(True)
            self.show_pillar_check.stateChanged.connect(self._toggle_pillar_visibility)
            file_layout.addWidget(self.show_pillar_check)

            # Pillar circle size control
            pillar_size_layout = QHBoxLayout()
            pillar_size_layout.addWidget(QLabel("Circle size:"))
            self.pillar_size_slider = QSlider(Qt.Horizontal)
            self.pillar_size_slider.setMinimum(1)
            self.pillar_size_slider.setMaximum(100)
            self.pillar_size_slider.setValue(100)  # Default to 10.0 (1.0x actual size)
            self.pillar_size_slider.valueChanged.connect(self._on_pillar_size_change)
            pillar_size_layout.addWidget(self.pillar_size_slider)
            self.pillar_size_label = QLabel("10.0")
            pillar_size_layout.addWidget(self.pillar_size_label)
            file_layout.addLayout(pillar_size_layout)

            layout.addWidget(file_group)
        else:
            # Create dummy attributes for embedded mode
            self.video_label = QLabel()
            self.file_status_label = QLabel()
            self.load_btn = QPushButton()
            self.reload_btn = QPushButton()
            self.progress = QProgressBar()
            self.show_pillar_check = QCheckBox()
            self.pillar_size_slider = QSlider()
            self.pillar_size_label = QLabel()

        # === Navigation (hidden in embedded mode) ===
        if not self.embedded_mode:
            nav_group = QGroupBox("2. Navigate")
            nav_layout = QVBoxLayout()
            nav_group.setLayout(nav_layout)

            # Frame info
            self.frame_label = QLabel("Frame: -- / --")
            font = QFont()
            font.setPointSize(14)
            font.setBold(True)
            self.frame_label.setFont(font)
            nav_layout.addWidget(self.frame_label)

            self.segment_label = QLabel("Segment: -- / --")
            nav_layout.addWidget(self.segment_label)

            # Playback
            play_layout = QHBoxLayout()

            self.play_rev_btn = QPushButton("◀ Rev")
            self.play_rev_btn.clicked.connect(self._play_reverse)
            self.play_rev_btn.setEnabled(False)
            play_layout.addWidget(self.play_rev_btn)

            self.play_btn = QPushButton("▶ Play")
            self.play_btn.clicked.connect(self._play_forward)
            self.play_btn.setEnabled(False)
            play_layout.addWidget(self.play_btn)

            self.stop_btn = QPushButton("⏹")
            self.stop_btn.clicked.connect(self._stop_play)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setMaximumWidth(30)
            play_layout.addWidget(self.stop_btn)
            nav_layout.addLayout(play_layout)

            # Speed buttons (including slow motion)
            speed_layout = QHBoxLayout()
            speed_layout.addWidget(QLabel("Speed:"))
            self.speed_buttons = {}
            for speed in [0.25, 0.5, 1, 2, 4, 8, 16]:
                label = f"{speed}x" if speed >= 1 else f"1/{int(1/speed)}x"
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setMaximumWidth(45)
                btn.clicked.connect(lambda checked, s=speed: self._set_speed(s))
                self.speed_buttons[speed] = btn
                speed_layout.addWidget(btn)
            self.speed_buttons[1].setChecked(True)
            nav_layout.addLayout(speed_layout)

            # Frame stepping
            step_layout = QHBoxLayout()
            for delta, label in [(-100, "-100"), (-10, "-10"), (-1, "-1"),
                                 (1, "+1"), (10, "+10"), (100, "+100")]:
                btn = QPushButton(label)
                btn.clicked.connect(lambda checked, d=delta: self._jump_frames(d))
                step_layout.addWidget(btn)
            nav_layout.addLayout(step_layout)

            # Segment navigation
            seg_layout = QHBoxLayout()
            self.prev_seg_btn = QPushButton("<< Prev Segment (P)")
            self.prev_seg_btn.clicked.connect(self._prev_segment)
            self.prev_seg_btn.setEnabled(False)
            seg_layout.addWidget(self.prev_seg_btn)

            self.next_seg_btn = QPushButton("Next Segment (N) >>")
            self.next_seg_btn.clicked.connect(self._next_segment)
            self.next_seg_btn.setEnabled(False)
            seg_layout.addWidget(self.next_seg_btn)
            nav_layout.addLayout(seg_layout)

            # Jump to segment start/end
            jump_layout = QHBoxLayout()
            self.jump_start_btn = QPushButton("→ Segment Start")
            self.jump_start_btn.clicked.connect(self._jump_to_segment_start)
            self.jump_start_btn.setEnabled(False)
            jump_layout.addWidget(self.jump_start_btn)

            self.jump_end_btn = QPushButton("→ Segment End")
            self.jump_end_btn.clicked.connect(self._jump_to_segment_end)
            self.jump_end_btn.setEnabled(False)
            jump_layout.addWidget(self.jump_end_btn)
            nav_layout.addLayout(jump_layout)

            # Jump to algorithm-detected key frames
            algo_jump_layout = QHBoxLayout()
            self.jump_interaction_btn = QPushButton("→ Interaction")
            self.jump_interaction_btn.setToolTip("Jump to detected pellet interaction frame")
            self.jump_interaction_btn.clicked.connect(self._jump_to_interaction)
            self.jump_interaction_btn.setEnabled(False)
            algo_jump_layout.addWidget(self.jump_interaction_btn)

            self.jump_known_btn = QPushButton("→ Outcome Known")
            self.jump_known_btn.setToolTip("Jump to frame where outcome became clear")
            self.jump_known_btn.clicked.connect(self._jump_to_outcome_known)
            self.jump_known_btn.setEnabled(False)
            algo_jump_layout.addWidget(self.jump_known_btn)

            self.jump_reach_btn = QPushButton("→ Causal Reach")
            self.jump_reach_btn.setToolTip("Jump to reach that caused this outcome")
            self.jump_reach_btn.clicked.connect(self._jump_to_causal_reach)
            self.jump_reach_btn.setEnabled(False)
            algo_jump_layout.addWidget(self.jump_reach_btn)
            nav_layout.addLayout(algo_jump_layout)

            layout.addWidget(nav_group)
        else:
            # Create dummy attributes for embedded mode
            self.frame_label = QLabel()
            self.segment_label = QLabel()
            self.play_rev_btn = QPushButton()
            self.play_btn = QPushButton()
            self.stop_btn = QPushButton()
            self.speed_buttons = {}
            self.prev_seg_btn = QPushButton()
            self.next_seg_btn = QPushButton()
            self.jump_start_btn = QPushButton()
            self.jump_end_btn = QPushButton()
            self.jump_interaction_btn = QPushButton()
            self.jump_known_btn = QPushButton()
            self.jump_reach_btn = QPushButton()
        
        # === Current Segment Outcome ===
        outcome_title = "Annotate Outcome" if self.embedded_mode else "3. Annotate Outcome"
        outcome_group = QGroupBox(outcome_title)
        outcome_layout = QVBoxLayout()
        outcome_group.setLayout(outcome_layout)
        
        # Current outcome display
        self.current_outcome_label = QLabel("Current: --")
        font2 = QFont()
        font2.setPointSize(12)
        font2.setBold(True)
        self.current_outcome_label.setFont(font2)
        outcome_layout.addWidget(self.current_outcome_label)
        
        # Interaction frame
        interact_layout = QHBoxLayout()
        interact_layout.addWidget(QLabel("Interaction frame:"))
        self.interaction_label = QLabel("--")
        interact_layout.addWidget(self.interaction_label)
        self.set_interaction_btn = QPushButton("Set HERE (I)")
        self.set_interaction_btn.clicked.connect(self._set_interaction_frame)
        self.set_interaction_btn.setEnabled(False)
        interact_layout.addWidget(self.set_interaction_btn)
        self.clear_interaction_btn = QPushButton("Clear")
        self.clear_interaction_btn.clicked.connect(self._clear_interaction_frame)
        self.clear_interaction_btn.setEnabled(False)
        interact_layout.addWidget(self.clear_interaction_btn)
        outcome_layout.addLayout(interact_layout)
        
        # Outcome known frame
        known_layout = QHBoxLayout()
        known_layout.addWidget(QLabel("Outcome known frame:"))
        self.outcome_known_label = QLabel("--")
        known_layout.addWidget(self.outcome_known_label)
        self.set_known_btn = QPushButton("Set HERE (K)")
        self.set_known_btn.clicked.connect(self._set_outcome_known_frame)
        self.set_known_btn.setEnabled(False)
        known_layout.addWidget(self.set_known_btn)
        self.clear_known_btn = QPushButton("Clear")
        self.clear_known_btn.clicked.connect(self._clear_outcome_known_frame)
        self.clear_known_btn.setEnabled(False)
        known_layout.addWidget(self.clear_known_btn)
        outcome_layout.addLayout(known_layout)
        
        # Outcome classification
        outcome_layout.addWidget(QLabel("Outcome:"))
        
        self.outcome_group = QButtonGroup()
        outcome_btn_layout = QVBoxLayout()
        
        self.outcome_radios = {}
        for outcome in OUTCOMES:
            radio = QRadioButton(OUTCOME_LABELS[outcome])
            radio.setStyleSheet(f"color: {OUTCOME_COLORS[outcome]};")
            radio.toggled.connect(lambda checked, o=outcome: self._on_outcome_selected(o, checked))
            radio.setEnabled(False)
            self.outcome_group.addButton(radio)
            self.outcome_radios[outcome] = radio
            outcome_btn_layout.addWidget(radio)
        
        outcome_layout.addLayout(outcome_btn_layout)
        
        # Accept as-is button (explicit confirmation without changes)
        self.accept_btn = QPushButton("Accept as-is (Enter) - algorithm is correct")
        self.accept_btn.clicked.connect(self._accept_current)
        self.accept_btn.setEnabled(False)
        self.accept_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.accept_btn.setToolTip("Mark this outcome as reviewed - algorithm result is correct")
        outcome_layout.addWidget(self.accept_btn)

        # Confirm untouched button
        self.confirm_untouched_btn = QPushButton("Confirm UNTOUCHED at segment end")
        self.confirm_untouched_btn.clicked.connect(self._confirm_untouched)
        self.confirm_untouched_btn.setEnabled(False)
        self.confirm_untouched_btn.setStyleSheet(f"background-color: {OUTCOME_COLORS['untouched']}; color: white;")
        outcome_layout.addWidget(self.confirm_untouched_btn)
        
        # Algorithm info
        self.algo_info = QLabel("Algorithm said: --")
        self.algo_info.setWordWrap(True)
        self.algo_info.setStyleSheet("color: gray;")
        outcome_layout.addWidget(self.algo_info)
        
        layout.addWidget(outcome_group)
        
        # === Flag Segment ===
        flag_title = "Flag Segment" if self.embedded_mode else "4. Flag Segment"
        flag_group = QGroupBox(flag_title)
        flag_layout = QVBoxLayout()
        flag_group.setLayout(flag_layout)
        
        self.flag_check = QCheckBox("Flag this segment for review")
        self.flag_check.stateChanged.connect(self._toggle_flag)
        self.flag_check.setEnabled(False)
        flag_layout.addWidget(self.flag_check)
        
        flag_reason_layout = QHBoxLayout()
        flag_reason_layout.addWidget(QLabel("Reason:"))
        self.flag_reason = QLineEdit()
        self.flag_reason.setPlaceholderText("Why needs review...")
        self.flag_reason.setEnabled(False)
        self.flag_reason.textChanged.connect(self._update_flag_reason)
        flag_reason_layout.addWidget(self.flag_reason)
        flag_layout.addLayout(flag_reason_layout)
        
        layout.addWidget(flag_group)
        
        # === Segment List ===
        list_title = "Segments to Review" if self.embedded_mode else "5. Segments to Review"
        list_group = QGroupBox(list_title)
        list_layout = QVBoxLayout()
        list_group.setLayout(list_layout)

        # Filter toggle - by default only show items needing review
        self.show_all_check = QCheckBox("Show all segments (including high-confidence)")
        self.show_all_check.setChecked(False)  # Default: only show items needing review
        self.show_all_check.stateChanged.connect(self._update_segment_list)
        list_layout.addWidget(self.show_all_check)

        # Review progress indicator
        self.review_progress_label = QLabel("Needs review: 0 | Done: 0/0")
        self.review_progress_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        list_layout.addWidget(self.review_progress_label)

        self.summary_label = QLabel("R: -- | D: -- | U: --")
        list_layout.addWidget(self.summary_label)
        
        self.segment_list = QListWidget()
        self.segment_list.itemClicked.connect(self._select_segment)
        self.segment_list.itemDoubleClicked.connect(self._jump_to_segment_from_list)
        self.segment_list.setMaximumHeight(120)
        list_layout.addWidget(self.segment_list)
        
        layout.addWidget(list_group)

        # === Algo vs GT Comparison ===
        comparison_group = QGroupBox("Algo vs GT Comparison")
        comparison_group.setCheckable(True)
        comparison_group.setChecked(True)  # Expanded by default so UI is visible
        comparison_layout = QVBoxLayout()
        comparison_group.setLayout(comparison_layout)

        self.comparison_panel = ComparisonPanel()
        self.comparison_panel.item_selected.connect(self._on_comparison_item_selected)
        self.comparison_panel.validation_saved.connect(self._save_validated)
        if not self.embedded_mode:
            # GT save only available in standalone mode, not Review Tool
            self.comparison_panel.gt_saved.connect(self._save_ground_truth)
        comparison_layout.addWidget(self.comparison_panel)

        layout.addWidget(comparison_group)

        # === Save ===
        save_title = "Save" if self.embedded_mode else "6. Save"
        save_group = QGroupBox(save_title)
        save_layout = QVBoxLayout()
        save_group.setLayout(save_layout)

        # Use the new clear save panel with progress option
        # In embedded mode (Review Tool), hide GT option - Review Tool edits algo files only
        self.save_panel = SimpleSavePanel(show_progress_save=True, review_mode=self.embedded_mode)
        self.save_panel.save_progress.connect(self._save_progress)
        self.save_panel.save_validated.connect(self._save_validated)
        if not self.embedded_mode:
            self.save_panel.save_ground_truth.connect(self._save_ground_truth)
        save_layout.addWidget(self.save_panel)

        # Keep references for compatibility
        self.status_label = self.save_panel.status_label
        self.save_btn = self.save_panel.save_btn
        self.save_gt_btn = self.save_panel.save_gt_btn
        self.save_progress_btn = self.save_panel.save_progress_btn

        layout.addWidget(save_group)

        layout.addStretch()
    
    def _setup_keybindings(self):
        """Set up keyboard shortcuts."""
        @self.viewer.bind_key('Space', overwrite=True)
        def toggle_play(viewer):
            self._toggle_play()
        
        @self.viewer.bind_key('Shift-r', overwrite=True)
        def reverse_play(viewer):
            if self.is_playing and self.playback_direction == -1:
                self._stop_play()
            else:
                self._play_reverse()
        
        @self.viewer.bind_key('Left', overwrite=True)
        def step_back(viewer):
            self._jump_frames(-1)
        
        @self.viewer.bind_key('Right', overwrite=True)
        def step_forward(viewer):
            self._jump_frames(1)
        
        @self.viewer.bind_key('Shift-Left', overwrite=True)
        def step_back_10(viewer):
            self._jump_frames(-10)
        
        @self.viewer.bind_key('Shift-Right', overwrite=True)
        def step_forward_10(viewer):
            self._jump_frames(10)
        
        @self.viewer.bind_key('n', overwrite=True)
        def next_seg(viewer):
            self._next_segment()
        
        @self.viewer.bind_key('p', overwrite=True)
        def prev_seg(viewer):
            self._prev_segment()
        
        @self.viewer.bind_key('i', overwrite=True)
        def set_interaction(viewer):
            self._set_interaction_frame()
        
        @self.viewer.bind_key('k', overwrite=True)
        def set_known(viewer):
            self._set_outcome_known_frame()
        
        @self.viewer.bind_key('r', overwrite=True)
        def set_retrieved(viewer):
            self._set_outcome('retrieved')
        
        @self.viewer.bind_key('d', overwrite=True)
        def set_displaced_sa(viewer):
            self._set_outcome('displaced_sa')
        
        @self.viewer.bind_key('o', overwrite=True)
        def set_displaced_outside(viewer):
            self._set_outcome('displaced_outside')
        
        @self.viewer.bind_key('u', overwrite=True)
        def set_untouched(viewer):
            self._set_outcome('untouched')
        
        @self.viewer.bind_key('Control-s', overwrite=True)
        def save(viewer):
            self._save_progress()

        @self.viewer.bind_key('Return', overwrite=True)
        def accept_current(viewer):
            self._accept_current()
    
    def _load_video(self):
        """Load video via file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video files (*.mp4 *.avi *.mkv);;All files (*)"
        )
        if not path:
            return
        self._load_video_from_path(Path(path))

    def _load_video_from_path(self, video_path: Path):
        """Load video and associated files from a path (for CLI/programmatic use)."""
        self.video_path = video_path
        video_dir = self.video_path.parent
        video_stem = self.video_path.stem

        video_id = video_stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]
        
        self.video_label.setText(f"Loading: {self.video_path.name}")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        try:
            # Find DLC file
            dlc_files = list(video_dir.glob(f"{video_id}*DLC*.h5"))
            if dlc_files:
                self.dlc_path = dlc_files[0]
                self._load_dlc()
            else:
                self.dlc_df = None
                show_warning("No DLC file found")
            
            self.progress.setValue(20)
            
            # Find segments
            seg_patterns = [
                f"{video_id}_seg_validation.json",
                f"{video_id}_segments_v2.json",
                f"{video_id}_segments.json"
            ]
            self.segments_path = None
            for pattern in seg_patterns:
                candidate = video_dir / pattern
                if candidate.exists():
                    self.segments_path = candidate
                    break
            
            if not self.segments_path:
                show_error("No segments file found!")
                return
            
            self._load_segments()
            self.progress.setValue(30)

            # Find outcomes (ground truth takes priority)
            outcome_patterns = [
                f"{video_id}_outcome_ground_truth.json",  # Ground truth first
                f"{video_id}_pellet_outcomes.json",
            ]
            self.outcomes_path = None
            self._using_outcome_ground_truth = False
            for pattern in outcome_patterns:
                candidate = video_dir / pattern
                if candidate.exists():
                    self.outcomes_path = candidate
                    if 'ground_truth' in pattern:
                        self._using_outcome_ground_truth = True
                    break

            if self.outcomes_path:
                self._load_outcomes()
            else:
                show_warning(
                    f"No outcomes file found for {video_id}.\n"
                    "Run the pipeline first to generate outcome detection results."
                )
                return
            
            self.progress.setValue(40)

            # Load video (skip if same video already loaded)
            if self.cached_video_path != self.video_path:
                self._load_video_frames()
                self.cached_video_path = self.video_path
            else:
                show_info("Video already loaded, skipping reload")

            # Add DLC overlay
            if self.dlc_df is not None:
                self._add_points_layer()
                self._add_pillar_shapes_layer()

            self.progress.setValue(95)
            
            # Enable controls
            self._enable_controls(True)
            self.reload_btn.setEnabled(True)  # Enable reload button after first load

            # Connect frame change
            self.viewer.dims.events.current_step.connect(self._on_frame_change)
            
            self.video_label.setText(
                f"{self.video_path.name}\n"
                f"Segments: {len(self.boundaries)-1}"
            )

            # Check for related files
            self._update_file_status()

            self.progress.setValue(100)

            # Process events to ensure progress bar shows 100%
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()

            # Check for work in progress
            if self.outcomes_data and self.outcomes_data.get('work_in_progress'):
                last_seg = self.outcomes_data.get('last_segment', 1)
                last_frame = self.outcomes_data.get('last_frame', 0)
                last_user = self.outcomes_data.get('last_edited_by', 'unknown')
                last_time = self.outcomes_data.get('last_edited_at', '')[:10]

                reply = QMessageBox.question(
                    self, "Resume Work?",
                    f"Previous session found:\n"
                    f"  By: {last_user}\n"
                    f"  Date: {last_time}\n"
                    f"  Segment {last_seg}, frame {last_frame}\n\n"
                    f"Resume from where you left off?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self._goto_segment(last_seg)
                    self.viewer.dims.set_current_step(0, last_frame)
                    show_info(f"Resumed at segment {last_seg}, frame {last_frame}")
                else:
                    self._goto_segment(1)
            else:
                # Go to first segment
                self._goto_segment(1)

            # Hide progress bar after successful load
            self.progress.setVisible(False)

        except Exception as e:
            show_error(f"Error loading: {e}")
            import traceback
            traceback.print_exc()
            # Keep progress bar visible on error so user can see what happened
            # It will be hidden on next successful load
    
    def _reload_data_only(self):
        """Reload DLC, segments, and outcomes without reloading video frames."""
        if not self.video_path:
            show_warning("No video loaded yet")
            return

        try:
            show_info("Reloading data (keeping video frames)...")

            # Reload DLC, segments, outcomes
            self._load_dlc()
            self._load_segments()
            self._load_outcomes()

            # Refresh DLC points and pillar overlay
            if self.dlc_df is not None:
                self._add_points_layer()
                self._add_pillar_shapes_layer()

            # Update display
            self._update_file_status()
            self._goto_segment(self.current_segment)  # Refresh current segment display

            show_info("Data reloaded successfully!")

        except Exception as e:
            show_error(f"Error reloading data: {e}")
            import traceback
            traceback.print_exc()

    def _load_data_only(self, video_path: Path):
        """
        Load just the data files (DLC, segments, outcomes) without loading the video.

        Used when the video is already loaded via shared state manager.
        The shared video layer and frame data should already be set on self:
            self._shared_video_layer, self._shared_video_frames,
            self._shared_n_frames, self._shared_video_fps
        """
        self.video_path = video_path
        video_dir = self.video_path.parent
        video_stem = self.video_path.stem

        video_id = video_stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        self.video_label.setText(f"Loading data: {self.video_path.name}")

        try:
            # Use shared video data
            if hasattr(self, '_shared_video_layer') and self._shared_video_layer is not None:
                self.video_layer = self._shared_video_layer
                self.n_frames = self._shared_n_frames
                self.fps = self._shared_video_fps
            else:
                show_error("No video loaded - load a video first")
                return

            # Find and load DLC file
            dlc_files = list(video_dir.glob(f"{video_id}*DLC*.h5"))
            if dlc_files:
                self.dlc_path = dlc_files[0]
                self._load_dlc()
            else:
                self.dlc_df = None
                show_warning("No DLC file found")

            # Find segments file
            seg_patterns = [
                f"{video_id}_seg_validation.json",
                f"{video_id}_segments_v2.json",
                f"{video_id}_segments.json"
            ]
            self.segments_path = None
            for pattern in seg_patterns:
                candidate = video_dir / pattern
                if candidate.exists():
                    self.segments_path = candidate
                    break

            if not self.segments_path:
                show_error("No segments file found!")
                return

            self._load_segments()

            # Find outcomes file (ground truth takes priority)
            outcome_patterns = [
                f"{video_id}_outcome_ground_truth.json",  # Ground truth first
                f"{video_id}_pellet_outcomes.json",
            ]
            self.outcomes_path = None
            self._using_outcome_ground_truth = False
            for pattern in outcome_patterns:
                candidate = video_dir / pattern
                if candidate.exists():
                    self.outcomes_path = candidate
                    if 'ground_truth' in pattern:
                        self._using_outcome_ground_truth = True
                    break

            if self.outcomes_path:
                self._load_outcomes()
            else:
                show_warning(
                    f"No outcomes file found for {video_id}.\n"
                    "Run the pipeline first to generate outcome detection results."
                )
                return

            # Add DLC overlay
            if self.dlc_df is not None:
                self._add_points_layer()
                self._add_pillar_shapes_layer()

            # Enable controls
            self._enable_controls(True)
            self.reload_btn.setEnabled(True)

            # Connect frame change
            self.viewer.dims.events.current_step.connect(self._on_frame_change)

            self.video_label.setText(
                f"{self.video_path.name}\n"
                f"Segments: {len(self.boundaries)-1}"
            )

            # Check for related files
            self._update_file_status()

            # Check for work in progress
            if self.outcomes_data and self.outcomes_data.get('work_in_progress'):
                last_seg = self.outcomes_data.get('last_segment', 1)
                last_frame = self.outcomes_data.get('last_frame', 0)
                last_user = self.outcomes_data.get('last_edited_by', 'unknown')
                last_time = self.outcomes_data.get('last_edited_at', '')[:10]

                reply = QMessageBox.question(
                    self, "Resume Work?",
                    f"Previous session found:\n"
                    f"  By: {last_user}\n"
                    f"  Date: {last_time}\n"
                    f"  Segment {last_seg}, frame {last_frame}\n\n"
                    f"Resume from where you left off?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self._goto_segment(last_seg)
                    self.viewer.dims.set_current_step(0, last_frame)
                    show_info(f"Resumed at segment {last_seg}, frame {last_frame}")
                else:
                    self._goto_segment(1)
            else:
                self._goto_segment(1)

        except Exception as e:
            show_error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()

    def _load_dlc(self):
        df = pd.read_hdf(self.dlc_path)
        df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
        self.dlc_df = df
    
    def _load_segments(self):
        with open(self.segments_path) as f:
            data = json.load(f)
        self.boundaries = data.get('boundaries', data.get('validated_boundaries', []))
    
    def _load_outcomes(self):
        with open(self.outcomes_path) as f:
            self.outcomes_data = json.load(f)
        self.original_outcomes_json = json.dumps(self.outcomes_data)
    
    def _run_detection(self):
        try:
            from mousereach.outcomes.core import PelletOutcomeDetector
            
            detector = PelletOutcomeDetector()
            
            video_id = self.video_path.stem
            if 'DLC_' in video_id:
                video_id = video_id.split('DLC_')[0]
            
            # Check for reaches file
            reaches_path = self.video_path.parent / f"{video_id}_reaches.json"
            if not reaches_path.exists():
                reaches_path = None
            
            results = detector.detect(self.dlc_path, self.segments_path, reaches_path)
            
            self.outcomes_path = self.video_path.parent / f"{video_id}_pellet_outcomes.json"
            detector.save_results(results, self.outcomes_path)
            
            self._load_outcomes()
            show_info(f"Detection complete: {results.summary}")
            
        except Exception as e:
            show_error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_video_frames(self):
        # Use preview if it exists, otherwise use original (don't auto-create)
        video_stem = self.video_path.stem.replace('_preview', '')
        if 'DLC_' in video_stem:
            video_stem = video_stem.split('DLC_')[0]
        preview_path = self.video_path.parent / f"{video_stem}_preview.mp4"

        if '_preview' not in self.video_path.stem and preview_path.exists():
            print(f"Using existing preview video: {preview_path.name}")
            actual_video = preview_path
            self.scale_factor = 0.75
        else:
            actual_video = self.video_path
            self.scale_factor = 1.0

        video_path_str = str(actual_video)

        # OpenCV VideoCapture
        cap = cv2.VideoCapture(video_path_str)

        if not cap.isOpened():
            # Try with forward slashes
            video_path_str = video_path_str.replace('\\', '/')
            cap = cv2.VideoCapture(video_path_str)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

        frames = []
        errors = 0
        for i in range(self.n_frames):
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if frames:
                        frames.append(frames[-1].copy())
                    continue
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except (cv2.error, MemoryError, SystemError) as e:
                errors += 1
                if frames:
                    frames.append(frames[-1].copy())
                if errors > 100:
                    cap.release()
                    raise RuntimeError(f"Too many frame read errors ({errors}). Error: {e}")
                continue

            if i % 500 == 0:
                self.progress.setValue(40 + int(50 * i / self.n_frames))
                from qtpy.QtWidgets import QApplication
                QApplication.processEvents()

        cap.release()
        
        if not frames:
            raise RuntimeError("No frames read from video")
        
        if errors > 0:
            show_warning(f"Loaded with {errors} frame read errors")
        
        self.n_frames = len(frames)

        # Remove old video layer if it exists and is still in the viewer
        if self.video_layer is not None and self.video_layer in self.viewer.layers:
            self.viewer.layers.remove(self.video_layer)

        self.video_layer = self.viewer.add_image(
            np.stack(frames),
            name=self.video_path.stem,
            rgb=True
        )
    
    def _add_points_layer(self):
        if self.dlc_df is None:
            return
        
        bodyparts = []
        for col in self.dlc_df.columns:
            if col.endswith('_x'):
                bodyparts.append(col[:-2])
        bodyparts = sorted(set(bodyparts))
        
        if not bodyparts:
            return
        
        colors_base = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5]
        ]
        bp_colors = {bp: colors_base[i % len(colors_base)] for i, bp in enumerate(bodyparts)}
        
        points_data = []
        point_colors = []
        
        for frame_idx in range(len(self.dlc_df)):
            for bp in bodyparts:
                x = self.dlc_df.iloc[frame_idx].get(f'{bp}_x', np.nan)
                y = self.dlc_df.iloc[frame_idx].get(f'{bp}_y', np.nan)
                likelihood = self.dlc_df.iloc[frame_idx].get(f'{bp}_likelihood', 0)
                
                if likelihood > 0.5 and not np.isnan(x) and not np.isnan(y):
                    # Scale coordinates to match downsampled video
                    scale = getattr(self, 'scale_factor', 1.0)
                    points_data.append([frame_idx, y * scale, x * scale])
                    point_colors.append(bp_colors[bp] + [0.15])
        
        if points_data:
            # Remove old points layer if it exists and is still in the viewer
            if self.points_layer is not None and self.points_layer in self.viewer.layers:
                self.viewer.layers.remove(self.points_layer)

            self.points_layer = self.viewer.add_points(
                np.array(points_data),
                name='DLC Points',
                size=6,
                face_color=np.array(point_colors),
            )

    def _add_pillar_shapes_layer(self):
        """Initialize empty pillar shapes layer."""
        if self.dlc_df is None:
            return

        # Remove old pillar shapes layer if it exists
        if self.pillar_shapes_layer is not None and self.pillar_shapes_layer in self.viewer.layers:
            self.viewer.layers.remove(self.pillar_shapes_layer)

        # Add empty shapes layer that we'll update per frame
        self.pillar_shapes_layer = self.viewer.add_shapes(
            name='Pillar Position (Geometric)',
            edge_width=2,
            edge_color='cyan',
            face_color='transparent',
            opacity=0.9
        )

        # Make the layer editable so user can move the circle
        self.pillar_shapes_layer.editable = True
        self.pillar_shapes_layer.mode = 'select'  # Start in select mode for moving

        # Add keyboard bindings for arrow key movement
        @self.pillar_shapes_layer.bind_key('Up')
        def move_up(layer):
            if len(layer.selected_data) > 0 and len(layer.data) > 0:
                layer.data[0][:, 0] -= 1  # Move up (decrease y)
                layer.refresh()

        @self.pillar_shapes_layer.bind_key('Down')
        def move_down(layer):
            if len(layer.selected_data) > 0 and len(layer.data) > 0:
                layer.data[0][:, 0] += 1  # Move down (increase y)
                layer.refresh()

        @self.pillar_shapes_layer.bind_key('Left')
        def move_left(layer):
            if len(layer.selected_data) > 0 and len(layer.data) > 0:
                layer.data[0][:, 1] -= 1  # Move left (decrease x)
                layer.refresh()

        @self.pillar_shapes_layer.bind_key('Right')
        def move_right(layer):
            if len(layer.selected_data) > 0 and len(layer.data) > 0:
                layer.data[0][:, 1] += 1  # Move right (increase x)
                layer.refresh()

        # Connect to frame change event to update pillar circle
        self.viewer.dims.events.current_step.connect(self._update_pillar_circle)

        # Draw initial circle
        self._update_pillar_circle()

    def _update_pillar_circle(self, event=None):
        """Update pillar circle for current frame."""
        if self.pillar_shapes_layer is None or self.dlc_df is None:
            return

        frame_idx = self.viewer.dims.current_step[0]

        if frame_idx >= len(self.dlc_df):
            return

        row = self.dlc_df.iloc[frame_idx]

        # Get scale factor (for preview video compatibility)
        scale = getattr(self, 'scale_factor', 1.0)

        # Get SA corners (scaled for preview video)
        sabl_x = row.get('SABL_x', np.nan) * scale
        sabl_y = row.get('SABL_y', np.nan) * scale
        sabr_x = row.get('SABR_x', np.nan) * scale
        sabr_y = row.get('SABR_y', np.nan) * scale

        # Clear existing shapes
        self.pillar_shapes_layer.data = []

        # Skip if SA corners not available
        if np.isnan([sabl_x, sabl_y, sabr_x, sabr_y]).any():
            return

        # Compute ruler length for this frame
        ruler = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)

        # SA midpoint
        mid_x = (sabl_x + sabr_x) / 2
        mid_y = (sabl_y + sabr_y) / 2

        # Geometric pillar position: 0.944 ruler units perpendicular from SA midpoint
        pillar_x = mid_x
        pillar_y = mid_y - (0.944 * ruler)

        # Pillar physical diameter: 4.125mm
        # Ruler (SABL-SABR) = 9mm = 1.0 ruler units (by definition)
        # Pillar radius = (4.125/9)/2 = 0.229 ruler units
        # Slider allows manual adjustment (default 10.0 = 1.0x, range 0.1-10.0 = 0.01x-1.0x)
        pillar_diameter_mm = 4.125
        ruler_mm = 9.0
        pillar_radius_ruler_units = (pillar_diameter_mm / ruler_mm) / 2.0  # = 0.229
        slider_multiplier = self.pillar_size_slider.value() / 100.0  # 1-100 -> 0.01-1.0
        pillar_radius = slider_multiplier * pillar_radius_ruler_units * ruler

        # Create circle as polygon with multiple points
        n_points = 32
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        circle_y = pillar_y + pillar_radius * np.sin(angles)
        circle_x = pillar_x + pillar_radius * np.cos(angles)
        circle_data = np.column_stack([circle_y, circle_x])

        # Set the data directly (this replaces all shapes)
        self.pillar_shapes_layer.data = [circle_data]

    def _on_pillar_size_change(self, value):
        """Update pillar circle size and redraw."""
        self.pillar_size_label.setText(f"{value / 10.0:.1f}")
        self._update_pillar_circle()

    def _toggle_pillar_visibility(self, state):
        """Toggle visibility of the pillar shapes layer."""
        if self.pillar_shapes_layer is not None and self.pillar_shapes_layer in self.viewer.layers:
            self.pillar_shapes_layer.visible = (state == Qt.Checked)

    def _update_file_status(self):
        """Check for related files and update status indicator."""
        if not self.video_path:
            return

        video_dir = self.video_path.parent
        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        status_parts = []

        # Check for ground truth file
        gt_path = video_dir / f"{video_id}_outcome_ground_truth.json"
        if gt_path.exists():
            status_parts.append('<span style="color: #4CAF50; font-weight: bold;">⬤ Ground Truth exists</span>')

        # Check for validation file
        val_path = video_dir / f"{video_id}_outcomes_validation.json"
        if val_path.exists():
            status_parts.append('<span style="color: #2196F3; font-weight: bold;">⬤ Validated</span>')

        if status_parts:
            self.file_status_label.setText(" | ".join(status_parts))
        else:
            self.file_status_label.setText('<span style="color: #999;">No ground truth or validation files found</span>')

    def _enable_controls(self, enabled: bool):
        self.play_btn.setEnabled(enabled)
        self.play_rev_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.prev_seg_btn.setEnabled(enabled)
        self.next_seg_btn.setEnabled(enabled)
        self.jump_start_btn.setEnabled(enabled)
        self.jump_end_btn.setEnabled(enabled)
        self.jump_interaction_btn.setEnabled(enabled)
        self.jump_known_btn.setEnabled(enabled)
        self.jump_reach_btn.setEnabled(enabled)
        self.set_interaction_btn.setEnabled(enabled)
        self.clear_interaction_btn.setEnabled(enabled)
        self.set_known_btn.setEnabled(enabled)
        self.clear_known_btn.setEnabled(enabled)
        self.accept_btn.setEnabled(enabled)
        self.confirm_untouched_btn.setEnabled(enabled)
        self.flag_check.setEnabled(enabled)
        self.flag_reason.setEnabled(enabled)
        self.save_progress_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.save_gt_btn.setEnabled(enabled)

        for radio in self.outcome_radios.values():
            radio.setEnabled(enabled)
    
    def _on_frame_change(self, event):
        frame_idx = self.viewer.dims.current_step[0]
        
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.frame_label.setText(f"Frame: {frame_idx} / {self.n_frames}  ({mins}:{secs:05.2f})")
    
    def _get_segment_data(self) -> Optional[Dict]:
        if not self.outcomes_data:
            return None
        for seg in self.outcomes_data.get('segments', []):
            if seg['segment_num'] == self.current_segment:
                return seg
        return None
    
    def _update_display(self):
        seg_data = self._get_segment_data()
        n_segments = len(self.boundaries) - 1
        
        if not seg_data:
            self.segment_label.setText(f"Segment: {self.current_segment} / {n_segments}")
            return
        
        flagged = seg_data.get('flagged_for_review', False)
        flag_text = " [FLAGGED]" if flagged else ""
        verified = " ✓" if seg_data.get('human_verified', False) else ""
        
        self.segment_label.setText(
            f"Segment: {self.current_segment} / {n_segments}{flag_text}{verified}"
        )
        
        # Current outcome
        outcome = seg_data.get('outcome', 'uncertain')
        color = OUTCOME_COLORS.get(outcome, '#000000')
        self.current_outcome_label.setText(f"Current: {outcome.upper()}")
        self.current_outcome_label.setStyleSheet(f"color: {color};")
        
        # Interaction frame
        interaction = seg_data.get('interaction_frame')
        self.interaction_label.setText(str(interaction) if interaction else "--")
        
        # Outcome known frame
        known = seg_data.get('outcome_known_frame')
        self.outcome_known_label.setText(str(known) if known else "--")
        
        # Radio buttons
        for o, radio in self.outcome_radios.items():
            radio.blockSignals(True)
            radio.setChecked(outcome == o)
            radio.blockSignals(False)
        
        # Algorithm info
        orig = seg_data.get('original_outcome')
        conf = seg_data.get('confidence', 0)
        if orig and orig != outcome:
            self.algo_info.setText(f"Algorithm said: {orig} (conf: {conf:.0%}) - CHANGED")
        else:
            self.algo_info.setText(f"Algorithm said: {outcome} (conf: {conf:.0%})")
        
        # Flag
        self.flag_check.blockSignals(True)
        self.flag_check.setChecked(flagged)
        self.flag_check.blockSignals(False)
        self.flag_reason.setText(seg_data.get('flag_reason', '') or '')
        
        # Update segment list
        self._update_segment_list()
        
        # Update summary
        self._update_summary()
    
    def _update_segment_list(self, *args):
        self.segment_list.clear()

        if not self.outcomes_data:
            self.review_progress_label.setText("Needs review: 0 | Done: 0/0")
            return

        segments = self.outcomes_data.get('segments', [])
        total = len(segments)

        # Count segments by category
        reviewed = sum(1 for seg in segments if seg.get('human_verified', False))
        needs_review = sum(1 for seg in segments
                          if not seg.get('human_verified', False)
                          and (seg.get('confidence', 0) < 0.85 or seg.get('flagged_for_review', False)))
        auto_accepted = total - reviewed - needs_review  # High confidence, not flagged, not reviewed

        # Update progress label
        if needs_review == 0 and reviewed == total:
            self.review_progress_label.setText(f"ALL DONE! ({total} segments)")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif needs_review == 0:
            self.review_progress_label.setText(f"No review needed | {auto_accepted} auto-accepted, {reviewed} manually reviewed")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self.review_progress_label.setText(f"Needs review: {needs_review} | Done: {reviewed}/{total}")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #FF9800;" if needs_review > 0 else "font-weight: bold; color: #2196F3;")

        # Filter segments: by default only show items needing review
        show_all = self.show_all_check.isChecked()

        def needs_attention(seg):
            """Returns True if segment needs human review."""
            if seg.get('human_verified', False):
                return False  # Already reviewed
            conf = seg.get('confidence', 0)
            flagged = seg.get('flagged_for_review', False)
            return conf < 0.85 or flagged  # Low confidence or flagged

        if show_all:
            filtered_segments = segments
        else:
            filtered_segments = [seg for seg in segments if needs_attention(seg)]

        # Sort: low confidence first, then by confidence descending
        def sort_key(seg):
            conf = seg.get('confidence', 0.5)
            flagged = seg.get('flagged_for_review', False)
            if flagged:
                return (0, 0)  # Flagged items first
            return (1, -conf)  # Then by confidence (low first)

        sorted_segments = sorted(filtered_segments, key=sort_key)

        from qtpy.QtGui import QColor

        for seg in sorted_segments:
            outcome = seg.get('outcome', '?')
            verified = seg.get('human_verified', False)
            flagged = seg.get('flagged_for_review', False)
            conf = seg.get('confidence', 0.5)

            # Status indicators
            status_icon = "✓" if verified else ("⚠" if flagged else "○")
            conf_str = f" ({conf:.0%})"

            item = QListWidgetItem(f"{status_icon} S{seg['segment_num']}: {outcome}{conf_str}")

            # Color code by confidence
            if verified:
                item.setForeground(QColor('#4CAF50'))  # Green for verified
            elif flagged:
                item.setForeground(QColor('#9C27B0'))  # Purple for flagged
            elif conf < 0.7:
                item.setForeground(QColor('#f44336'))  # Red for low confidence
            elif conf < 0.85:
                item.setForeground(QColor('#FF9800'))  # Orange for medium confidence
            else:
                item.setForeground(QColor('#8BC34A'))  # Light green for high confidence

            # Store segment number for lookup
            item.setData(Qt.UserRole, seg['segment_num'])
            self.segment_list.addItem(item)

        # Find and select current segment in list (if visible)
        for i in range(self.segment_list.count()):
            if self.segment_list.item(i).data(Qt.UserRole) == self.current_segment:
                self.segment_list.setCurrentRow(i)
                break

        # Update comparison panel
        self._update_comparison_panel()
    
    def _update_summary(self):
        if not self.outcomes_data:
            return
        
        summary = self.outcomes_data.get('summary', {})
        retrieved = summary.get('retrieved', 0)
        displaced_sa = summary.get('displaced_sa', 0)
        displaced_out = summary.get('displaced_outside', 0)
        untouched = summary.get('untouched', 0)
        
        self.summary_label.setText(
            f"R: {retrieved} | D(SA): {displaced_sa} | D(out): {displaced_out} | U: {untouched}"
        )

    def _on_comparison_item_selected(self, index: int):
        """Handle selection in comparison panel - jump to that segment."""
        if self.outcomes_data:
            segments = self.outcomes_data.get('segments', [])
            if 0 <= index < len(segments):
                self.current_segment = index + 1
                self._update_segment_display()
                self.segment_list.setCurrentRow(index)

    def _update_comparison_panel(self):
        """Update the comparison panel with current vs original outcomes."""
        if not hasattr(self, 'comparison_panel') or not self.outcomes_data:
            return

        # Get current outcomes
        current_outcomes = self.outcomes_data.get('segments', [])

        # Get original outcomes for comparison
        original_outcomes = []
        if hasattr(self, 'original_outcomes_json') and self.original_outcomes_json:
            orig_data = json.loads(self.original_outcomes_json)
            original_outcomes = orig_data.get('segments', [])

        if not original_outcomes:
            original_outcomes = current_outcomes

        # Check if GT file exists
        gt_exists = False
        if self.video_path:
            video_id = self.video_path.stem.split('DLC')[0].rstrip('_')
            gt_path = self.video_path.parent / f"{video_id}_outcomes_ground_truth.json"
            gt_exists = gt_path.exists()

        # Create comparison items
        items = create_outcome_comparison(original_outcomes, current_outcomes)
        self.comparison_panel.set_items(items, gt_exists=gt_exists)

    # === Playback ===
    
    def _set_speed(self, speed: float):
        """Set playback speed multiplier."""
        self.playback_speed = speed
        # Update button states
        for s, btn in self.speed_buttons.items():
            btn.setChecked(s == speed)
        # If playing, adjust timer
        if self.is_playing:
            self._update_playback_timer()
    
    def _update_playback_timer(self):
        """Update timer interval based on speed."""
        self.playback_timer.stop()
        if self.playback_speed >= 1:
            # For 1x and faster, use base interval and skip frames
            interval = max(1, int(1000 / self.fps))
        else:
            # For slow motion, increase interval
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
        self.play_btn.setText("▶ Play")
        self.play_rev_btn.setText("◀ Rev")
    
    def _start_playback(self):
        """Start playback in current direction."""
        if self.is_playing:
            self._stop_play()
            return
        
        self.is_playing = True
        self._update_playback_timer()
        
        if self.playback_direction == 1:
            self.play_btn.setText("⏸")
            self.play_rev_btn.setText("◀ Rev")
        else:
            self.play_rev_btn.setText("⏸")
            self.play_btn.setText("▶ Play")
    
    def _toggle_play(self):
        """Toggle forward playback (for Space key)."""
        if self.is_playing:
            self._stop_play()
        else:
            self.playback_direction = 1
            self._start_playback()
    
    def _playback_step(self):
        """Advance frames during playback."""
        current = self.viewer.dims.current_step[0]
        
        # Calculate frame skip based on speed
        if self.playback_speed >= 1:
            skip = int(self.playback_speed)
        else:
            skip = 1
        
        new_frame = current + (skip * self.playback_direction)
        
        if 0 <= new_frame < self.n_frames:
            self.viewer.dims.set_current_step(0, new_frame)
        else:
            # Hit beginning or end
            if new_frame < 0:
                self.viewer.dims.set_current_step(0, 0)
            else:
                self.viewer.dims.set_current_step(0, self.n_frames - 1)
            self._stop_play()
    
    def _jump_frames(self, delta: int):
        current = self.viewer.dims.current_step[0]
        new_frame = max(0, min(self.n_frames - 1, current + delta))
        self.viewer.dims.set_current_step(0, new_frame)
    
    # === Segment Navigation ===

    def goto_segment(self, seg_num: int):
        """Jump to a segment (public API for external callers)."""
        self._goto_segment(seg_num)

    def _goto_segment(self, seg_num: int):
        if seg_num < 1 or seg_num > len(self.boundaries) - 1:
            return

        self.current_segment = seg_num
        self.viewer.dims.set_current_step(0, self.boundaries[seg_num - 1])
        self._update_display()
    
    def _prev_segment(self):
        if self.current_segment > 1:
            self._goto_segment(self.current_segment - 1)
    
    def _next_segment(self):
        if self.current_segment < len(self.boundaries) - 1:
            self._goto_segment(self.current_segment + 1)
    
    def _jump_to_segment_start(self):
        if self.current_segment >= 1:
            self.viewer.dims.set_current_step(0, self.boundaries[self.current_segment - 1])
    
    def _jump_to_segment_end(self):
        if self.current_segment < len(self.boundaries):
            self.viewer.dims.set_current_step(0, self.boundaries[self.current_segment] - 1)
    
    def _select_segment(self, item):
        # Get segment number from item data (list is sorted by confidence)
        seg_num = item.data(Qt.UserRole)
        if seg_num:
            self.current_segment = seg_num
            self._update_display()

    def _jump_to_segment_from_list(self, item):
        # Get segment number from item data (list is sorted by confidence)
        seg_num = item.data(Qt.UserRole)
        if seg_num:
            self._goto_segment(seg_num)

    def _jump_to_interaction(self):
        """Jump to the detected interaction frame for this segment."""
        seg_data = self._get_segment_data()
        if not seg_data:
            return

        interaction_frame = seg_data.get('interaction_frame')
        if interaction_frame is not None:
            self.viewer.dims.set_current_step(0, interaction_frame)
            show_info(f"Jumped to interaction frame {interaction_frame}")
        else:
            show_warning("No interaction frame detected for this segment")

    def _jump_to_outcome_known(self):
        """Jump to the frame where outcome became determinable."""
        seg_data = self._get_segment_data()
        if not seg_data:
            return

        known_frame = seg_data.get('outcome_known_frame')
        if known_frame is not None:
            self.viewer.dims.set_current_step(0, known_frame)
            show_info(f"Jumped to outcome known frame {known_frame}")
        else:
            show_warning("No outcome known frame detected for this segment")

    def _jump_to_causal_reach(self):
        """Jump to the reach that caused this outcome."""
        seg_data = self._get_segment_data()
        if not seg_data:
            return

        reach_frame = seg_data.get('causal_reach_frame')
        if reach_frame is not None:
            self.viewer.dims.set_current_step(0, reach_frame)
            show_info(f"Jumped to causal reach frame {reach_frame}")
        else:
            show_warning("No causal reach detected for this segment")

    # === Annotation ===
    
    def _set_interaction_frame(self):
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        
        frame = self.viewer.dims.current_step[0]
        seg_data['interaction_frame'] = frame
        seg_data['human_verified'] = True
        
        self._increment_corrections()
        self._update_display()
        show_info(f"Interaction frame set to {frame}")
    
    def _clear_interaction_frame(self):
        seg_data = self._get_segment_data()
        if seg_data:
            seg_data['interaction_frame'] = None
            self._update_display()
    
    def _set_outcome_known_frame(self):
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        
        frame = self.viewer.dims.current_step[0]
        seg_data['outcome_known_frame'] = frame
        seg_data['human_verified'] = True
        
        self._increment_corrections()
        self._update_display()
        show_info(f"Outcome known frame set to {frame}")
    
    def _clear_outcome_known_frame(self):
        seg_data = self._get_segment_data()
        if seg_data:
            seg_data['outcome_known_frame'] = None
            self._update_display()
    
    def _on_outcome_selected(self, outcome: str, checked: bool):
        if checked:
            self._set_outcome(outcome)
    
    def _set_outcome(self, outcome: str):
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        
        old_outcome = seg_data.get('outcome')
        
        if old_outcome != outcome:
            if seg_data.get('original_outcome') is None:
                seg_data['original_outcome'] = old_outcome
            
            seg_data['outcome'] = outcome
            seg_data['human_verified'] = True
            seg_data['confidence'] = 1.0
            
            self._increment_corrections()
            self._recalculate_summary()
            
            show_info(f"Outcome set to {outcome}")
        
        self._update_display()
    
    def _confirm_untouched(self):
        """Confirm pellet was untouched at segment end."""
        # Jump to segment end first
        self._jump_to_segment_end()
        
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        
        seg_data['outcome'] = 'untouched'
        seg_data['interaction_frame'] = None
        seg_data['outcome_known_frame'] = self.boundaries[self.current_segment] - 1
        seg_data['human_verified'] = True
        seg_data['confidence'] = 1.0
        
        if seg_data.get('original_outcome') is None:
            seg_data['original_outcome'] = seg_data.get('outcome')
        
        self._increment_corrections()
        self._recalculate_summary()
        self._update_display()
        show_info("Confirmed untouched")

    def _accept_current(self):
        """Accept current segment outcome as-is (algorithm is correct).

        This provides explicit confirmation that the user reviewed this segment
        and agrees with the algorithm's classification, without making changes.
        """
        seg_data = self._get_segment_data()
        if not seg_data:
            return

        seg_data['human_verified'] = True
        seg_data['verified_by'] = get_username()
        seg_data['verified_at'] = datetime.now().isoformat()

        self._update_display()

        # Auto-advance to next unreviewed segment
        next_unreviewed = self._find_next_unreviewed()
        if next_unreviewed:
            show_info(f"Segment {self.current_segment} accepted. Moving to segment {next_unreviewed}...")
            self._goto_segment(next_unreviewed)
        else:
            show_info(f"Segment {self.current_segment} accepted. All segments reviewed!")

    def _find_next_unreviewed(self) -> Optional[int]:
        """Find the next segment that needs human review (low confidence or flagged)."""
        if not self.outcomes_data:
            return None

        def needs_review(seg):
            """Returns True if segment needs human attention."""
            if seg.get('human_verified', False):
                return False
            conf = seg.get('confidence', 0)
            flagged = seg.get('flagged_for_review', False)
            return conf < 0.85 or flagged

        # First look for segments needing review after current
        for seg in self.outcomes_data.get('segments', []):
            if seg['segment_num'] > self.current_segment and needs_review(seg):
                return seg['segment_num']

        # Then wrap around and check from beginning
        for seg in self.outcomes_data.get('segments', []):
            if seg['segment_num'] < self.current_segment and needs_review(seg):
                return seg['segment_num']

        return None  # All done - no segments need review

    def _increment_corrections(self):
        self.outcomes_data['corrections_made'] = self.outcomes_data.get('corrections_made', 0) + 1
    
    def _recalculate_summary(self):
        if not self.outcomes_data:
            return
        
        counts = {o: 0 for o in OUTCOMES}
        counts['no_pellet'] = 0
        counts['uncertain'] = 0
        
        for seg in self.outcomes_data['segments']:
            outcome = seg.get('outcome', 'uncertain')
            if outcome in counts:
                counts[outcome] += 1
        
        self.outcomes_data['summary'] = {
            'total_segments': len(self.outcomes_data['segments']),
            'retrieved': counts['retrieved'],
            'displaced_sa': counts['displaced_sa'],
            'displaced_outside': counts['displaced_outside'],
            'untouched': counts['untouched'],
            'no_pellet': counts['no_pellet'],
            'uncertain': counts['uncertain'],
            'flagged': sum(1 for s in self.outcomes_data['segments'] if s.get('flagged_for_review'))
        }
    
    # === Flagging ===
    
    def _toggle_flag(self, state):
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        
        seg_data['flagged_for_review'] = bool(state)
        
        flagged_count = sum(1 for s in self.outcomes_data['segments'] if s.get('flagged_for_review'))
        self.outcomes_data['segments_flagged'] = flagged_count
        
        self._update_display()
    
    def _update_flag_reason(self, text):
        seg_data = self._get_segment_data()
        if seg_data:
            seg_data['flag_reason'] = text
    
    # === Save ===
    
    def _save_progress(self):
        """Save work in progress - can resume later."""
        if not self.outcomes_path or not self.outcomes_data:
            return
        
        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]
        
        # Track progress
        current_frame = self.viewer.dims.current_step[0]
        self.outcomes_data['work_in_progress'] = True
        self.outcomes_data['last_edited_by'] = get_username()
        self.outcomes_data['last_edited_at'] = datetime.now().isoformat()
        self.outcomes_data['last_segment'] = self.current_segment
        self.outcomes_data['last_frame'] = int(current_frame)
        
        self._recalculate_summary()
        
        with open(self.outcomes_path, 'w') as f:
            json.dump(self.outcomes_data, f, indent=2)
        
        self.status_label.setText(f"Progress saved @ seg {self.current_segment}")
        show_info(f"Progress saved - segment {self.current_segment}, frame {current_frame}")
    
    def _save_validated(self):
        """Save as validated and mark complete.

        New architecture (v2.3+): Files stay in Processing/, validation status
        is set in JSON metadata. No folder-based triage.
        """
        if not self.outcomes_path or not self.outcomes_data:
            return

        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        # Count changes from original
        original_data = json.loads(self.original_outcomes_json) if self.original_outcomes_json else {}
        original_summary = original_data.get('summary', {})

        # Calculate detailed changes
        changed_outcomes = []
        original_segments = {s['segment_num']: s for s in original_data.get('segments', [])}
        for seg in self.outcomes_data.get('segments', []):
            orig_seg = original_segments.get(seg['segment_num'], {})
            if seg.get('outcome') != orig_seg.get('outcome'):
                changed_outcomes.append({
                    'segment': seg['segment_num'],
                    'original': orig_seg.get('outcome'),
                    'corrected': seg.get('outcome')
                })

        # Build validation_record (embedded audit trail)
        validation_record = {
            'validated_by': get_username(),
            'validated_at': datetime.now().isoformat(),
            'algorithm_version': self.outcomes_data.get('detector_version'),
            'original_summary': original_summary,
            'outcome_changes': changed_outcomes,
            'corrections_made': self.outcomes_data.get('corrections_made', 0),
            'segments_flagged': self.outcomes_data.get('segments_flagged', 0),
        }

        self._recalculate_summary()

        # Mark validated
        self.outcomes_data['validation_status'] = 'validated'  # KEY: marks as human-reviewed
        self.outcomes_data['validation_record'] = validation_record  # Embedded audit trail
        self.outcomes_data['validated'] = True  # Legacy compatibility
        self.outcomes_data['validated_by'] = get_username()
        self.outcomes_data['validated_at'] = datetime.now().isoformat()
        self.outcomes_data['work_in_progress'] = False  # Clear WIP flag

        with open(self.outcomes_path, 'w') as f:
            json.dump(self.outcomes_data, f, indent=2)

        # Update pipeline index
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_validation_changed(video_id, "outcome", "validated")
            index.save()
        except Exception:
            pass  # Silent fail - index will catch up on next rebuild

        # Log performance metrics (algorithm vs human comparison)
        try:
            from mousereach.performance import PerformanceLogger
            original_data = json.loads(self.original_outcomes_json) if self.original_outcomes_json else {}
            PerformanceLogger().log_outcome(
                video_id=video_id,
                algo_data=original_data,
                human_data=self.outcomes_data,
                algo_version=self.outcomes_data.get('detector_version'),
                validator=get_username()
            )
        except Exception:
            pass  # Don't block validation if logging fails

        changes = len(changed_outcomes)
        changes_text = f"{changes} outcomes changed" if changes else "no changes"
        self.status_label.setText(f"Validated: {video_id} ({changes_text})")
        show_info(f"Saved validated outcomes ({changes_text})")

    def _save_ground_truth(self):
        """Save as ground truth for algorithm development.

        GT files only contain items the human explicitly interacted with.
        Algorithm-seeded items the human never touched are NOT saved.
        This makes GT files true training data - only human-provided examples.

        A segment is included in GT if:
        - human_verified == True (user explicitly verified/set the outcome), OR
        - human_corrected == True (user changed the outcome from algorithm's prediction)
        """
        if not self.outcomes_data:
            return

        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        output_path = self.video_path.parent / f"{video_id}_outcome_ground_truth.json"

        self._recalculate_summary()

        # Build GT segments - ONLY include human-interacted segments
        gt_segments = []
        total_segments_in_video = len(self.outcomes_data['segments'])

        for seg in self.outcomes_data['segments']:
            is_verified = seg.get('human_verified', False)
            is_corrected = seg.get('human_corrected', False)

            # Only include in GT if human interacted with this segment
            if is_verified or is_corrected:
                gt_seg = {
                    'segment_num': seg.get('segment_num'),
                    'start_frame': seg.get('start_frame'),
                    'end_frame': seg.get('end_frame'),
                    'outcome': seg.get('outcome'),
                    'confidence': 1.0,  # Human-verified = full confidence
                    'interaction_frame': seg.get('interaction_frame'),
                    'outcome_known_frame': seg.get('outcome_known_frame'),
                    'human_verified': True,
                    'human_corrected': is_corrected,
                    'verified_by': get_username(),
                    'verified_at': datetime.now().isoformat(),
                    'flagged_for_review': seg.get('flagged_for_review', False),
                    'flag_reason': seg.get('flag_reason'),
                    # Preserve original if corrected
                    'original_outcome': seg.get('original_outcome') if is_corrected else None,
                }
                gt_segments.append(gt_seg)

        # Warn if no human-interacted items
        if not gt_segments:
            show_warning("No outcomes were verified or corrected. GT file will be empty.")

        gt_data = {
            'video_name': video_id,
            'type': 'ground_truth',
            'created_by': get_username(),
            'created_at': datetime.now().isoformat(),
            'n_segments': len(gt_segments),
            'segments': gt_segments,
            # Completeness tracking
            'gt_complete': len(gt_segments) > 0,
            'total_segments_in_video': total_segments_in_video,
            'segments_annotated': len(gt_segments),
        }

        with open(output_path, 'w') as f:
            json.dump(gt_data, f, indent=2)

        # Update pipeline index with GT status
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_gt_created(video_id, "outcome", is_complete=(len(gt_segments) > 0))
            index.save()
        except Exception:
            pass  # Silent fail - index will catch up on next rebuild

        status_text = f"{len(gt_segments)}/{total_segments_in_video} outcomes"
        self.status_label.setText(f"Saved GT: {output_path.name} ({status_text})")
        show_info(f"Saved ground truth: {status_text}")


def main():
    """Launch the annotator."""
    viewer = napari.Viewer(title="MouseReach Pellet Outcome Annotator")
    widget = PelletOutcomeAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Pellet Outcomes", area="right")
    
    print("\nKeyboard shortcuts:")
    print("  Space       - Play forward")
    print("  Shift+R     - Play reverse")
    print("  Left/Right  - Step 1 frame")
    print("  Shift+L/R   - Step 10 frames")
    print("  N/P         - Next/prev segment")
    print("  I           - Set interaction frame")
    print("  K           - Set outcome known frame")
    print("  R           - Set Retrieved")
    print("  D           - Set Displaced (SA)")
    print("  O           - Set Displaced (outside)")
    print("  U           - Set Untouched")
    print("  Ctrl+S      - Save progress")
    
    napari.run()


if __name__ == "__main__":
    main()
