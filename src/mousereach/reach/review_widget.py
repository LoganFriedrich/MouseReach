"""
MouseReach Reach Annotator
=====================

Napari widget for reviewing and correcting reach detection results.

Workflow:
1. Load video (auto-loads DLC, segments, algorithm reaches)
2. Watch video at normal speed or step through frames
3. Verify/correct reach start and end frames
4. Add missed reaches
5. Delete false positives
6. Flag messy segments for review
7. Save validated or ground truth

Install as plugin:
    pip install -e .
    # Then: Plugins → MouseReach Reach Detection → Reach Annotator

Or run standalone:
    python -m mousereach_reach_detection._napari_widget
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import json
import os
from datetime import datetime
from dataclasses import asdict

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QSpinBox, QListWidgetItem, QCheckBox, QLineEdit, QSlider, QScrollArea
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont, QColor

import napari
from napari.utils.notifications import show_info, show_warning, show_error
import pandas as pd
import cv2

from mousereach.review import ComparisonPanel, create_reach_comparison
from mousereach.review.save_panel import SimpleSavePanel


def get_username():
    """Get current username."""
    try:
        return os.getlogin()
    except (OSError, AttributeError):
        return os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))


class ReachAnnotatorWidget(QWidget):
    """
    Widget for annotating/correcting reach detection results.
    """

    def __init__(self, napari_viewer: napari.Viewer, embedded_mode: bool = False):
        super().__init__()
        self.viewer = napari_viewer
        self.embedded_mode = embedded_mode  # When True, hides video load and nav controls

        # Data
        self.video_path = None
        self.dlc_path = None
        self.segments_path = None
        self.reaches_path = None
        self.outcomes_path = None  # Pellet outcomes file path (for saving causal reach changes)

        self.dlc_df = None
        self.boundaries = []
        self.reaches_data = None
        self.outcomes_data = None  # Pellet outcome data for causal reach display
        self.outcomes_modified = False  # Track if causal reach was changed
        self.original_reaches_json = None  # For change tracking

        # State
        self.n_frames = 0
        self.fps = 60.0
        self.current_segment = 1
        self.current_reach_idx = 0
        self.is_playing = False
        self.playback_speed = 1  # 1x, 2x, 4x, 8x, 16x
        self.playback_direction = 1  # 1 = forward, -1 = backward
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        # Annotation state
        self.annotating_new_reach = False
        self.new_reach_start = None

        # Layers
        self.video_layer = None
        self.points_layer = None

        # Interesting frames (nose near box opening)
        self.interesting_frames = []

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
                "Space=play, R=reverse, 1-5=speed (1x/2x/4x/8x/16x)\n"
                "J/K=jump to RH visible, N/P=next/prev reach\n"
                "S/E=set start/end, A=add reach, Del=delete"
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

            self.progress = QProgressBar()
            self.progress.setVisible(False)
            file_layout.addWidget(self.progress)

            layout.addWidget(file_group)
        else:
            # Create dummy attributes for embedded mode
            self.video_label = QLabel()
            self.file_status_label = QLabel()
            self.load_btn = QPushButton()
            self.progress = QProgressBar()

        # === Navigation & Playback (hidden in embedded mode) ===
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

            # Playback controls
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

            play_layout.addWidget(QLabel("Speed:"))
            self.speed_buttons = {}
            for speed in [1, 2, 4, 8, 16]:
                btn = QPushButton(f"{speed}x")
                btn.setCheckable(True)
                btn.setMaximumWidth(40)
                btn.clicked.connect(lambda checked, s=speed: self._set_speed(s))
                self.speed_buttons[speed] = btn
                play_layout.addWidget(btn)
            self.speed_buttons[1].setChecked(True)
            nav_layout.addLayout(play_layout)

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
            self.prev_seg_btn = QPushButton("<< Prev Segment")
            self.prev_seg_btn.clicked.connect(self._prev_segment)
            self.prev_seg_btn.setEnabled(False)
            seg_layout.addWidget(self.prev_seg_btn)

            self.next_seg_btn = QPushButton("Next Segment >>")
            self.next_seg_btn.clicked.connect(self._next_segment)
            self.next_seg_btn.setEnabled(False)
            seg_layout.addWidget(self.next_seg_btn)
            nav_layout.addLayout(seg_layout)

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
        
        # === Reach List & Navigation ===
        reach_title = "Reaches in Segment" if self.embedded_mode else "3. Reaches in Segment"
        reach_group = QGroupBox(reach_title)
        reach_layout = QVBoxLayout()
        reach_group.setLayout(reach_layout)
        
        # Reach navigation
        reach_nav = QHBoxLayout()
        self.prev_reach_btn = QPushButton("< Prev Reach (P)")
        self.prev_reach_btn.clicked.connect(self._prev_reach)
        self.prev_reach_btn.setEnabled(False)
        reach_nav.addWidget(self.prev_reach_btn)
        
        self.next_reach_btn = QPushButton("Next Reach (N) >")
        self.next_reach_btn.clicked.connect(self._next_reach)
        self.next_reach_btn.setEnabled(False)
        reach_nav.addWidget(self.next_reach_btn)
        reach_layout.addLayout(reach_nav)
        
        # Current reach info
        self.reach_info = QLabel("No reach selected")
        self.reach_info.setWordWrap(True)
        reach_layout.addWidget(self.reach_info)
        
        # Jump to reach frames
        jump_layout = QHBoxLayout()
        self.jump_start_btn = QPushButton("→ Start")
        self.jump_start_btn.clicked.connect(lambda: self._jump_to_reach_frame('start'))
        self.jump_start_btn.setEnabled(False)
        jump_layout.addWidget(self.jump_start_btn)
        
        self.jump_end_btn = QPushButton("→ End")
        self.jump_end_btn.clicked.connect(lambda: self._jump_to_reach_frame('end'))
        self.jump_end_btn.setEnabled(False)
        jump_layout.addWidget(self.jump_end_btn)
        reach_layout.addLayout(jump_layout)

        # Causal reach control (shown when outcome data is loaded)
        causal_layout = QHBoxLayout()
        self.mark_causal_btn = QPushButton("Mark as Causal Reach")
        self.mark_causal_btn.setToolTip(
            "Mark the selected reach as the one that caused the pellet outcome"
        )
        self.mark_causal_btn.clicked.connect(self._mark_as_causal_reach)
        self.mark_causal_btn.setEnabled(False)
        self.mark_causal_btn.setVisible(False)
        causal_layout.addWidget(self.mark_causal_btn)
        reach_layout.addLayout(causal_layout)

        # Review progress indicator
        self.review_progress_label = QLabel("Reviewed: 0/0 reaches")
        self.review_progress_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        reach_layout.addWidget(self.review_progress_label)

        # Reach list
        self.reach_list = QListWidget()
        self.reach_list.itemClicked.connect(self._select_reach)
        self.reach_list.itemDoubleClicked.connect(self._jump_to_reach)
        self.reach_list.setMaximumHeight(100)
        reach_layout.addWidget(self.reach_list)

        # Outcome legend (shown when outcome data is loaded)
        self.outcome_legend = QLabel(
            "Outcome: [R]=Retrieved [D]=Disp.SA [O]=Disp.Out [U]=Untouched"
        )
        self.outcome_legend.setStyleSheet("font-size: 9px; color: #888888;")
        self.outcome_legend.setVisible(False)
        reach_layout.addWidget(self.outcome_legend)

        layout.addWidget(reach_group)
        
        # === Edit Reaches ===
        edit_title = "Edit Reaches" if self.embedded_mode else "4. Edit Reaches"
        edit_group = QGroupBox(edit_title)
        edit_layout = QVBoxLayout()
        edit_group.setLayout(edit_layout)
        
        # Correct current reach
        correct_layout = QHBoxLayout()
        self.set_start_btn = QPushButton("Set START here (S)")
        self.set_start_btn.clicked.connect(self._set_reach_start)
        self.set_start_btn.setEnabled(False)
        correct_layout.addWidget(self.set_start_btn)
        
        self.set_end_btn = QPushButton("Set END here (E)")
        self.set_end_btn.clicked.connect(self._set_reach_end)
        self.set_end_btn.setEnabled(False)
        correct_layout.addWidget(self.set_end_btn)
        edit_layout.addLayout(correct_layout)
        
        # Add new reach
        add_layout = QHBoxLayout()
        self.add_reach_btn = QPushButton("Add NEW reach starting here (A)")
        self.add_reach_btn.clicked.connect(self._start_add_reach)
        self.add_reach_btn.setEnabled(False)
        add_layout.addWidget(self.add_reach_btn)
        edit_layout.addLayout(add_layout)
        
        self.add_status = QLabel("")
        edit_layout.addWidget(self.add_status)
        
        self.finish_add_btn = QPushButton("Finish reach at current frame")
        self.finish_add_btn.clicked.connect(self._finish_add_reach)
        self.finish_add_btn.setVisible(False)
        edit_layout.addWidget(self.finish_add_btn)
        
        self.cancel_add_btn = QPushButton("Cancel adding reach")
        self.cancel_add_btn.clicked.connect(self._cancel_add_reach)
        self.cancel_add_btn.setVisible(False)
        edit_layout.addWidget(self.cancel_add_btn)
        
        # Delete reach
        self.delete_btn = QPushButton("Delete selected reach (DEL)")
        self.delete_btn.clicked.connect(self._delete_reach)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("background-color: #f44336; color: white;")
        edit_layout.addWidget(self.delete_btn)

        # Exclude reach from analysis
        exclude_layout = QHBoxLayout()
        self.exclude_check = QCheckBox("Exclude from analysis (X)")
        self.exclude_check.stateChanged.connect(self._toggle_exclude_reach)
        self.exclude_check.setEnabled(False)
        self.exclude_check.setToolTip("Mark this reach as unreliable - will be excluded from exports")
        exclude_layout.addWidget(self.exclude_check)
        edit_layout.addLayout(exclude_layout)

        exclude_reason_layout = QHBoxLayout()
        exclude_reason_layout.addWidget(QLabel("Reason:"))
        self.exclude_reason = QLineEdit()
        self.exclude_reason.setPlaceholderText("e.g., tracking lost, ambiguous, incomplete")
        self.exclude_reason.setEnabled(False)
        self.exclude_reason.textChanged.connect(self._update_exclude_reason)
        exclude_reason_layout.addWidget(self.exclude_reason)
        edit_layout.addLayout(exclude_reason_layout)

        # Accept reach as-is (explicit confirmation)
        self.accept_reach_btn = QPushButton("Accept reach as-is (Enter) - algorithm is correct")
        self.accept_reach_btn.clicked.connect(self._accept_current_reach)
        self.accept_reach_btn.setEnabled(False)
        self.accept_reach_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.accept_reach_btn.setToolTip("Mark this reach as reviewed - algorithm result is correct")
        edit_layout.addWidget(self.accept_reach_btn)

        layout.addWidget(edit_group)

        # === Flag Segment ===
        flag_title = "Flag Segment" if self.embedded_mode else "5. Flag Segment"
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
        self.save_gt_btn = self.save_panel.save_gt_btn  # May be None in review_mode
        self.save_progress_btn = self.save_panel.save_progress_btn

        layout.addWidget(save_group)

        # Add stretch at bottom
        layout.addStretch()
    
    def _setup_keybindings(self):
        """Set up keyboard shortcuts."""
        @self.viewer.bind_key('Space', overwrite=True)
        def toggle_play(viewer):
            self._toggle_play()
        
        @self.viewer.bind_key('r', overwrite=True)
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
        def next_reach(viewer):
            self._next_reach()
        
        @self.viewer.bind_key('p', overwrite=True)
        def prev_reach(viewer):
            self._prev_reach()
        
        @self.viewer.bind_key('s', overwrite=True)
        def set_start(viewer):
            self._set_reach_start()
        
        @self.viewer.bind_key('e', overwrite=True)
        def set_end(viewer):
            self._set_reach_end()
        
        @self.viewer.bind_key('a', overwrite=True)
        def add_reach(viewer):
            if self.annotating_new_reach:
                self._finish_add_reach()
            else:
                self._start_add_reach()
        
        @self.viewer.bind_key('Delete', overwrite=True)
        def delete_reach(viewer):
            self._delete_reach()
        
        @self.viewer.bind_key('Escape', overwrite=True)
        def cancel(viewer):
            self._cancel_add_reach()
        
        @self.viewer.bind_key('Control-s', overwrite=True)
        def save(viewer):
            self._save_progress()

        @self.viewer.bind_key('x', overwrite=True)
        def exclude_reach(viewer):
            self._toggle_exclude_reach_key()

        @self.viewer.bind_key('Return', overwrite=True)
        def accept_reach(viewer):
            self._accept_current_reach()

        # Jump to interesting frames (RH visible)
        @self.viewer.bind_key('j', overwrite=True)
        def next_interesting(viewer):
            self._jump_to_interesting(forward=True)
        
        @self.viewer.bind_key('k', overwrite=True)
        def prev_interesting(viewer):
            self._jump_to_interesting(forward=False)
        
        # Playback speed shortcuts (1-5 keys)
        @self.viewer.bind_key('1', overwrite=True)
        def speed_1x(viewer):
            self._set_speed(1)
        
        @self.viewer.bind_key('2', overwrite=True)
        def speed_2x(viewer):
            self._set_speed(2)
        
        @self.viewer.bind_key('3', overwrite=True)
        def speed_4x(viewer):
            self._set_speed(4)
        
        @self.viewer.bind_key('4', overwrite=True)
        def speed_8x(viewer):
            self._set_speed(8)
        
        @self.viewer.bind_key('5', overwrite=True)
        def speed_16x(viewer):
            self._set_speed(16)
    
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

        # Get video_id
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
                show_warning("No DLC file found - loading without tracking overlay")
            
            self.progress.setValue(20)
            
            # Find segments file
            seg_patterns = [
                f"{video_id}_seg_validation.json",
                f"{video_id}_segments_v2.json",
                f"{video_id}_segments.json",
                f"{video_id}_seg_ground_truth.json"
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

            # Find reaches file (ground truth takes priority)
            reach_patterns = [
                f"{video_id}_reach_ground_truth.json",  # Ground truth first
                f"{video_id}_reaches.json",
            ]
            self.reaches_path = None
            self._using_reach_ground_truth = False
            for pattern in reach_patterns:
                candidate = video_dir / pattern
                if candidate.exists():
                    self.reaches_path = candidate
                    if 'ground_truth' in pattern:
                        self._using_reach_ground_truth = True
                    break

            if self.reaches_path:
                self._load_reaches()
            else:
                show_warning(
                    f"No reaches file found for {video_id}.\n"
                    "Run the pipeline first to generate reach detection results."
                )
                return

            # Load pellet outcomes (optional - for causal reach display)
            self._load_outcomes(video_dir, video_id)

            self.progress.setValue(40)
            
            # Load video frames
            self._load_video_frames()
            
            # Add DLC overlay
            if self.dlc_df is not None:
                self._add_points_layer()
            
            self.progress.setValue(95)
            
            # Enable controls
            self._enable_controls(True)
            
            # Connect frame change
            self.viewer.dims.events.current_step.connect(self._on_frame_change)
            
            # Update display
            self.video_label.setText(
                f"{self.video_path.name}\n"
                f"Segments: {len(self.boundaries)-1} | "
                f"Total reaches: {self._count_total_reaches()}"
            )

            # Check for related files
            self._update_file_status()
            
            self.progress.setValue(100)
            
            # Check for work in progress
            if self.reaches_data and self.reaches_data.get('work_in_progress'):
                last_seg = self.reaches_data.get('last_segment', 1)
                last_frame = self.reaches_data.get('last_frame', 0)
                last_user = self.reaches_data.get('last_edited_by', 'unknown')
                last_time = self.reaches_data.get('last_edited_at', '')[:10]  # Just date
                
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
            
        except Exception as e:
            show_error(f"Error loading: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.progress.setVisible(False)

    def _load_data_only(self, video_path: Path):
        """
        Load just the data files (DLC, segments, reaches) without loading the video.

        Used when the video is already loaded via shared state manager.
        The shared video layer and frame data should already be set on self:
            self._shared_video_layer, self._shared_video_frames,
            self._shared_n_frames, self._shared_video_fps
        """
        self.video_path = video_path
        video_dir = self.video_path.parent
        video_stem = self.video_path.stem

        # Get video_id
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
                show_warning("No DLC file found - loading without tracking overlay")

            # Find segments file
            seg_patterns = [
                f"{video_id}_seg_validation.json",
                f"{video_id}_segments_v2.json",
                f"{video_id}_segments.json",
                f"{video_id}_seg_ground_truth.json"
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

            # Find reaches file (ground truth takes priority)
            reach_patterns = [
                f"{video_id}_reach_ground_truth.json",  # Ground truth first
                f"{video_id}_reaches.json",
            ]
            self.reaches_path = None
            self._using_reach_ground_truth = False
            for pattern in reach_patterns:
                candidate = video_dir / pattern
                if candidate.exists():
                    self.reaches_path = candidate
                    if 'ground_truth' in pattern:
                        self._using_reach_ground_truth = True
                    break

            if self.reaches_path:
                self._load_reaches()
            else:
                show_warning(
                    f"No reaches file found for {video_id}.\n"
                    "Run the pipeline first to generate reach detection results."
                )
                return

            # Load pellet outcomes (optional - for causal reach display)
            self._load_outcomes(video_dir, video_id)

            # Add DLC overlay
            if self.dlc_df is not None:
                self._add_points_layer()

            # Enable controls
            self._enable_controls(True)

            # Connect frame change
            self.viewer.dims.events.current_step.connect(self._on_frame_change)

            # Update display
            self.video_label.setText(
                f"{self.video_path.name}\n"
                f"Segments: {len(self.boundaries)-1} | "
                f"Total reaches: {self._count_total_reaches()}"
            )

            # Check for related files
            self._update_file_status()

            # Check for work in progress
            if self.reaches_data and self.reaches_data.get('work_in_progress'):
                last_seg = self.reaches_data.get('last_segment', 1)
                last_frame = self.reaches_data.get('last_frame', 0)
                last_user = self.reaches_data.get('last_edited_by', 'unknown')
                last_time = self.reaches_data.get('last_edited_at', '')[:10]

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
        """Load DLC tracking data."""
        df = pd.read_hdf(self.dlc_path)
        df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
        self.dlc_df = df
        
        # Compute interesting frames (nose near/past box opening)
        self._compute_interesting_frames()
    
    def _compute_interesting_frames(self):
        """
        Find frames where any right hand point is visible.
        These are the only frames where reaching is happening.
        
        Reduces review from ~54k frames to ~200-5000 frames.
        """
        self.interesting_frames = []
        
        if self.dlc_df is None:
            return
        
        # Right hand points
        rh_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
        
        for frame_idx in range(len(self.dlc_df)):
            for point in rh_points:
                col = f"{point}_likelihood"
                if col in self.dlc_df.columns:
                    if self.dlc_df.iloc[frame_idx][col] > 0.5:
                        self.interesting_frames.append(frame_idx)
                        break  # Found one visible point, move to next frame
        
        pct = len(self.interesting_frames) / len(self.dlc_df) * 100 if len(self.dlc_df) > 0 else 0
        print(f"Found {len(self.interesting_frames)} interesting frames ({pct:.1f}% - right hand visible)")
    
    def _jump_to_interesting(self, forward: bool = True):
        """Jump to next/previous interesting frame."""
        if not self.interesting_frames:
            show_warning("No interesting frames computed (need DLC data with Nose, BOXL, BOXR)")
            return
        
        current_frame = self.viewer.dims.current_step[0]
        
        if forward:
            # Find next interesting frame after current
            for frame in self.interesting_frames:
                if frame > current_frame:
                    self.viewer.dims.set_current_step(0, frame)
                    self._update_status()
                    return
            # Wrap to beginning
            self.viewer.dims.set_current_step(0, self.interesting_frames[0])
        else:
            # Find previous interesting frame before current
            for frame in reversed(self.interesting_frames):
                if frame < current_frame:
                    self.viewer.dims.set_current_step(0, frame)
                    self._update_status()
                    return
            # Wrap to end
            self.viewer.dims.set_current_step(0, self.interesting_frames[-1])
        
        self._update_status()
    
    def _load_segments(self):
        """Load segment boundaries."""
        with open(self.segments_path) as f:
            data = json.load(f)
        self.boundaries = data.get('boundaries', data.get('validated_boundaries', []))
    
    def _load_reaches(self):
        """Load reaches data."""
        with open(self.reaches_path) as f:
            self.reaches_data = json.load(f)
        # Store original for change tracking
        self.original_reaches_json = json.dumps(self.reaches_data)

    def _load_outcomes(self, video_dir: Path, video_id: str):
        """Load pellet outcomes data (optional - for causal reach display).

        Searches for the pellet outcomes JSON in the same directory as the video.
        If found, enables per-reach outcome tags and causal reach highlighting.
        If not found, the widget works normally without outcome display.
        """
        self.outcomes_path = None
        outcome_patterns = [
            f"{video_id}_pellet_outcomes.json",
        ]
        for pattern in outcome_patterns:
            candidate = video_dir / pattern
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        self.outcomes_data = json.load(f)
                    self.outcomes_path = candidate
                    self.outcome_legend.setVisible(True)
                    self.mark_causal_btn.setVisible(True)
                    self.mark_causal_btn.setEnabled(True)
                except Exception as e:
                    print(f"Warning: Could not load outcomes file: {e}")
                    self.outcomes_data = None
                return
        # No outcome file found - that's fine, widget works without it
        self.outcomes_data = None
        self.outcome_legend.setVisible(False)
        self.mark_causal_btn.setVisible(False)
        self.mark_causal_btn.setEnabled(False)

    def _get_outcome_for_segment(self, segment_num: int) -> Optional[Dict]:
        """Get the pellet outcome data for a given segment number.

        Returns the outcome dict from the outcomes JSON, or None if no
        outcome data is loaded or the segment is not found.
        """
        if not self.outcomes_data:
            return None
        for seg in self.outcomes_data.get('segments', []):
            if seg.get('segment_num') == segment_num:
                return seg
        return None

    def _get_reach_outcome_tag(self, reach: Dict) -> Optional[str]:
        """Get an outcome tag string if this reach is the causal reach.

        Returns a tag like '[R]', '[D]', '[O]', '[U]' if this reach is the
        causal reach for the current segment's pellet outcome.
        Returns None if this reach is not the causal reach, or if no
        outcome data is available.
        """
        outcome_seg = self._get_outcome_for_segment(self.current_segment)
        if not outcome_seg:
            return None

        causal_id = outcome_seg.get('causal_reach_id')
        if causal_id is None:
            return None

        # Match by reach_id if available, otherwise by reach_num
        reach_id = reach.get('reach_id', reach.get('reach_num'))
        if reach_id != causal_id:
            return None

        # This is the causal reach - return outcome tag
        outcome = outcome_seg.get('outcome', '')
        tag_map = {
            'retrieved': '[R]',
            'displaced_sa': '[D]',
            'displaced_outside': '[O]',
            'untouched': '[U]',
            'no_pellet': '[NP]',
            'uncertain': '[?]',
        }
        return tag_map.get(outcome, f'[{outcome[:1].upper()}]')

    def _mark_as_causal_reach(self):
        """Mark the currently selected reach as the causal reach for this segment.

        Updates the outcomes data to point to this reach as the one that caused
        the pellet outcome. The change is saved immediately to the outcomes file.
        """
        if not self.outcomes_data:
            show_warning("No outcomes data loaded")
            return

        reaches = self._get_segment_reaches()
        if not reaches or self.current_reach_idx >= len(reaches):
            show_warning("No reach selected")
            return

        reach = reaches[self.current_reach_idx]
        reach_id = reach.get('reach_id', reach.get('reach_num'))

        # Find and update the outcome for this segment
        outcome_seg = self._get_outcome_for_segment(self.current_segment)
        if not outcome_seg:
            show_warning(f"No outcome data for segment {self.current_segment}")
            return

        old_causal = outcome_seg.get('causal_reach_id')
        if old_causal == reach_id:
            show_info(f"Reach #{reach_id} is already marked as causal")
            return

        # Update the outcome data
        outcome_seg['causal_reach_id'] = reach_id
        outcome_seg['causal_reach_frame'] = reach.get('apex_frame', reach.get('start_frame'))
        outcome_seg['human_verified'] = True
        self.outcomes_modified = True

        # Save immediately
        self._save_outcomes()

        # Refresh display
        self._update_reach_list()
        self._update_reach_info()

        show_info(f"Reach #{reach_id} marked as causal for segment {self.current_segment}")

    def _save_outcomes(self):
        """Save the outcomes data back to the JSON file."""
        if not self.outcomes_path or not self.outcomes_data:
            return

        try:
            # Update metadata
            self.outcomes_data['last_edited_by'] = get_username()
            self.outcomes_data['last_edited_at'] = datetime.now().isoformat()

            with open(self.outcomes_path, 'w') as f:
                json.dump(self.outcomes_data, f, indent=2)

            self.outcomes_modified = False
        except Exception as e:
            show_error(f"Failed to save outcomes: {e}")

    def _run_detection(self):
        """Run reach detection algorithm."""
        try:
            from mousereach.reach.core import ReachDetector
            
            detector = ReachDetector()
            results = detector.detect(self.dlc_path, self.segments_path)
            
            video_id = self.video_path.stem
            if 'DLC_' in video_id:
                video_id = video_id.split('DLC_')[0]
            
            self.reaches_path = self.video_path.parent / f"{video_id}_reaches.json"
            detector.save_results(results, self.reaches_path)
            
            self._load_reaches()
            show_info(f"Detection complete: {results.summary['total_reaches']} reaches")
            
        except Exception as e:
            show_error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_video_frames(self):
        """Load video frames into napari."""
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

        # Try to handle network paths better
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
            raise RuntimeError("No frames could be read from video")
        
        if errors > 0:
            show_warning(f"Loaded with {errors} frame read errors")
        
        self.n_frames = len(frames)
        
        if self.video_layer is not None:
            self.viewer.layers.remove(self.video_layer)
        
        self.video_layer = self.viewer.add_image(
            np.stack(frames),
            name=self.video_path.stem,
            rgb=True
        )
    
    def _add_points_layer(self):
        """Add DLC tracking points as overlay."""
        if self.dlc_df is None:
            return
        
        # Find bodyparts
        bodyparts = []
        for col in self.dlc_df.columns:
            if col.endswith('_x'):
                bodyparts.append(col[:-2])
        bodyparts = sorted(set(bodyparts))
        
        if not bodyparts:
            return
        
        # Assign colors
        colors_base = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5]
        ]
        bp_colors = {bp: colors_base[i % len(colors_base)] for i, bp in enumerate(bodyparts)}
        
        # Collect all points
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
            if self.points_layer is not None:
                self.viewer.layers.remove(self.points_layer)
            
            self.points_layer = self.viewer.add_points(
                np.array(points_data),
                name='DLC Points',
                size=6,
                face_color=np.array(point_colors),
            )
    
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
        gt_path = video_dir / f"{video_id}_reach_ground_truth.json"
        if gt_path.exists():
            status_parts.append('<span style="color: #4CAF50; font-weight: bold;">⬤ Ground Truth exists</span>')

        # Check for validation file
        val_path = video_dir / f"{video_id}_reaches_validation.json"
        if val_path.exists():
            status_parts.append('<span style="color: #2196F3; font-weight: bold;">⬤ Validated</span>')

        if status_parts:
            self.file_status_label.setText(" | ".join(status_parts))
        else:
            self.file_status_label.setText('<span style="color: #999;">No ground truth or validation files found</span>')

    def _enable_controls(self, enabled: bool):
        """Enable/disable controls."""
        self.play_btn.setEnabled(enabled)
        self.play_rev_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.prev_seg_btn.setEnabled(enabled)
        self.next_seg_btn.setEnabled(enabled)
        self.prev_reach_btn.setEnabled(enabled)
        self.next_reach_btn.setEnabled(enabled)
        self.jump_start_btn.setEnabled(enabled)
        self.jump_end_btn.setEnabled(enabled)
        self.set_start_btn.setEnabled(enabled)
        self.set_end_btn.setEnabled(enabled)
        self.add_reach_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
        self.accept_reach_btn.setEnabled(enabled)
        self.exclude_check.setEnabled(enabled)
        self.exclude_reason.setEnabled(enabled)
        self.flag_check.setEnabled(enabled)
        self.flag_reason.setEnabled(enabled)
        self.save_progress_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.save_gt_btn.setEnabled(enabled)
    
    def _on_frame_change(self, event):
        """Handle frame change."""
        frame_idx = self.viewer.dims.current_step[0]
        
        # Update frame display
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.frame_label.setText(f"Frame: {frame_idx} / {self.n_frames}  ({mins}:{secs:05.2f})")
        
        # Update segment info
        seg = self._get_segment_for_frame(frame_idx)
        if seg != self.current_segment:
            self.current_segment = seg
            self.current_reach_idx = 0
            self._update_display()
    
    def _get_segment_for_frame(self, frame: int) -> int:
        """Get segment number for a frame."""
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= frame < self.boundaries[i + 1]:
                return i + 1
        return len(self.boundaries) - 1
    
    def _get_segment_reaches(self) -> List[Dict]:
        """Get reaches for current segment."""
        if not self.reaches_data:
            return []
        for seg in self.reaches_data.get('segments', []):
            if seg['segment_num'] == self.current_segment:
                return seg.get('reaches', [])
        return []
    
    def _get_segment_data(self) -> Optional[Dict]:
        """Get segment data for current segment."""
        if not self.reaches_data:
            return None
        for seg in self.reaches_data.get('segments', []):
            if seg['segment_num'] == self.current_segment:
                return seg
        return None
    
    def _sort_and_renumber_reaches(self):
        """Sort current segment's reaches by start frame and renumber sequentially.

        Called after editing a reach start frame to maintain temporal ordering.
        """
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        reaches = seg_data.get('reaches', [])
        reaches.sort(key=lambda r: r['start_frame'])
        for i, reach in enumerate(reaches):
            reach['reach_num'] = i + 1

    def _count_total_reaches(self) -> int:
        """Count total reaches in video."""
        if not self.reaches_data:
            return 0
        return sum(seg.get('n_reaches', 0) for seg in self.reaches_data.get('segments', []))
    
    def _update_display(self):
        """Update all display elements."""
        # Segment info
        n_segments = len(self.boundaries) - 1
        reaches = self._get_segment_reaches()
        seg_data = self._get_segment_data()
        
        flagged = seg_data.get('flagged_for_review', False) if seg_data else False
        flag_text = " [FLAGGED]" if flagged else ""
        
        self.segment_label.setText(
            f"Segment: {self.current_segment} / {n_segments} | "
            f"Reaches: {len(reaches)}{flag_text}"
        )
        
        # Flag checkbox
        self.flag_check.blockSignals(True)
        self.flag_check.setChecked(flagged)
        self.flag_check.blockSignals(False)
        
        self.flag_reason.setText(seg_data.get('flag_reason', '') if seg_data else '')
        
        # Reach list
        self._update_reach_list()
        
        # Reach info
        self._update_reach_info()
    
    def _update_reach_list(self):
        """Update reach list widget."""
        self.reach_list.clear()
        reaches = self._get_segment_reaches()

        # Calculate review progress across all segments
        total_reaches = 0
        reviewed_reaches = 0
        if self.reaches_data:
            for seg in self.reaches_data.get('segments', []):
                for r in seg.get('reaches', []):
                    total_reaches += 1
                    if r.get('human_verified', False):
                        reviewed_reaches += 1

        # Update progress label with color coding
        if total_reaches == 0:
            self.review_progress_label.setText("Reviewed: 0/0 reaches")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        elif reviewed_reaches == total_reaches:
            self.review_progress_label.setText(f"Reviewed: {reviewed_reaches}/{total_reaches} - ALL DONE!")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif reviewed_reaches > 0:
            self.review_progress_label.setText(f"Reviewed: {reviewed_reaches}/{total_reaches} reaches")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        else:
            self.review_progress_label.setText(f"Reviewed: {reviewed_reaches}/{total_reaches} reaches")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #2196F3;")

        for i, r in enumerate(reaches):
            source = r.get('source', 'algorithm')
            verified = r.get('human_verified', False)
            corrected = r.get('human_corrected', False)
            excluded = r.get('exclude_from_analysis', False)

            # Status icon: ✓ = verified, ○ = pending
            status_icon = "✓" if verified else "○"
            source_tag = " [+]" if source == 'human_added' else ""
            corrected_tag = " [c]" if corrected else ""
            note_tag = " [!]" if r.get('review_note') else ""
            excluded_tag = " [X]" if excluded else ""

            # Outcome tag for causal reach
            outcome_tag = self._get_reach_outcome_tag(r)
            outcome_suffix = f" {outcome_tag}" if outcome_tag else ""

            item = QListWidgetItem(
                f"{status_icon} #{r['reach_num']}: {r['start_frame']}-{r['end_frame']} "
                f"({r['duration_frames']}f){source_tag}{corrected_tag}{note_tag}{excluded_tag}{outcome_suffix}"
            )

            # Color code by status
            if excluded:
                item.setBackground(QColor(255, 200, 200))  # Light red
                tooltip = r.get('exclude_reason', 'Excluded from analysis')
                item.setToolTip(f"EXCLUDED: {tooltip}")
            elif outcome_tag:
                # Causal reach gets outcome-colored text
                outcome_colors = {
                    '[R]': QColor('#4CAF50'),   # Green - retrieved
                    '[D]': QColor('#FF9800'),   # Orange - displaced SA
                    '[O]': QColor('#F44336'),   # Red - displaced outside
                    '[U]': QColor('#2196F3'),   # Blue - untouched
                    '[NP]': QColor('#9E9E9E'),  # Gray - no pellet
                    '[?]': QColor('#9C27B0'),   # Purple - uncertain
                }
                color = outcome_colors.get(outcome_tag, QColor('#FF9800'))
                item.setForeground(color)
                item.setToolTip(f"CAUSAL REACH - pellet outcome: {outcome_tag}")
            elif verified:
                item.setForeground(QColor('#4CAF50'))  # Green for verified
            elif r.get('review_note'):
                item.setToolTip(r['review_note'])
                item.setBackground(QColor(255, 255, 200))  # Light yellow
            self.reach_list.addItem(item)

        if reaches and self.current_reach_idx < len(reaches):
            self.reach_list.setCurrentRow(self.current_reach_idx)

        # Update comparison panel
        self._update_comparison_panel()
    
    def _update_reach_info(self):
        """Update current reach info."""
        reaches = self._get_segment_reaches()

        if not reaches:
            self.reach_info.setText("No reaches in this segment")
            self.exclude_check.setEnabled(False)
            self.exclude_reason.setEnabled(False)
            return

        if self.current_reach_idx >= len(reaches):
            self.current_reach_idx = len(reaches) - 1

        r = reaches[self.current_reach_idx]

        source = "Human added" if r.get('source') == 'human_added' else "Algorithm"
        corrected = " (corrected)" if r.get('human_corrected', False) else ""
        excluded = " [EXCLUDED]" if r.get('exclude_from_analysis', False) else ""
        note = r.get('review_note', '')
        note_line = f"\n>> {note}" if note else ""
        exclude_line = f"\nExclude reason: {r.get('exclude_reason', 'N/A')}" if r.get('exclude_from_analysis') else ""

        # Check if this reach is the causal reach for the segment's pellet outcome
        outcome_tag = self._get_reach_outcome_tag(r)
        outcome_line = ""
        if outcome_tag:
            outcome_seg = self._get_outcome_for_segment(self.current_segment)
            if outcome_seg:
                outcome_name = outcome_seg.get('outcome', 'unknown')
                conf = outcome_seg.get('confidence', 0)
                interaction = outcome_seg.get('interaction_frame')
                interaction_str = f", interaction frame: {interaction}" if interaction else ""
                outcome_line = f"\nOutcome: {outcome_name} (conf: {conf:.2f}) [CAUSAL REACH]{interaction_str}"

        self.reach_info.setText(
            f"Reach {self.current_reach_idx + 1}/{len(reaches)}{corrected}{excluded}\n"
            f"Start: {r['start_frame']} | End: {r['end_frame']} | "
            f"Duration: {r['duration_frames']}f\n"
            f"Source: {source}{outcome_line}{note_line}{exclude_line}"
        )

        # Update exclusion checkbox state (without triggering handler)
        self.exclude_check.blockSignals(True)
        self.exclude_check.setChecked(r.get('exclude_from_analysis', False))
        self.exclude_check.blockSignals(False)
        self.exclude_check.setEnabled(True)

        # Update reason text
        self.exclude_reason.blockSignals(True)
        self.exclude_reason.setText(r.get('exclude_reason', '') or '')
        self.exclude_reason.blockSignals(False)
        self.exclude_reason.setEnabled(r.get('exclude_from_analysis', False))

    def _on_comparison_item_selected(self, index: int):
        """Handle selection in comparison panel - jump to that reach."""
        reaches = self._get_segment_reaches()
        if 0 <= index < len(reaches):
            self.current_reach_idx = index
            self._update_reach_info()
            self.reach_list.setCurrentRow(index)
            # Jump to reach start frame
            r = reaches[index]
            self.viewer.dims.set_current_step(0, r['start_frame'])

    def _update_comparison_panel(self):
        """Update the comparison panel with current vs original reaches."""
        if not hasattr(self, 'comparison_panel') or not self.reaches_data:
            return

        # Get current segment's reaches
        current_reaches = self._get_segment_reaches()

        # Get original reaches for comparison (from original_reaches_json if available)
        original_reaches = []
        if self.original_reaches_json:
            orig_data = json.loads(self.original_reaches_json)
            for seg in orig_data.get('segments', []):
                if seg.get('segment_num') == self.current_segment:
                    original_reaches = seg.get('reaches', [])
                    break

        if not original_reaches:
            original_reaches = current_reaches

        # Check if GT file exists
        gt_exists = False
        if self.video_path:
            video_id = self.video_path.stem.split('DLC')[0].rstrip('_')
            gt_path = self.video_path.parent / f"{video_id}_reach_ground_truth.json"
            gt_exists = gt_path.exists()

        # Create comparison items
        items = create_reach_comparison(original_reaches, current_reaches, self.current_segment)
        self.comparison_panel.set_items(items, gt_exists=gt_exists)

    # === Playback ===
    
    def _set_speed(self, speed: int):
        """Set playback speed multiplier."""
        self.playback_speed = speed
        # Update button states
        for s, btn in self.speed_buttons.items():
            btn.setChecked(s == speed)
        # If playing, restart timer with new speed
        if self.is_playing:
            self.playback_timer.stop()
            interval = max(1, int(1000 / self.fps))
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
        if self.is_playing and self.playback_direction == 1:
            # Already playing forward, stop
            self._stop_play()
            return
        elif self.is_playing and self.playback_direction == -1:
            # Already playing reverse, stop
            self._stop_play()
            return
        
        self.is_playing = True
        interval = max(1, int(1000 / self.fps))
        self.playback_timer.start(interval)
        
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
        """Advance frames during playback (skips frames for higher speeds)."""
        current = self.viewer.dims.current_step[0]
        # Skip frames based on speed multiplier and direction
        new_frame = current + (self.playback_speed * self.playback_direction)
        
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
        """Jump forward/backward by delta frames."""
        current = self.viewer.dims.current_step[0]
        new_frame = max(0, min(self.n_frames - 1, current + delta))
        self.viewer.dims.set_current_step(0, new_frame)
    
    # === Segment Navigation ===

    def goto_segment(self, seg_num: int):
        """Jump to a segment (public API for external callers)."""
        self._goto_segment(seg_num)

    def _goto_segment(self, seg_num: int):
        """Jump to a segment."""
        if seg_num < 1 or seg_num > len(self.boundaries) - 1:
            return

        self.current_segment = seg_num
        self.current_reach_idx = 0

        # Jump to segment start
        self.viewer.dims.set_current_step(0, self.boundaries[seg_num - 1])
        self._update_display()
    
    def _prev_segment(self):
        if self.current_segment > 1:
            self._goto_segment(self.current_segment - 1)
    
    def _next_segment(self):
        if self.current_segment < len(self.boundaries) - 1:
            self._goto_segment(self.current_segment + 1)
    
    # === Reach Navigation ===
    
    def _prev_reach(self):
        reaches = self._get_segment_reaches()
        if self.current_reach_idx > 0:
            self.current_reach_idx -= 1
            self._update_reach_info()
            self._update_reach_list()
            self._jump_to_reach_frame('start')
    
    def _next_reach(self):
        reaches = self._get_segment_reaches()
        if self.current_reach_idx < len(reaches) - 1:
            self.current_reach_idx += 1
            self._update_reach_info()
            self._update_reach_list()
            self._jump_to_reach_frame('start')
    
    def _select_reach(self, item):
        """Select reach from list."""
        self.current_reach_idx = self.reach_list.row(item)
        self._update_reach_info()
    
    def _jump_to_reach(self, item):
        """Jump to reach on double-click."""
        self.current_reach_idx = self.reach_list.row(item)
        self._update_reach_info()
        self._jump_to_reach_frame('start')
    
    def _jump_to_reach_frame(self, which: str):
        """Jump to start or end of current reach."""
        reaches = self._get_segment_reaches()
        if not reaches or self.current_reach_idx >= len(reaches):
            return
        
        r = reaches[self.current_reach_idx]
        frame = r['start_frame'] if which == 'start' else r['end_frame']
        self.viewer.dims.set_current_step(0, frame)
    
    # === Edit Reaches ===
    
    def _set_reach_start(self):
        """Set current reach start to current frame."""
        reaches = self._get_segment_reaches()
        if not reaches or self.current_reach_idx >= len(reaches):
            return
        
        frame = self.viewer.dims.current_step[0]
        r = reaches[self.current_reach_idx]
        
        if frame >= r['end_frame']:
            show_warning("Start must be before end")
            return
        
        # Track correction
        if not r.get('human_corrected'):
            r['original_start'] = r['start_frame']
            r['human_corrected'] = True
        
        r['start_frame'] = frame
        r['duration_frames'] = r['end_frame'] - frame

        # Re-sort and renumber in case start moved past another reach
        self._sort_and_renumber_reaches()

        self._increment_corrections()
        self._update_display()
        show_info(f"Reach start set to frame {frame}")

    def _set_reach_end(self):
        """Set current reach end to current frame."""
        reaches = self._get_segment_reaches()
        if not reaches or self.current_reach_idx >= len(reaches):
            return

        frame = self.viewer.dims.current_step[0]
        r = reaches[self.current_reach_idx]

        if frame <= r['start_frame']:
            show_warning("End must be after start")
            return

        # Track correction
        if not r.get('human_corrected'):
            r['original_end'] = r['end_frame']
            r['human_corrected'] = True

        r['end_frame'] = frame
        r['duration_frames'] = frame - r['start_frame']

        self._increment_corrections()
        self._update_display()
        show_info(f"Reach end set to frame {frame}")

    def _accept_current_reach(self):
        """Accept current reach as-is (algorithm is correct).

        This provides explicit confirmation that the user reviewed this reach
        and agrees with the algorithm's detection, without making changes.
        """
        reaches = self._get_segment_reaches()
        if not reaches:
            show_warning("No reaches in this segment")
            return

        if self.current_reach_idx >= len(reaches):
            return

        r = reaches[self.current_reach_idx]
        r['human_verified'] = True
        r['verified_by'] = get_username()
        r['verified_at'] = datetime.now().isoformat()

        self._update_display()

        # Auto-advance to next unreviewed reach
        next_unreviewed = self._find_next_unreviewed_reach()
        if next_unreviewed is not None:
            seg_num, reach_idx = next_unreviewed
            if seg_num == self.current_segment:
                self.current_reach_idx = reach_idx
                self._update_display()
                show_info(f"Reach accepted. Moving to reach #{reach_idx + 1}...")
            else:
                self.current_segment = seg_num
                self.current_reach_idx = reach_idx
                self._goto_segment(seg_num)
                show_info(f"Reach accepted. Moving to segment {seg_num}, reach #{reach_idx + 1}...")
        else:
            show_info(f"Reach accepted. All reaches reviewed!")

    def _find_next_unreviewed_reach(self):
        """Find the next reach that hasn't been reviewed yet.

        Returns tuple (segment_num, reach_idx) or None if all reviewed.
        """
        if not self.reaches_data:
            return None

        # First check remaining reaches in current segment
        reaches = self._get_segment_reaches()
        for i in range(self.current_reach_idx + 1, len(reaches)):
            if not reaches[i].get('human_verified'):
                return (self.current_segment, i)

        # Then check subsequent segments
        for seg in self.reaches_data.get('segments', []):
            if seg['segment_num'] > self.current_segment:
                for i, r in enumerate(seg.get('reaches', [])):
                    if not r.get('human_verified'):
                        return (seg['segment_num'], i)

        # Wrap around to earlier segments
        for seg in self.reaches_data.get('segments', []):
            if seg['segment_num'] < self.current_segment:
                for i, r in enumerate(seg.get('reaches', [])):
                    if not r.get('human_verified'):
                        return (seg['segment_num'], i)

        return None

    def _start_add_reach(self):
        """Start adding a new reach."""
        self.annotating_new_reach = True
        self.new_reach_start = self.viewer.dims.current_step[0]
        
        self.add_status.setText(f"Adding reach: START={self.new_reach_start}, now navigate to END and press A")
        self.finish_add_btn.setVisible(True)
        self.cancel_add_btn.setVisible(True)
        self.add_reach_btn.setText("Finish at current frame (A)")
    
    def _finish_add_reach(self):
        """Finish adding the new reach."""
        if not self.annotating_new_reach:
            return
        
        end_frame = self.viewer.dims.current_step[0]
        
        if end_frame <= self.new_reach_start:
            show_warning("End must be after start")
            return
        
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        
        # Generate IDs
        reaches = seg_data['reaches']
        existing_nums = [r['reach_num'] for r in reaches]
        next_num = max(existing_nums) + 1 if existing_nums else 1
        
        max_id = 0
        for seg in self.reaches_data['segments']:
            for r in seg['reaches']:
                max_id = max(max_id, r.get('reach_id', 0))
        
        # Create new reach
        new_reach = {
            'reach_id': max_id + 1,
            'reach_num': next_num,
            'start_frame': self.new_reach_start,
            'apex_frame': (self.new_reach_start + end_frame) // 2,  # Estimate
            'end_frame': end_frame,
            'duration_frames': end_frame - self.new_reach_start,
            'max_extent_pixels': 0,
            'max_extent_ruler': 0,
            'source': 'human_added',
            'human_corrected': False,
            'original_start': None,
            'original_end': None
        }
        
        reaches.append(new_reach)

        # Sort by start frame and renumber so reaches are always in temporal order
        reaches.sort(key=lambda r: r['start_frame'])
        for i, reach in enumerate(reaches):
            reach['reach_num'] = i + 1

        seg_data['n_reaches'] = len(reaches)

        # Track addition
        self.reaches_data['reaches_added'] = self.reaches_data.get('reaches_added', 0) + 1

        self._cancel_add_reach()  # Reset UI
        self._update_display()
        show_info(f"Added reach: {self.new_reach_start}-{end_frame}")
    
    def _cancel_add_reach(self):
        """Cancel adding a reach."""
        self.annotating_new_reach = False
        self.new_reach_start = None
        self.add_status.setText("")
        self.finish_add_btn.setVisible(False)
        self.cancel_add_btn.setVisible(False)
        self.add_reach_btn.setText("Add NEW reach starting here (A)")
    
    def _delete_reach(self):
        """Delete the current reach."""
        reaches = self._get_segment_reaches()
        if not reaches or self.current_reach_idx >= len(reaches):
            return
        
        r = reaches[self.current_reach_idx]
        
        # Confirm
        reply = QMessageBox.question(
            self, "Delete Reach",
            f"Delete reach #{r['reach_num']} ({r['start_frame']}-{r['end_frame']})?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        
        seg_data = self._get_segment_data()
        seg_data['reaches'].remove(r)

        # Renumber remaining reaches to keep sequential order
        for i, reach in enumerate(seg_data['reaches']):
            reach['reach_num'] = i + 1

        seg_data['n_reaches'] = len(seg_data['reaches'])

        # Track deletion
        self.reaches_data['reaches_removed'] = self.reaches_data.get('reaches_removed', 0) + 1

        if self.current_reach_idx > 0:
            self.current_reach_idx -= 1
        
        self._update_display()
        show_info(f"Deleted reach")
    
    def _increment_corrections(self):
        """Increment corrections counter."""
        self.reaches_data['corrections_made'] = self.reaches_data.get('corrections_made', 0) + 1

    # === Reach Exclusion ===

    def _toggle_exclude_reach(self, state):
        """Toggle exclusion flag on current reach (from checkbox)."""
        reaches = self._get_segment_reaches()
        if not reaches or self.current_reach_idx >= len(reaches):
            return

        r = reaches[self.current_reach_idx]
        r['exclude_from_analysis'] = bool(state)

        # Enable/disable reason field
        self.exclude_reason.setEnabled(bool(state))
        if not state:
            r['exclude_reason'] = None

        self._update_reach_list()
        self._update_reach_info()

    def _toggle_exclude_reach_key(self):
        """Toggle exclusion via 'X' keybinding."""
        if not self.exclude_check.isEnabled():
            return
        self.exclude_check.setChecked(not self.exclude_check.isChecked())

    def _update_exclude_reason(self, text):
        """Update exclusion reason for current reach."""
        reaches = self._get_segment_reaches()
        if not reaches or self.current_reach_idx >= len(reaches):
            return

        r = reaches[self.current_reach_idx]
        r['exclude_reason'] = text if text else None

    # === Flagging ===

    def _toggle_flag(self, state):
        """Toggle flag on current segment."""
        seg_data = self._get_segment_data()
        if not seg_data:
            return
        
        seg_data['flagged_for_review'] = bool(state)
        
        # Update count
        flagged_count = sum(1 for s in self.reaches_data['segments'] if s.get('flagged_for_review'))
        self.reaches_data['segments_flagged'] = flagged_count
        
        self._update_display()
    
    def _update_flag_reason(self, text):
        """Update flag reason."""
        seg_data = self._get_segment_data()
        if seg_data:
            seg_data['flag_reason'] = text
    
    # === Save ===
    
    def _save_progress(self):
        """Save work in progress - can resume later."""
        if not self.reaches_path or not self.reaches_data:
            return
        
        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]
        
        # Track progress (where user left off)
        current_frame = self.viewer.dims.current_step[0]
        self.reaches_data['work_in_progress'] = True
        self.reaches_data['last_edited_by'] = get_username()
        self.reaches_data['last_edited_at'] = datetime.now().isoformat()
        self.reaches_data['last_segment'] = self.current_segment
        self.reaches_data['last_frame'] = int(current_frame)
        
        # Recalculate summary
        total = sum(s['n_reaches'] for s in self.reaches_data['segments'])
        self.reaches_data['summary']['total_reaches'] = total
        
        # Save
        with open(self.reaches_path, 'w') as f:
            json.dump(self.reaches_data, f, indent=2)
        
        self.status_label.setText(f"Progress saved @ seg {self.current_segment}, frame {current_frame}")
        show_info(f"Progress saved - segment {self.current_segment}, frame {current_frame}")
    
    def _save_validated(self):
        """Save as validated and mark complete.

        New architecture (v2.3+): Files stay in Processing/, validation status
        is set in JSON metadata. No folder-based triage.
        """
        if not self.reaches_path or not self.reaches_data:
            return

        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        # Count changes from original
        original_data = json.loads(self.original_reaches_json) if self.original_reaches_json else {}
        original_total = sum(s.get('n_reaches', 0) for s in original_data.get('segments', []))
        current_total = sum(s['n_reaches'] for s in self.reaches_data['segments'])

        # Build validation_record (embedded audit trail)
        validation_record = {
            'validated_by': get_username(),
            'validated_at': datetime.now().isoformat(),
            'algorithm_version': self.reaches_data.get('detector_version'),
            'original_total_reaches': original_total,
            'final_total_reaches': current_total,
            'corrections_made': self.reaches_data.get('corrections_made', 0),
            'reaches_added': self.reaches_data.get('reaches_added', 0),
            'reaches_removed': self.reaches_data.get('reaches_removed', 0),
            'segments_flagged': self.reaches_data.get('segments_flagged', 0),
        }

        # Mark validated (complete)
        self.reaches_data['validation_status'] = 'validated'  # KEY: marks as human-reviewed
        self.reaches_data['validation_record'] = validation_record  # Embedded audit trail
        self.reaches_data['validated'] = True  # Legacy compatibility
        self.reaches_data['validated_by'] = get_username()
        self.reaches_data['validated_at'] = datetime.now().isoformat()
        self.reaches_data['work_in_progress'] = False  # Clear WIP flag

        # Recalculate summary
        self.reaches_data['summary']['total_reaches'] = current_total

        # Save
        with open(self.reaches_path, 'w') as f:
            json.dump(self.reaches_data, f, indent=2)

        # Update pipeline index
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_validation_changed(video_id, "reach", "validated")
            index.save()
        except Exception:
            pass  # Silent fail - index will catch up on next rebuild

        # Log performance metrics (algorithm vs human comparison)
        try:
            from mousereach.performance import PerformanceLogger
            original_data = json.loads(self.original_reaches_json) if self.original_reaches_json else {}
            PerformanceLogger().log_reach_detection(
                video_id=video_id,
                algo_data=original_data,
                human_data=self.reaches_data,
                algo_version=self.reaches_data.get('detector_version'),
                validator=get_username()
            )
        except Exception:
            pass  # Don't block validation if logging fails

        changes = validation_record['corrections_made'] + validation_record['reaches_added'] + validation_record['reaches_removed']
        changes_text = f"{changes} changes" if changes else "no changes"
        self.status_label.setText(f"Validated: {video_id} ({changes_text})")
        show_info(f"Saved validated reaches ({changes_text})")

    def _save_ground_truth(self):
        """Save as ground truth for algorithm development.

        GT files only contain items the human explicitly interacted with.
        Algorithm-seeded items the human never touched are NOT saved.
        This makes GT files true training data - only human-provided examples.

        A reach is included in GT if:
        - human_corrected == True (user changed the start/end frame), OR
        - source == 'human_added' (user manually added this reach), OR
        - human_verified == True (user explicitly marked as verified)
        """
        if not self.reaches_data:
            return

        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        output_path = self.video_path.parent / f"{video_id}_reach_ground_truth.json"

        # Build GT segments - ONLY include human-interacted reaches
        gt_segments = []
        total_reaches_in_video = 0
        gt_reaches_saved = 0

        for seg in self.reaches_data['segments']:
            total_reaches_in_video += len(seg.get('reaches', []))

            gt_reaches = []
            for r in seg.get('reaches', []):
                # Determine if this reach is human-verified
                # True if: human corrected it, human added it, or explicitly marked verified
                is_verified = (
                    r.get('human_corrected', False) or
                    r.get('source') == 'human_added' or
                    r.get('human_verified', False)  # Preserve explicit marking
                )

                # Only include in GT if human interacted with this reach
                if is_verified:
                    gt_reach = {
                        'reach_id': r.get('reach_id'),
                        'reach_num': r.get('reach_num'),
                        'start_frame': r['start_frame'],
                        'apex_frame': r.get('apex_frame'),
                        'end_frame': r['end_frame'],
                        'duration_frames': r.get('duration_frames'),
                        'source': r.get('source', 'algorithm'),
                        'human_corrected': r.get('human_corrected', False),
                        'human_verified': True,
                        'verified_by': get_username(),
                        'verified_at': datetime.now().isoformat(),
                        'exclude_from_analysis': r.get('exclude_from_analysis', False),
                        'exclude_reason': r.get('exclude_reason'),
                        # Preserve original values if corrected
                        'original_start_frame': r.get('original_start_frame'),
                        'original_end_frame': r.get('original_end_frame'),
                    }
                    gt_reaches.append(gt_reach)
                    gt_reaches_saved += 1

            # Only include segment in GT if it has human-verified reaches
            if gt_reaches:
                gt_seg = {
                    'segment_num': seg['segment_num'],
                    'start_frame': seg.get('start_frame'),
                    'end_frame': seg.get('end_frame'),
                    'flagged_for_review': seg.get('flagged_for_review', False),
                    'flag_reason': seg.get('flag_reason'),
                    'reaches': gt_reaches,
                    'human_verified': True,
                    'n_reaches': len(gt_reaches),
                }
                gt_segments.append(gt_seg)

        # Warn if no human-interacted items
        if gt_reaches_saved == 0:
            show_warning("No reaches were verified or corrected. GT file will be empty.")

        gt_data = {
            'video_name': video_id,
            'type': 'ground_truth',
            'created_by': get_username(),
            'created_at': datetime.now().isoformat(),
            'n_segments': len(gt_segments),
            'segments': gt_segments,
            # Completeness tracking
            'gt_complete': gt_reaches_saved > 0,
            'total_reaches_in_video': total_reaches_in_video,
            'reaches_annotated': gt_reaches_saved,
        }

        with open(output_path, 'w') as f:
            json.dump(gt_data, f, indent=2)

        # Update pipeline index with GT status
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_gt_created(video_id, "reach", is_complete=(gt_reaches_saved > 0))
            index.save()
        except Exception:
            pass  # Silent fail - index will catch up on next rebuild

        status_text = f"{gt_reaches_saved}/{total_reaches_in_video} reaches"
        self.status_label.setText(f"Saved GT: {output_path.name} ({status_text})")
        show_info(f"Saved ground truth: {status_text}")


def main():
    """Launch the annotator."""
    viewer = napari.Viewer(title="MouseReach Reach Annotator")
    widget = ReachAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Reach Annotator", area="right")
    
    print("\nKeyboard shortcuts:")
    print("  Space       - Play forward")
    print("  R           - Play reverse")
    print("  1-5         - Speed (1x/2x/4x/8x/16x)")
    print("  Left/Right  - Step 1 frame")
    print("  Shift+L/R   - Step 10 frames")
    print("  J/K         - Jump to next/prev RH visible frame")
    print("  N/P         - Next/prev reach")
    print("  S           - Set reach start")
    print("  E           - Set reach end")
    print("  A           - Add new reach (press twice: start, end)")
    print("  DEL         - Delete reach")
    print("  Ctrl+S      - Save progress")
    
    napari.run()


if __name__ == "__main__":
    main()
