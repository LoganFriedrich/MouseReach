"""
MouseReach Unified Review Tool
============================

Single tabbed widget combining all three review tools:
- Segmentation (boundary review)
- Reach detection review
- Pellet outcome review

Features:
- Load video once, shared across all tabs
- Tab switching auto-loads relevant data
- Keybindings are context-aware per active tab
- Integrates with MouseReachStateManager for shared state

Usage:
    mousereach-review-tool video.mp4
    # Or via Plugins → MouseReach → Unified Review Tool
"""

import numpy as np
from pathlib import Path
from typing import Optional
import json

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QTabWidget, QScrollArea,
    QDialog, QTextBrowser, QComboBox, QSpinBox
)
from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QFont

from typing import List, Tuple

import napari
from napari.utils.notifications import show_info, show_warning, show_error


# Help text for unified tool
HELP_TEXT = """
<h2>MouseReach Unified Review Tool</h2>

<h3>Overview</h3>
<p>This tool combines all three review steps into one interface with tabs:</p>
<ul>
<li><b>Boundaries</b> - Review segment boundary frames</li>
<li><b>Reaches</b> - Review reach detection (start/end frames)</li>
<li><b>Outcomes</b> - Review pellet outcome classification</li>
</ul>

<h3>Workflow</h3>
<ol>
<li>Load a video file - it will be shared across all tabs</li>
<li>Start with Boundaries tab, then move to Reaches, then Outcomes</li>
<li>Use keyboard shortcuts for efficient review</li>
<li>Save your work in each tab before switching</li>
</ol>

<h3>Common Keyboard Shortcuts</h3>
<ul>
<li><b>Enter</b> - Accept current item as-is (algorithm correct)</li>
<li><b>Left/Right</b> - Move 1 frame</li>
<li><b>Shift+Left/Right</b> - Move 10 frames</li>
<li><b>Space</b> - Context-dependent (set boundary, play/pause)</li>
<li><b>S</b> - Save validated</li>
</ul>

<h3>Tab-Specific Shortcuts</h3>
<p><b>Boundaries:</b> Enter = accept, N/P = next/prev, Space = set here</p>
<p><b>Reaches:</b> Enter = accept, N/P = next/prev, S/E = set start/end, A = add</p>
<p><b>Outcomes:</b> Enter = accept, R/D/O/U = set outcome, I/K = mark frames</p>
"""


class UnifiedReviewWidget(QWidget):
    """
    Unified review widget with tabs for segmentation, reach, and outcome review.

    Shares a single video layer across all tabs for efficient memory usage.
    """

    # Signal emitted when data is saved (for state manager integration)
    data_saved = Signal(Path)

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Shared state
        self.video_path: Optional[Path] = None
        self.video_layer = None
        self.n_frames = 0
        self.fps = 60.0
        self.scale_factor = 1.0

        # Segment boundaries (loaded from seg widget)
        self.boundaries: List[int] = []
        self.current_item_idx = 0  # Current boundary/reach/segment index

        # Playback state
        self.is_playing = False
        self.playback_speed = 1.0
        self.playback_direction = 1  # 1 = forward, -1 = backward
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        # Video dropdown data: list of (video_id, video_path, review_type)
        self._dropdown_items: List[Tuple[str, Path, str]] = []

        # Child widgets (created lazily)
        self._seg_widget = None
        self._reach_widget = None
        self._outcome_widget = None

        # Track which tabs have been initialized (UI created)
        self._tab_initialized = {'seg': False, 'reach': False, 'outcome': False}
        # Track which tabs have had data loaded (separate from UI init)
        self._data_loaded = {'seg': False, 'reach': False, 'outcome': False}

        self._build_ui()
        self._setup_common_keybindings()
        self._populate_video_dropdown()

        # Start background initialization of tab widgets after UI renders
        QTimer.singleShot(0, self._init_tabs_background)

    def _show_help(self):
        """Show help dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Unified Review Tool - Help")
        dialog.setMinimumSize(550, 450)
        layout = QVBoxLayout()
        dialog.setLayout(layout)

        text = QTextBrowser()
        text.setHtml(HELP_TEXT)
        text.setOpenExternalLinks(True)
        layout.addWidget(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec_()

    def _build_ui(self):
        """Build the unified widget UI."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(main_layout)

        # === Header with Help Button ===
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>MouseReach Review Tool</b>")
        header_label.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        help_btn = QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                border-radius: 12px;
                background-color: #555;
                color: white;
            }
            QPushButton:hover {
                background-color: #777;
            }
        """)
        help_btn.setToolTip("Click for help")
        help_btn.clicked.connect(self._show_help)
        header_layout.addWidget(help_btn)
        main_layout.addLayout(header_layout)

        # === Video Selection Section (dropdown) ===
        video_group = QGroupBox("Video")
        video_layout = QVBoxLayout()
        video_group.setLayout(video_layout)

        # Video dropdown
        dropdown_layout = QHBoxLayout()
        self.video_combo = QComboBox()
        self.video_combo.setPlaceholderText("Select video to review...")
        self.video_combo.currentIndexChanged.connect(self._on_video_selected)
        self.video_combo.setMinimumWidth(200)
        dropdown_layout.addWidget(self.video_combo, stretch=1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Refresh list of videos needing review")
        self.refresh_btn.clicked.connect(self._populate_video_dropdown)
        self.refresh_btn.setMaximumWidth(70)
        dropdown_layout.addWidget(self.refresh_btn)
        video_layout.addLayout(dropdown_layout)

        # Video status label
        self.video_label = QLabel("No video loaded")
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("color: #888;")
        video_layout.addWidget(self.video_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        video_layout.addWidget(self.progress)

        main_layout.addWidget(video_group)

        # === Shared Navigation Bar ===
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()
        nav_group.setLayout(nav_layout)

        # Frame display row
        frame_row = QHBoxLayout()
        self.frame_label = QLabel("Frame: -- / --")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.frame_label.setFont(font)
        frame_row.addWidget(self.frame_label)

        frame_row.addStretch()

        self.time_label = QLabel("Time: --:--")
        frame_row.addWidget(self.time_label)
        nav_layout.addLayout(frame_row)

        # Playback controls row
        play_row = QHBoxLayout()

        self.play_rev_btn = QPushButton("◀ Rev")
        self.play_rev_btn.clicked.connect(self._play_reverse)
        self.play_rev_btn.setEnabled(False)
        self.play_rev_btn.setMaximumWidth(50)
        play_row.addWidget(self.play_rev_btn)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._play_forward)
        self.play_btn.setEnabled(False)
        self.play_btn.setMaximumWidth(50)
        play_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.clicked.connect(self._stop_play)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMaximumWidth(30)
        play_row.addWidget(self.stop_btn)

        play_row.addStretch()

        # Speed buttons
        play_row.addWidget(QLabel("Speed:"))
        self.speed_buttons = {}
        for speed in [0.5, 1, 2, 4, 8]:
            label = f"{speed}x" if speed >= 1 else f"1/{int(1/speed)}x"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMaximumWidth(35)
            btn.clicked.connect(lambda checked, s=speed: self._set_speed(s))
            self.speed_buttons[speed] = btn
            play_row.addWidget(btn)
        self.speed_buttons[1].setChecked(True)
        nav_layout.addLayout(play_row)

        # Frame jump buttons row
        jump_row = QHBoxLayout()

        self.back_seg_btn = QPushButton("<<< -seg")
        self.back_seg_btn.clicked.connect(lambda: self._jump_to_prev_segment())
        self.back_seg_btn.setEnabled(False)
        jump_row.addWidget(self.back_seg_btn)

        self.back_100_btn = QPushButton("<< -100")
        self.back_100_btn.clicked.connect(lambda: self._jump_frames(-100))
        jump_row.addWidget(self.back_100_btn)

        self.back_10_btn = QPushButton("< -10")
        self.back_10_btn.clicked.connect(lambda: self._jump_frames(-10))
        jump_row.addWidget(self.back_10_btn)

        self.fwd_10_btn = QPushButton("+10 >")
        self.fwd_10_btn.clicked.connect(lambda: self._jump_frames(10))
        jump_row.addWidget(self.fwd_10_btn)

        self.fwd_100_btn = QPushButton("+100 >>")
        self.fwd_100_btn.clicked.connect(lambda: self._jump_frames(100))
        jump_row.addWidget(self.fwd_100_btn)

        self.fwd_seg_btn = QPushButton("+seg >>>")
        self.fwd_seg_btn.clicked.connect(lambda: self._jump_to_next_segment())
        self.fwd_seg_btn.setEnabled(False)
        jump_row.addWidget(self.fwd_seg_btn)

        nav_layout.addLayout(jump_row)

        # Item navigation row (context-aware: Boundary/Reach/Segment)
        item_row = QHBoxLayout()

        self.prev_item_btn = QPushButton("<< Prev Item")
        self.prev_item_btn.clicked.connect(self._prev_item)
        self.prev_item_btn.setEnabled(False)
        item_row.addWidget(self.prev_item_btn)

        self.item_label = QLabel("Item: -- / --")
        self.item_label.setAlignment(Qt.AlignCenter)
        item_row.addWidget(self.item_label, stretch=1)

        self.next_item_btn = QPushButton("Next Item >>")
        self.next_item_btn.clicked.connect(self._next_item)
        self.next_item_btn.setEnabled(False)
        item_row.addWidget(self.next_item_btn)

        nav_layout.addLayout(item_row)

        main_layout.addWidget(nav_group)

        # === Tab Widget for Review Modes ===
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Create placeholder tabs (widgets loaded on demand)
        self.tab_widget.addTab(QWidget(), "Boundaries")
        self.tab_widget.addTab(QWidget(), "Reaches")
        self.tab_widget.addTab(QWidget(), "Outcomes")

        # Initially disable reach/outcome tabs until dependencies are met
        self.tab_widget.setTabEnabled(1, False)  # Reaches needs segments
        self.tab_widget.setTabEnabled(2, False)  # Outcomes needs segments

        main_layout.addWidget(self.tab_widget, stretch=1)

        # === Status Bar ===
        self.status_label = QLabel("Select a video from the dropdown to begin")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        main_layout.addWidget(self.status_label)

        # Connect frame change event
        self.viewer.dims.events.current_step.connect(self._on_frame_change)

    def _populate_video_dropdown(self):
        """Populate dropdown with videos needing review from PipelineIndex."""
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        self._dropdown_items = []

        try:
            from mousereach.index import PipelineIndex
            from mousereach.config import PROCESSING_ROOT

            index = PipelineIndex()
            index.load()

            # Get videos needing review for each step
            seg_videos = index.get_needs_seg_review()
            reach_videos = index.get_needs_reach_review()
            outcome_videos = index.get_needs_outcome_review()

            # Add separator and items for each category
            if seg_videos:
                self.video_combo.addItem("── Segmentation Review ──")
                self.video_combo.model().item(self.video_combo.count() - 1).setEnabled(False)
                for video_id in seg_videos[:20]:  # Limit to 20 per category
                    video_path = self._find_video_path(video_id, PROCESSING_ROOT)
                    if video_path:
                        self._dropdown_items.append((video_id, video_path, 'seg'))
                        self.video_combo.addItem(f"  {video_id}")

            if reach_videos:
                self.video_combo.addItem("── Reach Review ──")
                self.video_combo.model().item(self.video_combo.count() - 1).setEnabled(False)
                for video_id in reach_videos[:20]:
                    video_path = self._find_video_path(video_id, PROCESSING_ROOT)
                    if video_path:
                        self._dropdown_items.append((video_id, video_path, 'reach'))
                        self.video_combo.addItem(f"  {video_id}")

            if outcome_videos:
                self.video_combo.addItem("── Outcome Review ──")
                self.video_combo.model().item(self.video_combo.count() - 1).setEnabled(False)
                for video_id in outcome_videos[:20]:
                    video_path = self._find_video_path(video_id, PROCESSING_ROOT)
                    if video_path:
                        self._dropdown_items.append((video_id, video_path, 'outcome'))
                        self.video_combo.addItem(f"  {video_id}")

            # Add Browse option at the end
            self.video_combo.addItem("────────────")
            self.video_combo.model().item(self.video_combo.count() - 1).setEnabled(False)
            self._dropdown_items.append(("BROWSE", None, "browse"))
            self.video_combo.addItem("Browse for video...")

            total = len(seg_videos) + len(reach_videos) + len(outcome_videos)
            self.status_label.setText(f"Found {total} videos needing review")

        except Exception as e:
            # Fallback if index not available
            self._dropdown_items.append(("BROWSE", None, "browse"))
            self.video_combo.addItem("Browse for video...")
            self.status_label.setText(f"Index unavailable: {e}")

        self.video_combo.blockSignals(False)

    def _find_video_path(self, video_id: str, processing_root: Path) -> Optional[Path]:
        """Find the video file path for a given video ID."""
        # Look in Processing folder
        for mp4 in processing_root.glob(f"**/{video_id}*.mp4"):
            if '_preview' not in mp4.name:
                return mp4
        return None

    def _on_video_selected(self, index: int):
        """Handle dropdown selection - load video or open browse dialog."""
        if index < 0:
            return

        # Find the corresponding item in our list
        # We need to account for separator items (disabled items in dropdown)
        item_text = self.video_combo.itemText(index)

        # Skip if separator
        if item_text.startswith("──") or not item_text.strip():
            return

        # Find matching item
        item_text_clean = item_text.strip()

        for video_id, video_path, review_type in self._dropdown_items:
            if video_id == "BROWSE" and "Browse" in item_text:
                self._browse_for_video()
                return
            elif video_id == item_text_clean or item_text_clean == video_id:
                if video_path:
                    self._load_video_from_path(video_path)
                    # Auto-switch to appropriate tab
                    tab_map = {'seg': 0, 'reach': 1, 'outcome': 2}
                    if review_type in tab_map:
                        self.tab_widget.setCurrentIndex(tab_map[review_type])
                return

    def _browse_for_video(self):
        """Open file dialog to browse for a video."""
        from mousereach.config import Paths

        default_dir = ""
        for candidate in [Paths.PROCESSING_ROOT, Path.cwd()]:
            if candidate.exists():
                default_dir = str(candidate)
                break

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video",
            default_dir, "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        if path:
            self._load_video_from_path(Path(path))

    def _load_video(self):
        """Load video via file dialog (legacy method, now uses dropdown)."""
        self._browse_for_video()

    def _load_video_from_path(self, video_path: Path):
        """Load video and initialize the first tab."""
        import cv2

        # Reset data loaded flags for new video
        self._data_loaded = {'seg': False, 'reach': False, 'outcome': False}

        self.video_path = video_path
        self.video_label.setText(f"Loading: {video_path.name}")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        try:
            # Use preview if it exists, otherwise use original (don't auto-create)
            video_stem = video_path.stem.replace('_preview', '')
            preview_path = video_path.parent / f"{video_stem}_preview.mp4"

            if '_preview' not in video_path.stem and preview_path.exists():
                actual_video = preview_path
                self.scale_factor = 0.75
                print(f"Using existing preview video: {preview_path.name}")
            else:
                actual_video = video_path
                self.scale_factor = 1.0

            # Load video frames
            cap = cv2.VideoCapture(str(actual_video))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

            frames = []
            for i in range(self.n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if i % 500 == 0:
                    self.progress.setValue(int(80 * i / self.n_frames))
                    from qtpy.QtWidgets import QApplication
                    QApplication.processEvents()

            cap.release()
            self.n_frames = len(frames)
            self.progress.setValue(85)

            # Remove old video layer if exists
            if self.video_layer is not None:
                try:
                    self.viewer.layers.remove(self.video_layer)
                except (ValueError, AttributeError):
                    pass  # Layer already removed or doesn't exist

            # Add video to viewer
            self.video_layer = self.viewer.add_image(
                np.stack(frames),
                name=video_path.stem,
                rgb=True
            )

            self.progress.setValue(95)

            # Initialize the segmentation tab
            self._init_seg_widget()
            self._load_data_into_seg_widget()

            # Check if we can enable other tabs
            self._update_tab_availability()

            self.video_label.setText(f"Loaded: {video_path.name} ({self.n_frames} frames)")
            self.progress.setValue(100)
            self.status_label.setText("Video loaded. Review boundaries, then switch tabs.")

            # Load boundaries for segment navigation
            self._load_boundaries()

            # Enable navigation controls
            self._enable_nav_controls(True)
            self._update_item_label()

        except Exception as e:
            show_error(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.progress.setVisible(False)

    def _load_boundaries(self):
        """Load segment boundaries from the segments file."""
        if self.video_path is None:
            return

        video_stem = self.video_path.stem.replace('_preview', '')
        if 'DLC' in video_stem:
            video_stem = video_stem.split('DLC')[0].rstrip('_')

        # Try to find segments file
        seg_patterns = [
            f"{video_stem}_seg_validation.json",
            f"{video_stem}_segments_v2.json",
            f"{video_stem}_segments.json",
        ]

        for pattern in seg_patterns:
            seg_path = self.video_path.parent / pattern
            if seg_path.exists():
                try:
                    with open(seg_path) as f:
                        data = json.load(f)
                    self.boundaries = data.get('boundaries', data.get('validated_boundaries', []))
                    return
                except Exception:
                    pass

        self.boundaries = []

    def _init_tabs_background(self):
        """Initialize tab widgets progressively in background.

        This creates the widget UI structure without loading data,
        so users can see the interface before loading a video.
        """
        # Stagger initialization to avoid blocking UI
        QTimer.singleShot(50, self._init_seg_widget_background)
        QTimer.singleShot(150, self._init_reach_widget_background)
        QTimer.singleShot(250, self._init_outcome_widget_background)

    def _init_seg_widget_background(self):
        """Initialize boundaries widget UI in background (no data)."""
        if self._seg_widget is not None:
            return

        from mousereach.segmentation.review_widget import BoundaryReviewWidget

        self._seg_widget = BoundaryReviewWidget(self.viewer, embedded_mode=True)

        # Replace placeholder with actual widget
        self.tab_widget.removeTab(0)
        self.tab_widget.insertTab(0, self._seg_widget, "Boundaries")
        self.tab_widget.setCurrentIndex(0)

        self._tab_initialized['seg'] = True

    def _init_reach_widget_background(self):
        """Initialize reaches widget UI in background (no data)."""
        if self._reach_widget is not None:
            return

        from mousereach.reach.review_widget import ReachAnnotatorWidget

        self._reach_widget = ReachAnnotatorWidget(self.viewer, embedded_mode=True)

        # Replace placeholder
        self.tab_widget.removeTab(1)
        self.tab_widget.insertTab(1, self._reach_widget, "Reaches")

        self._tab_initialized['reach'] = True

    def _init_outcome_widget_background(self):
        """Initialize outcomes widget UI in background (no data)."""
        if self._outcome_widget is not None:
            return

        from mousereach.outcomes.review_widget import PelletOutcomeAnnotatorWidget

        self._outcome_widget = PelletOutcomeAnnotatorWidget(self.viewer, embedded_mode=True)

        # Replace placeholder
        self.tab_widget.removeTab(2)
        self.tab_widget.insertTab(2, self._outcome_widget, "Outcomes")

        self._tab_initialized['outcome'] = True

    def _init_seg_widget(self):
        """Initialize the segmentation review widget and set shared video state."""
        # Create widget if not already created by background init
        if self._seg_widget is None:
            from mousereach.segmentation.review_widget import BoundaryReviewWidget
            self._seg_widget = BoundaryReviewWidget(self.viewer, embedded_mode=True)

            # Replace placeholder with actual widget
            self.tab_widget.removeTab(0)
            self.tab_widget.insertTab(0, self._seg_widget, "Boundaries")
            self.tab_widget.setCurrentIndex(0)

            self._tab_initialized['seg'] = True

        # Always update shared video state (may have been created by background init)
        self._seg_widget._shared_video_layer = self.video_layer
        self._seg_widget._shared_n_frames = self.n_frames
        self._seg_widget._shared_video_fps = self.fps
        self._seg_widget.scale_factor = self.scale_factor

    def _init_reach_widget(self):
        """Initialize the reach review widget and set shared video state."""
        # Create widget if not already created by background init
        if self._reach_widget is None:
            from mousereach.reach.review_widget import ReachAnnotatorWidget
            self._reach_widget = ReachAnnotatorWidget(self.viewer, embedded_mode=True)

            # Replace placeholder
            self.tab_widget.removeTab(1)
            self.tab_widget.insertTab(1, self._reach_widget, "Reaches")

            self._tab_initialized['reach'] = True

        # Always update shared video state (may have been created by background init)
        self._reach_widget._shared_video_layer = self.video_layer
        self._reach_widget._shared_n_frames = self.n_frames
        self._reach_widget._shared_video_fps = self.fps
        self._reach_widget.scale_factor = self.scale_factor

    def _init_outcome_widget(self):
        """Initialize the outcome review widget and set shared video state."""
        # Create widget if not already created by background init
        if self._outcome_widget is None:
            from mousereach.outcomes.review_widget import PelletOutcomeAnnotatorWidget
            self._outcome_widget = PelletOutcomeAnnotatorWidget(self.viewer, embedded_mode=True)

            # Replace placeholder
            self.tab_widget.removeTab(2)
            self.tab_widget.insertTab(2, self._outcome_widget, "Outcomes")

            self._tab_initialized['outcome'] = True

        # Always update shared video state (may have been created by background init)
        self._outcome_widget._shared_video_layer = self.video_layer
        self._outcome_widget._shared_n_frames = self.n_frames
        self._outcome_widget._shared_video_fps = self.fps
        self._outcome_widget.scale_factor = self.scale_factor

    def _load_data_into_seg_widget(self):
        """Load data into the segmentation widget using shared video."""
        if self._seg_widget is None or self.video_path is None:
            return

        self._seg_widget._load_data_only(self.video_path)

    def _load_data_into_reach_widget(self):
        """Load data into the reach widget using shared video."""
        if self._reach_widget is None or self.video_path is None:
            return

        # Check if _load_data_only exists, otherwise use full load
        if hasattr(self._reach_widget, '_load_data_only'):
            self._reach_widget._load_data_only(self.video_path)
        else:
            # Fallback: set shared state and load
            self._reach_widget.video_path = self.video_path
            self._reach_widget.video_layer = self.video_layer
            self._reach_widget.n_frames = self.n_frames
            self._reach_widget.fps = self.fps

    def _load_data_into_outcome_widget(self):
        """Load data into the outcome widget using shared video."""
        if self._outcome_widget is None or self.video_path is None:
            return

        if hasattr(self._outcome_widget, '_load_data_only'):
            self._outcome_widget._load_data_only(self.video_path)
        else:
            self._outcome_widget.video_path = self.video_path
            self._outcome_widget.video_layer = self.video_layer
            self._outcome_widget.n_frames = self.n_frames
            self._outcome_widget.fps = self.fps

    def _update_tab_availability(self):
        """Enable/disable tabs based on data availability."""
        if self.video_path is None:
            self.tab_widget.setTabEnabled(0, False)
            self.tab_widget.setTabEnabled(1, False)
            self.tab_widget.setTabEnabled(2, False)
            return

        # Boundaries tab always available if video loaded
        self.tab_widget.setTabEnabled(0, True)

        # Check for segments file (required for reaches and outcomes)
        video_stem = self.video_path.stem.replace('_preview', '')
        if 'DLC' in video_stem:
            video_stem = video_stem.split('DLC')[0].rstrip('_')

        seg_patterns = [
            f"{video_stem}_segments.json",
            f"{video_stem}_seg_validation.json",
            f"{video_stem}_seg_ground_truth.json",
        ]

        has_segments = any(
            (self.video_path.parent / p).exists() for p in seg_patterns
        )

        # Enable reaches if segments exist
        self.tab_widget.setTabEnabled(1, has_segments)

        # Enable outcomes if segments exist
        self.tab_widget.setTabEnabled(2, has_segments)

        # Update status
        if not has_segments:
            self.status_label.setText("Complete Boundaries first to unlock Reaches and Outcomes tabs")

    def _on_tab_changed(self, index: int):
        """Handle tab switch - load data and update keybindings."""
        if self.video_path is None:
            return

        tab_names = ['seg', 'reach', 'outcome']
        tab_name = tab_names[index] if index < len(tab_names) else None

        # Initialize widget and set shared video state
        # ALWAYS call init methods - they handle both widget creation AND shared video state setup
        # (Background init creates widget but doesn't set video state)
        if tab_name == 'reach':
            self._init_reach_widget()
        elif tab_name == 'outcome':
            self._init_outcome_widget()

        # Load data if not already loaded (separate from widget init!)
        if tab_name == 'reach' and not self._data_loaded['reach']:
            self._load_data_into_reach_widget()
            self._data_loaded['reach'] = True
        elif tab_name == 'outcome' and not self._data_loaded['outcome']:
            self._load_data_into_outcome_widget()
            self._data_loaded['outcome'] = True

        # Update keybindings for the active tab
        self._update_keybindings_for_tab(index)

        # Update tab availability (in case data was saved)
        self._update_tab_availability()

        # Update item navigation labels for current tab
        self._update_item_label()

        # Status update
        tab_labels = ['Boundaries', 'Reaches', 'Outcomes']
        if index < len(tab_labels):
            self.status_label.setText(f"Reviewing: {tab_labels[index]}")

    def _setup_common_keybindings(self):
        """Set up keybindings that work across all tabs."""
        @self.viewer.bind_key('Left', overwrite=True)
        def left_key(viewer):
            self._jump_frames(-1)

        @self.viewer.bind_key('Right', overwrite=True)
        def right_key(viewer):
            self._jump_frames(1)

        @self.viewer.bind_key('Shift-Left', overwrite=True)
        def shift_left_key(viewer):
            self._jump_frames(-10)

        @self.viewer.bind_key('Shift-Right', overwrite=True)
        def shift_right_key(viewer):
            self._jump_frames(10)

        @self.viewer.bind_key('Space', overwrite=True)
        def toggle_play(viewer):
            if self.is_playing:
                self._stop_play()
            else:
                self._play_forward()

        @self.viewer.bind_key('n', overwrite=True)
        def next_item(viewer):
            self._next_item()

        @self.viewer.bind_key('p', overwrite=True)
        def prev_item(viewer):
            self._prev_item()

    def _update_keybindings_for_tab(self, tab_index: int):
        """Update context-sensitive keybindings based on active tab."""
        # The individual widgets set up their own keybindings on the viewer
        # This method can be extended if we need to swap bindings dynamically
        pass

    def _jump_frames(self, delta: int):
        """Jump forward/backward by delta frames."""
        if self.n_frames == 0:
            return
        current = self.viewer.dims.current_step[0]
        new_frame = max(0, min(self.n_frames - 1, current + delta))
        self.viewer.dims.set_current_step(0, new_frame)

    # === Frame Change Handler ===

    def _on_frame_change(self, event=None):
        """Update frame display when frame changes."""
        if self.n_frames == 0:
            return

        frame_idx = self.viewer.dims.current_step[0]
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60

        self.frame_label.setText(f"Frame: {frame_idx} / {self.n_frames}")
        self.time_label.setText(f"Time: {mins}:{secs:05.2f}")

    # === Playback Controls ===

    def _set_speed(self, speed: float):
        """Set playback speed."""
        self.playback_speed = speed
        for s, btn in self.speed_buttons.items():
            btn.setChecked(s == speed)
        if self.is_playing:
            self._update_playback_timer()

    def _update_playback_timer(self):
        """Update timer interval based on speed."""
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

    def _playback_step(self):
        """Advance frames during playback."""
        current = self.viewer.dims.current_step[0]

        if self.playback_speed >= 1:
            skip = int(self.playback_speed)
        else:
            skip = 1

        new_frame = current + (skip * self.playback_direction)

        if 0 <= new_frame < self.n_frames:
            self.viewer.dims.set_current_step(0, new_frame)
        else:
            if new_frame < 0:
                self.viewer.dims.set_current_step(0, 0)
            else:
                self.viewer.dims.set_current_step(0, self.n_frames - 1)
            self._stop_play()

    # === Segment Navigation ===

    def _jump_to_prev_segment(self):
        """Jump to previous segment boundary."""
        if not self.boundaries or len(self.boundaries) < 2:
            return

        current = self.viewer.dims.current_step[0]

        # Find the previous boundary
        for i in range(len(self.boundaries) - 1, -1, -1):
            if self.boundaries[i] < current - 5:  # Allow some tolerance
                self.viewer.dims.set_current_step(0, self.boundaries[i])
                return

        # If no previous, go to first
        self.viewer.dims.set_current_step(0, self.boundaries[0])

    def _jump_to_next_segment(self):
        """Jump to next segment boundary."""
        if not self.boundaries or len(self.boundaries) < 2:
            return

        current = self.viewer.dims.current_step[0]

        # Find the next boundary
        for boundary in self.boundaries:
            if boundary > current + 5:  # Allow some tolerance
                self.viewer.dims.set_current_step(0, boundary)
                return

        # If no next, go to last
        self.viewer.dims.set_current_step(0, self.boundaries[-1])

    # === Item Navigation (context-aware) ===

    def _prev_item(self):
        """Navigate to previous item (boundary/reach/segment based on active tab)."""
        tab_idx = self.tab_widget.currentIndex()

        if tab_idx == 0 and self._seg_widget:
            # Boundaries tab - navigate to previous boundary
            if hasattr(self._seg_widget, '_prev_boundary'):
                self._seg_widget._prev_boundary()
            self._update_item_label()
        elif tab_idx == 1 and self._reach_widget:
            # Reaches tab - navigate to previous reach
            if hasattr(self._reach_widget, '_prev_reach'):
                self._reach_widget._prev_reach()
            self._update_item_label()
        elif tab_idx == 2 and self._outcome_widget:
            # Outcomes tab - navigate to previous segment
            if hasattr(self._outcome_widget, '_prev_segment'):
                self._outcome_widget._prev_segment()
            self._update_item_label()

    def _next_item(self):
        """Navigate to next item (boundary/reach/segment based on active tab)."""
        tab_idx = self.tab_widget.currentIndex()

        if tab_idx == 0 and self._seg_widget:
            if hasattr(self._seg_widget, '_next_boundary'):
                self._seg_widget._next_boundary()
            self._update_item_label()
        elif tab_idx == 1 and self._reach_widget:
            if hasattr(self._reach_widget, '_next_reach'):
                self._reach_widget._next_reach()
            self._update_item_label()
        elif tab_idx == 2 and self._outcome_widget:
            if hasattr(self._outcome_widget, '_next_segment'):
                self._outcome_widget._next_segment()
            self._update_item_label()

    def _update_item_label(self):
        """Update the item navigation label based on active tab."""
        tab_idx = self.tab_widget.currentIndex()

        if tab_idx == 0 and self._seg_widget:
            # Boundaries
            current = getattr(self._seg_widget, 'current_boundary_idx', 0) + 1
            total = len(getattr(self._seg_widget, 'boundaries', []))
            self.prev_item_btn.setText("<< Prev Boundary")
            self.next_item_btn.setText("Next Boundary >>")
            self.item_label.setText(f"Boundary: {current} / {total}")
        elif tab_idx == 1 and self._reach_widget:
            # Reaches
            current = getattr(self._reach_widget, 'current_reach_idx', 0) + 1
            reaches = getattr(self._reach_widget, 'reaches_data', {})
            segment = getattr(self._reach_widget, 'current_segment', 1)
            seg_reaches = []
            if reaches:
                for seg in reaches.get('segments', []):
                    if seg.get('segment_num') == segment:
                        seg_reaches = seg.get('reaches', [])
                        break
            total = len(seg_reaches)
            self.prev_item_btn.setText("<< Prev Reach")
            self.next_item_btn.setText("Next Reach >>")
            self.item_label.setText(f"Reach: {current} / {total}")
        elif tab_idx == 2 and self._outcome_widget:
            # Outcomes (segments)
            current = getattr(self._outcome_widget, 'current_segment', 1)
            total = len(getattr(self._outcome_widget, 'boundaries', [])) - 1
            self.prev_item_btn.setText("<< Prev Segment")
            self.next_item_btn.setText("Next Segment >>")
            self.item_label.setText(f"Segment: {current} / {total}")
        else:
            self.item_label.setText("Item: -- / --")

    def _enable_nav_controls(self, enabled: bool):
        """Enable or disable navigation controls."""
        self.play_btn.setEnabled(enabled)
        self.play_rev_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.back_seg_btn.setEnabled(enabled and len(self.boundaries) > 1)
        self.fwd_seg_btn.setEnabled(enabled and len(self.boundaries) > 1)
        self.prev_item_btn.setEnabled(enabled)
        self.next_item_btn.setEnabled(enabled)


def main():
    """Launch the unified review tool."""
    import argparse

    parser = argparse.ArgumentParser(description="MouseReach Unified Review Tool")
    parser.add_argument('video', nargs='?', type=Path, help="Video file to load")
    parser.add_argument('--tab', choices=['seg', 'reach', 'outcome'], default='seg',
                       help="Initial tab to show")
    args = parser.parse_args()

    viewer = napari.Viewer(title="MouseReach Review Tool")
    widget = UnifiedReviewWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Review", area="right")

    print("\n" + "=" * 50)
    print("MouseReach Unified Review Tool")
    print("=" * 50)
    print("\nCommon shortcuts:")
    print("  Left/Right      - Move 1 frame")
    print("  Shift+Left/Right - Move 10 frames")
    print("\nBoundaries tab: N/P = nav, SPACE = set boundary")
    print("Reaches tab: N/P = nav, S/E = start/end, A = add")
    print("Outcomes tab: R/D/O/U = outcome, I/K = frames")
    print("=" * 50)

    if args.video:
        widget._load_video_from_path(args.video)

    napari.run()


if __name__ == "__main__":
    main()
