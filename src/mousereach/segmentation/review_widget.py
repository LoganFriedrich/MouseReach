"""
MouseReach Boundary Review Tool
===========================

Napari widget for reviewing and correcting segment boundaries.

Features:
- Shows DLC tracking points on video
- Pre-loads algorithm boundaries (just correct the wrong ones)
- Better frame navigation
- Motion-based jumping

Install as plugin:
    pip install -e .
    # Then: Plugins → MouseReach Segmentation → Boundary Review Tool

Or run standalone:
    python -m mousereach_segmentation.review
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import json

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QSpinBox, QScrollArea, QDialog, QTextBrowser
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

import napari
from napari.utils.notifications import show_info, show_warning, show_error
import pandas as pd

from mousereach.review import ComparisonPanel, create_boundary_comparison
from mousereach.review.save_panel import SimpleSavePanel


# Help text for this widget
HELP_TEXT = """
<h2>Step 2: Segmentation Review</h2>

<h3>What This Does</h3>
<p>Reviews and corrects the automatically detected pellet presentation boundaries.</p>

<h3>Normal Workflow</h3>
<ol>
<li><b>Load Video</b> - Select a video file, DLC data loads automatically</li>
<li><b>Navigate</b> - Use buttons or keyboard to move between boundaries</li>
<li><b>Adjust</b> - Press SPACE to set a boundary to the current frame</li>
<li><b>Save Validated</b> - Saves results and updates validation status</li>
</ol>

<h3>Keyboard Shortcuts</h3>
<ul>
<li><b>ENTER</b> - Accept boundary as-is (algorithm is correct)</li>
<li><b>SPACE</b> - Set current boundary to this frame</li>
<li><b>N</b> - Next boundary</li>
<li><b>P</b> - Previous boundary</li>
<li><b>S</b> - Save validated</li>
<li><b>Left/Right</b> - Move 1 frame</li>
<li><b>Shift+Left/Right</b> - Move 10 frames</li>
</ul>

<h3>What is a "Boundary"?</h3>
<p>Each boundary marks where one pellet presentation ends and another begins.
Look for the frame where the SABL (left pellet marker) is centered in the slit.</p>

<h3>Ground Truth (Development Only)</h3>
<p style="color: #888;">The "Save as Ground Truth" button is for developers measuring
algorithm accuracy. Normal users should use "Save Validated" instead.</p>
"""


class BoundaryReviewWidget(QWidget):
    """
    Widget for annotating/correcting segment boundaries.
    """

    def __init__(self, napari_viewer: napari.Viewer, embedded_mode: bool = False):
        super().__init__()
        self.viewer = napari_viewer
        self.embedded_mode = embedded_mode  # When True, hides video load and nav controls
        self.boundaries: List[int] = []
        self.boundaries_reviewed: List[bool] = []  # Track which boundaries have been reviewed
        self.fps = 60.0
        self.video_layer = None
        self.points_layer = None
        self.video_path = None
        self.dlc_path = None
        self.dlc_df = None
        self.n_frames = 0
        self.current_boundary_idx = 0

        # Anomaly annotation support
        self.anomalies = []  # Raw anomaly text from algorithm
        self.anomaly_annotations = {}  # User annotations: {anomaly_text: {cause, notes}}

        self._build_ui()
        if not embedded_mode:
            self._setup_keybindings()

    def _show_help(self):
        """Show help dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Step 2b: Review Boundaries - Help")
        dialog.setMinimumSize(500, 400)
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

        # === Header with Help Button ===
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Step 2b: Review Boundaries</b>")
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
        layout.addLayout(header_layout)

        # === Instructions (shorter now) ===
        if not self.embedded_mode:
            instructions = QLabel(
                "Review and correct pellet presentation boundaries.\n"
                "Click ? for detailed instructions."
            )
            instructions.setWordWrap(True)
            instructions.setStyleSheet("color: #888; padding: 5px;")
            layout.addWidget(instructions)

        # === File Selection (hidden in embedded mode) ===
        if not self.embedded_mode:
            file_group = QGroupBox("1. Load Video")
            file_layout = QVBoxLayout()
            file_group.setLayout(file_layout)

            self.video_label = QLabel("No video loaded")
            self.video_label.setWordWrap(True)
            file_layout.addWidget(self.video_label)

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
            self.load_btn = QPushButton()
            self.progress = QProgressBar()

        # === Navigation (hidden in embedded mode) ===
        if not self.embedded_mode:
            nav_group = QGroupBox("2. Navigate")
            nav_layout = QVBoxLayout()
            nav_group.setLayout(nav_layout)

            # Frame display
            self.frame_label = QLabel("Frame: -- / --")
            font = QFont()
            font.setPointSize(14)
            font.setBold(True)
            self.frame_label.setFont(font)
            nav_layout.addWidget(self.frame_label)

            self.time_label = QLabel("Time: --:--")
            nav_layout.addWidget(self.time_label)

            # Frame jump buttons
            jump_layout = QHBoxLayout()

            self.back_seg_btn = QPushButton("<<< -1 seg")
            self.back_seg_btn.clicked.connect(lambda: self._jump_frames(-1837))
            jump_layout.addWidget(self.back_seg_btn)

            self.back_100_btn = QPushButton("<< -100")
            self.back_100_btn.clicked.connect(lambda: self._jump_frames(-100))
            jump_layout.addWidget(self.back_100_btn)

            self.back_10_btn = QPushButton("< -10")
            self.back_10_btn.clicked.connect(lambda: self._jump_frames(-10))
            jump_layout.addWidget(self.back_10_btn)

            self.fwd_10_btn = QPushButton("+10 >")
            self.fwd_10_btn.clicked.connect(lambda: self._jump_frames(10))
            jump_layout.addWidget(self.fwd_10_btn)

            self.fwd_100_btn = QPushButton("+100 >>")
            self.fwd_100_btn.clicked.connect(lambda: self._jump_frames(100))
            jump_layout.addWidget(self.fwd_100_btn)

            self.fwd_seg_btn = QPushButton("+1 seg >>>")
            self.fwd_seg_btn.clicked.connect(lambda: self._jump_frames(1837))
            jump_layout.addWidget(self.fwd_seg_btn)

            nav_layout.addLayout(jump_layout)

            # Jump to frame
            goto_layout = QHBoxLayout()
            goto_layout.addWidget(QLabel("Go to frame:"))
            self.goto_spin = QSpinBox()
            self.goto_spin.setRange(0, 999999)
            self.goto_spin.valueChanged.connect(self._goto_frame)
            goto_layout.addWidget(self.goto_spin)
            nav_layout.addLayout(goto_layout)

            layout.addWidget(nav_group)
        else:
            # Create dummy attributes for embedded mode
            self.frame_label = QLabel()
            self.time_label = QLabel()
            self.goto_spin = QSpinBox()
        
        # === Boundaries ===
        bounds_title = "Boundaries (21 total)" if self.embedded_mode else "3. Boundaries (21 total)"
        bounds_group = QGroupBox(bounds_title)
        bounds_layout = QVBoxLayout()
        bounds_group.setLayout(bounds_layout)
        
        # Current boundary being edited
        self.current_bound_label = QLabel("Boundary: --")
        font2 = QFont()
        font2.setPointSize(12)
        font2.setBold(True)
        self.current_bound_label.setFont(font2)
        bounds_layout.addWidget(self.current_bound_label)
        
        # Boundary navigation
        bound_nav_layout = QHBoxLayout()
        
        self.prev_bound_btn = QPushButton("<< Prev Boundary")
        self.prev_bound_btn.clicked.connect(self._prev_boundary)
        self.prev_bound_btn.setEnabled(False)
        bound_nav_layout.addWidget(self.prev_bound_btn)
        
        self.next_bound_btn = QPushButton("Next Boundary >>")
        self.next_bound_btn.clicked.connect(self._next_boundary)
        self.next_bound_btn.setEnabled(False)
        bound_nav_layout.addWidget(self.next_bound_btn)
        
        bounds_layout.addLayout(bound_nav_layout)
        
        # Set boundary to current frame
        self.set_bound_btn = QPushButton("Set Boundary HERE (SPACE)")
        self.set_bound_btn.clicked.connect(self._set_current_boundary)
        self.set_bound_btn.setEnabled(False)
        self.set_bound_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        bounds_layout.addWidget(self.set_bound_btn)

        # Accept boundary as-is button
        self.accept_bound_btn = QPushButton("Accept boundary as-is (Enter) - algorithm is correct")
        self.accept_bound_btn.clicked.connect(self._accept_current_boundary)
        self.accept_bound_btn.setEnabled(False)
        self.accept_bound_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.accept_bound_btn.setToolTip("Mark this boundary as reviewed - algorithm result is correct")
        bounds_layout.addWidget(self.accept_bound_btn)

        # Review progress indicator
        self.review_progress_label = QLabel("Reviewed: 0/0 boundaries")
        self.review_progress_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        bounds_layout.addWidget(self.review_progress_label)

        # List of all boundaries
        self.bounds_list = QListWidget()
        self.bounds_list.itemClicked.connect(self._select_boundary)
        self.bounds_list.itemDoubleClicked.connect(self._jump_to_selected_boundary)
        bounds_layout.addWidget(self.bounds_list)
        
        layout.addWidget(bounds_group)

        # === Anomaly Annotations ===
        anomaly_title = "Anomaly Annotations (Optional)" if self.embedded_mode else "4. Anomaly Annotations (Optional)"
        anomaly_group = QGroupBox(anomaly_title)
        anomaly_layout = QVBoxLayout()
        anomaly_group.setLayout(anomaly_layout)

        info_label = QLabel("Annotate what you see for algorithm training:")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        anomaly_layout.addWidget(info_label)

        self.anomaly_list = QListWidget()
        self.anomaly_list.setMaximumHeight(150)
        self.anomaly_list.itemDoubleClicked.connect(self._annotate_anomaly)
        anomaly_layout.addWidget(self.anomaly_list)

        annotate_btn = QPushButton("Annotate Selected Anomaly")
        annotate_btn.clicked.connect(self._annotate_anomaly)
        anomaly_layout.addWidget(annotate_btn)

        layout.addWidget(anomaly_group)

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
        save_title = "Save" if self.embedded_mode else "5. Save"
        save_group = QGroupBox(save_title)
        save_layout = QVBoxLayout()
        save_group.setLayout(save_layout)

        # Use the new clear save panel
        # In embedded mode (Review Tool), hide GT option - Review Tool edits algo files only
        self.save_panel = SimpleSavePanel(review_mode=self.embedded_mode)
        self.save_panel.save_validated.connect(self._save_validated)
        if not self.embedded_mode:
            self.save_panel.save_ground_truth.connect(self._save_ground_truth)
        save_layout.addWidget(self.save_panel)

        # Keep reference to status label for compatibility
        self.status_label = self.save_panel.status_label

        # Keep references to buttons for enable/disable
        self.save_btn = self.save_panel.save_btn
        self.save_gt_btn = self.save_panel.save_gt_btn

        layout.addWidget(save_group)
    
    def _setup_keybindings(self):
        """Set up keyboard shortcuts."""
        @self.viewer.bind_key('Space', overwrite=True)
        def set_boundary_key(viewer):
            self._set_current_boundary()
        
        @self.viewer.bind_key('s', overwrite=True)
        def save_key(viewer):
            self._save()
        
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
        
        @self.viewer.bind_key('n', overwrite=True)
        def next_bound_key(viewer):
            self._next_boundary()
        
        @self.viewer.bind_key('p', overwrite=True)
        def prev_bound_key(viewer):
            self._prev_boundary()

        @self.viewer.bind_key('Return', overwrite=True)
        def accept_bound_key(viewer):
            self._accept_current_boundary()

    def _load_video(self):
        """Load video via file dialog."""
        from mousereach.config import Paths
        # Default to Processing folder if it exists
        default_dir = ""
        for candidate in [
            Paths.SEG_NEEDS_REVIEW,
            Path(__file__).parent.parent.parent.parent / "dev_SampleData",  # Project dev_SampleData
        ]:
            if candidate.exists():
                default_dir = str(candidate)
                break

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video",
            default_dir, "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        if not path:
            return

        self._load_video_from_path(Path(path))

    def _load_video_from_path(self, video_path: Path):
        """Load video, DLC data, and algorithm boundaries from a path."""
        import cv2

        self.video_path = video_path
        self.video_label.setText(f"Loading: {self.video_path.name}")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        try:
            # Find DLC file (use original video stem, not preview)
            video_stem = self.video_path.stem.replace('_preview', '')
            dlc_files = list(self.video_path.parent.glob(f"{video_stem}DLC*.h5"))
            if not dlc_files:
                dlc_files = list(self.video_path.parent.glob(f"{video_stem}DLC*.csv"))

            if dlc_files:
                self.dlc_path = dlc_files[0]
                self._load_dlc()
            else:
                self.dlc_df = None
                show_info("No DLC file found - loading without tracking overlay")

            # Check for compressed preview version (saves memory)
            # Retry with increasing compression if memory errors occur
            preview_path = self.video_path.parent / f"{video_stem}_preview.mp4"
            compression_attempts = 0
            max_attempts = 3

            while compression_attempts < max_attempts:
                try:
                    if '_preview' not in self.video_path.stem:
                        if not preview_path.exists() or compression_attempts > 0:
                            # Create or recreate preview with appropriate compression
                            if compression_attempts == 0:
                                print(f"Creating preview video (one-time): {preview_path.name}")
                                show_info("Creating compressed preview (one-time)...")
                            else:
                                print(f"Recreating with higher compression (attempt {compression_attempts + 1})...")
                                show_info(f"Increasing compression (attempt {compression_attempts + 1})...")
                                # Delete old preview
                                if preview_path.exists():
                                    preview_path.unlink()

                            from mousereach.video_prep.compress import create_preview
                            # Each attempt reduces scale by 10% and increases CRF
                            scale = max(0.3, 0.75 - (compression_attempts * 0.15))
                            crf = min(40, 28 + (compression_attempts * 5))
                            create_preview(self.video_path, scale=scale, crf=crf, overwrite=True)

                        if preview_path.exists():
                            print(f"Using preview video: {preview_path.name}")
                            actual_video = preview_path
                            self.scale_factor = 0.75
                        else:
                            print("Preview creation failed, using original video")
                            actual_video = self.video_path
                            self.scale_factor = 1.0
                    else:
                        actual_video = self.video_path
                        self.scale_factor = 1.0

                    # Load video
                    cap = cv2.VideoCapture(str(actual_video))

                    if not cap.isOpened():
                        raise RuntimeError(f"Could not open video: {self.video_path}")

                    self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Estimate memory needed (frames * height * width * 3 bytes * 2 for safety margin)
                    estimated_mb = (self.n_frames * height * width * 3 * 2) / (1024 * 1024)
                    print(f"Video: {self.n_frames} frames @ {width}x{height}, estimated memory: {estimated_mb:.0f} MB")

                    frames = []
                    bad_frames = 0
                    for i in range(self.n_frames):
                        try:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                bad_frames += 1
                                # Use previous frame or black frame
                                if frames:
                                    frames.append(frames[-1].copy())
                                continue

                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        except (cv2.error, SystemError) as e:
                            bad_frames += 1
                            if frames:
                                frames.append(frames[-1].copy())
                            continue

                        if i % 500 == 0:
                            self.progress.setValue(int(80 * i / self.n_frames))
                            from qtpy.QtWidgets import QApplication
                            QApplication.processEvents()

                    cap.release()
                    # Success - break out of retry loop
                    break

                except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
                    if cap and cap.isOpened():
                        cap.release()
                    compression_attempts += 1
                    if compression_attempts >= max_attempts:
                        raise MemoryError(
                            f"Unable to load video even with maximum compression. "
                            f"Video is too large for available RAM ({estimated_mb:.0f} MB required)."
                        )
                    print(f"Memory error, retrying with higher compression...")
            
            if bad_frames > 0:
                print(f"Warning: {bad_frames} bad frames encountered")
            
            if len(frames) == 0:
                raise RuntimeError("No valid frames could be read from video")
            
            self.n_frames = len(frames)
            self.progress.setValue(85)
            
            # Add video to viewer
            if self.video_layer is not None:
                self.viewer.layers.remove(self.video_layer)
            
            self.video_layer = self.viewer.add_image(
                np.stack(frames),
                name=self.video_path.stem,
                rgb=True
            )
            
            self.progress.setValue(90)
            
            # Add DLC points overlay
            if self.dlc_df is not None:
                self._add_points_layer()
            
            self.progress.setValue(95)
            
            # Load algorithm boundaries or create defaults
            self._load_algorithm_boundaries()
            
            # Enable controls
            self.set_bound_btn.setEnabled(True)
            self.accept_bound_btn.setEnabled(True)
            self.prev_bound_btn.setEnabled(True)
            self.next_bound_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.save_gt_btn.setEnabled(True)
            self.goto_spin.setRange(0, self.n_frames - 1)

            # Connect frame change
            self.viewer.dims.events.current_step.connect(self._on_frame_change)

            self.video_label.setText(f"Loaded: {self.video_path.name}")
            self.progress.setValue(100)
            
            # Go to first boundary
            self.current_boundary_idx = 0
            self._update_current_boundary_display()
            self._jump_to_current_boundary()
            
        except Exception as e:
            show_error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.progress.setVisible(False)

    def _load_data_only(self, video_path: Path):
        """
        Load just the data files (DLC, boundaries) without loading the video.

        Used when the video is already loaded via shared state manager.
        The shared video layer and frame data should already be set on self:
            self._shared_video_layer, self._shared_video_frames,
            self._shared_n_frames, self._shared_video_fps
        """
        self.video_path = video_path
        self.video_label.setText(f"Loading data: {self.video_path.name}")

        try:
            # Use shared video data
            if hasattr(self, '_shared_video_layer') and self._shared_video_layer is not None:
                self.video_layer = self._shared_video_layer
                self.n_frames = self._shared_n_frames
                self.fps = self._shared_video_fps
            else:
                # Fallback: no shared video, can't proceed
                show_error("No video loaded - load a video first")
                return

            # Find and load DLC file
            video_stem = self.video_path.stem
            dlc_files = list(self.video_path.parent.glob(f"{video_stem}DLC*.h5"))
            if not dlc_files:
                dlc_files = list(self.video_path.parent.glob(f"{video_stem}DLC*.csv"))

            if dlc_files:
                self.dlc_path = dlc_files[0]
                self._load_dlc()
            else:
                self.dlc_df = None
                show_info("No DLC file found - loading without tracking overlay")

            # Add DLC points overlay
            if self.dlc_df is not None:
                self._add_points_layer()

            # Load algorithm boundaries
            self._load_algorithm_boundaries()

            # Enable controls
            self.set_bound_btn.setEnabled(True)
            self.accept_bound_btn.setEnabled(True)
            self.prev_bound_btn.setEnabled(True)
            self.next_bound_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.save_gt_btn.setEnabled(True)
            self.goto_spin.setRange(0, self.n_frames - 1)

            # Connect frame change
            self.viewer.dims.events.current_step.connect(self._on_frame_change)

            self.video_label.setText(f"Loaded: {self.video_path.name}")

            # Go to first boundary
            self.current_boundary_idx = 0
            self._update_current_boundary_display()
            self._jump_to_current_boundary()

        except Exception as e:
            show_error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()

    def _load_dlc(self):
        """Load DLC tracking data."""
        if self.dlc_path.suffix == '.h5':
            df = pd.read_hdf(self.dlc_path)
        else:
            df = pd.read_csv(self.dlc_path, header=[0, 1, 2], index_col=0)
        
        # Flatten columns
        df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
        self.dlc_df = df
    
    def _add_points_layer(self):
        """Add all DLC tracking points as napari points layer with distinct colors."""
        if self.dlc_df is None:
            return
        
        # Find all bodyparts
        bodyparts = []
        for col in self.dlc_df.columns:
            if col.endswith('_x'):
                bodyparts.append(col[:-2])  # Remove '_x'
        bodyparts = sorted(set(bodyparts))
        
        if not bodyparts:
            return
        
        # Assign distinct colors to each bodypart
        # Using a colormap that's easy to distinguish
        colors_base = [
            [1, 0, 0],      # red
            [0, 1, 0],      # green
            [0, 0, 1],      # blue
            [1, 1, 0],      # yellow
            [1, 0, 1],      # magenta
            [0, 1, 1],      # cyan
            [1, 0.5, 0],    # orange
            [0.5, 0, 1],    # purple
            [0, 1, 0.5],    # spring green
            [1, 0, 0.5],    # pink
            [0.5, 1, 0],    # lime
            [0, 0.5, 1],    # sky blue
            [1, 0.5, 0.5],  # salmon
            [0.5, 1, 0.5],  # light green
            [0.5, 0.5, 1],  # light blue
            [1, 1, 0.5],    # light yellow
            [1, 0.5, 1],    # light magenta
        ]
        
        bp_colors = {}
        for i, bp in enumerate(bodyparts):
            bp_colors[bp] = colors_base[i % len(colors_base)]
        
        # Collect all points with their colors
        points_data = []
        point_colors = []
        
        for frame_idx in range(len(self.dlc_df)):
            for bp in bodyparts:
                x_col = f'{bp}_x'
                y_col = f'{bp}_y'
                like_col = f'{bp}_likelihood'
                
                if x_col in self.dlc_df.columns and y_col in self.dlc_df.columns:
                    x = self.dlc_df.iloc[frame_idx][x_col]
                    y = self.dlc_df.iloc[frame_idx][y_col]
                    likelihood = self.dlc_df.iloc[frame_idx].get(like_col, 1.0)

                    if likelihood > 0.5 and not np.isnan(x) and not np.isnan(y):
                        # Scale coordinates to match downsampled video
                        scale = getattr(self, 'scale_factor', 1.0)
                        points_data.append([frame_idx, y * scale, x * scale])  # napari uses [t, y, x]
                        point_colors.append(bp_colors[bp] + [0.15])  # Add alpha
        
        if points_data:
            if self.points_layer is not None:
                self.viewer.layers.remove(self.points_layer)
            
            self.points_layer = self.viewer.add_points(
                np.array(points_data),
                name='DLC Points',
                size=6,
                face_color=np.array(point_colors),
            )
            
            # Print legend
            print("\nDLC Point Colors:")
            for bp, color in bp_colors.items():
                r, g, b = [int(c * 255) for c in color]
                print(f"  {bp}: RGB({r}, {g}, {b})")
    
    def _load_algorithm_boundaries(self):
        """Load pre-computed boundaries or run robust segmenter."""
        
        # Get current segmenter version
        try:
            from mousereach.segmentation.core.segmenter_robust import SEGMENTER_VERSION
            current_version = SEGMENTER_VERSION
        except ImportError:
            current_version = None
        
        # Extract video_id from video path (handle both mp4 and h5 loading)
        video_id = self.video_path.stem
        if 'DLC' in video_id:
            video_id = video_id.split('DLC')[0]
            if video_id.endswith('_'):
                video_id = video_id[:-1]
        
        # Try to find segments file (check multiple patterns, priority order)
        # Ground truth files take highest priority - they're the authoritative source
        seg_file = None
        self._using_ground_truth = False
        for pattern in [
            f"{video_id}_seg_ground_truth.json",  # Ground truth (highest priority)
            f"{video_id}_seg_validation.json",    # Human-validated
            f"{video_id}_segments.json",          # Algorithm output
            f"{video_id}_segments_v2.json",       # Old v2 format
        ]:
            candidate = self.video_path.parent / pattern
            if candidate.exists():
                seg_file = candidate
                if 'ground_truth' in pattern:
                    self._using_ground_truth = True
                break

        # Also try glob in case of slight naming differences
        if not seg_file:
            # Check for ground truth first
            gt_files = list(self.video_path.parent.glob(f"{video_id}*_ground_truth.json"))
            if gt_files:
                seg_file = gt_files[0]
                self._using_ground_truth = True
            else:
                seg_files = list(self.video_path.parent.glob(f"{video_id}*_segments*.json"))
                if seg_files:
                    seg_file = seg_files[0]
        
        if seg_file:
            with open(seg_file) as f:
                data = json.load(f)

            # Ground truth files are authoritative - show special status
            if self._using_ground_truth:
                n_bounds = len(data.get('boundaries', []))
                self.status_label.setText(f"✓ GROUND TRUTH - {seg_file.name} ({n_bounds} boundaries)")
                from napari.utils.notifications import show_info
                show_info(f"Using ground truth boundaries from {seg_file.name}")
            else:
                file_version = data.get('segmenter_version', '1.0.0')

                # Check if outdated
                if current_version and file_version != current_version:
                    from napari.utils.notifications import show_warning
                    show_warning(f"Segments file is outdated (v{file_version} vs v{current_version}). Consider re-running batch_segment.py")
                    self.status_label.setText(f"⚠ OUTDATED v{file_version} - {seg_file.name}")
                else:
                    conf = data.get('overall_confidence', 0)
                    anomalies = data.get('anomalies', [])
                    status = f"Loaded {seg_file.name} (v{file_version}, conf={conf:.2f})"
                    if anomalies:
                        status += f", {len(anomalies)} anomalies"
                    self.status_label.setText(status)

                    if anomalies:
                        from napari.utils.notifications import show_warning
                        for a in anomalies[:3]:
                            show_warning(a)
            
            self.boundaries = data.get('boundaries', [])
            self._original_boundaries = self.boundaries.copy()  # Store for change tracking
            self._algorithm_version = data.get('segmenter_version')  # Store for validation record

            # Load anomalies and any existing annotations
            self.anomalies = data.get('anomalies', [])
            self.anomaly_annotations = data.get('anomaly_annotations', {})

            self._update_bounds_list()
            self._update_anomaly_list()
            return
        
        # No pre-computed file - try running robust segmenter
        try:
            from mousereach.segmentation.core.segmenter_robust import segment_video_robust, print_diagnostics
            
            print(f"\nNo pre-computed segments found, running robust segmenter...")
            boundaries, diag = segment_video_robust(self.dlc_path)
            self.boundaries = boundaries

            print_diagnostics(diag)

            confidence = np.mean(diag.boundary_confidences)
            n_anomalies = len(diag.anomalies)

            status = f"Computed: {diag.n_primary_candidates} detections, conf={confidence:.2f}"
            if n_anomalies > 0:
                status += f", {n_anomalies} anomalies"
            self.status_label.setText(status)

            # Load anomalies from diagnostics
            self.anomalies = diag.anomalies
            self.anomaly_annotations = {}

            if diag.anomalies:
                from napari.utils.notifications import show_warning
                for a in diag.anomalies[:3]:
                    show_warning(a)

            self._update_bounds_list()
            self._update_anomaly_list()
            return
            
        except ImportError:
            pass
        except Exception as e:
            print(f"Robust segmenter failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback: try old segments file
        seg_files = list(self.video_path.parent.glob(f"{self.video_path.stem}*_segments.json"))
        
        if seg_files:
            with open(seg_files[0]) as f:
                data = json.load(f)
            self.boundaries = data.get('boundaries', [])
            self.status_label.setText(f"⚠ OLD FORMAT - {seg_files[0].name}")
            from napari.utils.notifications import show_warning
            show_warning("Using old segments format. Run batch_segment.py to upgrade.")
        else:
            # Last resort: evenly spaced
            interval = self.n_frames / 22
            self.boundaries = [int((i + 1) * interval) for i in range(21)]
            self.status_label.setText("Using evenly spaced defaults")
        
        while len(self.boundaries) < 21:
            self.boundaries.append(self.n_frames - 100)
        self.boundaries = self.boundaries[:21]
        
        # Store original for change tracking
        self._original_boundaries = self.boundaries.copy()
        
        self._update_bounds_list()
    
    def _on_frame_change(self, event):
        """Update display when frame changes."""
        frame_idx = self.viewer.dims.current_step[0]
        
        self.frame_label.setText(f"Frame: {frame_idx} / {self.n_frames}")
        
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.time_label.setText(f"Time: {mins}:{secs:05.2f}")
        
        # Don't update spinbox if it triggered this (avoid loop)
        if self.goto_spin.value() != frame_idx:
            self.goto_spin.blockSignals(True)
            self.goto_spin.setValue(frame_idx)
            self.goto_spin.blockSignals(False)
    
    def _jump_frames(self, delta: int):
        """Jump forward/backward by delta frames."""
        current = self.viewer.dims.current_step[0]
        new_frame = max(0, min(self.n_frames - 1, current + delta))
        self.viewer.dims.set_current_step(0, new_frame)
    
    def _jump_seconds(self, seconds: float):
        """Jump forward/backward by seconds."""
        delta_frames = int(seconds * self.fps)
        self._jump_frames(delta_frames)
    
    def _goto_frame(self, frame: int):
        """Jump to specific frame."""
        self.viewer.dims.set_current_step(0, frame)
    
    def _prev_boundary(self):
        """Go to previous boundary."""
        if self.current_boundary_idx > 0:
            self.current_boundary_idx -= 1
            self._update_current_boundary_display()
            self._jump_to_current_boundary()
    
    def _next_boundary(self):
        """Go to next boundary."""
        if self.current_boundary_idx < 20:
            self.current_boundary_idx += 1
            self._update_current_boundary_display()
            self._jump_to_current_boundary()
    
    def goto_boundary(self, boundary_idx: int):
        """Jump to a specific boundary (public API for external callers).

        Args:
            boundary_idx: Index of boundary (0-20)
        """
        if boundary_idx < 0 or boundary_idx > 20:
            return
        if not self.boundaries or boundary_idx >= len(self.boundaries):
            return

        self.current_boundary_idx = boundary_idx
        self._update_current_boundary_display()
        self._jump_to_current_boundary()

    def _jump_to_current_boundary(self):
        """Jump viewer to current boundary frame."""
        if self.boundaries and self.current_boundary_idx < len(self.boundaries):
            self.viewer.dims.set_current_step(0, self.boundaries[self.current_boundary_idx])
    
    def _set_current_boundary(self):
        """Set the current boundary to the current frame."""
        if not self.boundaries:
            return

        frame_idx = self.viewer.dims.current_step[0]
        self.boundaries[self.current_boundary_idx] = frame_idx

        # Mark as reviewed when changed
        if self.current_boundary_idx < len(self.boundaries_reviewed):
            self.boundaries_reviewed[self.current_boundary_idx] = True

        self._update_bounds_list()
        self._update_current_boundary_display()

        show_info(f"Boundary {self.current_boundary_idx + 1} set to frame {frame_idx}")

    def _accept_current_boundary(self):
        """Accept current boundary as-is (algorithm is correct).

        This provides explicit confirmation that the user reviewed this boundary
        and agrees with the algorithm's detection, without making changes.
        """
        if not self.boundaries:
            return

        idx = self.current_boundary_idx
        if idx < len(self.boundaries_reviewed):
            self.boundaries_reviewed[idx] = True

        self._update_bounds_list()
        self._update_current_boundary_display()

        # Auto-advance to next unreviewed boundary
        next_unreviewed = self._find_next_unreviewed_boundary()
        if next_unreviewed is not None:
            self.current_boundary_idx = next_unreviewed
            self._update_current_boundary_display()
            self._jump_to_current_boundary()
            show_info(f"Boundary {idx + 1} accepted. Moving to boundary {next_unreviewed + 1}...")
        else:
            show_info(f"Boundary {idx + 1} accepted. All boundaries reviewed!")

    def _find_next_unreviewed_boundary(self) -> Optional[int]:
        """Find the next boundary that hasn't been reviewed yet."""
        if not self.boundaries_reviewed:
            return None

        # First check boundaries after current
        for i in range(self.current_boundary_idx + 1, len(self.boundaries_reviewed)):
            if not self.boundaries_reviewed[i]:
                return i

        # Then wrap around and check from beginning
        for i in range(0, self.current_boundary_idx):
            if not self.boundaries_reviewed[i]:
                return i

        return None

    def _update_current_boundary_display(self):
        """Update the current boundary label."""
        idx = self.current_boundary_idx
        
        if idx == 0:
            desc = "End of garbage_pre / Start of SA1"
        elif idx == 20:
            desc = "End of SA20 / Start of garbage_post"
        else:
            desc = f"SA{idx} → SA{idx + 1}"
        
        self.current_bound_label.setText(f"Boundary {idx + 1}/21: {desc}")
        
        # Highlight in list
        self.bounds_list.setCurrentRow(idx)
    
    def _update_bounds_list(self):
        """Update the boundaries list widget."""
        self.bounds_list.clear()

        # Initialize boundaries_reviewed if needed
        if len(self.boundaries_reviewed) != len(self.boundaries):
            self.boundaries_reviewed = [False] * len(self.boundaries)

        # Calculate review progress
        total = len(self.boundaries)
        reviewed = sum(self.boundaries_reviewed)

        # Update progress label with color coding
        if total == 0:
            self.review_progress_label.setText("Reviewed: 0/0 boundaries")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        elif reviewed == total:
            self.review_progress_label.setText(f"Reviewed: {reviewed}/{total} - ALL DONE!")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif reviewed > 0:
            self.review_progress_label.setText(f"Reviewed: {reviewed}/{total} boundaries")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        else:
            self.review_progress_label.setText(f"Reviewed: {reviewed}/{total} boundaries")
            self.review_progress_label.setStyleSheet("font-weight: bold; color: #2196F3;")

        from qtpy.QtGui import QColor

        for i, b in enumerate(self.boundaries):
            time_sec = b / self.fps
            mins = int(time_sec // 60)
            secs = time_sec % 60

            if i == 0:
                label = f"B1 (→SA1)"
            elif i == 20:
                label = f"B21 (SA20→)"
            else:
                label = f"B{i+1} (SA{i}→{i+1})"

            # Show review status
            is_reviewed = self.boundaries_reviewed[i] if i < len(self.boundaries_reviewed) else False
            status_icon = "✓" if is_reviewed else "○"

            from qtpy.QtWidgets import QListWidgetItem
            item = QListWidgetItem(f"{status_icon} {label}: {b} ({mins}:{secs:04.1f})")

            # Color code: reviewed = green, not reviewed = default
            if is_reviewed:
                item.setForeground(QColor('#4CAF50'))

            self.bounds_list.addItem(item)

        # Update comparison panel
        self._update_comparison_panel()
    
    def _select_boundary(self, item):
        """Select a boundary from the list."""
        self.current_boundary_idx = self.bounds_list.row(item)
        self._update_current_boundary_display()
    
    def _jump_to_selected_boundary(self, item):
        """Jump to boundary when double-clicked."""
        self.current_boundary_idx = self.bounds_list.row(item)
        self._update_current_boundary_display()
        self._jump_to_current_boundary()

    def _on_comparison_item_selected(self, index: int):
        """Handle selection in comparison panel - jump to that boundary."""
        if 0 <= index < len(self.boundaries):
            self.current_boundary_idx = index
            self._update_current_boundary_display()
            self._jump_to_current_boundary()

    def _update_comparison_panel(self):
        """Update the comparison panel with current boundaries vs original."""
        if not hasattr(self, 'comparison_panel') or not self.boundaries:
            return

        # Get original boundaries (algorithm output)
        original = getattr(self, '_original_boundaries', self.boundaries)

        # Check if GT file exists
        gt_exists = getattr(self, '_using_ground_truth', False)

        # Create comparison items
        items = create_boundary_comparison(original, self.boundaries)
        self.comparison_panel.set_items(items, gt_exists=gt_exists)

    def _save(self):
        """Save as validated output (for pipeline use)."""
        self._save_validated()
    
    def _save_validated(self):
        """Save corrected boundaries as validated output for pipeline.

        New architecture (v2.3+): Files stay in Processing/, validation status
        is set in JSON metadata. No folder-based triage.
        """
        if not self.video_path:
            return

        import os
        from datetime import datetime

        # Get username
        try:
            username = os.getlogin()
        except (OSError, AttributeError):
            username = os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))

        # Determine video_id (strip DLC suffix if present)
        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        # Load original boundaries for change tracking
        original_boundaries = getattr(self, '_original_boundaries', self.boundaries)

        # Build detailed change list and per-boundary correction tracking
        changes = []
        boundary_corrections = {}
        timestamp = datetime.now().isoformat()

        for i, (orig, curr) in enumerate(zip(original_boundaries, self.boundaries)):
            if orig != curr:
                changes.append({
                    "index": i,
                    "original": orig,
                    "corrected": curr,
                    "delta": curr - orig
                })
                # Per-boundary correction tracking (like reaches/outcomes have)
                boundary_corrections[str(i)] = {
                    "human_corrected": True,
                    "original_frame": orig,
                    "corrected_by": username,
                    "corrected_at": timestamp
                }
            else:
                # Boundary unchanged - mark as not corrected
                boundary_corrections[str(i)] = {
                    "human_corrected": False,
                    "original_frame": None,
                    "corrected_by": None,
                    "corrected_at": None
                }

        # Build validation_record (audit trail embedded in main JSON)
        validation_record = {
            "validated_by": username,
            "validated_at": timestamp,
            "algorithm_version": getattr(self, '_algorithm_version', None),
            "original_boundaries": original_boundaries,
            "changes_made": changes,
            "total_items": len(self.boundaries),
            "items_changed": len(changes)
        }

        # Save updated segments file (overwrites algorithm output)
        segments_path = self.video_path.parent / f"{video_id}_segments.json"

        # Load existing data if present (to preserve algorithm diagnostics)
        existing_data = {}
        if segments_path.exists():
            try:
                with open(segments_path, 'r') as f:
                    existing_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        # Update with validated info
        existing_data.update({
            "video_name": video_id,
            "boundaries": self.boundaries,
            "n_boundaries": len(self.boundaries),
            "boundary_corrections": boundary_corrections,  # Per-boundary human correction tracking
            "validation_status": "validated",  # KEY: marks as human-reviewed
            "validation_record": validation_record,  # Embedded audit trail
        })

        with open(segments_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

        # Update pipeline index
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_validation_changed(video_id, "seg", "validated")
            index.save()
        except Exception:
            pass  # Silent fail - index will catch up on next rebuild

        # Log performance metrics (algorithm vs human comparison)
        try:
            from mousereach.performance import PerformanceLogger
            PerformanceLogger().log_segmentation(
                video_id=video_id,
                algo_boundaries=original_boundaries,
                human_boundaries=self.boundaries,
                algo_version=getattr(self, '_algorithm_version', None),
                validator=username
            )
        except Exception:
            pass  # Don't block validation if logging fails

        changes_text = f"{len(changes)} boundaries corrected" if changes else "no changes"
        self.status_label.setText(f"Validated: {video_id} ({changes_text})")
        show_info(f"Saved validated segmentation ({changes_text})")

    def _save_ground_truth(self):
        """Save as ground truth (for dev/accuracy measurement only).

        GT files only contain items the human explicitly interacted with.
        Algorithm-seeded items the human never touched are NOT saved.
        This makes GT files true training data - only human-provided examples.

        A boundary is included in GT if:
        - boundaries_reviewed[i] == True (user explicitly accepted as correct), OR
        - The boundary was corrected (frame differs from original)
        """
        if not self.video_path:
            return

        import os
        from datetime import datetime

        try:
            username = os.getlogin()
        except (OSError, AttributeError):
            username = os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))

        video_id = self.video_path.stem
        if 'DLC_' in video_id:
            video_id = video_id.split('DLC_')[0]

        output_path = self.video_path.parent / f"{video_id}_seg_ground_truth.json"

        # Track verification status
        # An anomaly is verified if it has an annotation
        total_anomalies = len(self.anomalies)
        verified_anomalies = len(self.anomaly_annotations)

        # Get original boundaries for comparison if available
        original_boundaries = getattr(self, '_original_boundaries', self.boundaries)

        # Build boundaries with verification metadata - ONLY for human-interacted items
        boundaries_with_meta = []
        gt_boundary_frames = []  # Simple list for backwards compat

        for i, frame in enumerate(self.boundaries):
            was_corrected = False
            if i < len(original_boundaries):
                was_corrected = (frame != original_boundaries[i])

            # Check if user explicitly reviewed/accepted this boundary
            was_reviewed = False
            if i < len(self.boundaries_reviewed):
                was_reviewed = self.boundaries_reviewed[i]

            # Only include in GT if human interacted with this boundary
            if was_reviewed or was_corrected:
                boundaries_with_meta.append({
                    'index': i,
                    'frame': frame,
                    'human_corrected': was_corrected,
                    'human_verified': True,  # Human interacted = verified
                    'verified_by': username,
                    'verified_at': datetime.now().isoformat(),
                    'original_frame': original_boundaries[i] if was_corrected and i < len(original_boundaries) else None,
                })
                gt_boundary_frames.append(frame)

        # Warn if no human-interacted items
        if not boundaries_with_meta:
            show_warning("No boundaries were reviewed or corrected. GT file will be empty.")

        # GT completeness: based on what's actually in the GT file
        # (different from anomaly-based completeness which is for the whole video)
        gt_complete = len(boundaries_with_meta) > 0

        data = {
            "video_name": video_id,
            "type": "ground_truth",
            "created_by": username,
            "created_at": datetime.now().isoformat(),
            "boundaries": gt_boundary_frames,  # Only human-verified boundary frames
            "boundaries_with_meta": boundaries_with_meta,  # Only human-interacted items
            "n_boundaries": len(boundaries_with_meta),
            "total_frames": self.n_frames,
            "fps": self.fps,
            "anomalies": self.anomalies,
            "anomaly_annotations": self.anomaly_annotations,
            "annotation_notes": "",
            # Completeness tracking
            "gt_complete": gt_complete,
            "total_anomalies": total_anomalies,
            "verified_anomalies": verified_anomalies,
            "human_verified": gt_complete,  # Legacy compatibility
            # Track what portion of the video was annotated
            "total_boundaries_in_video": len(self.boundaries),
            "boundaries_annotated": len(boundaries_with_meta),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Update pipeline index with GT status
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_gt_created(video_id, "seg", is_complete=gt_complete)
            index.save()
        except Exception:
            pass  # Silent fail - index will catch up on next rebuild

        # Status message showing what was saved
        status_text = f"{len(boundaries_with_meta)}/{len(self.boundaries)} boundaries"
        self.status_label.setText(f"Saved GT: {output_path.name} ({status_text})")
        show_info(f"Saved ground truth: {status_text}")

    def _update_anomaly_list(self):
        """Update the anomaly list widget with current anomalies."""
        self.anomaly_list.clear()

        if not self.anomalies:
            self.anomaly_list.addItem("(No anomalies detected)")
            return

        for anom in self.anomalies:
            # Check if this anomaly has an annotation
            if anom in self.anomaly_annotations:
                annot = self.anomaly_annotations[anom]
                display = f"✓ {anom[:60]}... [{annot.get('cause', 'Unknown')}]"
            else:
                display = f"  {anom}"
            self.anomaly_list.addItem(display)

    def _annotate_anomaly(self):
        """Open dialog to annotate the selected anomaly."""
        from qtpy.QtWidgets import QComboBox, QTextEdit, QDialogButtonBox

        selected_items = self.anomaly_list.selectedItems()
        if not selected_items:
            show_error("Please select an anomaly to annotate")
            return

        idx = self.anomaly_list.row(selected_items[0])
        if idx >= len(self.anomalies):
            return

        anomaly_text = self.anomalies[idx]

        # Create annotation dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Annotate Anomaly")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout()
        dialog.setLayout(layout)

        # Show anomaly text
        layout.addWidget(QLabel("<b>Anomaly:</b>"))
        anom_label = QLabel(anomaly_text)
        anom_label.setWordWrap(True)
        anom_label.setStyleSheet("background-color: #333; padding: 10px; border-radius: 5px;")
        layout.addWidget(anom_label)

        # Cause selection
        layout.addWidget(QLabel("<b>What caused this?</b>"))
        cause_combo = QComboBox()
        cause_combo.addItems([
            "Unknown",
            "Stuck tray (manual advance)",
            "Stuck tray (mouse freed it)",
            "Operator paused session",
            "Operator adjusted tray",
            "Mouse hesitated/delayed",
            "Early session termination",
            "Equipment malfunction (tray too fast)",
            "Equipment malfunction (tray too slow)",
            "Normal variation",
            "Boundaries correct (not an error)",
            "Boundaries incorrect (algorithm error)"
        ])

        # Pre-fill if already annotated
        if anomaly_text in self.anomaly_annotations:
            existing_cause = self.anomaly_annotations[anomaly_text].get('cause', 'Unknown')
            idx_to_set = cause_combo.findText(existing_cause)
            if idx_to_set >= 0:
                cause_combo.setCurrentIndex(idx_to_set)

        layout.addWidget(cause_combo)

        # Notes field
        layout.addWidget(QLabel("<b>Additional notes:</b>"))
        notes_edit = QTextEdit()
        notes_edit.setMaximumHeight(100)
        notes_edit.setPlaceholderText("Optional: Describe what you see in the video...")

        # Pre-fill notes if already annotated
        if anomaly_text in self.anomaly_annotations:
            existing_notes = self.anomaly_annotations[anomaly_text].get('notes', '')
            notes_edit.setPlainText(existing_notes)

        layout.addWidget(notes_edit)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_():
            # Save annotation
            self.anomaly_annotations[anomaly_text] = {
                'cause': cause_combo.currentText(),
                'notes': notes_edit.toPlainText()
            }
            self._update_anomaly_list()
            show_info(f"Annotation saved for: {anomaly_text[:50]}...")


def main():
    """Launch the review tool."""
    viewer = napari.Viewer(title="MouseReach Boundary Review Tool")
    widget = BoundaryReviewWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Review", area="right")
    
    print("\nKeyboard shortcuts:")
    print("  SPACE     - Set current boundary to this frame")
    print("  N         - Next boundary")
    print("  P         - Previous boundary")
    print("  S         - Save validated (for pipeline)")
    print("  Left/Right - Move 1 frame")
    print("  Shift+Left/Right - Move 10 frames")
    
    napari.run()


if __name__ == "__main__":
    main()
