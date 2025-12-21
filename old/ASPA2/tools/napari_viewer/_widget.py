"""
ASPA2 Napari Widget
===================

Widget for viewing video segments in napari.
Imports segmentation logic from aspa2_core.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
import json
import sys

# Add parent paths so we can import aspa2_core
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QCheckBox
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

import napari
from napari.utils.notifications import show_info, show_error


def get_segment_info(frame_idx: int, boundaries: List[int]) -> tuple:
    """Get segment index and name for a frame."""
    for i, b in enumerate(boundaries):
        if frame_idx < b:
            if i == 0:
                return 0, "garbage_pre"
            else:
                return i, f"pellet_{i}"
    return 21, "garbage_post"


class SegmentViewerWidget(QWidget):
    """
    Main widget for ASPA2 segment viewing.
    
    Provides file selection, segment loading/computation, and real-time
    segment info display as user scrubs through video.
    """
    
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.boundaries = []
        self.fps = 60.0
        self.video_layer = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # === File Selection Group ===
        file_group = QGroupBox("1. Load Files")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        # Video button
        video_layout = QHBoxLayout()
        self.video_label = QLabel("Video: (none)")
        self.video_label.setWordWrap(True)
        self.video_btn = QPushButton("Select Video...")
        self.video_btn.clicked.connect(self._select_video)
        video_layout.addWidget(self.video_label, stretch=1)
        video_layout.addWidget(self.video_btn)
        file_layout.addLayout(video_layout)
        
        # DLC button (usually auto-detected)
        dlc_layout = QHBoxLayout()
        self.dlc_label = QLabel("DLC: (auto-detect)")
        self.dlc_label.setWordWrap(True)
        self.dlc_btn = QPushButton("Override...")
        self.dlc_btn.clicked.connect(self._select_dlc)
        dlc_layout.addWidget(self.dlc_label, stretch=1)
        dlc_layout.addWidget(self.dlc_btn)
        file_layout.addLayout(dlc_layout)
        
        # Recompute checkbox
        self.recompute_check = QCheckBox("Recompute segments (ignore existing)")
        file_layout.addWidget(self.recompute_check)
        
        # Load button
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self._load_and_compute)
        self.load_btn.setEnabled(False)
        file_layout.addWidget(self.load_btn)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        file_layout.addWidget(self.progress)
        
        layout.addWidget(file_group)
        
        # === Segment Info Group ===
        info_group = QGroupBox("2. Current Segment")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)
        
        # Big segment label
        self.segment_label = QLabel("--")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.segment_label.setFont(font)
        self.segment_label.setAlignment(Qt.AlignCenter)
        self.segment_label.setStyleSheet("color: gray;")
        info_layout.addWidget(self.segment_label)
        
        # Frame/time info
        self.frame_label = QLabel("Frame: -- / --")
        self.frame_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.frame_label)
        
        self.time_label = QLabel("Time: --:--")
        self.time_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.time_label)
        
        layout.addWidget(info_group)
        
        # === Boundaries Group ===
        bounds_group = QGroupBox("3. Navigation")
        bounds_layout = QVBoxLayout()
        bounds_group.setLayout(bounds_layout)
        
        self.bounds_label = QLabel("Load a video to see boundaries")
        self.bounds_label.setWordWrap(True)
        bounds_layout.addWidget(self.bounds_label)
        
        # Jump buttons
        jump_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<< Prev Boundary")
        self.prev_btn.clicked.connect(self._prev_boundary)
        self.prev_btn.setEnabled(False)
        self.next_btn = QPushButton("Next Boundary >>")
        self.next_btn.clicked.connect(self._next_boundary)
        self.next_btn.setEnabled(False)
        jump_layout.addWidget(self.prev_btn)
        jump_layout.addWidget(self.next_btn)
        bounds_layout.addLayout(jump_layout)
        
        layout.addWidget(bounds_group)
        
        # Stretch at bottom
        layout.addStretch()
        
        # Store paths
        self.video_path = None
        self.dlc_path = None
        self.segments_path = None
    
    def _select_video(self):
        """Open file dialog for video, auto-detect DLC file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video",
            "", "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        if path:
            self.video_path = Path(path)
            self.video_label.setText(f"Video: {self.video_path.name}")
            
            # Auto-detect DLC and segments files
            self._auto_detect_files()
            self._check_ready()
    
    def _auto_detect_files(self):
        """Try to find matching DLC and segments files for selected video."""
        if not self.video_path:
            return
        
        video_dir = self.video_path.parent
        video_stem = self.video_path.stem  # e.g., "20250624_CNT0115_P2"
        
        # Look for existing segments file first
        seg_matches = list(video_dir.glob(f"{video_stem}*_segments.json"))
        if seg_matches:
            self.segments_path = seg_matches[0]
        else:
            self.segments_path = None
        
        # Look for DLC files matching this video
        h5_matches = list(video_dir.glob(f"{video_stem}DLC*.h5"))
        csv_matches = list(video_dir.glob(f"{video_stem}DLC*.csv"))
        
        # Prefer .h5 over .csv
        if h5_matches:
            self.dlc_path = h5_matches[0]
        elif csv_matches:
            self.dlc_path = csv_matches[0]
        else:
            # Try looser match
            prefix = video_stem.split('_')[0]
            h5_matches = list(video_dir.glob(f"{prefix}*DLC*.h5"))
            if h5_matches:
                self.dlc_path = h5_matches[0]
            else:
                self.dlc_path = None
        
        # Update labels
        if self.segments_path:
            self.dlc_label.setText(f"Segments: {self.segments_path.name} (found)")
        elif self.dlc_path:
            self.dlc_label.setText(f"DLC: {self.dlc_path.name} (will compute)")
        else:
            self.dlc_label.setText("DLC: (not found - select manually)")
    
    def _select_dlc(self):
        """Open file dialog for DLC file (manual override)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DLC File",
            "", "DLC Files (*.h5 *.csv);;All Files (*)"
        )
        if path:
            self.dlc_path = Path(path)
            self.segments_path = None  # Clear auto-detected segments
            self.dlc_label.setText(f"DLC: {self.dlc_path.name}")
            self._check_ready()
    
    def _check_ready(self):
        """Enable load button if we have what we need."""
        ready = self.video_path is not None and (
            self.segments_path is not None or self.dlc_path is not None
        )
        self.load_btn.setEnabled(ready)
    
    def _load_and_compute(self):
        """Load video and segments."""
        import cv2
        
        self.load_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        try:
            # Get segments - either load existing or compute
            recompute = self.recompute_check.isChecked()
            
            if self.segments_path and not recompute:
                # Load existing segments
                show_info(f"Loading segments from {self.segments_path.name}")
                with open(self.segments_path) as f:
                    seg_data = json.load(f)
                self.boundaries = seg_data['boundaries']
                self.fps = seg_data.get('fps', 60.0)
                confidence = seg_data.get('confidence', 0)
                interval = seg_data.get('interval_frames', 0)
            else:
                # Compute segments from DLC
                if not self.dlc_path:
                    show_error("No DLC file available")
                    return
                
                show_info("Computing segments...")
                from aspa2_core import segment_video
                
                result = segment_video(self.dlc_path)
                self.boundaries = result.boundaries
                self.fps = result.fps
                confidence = result.confidence
                interval = result.interval_frames
            
            self.progress.setValue(20)
            
            # Load video
            show_info(f"Loading video: {self.video_path.name}")
            cap = cv2.VideoCapture(str(self.video_path))
            
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if video_fps > 0:
                self.fps = video_fps
            
            frames = []
            for i in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if i % 500 == 0:
                    pct = 20 + int(70 * i / n_frames)
                    self.progress.setValue(pct)
                    from qtpy.QtWidgets import QApplication
                    QApplication.processEvents()
            
            cap.release()
            
            self.progress.setValue(95)
            
            # Add to viewer
            if self.video_layer is not None:
                self.viewer.layers.remove(self.video_layer)
            
            video_stack = np.stack(frames)
            self.video_layer = self.viewer.add_image(
                video_stack, 
                name=self.video_path.stem,
                rgb=True
            )
            
            # Update UI
            self.bounds_label.setText(
                f"21 boundaries at ~{interval/self.fps:.1f}s intervals\n"
                f"Confidence: {confidence:.2f}"
            )
            
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            
            # Connect to frame changes
            self.viewer.dims.events.current_step.connect(self._on_frame_change)
            
            self.progress.setValue(100)
            show_info(f"Loaded {len(frames)} frames with {len(self.boundaries)} boundaries")
            
            # Trigger initial update
            self._on_frame_change(None)
            
        except Exception as e:
            show_error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.progress.setVisible(False)
            self.load_btn.setEnabled(True)
    
    def _on_frame_change(self, event):
        """Update display when frame changes."""
        if not self.boundaries:
            return
        
        frame_idx = self.viewer.dims.current_step[0]
        seg_idx, seg_name = get_segment_info(frame_idx, self.boundaries)
        
        # Update labels
        if "garbage" in seg_name:
            self.segment_label.setText(seg_name)
            self.segment_label.setStyleSheet("color: gray;")
        else:
            self.segment_label.setText(f"Pellet {seg_idx}")
            self.segment_label.setStyleSheet("color: green;")
        
        n_frames = self.video_layer.data.shape[0] if self.video_layer else 0
        self.frame_label.setText(f"Frame: {frame_idx} / {n_frames}")
        
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.time_label.setText(f"Time: {mins}:{secs:05.2f}")
    
    def _prev_boundary(self):
        """Jump to previous boundary."""
        if not self.boundaries:
            return
        
        current = self.viewer.dims.current_step[0]
        
        for b in reversed(self.boundaries):
            if b < current - 1:
                self.viewer.dims.set_current_step(0, b)
                return
        
        self.viewer.dims.set_current_step(0, 0)
    
    def _next_boundary(self):
        """Jump to next boundary."""
        if not self.boundaries:
            return
        
        current = self.viewer.dims.current_step[0]
        
        for b in self.boundaries:
            if b > current + 1:
                self.viewer.dims.set_current_step(0, b)
                return
