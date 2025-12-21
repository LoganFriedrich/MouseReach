"""
ASPA2 Boundary Annotator v2
===========================

Annotate/correct segment boundaries with DLC point overlay.

Features:
- Shows DLC tracking points on video
- Pre-loads algorithm boundaries (just correct the wrong ones)
- Better frame navigation
- Motion-based jumping

Usage:
    python boundary_annotator_v2.py
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import json
import sys

# Add parent paths so we can import aspa2_core
_script_dir = Path(__file__).parent
_aspa2_root = _script_dir.parent
if str(_aspa2_root) not in sys.path:
    sys.path.insert(0, str(_aspa2_root))

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QSpinBox
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

import napari
from napari.utils.notifications import show_info, show_error
import pandas as pd


class BoundaryAnnotatorWidget(QWidget):
    """
    Widget for annotating/correcting segment boundaries.
    """
    
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.boundaries: List[int] = []
        self.fps = 60.0
        self.video_layer = None
        self.points_layer = None
        self.video_path = None
        self.dlc_path = None
        self.dlc_df = None
        self.n_frames = 0
        self.current_boundary_idx = 0
        
        self._build_ui()
        self._setup_keybindings()
    
    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # === Instructions ===
        instructions = QLabel(
            "Goal: Mark frame where SABL is centered in cage opening.\n\n"
            "1. Load video (auto-loads DLC + algorithm boundaries)\n"
            "2. Use navigation to review each boundary\n"
            "3. Adjust any that are wrong\n"
            "4. Save when done"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # === File Selection ===
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
        
        # === Navigation ===
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
        
        # === Boundaries ===
        bounds_group = QGroupBox("3. Boundaries (21 total)")
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
        
        # List of all boundaries
        self.bounds_list = QListWidget()
        self.bounds_list.itemClicked.connect(self._select_boundary)
        self.bounds_list.itemDoubleClicked.connect(self._jump_to_selected_boundary)
        bounds_layout.addWidget(self.bounds_list)
        
        layout.addWidget(bounds_group)
        
        # === Save ===
        save_group = QGroupBox("4. Save")
        save_layout = QVBoxLayout()
        save_group.setLayout(save_layout)
        
        self.save_btn = QPushButton("Save Ground Truth (S)")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        save_layout.addWidget(self.status_label)
        
        layout.addWidget(save_group)
    
    def _setup_keybindings(self):
        """Set up keyboard shortcuts."""
        @self.viewer.bind_key('Space')
        def set_boundary_key(viewer):
            self._set_current_boundary()
        
        @self.viewer.bind_key('s')
        def save_key(viewer):
            self._save()
        
        @self.viewer.bind_key('Left')
        def left_key(viewer):
            self._jump_frames(-1)
        
        @self.viewer.bind_key('Right')
        def right_key(viewer):
            self._jump_frames(1)
        
        @self.viewer.bind_key('Shift-Left')
        def shift_left_key(viewer):
            self._jump_frames(-10)
        
        @self.viewer.bind_key('Shift-Right')
        def shift_right_key(viewer):
            self._jump_frames(10)
        
        @self.viewer.bind_key('n')
        def next_bound_key(viewer):
            self._next_boundary()
        
        @self.viewer.bind_key('p')
        def prev_bound_key(viewer):
            self._prev_boundary()
    
    def _load_video(self):
        """Load video, DLC data, and algorithm boundaries."""
        import cv2
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video",
            "", "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        if not path:
            return
        
        self.video_path = Path(path)
        self.video_label.setText(f"Loading: {self.video_path.name}")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        try:
            # Find DLC file
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
            
            # Load video
            cap = cv2.VideoCapture(str(self.video_path))
            
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.video_path}")
            
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            
            frames = []
            bad_frames = 0
            for i in range(self.n_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    bad_frames += 1
                    # Use previous frame or black frame
                    if frames:
                        frames.append(frames[-1].copy())
                    continue
                
                try:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                except cv2.error as e:
                    bad_frames += 1
                    if frames:
                        frames.append(frames[-1].copy())
                    continue
                
                if i % 500 == 0:
                    self.progress.setValue(int(80 * i / self.n_frames))
                    from qtpy.QtWidgets import QApplication
                    QApplication.processEvents()
            
            cap.release()
            
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
            self.prev_bound_btn.setEnabled(True)
            self.next_bound_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
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
                        points_data.append([frame_idx, y, x])  # napari uses [t, y, x]
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
            from aspa2_core.segmenter_robust import SEGMENTER_VERSION
            current_version = SEGMENTER_VERSION
        except ImportError:
            current_version = None
        
        # First, try to find pre-computed segments_v2.json (from robust segmenter)
        seg_v2_files = list(self.video_path.parent.glob(f"{self.video_path.stem}*_segments_v2.json"))
        
        if seg_v2_files:
            with open(seg_v2_files[0]) as f:
                data = json.load(f)
            
            file_version = data.get('segmenter_version', '1.0.0')
            
            # Check if outdated
            if current_version and file_version != current_version:
                from napari.utils.notifications import show_warning
                show_warning(f"Segments file is outdated (v{file_version} vs v{current_version}). Consider re-running batch_segment.py")
                self.status_label.setText(f"⚠ OUTDATED v{file_version} - {seg_v2_files[0].name}")
            else:
                conf = data.get('overall_confidence', 0)
                anomalies = data.get('anomalies', [])
                status = f"Loaded {seg_v2_files[0].name} (v{file_version}, conf={conf:.2f})"
                if anomalies:
                    status += f", {len(anomalies)} anomalies"
                self.status_label.setText(status)
                
                if anomalies:
                    from napari.utils.notifications import show_warning
                    for a in anomalies[:3]:
                        show_warning(a)
            
            self.boundaries = data.get('boundaries', [])
            self._update_bounds_list()
            return
        
        # No pre-computed file - try running robust segmenter
        try:
            from aspa2_core.segmenter_robust import segment_video_robust, print_diagnostics
            
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
            
            if diag.anomalies:
                from napari.utils.notifications import show_warning
                for a in diag.anomalies[:3]:
                    show_warning(a)
            
            self._update_bounds_list()
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
        
        self._update_bounds_list()
        self._update_current_boundary_display()
        
        show_info(f"Boundary {self.current_boundary_idx + 1} set to frame {frame_idx}")
    
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
            
            self.bounds_list.addItem(f"{label}: {b} ({mins}:{secs:04.1f})")
    
    def _select_boundary(self, item):
        """Select a boundary from the list."""
        self.current_boundary_idx = self.bounds_list.row(item)
        self._update_current_boundary_display()
    
    def _jump_to_selected_boundary(self, item):
        """Jump to boundary when double-clicked."""
        self.current_boundary_idx = self.bounds_list.row(item)
        self._update_current_boundary_display()
        self._jump_to_current_boundary()
    
    def _save(self):
        """Save ground truth."""
        if not self.video_path:
            return
        
        output_path = self.video_path.parent / f"{self.video_path.stem}_ground_truth.json"
        
        data = {
            "video_name": self.video_path.stem,
            "video_file": self.video_path.name,
            "total_frames": self.n_frames,
            "fps": self.fps,
            "boundaries": self.boundaries,
            "n_boundaries": len(self.boundaries),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.status_label.setText(f"Saved to {output_path.name}")
        show_info(f"Saved to {output_path.name}")


def main():
    """Launch the annotator."""
    viewer = napari.Viewer(title="ASPA2 Boundary Annotator")
    widget = BoundaryAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Annotator", area="right")
    
    print("\nKeyboard shortcuts:")
    print("  SPACE     - Set current boundary to this frame")
    print("  N         - Next boundary")
    print("  P         - Previous boundary")
    print("  S         - Save")
    print("  Left/Right - Move 1 frame")
    print("  Shift+Left/Right - Move 10 frames")
    
    napari.run()


if __name__ == "__main__":
    main()
