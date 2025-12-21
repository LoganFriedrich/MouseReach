"""
ASPA2 Boundary Annotator
========================

Tool for manually annotating segment boundaries (ground truth).

Usage:
    python boundary_annotator.py

Controls:
    - Scrub video with slider
    - Press SPACE or click "Mark Boundary" when at a boundary
    - Press Z to undo last boundary
    - Press S to save
    - Boundaries auto-save when you reach 21

Output:
    Creates *_ground_truth.json next to the video file
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import json

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QListWidget, QMessageBox
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

import napari
from napari.utils.notifications import show_info, show_error


class BoundaryAnnotatorWidget(QWidget):
    """
    Widget for annotating segment boundaries.
    """
    
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.boundaries: List[int] = []
        self.fps = 60.0
        self.video_layer = None
        self.video_path = None
        self.n_frames = 0
        
        self._build_ui()
        self._setup_keybindings()
    
    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # === Instructions ===
        instructions = QLabel(
            "1. Load video\n"
            "2. Scrub to each boundary (where SA moves)\n"
            "3. Press SPACE or click 'Mark Boundary'\n"
            "4. Repeat for all 21 boundaries\n"
            "5. Save when done"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # === File Selection ===
        file_group = QGroupBox("Video")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        self.video_label = QLabel("No video loaded")
        self.video_label.setWordWrap(True)
        file_layout.addWidget(self.video_label)
        
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Video...")
        self.load_btn.clicked.connect(self._load_video)
        btn_layout.addWidget(self.load_btn)
        
        self.load_existing_btn = QPushButton("Load Existing...")
        self.load_existing_btn.clicked.connect(self._load_existing)
        self.load_existing_btn.setEnabled(False)
        btn_layout.addWidget(self.load_existing_btn)
        file_layout.addLayout(btn_layout)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        file_layout.addWidget(self.progress)
        
        layout.addWidget(file_group)
        
        # === Current Position ===
        pos_group = QGroupBox("Position")
        pos_layout = QVBoxLayout()
        pos_group.setLayout(pos_layout)
        
        self.frame_label = QLabel("Frame: -- / --")
        font = QFont()
        font.setPointSize(14)
        self.frame_label.setFont(font)
        pos_layout.addWidget(self.frame_label)
        
        self.time_label = QLabel("Time: --:--")
        pos_layout.addWidget(self.time_label)
        
        layout.addWidget(pos_group)
        
        # === Boundaries ===
        bounds_group = QGroupBox("Boundaries (need 21)")
        bounds_layout = QVBoxLayout()
        bounds_group.setLayout(bounds_layout)
        
        self.bounds_list = QListWidget()
        self.bounds_list.itemDoubleClicked.connect(self._jump_to_boundary)
        bounds_layout.addWidget(self.bounds_list)
        
        self.count_label = QLabel("0 / 21 boundaries marked")
        font = QFont()
        font.setBold(True)
        self.count_label.setFont(font)
        bounds_layout.addWidget(self.count_label)
        
        # Buttons
        mark_layout = QHBoxLayout()
        
        self.mark_btn = QPushButton("Mark Boundary (SPACE)")
        self.mark_btn.clicked.connect(self._mark_boundary)
        self.mark_btn.setEnabled(False)
        mark_layout.addWidget(self.mark_btn)
        
        self.undo_btn = QPushButton("Undo (Z)")
        self.undo_btn.clicked.connect(self._undo_boundary)
        self.undo_btn.setEnabled(False)
        mark_layout.addWidget(self.undo_btn)
        
        bounds_layout.addLayout(mark_layout)
        
        self.save_btn = QPushButton("Save Ground Truth")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        bounds_layout.addWidget(self.save_btn)
        
        layout.addWidget(bounds_group)
        
        # === Compare ===
        compare_group = QGroupBox("Compare to Algorithm")
        compare_layout = QVBoxLayout()
        compare_group.setLayout(compare_layout)
        
        self.compare_btn = QPushButton("Load Algorithm Result...")
        self.compare_btn.clicked.connect(self._load_algorithm)
        self.compare_btn.setEnabled(False)
        compare_layout.addWidget(self.compare_btn)
        
        self.compare_label = QLabel("")
        self.compare_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_label)
        
        layout.addWidget(compare_group)
        
        layout.addStretch()
    
    def _setup_keybindings(self):
        """Set up keyboard shortcuts."""
        @self.viewer.bind_key('Space')
        def mark_boundary_key(viewer):
            self._mark_boundary()
        
        @self.viewer.bind_key('z')
        def undo_key(viewer):
            self._undo_boundary()
        
        @self.viewer.bind_key('s')
        def save_key(viewer):
            self._save()
    
    def _load_video(self):
        """Load a video file."""
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
            cap = cv2.VideoCapture(str(self.video_path))
            
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            
            frames = []
            for i in range(self.n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if i % 500 == 0:
                    self.progress.setValue(int(90 * i / self.n_frames))
                    from qtpy.QtWidgets import QApplication
                    QApplication.processEvents()
            
            cap.release()
            
            # Add to viewer
            if self.video_layer is not None:
                self.viewer.layers.remove(self.video_layer)
            
            self.video_layer = self.viewer.add_image(
                np.stack(frames),
                name=self.video_path.stem,
                rgb=True
            )
            
            # Reset boundaries
            self.boundaries = []
            self._update_bounds_list()
            
            # Enable buttons
            self.mark_btn.setEnabled(True)
            self.load_existing_btn.setEnabled(True)
            self.compare_btn.setEnabled(True)
            
            # Connect frame change
            self.viewer.dims.events.current_step.connect(self._on_frame_change)
            
            self.video_label.setText(f"Loaded: {self.video_path.name}")
            self.progress.setValue(100)
            self._on_frame_change(None)
            
            # Check for existing ground truth
            gt_path = self.video_path.parent / f"{self.video_path.stem}_ground_truth.json"
            if gt_path.exists():
                reply = QMessageBox.question(
                    self, "Existing Annotations",
                    f"Found existing ground truth:\n{gt_path.name}\n\nLoad it?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self._load_ground_truth(gt_path)
            
        except Exception as e:
            show_error(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.progress.setVisible(False)
    
    def _on_frame_change(self, event):
        """Update display when frame changes."""
        frame_idx = self.viewer.dims.current_step[0]
        
        self.frame_label.setText(f"Frame: {frame_idx} / {self.n_frames}")
        
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.time_label.setText(f"Time: {mins}:{secs:05.2f}")
    
    def _mark_boundary(self):
        """Mark current frame as a boundary."""
        if self.video_layer is None:
            return
        
        frame_idx = self.viewer.dims.current_step[0]
        
        # Check if already marked nearby
        for b in self.boundaries:
            if abs(b - frame_idx) < 10:
                show_info(f"Already have boundary at {b} (within 10 frames)")
                return
        
        self.boundaries.append(frame_idx)
        self.boundaries.sort()
        
        self._update_bounds_list()
        self.undo_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        show_info(f"Marked boundary {len(self.boundaries)}/21 at frame {frame_idx}")
        
        if len(self.boundaries) == 21:
            show_info("All 21 boundaries marked! Don't forget to save.")
    
    def _undo_boundary(self):
        """Remove last boundary."""
        if not self.boundaries:
            return
        
        removed = self.boundaries.pop()
        self._update_bounds_list()
        
        show_info(f"Removed boundary at frame {removed}")
        
        if not self.boundaries:
            self.undo_btn.setEnabled(False)
    
    def _update_bounds_list(self):
        """Update the boundaries list widget."""
        self.bounds_list.clear()
        
        for i, b in enumerate(self.boundaries):
            time_sec = b / self.fps
            mins = int(time_sec // 60)
            secs = time_sec % 60
            
            if i == 0:
                label = f"Boundary 1 (end of garbage_pre)"
            elif i == 20:
                label = f"Boundary 21 (start of garbage_post)"
            else:
                label = f"Boundary {i+1} (SA{i} → SA{i+1})"
            
            self.bounds_list.addItem(f"{label}: frame {b} ({mins}:{secs:05.2f})")
        
        self.count_label.setText(f"{len(self.boundaries)} / 21 boundaries marked")
        
        if len(self.boundaries) == 21:
            self.count_label.setStyleSheet("color: green;")
        else:
            self.count_label.setStyleSheet("color: orange;")
    
    def _jump_to_boundary(self, item):
        """Jump to a boundary when double-clicked in list."""
        idx = self.bounds_list.row(item)
        if idx < len(self.boundaries):
            self.viewer.dims.set_current_step(0, self.boundaries[idx])
    
    def _save(self):
        """Save ground truth to JSON."""
        if not self.video_path:
            return
        
        if len(self.boundaries) != 21:
            reply = QMessageBox.question(
                self, "Incomplete",
                f"Only {len(self.boundaries)}/21 boundaries marked.\nSave anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        output_path = self.video_path.parent / f"{self.video_path.stem}_ground_truth.json"
        
        data = {
            "video_name": self.video_path.stem,
            "video_file": self.video_path.name,
            "total_frames": self.n_frames,
            "fps": self.fps,
            "boundaries": self.boundaries,
            "n_boundaries": len(self.boundaries),
            "complete": len(self.boundaries) == 21,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        show_info(f"Saved to {output_path.name}")
    
    def _load_existing(self):
        """Load existing ground truth."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Ground Truth JSON",
            str(self.video_path.parent) if self.video_path else "",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._load_ground_truth(Path(path))
    
    def _load_ground_truth(self, path: Path):
        """Load ground truth from file."""
        with open(path) as f:
            data = json.load(f)
        
        self.boundaries = data.get("boundaries", [])
        self._update_bounds_list()
        
        self.undo_btn.setEnabled(bool(self.boundaries))
        self.save_btn.setEnabled(bool(self.boundaries))
        
        show_info(f"Loaded {len(self.boundaries)} boundaries from {path.name}")
    
    def _load_algorithm(self):
        """Load algorithm result and compare."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Algorithm Segments JSON",
            str(self.video_path.parent) if self.video_path else "",
            "JSON Files (*_segments.json *.json);;All Files (*)"
        )
        if not path:
            return
        
        with open(path) as f:
            data = json.load(f)
        
        algo_bounds = data.get("boundaries", [])
        
        if len(self.boundaries) == 0:
            self.compare_label.setText("Mark ground truth boundaries first!")
            return
        
        # Compare
        n_compare = min(len(self.boundaries), len(algo_bounds))
        errors = []
        for i in range(n_compare):
            diff = algo_bounds[i] - self.boundaries[i]
            errors.append(diff)
        
        if errors:
            mae = np.mean(np.abs(errors))
            max_err = max(errors, key=abs)
            
            self.compare_label.setText(
                f"Compared {n_compare} boundaries:\n"
                f"Mean absolute error: {mae:.1f} frames ({mae/self.fps:.2f}s)\n"
                f"Max error: {max_err} frames ({max_err/self.fps:.2f}s)\n"
                f"Errors: {errors[:5]}..."
            )
        else:
            self.compare_label.setText("No boundaries to compare")


def main():
    """Launch the annotator."""
    viewer = napari.Viewer(title="ASPA2 Boundary Annotator")
    widget = BoundaryAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Annotator", area="right")
    napari.run()


if __name__ == "__main__":
    main()
