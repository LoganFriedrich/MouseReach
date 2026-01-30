"""
MouseReach Pillar Geometry Annotator
================================

Napari widget for annotating the true pillar position in videos.

Purpose:
- Validate geometric pillar calculation
- Learn corrections to improve algorithm
- Determine true "on pillar" threshold distance

Workflow:
1. Load video (auto-loads DLC data)
2. Navigate to key frames (segment starts, tray movements, interactions)
3. Algorithm auto-places circle at calculated pillar position
4. User adjusts circle position/size if wrong
5. Save annotations with auto vs manual comparison

Output: JSON file with pillar geometry ground truth
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json
import os
from datetime import datetime

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QListWidget, QMessageBox,
    QListWidgetItem, QLineEdit, QTextEdit, QCheckBox
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

import napari
from napari.utils.notifications import show_info, show_warning, show_error
import pandas as pd
import cv2


def get_username():
    """Get current username."""
    try:
        return os.getlogin()
    except (OSError, AttributeError):
        return os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))


class PillarGeometryAnnotatorWidget(QWidget):
    """
    Widget for annotating true pillar geometry.

    Auto-populates circle based on geometric calculation,
    user corrects if needed, tracks corrections for algorithm improvement.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Data
        self.video_path = None
        self.dlc_path = None
        self.segments_path = None
        self.dlc_df = None
        self.boundaries = []

        # State
        self.n_frames = 0
        self.fps = 60.0
        self.current_frame = 0
        self.ruler_length = None  # Pixels (SABL-SABR distance)

        # Annotations
        self.pillar_annotations = []  # List of {frame, auto_x, auto_y, auto_r, user_x, user_y, user_r, modified, ...}
        self.current_annotation_idx = None

        # Layers
        self.video_layer = None
        self.pillar_circle_layer = None  # Shapes layer for pillar circle
        self.sa_points_layer = None  # Points layer for SABL/SABR reference

        self._build_ui()
        self._setup_keybindings()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)

        # === Instructions ===
        instructions = QLabel(
            "Pillar Geometry Annotator\n\n"
            "1. Load video\n"
            "2. Navigate to key frames\n"
            "3. Adjust circle if algorithm is wrong\n"
            "4. Save annotations\n\n"
            "Arrow keys = move circle, +/- = resize"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background-color: #f0f0f0; padding: 8px; border-radius: 4px;")
        layout.addWidget(instructions)

        # === File Loading ===
        file_group = QGroupBox("1. Load Video")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        self.video_label = QLabel("No video loaded")
        self.video_label.setWordWrap(True)
        file_layout.addWidget(self.video_label)

        self.load_btn = QPushButton("Select Video...")
        self.load_btn.clicked.connect(self._load_video)
        file_layout.addWidget(self.load_btn)

        layout.addWidget(file_group)

        # === Frame Navigation ===
        nav_group = QGroupBox("2. Navigate to Key Frames")
        nav_layout = QVBoxLayout()
        nav_group.setLayout(nav_layout)

        self.frame_label = QLabel("Frame: --")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.frame_label.setFont(font)
        nav_layout.addWidget(self.frame_label)

        # Quick navigation buttons
        nav_btns_layout = QHBoxLayout()

        self.prev_seg_btn = QPushButton("◀ Prev Segment")
        self.prev_seg_btn.clicked.connect(self._goto_prev_segment)
        self.prev_seg_btn.setEnabled(False)
        nav_btns_layout.addWidget(self.prev_seg_btn)

        self.next_seg_btn = QPushButton("Next Segment ▶")
        self.next_seg_btn.clicked.connect(self._goto_next_segment)
        self.next_seg_btn.setEnabled(False)
        nav_btns_layout.addWidget(self.next_seg_btn)

        nav_layout.addLayout(nav_btns_layout)

        self.annotation_note = QLineEdit()
        self.annotation_note.setPlaceholderText("Note about this frame (e.g., 'pellet on pillar', 'after tray move')")
        nav_layout.addWidget(QLabel("Annotation Note:"))
        nav_layout.addWidget(self.annotation_note)

        layout.addWidget(nav_group)

        # === Circle Adjustment ===
        circle_group = QGroupBox("3. Adjust Pillar Circle")
        circle_layout = QVBoxLayout()
        circle_group.setLayout(circle_layout)

        self.circle_info_label = QLabel("Circle: Auto-positioned")
        circle_layout.addWidget(self.circle_info_label)

        adjustment_help = QLabel(
            "• Drag circle to move\n"
            "• Drag edge to resize\n"
            "• Arrow keys for fine adjustment\n"
            "• +/- keys to resize"
        )
        adjustment_help.setStyleSheet("font-size: 10px; color: #666;")
        circle_layout.addWidget(adjustment_help)

        save_frame_layout = QHBoxLayout()

        self.save_frame_btn = QPushButton("✓ Save Current Frame")
        self.save_frame_btn.clicked.connect(self._save_current_frame)
        self.save_frame_btn.setEnabled(False)
        save_frame_layout.addWidget(self.save_frame_btn)

        self.reset_circle_btn = QPushButton("↺ Reset to Auto")
        self.reset_circle_btn.clicked.connect(self._reset_circle_to_auto)
        self.reset_circle_btn.setEnabled(False)
        save_frame_layout.addWidget(self.reset_circle_btn)

        circle_layout.addLayout(save_frame_layout)

        layout.addWidget(circle_group)

        # === Annotations List ===
        list_group = QGroupBox("4. Saved Annotations")
        list_layout = QVBoxLayout()
        list_group.setLayout(list_layout)

        self.annotations_list = QListWidget()
        self.annotations_list.itemClicked.connect(self._on_annotation_selected)
        list_layout.addWidget(self.annotations_list)

        list_btns_layout = QHBoxLayout()

        self.delete_annotation_btn = QPushButton("🗑 Delete")
        self.delete_annotation_btn.clicked.connect(self._delete_current_annotation)
        self.delete_annotation_btn.setEnabled(False)
        list_btns_layout.addWidget(self.delete_annotation_btn)

        list_layout.addLayout(list_btns_layout)

        layout.addWidget(list_group)

        # === Save/Export ===
        save_group = QGroupBox("5. Save Annotations")
        save_layout = QVBoxLayout()
        save_group.setLayout(save_layout)

        self.save_count_label = QLabel("0 annotations saved")
        save_layout.addWidget(self.save_count_label)

        self.save_all_btn = QPushButton("💾 Save All Annotations")
        self.save_all_btn.clicked.connect(self._save_annotations)
        self.save_all_btn.setEnabled(False)
        save_layout.addWidget(self.save_all_btn)

        layout.addWidget(save_group)

        layout.addStretch()

    def _setup_keybindings(self):
        """Setup keyboard shortcuts."""
        @self.viewer.bind_key('Left')
        def move_circle_left(viewer):
            if self.pillar_circle_layer and len(self.pillar_circle_layer.data) > 0:
                self._nudge_circle(-1, 0)

        @self.viewer.bind_key('Right')
        def move_circle_right(viewer):
            if self.pillar_circle_layer and len(self.pillar_circle_layer.data) > 0:
                self._nudge_circle(1, 0)

        @self.viewer.bind_key('Up')
        def move_circle_up(viewer):
            if self.pillar_circle_layer and len(self.pillar_circle_layer.data) > 0:
                self._nudge_circle(0, -1)

        @self.viewer.bind_key('Down')
        def move_circle_down(viewer):
            if self.pillar_circle_layer and len(self.pillar_circle_layer.data) > 0:
                self._nudge_circle(0, 1)

        @self.viewer.bind_key('+')
        @self.viewer.bind_key('=')
        def increase_radius(viewer):
            if self.pillar_circle_layer and len(self.pillar_circle_layer.data) > 0:
                self._resize_circle(1)

        @self.viewer.bind_key('-')
        def decrease_radius(viewer):
            if self.pillar_circle_layer and len(self.pillar_circle_layer.data) > 0:
                self._resize_circle(-1)

        @self.viewer.bind_key('s')
        def save_frame(viewer):
            if self.save_frame_btn.isEnabled():
                self._save_current_frame()

        @self.viewer.bind_key('n')
        def next_segment(viewer):
            if self.next_seg_btn.isEnabled():
                self._goto_next_segment()

        @self.viewer.bind_key('p')
        def prev_segment(viewer):
            if self.prev_seg_btn.isEnabled():
                self._goto_prev_segment()

    def _load_video(self):
        """Load video and associated DLC/segments data."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )

        if not file_path:
            return

        self.video_path = Path(file_path)
        video_name = self.video_path.stem
        video_dir = self.video_path.parent

        # Find DLC and segments files
        dlc_pattern = f"{video_name}DLC*.h5"
        dlc_files = list(video_dir.glob(dlc_pattern))

        if not dlc_files:
            show_error(f"No DLC file found matching: {dlc_pattern}")
            return

        self.dlc_path = dlc_files[0]

        seg_pattern = f"{video_name}_segments.json"
        seg_files = list(video_dir.glob(seg_pattern))

        if not seg_files:
            show_error(f"No segments file found: {seg_pattern}")
            return

        self.segments_path = seg_files[0]

        # Load data
        try:
            self._do_load()
        except Exception as e:
            show_error(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()

    def _do_load(self):
        """Actually load and display data."""
        # Load DLC
        self.dlc_df = pd.read_hdf(self.dlc_path)

        # Flatten multi-index columns
        scorer = self.dlc_df.columns.get_level_values(0)[0]
        self.dlc_df.columns = [f"{bp}_{coord}" if coord != 'likelihood' else f"{bp}_likelihood"
                               for _, bp, coord in self.dlc_df.columns]

        # Load segments
        with open(self.segments_path) as f:
            seg_data = json.load(f)
            self.boundaries = seg_data['boundaries']
            self.fps = seg_data.get('fps', 60.0)

        # Load video
        cap = cv2.VideoCapture(str(self.video_path))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        frames_array = np.array(frames)

        # Add video layer
        if self.video_layer is not None and self.video_layer in self.viewer.layers:
            self.viewer.layers.remove(self.video_layer)

        self.video_layer = self.viewer.add_image(
            frames_array,
            name=self.video_path.stem,
            colormap='gray'
        )

        # Compute ruler (median SABL-SABR distance)
        sabl_x = self.dlc_df['SABL_x'].values
        sabl_y = self.dlc_df['SABL_y'].values
        sabr_x = self.dlc_df['SABR_x'].values
        sabr_y = self.dlc_df['SABR_y'].values

        rulers = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)
        self.ruler_length = np.median(rulers[rulers > 0])

        # Add SA reference points layer
        if self.sa_points_layer is not None and self.sa_points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.sa_points_layer)

        # Show SABL/SABR for current frame
        self._update_sa_points()

        # Initialize pillar circle layer
        if self.pillar_circle_layer is not None and self.pillar_circle_layer in self.viewer.layers:
            self.viewer.layers.remove(self.pillar_circle_layer)

        self.pillar_circle_layer = self.viewer.add_shapes(
            name="Pillar Circle (adjust if wrong)",
            shape_type='ellipse',
            edge_color='cyan',
            edge_width=2,
            face_color='transparent'
        )

        # Update UI
        self.video_label.setText(f"{self.video_path.name}\n{self.n_frames} frames, {len(self.boundaries)} segments")
        self.prev_seg_btn.setEnabled(True)
        self.next_seg_btn.setEnabled(True)
        self.save_frame_btn.setEnabled(True)
        self.reset_circle_btn.setEnabled(True)

        # Go to first segment
        self.current_frame = self.boundaries[0]
        self.viewer.dims.current_step = (self.current_frame, 0, 0)
        self._update_frame_display()

        show_info(f"Loaded {self.video_path.name}")

    def _compute_auto_pillar(self, frame_idx: int) -> Tuple[float, float, float]:
        """
        Compute automatic pillar position using geometric calculation.

        Returns:
            (center_x, center_y, radius) in pixels
        """
        row = self.dlc_df.iloc[frame_idx]

        sabl_x = row['SABL_x']
        sabl_y = row['SABL_y']
        sabr_x = row['SABR_x']
        sabr_y = row['SABR_y']

        # SA midpoint
        mid_x = (sabl_x + sabr_x) / 2
        mid_y = (sabl_y + sabr_y) / 2

        # Ruler for this frame
        ruler = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)

        # Geometric pillar: 0.944 ruler units perpendicular from SA midpoint
        # In image coordinates, "above" SA means negative Y
        pillar_x = mid_x
        pillar_y = mid_y - (0.944 * ruler)

        # Estimate radius: assume pillar diameter ~= pellet diameter ~= 0.20 ruler units
        # So radius = 0.10 ruler units
        radius = 0.10 * ruler

        return pillar_x, pillar_y, radius

    def _update_sa_points(self):
        """Update SABL/SABR reference points for current frame."""
        if self.dlc_df is None:
            return

        frame_idx = int(self.viewer.dims.current_step[0])
        row = self.dlc_df.iloc[frame_idx]

        sabl = [frame_idx, row['SABL_y'], row['SABL_x']]
        sabr = [frame_idx, row['SABR_y'], row['SABR_x']]

        if self.sa_points_layer is not None and self.sa_points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.sa_points_layer)

        self.sa_points_layer = self.viewer.add_points(
            [sabl, sabr],
            name="SA Reference (SABL/SABR)",
            face_color=['yellow', 'yellow'],
            size=8,
            edge_color='black',
            edge_width=1
        )

    def _update_pillar_circle(self):
        """Update pillar circle for current frame."""
        frame_idx = int(self.viewer.dims.current_step[0])

        # Compute auto position
        auto_x, auto_y, auto_r = self._compute_auto_pillar(frame_idx)

        # Create circle shape (napari uses [y, x] coordinates)
        # Ellipse is defined by bounding box corners
        circle = np.array([
            [frame_idx, auto_y - auto_r, auto_x - auto_r],  # top-left
            [frame_idx, auto_y + auto_r, auto_x + auto_r]   # bottom-right
        ])

        # Update shapes layer
        self.pillar_circle_layer.data = [circle]
        self.pillar_circle_layer.shape_type = ['ellipse']

        self._update_circle_info(auto_x, auto_y, auto_r, is_auto=True)

    def _update_circle_info(self, center_x, center_y, radius, is_auto=True):
        """Update circle info label."""
        status = "Auto-positioned" if is_auto else "User-adjusted"
        self.circle_info_label.setText(
            f"Circle: {status}\n"
            f"Center: ({center_x:.1f}, {center_y:.1f})\n"
            f"Radius: {radius:.1f} px ({radius/self.ruler_length:.3f} ruler units)"
        )

    def _update_frame_display(self):
        """Update frame label and auto-populate pillar circle."""
        frame_idx = int(self.viewer.dims.current_step[0])
        self.current_frame = frame_idx

        seg_num = self._get_segment_num(frame_idx)

        self.frame_label.setText(f"Frame: {frame_idx} / {self.n_frames}")

        # Update SA reference points
        self._update_sa_points()

        # Auto-populate pillar circle
        self._update_pillar_circle()

    def _get_segment_num(self, frame_idx: int) -> int:
        """Get segment number for frame."""
        for i, bound in enumerate(self.boundaries):
            if frame_idx < bound:
                return i
        return len(self.boundaries)

    def _goto_next_segment(self):
        """Navigate to start of next segment."""
        seg_num = self._get_segment_num(self.current_frame)
        if seg_num < len(self.boundaries):
            self.current_frame = self.boundaries[seg_num]
            self.viewer.dims.current_step = (self.current_frame, 0, 0)
            self._update_frame_display()

    def _goto_prev_segment(self):
        """Navigate to start of previous segment."""
        seg_num = self._get_segment_num(self.current_frame)
        if seg_num > 0:
            self.current_frame = self.boundaries[seg_num - 1]
            self.viewer.dims.current_step = (self.current_frame, 0, 0)
            self._update_frame_display()

    def _nudge_circle(self, dx: int, dy: int):
        """Move circle by small amount."""
        if len(self.pillar_circle_layer.data) == 0:
            return

        circle = self.pillar_circle_layer.data[0].copy()

        # Move both corners by same amount (preserves size)
        circle[:, 2] += dx  # x
        circle[:, 1] += dy  # y

        self.pillar_circle_layer.data = [circle]

        # Update info (no longer auto)
        center_y = (circle[0, 1] + circle[1, 1]) / 2
        center_x = (circle[0, 2] + circle[1, 2]) / 2
        radius = (circle[1, 1] - circle[0, 1]) / 2

        self._update_circle_info(center_x, center_y, radius, is_auto=False)

    def _resize_circle(self, delta_r: float):
        """Resize circle."""
        if len(self.pillar_circle_layer.data) == 0:
            return

        circle = self.pillar_circle_layer.data[0].copy()

        # Get current center
        center_y = (circle[0, 1] + circle[1, 1]) / 2
        center_x = (circle[0, 2] + circle[1, 2]) / 2

        # Get current radius
        radius = (circle[1, 1] - circle[0, 1]) / 2

        # Update radius
        new_radius = max(1, radius + delta_r)

        # Update corners
        circle[0, 1] = center_y - new_radius
        circle[0, 2] = center_x - new_radius
        circle[1, 1] = center_y + new_radius
        circle[1, 2] = center_x + new_radius

        self.pillar_circle_layer.data = [circle]

        self._update_circle_info(center_x, center_y, new_radius, is_auto=False)

    def _reset_circle_to_auto(self):
        """Reset circle to automatic position."""
        self._update_pillar_circle()
        show_info("Circle reset to automatic position")

    def _save_current_frame(self):
        """Save current frame's pillar annotation."""
        frame_idx = int(self.viewer.dims.current_step[0])

        if len(self.pillar_circle_layer.data) == 0:
            show_warning("No circle to save")
            return

        circle = self.pillar_circle_layer.data[0]

        # Extract current circle
        user_center_y = (circle[0, 1] + circle[1, 1]) / 2
        user_center_x = (circle[0, 2] + circle[1, 2]) / 2
        user_radius = (circle[1, 1] - circle[0, 1]) / 2

        # Compute auto values
        auto_x, auto_y, auto_r = self._compute_auto_pillar(frame_idx)

        # Check if user modified
        dx = user_center_x - auto_x
        dy = user_center_y - auto_y
        dr = user_radius - auto_r

        modified = abs(dx) > 0.5 or abs(dy) > 0.5 or abs(dr) > 0.5
        moved = abs(dx) > 0.5 or abs(dy) > 0.5
        resized = abs(dr) > 0.5

        annotation = {
            'frame': int(frame_idx),
            'auto_pillar_center_x': float(auto_x),
            'auto_pillar_center_y': float(auto_y),
            'auto_pillar_radius_pixels': float(auto_r),
            'user_pillar_center_x': float(user_center_x),
            'user_pillar_center_y': float(user_center_y),
            'user_pillar_radius_pixels': float(user_radius),
            'user_modified': modified,
            'moved': moved,
            'resized': resized,
            'correction_delta_x': float(dx),
            'correction_delta_y': float(dy),
            'correction_delta_radius': float(dr),
            'note': self.annotation_note.text().strip(),
            'ruler_length_pixels': float(self.ruler_length),
            'timestamp': datetime.now().isoformat()
        }

        # Check if frame already annotated
        existing_idx = None
        for i, ann in enumerate(self.pillar_annotations):
            if ann['frame'] == frame_idx:
                existing_idx = i
                break

        if existing_idx is not None:
            self.pillar_annotations[existing_idx] = annotation
            show_info(f"Updated annotation for frame {frame_idx}")
        else:
            self.pillar_annotations.append(annotation)
            show_info(f"Saved annotation for frame {frame_idx}")

        self._update_annotations_list()
        self.save_all_btn.setEnabled(True)

    def _update_annotations_list(self):
        """Update list of saved annotations."""
        self.annotations_list.clear()

        for ann in sorted(self.pillar_annotations, key=lambda a: a['frame']):
            frame = ann['frame']
            status = "✎ Modified" if ann['user_modified'] else "✓ Auto"
            note = f" - {ann['note']}" if ann['note'] else ""

            item = QListWidgetItem(f"Frame {frame}: {status}{note}")
            item.setData(Qt.UserRole, ann)
            self.annotations_list.addItem(item)

        self.save_count_label.setText(f"{len(self.pillar_annotations)} annotations saved")
        self.delete_annotation_btn.setEnabled(len(self.pillar_annotations) > 0)

    def _on_annotation_selected(self, item):
        """Jump to selected annotation frame."""
        ann = item.data(Qt.UserRole)
        frame = ann['frame']

        self.viewer.dims.current_step = (frame, 0, 0)
        self._update_frame_display()

        # Load user's annotation into circle
        user_x = ann['user_pillar_center_x']
        user_y = ann['user_pillar_center_y']
        user_r = ann['user_pillar_radius_pixels']

        circle = np.array([
            [frame, user_y - user_r, user_x - user_r],
            [frame, user_y + user_r, user_x + user_r]
        ])

        self.pillar_circle_layer.data = [circle]
        self._update_circle_info(user_x, user_y, user_r, is_auto=not ann['user_modified'])

        # Load note
        self.annotation_note.setText(ann.get('note', ''))

    def _delete_current_annotation(self):
        """Delete currently selected annotation."""
        current_item = self.annotations_list.currentItem()
        if not current_item:
            show_warning("No annotation selected")
            return

        ann = current_item.data(Qt.UserRole)

        reply = QMessageBox.question(
            self,
            "Delete Annotation",
            f"Delete annotation for frame {ann['frame']}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.pillar_annotations = [a for a in self.pillar_annotations if a['frame'] != ann['frame']]
            self._update_annotations_list()
            show_info(f"Deleted annotation for frame {ann['frame']}")

    def _save_annotations(self):
        """Save all annotations to JSON file."""
        if not self.pillar_annotations:
            show_warning("No annotations to save")
            return

        # Default save path
        default_path = self.video_path.parent / f"{self.video_path.stem}_pillar_geometry.json"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pillar Geometry Annotations",
            str(default_path),
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        output = {
            'video_name': self.video_path.stem,
            'created_by': get_username(),
            'created_at': datetime.now().isoformat(),
            'ruler_length_pixels': float(self.ruler_length),
            'n_annotations': len(self.pillar_annotations),
            'annotations': self.pillar_annotations
        }

        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)

        show_info(f"Saved {len(self.pillar_annotations)} annotations to {Path(file_path).name}")


# Napari plugin registration
@napari.hookimpl
def napari_experimental_provide_dock_widget():
    return PillarGeometryAnnotatorWidget


# Standalone execution
if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    widget = PillarGeometryAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, name='Pillar Geometry')
    napari.run()
