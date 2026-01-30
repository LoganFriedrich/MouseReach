"""
Napari widget for validating reach-outcome linkages.

Allows manual verification and correction of which reach caused each pellet outcome.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

import napari
import numpy as np
import pandas as pd
from napari.layers import Image, Shapes, Points
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QGroupBox, QTextEdit, QCheckBox
)
from qtpy.QtCore import Qt


class ReachOutcomeValidator(QWidget):
    """Widget for validating reach-outcome linkages."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Data
        self.video_path: Optional[Path] = None
        self.dlc_path: Optional[Path] = None
        self.reaches_path: Optional[Path] = None
        self.outcomes_path: Optional[Path] = None
        self.features_path: Optional[Path] = None

        self.dlc_df: Optional[pd.DataFrame] = None
        self.reaches_data: Optional[Dict] = None
        self.outcomes_data: Optional[Dict] = None
        self.features_data: Optional[Dict] = None

        # Current state
        self.current_segment_idx: int = 0
        self.current_segment_data: Optional[Dict] = None
        self.video_frames_cache: Optional[np.ndarray] = None

        # Layers
        self.image_layer: Optional[Image] = None
        self.reach_shapes_layer: Optional[Shapes] = None
        self.pellet_points_layer: Optional[Points] = None
        self.interaction_marker_layer: Optional[Points] = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout()

        # File selection group
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()

        self.file_label = QLabel("No files loaded")
        file_layout.addWidget(self.file_label)

        load_btn = QPushButton("Load Video + Features")
        load_btn.clicked.connect(self._load_files)
        file_layout.addWidget(load_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        file_layout.addWidget(self.progress)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Segment navigation
        nav_group = QGroupBox("Segment Navigation")
        nav_layout = QVBoxLayout()

        seg_nav_row = QHBoxLayout()
        seg_nav_row.addWidget(QLabel("Segment:"))

        self.segment_spinner = QSpinBox()
        self.segment_spinner.setMinimum(1)
        self.segment_spinner.setMaximum(20)
        self.segment_spinner.valueChanged.connect(self._on_segment_changed)
        seg_nav_row.addWidget(self.segment_spinner)

        prev_seg_btn = QPushButton("◀ Prev")
        prev_seg_btn.clicked.connect(self._prev_segment)
        seg_nav_row.addWidget(prev_seg_btn)

        next_seg_btn = QPushButton("Next ▶")
        next_seg_btn.clicked.connect(self._next_segment)
        seg_nav_row.addWidget(next_seg_btn)

        seg_nav_row.addStretch()
        nav_layout.addLayout(seg_nav_row)

        # Segment info
        self.segment_info_label = QLabel("No segment loaded")
        nav_layout.addWidget(self.segment_info_label)

        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)

        # Outcome info
        outcome_group = QGroupBox("Outcome Information")
        outcome_layout = QVBoxLayout()

        self.outcome_label = QLabel("Outcome: -")
        outcome_layout.addWidget(self.outcome_label)

        self.interaction_frame_label = QLabel("Interaction frame: -")
        outcome_layout.addWidget(self.interaction_frame_label)

        self.causal_reach_label = QLabel("Causal reach: -")
        outcome_layout.addWidget(self.causal_reach_label)

        outcome_group.setLayout(outcome_layout)
        layout.addWidget(outcome_group)

        # Reaches table
        reaches_group = QGroupBox("Reaches in This Segment")
        reaches_layout = QVBoxLayout()

        self.reaches_table = QTableWidget()
        self.reaches_table.setColumnCount(7)
        self.reaches_table.setHorizontalHeaderLabels([
            "Reach ID", "Start", "Apex", "End", "Duration", "Extent (mm)", "Causal?"
        ])
        self.reaches_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.reaches_table.setSelectionMode(QTableWidget.SingleSelection)
        self.reaches_table.itemSelectionChanged.connect(self._on_reach_selected)
        reaches_layout.addWidget(self.reaches_table)

        # Jump to buttons
        jump_row = QHBoxLayout()

        jump_start_btn = QPushButton("Jump to Start")
        jump_start_btn.clicked.connect(self._jump_to_reach_start)
        jump_row.addWidget(jump_start_btn)

        jump_apex_btn = QPushButton("Jump to Apex")
        jump_apex_btn.clicked.connect(self._jump_to_reach_apex)
        jump_row.addWidget(jump_apex_btn)

        jump_interaction_btn = QPushButton("Jump to Interaction")
        jump_interaction_btn.clicked.connect(self._jump_to_interaction)
        jump_row.addWidget(jump_interaction_btn)

        reaches_layout.addLayout(jump_row)

        # Set as causal reach button
        set_causal_btn = QPushButton("Set Selected as Causal Reach")
        set_causal_btn.clicked.connect(self._set_causal_reach)
        reaches_layout.addWidget(set_causal_btn)

        reaches_group.setLayout(reaches_layout)
        layout.addWidget(reaches_group)

        # Validation controls
        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout()

        self.verified_checkbox = QCheckBox("Mark as verified")
        self.verified_checkbox.stateChanged.connect(self._on_verified_changed)
        validation_layout.addWidget(self.verified_checkbox)

        self.notes_label = QLabel("Notes:")
        validation_layout.addWidget(self.notes_label)

        self.notes_text = QTextEdit()
        self.notes_text.setMaximumHeight(60)
        validation_layout.addWidget(self.notes_text)

        validation_group.setLayout(validation_layout)
        layout.addWidget(validation_group)

        # Save buttons
        save_layout = QHBoxLayout()

        save_validation_btn = QPushButton("Save as Validation")
        save_validation_btn.clicked.connect(self._save_validation)
        save_layout.addWidget(save_validation_btn)

        save_ground_truth_btn = QPushButton("Save as Ground Truth")
        save_ground_truth_btn.clicked.connect(self._save_ground_truth)
        save_layout.addWidget(save_ground_truth_btn)

        layout.addLayout(save_layout)

        # Statistics
        stats_group = QGroupBox("Validation Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("No data loaded")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        self.setLayout(layout)

    def _load_files(self):
        """Load video and feature files."""
        # For now, prompt user to select features file
        # Then derive other paths from it
        from qtpy.QtWidgets import QFileDialog

        features_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Features JSON",
            str(Path.home()),
            "Feature Files (*_features.json)"
        )

        if not features_file:
            return

        self.features_path = Path(features_file)
        video_name = self.features_path.stem.replace('_features', '')
        base_dir = self.features_path.parent

        # Find other files
        self.dlc_path = list(base_dir.glob(f"{video_name}*DLC*.h5"))[0]
        self.reaches_path = base_dir / f"{video_name}_reaches.json"
        self.outcomes_path = base_dir / f"{video_name}_pellet_outcomes.json"

        # Find video
        video_files = list(base_dir.glob(f"{video_name}*.mp4")) + list(base_dir.glob(f"{video_name}*.avi"))
        if video_files:
            self.video_path = video_files[0]

        self._load_data()

    def _load_data(self):
        """Load all data files."""
        self.progress.setVisible(True)
        self.progress.setValue(0)

        try:
            # Load DLC
            self.progress.setValue(10)
            self.dlc_df = pd.read_hdf(self.dlc_path)
            self.dlc_df.columns = ['_'.join(col[1:]) for col in self.dlc_df.columns]

            # Load reaches
            self.progress.setValue(30)
            with open(self.reaches_path) as f:
                self.reaches_data = json.load(f)

            # Load outcomes
            self.progress.setValue(50)
            with open(self.outcomes_path) as f:
                self.outcomes_data = json.load(f)

            # Load features
            self.progress.setValue(70)
            with open(self.features_path) as f:
                self.features_data = json.load(f)

            # Load video if available
            if self.video_path and self.video_path.exists():
                self.progress.setValue(80)
                self._load_video()

            self.progress.setValue(100)
            self.progress.setVisible(False)

            # Update UI
            self.file_label.setText(f"Loaded: {self.features_path.name}")
            self.segment_spinner.setMaximum(self.features_data['n_segments'])

            # Load first segment
            self.current_segment_idx = 0
            self._load_segment(0)

            self._update_stats()

        except Exception as e:
            self.file_label.setText(f"Error loading: {e}")
            self.progress.setVisible(False)

    def _load_video(self):
        """Load video frames."""
        import cv2

        video_path_str = str(self.video_path)

        cap = cv2.VideoCapture(video_path_str)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        self.video_frames_cache = np.array(frames)

        # Add to napari
        if self.image_layer is not None and self.image_layer in self.viewer.layers:
            self.viewer.layers.remove(self.image_layer)

        self.image_layer = self.viewer.add_image(
            self.video_frames_cache,
            name='Video',
            colormap='gray'
        )

    def _load_segment(self, seg_idx: int):
        """Load a specific segment."""
        if self.features_data is None:
            return

        seg_data = self.features_data['segments'][seg_idx]
        outcome_data = self.outcomes_data['segments'][seg_idx]
        reaches_seg_data = self.reaches_data['segments'][seg_idx]

        self.current_segment_data = {
            'features': seg_data,
            'outcome': outcome_data,
            'reaches': reaches_seg_data
        }

        # Update UI
        seg_num = seg_data['segment_num']
        start_frame = seg_data['start_frame']
        end_frame = seg_data['end_frame']

        self.segment_info_label.setText(
            f"Segment {seg_num}: Frames {start_frame}-{end_frame} ({end_frame - start_frame} frames)"
        )

        outcome = seg_data['outcome']
        confidence = seg_data['outcome_confidence']
        self.outcome_label.setText(f"Outcome: {outcome} (confidence: {confidence:.2f})")

        interaction_frame = outcome_data.get('interaction_frame')
        if interaction_frame:
            self.interaction_frame_label.setText(f"Interaction frame: {interaction_frame}")
        else:
            self.interaction_frame_label.setText("Interaction frame: None")

        causal_reach_id = seg_data.get('causal_reach_id')
        if causal_reach_id:
            self.causal_reach_label.setText(f"Causal reach: {causal_reach_id}")
        else:
            self.causal_reach_label.setText("Causal reach: None")

        # Populate reaches table
        self._populate_reaches_table(seg_data['reaches'])

        # Update verified checkbox
        self.verified_checkbox.setChecked(outcome_data.get('human_verified', False))

        # Update visualization
        self._update_visualization()

    def _populate_reaches_table(self, reaches: List[Dict]):
        """Populate the reaches table."""
        self.reaches_table.setRowCount(len(reaches))

        for i, reach in enumerate(reaches):
            self.reaches_table.setItem(i, 0, QTableWidgetItem(str(reach['reach_id'])))
            self.reaches_table.setItem(i, 1, QTableWidgetItem(str(reach['start_frame'])))
            self.reaches_table.setItem(i, 2, QTableWidgetItem(str(reach.get('apex_frame', '-'))))
            self.reaches_table.setItem(i, 3, QTableWidgetItem(str(reach['end_frame'])))
            self.reaches_table.setItem(i, 4, QTableWidgetItem(str(reach['duration_frames'])))

            extent_mm = reach.get('max_extent_mm')
            extent_str = f"{extent_mm:.2f}" if extent_mm is not None else '-'
            self.reaches_table.setItem(i, 5, QTableWidgetItem(extent_str))

            causal_str = "YES" if reach.get('causal_reach', False) else ""
            causal_item = QTableWidgetItem(causal_str)
            if reach.get('causal_reach', False):
                causal_item.setBackground(Qt.green)
            self.reaches_table.setItem(i, 6, causal_item)

        self.reaches_table.resizeColumnsToContents()

    def _update_visualization(self):
        """Update napari visualization for current segment."""
        if self.current_segment_data is None:
            return

        seg_data = self.current_segment_data['features']
        start_frame = seg_data['start_frame']
        end_frame = seg_data['end_frame']

        # Jump to segment start
        self.viewer.dims.set_point(0, start_frame)

        # Update reach shapes layer
        self._update_reach_shapes()

        # Update pellet trajectory
        self._update_pellet_points()

        # Update interaction marker
        self._update_interaction_marker()

    def _update_reach_shapes(self):
        """Draw bounding boxes for all reaches in segment."""
        if self.current_segment_data is None:
            return

        # Remove old layer
        if self.reach_shapes_layer is not None and self.reach_shapes_layer in self.viewer.layers:
            self.viewer.layers.remove(self.reach_shapes_layer)

        reaches = self.current_segment_data['features']['reaches']

        # Create rectangles for each reach (temporal extent)
        shapes_data = []
        shape_types = []
        colors = []

        for reach in reaches:
            start = reach['start_frame']
            end = reach['end_frame']

            # Create a line at y=50 showing temporal extent
            line_data = np.array([[start, 50], [end, 50]])
            shapes_data.append(line_data)
            shape_types.append('line')

            # Color: green if causal, blue otherwise
            if reach.get('causal_reach', False):
                colors.append('green')
            else:
                colors.append('cyan')

        if shapes_data:
            self.reach_shapes_layer = self.viewer.add_shapes(
                shapes_data,
                shape_type=shape_types,
                edge_color=colors,
                edge_width=3,
                name='Reaches',
                opacity=0.8
            )

    def _update_pellet_points(self):
        """Show pellet trajectory during segment."""
        if self.current_segment_data is None or self.dlc_df is None:
            return

        # Remove old layer
        if self.pellet_points_layer is not None and self.pellet_points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.pellet_points_layer)

        seg_data = self.current_segment_data['features']
        start_frame = seg_data['start_frame']
        end_frame = seg_data['end_frame']

        seg_df = self.dlc_df.iloc[start_frame:end_frame]

        # Get pellet positions
        pellet_x = seg_df['Pellet_x'].values
        pellet_y = seg_df['Pellet_y'].values
        pellet_conf = seg_df['Pellet_likelihood'].values

        # Filter high confidence
        valid = pellet_conf > 0.5

        points_data = []
        for i, (x, y, v) in enumerate(zip(pellet_x, pellet_y, valid)):
            if v:
                frame_idx = start_frame + i
                points_data.append([frame_idx, y, x])

        if points_data:
            self.pellet_points_layer = self.viewer.add_points(
                np.array(points_data),
                name='Pellet',
                size=3,
                face_color='yellow',
                opacity=0.6
            )

    def _update_interaction_marker(self):
        """Mark the interaction frame."""
        if self.current_segment_data is None:
            return

        # Remove old layer
        if self.interaction_marker_layer is not None and self.interaction_marker_layer in self.viewer.layers:
            self.viewer.layers.remove(self.interaction_marker_layer)

        interaction_frame = self.current_segment_data['outcome'].get('interaction_frame')

        if interaction_frame is not None and self.dlc_df is not None:
            # Get pellet position at interaction
            row = self.dlc_df.iloc[interaction_frame]
            pellet_x = row.get('Pellet_x')
            pellet_y = row.get('Pellet_y')

            if not np.isnan(pellet_x) and not np.isnan(pellet_y):
                self.interaction_marker_layer = self.viewer.add_points(
                    np.array([[interaction_frame, pellet_y, pellet_x]]),
                    name='Interaction',
                    size=10,
                    face_color='red',
                    symbol='cross'
                )

    def _on_segment_changed(self, value):
        """Handle segment spinner change."""
        self.current_segment_idx = value - 1
        self._load_segment(self.current_segment_idx)

    def _prev_segment(self):
        """Go to previous segment."""
        if self.current_segment_idx > 0:
            self.segment_spinner.setValue(self.current_segment_idx)

    def _next_segment(self):
        """Go to next segment."""
        if self.current_segment_idx < len(self.features_data['segments']) - 1:
            self.segment_spinner.setValue(self.current_segment_idx + 2)

    def _on_reach_selected(self):
        """Handle reach selection in table."""
        selected_rows = self.reaches_table.selectedIndexes()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        # Could highlight the selected reach
        pass

    def _jump_to_reach_start(self):
        """Jump to start of selected reach."""
        selected_rows = self.reaches_table.selectedIndexes()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        reach = self.current_segment_data['features']['reaches'][row]
        self.viewer.dims.set_point(0, reach['start_frame'])

    def _jump_to_reach_apex(self):
        """Jump to apex of selected reach."""
        selected_rows = self.reaches_table.selectedIndexes()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        reach = self.current_segment_data['features']['reaches'][row]
        apex = reach.get('apex_frame')
        if apex:
            self.viewer.dims.set_point(0, apex)

    def _jump_to_interaction(self):
        """Jump to interaction frame."""
        interaction_frame = self.current_segment_data['outcome'].get('interaction_frame')
        if interaction_frame:
            self.viewer.dims.set_point(0, interaction_frame)

    def _set_causal_reach(self):
        """Set selected reach as causal."""
        selected_rows = self.reaches_table.selectedIndexes()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        reach = self.current_segment_data['features']['reaches'][row]

        # Update causal reach
        # First, clear all causal flags
        for r in self.current_segment_data['features']['reaches']:
            r['causal_reach'] = False

        # Set new causal reach
        reach['causal_reach'] = True
        self.current_segment_data['features']['causal_reach_id'] = reach['reach_id']

        # Update outcome data
        self.current_segment_data['outcome']['causal_reach_id'] = reach['reach_id']
        self.current_segment_data['outcome']['causal_reach_frame'] = reach.get('apex_frame')

        # Mark as modified
        self.current_segment_data['outcome']['human_verified'] = True

        # Refresh display
        self._populate_reaches_table(self.current_segment_data['features']['reaches'])
        self._update_reach_shapes()

        self.causal_reach_label.setText(f"Causal reach: {reach['reach_id']}")

    def _on_verified_changed(self, state):
        """Handle verified checkbox change."""
        if self.current_segment_data:
            self.current_segment_data['outcome']['human_verified'] = (state == Qt.Checked)

    def _save_validation(self):
        """Save as validation (overwrites existing files)."""
        if self.outcomes_data is None or self.features_data is None:
            return

        try:
            # Mark all verified segments
            from datetime import datetime
            timestamp = datetime.now().isoformat()

            for seg in self.outcomes_data['segments']:
                if seg.get('human_verified', False):
                    seg['validated_at'] = timestamp
                    seg['validated_by'] = self._get_username()

            # Save updated outcomes
            with open(self.outcomes_path, 'w') as f:
                json.dump(self.outcomes_data, f, indent=2)

            # Save updated features
            with open(self.features_path, 'w') as f:
                json.dump(self.features_data, f, indent=2)

            # Update pipeline index
            try:
                from mousereach.index import PipelineIndex
                from mousereach.config import get_video_id
                video_id = get_video_id(self.features_path)
                index = PipelineIndex()
                index.load()
                index.record_validation_changed(video_id, "kinematics", "validated")
                index.save()
            except Exception as idx_e:
                print(f"Index update warning: {idx_e}")

            self.file_label.setText(f"Saved validation: {self.features_path.name}")

        except Exception as e:
            self.file_label.setText(f"Error saving: {e}")

    def _save_ground_truth(self):
        """Save as ground truth (creates new _ground_truth.json files)."""
        if self.outcomes_data is None or self.features_data is None:
            return

        try:
            from datetime import datetime
            import copy

            # Create ground truth versions
            gt_outcomes = copy.deepcopy(self.outcomes_data)
            gt_features = copy.deepcopy(self.features_data)

            # Add ground truth metadata
            timestamp = datetime.now().isoformat()
            username = self._get_username()

            gt_outcomes['ground_truth'] = True
            gt_outcomes['created_at'] = timestamp
            gt_outcomes['created_by'] = username

            gt_features['ground_truth'] = True
            gt_features['created_at'] = timestamp
            gt_features['created_by'] = username

            # Mark all as verified
            for seg in gt_outcomes['segments']:
                seg['human_verified'] = True
                seg['validated_at'] = timestamp
                seg['validated_by'] = username

            # Save ground truth files
            video_name = self.features_path.stem.replace('_features', '')
            base_dir = self.features_path.parent

            gt_outcomes_path = base_dir / f"{video_name}_outcome_ground_truth.json"
            gt_features_path = base_dir / f"{video_name}_features_ground_truth.json"

            with open(gt_outcomes_path, 'w') as f:
                json.dump(gt_outcomes, f, indent=2)

            with open(gt_features_path, 'w') as f:
                json.dump(gt_features, f, indent=2)

            # Update pipeline index with ground truth status
            try:
                from mousereach.index import PipelineIndex
                index = PipelineIndex()
                index.load()
                # Check if all segments are verified
                all_verified = all(seg.get('human_verified', False) for seg in gt_outcomes.get('segments', []))
                index.record_gt_created(video_name, "kinematics", is_complete=all_verified)
                index.save()
            except Exception as idx_e:
                print(f"Index update warning: {idx_e}")

            self.file_label.setText(f"Saved ground truth: {gt_outcomes_path.name}, {gt_features_path.name}")

        except Exception as e:
            self.file_label.setText(f"Error saving ground truth: {e}")

    def _get_username(self):
        """Get current username."""
        import os
        try:
            return os.getlogin()
        except (OSError, AttributeError):
            return os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))

    def _update_stats(self):
        """Update validation statistics."""
        if self.outcomes_data is None:
            return

        total = len(self.outcomes_data['segments'])
        verified = sum(1 for seg in self.outcomes_data['segments'] if seg.get('human_verified', False))

        self.stats_label.setText(
            f"Verified: {verified}/{total} segments ({verified/total*100:.1f}%)"
        )


# Napari plugin registration
@napari.viewer.Viewer.bind_key('v')
def launch_validator(viewer):
    """Launch reach-outcome validator (press 'v')."""
    validator = ReachOutcomeValidator(viewer)
    viewer.window.add_dock_widget(validator, name='Reach-Outcome Validator')
