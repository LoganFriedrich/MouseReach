"""
Napari widget for viewing reach-outcome data.

Simple viewer to explore all the extracted data overlaid on video:
- DLC tracking points
- Reaches with temporal extents
- Pellet trajectories
- Interaction markers
- Outcome labels
- Feature values

No editing - just for exploration and presentation.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

import napari
import numpy as np
import pandas as pd
from napari.layers import Image, Shapes, Points, Labels
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QTableWidget, QTableWidgetItem, QProgressBar,
    QGroupBox, QCheckBox, QSlider, QScrollArea
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont


class DataViewerWidget(QWidget):
    """Widget for viewing reach-outcome data (read-only)."""

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
        self.n_segments: int = 20
        self.is_playing: bool = False
        self.playback_speed: int = 1

        # Layers
        self.video_layer: Optional[Image] = None
        self.dlc_points_layer: Optional[Points] = None
        self.reach_shapes_layer: Optional[Shapes] = None
        self.pellet_points_layer: Optional[Points] = None
        self.interaction_marker_layer: Optional[Points] = None

        # Playback
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the widget UI."""
        # Main layout with scroll area
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll)

        inner_widget = QWidget()
        scroll.setWidget(inner_widget)
        layout = QVBoxLayout()
        inner_widget.setLayout(layout)

        # Instructions
        instructions = QLabel(
            "Space=play, 1-5=speed (1x/2x/4x/8x/16x), ←/→=frame step\n"
            "N/P=next/prev segment, J=jump to interaction"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # File loading
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()

        self.file_label = QLabel("No files loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        load_btn = QPushButton("Select Features JSON...")
        load_btn.clicked.connect(self._load_files)
        file_layout.addWidget(load_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        file_layout.addWidget(self.progress)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Playback controls
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout()

        # Frame info
        self.frame_label = QLabel("Frame: -- / --")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.frame_label.setFont(font)
        playback_layout.addWidget(self.frame_label)

        # Play buttons
        play_row = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        play_row.addWidget(self.play_btn)

        play_row.addWidget(QLabel("Speed:"))
        self.speed_buttons = {}
        for speed in [1, 2, 4, 8, 16]:
            btn = QPushButton(f"{speed}x")
            btn.setCheckable(True)
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, s=speed: self._set_speed(s))
            self.speed_buttons[speed] = btn
            play_row.addWidget(btn)
        self.speed_buttons[1].setChecked(True)

        playback_layout.addLayout(play_row)

        # Frame stepping
        step_row = QHBoxLayout()
        for delta, label in [(-100, "-100"), (-10, "-10"), (-1, "-1"), (1, "+1"), (10, "+10"), (100, "+100")]:
            btn = QPushButton(label)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, d=delta: self._step_frame(d))
            step_row.addWidget(btn)
        playback_layout.addLayout(step_row)

        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)

        # Segment navigation
        seg_group = QGroupBox("Segment Navigation")
        seg_layout = QVBoxLayout()

        self.segment_label = QLabel("Segment: -- / --")
        seg_layout.addWidget(self.segment_label)

        seg_nav_row = QHBoxLayout()
        seg_nav_row.addWidget(QLabel("Segment:"))

        self.segment_spinner = QSpinBox()
        self.segment_spinner.setMinimum(1)
        self.segment_spinner.setMaximum(20)
        self.segment_spinner.valueChanged.connect(self._on_segment_changed)
        seg_nav_row.addWidget(self.segment_spinner)

        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self._prev_segment)
        seg_nav_row.addWidget(prev_btn)

        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self._next_segment)
        seg_nav_row.addWidget(next_btn)

        seg_nav_row.addStretch()
        seg_layout.addLayout(seg_nav_row)

        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)

        # Segment info
        info_group = QGroupBox("Current Segment Info")
        info_layout = QVBoxLayout()

        self.outcome_label = QLabel("Outcome: -")
        info_layout.addWidget(self.outcome_label)

        self.interaction_label = QLabel("Interaction: -")
        info_layout.addWidget(self.interaction_label)

        self.causal_reach_label = QLabel("Causal reach: -")
        info_layout.addWidget(self.causal_reach_label)

        self.confidence_label = QLabel("Confidence: -")
        info_layout.addWidget(self.confidence_label)

        jump_btn = QPushButton("Jump to Interaction Frame")
        jump_btn.clicked.connect(self._jump_to_interaction)
        info_layout.addWidget(jump_btn)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Reaches table
        reaches_group = QGroupBox("Reaches in Segment")
        reaches_layout = QVBoxLayout()

        self.reaches_table = QTableWidget()
        self.reaches_table.setColumnCount(8)
        self.reaches_table.setHorizontalHeaderLabels([
            "ID", "Start", "Apex", "End", "Dur", "Extent(mm)", "Velocity", "Causal"
        ])
        self.reaches_table.setMaximumHeight(200)
        reaches_layout.addWidget(self.reaches_table)

        reaches_group.setLayout(reaches_layout)
        layout.addWidget(reaches_group)

        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()

        self.show_dlc_check = QCheckBox("Show DLC Points")
        self.show_dlc_check.setChecked(True)
        self.show_dlc_check.stateChanged.connect(self._update_layer_visibility)
        display_layout.addWidget(self.show_dlc_check)

        self.show_pellet_check = QCheckBox("Show Pellet Trajectory")
        self.show_pellet_check.setChecked(True)
        self.show_pellet_check.stateChanged.connect(self._update_layer_visibility)
        display_layout.addWidget(self.show_pellet_check)

        self.show_reaches_check = QCheckBox("Show Reach Extents")
        self.show_reaches_check.setChecked(True)
        self.show_reaches_check.stateChanged.connect(self._update_layer_visibility)
        display_layout.addWidget(self.show_reaches_check)

        self.show_interaction_check = QCheckBox("Show Interaction Marker")
        self.show_interaction_check.setChecked(True)
        self.show_interaction_check.stateChanged.connect(self._update_layer_visibility)
        display_layout.addWidget(self.show_interaction_check)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Stats
        stats_group = QGroupBox("Video Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("No data loaded")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()

    def _load_files(self):
        """Load video and feature files."""
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

            # Add DLC points layer
            self._add_dlc_points_layer()

            self.progress.setValue(100)
            self.progress.setVisible(False)

            # Update UI
            self.file_label.setText(f"Loaded: {self.features_path.name}")
            self.n_segments = self.features_data['n_segments']
            self.segment_spinner.setMaximum(self.n_segments)
            self.play_btn.setEnabled(True)

            # Load first segment
            self.current_segment_idx = 0
            self._load_segment(0)

            self._update_stats()

            # Connect viewer events
            self.viewer.dims.events.current_step.connect(self._on_frame_changed)

        except Exception as e:
            self.file_label.setText(f"Error loading: {e}")
            self.progress.setVisible(False)
            import traceback
            traceback.print_exc()

    def _load_video(self):
        """Load video frames."""
        import cv2

        # Check for compressed preview version (saves memory)
        video_stem = self.video_path.stem.replace('_preview', '')
        if 'DLC_' in video_stem:
            video_stem = video_stem.split('DLC_')[0]
        preview_path = self.video_path.parent / f"{video_stem}_preview.mp4"

        if '_preview' not in self.video_path.stem:
            if not preview_path.exists():
                # Auto-create preview on first load
                print(f"Creating preview video (one-time): {preview_path.name}")
                from napari.utils.notifications import show_info
                show_info("Creating compressed preview (one-time)...")
                from mousereach.video_prep.compress import create_preview
                create_preview(self.video_path)

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

        video_path_str = str(actual_video)

        cap = cv2.VideoCapture(video_path_str)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        video_frames = np.array(frames)

        # Add to napari
        if self.video_layer is not None and self.video_layer in self.viewer.layers:
            self.viewer.layers.remove(self.video_layer)

        self.video_layer = self.viewer.add_image(
            video_frames,
            name='Video',
            rgb=True
        )

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
            [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5]
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
                    point_colors.append(bp_colors[bp] + [0.3])

        if points_data:
            if self.dlc_points_layer is not None and self.dlc_points_layer in self.viewer.layers:
                self.viewer.layers.remove(self.dlc_points_layer)

            self.dlc_points_layer = self.viewer.add_points(
                np.array(points_data),
                name='DLC Tracking',
                size=4,
                face_color=np.array(point_colors)
            )

    def _load_segment(self, seg_idx: int):
        """Load a specific segment."""
        if self.features_data is None:
            return

        seg_data = self.features_data['segments'][seg_idx]
        outcome_data = self.outcomes_data['segments'][seg_idx]

        # Update labels
        seg_num = seg_data['segment_num']
        start_frame = seg_data['start_frame']
        end_frame = seg_data['end_frame']

        self.segment_label.setText(f"Segment {seg_num}/{self.n_segments}: Frames {start_frame}-{end_frame}")

        outcome = seg_data['outcome']
        confidence = seg_data['outcome_confidence']
        self.outcome_label.setText(f"Outcome: {outcome}")
        self.confidence_label.setText(f"Confidence: {confidence:.2f}")

        interaction_frame = outcome_data.get('interaction_frame')
        if interaction_frame:
            self.interaction_label.setText(f"Interaction: Frame {interaction_frame}")
        else:
            self.interaction_label.setText("Interaction: None")

        causal_reach_id = seg_data.get('causal_reach_id')
        if causal_reach_id:
            self.causal_reach_label.setText(f"Causal reach: ID {causal_reach_id}")
        else:
            self.causal_reach_label.setText("Causal reach: None")

        # Populate reaches table
        self._populate_reaches_table(seg_data['reaches'])

        # Update visualization
        self._update_segment_visualization(seg_data, outcome_data)

        # Jump to segment start
        self.viewer.dims.set_point(0, start_frame)

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

            velocity = reach.get('peak_velocity_px_per_frame')
            vel_str = f"{velocity:.1f}" if velocity is not None else '-'
            self.reaches_table.setItem(i, 6, QTableWidgetItem(vel_str))

            causal_str = "YES" if reach.get('causal_reach', False) else ""
            causal_item = QTableWidgetItem(causal_str)
            if reach.get('causal_reach', False):
                causal_item.setBackground(Qt.green)
            self.reaches_table.setItem(i, 7, causal_item)

        self.reaches_table.resizeColumnsToContents()

    def _update_segment_visualization(self, seg_data: Dict, outcome_data: Dict):
        """Update visualization for current segment."""
        # Update reach shapes
        self._update_reach_shapes(seg_data)

        # Update pellet trajectory
        self._update_pellet_trajectory(seg_data)

        # Update interaction marker
        self._update_interaction_marker(outcome_data)

    def _update_reach_shapes(self, seg_data: Dict):
        """Draw reach temporal extents."""
        if self.reach_shapes_layer is not None and self.reach_shapes_layer in self.viewer.layers:
            self.viewer.layers.remove(self.reach_shapes_layer)

        reaches = seg_data['reaches']
        shapes_data = []
        shape_types = []
        colors = []

        for reach in reaches:
            start = reach['start_frame']
            end = reach['end_frame']

            # Line showing temporal extent
            line_data = np.array([[start, 30], [end, 30]])
            shapes_data.append(line_data)
            shape_types.append('line')

            if reach.get('causal_reach', False):
                colors.append('lime')
            else:
                colors.append('cyan')

        if shapes_data:
            self.reach_shapes_layer = self.viewer.add_shapes(
                shapes_data,
                shape_type=shape_types,
                edge_color=colors,
                edge_width=4,
                name='Reaches',
                opacity=0.9
            )

    def _update_pellet_trajectory(self, seg_data: Dict):
        """Show pellet trajectory during segment."""
        if self.pellet_points_layer is not None and self.pellet_points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.pellet_points_layer)

        start_frame = seg_data['start_frame']
        end_frame = seg_data['end_frame']
        seg_df = self.dlc_df.iloc[start_frame:end_frame]

        pellet_x = seg_df['Pellet_x'].values
        pellet_y = seg_df['Pellet_y'].values
        pellet_conf = seg_df['Pellet_likelihood'].values

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
                size=5,
                face_color='yellow',
                opacity=0.7
            )

    def _update_interaction_marker(self, outcome_data: Dict):
        """Mark the interaction frame."""
        if self.interaction_marker_layer is not None and self.interaction_marker_layer in self.viewer.layers:
            self.viewer.layers.remove(self.interaction_marker_layer)

        interaction_frame = outcome_data.get('interaction_frame')

        if interaction_frame is not None:
            row = self.dlc_df.iloc[interaction_frame]
            pellet_x = row.get('Pellet_x')
            pellet_y = row.get('Pellet_y')

            if not np.isnan(pellet_x) and not np.isnan(pellet_y):
                self.interaction_marker_layer = self.viewer.add_points(
                    np.array([[interaction_frame, pellet_y, pellet_x]]),
                    name='Interaction',
                    size=15,
                    face_color='red',
                    symbol='cross'
                )

    def _update_layer_visibility(self):
        """Update visibility of layers based on checkboxes."""
        if self.dlc_points_layer:
            self.dlc_points_layer.visible = self.show_dlc_check.isChecked()
        if self.pellet_points_layer:
            self.pellet_points_layer.visible = self.show_pellet_check.isChecked()
        if self.reach_shapes_layer:
            self.reach_shapes_layer.visible = self.show_reaches_check.isChecked()
        if self.interaction_marker_layer:
            self.interaction_marker_layer.visible = self.show_interaction_check.isChecked()

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
        if self.current_segment_idx < self.n_segments - 1:
            self.segment_spinner.setValue(self.current_segment_idx + 2)

    def _jump_to_interaction(self):
        """Jump to interaction frame."""
        if self.features_data is None:
            return

        seg_data = self.features_data['segments'][self.current_segment_idx]
        outcome_data = self.outcomes_data['segments'][self.current_segment_idx]

        interaction_frame = outcome_data.get('interaction_frame')
        if interaction_frame:
            self.viewer.dims.set_point(0, interaction_frame)

    def _toggle_play(self):
        """Toggle playback."""
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        """Start playback."""
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        interval = int(1000 / (60 * self.playback_speed))
        self.playback_timer.start(interval)

    def _stop_play(self):
        """Stop playback."""
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.playback_timer.stop()

    def _playback_step(self):
        """Advance one frame during playback."""
        current_frame = self.viewer.dims.current_step[0]
        next_frame = current_frame + 1

        if next_frame >= len(self.dlc_df):
            next_frame = 0

        self.viewer.dims.set_point(0, next_frame)

    def _set_speed(self, speed: int):
        """Set playback speed."""
        self.playback_speed = speed

        for s, btn in self.speed_buttons.items():
            btn.setChecked(s == speed)

        if self.is_playing:
            interval = int(1000 / (60 * self.playback_speed))
            self.playback_timer.setInterval(interval)

    def _step_frame(self, delta: int):
        """Step forward/backward by delta frames."""
        current_frame = self.viewer.dims.current_step[0]
        new_frame = max(0, min(len(self.dlc_df) - 1, current_frame + delta))
        self.viewer.dims.set_point(0, new_frame)

    def _on_frame_changed(self, event):
        """Update frame label when frame changes."""
        if self.dlc_df is not None:
            current_frame = self.viewer.dims.current_step[0]
            total_frames = len(self.dlc_df)
            self.frame_label.setText(f"Frame: {current_frame} / {total_frames}")

    def _update_stats(self):
        """Update statistics label."""
        if self.features_data is None:
            return

        summary = self.features_data.get('summary', {})
        total_reaches = summary.get('total_reaches', 0)
        causal_reaches = summary.get('causal_reaches', 0)
        outcome_counts = summary.get('outcome_counts', {})

        stats_text = f"Total reaches: {total_reaches}\n"
        stats_text += f"Causal reaches: {causal_reaches}\n\n"
        stats_text += "Outcomes:\n"
        for outcome, count in outcome_counts.items():
            stats_text += f"  {outcome}: {count}\n"

        self.stats_label.setText(stats_text)


def main():
    """Launch Step 5 data viewer standalone."""
    import napari

    viewer = napari.Viewer(title="MouseReach Step 5: Feature Viewer")
    widget = DataViewerWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Step 5 - View Features", area="right")

    print("\nStep 5: Feature Viewer")
    print("=" * 40)
    print("View reach-outcome data overlaid on video.")
    print("Space=play, 1-5=speed, N/P=segments")
    print()

    napari.run()


if __name__ == "__main__":
    main()
