"""
ASPA2 Model Viewer (Napari)
==========================

PURPOSE:
    Visualize DLC tracking data with model overlays in napari.
    Shows:
    - Original video frames
    - Box model (fixed Reference, BOXL, BOXR points)
    - SA rectangle model (moves with tray)
    - Pellet/pillar positions
    - Per-pellet refined models during still periods
    - Current pellet number indicator

INPUT:
    - Video file (.mp4, .avi, etc.)
    - DLC output file (.csv or .h5)
    - Segmentation JSON (from aspa2_pellet_segmenter.py)
    - Per-pellet models JSON (from aspa2_pellet_model_builder.py) [optional]

OUTPUT:
    - Interactive napari viewer with layers
    - Optional: export annotated video

USAGE:
    Command line:
        python aspa2_model_viewer.py                    # Opens file dialogs
        python aspa2_model_viewer.py video.mp4 dlc.csv seg.json
        python aspa2_model_viewer.py video.mp4 dlc.csv seg.json --pellet-models models.json
    
    As module:
        from aspa2_model_viewer import ASPA2Viewer
        viewer = ASPA2Viewer(video_path, dlc_path, seg_path)
        viewer.run()

LAYERS:
    1. Video frames (image layer)
    2. DLC tracked points (points layer, color-coded by likelihood)
    3. Box model - fixed points (shapes layer, cyan)
    4. SA rectangle model - video-wide (shapes layer, yellow)
    5. Per-pellet SA model - refined (shapes layer, green) [optional]
    6. Pellet position marker (points layer, red)
    7. Pellet number text overlay

CONTROLS:
    - Scroll through frames with slider
    - Toggle layers on/off
    - Jump to pellet N with number keys or dropdown
    - Play/pause video

DEPENDENCIES:
    napari, numpy, pandas, scipy, opencv-python, tkinter
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import cv2

try:
    import napari
    from napari.layers import Image, Points, Shapes
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False
    print("WARNING: napari not installed. Install with: pip install napari[all]")


# Colors for different elements - using string names for napari compatibility
COLORS = {
    'box': 'cyan',
    'sa_global': 'yellow',
    'sa_pellet': 'green', 
    'pellet': 'red',
    'pillar': 'orange',
    'dlc_good': 'lime',
    'dlc_bad': 'red',
}


def load_dlc_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DLC output file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.h5':
        df = pd.read_hdf(filepath)
    elif filepath.suffix in ['.csv', '.gz']:
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
    return df


def load_json(filepath: Union[str, Path]) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class VideoReader:
    """Simple video reader with frame caching."""
    
    def __init__(self, video_path: Union[str, Path], cache_size: int = 100):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.cache = {}
        self.cache_size = cache_size
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get a single frame (with caching)."""
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Cache management
        if len(self.cache) >= self.cache_size:
            oldest = min(self.cache.keys())
            del self.cache[oldest]
        
        self.cache[frame_idx] = frame
        return frame
    
    def get_frames(self, start: int, end: int) -> np.ndarray:
        """Get a range of frames as 3D array."""
        frames = []
        for i in range(start, end):
            frames.append(self.get_frame(i))
        return np.array(frames)
    
    def close(self):
        self.cap.release()


class ASPA2Viewer:
    """Main viewer class for ASPA data visualization."""
    
    def __init__(self, 
                 video_path: Union[str, Path],
                 dlc_path: Union[str, Path],
                 segmentation_path: Union[str, Path],
                 pellet_models_path: Optional[Union[str, Path]] = None):
        
        self.video_path = Path(video_path)
        self.dlc_path = Path(dlc_path)
        self.segmentation_path = Path(segmentation_path)
        self.pellet_models_path = Path(pellet_models_path) if pellet_models_path else None
        
        # Load data
        print("Loading data...")
        self.video = VideoReader(video_path)
        self.dlc_df = load_dlc_data(dlc_path)
        self.segmentation = load_json(segmentation_path)
        
        if self.pellet_models_path:
            self.pellet_models = load_json(pellet_models_path)
        else:
            self.pellet_models = None
        
        self.n_frames = len(self.dlc_df)
        self.current_frame = 0
        self.current_pellet = 1
        
        # Extract models
        self.box_model = self.segmentation['box_model']
        self.sa_model = self.segmentation['sa_model']
        self.pellet_segments = self.segmentation['pellet_segments']
        
        print(f"Loaded: {self.n_frames} frames, {len(self.pellet_segments)} pellets")
        
        self.viewer = None
    
    def get_pellet_for_frame(self, frame: int) -> int:
        """Get which pellet number a frame belongs to."""
        for seg in self.pellet_segments:
            if seg['start_frame'] <= frame <= seg['end_frame']:
                return seg['pellet_num']
        return 0
    
    def get_box_shapes(self) -> Tuple[np.ndarray, List[str]]:
        """Get box model as shapes (points + lines)."""
        # Box points
        ref = [self.box_model['reference_y'], self.box_model['reference_x']]
        boxl = [self.box_model['boxl_y'], self.box_model['boxl_x']]
        boxr = [self.box_model['boxr_y'], self.box_model['boxr_x']]
        
        # Line connecting BOXL and BOXR (the opening)
        opening_line = np.array([boxl, boxr])
        
        return np.array([opening_line]), ['line']
    
    def get_sa_rectangle(self, frame: int) -> Optional[np.ndarray]:
        """Get SA rectangle for a specific frame."""
        row = self.dlc_df.iloc[frame]
        
        # Check if we have good tracking
        if row['SABL_likelihood'] < 0.5 or row['SABR_likelihood'] < 0.5:
            return None
        
        # Get corners from DLC data
        sabl = [row['SABL_y'], row['SABL_x']]
        sabr = [row['SABR_y'], row['SABR_x']]
        
        # Use model dimensions for top corners if not well tracked
        if row['SATL_likelihood'] > 0.5:
            satl = [row['SATL_y'], row['SATL_x']]
        else:
            satl = [sabl[0] + self.sa_model['height'], sabl[1]]
        
        if row['SATR_likelihood'] > 0.5:
            satr = [row['SATR_y'], row['SATR_x']]
        else:
            satr = [sabr[0] + self.sa_model['height'], sabr[1]]
        
        # Rectangle as polygon (counter-clockwise)
        return np.array([sabl, sabr, satr, satl])
    
    def get_pellet_model_rectangle(self, pellet_num: int) -> Optional[np.ndarray]:
        """Get per-pellet refined rectangle model."""
        if self.pellet_models is None:
            return None
        
        for pm in self.pellet_models['pellet_models']:
            if pm['pellet_num'] == pellet_num:
                rect = pm['rectangle']
                return np.array([
                    [rect['sabl_y'], rect['sabl_x']],
                    [rect['sabr_y'], rect['sabr_x']],
                    [rect['satr_y'], rect['satr_x']],
                    [rect['satl_y'], rect['satl_x']],
                ])
        return None
    
    def get_dlc_points(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get DLC tracked points with colors based on likelihood."""
        row = self.dlc_df.iloc[frame]
        
        points = []
        colors = []
        
        # All tracked body parts
        bodyparts = ['SABL', 'SABR', 'SATL', 'SATR', 'Pellet', 'Pillar',
                     'Reference', 'BOXL', 'BOXR']
        
        for bp in bodyparts:
            x_col = f'{bp}_x'
            y_col = f'{bp}_y'
            l_col = f'{bp}_likelihood'
            
            if x_col in row.index:
                x, y, like = row[x_col], row[y_col], row[l_col]
                points.append([y, x])  # napari uses y, x order
                
                # Color based on likelihood
                if like > 0.9:
                    colors.append(COLORS['dlc_good'])
                else:
                    colors.append(COLORS['dlc_bad'])
        
        return np.array(points), np.array(colors)
    
    def get_pellet_marker(self, frame: int) -> Optional[np.ndarray]:
        """Get pellet position marker."""
        row = self.dlc_df.iloc[frame]
        
        if row['Pellet_likelihood'] > 0.5:
            return np.array([[row['Pellet_y'], row['Pellet_x']]])
        return None
    
    def update_frame(self, frame: int):
        """Update all layers for a new frame."""
        self.current_frame = frame
        self.current_pellet = self.get_pellet_for_frame(frame)
        
        # Update SA corner points
        corners = self.get_sa_corner_points(frame)
        for name in ['SABL', 'SABR', 'SATL', 'SATR']:
            if name in self.corner_layers:
                self.corner_layers[name].data = np.array([corners[name]['pos']])
        
        # Update general model rectangle
        gen_rect = self.get_general_model_rectangle(frame)
        if gen_rect is not None:
            self.general_model_layer.data = [gen_rect]
        
        # Update SA rectangle (DLC live)
        sa_rect = self.get_sa_rectangle(frame)
        if sa_rect is not None:
            self.sa_layer.data = [sa_rect]
        
        # Update per-pellet model rectangle
        if self.pellet_models:
            pm_rect = self.get_pellet_model_rectangle(self.current_pellet)
            if pm_rect is not None:
                self.pellet_model_layer.data = [pm_rect]
            
            # Update pellet model position
            for pm in self.pellet_models['pellet_models']:
                if pm['pellet_num'] == self.current_pellet:
                    if pm.get('pellet_x') is not None:
                        self.pellet_model_pos_layer.data = np.array([[pm['pellet_y'], pm['pellet_x']]])
                        self.pellet_model_pos_layer.visible = True
                    else:
                        self.pellet_model_pos_layer.visible = False
                    break
        
        # Update pellet marker (DLC live)
        pellet_pos = self.get_pellet_marker(frame)
        if pellet_pos is not None:
            self.pellet_marker_layer.data = pellet_pos
            self.pellet_marker_layer.visible = True
        else:
            self.pellet_marker_layer.visible = False
        
        # Update title
        self.viewer.title = f"ASPA2 Viewer - Frame {frame} - Pellet {self.current_pellet}"
    
    def get_sa_corner_points(self, frame: int) -> dict:
        """Get individual SA corner points with labels."""
        row = self.dlc_df.iloc[frame]
        corners = {}
        for name in ['SABL', 'SABR', 'SATL', 'SATR']:
            x, y = row[f'{name}_x'], row[f'{name}_y']
            like = row[f'{name}_likelihood']
            corners[name] = {'pos': [y, x], 'likelihood': like}
        return corners
    
    def get_general_model_rectangle(self, frame: int) -> Optional[np.ndarray]:
        """Get the general (video-wide) SA model rectangle positioned at current SA location."""
        row = self.dlc_df.iloc[frame]
        
        # Need at least SABL to anchor the model
        if row['SABL_likelihood'] < 0.5:
            return None
        
        # Anchor at SABL position
        sabl_y, sabl_x = row['SABL_y'], row['SABL_x']
        
        # Use video-wide model dimensions
        width = self.sa_model['width']
        height = self.sa_model['height']
        
        # Build rectangle from model
        return np.array([
            [sabl_y, sabl_x],                    # SABL
            [sabl_y, sabl_x + width],            # SABR
            [sabl_y + height, sabl_x + width],   # SATR
            [sabl_y + height, sabl_x],           # SATL
        ])

    def run(self):
        """Launch the napari viewer."""
        if not HAS_NAPARI:
            print("ERROR: napari not installed. Cannot run viewer.")
            return
        
        print("Launching viewer...")
        
        # Create viewer
        self.viewer = napari.Viewer(title=f"ASPA2 Viewer - {self.video_path.name}")
        
        # Load first frame
        first_frame = self.video.get_frame(0)
        
        # Add video layer
        self.video_layer = self.viewer.add_image(
            first_frame,
            name='Video',
            colormap='gray' if first_frame.ndim == 2 else None
        )
        
        # === BOX MODEL (FIXED) ===
        box_shapes, box_types = self.get_box_shapes()
        self.box_layer = self.viewer.add_shapes(
            box_shapes,
            shape_type=box_types,
            edge_color='cyan',
            face_color='transparent',
            edge_width=3,
            name='BOX: Opening Line'
        )
        
        # Box reference points
        box_points = np.array([
            [self.box_model['reference_y'], self.box_model['reference_x']],
            [self.box_model['boxl_y'], self.box_model['boxl_x']],
            [self.box_model['boxr_y'], self.box_model['boxr_x']],
        ])
        self.box_points_layer = self.viewer.add_points(
            box_points,
            face_color='cyan',
            size=10,
            name='BOX: REF, BOXL, BOXR'
        )
        
        # === SA CORNERS (LIVE DLC TRACKING) - Individual layers ===
        corners = self.get_sa_corner_points(0)
        corner_names = ['SABL', 'SABR', 'SATL', 'SATR']
        corner_colors = ['orange', 'darkorange', 'goldenrod', 'gold']
        
        self.corner_layers = {}
        for i, name in enumerate(corner_names):
            pos = np.array([corners[name]['pos']])
            layer = self.viewer.add_points(
                pos,
                face_color=corner_colors[i],
                size=12,
                name=f'SA: {name} (DLC)'
            )
            self.corner_layers[name] = layer
        
        # === GENERAL MODEL RECTANGLE (VIDEO-WIDE) ===
        gen_rect = self.get_general_model_rectangle(0)
        if gen_rect is not None:
            self.general_model_layer = self.viewer.add_shapes(
                [gen_rect],
                shape_type='polygon',
                edge_color='yellow',
                face_color='transparent',
                edge_width=3,
                name='MODEL: General (Video-Wide SA)'
            )
        else:
            self.general_model_layer = self.viewer.add_shapes(
                [],
                shape_type='polygon',
                edge_color='yellow',
                name='MODEL: General (Video-Wide SA)'
            )
        
        # === SA RECTANGLE FROM DLC (LIVE) ===
        sa_rect = self.get_sa_rectangle(0)
        if sa_rect is not None:
            self.sa_layer = self.viewer.add_shapes(
                [sa_rect],
                shape_type='polygon',
                edge_color='orange',
                face_color='transparent',
                edge_width=2,
                name='SA: Rectangle (DLC Live)'
            )
        else:
            self.sa_layer = self.viewer.add_shapes(
                [],
                shape_type='polygon',
                edge_color='orange',
                name='SA: Rectangle (DLC Live)'
            )
        
        # === PER-PELLET MODEL RECTANGLE ===
        if self.pellet_models:
            pm_rect = self.get_pellet_model_rectangle(1)
            if pm_rect is not None:
                self.pellet_model_layer = self.viewer.add_shapes(
                    [pm_rect],
                    shape_type='polygon',
                    edge_color='lime',
                    face_color='transparent',
                    edge_width=3,
                    name='MODEL: Per-Pellet (Still Period)'
                )
            else:
                self.pellet_model_layer = self.viewer.add_shapes(
                    [],
                    shape_type='polygon',
                    edge_color='lime',
                    name='MODEL: Per-Pellet (Still Period)'
                )
        else:
            self.pellet_model_layer = self.viewer.add_shapes(
                [],
                name='MODEL: Per-Pellet (Still Period)'
            )
        
        # === PELLET POSITION (DLC LIVE) ===
        pellet_pos = self.get_pellet_marker(0)
        self.pellet_marker_layer = self.viewer.add_points(
            pellet_pos if pellet_pos is not None else np.empty((0, 2)),
            face_color='red',
            size=15,
            symbol='star',
            name='PELLET: Position (DLC Live)'
        )
        
        # === PELLET MODEL POSITION (from still period) ===
        if self.pellet_models:
            pm = self.pellet_models['pellet_models'][0]
            if pm.get('pellet_x') is not None:
                model_pellet_pos = np.array([[pm['pellet_y'], pm['pellet_x']]])
                self.pellet_model_pos_layer = self.viewer.add_points(
                    model_pellet_pos,
                    face_color='lime',
                    size=12,
                    symbol='diamond',
                    name='PELLET: Model Position (Expected)'
                )
            else:
                self.pellet_model_pos_layer = self.viewer.add_points(
                    np.empty((0, 2)),
                    name='PELLET: Model Position (Expected)'
                )
        else:
            self.pellet_model_pos_layer = self.viewer.add_points(
                np.empty((0, 2)),
                name='PELLET: Model Position (Expected)'
            )
        
        # Add frame slider
        @self.viewer.dims.events.current_step.connect
        def on_frame_change(event):
            frame = event.value[0] if isinstance(event.value, tuple) else event.value
            if 0 <= frame < self.n_frames:
                # Update video frame
                self.video_layer.data = self.video.get_frame(frame)
                self.update_frame(frame)
        
        # Set up dims for frame scrubbing
        self.viewer.dims.ndim = 3
        self.viewer.dims.set_range(0, (0, self.n_frames - 1, 1))
        
        # Add keyboard shortcuts
        @self.viewer.bind_key('Right')
        def next_frame(viewer):
            current = viewer.dims.current_step[0]
            viewer.dims.set_current_step(0, min(current + 1, self.n_frames - 1))
        
        @self.viewer.bind_key('Left')
        def prev_frame(viewer):
            current = viewer.dims.current_step[0]
            viewer.dims.set_current_step(0, max(current - 1, 0))
        
        @self.viewer.bind_key('Shift-Right')
        def next_pellet(viewer):
            for seg in self.pellet_segments:
                if seg['start_frame'] > self.current_frame:
                    viewer.dims.set_current_step(0, seg['start_frame'])
                    break
        
        @self.viewer.bind_key('Shift-Left')
        def prev_pellet(viewer):
            for seg in reversed(self.pellet_segments):
                if seg['start_frame'] < self.current_frame:
                    viewer.dims.set_current_step(0, seg['start_frame'])
                    break
        
        print("Viewer ready!")
        print("Controls:")
        print("  Left/Right arrows: Previous/Next frame")
        print("  Shift+Left/Right: Jump to previous/next pellet")
        print("  Layer visibility: Toggle in layer panel")
        
        napari.run()
    
    def export_video(self, output_path: Union[str, Path], 
                     start_frame: int = 0, end_frame: Optional[int] = None,
                     fps: float = 30.0):
        """Export annotated video."""
        output_path = Path(output_path)
        
        if end_frame is None:
            end_frame = self.n_frames
        
        print(f"Exporting frames {start_frame} to {end_frame}...")
        
        # Set up video writer
        first_frame = self.video.get_frame(0)
        h, w = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame_idx in range(start_frame, end_frame):
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{end_frame}")
            
            # Get frame
            frame = self.video.get_frame(frame_idx).copy()
            
            # Draw annotations
            frame = self._draw_annotations(frame, frame_idx)
            
            # Write
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Exported: {output_path}")
    
    def _draw_annotations(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw all annotations on a frame."""
        pellet_num = self.get_pellet_for_frame(frame_idx)
        
        # Draw box opening line
        boxl = (int(self.box_model['boxl_x']), int(self.box_model['boxl_y']))
        boxr = (int(self.box_model['boxr_x']), int(self.box_model['boxr_y']))
        cv2.line(frame, boxl, boxr, (0, 255, 255), 2)
        
        # Draw SA rectangle from DLC
        sa_rect = self.get_sa_rectangle(frame_idx)
        if sa_rect is not None:
            pts = sa_rect[:, ::-1].astype(np.int32)  # y,x to x,y
            cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
        
        # Draw per-pellet model rectangle
        if self.pellet_models:
            pm_rect = self.get_pellet_model_rectangle(pellet_num)
            if pm_rect is not None:
                pts = pm_rect[:, ::-1].astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Draw pellet marker
        row = self.dlc_df.iloc[frame_idx]
        if row['Pellet_likelihood'] > 0.5:
            pos = (int(row['Pellet_x']), int(row['Pellet_y']))
            cv2.circle(frame, pos, 8, (255, 0, 0), -1)
        
        # Draw pellet number
        cv2.putText(frame, f"Pellet {pellet_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {frame_idx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        return frame


def select_files_dialog() -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Open dialogs to select all required files."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        # Select video
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if not video_path:
            return None, None, None, None
        
        # Select DLC file
        dlc_path = filedialog.askopenfilename(
            title="Select DLC Output File",
            filetypes=[
                ("DLC files", "*.csv *.csv.gz *.h5"),
                ("All files", "*.*")
            ]
        )
        if not dlc_path:
            return None, None, None, None
        
        # Select segmentation
        seg_path = filedialog.askopenfilename(
            title="Select Segmentation JSON",
            filetypes=[
                ("JSON files", "*_segmentation.json *.json"),
                ("All files", "*.*")
            ]
        )
        if not seg_path:
            return None, None, None, None
        
        # Optional: pellet models
        pellet_path = filedialog.askopenfilename(
            title="Select Per-Pellet Models JSON (optional - Cancel to skip)",
            filetypes=[
                ("JSON files", "*_pellet_models.json *.json"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        return (
            Path(video_path),
            Path(dlc_path),
            Path(seg_path),
            Path(pellet_path) if pellet_path else None
        )
        
    except ImportError:
        print("tkinter not available. Please provide paths as arguments.")
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="View ASPA tracking data with model overlays in napari"
    )
    parser.add_argument('video_path', nargs='?', help="Path to video file")
    parser.add_argument('dlc_path', nargs='?', help="Path to DLC output file")
    parser.add_argument('segmentation_path', nargs='?', help="Path to segmentation JSON")
    parser.add_argument('--pellet-models', '-p', help="Path to per-pellet models JSON")
    parser.add_argument('--export', '-e', help="Export annotated video to path")
    parser.add_argument('--start', type=int, default=0, help="Start frame for export")
    parser.add_argument('--end', type=int, help="End frame for export")
    parser.add_argument('--fps', type=float, default=30.0, help="Export FPS")
    
    args = parser.parse_args()
    
    # Get paths
    if args.video_path and args.dlc_path and args.segmentation_path:
        video_path = Path(args.video_path)
        dlc_path = Path(args.dlc_path)
        seg_path = Path(args.segmentation_path)
        pellet_path = Path(args.pellet_models) if args.pellet_models else None
    else:
        video_path, dlc_path, seg_path, pellet_path = select_files_dialog()
    
    if video_path is None:
        print("Files not selected. Exiting.")
        return
    
    # Create viewer
    viewer = ASPA2Viewer(video_path, dlc_path, seg_path, pellet_path)
    
    if args.export:
        viewer.export_video(args.export, args.start, args.end, args.fps)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
