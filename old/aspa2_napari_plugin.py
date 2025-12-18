"""
ASPA2 Segment Viewer - Napari Plugin
=====================================

Install as napari plugin or run standalone.

Usage:
    # Standalone
    python aspa2_napari_plugin.py
    
    # Or from napari: Plugins > ASPA2 Segment Viewer
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional
import warnings

# Suppress some warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def load_video_frames(video_path: Path, max_frames: Optional[int] = None):
    """Load video frames into numpy array."""
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if max_frames:
        n_frames = min(n_frames, max_frames)
    
    print(f"Loading {n_frames} frames ({n_frames/fps:.1f}s)...")
    
    frames = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i+1}/{n_frames} frames...")
    
    cap.release()
    print(f"  Done loading {len(frames)} frames")
    
    return np.stack(frames), fps


def get_segment_info(frame_idx: int, boundaries: list) -> tuple:
    """Get segment index and name for a frame."""
    # boundaries should be [b1, b2, ..., b21]
    # Segment 0 = garbage_pre (frames 0 to b1)
    # Segment 1-20 = pellet_1 to pellet_20
    # Segment 21 = garbage_post (frames after b21)
    
    for i, b in enumerate(boundaries):
        if frame_idx < b:
            if i == 0:
                return 0, "garbage_pre"
            else:
                return i, f"pellet_{i}"
    
    return 21, "garbage_post"


class SegmentViewerWidget:
    """Widget showing current segment info."""
    
    def __init__(self, napari_viewer, boundaries: list, fps: float):
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QFont
        
        self.viewer = napari_viewer
        self.boundaries = boundaries
        self.fps = fps
        
        # Create widget
        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)
        
        # Big segment label
        self.segment_label = QLabel("Segment: --")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.segment_label.setFont(font)
        self.segment_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.segment_label)
        
        # Frame info label
        self.frame_label = QLabel("Frame: --")
        self.frame_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.frame_label)
        
        # Time label
        self.time_label = QLabel("Time: --")
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)
        
        # Boundary info
        self.boundary_label = QLabel("Boundaries: --")
        self.boundary_label.setWordWrap(True)
        layout.addWidget(self.boundary_label)
        
        # Show boundary info
        self.boundary_label.setText(
            f"Boundaries at frames:\n" + 
            ", ".join(str(b) for b in boundaries[:7]) + 
            "\n... " +
            ", ".join(str(b) for b in boundaries[-3:])
        )
        
        # Connect to dimension changes
        self.viewer.dims.events.current_step.connect(self.on_frame_change)
        
        # Initial update
        self.on_frame_change(None)
    
    def on_frame_change(self, event):
        """Update labels when frame changes."""
        frame_idx = self.viewer.dims.current_step[0]
        seg_idx, seg_name = get_segment_info(frame_idx, self.boundaries)
        
        time_sec = frame_idx / self.fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        
        self.segment_label.setText(f"Segment {seg_idx}: {seg_name}")
        self.frame_label.setText(f"Frame: {frame_idx}")
        self.time_label.setText(f"Time: {mins}:{secs:05.2f}")
        
        # Color code by segment type
        if "garbage" in seg_name:
            self.segment_label.setStyleSheet("color: gray;")
        else:
            self.segment_label.setStyleSheet("color: green;")


def view_segments(video_path: Path, boundaries: list, fps: float = 60.0):
    """Open napari viewer with video and segment overlay."""
    import napari
    
    print(f"\nOpening: {video_path.name}")
    
    # Load video
    frames, video_fps = load_video_frames(video_path)
    if video_fps > 0:
        fps = video_fps
    
    # Create viewer
    viewer = napari.Viewer(title=f'ASPA2: {video_path.name}')
    
    # Add video
    viewer.add_image(frames, name='video', rgb=True)
    
    # Add segment widget
    widget = SegmentViewerWidget(viewer, boundaries, fps)
    viewer.window.add_dock_widget(widget.widget, name='Segment Info', area='right')
    
    print("\nControls:")
    print("  - Drag slider or use arrow keys to navigate")
    print("  - Segment info updates in right panel")
    print("  - Close window when done")
    
    napari.run()


def interactive_select():
    """Interactive file selection."""
    import tkinter as tk
    from tkinter import filedialog, messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    result = messagebox.askyesno(
        "ASPA2 Segment Viewer",
        "Select a directory containing video and DLC files?\n\n"
        "Yes = Select directory\n"
        "No = Select files individually"
    )
    
    if result:  # Directory
        directory = filedialog.askdirectory(title="Select directory")
        if not directory:
            return None, None
        
        directory = Path(directory)
        
        # Find video files
        videos = list(directory.glob("*.mp4")) + list(directory.glob("*.avi"))
        if not videos:
            messagebox.showerror("Error", "No video files found")
            return None, None
        
        # Find DLC files (prefer .h5 over .csv)
        dlc_files = list(directory.glob("*DLC*.h5"))
        if not dlc_files:
            dlc_files = list(directory.glob("*DLC*.csv"))
        if not dlc_files:
            messagebox.showerror("Error", "No DLC files found")
            return None, None
        
        # Let user choose if multiple
        if len(videos) > 1:
            print("\nVideos found:")
            for i, v in enumerate(videos):
                print(f"  {i+1}. {v.name}")
            choice = input("Enter number (or Enter for first): ").strip()
            video = videos[int(choice)-1] if choice else videos[0]
        else:
            video = videos[0]
        
        # Try to match DLC to video by name
        video_stem = video.stem.replace("_P1", "").replace("_P2", "").replace("_P3", "").replace("_P4", "")
        matching_dlc = [d for d in dlc_files if video_stem in d.stem]
        
        if matching_dlc:
            dlc = matching_dlc[0]
        elif len(dlc_files) > 1:
            print("\nDLC files found:")
            for i, d in enumerate(dlc_files):
                print(f"  {i+1}. {d.name}")
            choice = input("Enter number (or Enter for first): ").strip()
            dlc = dlc_files[int(choice)-1] if choice else dlc_files[0]
        else:
            dlc = dlc_files[0]
        
        print(f"\nSelected: {video.name}")
        print(f"     DLC: {dlc.name}")
        
        return video, dlc
    
    else:  # Individual files
        video = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video", "*.mp4 *.avi"), ("All", "*.*")]
        )
        if not video:
            return None, None
        
        dlc = filedialog.askopenfilename(
            title="Select DLC file",
            filetypes=[("DLC", "*.h5 *.csv"), ("All", "*.*")]
        )
        if not dlc:
            return None, None
        
        return Path(video), Path(dlc)


def main():
    """Main entry point."""
    print("ASPA2 Segment Viewer")
    print("=" * 40)
    
    video_path, dlc_path = interactive_select()
    
    if not video_path or not dlc_path:
        print("No files selected")
        return
    
    # Compute segments
    print("\nComputing segments...")
    try:
        from aspa2_segment_finder import segment_video
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            seg_path = Path(f.name)
        
        result = segment_video(dlc_path, output_path=seg_path)
        boundaries = result.boundaries
        fps = result.fps
        
    except Exception as e:
        print(f"Error computing segments: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Open viewer
    try:
        view_segments(video_path, boundaries, fps)
    except Exception as e:
        print(f"Error opening viewer: {e}")
        import traceback
        traceback.print_exc()


# Napari plugin registration (for when installed as plugin)
try:
    from napari_plugin_engine import napari_hook_implementation
    
    @napari_hook_implementation
    def napari_experimental_provide_dock_widget():
        from magicgui import magicgui
        from pathlib import Path
        
        @magicgui(
            call_button="Load Video",
            video_path={"label": "Video file", "mode": "r", "filter": "*.mp4 *.avi"},
            dlc_path={"label": "DLC file", "mode": "r", "filter": "*.h5 *.csv"},
        )
        def segment_viewer_widget(
            video_path: Path = Path("."),
            dlc_path: Path = Path("."),
        ):
            """Load video with segment overlay."""
            if not video_path.exists() or not dlc_path.exists():
                print("Please select valid files")
                return
            
            from aspa2_segment_finder import segment_video
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                seg_path = Path(f.name)
            
            result = segment_video(dlc_path, output_path=seg_path)
            
            # This would need viewer access - for now just print
            print(f"Segments computed: {len(result.boundaries)} boundaries")
            print(f"Confidence: {result.confidence}")
        
        return segment_viewer_widget
        
except ImportError:
    # napari plugin engine not available - that's fine for standalone use
    pass


if __name__ == '__main__':
    main()
