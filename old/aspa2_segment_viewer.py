"""
ASPA2 Segment Viewer - Visual Verification
===========================================

Opens video in napari with segment overlay.
As you scrub through frames, shows current segment info.

Usage:
    python aspa2_segment_viewer.py video.mp4 segments.json
    
Or in Python:
    from aspa2_segment_viewer import view_segments
    view_segments('video.mp4', 'segments.json')
"""

import numpy as np
import json
from pathlib import Path
from typing import Union, Optional


def view_segments(video_path: Union[str, Path],
                  segments_path: Optional[Union[str, Path]] = None,
                  dlc_path: Optional[Union[str, Path]] = None):
    """
    Open video in napari with segment boundaries displayed.
    
    Args:
        video_path: Path to video file
        segments_path: Path to segments JSON file (optional - will compute if not provided)
        dlc_path: Path to DLC file (required if segments_path not provided)
    """
    import napari
    import dask.array as da
    from dask import delayed
    import cv2
    
    video_path = Path(video_path)
    
    # Load or compute segments
    if segments_path is not None:
        with open(segments_path) as f:
            seg_data = json.load(f)
        boundaries = seg_data['boundaries']
        fps = seg_data.get('fps', 60.0)
    elif dlc_path is not None:
        from aspa2_segment_finder import segment_video
        result = segment_video(dlc_path)
        boundaries = result.boundaries
        fps = result.fps
    else:
        raise ValueError("Must provide either segments_path or dlc_path")
    
    # Build segment lookup: frame -> (segment_index, segment_name)
    boundaries = [0] + boundaries + [999999999]  # Add start/end
    
    def get_segment_info(frame_idx):
        """Get segment info for a frame."""
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= frame_idx < boundaries[i + 1]:
                if i == 0:
                    return 0, "garbage_pre"
                elif i == len(boundaries) - 2:
                    return 21, "garbage_post"
                else:
                    return i, f"pellet_{i}"
        return -1, "unknown"
    
    # Lazy video loading with dask
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    @delayed
    def read_frame(frame_idx):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create lazy dask array
    lazy_frames = [da.from_delayed(read_frame(i), 
                                    shape=(height, width, 3), 
                                    dtype=np.uint8) 
                   for i in range(n_frames)]
    video_stack = da.stack(lazy_frames, axis=0)
    
    # Create napari viewer
    viewer = napari.Viewer(title=f'Segment Viewer: {video_path.name}')
    
    # Add video layer
    layer = viewer.add_image(video_stack, name='video', rgb=True)
    
    # Create text overlay for segment info
    @viewer.dims.events.current_step.connect
    def on_frame_change(event):
        frame_idx = viewer.dims.current_step[0]
        seg_idx, seg_name = get_segment_info(frame_idx)
        
        # Calculate time
        time_sec = frame_idx / fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        
        # Update window title with segment info
        viewer.title = (f'Frame {frame_idx}/{n_frames} | '
                       f'{mins}:{secs:05.2f} | '
                       f'Segment {seg_idx}: {seg_name}')
    
    # Add boundary markers as points layer
    boundary_points = [[b, height // 2, width // 2] for b in boundaries[1:-1]]
    if boundary_points:
        # Add vertical lines at boundaries (as shapes)
        boundary_frames = boundaries[1:-1]  # Exclude 0 and end
        
        # Create shapes for boundary markers (lines spanning height)
        shapes_data = []
        for b in boundary_frames:
            # Vertical line at boundary frame
            line = np.array([[b, 0, 0], [b, 0, width]])
            shapes_data.append(line)
        
        if shapes_data:
            viewer.add_shapes(shapes_data,
                            shape_type='line',
                            edge_color='red',
                            edge_width=3,
                            name='boundaries')
    
    # Trigger initial update
    on_frame_change(None)
    
    print(f"\nControls:")
    print(f"  - Drag slider to scrub through video")
    print(f"  - Arrow keys to step frame by frame")
    print(f"  - Segment info shown in window title")
    print(f"\nBoundaries at frames: {boundaries[1:-1]}")
    
    napari.run()


def view_with_dlc(video_path: Union[str, Path], dlc_path: Union[str, Path]):
    """
    Convenience function: compute segments from DLC file and view.
    """
    from aspa2_segment_finder import segment_video
    import tempfile
    
    # Compute segments
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        seg_path = f.name
    
    result = segment_video(dlc_path, output_path=Path(seg_path))
    
    # View
    view_segments(video_path, seg_path)


def interactive_select():
    """
    Interactive file selection using tkinter dialogs.
    Returns (video_path, dlc_path) or (video_path, segments_path).
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Ask for directory or let user pick files
    result = messagebox.askyesno(
        "ASPA2 Segment Viewer",
        "Do you want to select a directory containing video and DLC files?\n\n"
        "Yes = Select directory (will auto-match files)\n"
        "No = Select files individually"
    )
    
    if result:  # Select directory
        directory = filedialog.askdirectory(title="Select directory with video and DLC files")
        if not directory:
            print("No directory selected")
            return None, None
        
        directory = Path(directory)
        
        # Find video files
        video_files = list(directory.glob("*.mp4")) + list(directory.glob("*.avi"))
        if not video_files:
            messagebox.showerror("Error", "No video files (.mp4, .avi) found in directory")
            return None, None
        
        # Find DLC files
        dlc_files = list(directory.glob("*DLC*.h5")) + list(directory.glob("*DLC*.csv"))
        if not dlc_files:
            messagebox.showerror("Error", "No DLC files found in directory")
            return None, None
        
        # If multiple, let user choose
        if len(video_files) > 1:
            # Simple selection - just show list
            video_names = [f.name for f in video_files]
            print("\nMultiple videos found:")
            for i, name in enumerate(video_names):
                print(f"  {i+1}. {name}")
            choice = input("Enter number (or press Enter for first): ").strip()
            if choice:
                video_path = video_files[int(choice) - 1]
            else:
                video_path = video_files[0]
        else:
            video_path = video_files[0]
        
        if len(dlc_files) > 1:
            dlc_names = [f.name for f in dlc_files]
            print("\nMultiple DLC files found:")
            for i, name in enumerate(dlc_names):
                print(f"  {i+1}. {name}")
            choice = input("Enter number (or press Enter for first): ").strip()
            if choice:
                dlc_path = dlc_files[int(choice) - 1]
            else:
                dlc_path = dlc_files[0]
        else:
            dlc_path = dlc_files[0]
        
        print(f"\nSelected video: {video_path.name}")
        print(f"Selected DLC: {dlc_path.name}")
        
        return video_path, dlc_path
    
    else:  # Select files individually
        video_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi"), ("All files", "*.*")]
        )
        if not video_path:
            print("No video selected")
            return None, None
        
        # Ask if they have segments.json or DLC file
        has_segments = messagebox.askyesno(
            "Segments",
            "Do you have a pre-computed segments.json file?\n\n"
            "Yes = Select segments.json\n"
            "No = Select DLC file (will compute segments)"
        )
        
        if has_segments:
            segments_path = filedialog.askopenfilename(
                title="Select segments JSON file",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not segments_path:
                print("No segments file selected")
                return None, None
            return Path(video_path), Path(segments_path)
        else:
            dlc_path = filedialog.askopenfilename(
                title="Select DLC file",
                filetypes=[("DLC files", "*.h5 *.csv"), ("All files", "*.*")]
            )
            if not dlc_path:
                print("No DLC file selected")
                return None, None
            return Path(video_path), Path(dlc_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 3:
        # Command line mode
        video_path = sys.argv[1]
        
        if len(sys.argv) >= 4 and sys.argv[2] == '--dlc':
            view_with_dlc(video_path, sys.argv[3])
        else:
            view_segments(video_path, sys.argv[2])
    
    elif len(sys.argv) == 2:
        # Just video provided - need to find DLC/segments
        video_path = Path(sys.argv[1])
        
        # Look for matching DLC file in same directory
        dlc_files = list(video_path.parent.glob("*DLC*.h5"))
        seg_files = list(video_path.parent.glob("*_segments.json"))
        
        if seg_files:
            print(f"Found segments file: {seg_files[0].name}")
            view_segments(video_path, seg_files[0])
        elif dlc_files:
            print(f"Found DLC file: {dlc_files[0].name}")
            view_with_dlc(video_path, dlc_files[0])
        else:
            print("No DLC or segments file found in same directory")
            print("Launching interactive selector...")
            video_path, other_path = interactive_select()
            if video_path and other_path:
                if str(other_path).endswith('.json'):
                    view_segments(video_path, other_path)
                else:
                    view_with_dlc(video_path, other_path)
    
    else:
        # Interactive mode
        print("ASPA2 Segment Viewer")
        print("=" * 40)
        video_path, other_path = interactive_select()
        
        if video_path and other_path:
            if str(other_path).endswith('.json'):
                view_segments(video_path, other_path)
            else:
                view_with_dlc(video_path, other_path)
