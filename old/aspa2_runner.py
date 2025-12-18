"""
ASPA2 Pipeline Runner
=====================

PURPOSE:
    Main entry point for ASPA2 (Automated Skilled Pellet Assessment) analysis.
    
WORKFLOW:
    1. Select VIDEO file(s) - this is your primary input
    2. Script automatically finds the matching DLC file
    3. Outputs saved next to the video with matching names
    4. View results by selecting the same video

GUARANTEE:
    Every video produces exactly 22 segments:
    - Segment 0: garbage_pre (tray not yet in position)
    - Segments 1-20: pellet periods
    - Segment 21: garbage_post (tray has left)

FILE NAMING:
    Video:        20250701_CNT0110.mp4
    DLC file:     20250701_CNT0110*DLC*.csv (auto-found)
    Segmentation: 20250701_CNT0110_segmentation.json
    Models:       20250701_CNT0110_pellet_models.json

USAGE:
    GUI mode:
        python aspa2_runner.py
    
    Command line:
        python aspa2_runner.py --process video1.mp4 video2.mp4
        python aspa2_runner.py --view video.mp4

DEPENDENCIES:
    numpy, pandas, scipy, opencv-python
    Optional: napari (for viewer)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# Import ASPA2 modules - use v3 segmenter
try:
    from aspa2_segmenter_v3 import segment_video, load_dlc_data
    from aspa2_pellet_model_builder import build_pellet_models
    from aspa2_model_viewer import ASPA2Viewer
except ImportError:
    import importlib.util
    
    def import_from_file(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    script_dir = Path(__file__).parent
    segmenter = import_from_file("segmenter", script_dir / "aspa2_segmenter_v3.py")
    model_builder = import_from_file("model_builder", script_dir / "aspa2_pellet_model_builder.py")
    viewer_module = import_from_file("viewer", script_dir / "aspa2_model_viewer.py")
    
    segment_video = segmenter.segment_video
    load_dlc_data = segmenter.load_dlc_data
    build_pellet_models = model_builder.build_pellet_models
    ASPA2Viewer = viewer_module.ASPA2Viewer


def get_video_basename(video_path: Path) -> str:
    """Get the base name of a video (without extension)."""
    return video_path.stem


def get_video_fps(video_path: Path) -> float:
    """Get FPS from video file metadata."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return fps
    except Exception:
        pass
    return 60.0  # fallback default


def find_dlc_file(video_path: Path) -> Optional[Path]:
    """
    Find DLC output file matching a video.
    Searches in same directory and common subdirectories.
    """
    video_dir = video_path.parent
    video_base = get_video_basename(video_path)
    
    # Search patterns - DLC files contain the video name + DLC suffix
    search_dirs = [video_dir, video_dir / 'dlc', video_dir / 'DLC', video_dir.parent]
    extensions = ['.csv', '.csv.gz', '.h5']
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in extensions:
            # Look for files starting with video name and containing 'DLC'
            for f in search_dir.glob(f"{video_base}*DLC*{ext}"):
                return f
            for f in search_dir.glob(f"{video_base}*dlc*{ext}"):
                return f
    
    return None


def get_output_paths(video_path: Path) -> dict:
    """Get standard output file paths for a video."""
    video_dir = video_path.parent
    video_base = get_video_basename(video_path)
    
    return {
        'segmentation': video_dir / f"{video_base}_segmentation.json",
        'pellet_models': video_dir / f"{video_base}_pellet_models.json",
        'annotated_video': video_dir / f"{video_base}_annotated.mp4",
    }


def find_existing_outputs(video_path: Path) -> dict:
    """Find any existing output files for a video."""
    paths = get_output_paths(video_path)
    existing = {}
    for key, path in paths.items():
        if path.exists():
            existing[key] = path
    return existing


def process_video(video_path: Path, fps: Optional[float] = None, force: bool = False) -> dict:
    """
    Process a single video through the complete pipeline.
    
    Args:
        video_path: Path to video file
        fps: Video frame rate (auto-detected if None)
        force: Re-process even if outputs exist
    
    Returns:
        dict with results and paths
    """
    video_path = Path(video_path)
    video_base = get_video_basename(video_path)
    
    # Auto-detect FPS if not provided
    if fps is None:
        fps = get_video_fps(video_path)
    
    results = {
        'video': str(video_path),
        'video_name': video_base,
        'fps': fps,
        'timestamp': datetime.now().isoformat(),
    }
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"  FPS: {fps}")
    print(f"{'='*60}")
    
    # Check for existing outputs
    existing = find_existing_outputs(video_path)
    output_paths = get_output_paths(video_path)
    
    if existing.get('segmentation') and existing.get('pellet_models') and not force:
        print(f"  Already processed. Use --force to reprocess.")
        results['status'] = 'skipped'
        results.update({k: str(v) for k, v in existing.items()})
        return results
    
    # Find DLC file
    dlc_path = find_dlc_file(video_path)
    if dlc_path is None:
        print(f"  ERROR: Could not find DLC file for {video_base}")
        print(f"  Searched for: {video_base}*DLC*.csv/h5 in {video_path.parent}")
        results['status'] = 'error'
        results['error'] = 'DLC file not found'
        return results
    
    print(f"  DLC file: {dlc_path.name}")
    results['dlc_file'] = str(dlc_path)
    
    # Step 1: Segment into 22 segments (always)
    print("\n[1/2] Segmenting into 22 segments (20 pellet periods)...")
    try:
        seg_output = output_paths['segmentation']
        segmentation = segment_video(dlc_path, fps=fps, output_path=seg_output)
        
        results['segmentation'] = str(seg_output)
        results['n_segments'] = len(segmentation.segments)
        results['n_pellet_segments'] = sum(1 for s in segmentation.segments if s.segment_type == 'pellet')
        results['interval_seconds'] = segmentation.fundamental_interval_seconds
        results['segmentation_confidence'] = segmentation.overall_confidence
        results['boundaries_from_snaps'] = segmentation.n_boundaries_from_snaps
        results['boundaries_interpolated'] = segmentation.n_boundaries_interpolated
        
        print(f"   {results['n_segments']} segments, {results['n_pellet_segments']} pellet periods")
        print(f"   Interval: {results['interval_seconds']:.1f}s")
        print(f"   Confidence: {results['segmentation_confidence']:.2f} ({results['boundaries_from_snaps']}/21 from snaps)")
        
        if segmentation.flags:
            print(f"   Flags: {segmentation.flags}")
            results['segmentation_flags'] = segmentation.flags
            
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['error'] = str(e)
        return results
    
    # Step 2: Build per-pellet models
    print("\n[2/2] Building per-pellet models...")
    try:
        models_output = output_paths['pellet_models']
        models = build_pellet_models(dlc_path, seg_output, output_path=models_output)
        
        results['pellet_models'] = str(models_output)
        qualities = [m.model_quality for m in models]
        results['models_good'] = qualities.count('good')
        results['models_fair'] = qualities.count('fair')
        results['models_poor'] = qualities.count('poor')
        print(f"   Quality: {results['models_good']} good, {results['models_fair']} fair, {results['models_poor']} poor")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['model_error'] = str(e)
    
    results['status'] = 'success'
    print(f"\n   Saved: {seg_output.name}")
    print(f"   Saved: {models_output.name}")
    
    return results


def view_video(video_path: Path) -> bool:
    """
    View results for a video. Auto-finds all associated files.
    
    Returns:
        True if viewer launched successfully
    """
    video_path = Path(video_path)
    
    print(f"\nLoading: {video_path.name}")
    
    # Find DLC file
    dlc_path = find_dlc_file(video_path)
    if dlc_path is None:
        print(f"ERROR: Could not find DLC file for {video_path.name}")
        return False
    
    # Find outputs
    outputs = find_existing_outputs(video_path)
    
    if 'segmentation' not in outputs:
        print(f"ERROR: No segmentation found. Run processing first.")
        expected = get_output_paths(video_path)['segmentation']
        print(f"  Expected: {expected}")
        return False
    
    print(f"  Video: {video_path.name}")
    print(f"  DLC: {dlc_path.name}")
    print(f"  Segmentation: {outputs['segmentation'].name}")
    if 'pellet_models' in outputs:
        print(f"  Pellet models: {outputs['pellet_models'].name}")
    
    print("\nLaunching viewer...")
    
    try:
        viewer = ASPA2Viewer(
            video_path,
            dlc_path,
            outputs['segmentation'],
            outputs.get('pellet_models')
        )
        viewer.run()
        return True
    except Exception as e:
        print(f"ERROR launching viewer: {e}")
        import traceback
        traceback.print_exc()
        return False


class ASPA2Gui:
    """Simple GUI for ASPA2 pipeline."""
    
    def __init__(self):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
            self.tk = tk
            self.ttk = ttk
            self.filedialog = filedialog
            self.messagebox = messagebox
        except ImportError:
            print("ERROR: tkinter not available. Use command-line interface.")
            sys.exit(1)
        
        self.root = tk.Tk()
        self.root.title("ASPA2 - Automated Skilled Pellet Assessment")
        self.root.geometry("400x250")
        
        self._build_ui()
    
    def _build_ui(self):
        tk = self.tk
        ttk = self.ttk
        
        main = ttk.Frame(self.root, padding="20")
        main.grid(row=0, column=0, sticky="nsew")
        
        # Title
        ttk.Label(main, text="ASPA2", font=('Arial', 20, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 5)
        )
        ttk.Label(main, text="Automated Skilled Pellet Assessment", font=('Arial', 10)).grid(
            row=1, column=0, columnspan=2, pady=(0, 20)
        )
        
        # Main buttons
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Process Videos", 
                   command=self._process, width=20).pack(pady=5)
        ttk.Button(btn_frame, text="View Results", 
                   command=self._view, width=20).pack(pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Select videos to process or view\n(FPS auto-detected from video)")
        ttk.Label(main, textvariable=self.status_var, 
                  font=('Arial', 9), foreground='gray').grid(
            row=3, column=0, columnspan=2, pady=20
        )
    
    def _select_videos(self, title="Select Video Files") -> List[Path]:
        """Open dialog to select video files."""
        files = self.filedialog.askopenfilenames(
            title=title,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI"),
                ("All files", "*.*")
            ]
        )
        return [Path(f) for f in files] if files else []
    
    def _process(self):
        """Process selected videos."""
        videos = self._select_videos("Select Videos to Process")
        if not videos:
            return
        
        self.status_var.set(f"Processing {len(videos)} video(s)...")
        self.root.update()
        
        success = 0
        skipped = 0
        errors = 0
        
        for video in videos:
            try:
                result = process_video(video)  # FPS auto-detected
                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    errors += 1
            except Exception as e:
                print(f"Error processing {video.name}: {e}")
                errors += 1
        
        self.status_var.set(f"Done: {success} new, {skipped} skipped, {errors} errors")
        
        if errors == 0:
            self.messagebox.showinfo("Complete", 
                f"Processed: {success}\nSkipped (already done): {skipped}\n\n"
                f"Outputs saved next to each video.")
        else:
            self.messagebox.showwarning("Complete with errors",
                f"Processed: {success}\nSkipped: {skipped}\nErrors: {errors}\n\n"
                f"Check console for details.")
    
    def _view(self):
        """View results for a video."""
        videos = self._select_videos("Select Video to View")
        if not videos:
            return
        
        video = videos[0]  # View first selected
        
        self.status_var.set(f"Loading {video.name}...")
        self.root.update()
        
        if view_video(video):
            self.status_var.set("Viewer closed")
        else:
            self.status_var.set("Failed to launch viewer")
            self.messagebox.showerror("Error", 
                "Could not launch viewer.\n\n"
                "Make sure:\n"
                "1. Video has been processed\n"
                "2. DLC file exists in same folder\n"
                "3. napari is installed")
    
    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="ASPA2 - Automated Skilled Pellet Assessment"
    )
    parser.add_argument('--process', '-p', nargs='+', metavar='VIDEO',
                        help="Process video file(s)")
    parser.add_argument('--view', '-v', metavar='VIDEO',
                        help="View results for a video")
    parser.add_argument('--fps', type=float, default=60.0,
                        help="Video frame rate (default: 60)")
    parser.add_argument('--force', '-f', action='store_true',
                        help="Re-process even if outputs exist")
    
    args = parser.parse_args()
    
    # Process mode
    if args.process:
        for video in args.process:
            process_video(Path(video), fps=args.fps, force=args.force)
        return
    
    # View mode
    if args.view:
        view_video(Path(args.view))
        return
    
    # Default: GUI
    gui = ASPA2Gui()
    gui.run()


if __name__ == "__main__":
    main()
