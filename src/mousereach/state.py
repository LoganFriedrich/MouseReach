"""
MouseReach State Manager
===================

Shared state manager for coordinating between MouseReach widgets.

Features:
- Detects video layers added to napari (drag-drop, File menu, etc.)
- Tracks the "active" video path
- SHARES a single video layer across all widgets (load once, not 3x!)
- Notifies widgets when video changes
- Notifies downstream widgets when upstream data is saved

Usage in launch_all.py:
    state = MouseReachStateManager(viewer)
    state.register_widget("2b", widget_2b)
    state.register_widget("3b", widget_3b)
    # etc.

    # Load video once, all widgets use it:
    state.load_video("path/to/video.mp4")
"""

from pathlib import Path
from typing import Dict, Callable, Optional, List, Any
import numpy as np
from qtpy.QtCore import QObject, Signal
from mousereach.lazy_video import LazyVideoArray, smart_load_video


class MouseReachStateManager(QObject):
    """
    Manages shared state between MouseReach widgets.

    Responsibilities:
    - Watch for video layers in napari viewer
    - Track which video is "active"
    - Notify widgets when they should load/refresh data
    - Handle save events and propagate to downstream widgets
    """

    # Signals for widget communication
    video_changed = Signal(Path)  # Emitted when active video changes
    data_saved = Signal(str, Path)  # Emitted when a step saves (step_id, video_path)

    # Define the pipeline order and dependencies
    PIPELINE_ORDER = ["2b", "3b", "4b", "5"]
    DEPENDENCIES = {
        "3b": ["2b"],  # 3b depends on 2b output
        "4b": ["3b"],  # 4b depends on 3b output
        "5": ["2b", "3b", "4b"],  # 5 depends on all review steps
    }

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.widgets: Dict[str, Any] = {}  # step_id -> widget
        self.active_video_path: Optional[Path] = None
        self._video_layers: Dict[str, Path] = {}  # layer_name -> video_path

        # Shared video data (loaded once, used by all widgets)
        self._shared_video_layer = None  # The napari Image layer
        self._shared_video_frames = None  # numpy array of frames
        self._shared_video_fps = 60.0
        self._shared_n_frames = 0

        # Connect to napari layer events
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        # Check existing layers
        self._scan_existing_layers()

    def register_widget(self, step_id: str, widget):
        """Register a widget for state management."""
        self.widgets[step_id] = widget

        # Connect widget's save signal if it has one
        if hasattr(widget, 'data_saved'):
            widget.data_saved.connect(lambda p, s=step_id: self._on_widget_saved(s, p))

        # Connect widget to video_changed signal so it auto-loads when video changes
        # This enables cross-widget video sharing
        def on_video_changed(path, w=widget, sid=step_id):
            self._load_data_for_widget(sid, w)
        self.video_changed.connect(on_video_changed)

    def _scan_existing_layers(self):
        """Check for any video layers that already exist."""
        for layer in self.viewer.layers:
            self._check_layer_for_video(layer)

    def _on_layer_inserted(self, event):
        """Handle new layer being added to viewer."""
        layer = event.value
        self._check_layer_for_video(layer)

    def _on_layer_removed(self, event):
        """Handle layer being removed from viewer."""
        layer = event.value
        if layer.name in self._video_layers:
            removed_path = self._video_layers[layer.name]  # Save before deleting
            del self._video_layers[layer.name]
            # If this was the active video, clear it
            if self.active_video_path == removed_path:
                self.active_video_path = None

    def _check_layer_for_video(self, layer):
        """Check if a layer is a video and extract its path."""
        # napari Image layers from video files have source info
        if hasattr(layer, 'source') and layer.source:
            source = layer.source
            if hasattr(source, 'path') and source.path:
                path = Path(source.path)
                if path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                    self._video_layers[layer.name] = path
                    self._maybe_set_active_video(path)

        # Also check layer.metadata for path info (some plugins store it there)
        if hasattr(layer, 'metadata') and isinstance(layer.metadata, dict):
            if 'path' in layer.metadata:
                path = Path(layer.metadata['path'])
                if path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                    self._video_layers[layer.name] = path
                    self._maybe_set_active_video(path)

    def _maybe_set_active_video(self, path: Path):
        """Set active video if we don't have one, or if this is newer."""
        if self.active_video_path is None:
            self.set_active_video(path)
        # If we already have a video, don't auto-switch
        # User can manually select which video to use

    def set_active_video(self, path: Path):
        """Set the active video and notify widgets."""
        if path == self.active_video_path:
            return

        self.active_video_path = path
        print(f"[MouseReach] Active video: {path.name}")
        self.video_changed.emit(path)

    def load_video(self, path: Path, progress_callback=None):
        """
        Load a video ONCE and make it available to all widgets.

        Uses smart loading that checks available RAM and picks the best strategy:
        - Plenty of RAM: Large cache for smooth scrubbing
        - Moderate RAM: Standard lazy loading
        - Low RAM: Creates compressed preview automatically
        """
        path = Path(path)
        if not path.exists():
            print(f"[MouseReach] Video not found: {path}")
            return False

        try:
            # Use smart loader that picks optimal strategy based on RAM
            lazy_video, strategy = smart_load_video(path, progress_callback=progress_callback)
            self._shared_n_frames = lazy_video.n_frames
            self._shared_video_fps = lazy_video.fps
            self._shared_video_frames = lazy_video
            self._loading_strategy = strategy  # Store for debugging/display

            print(f"[MouseReach] Video ready: {lazy_video.n_frames} frames at {lazy_video.fps:.1f} fps")
            if strategy.get("used_preview"):
                print(f"[MouseReach] Using compressed preview (RAM-constrained)")
        except Exception as e:
            print(f"[MouseReach] Could not open video: {e}")
            return False
        if self._shared_video_layer is not None:
            try:
                self.viewer.layers.remove(self._shared_video_layer)
            except ValueError:
                pass
        self._shared_video_layer = self.viewer.add_image(
            self._shared_video_frames,
            name=path.stem,
            metadata={'path': str(path), 'mousereach_shared': True, 'lazy': True}
        )
        self.active_video_path = path
        self._video_layers[path.stem] = path
        self.video_changed.emit(path)
        self._load_data_into_widgets()
        if progress_callback:
            progress_callback(100)
        return True

    def _load_data_into_widgets(self):
        """Load associated data files into all widgets (but NOT the video - they share that)."""
        if not self.active_video_path:
            return

        for step_id, widget in self.widgets.items():
            if step_id in ["2b", "3b", "4b", "5"]:
                self._load_data_for_widget(step_id, widget)

    def _load_data_for_widget(self, step_id: str, widget):
        """Load just the data files for a widget (it uses the shared video layer)."""
        if not self.active_video_path:
            return

        # Give widget access to shared video info
        widget._shared_video_layer = self._shared_video_layer
        widget._shared_video_frames = self._shared_video_frames
        widget._shared_n_frames = self._shared_n_frames
        widget._shared_video_fps = self._shared_video_fps
        widget.video_path = self.active_video_path

        # If widget has a method to load just the data (not video), use it
        if hasattr(widget, '_load_data_only'):
            print(f"[MouseReach] Loading data for {step_id}: {self.active_video_path.name}")
            widget._load_data_only(self.active_video_path)
        elif hasattr(widget, '_load_associated_data'):
            print(f"[MouseReach] Loading data for {step_id}: {self.active_video_path.name}")
            widget._load_associated_data(self.active_video_path)

    def get_shared_video(self):
        """Get the shared video data."""
        return {
            'layer': self._shared_video_layer,
            'frames': self._shared_video_frames,
            'n_frames': self._shared_n_frames,
            'fps': self._shared_video_fps,
            'path': self.active_video_path,
        }

    def get_available_videos(self) -> List[Path]:
        """Get list of all video paths detected in viewer."""
        return list(set(self._video_layers.values()))

    def _detect_video_from_layers(self) -> Optional[Path]:
        """Scan napari layers for video and return path if found."""
        for layer in self.viewer.layers:
            # Check metadata first (our preferred storage)
            if hasattr(layer, 'metadata') and isinstance(layer.metadata, dict):
                path_str = layer.metadata.get('path')
                if path_str:
                    path = Path(path_str)
                    if path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                        return path

            # Check source.path (napari's native storage)
            if hasattr(layer, 'source') and layer.source:
                if hasattr(layer.source, 'path') and layer.source.path:
                    path = Path(layer.source.path)
                    if path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                        return path

            # Check layer name against common video patterns
            if hasattr(layer, 'name'):
                # Try to find a video file that matches the layer name
                for suffix in ['.mp4', '.avi', '.mkv', '.mov']:
                    # Check various locations
                    candidates = []
                    if hasattr(layer, 'data') and hasattr(layer.data, 'path'):
                        candidates.append(Path(layer.data.path))

        return None

    def load_active_video_into_widget(self, step_id: str):
        """Load the active video into a specific widget.

        If video is already loaded in shared state, uses _load_data_only to avoid
        reloading the video. Otherwise falls back to full _load_video_from_path.
        """
        # If no active video tracked, try to detect from napari layers
        if not self.active_video_path:
            detected = self._detect_video_from_layers()
            if detected:
                print(f"[MouseReach] Detected video from layers: {detected.name}")
                self.active_video_path = detected
                self._video_layers[detected.stem] = detected
            else:
                print(f"[MouseReach] No active video to load into {step_id}")
                return False

        widget = self.widgets.get(step_id)
        if not widget:
            print(f"[MouseReach] Widget {step_id} not registered")
            return False

        # Check if widget already has this video loaded
        if hasattr(widget, 'video_path') and widget.video_path == self.active_video_path:
            print(f"[MouseReach] Widget {step_id} already has this video loaded")
            return True

        # If shared video is loaded, use _load_data_only (much faster, shares video layer)
        if self._shared_video_layer is not None and hasattr(widget, '_load_data_only'):
            self._load_data_for_widget(step_id, widget)
            return True
        elif hasattr(widget, '_load_video_from_path'):
            print(f"[MouseReach] Loading video into {step_id} via _load_video_from_path")
            widget._load_video_from_path(self.active_video_path)
            return True
        elif hasattr(widget, '_load_data_for_video'):
            print(f"[MouseReach] Loading video into {step_id} via _load_data_for_video")
            widget._load_data_for_video(self.active_video_path)
            return True
        else:
            print(f"[MouseReach] Widget {step_id} doesn't support video loading")
            return False

    def _on_widget_saved(self, step_id: str, video_path: Path):
        """Handle a widget saving its data."""
        print(f"[MouseReach] Step {step_id} saved data for {video_path.name}")
        self.data_saved.emit(step_id, video_path)

        # Find downstream widgets that depend on this step
        for downstream_id, deps in self.DEPENDENCIES.items():
            if step_id in deps:
                downstream_widget = self.widgets.get(downstream_id)
                if downstream_widget:
                    # Check if downstream widget has the same video loaded
                    if hasattr(downstream_widget, 'video_path'):
                        if downstream_widget.video_path == video_path:
                            self._refresh_widget(downstream_id)

    def _refresh_widget(self, step_id: str):
        """Refresh a widget's data (reload from disk)."""
        widget = self.widgets.get(step_id)
        if not widget:
            return

        print(f"[MouseReach] Refreshing {step_id} with updated upstream data")

        # Try different refresh methods
        if hasattr(widget, '_refresh_data'):
            widget._refresh_data()
        elif hasattr(widget, '_reload_data'):
            widget._reload_data()
        elif hasattr(widget, 'video_path') and hasattr(widget, '_load_video_from_path'):
            # Reload the whole thing
            widget._load_video_from_path(widget.video_path)

    def notify_save(self, step_id: str, video_path: Path):
        """
        Call this from widgets when they save.

        Usage in widget save method:
            if hasattr(self, '_state_manager'):
                self._state_manager.notify_save("3b", self.video_path)
        """
        self._on_widget_saved(step_id, video_path)


def connect_widget_to_state(widget, state_manager: MouseReachStateManager, step_id: str):
    """
    Helper to connect a widget to the state manager.

    This patches the widget to notify state manager on save, and
    stores a reference so the widget can access shared state.
    """
    # Store reference to state manager
    widget._state_manager = state_manager
    widget._step_id = step_id

    # Register with state manager
    state_manager.register_widget(step_id, widget)

    # Patch save methods to notify state manager
    # Look for common save method names
    for save_method_name in ['_save_progress', '_save_validated', '_save_ground_truth', '_save']:
        if hasattr(widget, save_method_name):
            original_method = getattr(widget, save_method_name)

            def make_wrapped_save(orig, sid, sm):
                def wrapped_save(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    # Notify state manager after successful save
                    if hasattr(widget, 'video_path') and widget.video_path:
                        sm.notify_save(sid, widget.video_path)
                    return result
                return wrapped_save

            setattr(widget, save_method_name, make_wrapped_save(original_method, step_id, state_manager))
