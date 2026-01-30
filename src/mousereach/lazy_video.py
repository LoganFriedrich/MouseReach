"""Lazy video loading for fast startup on network drives."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import threading


class LazyVideoArray:
    """Numpy-like array that loads video frames on demand for fast startup.

    Performance optimizations (v2.3.1):
    - Keeps VideoCapture open persistently (avoids open/close per frame)
    - Tracks position to skip seeks for sequential reads
    - Thread-safe with separate locks for cache and capture
    """

    def __init__(self, video_path: Path, cache_size: int = 100):
        import cv2
        self.video_path = video_path
        self.cache_size = cache_size
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []
        self._lock = threading.Lock()  # For cache access

        # Persistent VideoCapture for performance (avoid open/close per frame)
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()  # For capture access
        self._current_pos = 0  # Track position for sequential optimization

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_shape = frame_rgb.shape
            self._frame_dtype = frame_rgb.dtype
            self._cache[0] = frame_rgb
            self._cache_order.append(0)
        else:
            self._frame_shape = (self.height, self.width, 3)
            self._frame_dtype = np.uint8

        # Keep capture open for reuse instead of releasing
        self._cap = cap
        self._current_pos = 1  # We just read frame 0

        self.shape = (self.n_frames,) + self._frame_shape
        self.dtype = self._frame_dtype
        self.ndim = 4

    def _read_frame(self, idx: int) -> np.ndarray:
        """Read a single frame, using persistent capture for performance.

        Optimizations:
        - Check cache first (fast path)
        - Reuse persistent VideoCapture (avoid open/close overhead)
        - Skip seek if reading sequentially (most common during playback)
        - Reopen capture if connection lost (network drive timeout)
        """
        import cv2

        # Fast path: check cache first
        with self._lock:
            if idx in self._cache:
                self._cache_order.remove(idx)
                self._cache_order.append(idx)
                return self._cache[idx]

        # Read from video using persistent capture
        with self._cap_lock:
            # Reopen if needed (connection timeout on network drives)
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(str(self.video_path))
                self._current_pos = 0

            # Only seek if not sequential (saves ~50ms per frame on network)
            if idx != self._current_pos:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret, frame = self._cap.read()
            self._current_pos = idx + 1  # Track where we are now

        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.zeros(self._frame_shape, dtype=self._frame_dtype)

        # Update cache
        with self._lock:
            self._cache[idx] = frame_rgb
            self._cache_order.append(idx)
            while len(self._cache_order) > self.cache_size:
                oldest = self._cache_order.pop(0)
                if oldest in self._cache:
                    del self._cache[oldest]
        return frame_rgb

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx = self.n_frames + idx
            return self._read_frame(idx)
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.n_frames)
            frames = [self._read_frame(i) for i in range(start, stop, step)]
            return np.stack(frames) if frames else np.empty((0,) + self._frame_shape, dtype=self._frame_dtype)
        elif isinstance(key, tuple):
            frame_key = key[0]
            rest = key[1:] if len(key) > 1 else ()
            if isinstance(frame_key, (int, np.integer)):
                frame = self._read_frame(int(frame_key))
                return frame[rest] if rest else frame
            elif isinstance(frame_key, slice):
                start, stop, step = frame_key.indices(self.n_frames)
                frames = np.stack([self._read_frame(i) for i in range(start, stop, step)])
                return frames[(slice(None),) + rest] if rest else frames
        raise IndexError(f"Unsupported index type: {type(key)}")

    def __len__(self):
        return self.n_frames

    def __array__(self, dtype=None):
        print(f"[MouseReach] Warning: Converting lazy video to full array ({self.n_frames} frames)")
        frames = [self._read_frame(i) for i in range(self.n_frames)]
        arr = np.stack(frames)
        return arr.astype(dtype) if dtype else arr

    def preload_range(self, start: int, end: int):
        for i in range(max(0, start), min(end, self.n_frames)):
            if i not in self._cache:
                self._read_frame(i)

    def close(self):
        """Release video capture resources.

        Call this when done with the video to free file handles.
        Safe to call multiple times.
        """
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


# =============================================================================
# Smart Loading Strategy
# =============================================================================

def get_available_ram_mb() -> float:
    """Get available RAM in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        # Fallback: assume 4GB available (conservative)
        return 4096


def estimate_video_memory_mb(video_path: Path) -> Tuple[float, dict]:
    """
    Estimate memory needed to load a video fully into RAM.

    Returns:
        (estimated_mb, video_info)
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0, {}

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()

    # RGB: 3 bytes per pixel
    bytes_per_frame = width * height * 3
    total_bytes = n_frames * bytes_per_frame
    estimated_mb = total_bytes / (1024 * 1024)

    video_info = {
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "bytes_per_frame": bytes_per_frame,
    }

    return estimated_mb, video_info


def get_loading_strategy(video_path: Path) -> dict:
    """
    Determine the best loading strategy based on available RAM.

    Returns a dict with:
        strategy: "lazy" | "lazy_large_cache" | "compressed"
        cache_size: Number of frames to cache (for lazy strategies)
        scale: Resolution scale factor (for compressed strategy)
        reason: Human-readable explanation

    The strategy prioritizes:
    1. Lazy loading with appropriate cache size (fastest startup)
    2. Only use compression if truly necessary (very large videos)
    """
    available_ram = get_available_ram_mb()
    estimated_mb, video_info = estimate_video_memory_mb(video_path)

    if estimated_mb == 0:
        return {
            "strategy": "lazy",
            "cache_size": 100,
            "scale": 1.0,
            "reason": "Could not read video info, using default lazy loading",
        }

    # Calculate ratio of available RAM to video size
    ratio = available_ram / estimated_mb if estimated_mb > 0 else float('inf')

    # Calculate optimal cache size based on available RAM
    # Reserve 60% of available RAM for other uses, use 40% for cache
    usable_ram_mb = available_ram * 0.4
    bytes_per_frame = video_info.get("bytes_per_frame", 1)
    max_cache_frames = int((usable_ram_mb * 1024 * 1024) / bytes_per_frame) if bytes_per_frame > 0 else 500

    # Cap cache size at reasonable values
    max_cache_frames = max(50, min(max_cache_frames, 2000))

    if ratio > 3.0:
        # Plenty of RAM - large cache for smooth scrubbing
        return {
            "strategy": "lazy_large_cache",
            "cache_size": min(max_cache_frames, 1000),
            "scale": 1.0,
            "reason": f"Ample RAM ({available_ram:.0f}MB available, video needs {estimated_mb:.0f}MB)",
        }
    elif ratio > 1.0:
        # Moderate RAM - medium cache
        return {
            "strategy": "lazy",
            "cache_size": min(max_cache_frames, 300),
            "scale": 1.0,
            "reason": f"Moderate RAM ({available_ram:.0f}MB available, video needs {estimated_mb:.0f}MB)",
        }
    elif ratio > 0.5:
        # Limited RAM - small cache
        return {
            "strategy": "lazy",
            "cache_size": min(max_cache_frames, 100),
            "scale": 1.0,
            "reason": f"Limited RAM ({available_ram:.0f}MB available, video needs {estimated_mb:.0f}MB)",
        }
    else:
        # Very limited RAM - need compressed preview
        # Calculate scale to fit in available RAM
        target_mb = available_ram * 0.3  # Target 30% of available RAM
        scale = min(0.75, np.sqrt(target_mb / estimated_mb)) if estimated_mb > 0 else 0.5
        scale = max(0.25, scale)  # Don't go below 25%

        return {
            "strategy": "compressed",
            "cache_size": 50,
            "scale": round(scale, 2),
            "reason": f"Low RAM ({available_ram:.0f}MB available, video needs {estimated_mb:.0f}MB) - using compressed preview",
        }


def smart_load_video(
    video_path: Path,
    force_strategy: Optional[str] = None,
    progress_callback=None
) -> Tuple["LazyVideoArray", dict]:
    """
    Smart video loader that picks the best strategy based on available RAM.

    Args:
        video_path: Path to video file
        force_strategy: Override auto-detection ("lazy", "compressed")
        progress_callback: Optional callback(percent, message)

    Returns:
        (video_array, strategy_info)
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get optimal strategy
    if force_strategy:
        strategy = {"strategy": force_strategy, "cache_size": 100, "scale": 0.75, "reason": "Forced"}
    else:
        strategy = get_loading_strategy(video_path)

    if progress_callback:
        progress_callback(10, f"Strategy: {strategy['reason']}")

    print(f"[MouseReach] Loading {video_path.name}")
    print(f"[MouseReach]   Strategy: {strategy['strategy']} (cache={strategy['cache_size']})")
    print(f"[MouseReach]   Reason: {strategy['reason']}")

    if strategy["strategy"] == "compressed":
        # Need to create/use compressed preview
        from mousereach.video_prep.compress import get_preview_path, create_preview

        preview_path = get_preview_path(video_path)

        if not preview_path.exists():
            if progress_callback:
                progress_callback(20, "Creating compressed preview...")
            print(f"[MouseReach]   Creating preview at {strategy['scale']*100:.0f}% scale...")
            create_preview(video_path, scale=strategy["scale"], overwrite=False)

        if preview_path.exists():
            if progress_callback:
                progress_callback(50, "Loading preview...")
            video_array = LazyVideoArray(preview_path, cache_size=strategy["cache_size"])
            strategy["used_preview"] = True
            strategy["preview_path"] = str(preview_path)
        else:
            # Fallback to original with small cache
            print(f"[MouseReach]   Preview creation failed, using original with minimal cache")
            video_array = LazyVideoArray(video_path, cache_size=50)
            strategy["used_preview"] = False
    else:
        # Use lazy loading with specified cache size
        if progress_callback:
            progress_callback(50, "Opening video...")
        video_array = LazyVideoArray(video_path, cache_size=strategy["cache_size"])
        strategy["used_preview"] = False

    if progress_callback:
        progress_callback(100, "Ready")

    return video_array, strategy
