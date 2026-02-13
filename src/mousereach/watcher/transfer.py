"""
File transfer utilities for MouseReach watcher.

Provides safe file copying with verification and stability detection
to ensure files are fully written before processing.
"""

import os
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def wait_for_stable(path: Path, check_interval: float = 5.0,
                    stable_duration: float = 10.0,
                    timeout: float = 300.0) -> bool:
    """
    Wait until a file's size stops changing.

    Polls file size every `check_interval` seconds. Returns True when
    size is unchanged for `stable_duration` seconds. Returns False if
    the file disappears or timeout is exceeded.

    Args:
        path: Path to the file to monitor
        check_interval: Seconds between size checks
        stable_duration: How long size must be unchanged to consider stable
        timeout: Maximum total wait time in seconds

    Returns:
        True if file is stable, False if timeout or file disappeared
    """
    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        return False

    start_time = time.time()
    last_size = path.stat().st_size
    stable_since = time.time()

    while (time.time() - start_time) < timeout:
        time.sleep(check_interval)

        if not path.exists():
            logger.warning(f"File disappeared during stability check: {path}")
            return False

        current_size = path.stat().st_size

        if current_size != last_size:
            last_size = current_size
            stable_since = time.time()
            logger.debug(f"File size changed to {current_size} bytes: {path.name}")
        elif (time.time() - stable_since) >= stable_duration:
            logger.debug(f"File stable at {current_size} bytes for {stable_duration}s: {path.name}")
            return True

    logger.warning(f"Stability check timed out after {timeout}s: {path}")
    return False


def safe_copy(src: Path, dst: Path, verify: bool = True) -> bool:
    """
    Copy a file with size verification.

    Args:
        src: Source file path
        dst: Destination file path
        verify: If True, verify file sizes match after copy

    Returns:
        True if copy succeeded (and verified if requested)
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Copying {src.name} -> {dst.parent}")
        shutil.copy2(str(src), str(dst))

        if verify:
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size
            if src_size != dst_size:
                logger.error(
                    f"Size mismatch after copy: src={src_size}, dst={dst_size} "
                    f"for {src.name}"
                )
                # Clean up bad copy
                dst.unlink(missing_ok=True)
                return False
            logger.debug(f"Copy verified: {src_size} bytes for {src.name}")

        return True

    except OSError as e:
        logger.error(f"Copy failed {src} -> {dst}: {e}")
        # Clean up partial copy
        if dst.exists():
            dst.unlink(missing_ok=True)
        return False


def safe_move(src: Path, dst: Path) -> bool:
    """
    Move a file safely using copy-then-delete.

    Uses copy+verify+delete instead of rename, since rename is not
    atomic across different drives/mounts.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        True if move succeeded
    """
    if not safe_copy(src, dst, verify=True):
        return False

    try:
        src.unlink()
        logger.debug(f"Moved {src.name} -> {dst.parent}")
        return True
    except OSError as e:
        logger.error(f"Failed to delete source after copy: {src}: {e}")
        # Copy succeeded, source delete failed — file exists in both places
        # This is not ideal but not data-losing
        return True  # The file IS at the destination


def check_file_stable_quick(path: Path, recorded_size: Optional[int],
                             min_stable_seconds: float = 60.0,
                             last_change_time: Optional[float] = None) -> tuple:
    """
    Quick (non-blocking) stability check for use in polling loops.

    Instead of waiting, this compares the current size against a previously
    recorded size and timestamps. Designed to be called once per scan cycle.

    Args:
        path: File to check
        recorded_size: Previously recorded size (None if first check)
        min_stable_seconds: Required stable duration
        last_change_time: Timestamp of last size change

    Returns:
        Tuple of (is_stable: bool, current_size: int, change_time: float)
        - is_stable: True if size unchanged for min_stable_seconds
        - current_size: Current file size
        - change_time: When size last changed (now if changed, previous if stable)
    """
    if not path.exists():
        return False, 0, time.time()

    current_size = path.stat().st_size
    now = time.time()

    if recorded_size is None or current_size != recorded_size:
        # First check or size changed
        return False, current_size, now

    # Size unchanged — check duration
    if last_change_time is None:
        return False, current_size, now

    elapsed = now - last_change_time
    is_stable = elapsed >= min_stable_seconds

    return is_stable, current_size, last_change_time


def verify_file_hash(path1: Path, path2: Path, algorithm: str = 'md5') -> bool:
    """
    Verify two files have identical content via hash comparison.

    Only use for critical files — this reads the entire file.

    Args:
        path1: First file
        path2: Second file
        algorithm: Hash algorithm ('md5', 'sha256')

    Returns:
        True if hashes match
    """
    def _hash_file(path: Path) -> str:
        h = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    try:
        hash1 = _hash_file(path1)
        hash2 = _hash_file(path2)
        match = hash1 == hash2
        if not match:
            logger.error(f"Hash mismatch: {path1.name} ({hash1[:8]}...) vs {path2.name} ({hash2[:8]}...)")
        return match
    except OSError as e:
        logger.error(f"Hash verification failed: {e}")
        return False
