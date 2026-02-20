"""
Machine diagnostics for the MouseReach watcher.

Provides informational tools for inspecting the current machine's drive
layout and verifying that configured paths are accessible. Used by the
mousereach-watch-info command.

All paths are configurable via mousereach-setup. This module does NOT
gate the watcher on specific drive letters — it only reports what it sees.
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DRIVE DETECTION (Windows only, informational)
# =============================================================================

def _drive_exists(letter: str) -> bool:
    """Check if a Windows drive letter is accessible."""
    if sys.platform != "win32":
        return False
    try:
        return Path(f"{letter}\\").exists()
    except OSError:
        return False


def _is_local_drive(letter: str) -> bool:
    """Check if a drive is local (not a network share).

    Uses Windows GetDriveType API:
      DRIVE_FIXED=3, DRIVE_REMOTE=4
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{letter}\\")
        return drive_type == 3  # DRIVE_FIXED
    except Exception:
        return False


def get_available_drives() -> dict:
    """Get all available drive letters and whether they're local or network.

    Returns:
        Dict mapping drive letter (e.g. "D:") to {"local": bool}
    """
    if sys.platform != "win32":
        return {}

    drives = {}
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        letter = f"{char}:"
        if _drive_exists(letter):
            drives[letter] = {"local": _is_local_drive(letter)}
    return drives


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configured_paths() -> list:
    """Check that the paths in config.json are accessible from this machine.

    Returns:
        List of (path_name, path_value, is_ok, message) tuples.
    """
    results = []
    try:
        from mousereach.config import _load_config, WatcherConfig
        config = _load_config()
    except Exception as e:
        return [("config.json", str(e), False, "Could not load config")]

    # NAS drive
    nas = config.get("nas_drive")
    if nas:
        p = Path(nas)
        ok = p.exists()
        results.append(("nas_drive", nas, ok,
                         "accessible" if ok else "NOT ACCESSIBLE from this machine"))
    else:
        results.append(("nas_drive", "(not set)", False, "not configured"))

    # Processing root
    proc = config.get("processing_root")
    if proc:
        p = Path(proc)
        ok = p.exists()
        results.append(("processing_root", proc, ok,
                         "accessible" if ok else "NOT ACCESSIBLE from this machine"))
    else:
        results.append(("processing_root", "(not set)", False, "REQUIRED - run mousereach-setup"))

    # Watcher-specific paths
    watcher = config.get("watcher", {})
    dlc = watcher.get("dlc_config_path")
    if dlc:
        p = Path(dlc)
        ok = p.exists()
        results.append(("watcher.dlc_config_path", dlc, ok,
                         "accessible" if ok else "NOT ACCESSIBLE from this machine"))

    return results


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_machine_info():
    """Print diagnostic info about drives, GPU, and configured path accessibility."""
    drives = get_available_drives()

    print("=" * 60)
    print("MouseReach Machine Diagnostics")
    print("=" * 60)
    print(f"\nPlatform: {sys.platform}")
    print()

    # Show drives
    print("Available Drives:")
    if not drives:
        print("  (none detected - not Windows or no drives accessible)")
    else:
        for letter, info in sorted(drives.items()):
            locality = "local" if info["local"] else "network"
            print(f"  {letter}  ({locality})")
    print()

    # Show GPU diagnostics (set up env first so TF/cuDNN paths are available)
    try:
        from mousereach.gpu import setup_gpu_env, print_gpu_info
        setup_gpu_env()
        print_gpu_info()
    except Exception as e:
        print(f"GPU diagnostics failed: {e}")
        print()

    # Show configured path validation
    print("Configured Paths:")
    results = validate_configured_paths()
    if not results:
        print("  (no configuration found - run mousereach-setup)")
    else:
        all_ok = True
        for name, value, ok, message in results:
            status = "OK" if ok else "!!"
            print(f"  [{status}] {name}: {value}")
            if not ok:
                print(f"       {message}")
                all_ok = False

        print()
        if all_ok:
            print("Watcher readiness: READY")
            print("  All configured paths are accessible from this machine.")
        else:
            print("Watcher readiness: NOT READY")
            print("  Some configured paths are not accessible.")
            print("  Run mousereach-setup to fix, or check network/drive mounts.")
