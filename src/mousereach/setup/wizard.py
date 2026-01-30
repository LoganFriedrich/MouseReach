"""Interactive configuration wizard for MouseReach.

This module provides a CLI-based setup wizard that guides users through
configuring MouseReach for their system. It saves configuration to a JSON file
in the user's home directory for reliable persistence.
"""

import os
import sys
import json
from pathlib import Path

# Config file location
CONFIG_DIR = Path.home() / ".mousereach"
CONFIG_FILE = CONFIG_DIR / "config.json"


def cli_main():
    """Main entry point for mousereach-setup command."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--show":
            show_config()
            return
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print_help()
            return
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use 'mousereach-setup --help' for options")
            return

    run_setup_wizard()


def print_help():
    """Print help message."""
    print(f"""
MouseReach Configuration Setup Wizard

Usage:
    mousereach-setup              Run interactive configuration wizard
    mousereach-setup --show       Show current configuration
    mousereach-setup --help       Show this help message

The wizard will prompt you for:
    - NAS/Archive Drive: Location of raw videos and final outputs
    - Pipeline Processing Root: Location of working pipeline folders

Configuration is saved to: {CONFIG_FILE}

Note: There are no default paths. MouseReach requires you to specify
      where YOUR pipeline folders are located on YOUR system.
""")


def run_setup_wizard():
    """Run the interactive setup wizard."""
    print("\n" + "=" * 60)
    print("MouseReach Configuration Setup Wizard")
    print("=" * 60)
    print("\nThis wizard will configure your MouseReach installation.")
    print("Press Enter to accept default values shown in [brackets].\n")

    # Load existing config for defaults
    existing = load_config()

    # Get NAS drive - use existing config or env var if available
    nas_default = existing.get("nas_drive") or os.getenv("MouseReach_NAS_DRIVE") or ""
    print("1. NAS/Archive Drive")
    print("   Location of raw videos and final outputs")
    print("   (optional - only needed for archiving)")
    nas_drive = prompt_path(
        "   Enter path",
        default=nas_default,
        must_exist=False,
        allow_empty=True
    )

    # Get processing root - use existing config or env var if available
    # NO hardcoded default - user must specify their own path
    proc_default = existing.get("processing_root") or os.getenv("MouseReach_PROCESSING_ROOT") or ""
    print("\n2. Pipeline Processing Root [REQUIRED]")
    print("   Location of working pipeline folders (DLC_Queue, Processing, etc.)")
    print("   Example: D:/MouseReach_Pipeline or /mnt/data/pipeline")
    proc_root = prompt_path(
        "   Enter path",
        default=proc_default,
        must_exist=False,
        allow_empty=False
    )

    # Save configuration
    print("\n" + "=" * 60)
    print("Saving configuration...")
    save_config(
        nas_drive=nas_drive,
        processing_root=proc_root
    )

    print("\n✓ Configuration saved!")
    print("\nYour settings:")
    print(f"  NAS Drive:         {nas_drive}")
    print(f"  Processing Root:   {proc_root}")

    if sys.platform != "win32":
        print("\nNote: On Linux/Mac, please run 'source ~/.bashrc' to apply changes")
        print("      or start a new terminal session.")

    print("\nNext steps:")
    print("  1. Verify: mousereach-setup --show")
    print("  2. Launch: mousereach")


def prompt_path(prompt_text: str, default: str, must_exist: bool = False, allow_empty: bool = False):
    """Prompt user for a path with validation.

    Args:
        prompt_text: Text to display before the input prompt
        default: Default value if user presses Enter
        must_exist: If True, validate that the path exists
        allow_empty: If True, allow empty/blank responses

    Returns:
        Path object for the configured directory, or None if allow_empty and empty
    """
    while True:
        if default:
            response = input(f"{prompt_text} [{default}]: ").strip()
        else:
            response = input(f"{prompt_text}: ").strip()

        path_str = response if response else default

        # Handle empty case
        if not path_str:
            if allow_empty:
                return None
            else:
                print("  ✗ This field is required. Please enter a path.")
                continue

        path = Path(path_str)

        if must_exist and not path.exists():
            print(f"  ⚠ Warning: {path} does not exist.")
            create = input("  Create it? [y/N]: ").lower().strip()
            if create == 'y':
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"  ✓ Created {path}")
                    return path
                except Exception as e:
                    print(f"  ✗ Error creating directory: {e}")
                    continue
            else:
                continue
        else:
            # Path doesn't need to exist (will be created later)
            return path


def save_config(nas_drive, processing_root: Path):
    """Save configuration to JSON file.

    Saves to ~/.mousereach/config.json for reliable cross-session persistence.

    Args:
        nas_drive: Path to NAS drive (can be None)
        processing_root: Path to pipeline processing root (required)
    """
    nas_str = str(nas_drive) if nas_drive else ""
    proc_str = str(processing_root)

    # Ensure config directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Save to JSON file (only include nas_drive if it's set)
    config = {
        "processing_root": proc_str
    }
    if nas_str:
        config["nas_drive"] = nas_str

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

        # Also set for current session
        if nas_str:
            os.environ["MouseReach_NAS_DRIVE"] = nas_str
        os.environ["MouseReach_PROCESSING_ROOT"] = proc_str
    except Exception as e:
        print(f"\n✗ Error saving configuration: {e}")


def load_config() -> dict:
    """Load configuration from JSON file.

    Returns:
        dict with 'nas_drive' and 'processing_root' keys, or empty dict if not found
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def show_config():
    """Display current MouseReach configuration."""
    # Load from config file first, then fall back to env vars
    config = load_config()

    nas_drive = config.get("nas_drive") or os.getenv("MouseReach_NAS_DRIVE")
    proc_root = config.get("processing_root") or os.getenv("MouseReach_PROCESSING_ROOT")

    print("\n" + "=" * 60)
    print("Current MouseReach Configuration")
    print("=" * 60)
    print(f"\nConfig file: {CONFIG_FILE}")
    print(f"  Exists: {CONFIG_FILE.exists()}")
    print(f"\nNAS Drive:         {nas_drive or '(not configured - optional)'}")
    print(f"Processing Root:   {proc_root or '(NOT CONFIGURED - REQUIRED)'}")

    if not proc_root:
        print("\n⚠️  PROCESSING_ROOT is not configured!")
        print("   Run 'mousereach-setup' to configure before using MouseReach.")
    elif not nas_drive:
        print("\nNote: NAS_DRIVE not configured (optional for basic pipeline)")
    print()


if __name__ == "__main__":
    cli_main()
