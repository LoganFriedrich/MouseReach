"""Interactive configuration wizard for MouseReach.

This module provides a CLI-based setup wizard that guides users through
configuring MouseReach for their system. It saves configuration to a JSON file
in the user's home directory for reliable persistence.

Machine identification priority:
  1. Local identity file (~/.mousereach/machine_role.json) - explicit declaration
  2. lab_profiles.json drive-pattern matching - auto-detection fallback
  3. Fully manual - user enters paths interactively

The local identity file is the simplest approach: drop a small JSON file on
each PC that says what role it plays. The wizard reads it and pre-fills
defaults from the matching lab profile.

To set up a machine's identity:
    mousereach-setup --set-role "NAS / DLC PC"

Or create ~/.mousereach/machine_role.json manually:
    {"role": "NAS / DLC PC"}

Edit lab_profiles.json to customize profiles for your lab, or delete it
to use the wizard with no pre-filled defaults.
"""

import os
import sys
import json
from pathlib import Path

# Config file location
CONFIG_DIR = Path.home() / ".mousereach"
CONFIG_FILE = CONFIG_DIR / "config.json"
MACHINE_ROLE_FILE = CONFIG_DIR / "machine_role.json"

# Lab profiles file (lab-specific, gitignored) with shipped template fallback
LAB_PROFILES_FILE = Path(__file__).parent / "lab_profiles.json"
LAB_PROFILES_EXAMPLE = Path(__file__).parent / "lab_profiles.json.example"


# =============================================================================
# LOCAL MACHINE IDENTITY
# =============================================================================

def _load_machine_role() -> str | None:
    """Load the local machine role from ~/.mousereach/machine_role.json.

    Returns:
        Role name string, or None if no identity file exists.
    """
    if not MACHINE_ROLE_FILE.exists():
        return None
    try:
        with open(MACHINE_ROLE_FILE, "r") as f:
            data = json.load(f)
        return data.get("role")
    except Exception:
        return None


def set_machine_role(role_name: str):
    """Write a machine identity file declaring this PC's role.

    Args:
        role_name: Must match a profile name in lab_profiles.json.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MACHINE_ROLE_FILE, "w") as f:
        json.dump({"role": role_name}, f, indent=2)


def list_available_roles() -> list:
    """Return list of profile names from lab_profiles.json."""
    profiles = _load_lab_profiles()
    return [p.get("name", "Unknown") for p in profiles]


# =============================================================================
# LAB PROFILE MATCHING
# =============================================================================

def _load_lab_profiles() -> list:
    """Load lab profiles from lab_profiles.json, falling back to .example template."""
    profiles_path = LAB_PROFILES_FILE
    if not profiles_path.exists():
        profiles_path = LAB_PROFILES_EXAMPLE
    if not profiles_path.exists():
        return []
    try:
        with open(profiles_path, "r") as f:
            data = json.load(f)
        return data.get("profiles", [])
    except Exception:
        return []


def _get_current_drives() -> dict:
    """Get available drives on this machine. Returns {letter: is_local}."""
    if sys.platform != "win32":
        return {}
    drives = {}
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive_path = Path(f"{char}:\\")
        try:
            if drive_path.exists():
                # Check if local via GetDriveType
                try:
                    import ctypes
                    drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{char}:\\")
                    drives[char] = (drive_type == 3)  # DRIVE_FIXED = local
                except Exception:
                    drives[char] = False
        except OSError:
            pass
    return drives


def _match_profile(profile: dict, drives: dict) -> bool:
    """Check if a profile's match rules fit the current machine's drives.

    Args:
        profile: A profile dict from lab_profiles.json
        drives: Dict of {letter: is_local} from _get_current_drives()
    """
    match = profile.get("match", {})
    available = set(drives.keys())

    # drives_present: all must exist
    for letter in match.get("drives_present", []):
        if letter.upper().rstrip(":") not in available:
            return False

    # drives_local: all must exist AND be local
    for letter in match.get("drives_local", []):
        letter = letter.upper().rstrip(":")
        if letter not in available or not drives.get(letter, False):
            return False

    # drives_absent: none of these should exist
    for letter in match.get("drives_absent", []):
        if letter.upper().rstrip(":") in available:
            return False

    return True


def _find_profile_by_name(name: str) -> dict | None:
    """Find a lab profile by its name (case-insensitive)."""
    profiles = _load_lab_profiles()
    for profile in profiles:
        if profile.get("name", "").lower() == name.lower():
            return profile
    return None


def detect_lab_profile() -> tuple:
    """Identify this machine's lab profile.

    Priority:
      1. Local identity file (~/.mousereach/machine_role.json)
      2. Drive-pattern matching against lab_profiles.json
      3. None (fully manual)

    Returns:
        (profile_dict, drives_dict, detection_method) where detection_method
        is "identity_file", "drive_match", or None.
    """
    drives = _get_current_drives()

    # Priority 1: Local identity file
    role = _load_machine_role()
    if role:
        profile = _find_profile_by_name(role)
        if profile:
            return profile, drives, "identity_file"

    # Priority 2: Drive-pattern matching
    profiles = _load_lab_profiles()
    for profile in profiles:
        if _match_profile(profile, drives):
            return profile, drives, "drive_match"

    return None, drives, None


# =============================================================================
# CLI ENTRY POINTS
# =============================================================================

def cli_main():
    """Main entry point for mousereach-setup command."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--show":
            show_config()
            return
        elif sys.argv[1] == "--set-role":
            if len(sys.argv) < 3:
                print("Usage: mousereach-setup --set-role \"Role Name\"")
                print("\nAvailable roles:")
                for name in list_available_roles():
                    print(f"  - {name}")
                return
            role_name = sys.argv[2]
            profile = _find_profile_by_name(role_name)
            if not profile:
                print(f"Unknown role: {role_name}")
                print("\nAvailable roles:")
                for name in list_available_roles():
                    print(f"  - {name}")
                return
            set_machine_role(role_name)
            print(f"Machine role set to: {role_name}")
            print(f"  Saved to: {MACHINE_ROLE_FILE}")
            print(f"  Description: {profile.get('description', '')}")
            print(f"\nRun 'mousereach-setup' to configure paths for this role.")
            return
        elif sys.argv[1] == "--list-roles":
            profiles = _load_lab_profiles()
            if not profiles:
                print("No lab profiles found.")
                print(f"  Expected file: {LAB_PROFILES_FILE}")
                return
            print("\nAvailable machine roles:")
            current_role = _load_machine_role()
            for p in profiles:
                name = p.get("name", "Unknown")
                desc = p.get("description", "")
                marker = " (current)" if current_role and current_role.lower() == name.lower() else ""
                print(f"  {name}{marker}")
                if desc:
                    print(f"    {desc}")
            print(f"\nSet role: mousereach-setup --set-role \"Role Name\"")
            print(f"Identity file: {MACHINE_ROLE_FILE}")
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
    mousereach-setup                   Run interactive configuration wizard
    mousereach-setup --show            Show current configuration
    mousereach-setup --set-role NAME   Declare this PC's role (saves locally)
    mousereach-setup --list-roles      Show available machine roles
    mousereach-setup --help            Show this help message

Machine identification priority:
  1. Local identity file (~/.mousereach/machine_role.json)
     Set with: mousereach-setup --set-role "NAS / DLC PC"
  2. Drive-pattern auto-detection (lab_profiles.json)
  3. Fully manual (user enters all paths)

When a role is identified, the wizard pre-fills defaults from the
matching lab profile. You can override any default during setup.

Configuration is saved to: {CONFIG_FILE}
Identity file: {MACHINE_ROLE_FILE}
Lab profiles file: {LAB_PROFILES_FILE}
""")


# =============================================================================
# SETUP WIZARD
# =============================================================================

def run_setup_wizard():
    """Run the interactive setup wizard."""
    print("\n" + "=" * 60)
    print("MouseReach Configuration Setup Wizard")
    print("=" * 60)

    # --- Lab profile detection ---
    profile, drives, method = detect_lab_profile()

    if drives:
        drive_strs = []
        for letter in sorted(drives.keys()):
            locality = "local" if drives[letter] else "network"
            drive_strs.append(f"{letter}: ({locality})")
        print(f"\nDrives: {', '.join(drive_strs)}")

    if profile:
        name = profile.get("name", "Unknown")
        desc = profile.get("description", "")
        if method == "identity_file":
            print(f"\nMachine role: {name}  (from {MACHINE_ROLE_FILE})")
        else:
            print(f"\nAuto-detected profile: {name}  (matched drive pattern)")
        if desc:
            print(f"  {desc}")
        print("  Defaults are pre-filled below. Press Enter to accept,")
        print("  or type a new value to override.")
    else:
        print("\n  No machine role set and no drive pattern matched.")
        print("  Tip: Run 'mousereach-setup --set-role \"Role Name\"' to declare")
        print("  this PC's role, or enter paths manually below.")

    print()

    # Merge defaults: existing config > profile defaults > nothing
    existing = load_config()
    profile_defaults = profile.get("defaults", {}) if profile else {}
    profile_watcher = profile_defaults.get("watcher", {})

    # --- Step 1: Data repository ---
    nas_default = (
        existing.get("nas_drive")
        or profile_defaults.get("nas_drive")
        or os.getenv("MouseReach_NAS_DRIVE")
        or ""
    )
    print("1. Data Repository")
    print("   Where is your long-term data storage? This is the root of the")
    print("   drive or network share where raw collage videos arrive and")
    print("   where final analyzed results are archived.")
    print()
    print("   If this PC has a direct/local connection to the storage,")
    print("   use the local path (faster than a mapped network drive).")
    nas_drive = prompt_path(
        "   Data repository path",
        default=nas_default,
        must_exist=False,
        allow_empty=True
    )

    # --- Step 2: Processing root ---
    proc_default = (
        existing.get("processing_root")
        or profile_defaults.get("processing_root")
        or os.getenv("MouseReach_PROCESSING_ROOT")
        or ""
    )
    print("\n2. Pipeline Processing Root [REQUIRED]")
    print("   Parent folder for all working pipeline directories.")
    print("   The watcher creates these subfolders automatically:")
    print("     <root>/DLC_Queue/     - videos waiting for DLC")
    print("     <root>/Processing/    - post-DLC analysis")
    print("     <root>/Failed/        - errors requiring investigation")
    if nas_drive and not proc_default:
        suggested = Path(nas_drive) / "! DLC Output"
        print(f"\n   Based on your data repository, this is typically: {suggested}")
        proc_default = str(suggested)
    proc_root = prompt_path(
        "   Enter path",
        default=proc_default,
        must_exist=False,
        allow_empty=False
    )

    # --- Step 3: Watcher configuration ---
    watcher_config = None
    print("\n3. Automated Watcher")
    print("   The watcher monitors your data repository for new collage")
    print("   videos and automatically runs the full pipeline: crop, DLC,")
    print("   segment, detect reaches, detect outcomes, archive.")
    print()
    print("   Requirements for the machine running the watcher:")
    print("     - Fast/local access to the data repository")
    print("     - A GPU (for DLC inference)")
    print("     - Access to the DLC model config.yaml")
    print()

    # Default to yes if the profile includes watcher config
    if profile_watcher.get("enabled"):
        setup_watcher = input("   Set up the watcher on this machine? [Y/n]: ").strip().lower()
        setup_watcher = setup_watcher != 'n'
    else:
        setup_watcher = input("   Set up the watcher on this machine? [y/N]: ").strip().lower()
        setup_watcher = setup_watcher == 'y'

    if setup_watcher:
        watcher_config = {}
        existing_watcher = existing.get("watcher", {})

        # DLC config path — priority: existing > profile > empty
        dlc_default = (
            existing_watcher.get("dlc_config_path")
            or profile_watcher.get("dlc_config_path")
            or ""
        )
        print("\n   DLC Model Configuration")
        print("   Path to your DeepLabCut model's config.yaml")
        dlc_path = prompt_path(
            "   DLC config.yaml path",
            default=dlc_default,
            must_exist=False,
            allow_empty=True
        )
        if dlc_path:
            watcher_config['dlc_config_path'] = str(dlc_path)

        # GPU device
        gpu_default = existing_watcher.get('dlc_gpu_device',
                       profile_watcher.get('dlc_gpu_device', 0))
        gpu_str = input(f"   GPU device number [{gpu_default}]: ").strip()
        watcher_config['dlc_gpu_device'] = int(gpu_str) if gpu_str else gpu_default

        # Poll interval
        poll_default = existing_watcher.get('poll_interval_seconds',
                        profile_watcher.get('poll_interval_seconds', 30))
        poll_str = input(f"   Poll interval in seconds [{poll_default}]: ").strip()
        watcher_config['poll_interval_seconds'] = int(poll_str) if poll_str else poll_default

        # Stability wait
        stable_default = existing_watcher.get('stability_wait_seconds',
                          profile_watcher.get('stability_wait_seconds', 60))
        stable_str = input(f"   File stability wait in seconds [{stable_default}]: ").strip()
        watcher_config['stability_wait_seconds'] = int(stable_str) if stable_str else stable_default

        watcher_config['enabled'] = True
        watcher_config['max_retries'] = existing_watcher.get('max_retries',
                                         profile_watcher.get('max_retries', 3))

    # --- Step 4: Create pipeline directories ---
    if proc_root:
        dirs_to_create = [
            proc_root / "DLC_Queue",
            proc_root / "Processing",
            proc_root / "Failed",
        ]
        if watcher_config:
            dirs_to_create.append(proc_root / "Quarantine")
        if nas_drive:
            nas_root = Path(nas_drive) / "! DLC Output" if str(nas_drive) != str(proc_root) else proc_root
            dirs_to_create.extend([
                nas_root / "Unanalyzed" / "Multi-Animal",
                nas_root / "Unanalyzed" / "Single_Animal",
                nas_root / "Analyzed",
            ])

        missing = [d for d in dirs_to_create if not d.exists()]
        if missing:
            print(f"\n4. Pipeline Directories")
            print(f"   {len(missing)} directories need to be created:")
            for d in missing:
                print(f"     {d}")
            create = input("   Create them now? [Y/n]: ").strip().lower()
            if create != 'n':
                for d in missing:
                    try:
                        d.mkdir(parents=True, exist_ok=True)
                        print(f"     Created: {d}")
                    except Exception as e:
                        print(f"     ERROR creating {d}: {e}")

    # --- Save configuration ---
    print("\n" + "=" * 60)
    print("Saving configuration...")
    save_config(
        nas_drive=nas_drive,
        processing_root=proc_root,
        watcher=watcher_config
    )

    print("\n  Configuration saved!")
    print("\nYour settings:")
    print(f"  Config file:       {CONFIG_FILE}")
    print(f"  Data Repository:   {nas_drive or '(not set)'}")
    print(f"  Processing Root:   {proc_root}")
    if watcher_config:
        print(f"  Watcher:           Enabled")
        print(f"    DLC Config:      {watcher_config.get('dlc_config_path', '(not set)')}")
        print(f"    GPU Device:      {watcher_config.get('dlc_gpu_device', 0)}")
        print(f"    Poll Interval:   {watcher_config.get('poll_interval_seconds', 30)}s")

    print("\nNext steps:")
    print("  1. Verify config:     mousereach-setup --show")
    if watcher_config:
        print("  2. Dry run (scan):    mousereach-watch --dry-run")
        print("  3. Process once:      mousereach-watch --once")
        print("  4. Start daemon:      mousereach-watch")
    else:
        print("  2. Launch napari:     mousereach")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
                print("  This field is required. Please enter a path.")
                continue

        path = Path(path_str)

        if must_exist and not path.exists():
            print(f"  Warning: {path} does not exist.")
            create = input("  Create it? [y/N]: ").lower().strip()
            if create == 'y':
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"  Created {path}")
                    return path
                except Exception as e:
                    print(f"  Error creating directory: {e}")
                    continue
            else:
                continue
        else:
            return path


def save_config(nas_drive, processing_root: Path, watcher: dict = None):
    """Save configuration to JSON file.

    Saves to ~/.mousereach/config.json for reliable cross-session persistence.

    Args:
        nas_drive: Path to NAS drive (can be None)
        processing_root: Path to pipeline processing root (required)
        watcher: Optional watcher configuration dict
    """
    nas_str = str(nas_drive) if nas_drive else ""
    proc_str = str(processing_root)

    # Ensure config directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing config to preserve other settings
    existing = load_config()

    # Save to JSON file (only include nas_drive if it's set)
    config = {
        "processing_root": proc_str
    }
    if nas_str:
        config["nas_drive"] = nas_str

    # Preserve or update watcher config
    if watcher is not None:
        config["watcher"] = watcher
    elif "watcher" in existing:
        config["watcher"] = existing["watcher"]

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

        # Also set for current session
        if nas_str:
            os.environ["MouseReach_NAS_DRIVE"] = nas_str
        os.environ["MouseReach_PROCESSING_ROOT"] = proc_str
    except Exception as e:
        print(f"\nError saving configuration: {e}")


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
    config = load_config()

    nas_drive = config.get("nas_drive") or os.getenv("MouseReach_NAS_DRIVE")
    proc_root = config.get("processing_root") or os.getenv("MouseReach_PROCESSING_ROOT")
    watcher = config.get("watcher", {})

    # Check lab profile
    profile, drives, method = detect_lab_profile()

    print("\n" + "=" * 60)
    print("Current MouseReach Configuration")
    print("=" * 60)

    if drives:
        drive_strs = []
        for letter in sorted(drives.keys()):
            locality = "local" if drives[letter] else "network"
            drive_strs.append(f"{letter}: ({locality})")
        print(f"\nDrives:            {', '.join(drive_strs)}")

    if profile:
        source = "identity file" if method == "identity_file" else "drive match"
        print(f"Machine Role:      {profile.get('name', 'Unknown')}  ({source})")
    else:
        print(f"Machine Role:      (not set - run mousereach-setup --set-role)")

    print(f"\nConfig file:       {CONFIG_FILE}")
    print(f"  Exists:          {CONFIG_FILE.exists()}")
    print(f"\nData Repository:   {nas_drive or '(not configured - optional)'}")
    print(f"Processing Root:   {proc_root or '(NOT CONFIGURED - REQUIRED)'}")

    if watcher:
        print(f"\nWatcher:           {'Enabled' if watcher.get('enabled') else 'Disabled'}")
        print(f"  DLC Config:      {watcher.get('dlc_config_path', '(not set)')}")
        print(f"  GPU Device:      {watcher.get('dlc_gpu_device', 0)}")
        print(f"  Poll Interval:   {watcher.get('poll_interval_seconds', 30)}s")
        print(f"  Stability Wait:  {watcher.get('stability_wait_seconds', 60)}s")
        print(f"  Auto Archive:    {watcher.get('auto_archive_approved', False)}")
        print(f"  Max Retries:     {watcher.get('max_retries', 3)}")
    else:
        print(f"\nWatcher:           Not configured")

    if not proc_root:
        print("\n  PROCESSING_ROOT is not configured!")
        print("   Run 'mousereach-setup' to configure before using MouseReach.")
    elif not nas_drive:
        print("\nNote: Data repository not configured (optional for basic pipeline)")

    print(f"\nLab profiles file: {LAB_PROFILES_FILE}")
    print(f"  Exists: {LAB_PROFILES_FILE.exists()}")
    print()


if __name__ == "__main__":
    cli_main()
