"""Interactive configuration wizard for MouseReach.

This module provides a CLI-based setup wizard that guides users through
configuring MouseReach for their system. It saves configuration to a JSON file
in the user's home directory for reliable persistence.

Machine identification priority:
  1. Local identity file (~/.mousereach/machine_role.json) - explicit declaration
  2. The lab's profiles file, matched by hostname and drive pattern
  3. Fully manual - user enters paths interactively

The lab's profiles file is looked for at the path in the
MOUSEREACH_LAB_PROFILES environment variable, then at
~/.mousereach/lab_profiles.json, then as lab_profiles.json beside this
module. The shipped lab_profiles.json.example is a template to copy from; it
is never matched against a machine.

A matched profile pre-fills every setting this machine does not already have
(the machine's own settings always win), including watcher settings the
wizard does not ask about (mode, also_process, pause_while_running,
recording_days, recording_hours, ...) and other sections such as "backup".

The local identity file is the simplest approach: drop a small JSON file on
each PC that says what role it plays. The wizard reads it and pre-fills
defaults from the matching lab profile.

To set up a machine's identity:
    mousereach-setup --set-role "NAS / DLC PC"

Or create ~/.mousereach/machine_role.json manually:
    {"role": "NAS / DLC PC"}

Copy lab_profiles.json.example, edit it for your lab and put it in one of
those places; with none, the wizard has no pre-filled defaults.
"""

import os
import sys
import json
from pathlib import Path

# Config file location
CONFIG_DIR = Path.home() / ".mousereach"
CONFIG_FILE = CONFIG_DIR / "config.json"
MACHINE_ROLE_FILE = CONFIG_DIR / "machine_role.json"

# The watcher settings this wizard actually prompts for. Everything else
# already in the config file is carried over untouched when the watcher
# section is rewritten -- see run_setup_wizard().
_WIZARD_PROMPTED_KEYS = frozenset({
    'dlc_config_path', 'dlc_gpu_device', 'poll_interval_seconds',
    'stability_wait_seconds', 'enabled', 'max_retries',
})

# The lab's machine profiles (lab-specific; never shipped). Looked for, in
# order: the file named by the MOUSEREACH_LAB_PROFILES environment variable
# (one shared copy that every machine can point at), then
# ~/.mousereach/lab_profiles.json (beside this machine's other lab settings),
# then lab_profiles.json beside this module (the original location). WHY the
# first two: the third is git-ignored, so no pull ever brings it and every
# re-clone loses it -- on 2026-10-22 none of the lab's GPU PCs had one.
#
# The shipped lab_profiles.json.example is a TEMPLATE to copy and edit. It is
# never matched against a machine. WHY: it used to be the fallback, and its
# drive patterns matched real machines and offered them a retired folder
# layout as their defaults (found 2026-10-22).
LAB_PROFILES_ENV = "MOUSEREACH_LAB_PROFILES"
USER_LAB_PROFILES_FILE = CONFIG_DIR / "lab_profiles.json"
LAB_PROFILES_FILE = Path(__file__).parent / "lab_profiles.json"
LAB_PROFILES_EXAMPLE = Path(__file__).parent / "lab_profiles.json.example"

# Profile sections the wizard handles itself; any other top-level section a
# profile gives (e.g. "backup") is added when this machine has none.
_PROFILE_HANDLED_SECTIONS = frozenset({
    'nas_drive', 'nas_root', 'processing_root', 'watcher'})


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

def lab_profiles_candidates() -> list:
    """Where this machine looks for the lab's profiles, in order (see
    LAB_PROFILES_ENV above)."""
    env = os.environ.get(LAB_PROFILES_ENV, "").strip()
    return ([Path(env)] if env else []) + [USER_LAB_PROFILES_FILE, LAB_PROFILES_FILE]


def lab_profiles_path():
    """The lab profiles file this machine uses, or None when it has none."""
    for p in lab_profiles_candidates():
        if p.is_file():
            return p
    return None


def _load_lab_profiles() -> list:
    """The lab's machine profiles; [] when this machine has no profiles file.
    Never the shipped template (see LAB_PROFILES_ENV above)."""
    env = os.environ.get(LAB_PROFILES_ENV, "").strip()
    if env and not Path(env).is_file():
        print(f"[!] {LAB_PROFILES_ENV} names {env}, which is not a file; "
              f"looking in the other places instead")
    profiles_path = lab_profiles_path()
    if profiles_path is None:
        return []
    try:
        with open(profiles_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("profiles", [])
    except Exception as e:
        print(f"[!] Could not read the lab profiles file {profiles_path}: {e}")
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


def _match_profile(profile: dict, drives: dict, hostname: str = None) -> bool:
    """Check if a profile's match rules fit this machine. Every rule given
    must hold.

    Args:
        profile: A profile dict from lab_profiles.json
        drives: Dict of {letter: is_local} from _get_current_drives()
        hostname: this machine's name (default: socket.gethostname())

    WHY the hostname rule is checked: profiles documented "hostname: match
    by machine hostname (most reliable)", but only the drive rules were
    read, so two PCs with the same drive layout both took the first
    profile listed (found 2026-10-22).
    """
    match = profile.get("match", {})
    available = set(drives.keys())

    # hostname: this machine's name, or one of a list of names
    wanted = match.get("hostname")
    if wanted:
        if hostname is None:
            import socket
            hostname = socket.gethostname()
        names = [wanted] if isinstance(wanted, str) else list(wanted)
        if hostname.lower() not in {str(n).lower() for n in names}:
            return False

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
        is "identity_file", "hostname_match" (the matched profile has a
        hostname rule), "drive_match" (drive rules only), or None.

    An identity file naming a role that is not in the profiles is said out
    loud, then matching goes on. WHY: it used to fall through silently, so a
    PC whose identity file still named a retired role looked declared while
    it was really being matched (found 2026-10-22).
    """
    drives = _get_current_drives()

    # Priority 1: Local identity file
    role = _load_machine_role()
    if role:
        profile = _find_profile_by_name(role)
        if profile:
            return profile, drives, "identity_file"
        print(f"[!] {MACHINE_ROLE_FILE} names the role \"{role}\", which is not in "
              f"the lab profiles; matching this machine instead. Fix it with: "
              f"mousereach-setup --set-role \"<role name>\"")

    # Priority 2: Hostname and drive-pattern matching
    profiles = _load_lab_profiles()
    for profile in profiles:
        if _match_profile(profile, drives):
            by_name = bool(profile.get("match", {}).get("hostname"))
            return profile, drives, "hostname_match" if by_name else "drive_match"

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
                print("No lab profiles found. Looked at, in order:")
                for p in lab_profiles_candidates():
                    print(f"  {p}")
                print(f"  (set {LAB_PROFILES_ENV} to point at a shared copy;")
                print(f"   template to copy: {LAB_PROFILES_EXAMPLE})")
                return
            # Say which file was read and which role this machine would get,
            # so running this on a PC is enough to check it (asked for by a
            # GPU node's session, 2026-10-22).
            print(f"\nLab profiles file: {lab_profiles_path()}")
            current_role = _load_machine_role()
            mine, _, how = detect_lab_profile()
            print("\nAvailable machine roles:")
            for p in profiles:
                name = p.get("name", "Unknown")
                desc = p.get("description", "")
                marker = ""
                if mine is not None and mine.get("name") == name:
                    marker = f"  <- THIS MACHINE ({how.replace('_', ' ')})"
                elif current_role and current_role.lower() == name.lower():
                    marker = " (current)"
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

The wizard asks, in order: the data repository (nas_drive); the shared
pipeline folder (nas_root -- the folder holding Unanalyzed, Processing,
Analyzed and pipeline_versions.json, the same one on every machine); the
local processing root; and the watcher settings. It then offers to create
the pipeline folders inside the shared pipeline folder.

Saving writes only those settings and keeps every other one already in the
config file (backup folders, watcher settings it does not ask about). There
is no guessed pipeline folder: with nas_root unset, every tool that needs
the shared folders stops and says so.

Configuration is saved to: {CONFIG_FILE}
Identity file: {MACHINE_ROLE_FILE}
Lab profiles file: {lab_profiles_path() or '(none -- looked at ' + ', '.join(str(p) for p in lab_profiles_candidates()) + ')'}
Template to copy: {LAB_PROFILES_EXAMPLE}
""")


# =============================================================================
# SETUP WIZARD
# =============================================================================

def _carry_unprompted(watcher_config: dict, existing_watcher: dict,
                      profile_watcher: dict) -> dict:
    """Fill ``watcher_config`` with every watcher setting the wizard does not
    ask about: this machine's own first, then the lab profile's. Existing
    values always win. Returns ``watcher_config``.

    WHY this machine's: save_config() used to REPLACE the whole watcher
    section, and on a processing server the wizard's own output drops mode,
    db_path, max_local_pending, also_process, staging_path and work_priority.

    WHY the profile's: only the five prompted settings were ever taken from
    a profile, so a machine set up from scratch lost the rest of its role --
    mode, also_process, and the pause for the recording program with its
    recording days and hours. A behaviour PC rebuilt that way would pose on
    top of a recording (found 2026-10-22).
    """
    for source in (existing_watcher or {}, profile_watcher or {}):
        for key, value in source.items():
            if key not in _WIZARD_PROMPTED_KEYS:
                watcher_config.setdefault(key, value)
    return watcher_config


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
        elif method == "hostname_match":
            print(f"\nAuto-detected profile: {name}  (matched this machine's hostname)")
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

    # --- Step 2: Shared pipeline folder (nas_root) ---
    # WHY asked for explicitly: every shared path (intake, posed videos, review
    # queues, Analyzed) hangs off it, and it used to be neither asked for nor
    # saved -- a save from this wizard deleted it, after which the code guessed
    # a retired layout (config.py says more).
    root_default = (
        existing.get("nas_root")
        or profile_defaults.get("nas_root")
        or ""
    )
    print("\n2. Shared Pipeline Folder (nas_root)")
    print("   The folder on your data repository that holds the pipeline itself:")
    print("   Unanalyzed/, Processing/, Analyzed/ and pipeline_versions.json.")
    print("   Every machine in the lab points at the same folder. Leave it blank")
    print("   only on a machine that never uses shared storage.")
    nas_root = prompt_path(
        "   Pipeline folder",
        default=root_default,
        must_exist=False,
        allow_empty=True
    )
    if nas_root and not (nas_root / "pipeline_versions.json").exists():
        print("   [!] There is no pipeline_versions.json in that folder yet. If this")
        print("       is a new pipeline, create it with: mousereach-version-check --init")

    # --- Step 3: Processing root ---
    proc_default = (
        existing.get("processing_root")
        or profile_defaults.get("processing_root")
        or os.getenv("MouseReach_PROCESSING_ROOT")
        or ""
    )
    print("\n3. Pipeline Processing Root [REQUIRED]")
    print("   Parent folder for all working pipeline directories.")
    print("   The watcher creates these subfolders automatically:")
    print("     <root>/DLC_Queue/     - videos waiting for DLC")
    print("     <root>/Processing/    - post-DLC analysis")
    print("     <root>/Failed/        - errors requiring investigation")
    print("   Use a fast local disk on this machine, not the shared folder.")
    proc_root = prompt_path(
        "   Enter path",
        default=proc_default,
        must_exist=False,
        allow_empty=False
    )

    # --- Step 4: Watcher configuration ---
    watcher_config = None
    print("\n4. Automated Watcher")
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

        _carry_unprompted(watcher_config, existing_watcher, profile_watcher)

    # --- Step 5: Create pipeline directories ---
    if proc_root:
        # Local compute working dirs (on the processing root, NOT the NAS pipe).
        dirs_to_create = [
            proc_root / "DLC_Queue",
            proc_root / "Processing",
        ]
        if nas_root:
            # The canonical NAS pipe tree, built in the shared folder given in
            # step 2 -- never guessed from the drive (it used to be built under
            # <drive>/! DLC Output, a layout retired in 2026) -- single source of truth in
            # pipeline.pipe_structure (Unanalyzed[Multi-Animal, Single_Animal]
            # / Processing[Posed, Repose_Queue, Review/{triage,deep_review},
            # Quarantine, Failed] / Analyzed[Archive]).
            from mousereach.pipeline.pipe_structure import TARGET_DIRS
            dirs_to_create.extend(nas_root / rel for rel in TARGET_DIRS)

        missing = [d for d in dirs_to_create if not d.exists()]
        if missing:
            print(f"\n5. Pipeline Directories")
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
        watcher=watcher_config,
        nas_root=nas_root,
        sections={k: v for k, v in profile_defaults.items()
                  if k not in _PROFILE_HANDLED_SECTIONS},
    )

    print("\n  Configuration saved!")
    print("\nYour settings:")
    print(f"  Config file:       {CONFIG_FILE}")
    print(f"  Data Repository:   {nas_drive or '(not set)'}")
    print(f"  Pipeline Folder:   {nas_root or '(not set)'}")
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


# save_config's "leave this setting as it is" (None or "" removes it instead).
KEEP = object()


def save_config(nas_drive=KEEP, processing_root: Path = None, watcher: dict = None,
                nas_root=KEEP, sections: dict = None):
    """Write the settings setup manages into ~/.mousereach/config.json, keeping
    EVERY other setting already in the file.

    WHY: this used to rewrite the file with processing_root, nas_drive and
    watcher only, so saving from mousereach-setup or the Setup panel silently
    deleted the shared pipeline folder (nas_root), the nightly backup's
    folders and anything else the file held (found 2026-10-22).

    Args:
        nas_drive, nas_root: KEEP leaves the setting as it is; None or "" removes
            it; a path sets it.
        processing_root: the local working folder; set when given.
        watcher: merged into the existing watcher section -- the keys given win,
            every other watcher setting is kept.
        sections: other top-level sections from a lab profile (e.g. "backup");
            each is added only when the file has no section of that name, so
            a machine's own settings are never replaced by a profile's.
    """
    existing = load_config()
    config = dict(existing)
    for key, value in (sections or {}).items():
        config.setdefault(key, value)
    if processing_root is not None:
        config["processing_root"] = str(processing_root)
    for key, value in (("nas_drive", nas_drive), ("nas_root", nas_root)):
        if value is KEEP:
            continue
        if value:
            config[key] = str(value)
        else:
            config.pop(key, None)
    if watcher is not None:
        merged = dict(existing.get("watcher") or {})
        merged.update(watcher)
        config["watcher"] = merged

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_name(CONFIG_FILE.name + ".tmp")
        tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(tmp, CONFIG_FILE)

        # Also set for current session
        if config.get("nas_drive"):
            os.environ["MouseReach_NAS_DRIVE"] = config["nas_drive"]
        if config.get("processing_root"):
            os.environ["MouseReach_PROCESSING_ROOT"] = config["processing_root"]
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
    nas_root = config.get("nas_root")
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
        source = method.replace("_", " ") if method else "?"
        print(f"Machine Role:      {profile.get('name', 'Unknown')}  ({source})")
    else:
        print(f"Machine Role:      (not set - run mousereach-setup --set-role)")

    print(f"\nConfig file:       {CONFIG_FILE}")
    print(f"  Exists:          {CONFIG_FILE.exists()}")
    print(f"\nData Repository:   {nas_drive or '(not configured - optional)'}")
    print(f"Pipeline Folder:   {nas_root or '(not configured - shared folders unavailable)'}")
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

    used = lab_profiles_path()
    print(f"\nLab profiles file: {used or '(none)'}")
    if used is None:
        print("  Looked at: " + ", ".join(str(p) for p in lab_profiles_candidates()))
    print()


if __name__ == "__main__":
    cli_main()
