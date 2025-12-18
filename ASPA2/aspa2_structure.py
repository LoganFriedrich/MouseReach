"""
ASPA2 Project Structure Manager
================================

Defines the canonical project structure and can:
- Show what the structure SHOULD be
- Check what currently EXISTS
- Create missing directories
- Identify misplaced files

Run:
    python aspa2_structure.py          # Show expected structure
    python aspa2_structure.py check    # Compare expected vs actual
    python aspa2_structure.py init     # Create missing directories
    python aspa2_structure.py report   # Detailed report
"""

from pathlib import Path
from typing import List
from dataclasses import dataclass

# ============================================================
# CANONICAL STRUCTURE DEFINITION
# Edit this to change what the project SHOULD look like
# ============================================================

STRUCTURE = {
    "aspa2_core": {
        "_description": "Core library - all algorithms live here",
        "_files": ["__init__.py"],
        "segmenter.py": "Video segmentation (find 21 boundaries)",
        "dlc_utils.py": "Load and preprocess DLC files",
        "calibration.py": "Ruler detection, px-to-mm conversion",
        "reach_detector.py": "Detect reach events within segments",
        "scorer.py": "Score reaches (success/fail/etc)",
    },
    "scripts": {
        "_description": "Pipeline scripts - run in order",
        "0_run_pipeline.py": "Run all steps in sequence",
        "1_segment.py": "Segment videos into pellet periods",
        "2_calibrate.py": "Calibrate pixel-to-mm from rulers",
        "3_reaches.py": "Detect reach events",
        "4_score.py": "Score each reach",
        "5_compile.py": "Compile results to spreadsheet",
    },
    "tools": {
        "_description": "GUI tools and plugins",
        "napari_viewer": {
            "_description": "Napari plugin for segment verification",
            "_files": ["__init__.py", "napari.yaml"],
            "_widget.py": "Main viewer widget",
        },
    },
    "config": {
        "_description": "Configuration files",
        "default_config.json": "Default pipeline settings",
    },
    "tests": {
        "_description": "Test files",
        "test_segmenter.py": "Segmenter unit tests",
        "test_data": {
            "_description": "Small test datasets",
        },
    },
    "_root_files": {
        "pyproject.toml": "Package configuration",
        "README.md": "Project documentation",
        "aspa2_structure.py": "This file - structure manager",
    },
}


@dataclass
class FileStatus:
    path: Path
    expected: bool
    exists: bool
    description: str = ""


def get_expected_paths(structure: dict, base: Path = Path("."), paths: List[FileStatus] = None) -> List[FileStatus]:
    """Recursively get all expected paths from structure definition."""
    if paths is None:
        paths = []
    
    for name, value in structure.items():
        if name.startswith("_"):
            if name == "_files":
                for f in value:
                    paths.append(FileStatus(base / f, expected=True, exists=False, description=""))
            continue
        
        path = base / name
        
        if isinstance(value, dict):
            desc = value.get("_description", "")
            paths.append(FileStatus(path, expected=True, exists=False, description=desc))
            get_expected_paths(value, path, paths)
        else:
            paths.append(FileStatus(path, expected=True, exists=False, description=value))
    
    return paths


def check_structure(root: Path = Path(".")) -> List[FileStatus]:
    """Check expected structure against actual filesystem."""
    all_structure = STRUCTURE.copy()
    root_files = all_structure.pop("_root_files", {})
    
    paths = get_expected_paths(all_structure, root)
    
    for name, desc in root_files.items():
        paths.append(FileStatus(root / name, expected=True, exists=False, description=desc))
    
    for status in paths:
        status.exists = status.path.exists()
    
    return paths


def find_unexpected_files(root: Path = Path("."), expected_paths: List[FileStatus] = None) -> List[Path]:
    """Find files that exist but aren't in the expected structure."""
    if expected_paths is None:
        expected_paths = check_structure(root)
    
    expected_set = {s.path.resolve() for s in expected_paths}
    
    unexpected = []
    for item in root.rglob("*"):
        if item.is_file():
            if any(part.startswith(".") or part == "__pycache__" for part in item.parts):
                continue
            if item.resolve() not in expected_set:
                unexpected.append(item)
    
    return unexpected


def print_structure():
    """Print the expected structure as a tree."""
    print("ASPA2 Expected Project Structure")
    print("=" * 50)
    
    def print_tree(structure: dict, indent: int = 0):
        prefix = "  " * indent
        for name, value in structure.items():
            if name.startswith("_"):
                if name == "_description":
                    print(f"{prefix}# {value}")
                continue
            
            if isinstance(value, dict):
                print(f"{prefix}{name}/")
                print_tree(value, indent + 1)
            else:
                print(f"{prefix}{name}")
    
    print_tree(STRUCTURE)


def print_check(root: Path = Path(".")):
    """Print comparison of expected vs actual."""
    print(f"Checking structure in: {root.resolve()}")
    print("=" * 50)
    
    paths = check_structure(root)
    
    missing = [p for p in paths if p.expected and not p.exists]
    present = [p for p in paths if p.expected and p.exists]
    
    print(f"\n✓ Present ({len(present)}):")
    for p in present:
        print(f"  {p.path}")
    
    print(f"\n✗ Missing ({len(missing)}):")
    for p in missing:
        print(f"  {p.path}")
    
    unexpected = find_unexpected_files(root, paths)
    if unexpected:
        print(f"\n? Unexpected ({len(unexpected)}):")
        for p in unexpected[:20]:
            print(f"  {p}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")


def init_structure(root: Path = Path(".")):
    """Create missing directories and placeholder files."""
    print(f"Initializing structure in: {root.resolve()}")
    print("=" * 50)
    
    paths = check_structure(root)
    
    created = []
    for status in paths:
        if not status.exists:
            if status.path.suffix:
                status.path.parent.mkdir(parents=True, exist_ok=True)
                if status.path.suffix == ".py":
                    content = f'"""\n{status.description or status.path.name}\n\nTODO: Implement\n"""\n'
                elif status.path.suffix == ".json":
                    content = "{}\n"
                elif status.path.suffix == ".md":
                    content = f"# {status.path.stem}\n\n{status.description or 'TODO'}\n"
                else:
                    content = ""
                status.path.write_text(content)
                created.append(status.path)
            else:
                status.path.mkdir(parents=True, exist_ok=True)
                created.append(status.path)
    
    print(f"Created {len(created)} items:")
    for p in created:
        print(f"  {p}")


def print_report(root: Path = Path(".")):
    """Detailed report with suggestions."""
    print(f"ASPA2 Structure Report")
    print(f"Root: {root.resolve()}")
    print("=" * 50)
    
    paths = check_structure(root)
    missing = [p for p in paths if p.expected and not p.exists]
    present = [p for p in paths if p.expected and p.exists]
    unexpected = find_unexpected_files(root, paths)
    
    print(f"\nSummary:")
    print(f"  Expected items: {len(paths)}")
    print(f"  Present: {len(present)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")
    
    print(f"\nCompleteness by section:")
    for section in ["aspa2_core", "scripts", "tools", "config", "tests"]:
        section_paths = [p for p in paths if len(p.path.parts) > 0 and p.path.parts[0] == section]
        if section_paths:
            section_present = sum(1 for p in section_paths if p.exists)
            pct = 100 * section_present / len(section_paths) if section_paths else 0
            print(f"  {section}: {section_present}/{len(section_paths)} ({pct:.0f}%)")
    
    print(f"\nSuggestions:")
    if missing:
        print(f"  - Run 'python aspa2_structure.py init' to create missing items")
    if unexpected:
        print(f"  - Review unexpected files - may need to be moved or added to STRUCTURE")


if __name__ == "__main__":
    import sys
    
    root = Path(".")
    
    if len(sys.argv) < 2:
        print_structure()
    elif sys.argv[1] == "check":
        print_check(root)
    elif sys.argv[1] == "init":
        init_structure(root)
    elif sys.argv[1] == "report":
        print_report(root)
    else:
        print("Usage:")
        print("  python aspa2_structure.py          # Show expected structure")
        print("  python aspa2_structure.py check    # Compare expected vs actual")
        print("  python aspa2_structure.py init     # Create missing directories")
        print("  python aspa2_structure.py report   # Detailed report")
