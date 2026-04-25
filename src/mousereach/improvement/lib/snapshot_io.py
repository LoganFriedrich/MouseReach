"""
Snapshot I/O helpers for the MouseReach Improvement Process.

Handles reading/writing manifest.json, resolving snapshot paths, and
managing the metrics/ directory within each snapshot.

Usage:
    from mousereach.improvement.lib.snapshot_io import (
        get_snapshots_root,
        snapshot_dir,
        write_snapshot,
        read_snapshot,
        list_snapshots,
    )

    root = get_snapshots_root()  # -> .../MouseReach_Pipeline/Improvement_Snapshots
    sd = snapshot_dir("segmentation", "seg_v2.1.3_phantom_first_post_validation")
    write_snapshot(sd, manifest)
    m = read_snapshot(sd)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

from .manifest import Manifest


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_snapshots_root() -> Path:
    """Return the Improvement_Snapshots directory under MouseReach_Pipeline.

    Resolution order:
      1. MOUSEREACH_SNAPSHOTS_ROOT environment variable (if set).
      2. CONNECTOME_ROOT / Behavior / MouseReach_Pipeline / Improvement_Snapshots.
      3. Y:\\2_Connectome\\Behavior\\MouseReach_Pipeline\\Improvement_Snapshots (fallback).
    """
    env = os.environ.get("MOUSEREACH_SNAPSHOTS_ROOT")
    if env:
        return Path(env)

    connectome_root = os.environ.get("CONNECTOME_ROOT")
    if connectome_root:
        return Path(connectome_root) / "Behavior" / "MouseReach_Pipeline" / "Improvement_Snapshots"

    return Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots")


def snapshot_dir(phase: str, snapshot_name: str,
                 root: Optional[Path] = None) -> Path:
    """Return the directory for a specific snapshot.

    Parameters
    ----------
    phase : str
        One of "segmentation", "reach_detection", "outcome", "features".
    snapshot_name : str
        Directory name of the snapshot (e.g. "seg_v2.1.3_phantom_first_post_validation").
    root : Path, optional
        Override the snapshots root. Defaults to get_snapshots_root().
    """
    if root is None:
        root = get_snapshots_root()
    return root / phase / snapshot_name


def vault_template_dir() -> Path:
    """Return the path to the vault_template shipped with the package."""
    return Path(__file__).parent / "vault_template"


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------

def write_snapshot(dest: Path, manifest: Manifest,
                   copy_vault_template: bool = True) -> Path:
    """Create a snapshot directory and write its manifest.json.

    Parameters
    ----------
    dest : Path
        Target snapshot directory (will be created if needed).
    manifest : Manifest
        Metadata to write.
    copy_vault_template : bool
        If True, copy the vault_template into dest/vault/ (creating
        .obsidian/ so Obsidian treats it as a vault).

    Returns
    -------
    Path
        The path to the written manifest.json.
    """
    dest.mkdir(parents=True, exist_ok=True)

    # Sub-directories
    (dest / "figures").mkdir(exist_ok=True)
    (dest / "metrics").mkdir(exist_ok=True)

    # Vault
    vault_dir = dest / "vault"
    if copy_vault_template:
        template = vault_template_dir()
        if template.exists():
            if not vault_dir.exists():
                shutil.copytree(str(template), str(vault_dir))
            else:
                # Ensure .obsidian exists even if vault was pre-created
                obsidian_dir = vault_dir / ".obsidian"
                obsidian_dir.mkdir(exist_ok=True)
                for cfg_name in ("app.json", "workspace.json"):
                    cfg = obsidian_dir / cfg_name
                    if not cfg.exists():
                        cfg.write_text("{}", encoding="utf-8")
        else:
            vault_dir.mkdir(exist_ok=True)
            obsidian_dir = vault_dir / ".obsidian"
            obsidian_dir.mkdir(exist_ok=True)
            (obsidian_dir / "app.json").write_text("{}", encoding="utf-8")
            (obsidian_dir / "workspace.json").write_text("{}", encoding="utf-8")
    else:
        vault_dir.mkdir(exist_ok=True)

    manifest_path = dest / "manifest.json"
    manifest.to_json(manifest_path)
    return manifest_path


def read_snapshot(dest: Path) -> Manifest:
    """Read a snapshot's manifest.json and return a Manifest object.

    Parameters
    ----------
    dest : Path
        Snapshot directory containing manifest.json.

    Raises
    ------
    FileNotFoundError
        If manifest.json does not exist in dest.
    """
    manifest_path = dest / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in {dest}. "
            f"Is this a valid snapshot directory?"
        )
    return Manifest.from_json(manifest_path)


def list_snapshots(phase: str, root: Optional[Path] = None) -> List[Path]:
    """List all snapshot directories for a given phase, sorted by name.

    Parameters
    ----------
    phase : str
        One of "segmentation", "reach_detection", "outcome", "features".
    root : Path, optional
        Override the snapshots root.

    Returns
    -------
    list of Path
        Sorted list of snapshot directory paths.
    """
    if root is None:
        root = get_snapshots_root()
    phase_dir = root / phase
    if not phase_dir.exists():
        return []
    return sorted(
        [d for d in phase_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    )


def ensure_metrics_dir(snapshot_path: Path) -> Path:
    """Ensure the metrics/ subdirectory exists and return its path."""
    metrics = snapshot_path / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    return metrics
