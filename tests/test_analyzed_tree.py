"""Walks of Analyzed never read superseded outputs, and still find live poses.

Superseded outputs keep their original names under Analyzed/Archive, so a
plain rglob cannot tell them from live results. DLC Model folders, on the
other hand, are live pose storage and must stay visible.
"""
import os
from pathlib import Path

import pytest

from mousereach.pipeline.analyzed_tree import (
    SUPERSEDED_DIR_NAMES,
    first_file,
    is_superseded_dir,
    iter_files,
)

STEM = "20250101_CNT0101_P1"
POSE = f"{STEM}DLC_resnet101_MPSAOct27shuffle3_100000.h5"


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    return p


def test_only_superseded_folders_are_superseded():
    assert is_superseded_dir("Archive")
    # MouseReach's own pre-modification backups keep original names too.
    assert is_superseded_dir("_archived")
    for name in ("DLC Model 4", "DLC Model 3", "Connectome", "CNT01", "Multi-Animal", "_x", ".y"):
        assert not is_superseded_dir(name), name


def test_archived_copies_are_never_found(tmp_path):
    an = tmp_path / "Analyzed"
    live = _touch(an / "Connectome" / "CNT01" / f"{STEM}_features.json")
    _touch(an / "Archive" / "DLC Model 4.0" / "seg2.2.4_reach8.1.0_out6.1.0_asn2.1.0" / f"{STEM}_features.json")
    _touch(an / "Connectome" / "CNT01" / "Archive" / f"{STEM}_features.json")
    assert list(iter_files(an, "*_features.json")) == [live]


def test_backup_copies_under_archived_are_never_found(tmp_path):
    """backfill-manifest-versions run with --root Analyzed/<project> used to
    copy originals to Analyzed/_archived/manifests_pre_version_backfill_<ts>/.
    With SUPERSEDED_DIR_NAMES == {"Archive"} (the old set) iter_files yielded
    that copy beside the live manifest, so this assert failed."""
    an = tmp_path / "Analyzed"
    live = _touch(an / "Connectome" / "CNT01" / f"{STEM}_processing_manifest.json")
    _touch(an / "_archived" / "manifests_pre_version_backfill_20260914_120000"
           / f"{STEM}_processing_manifest.json")
    assert list(iter_files(an, "*_processing_manifest.json")) == [live]


def test_live_pose_storage_stays_visible(tmp_path):
    an = tmp_path / "Analyzed"
    pose = _touch(an / "Connectome" / "DLC Model 4" / "CNT01" / POSE)
    _touch(an / "Archive" / "DLC Model 4.0" / POSE)
    assert list(iter_files(an, f"{STEM}DLC*.h5")) == [pose]
    assert first_file(an, f"{STEM}DLC*.h5") == pose


def test_root_is_walked_whatever_its_name_and_missing_root_is_empty(tmp_path):
    f = _touch(tmp_path / "Archive" / f"{STEM}_features.json")
    assert list(iter_files(tmp_path / "Archive", "*_features.json")) == [f]
    assert list(iter_files(tmp_path / "missing", "*")) == []
    assert list(iter_files(None, "*")) == []
    assert first_file(tmp_path / "missing", "*") is None


def test_default_pattern_lists_files_not_folders(tmp_path):
    f = _touch(tmp_path / "a" / "b.json")
    (tmp_path / "empty_dir").mkdir()
    assert list(iter_files(tmp_path)) == [f]


# --- listing errors: the rule rglob had ------------------------------------

def _fail_listing(monkeypatch, bad: Path, exc: OSError):
    """os.walk lists each folder with the os module's scandir; make exactly
    one folder unlistable."""
    real = os.scandir
    target = os.path.normcase(os.path.normpath(str(bad)))

    def fake(path="."):
        if os.path.normcase(os.path.normpath(os.fspath(path))) == target:
            raise exc
        return real(path)

    monkeypatch.setattr(os, "scandir", fake)


def test_network_error_listing_a_subfolder_raises(tmp_path, monkeypatch):
    """A NAS hiccup (WinError 59) listing one cohort folder. rglob raised it;
    os.walk's default onerror=None silently skipped the folder and returned
    only the other cohort's file -- a partial answer that looks complete. The
    old iter_files therefore did not raise and this test failed."""
    an = tmp_path / "Analyzed"
    _touch(an / "Connectome" / "CNT01" / f"{STEM}_features.json")
    bad = an / "Connectome" / "CNT02"
    _touch(bad / "20250101_CNT0201_P1_features.json")
    _fail_listing(monkeypatch, bad, OSError(
        22, "The specified network name is no longer available", str(bad), 59))

    with pytest.raises(OSError) as ei:
        list(iter_files(an, "*_features.json"))
    assert getattr(ei.value, "winerror", None) == 59


def test_permission_denied_subfolder_is_skipped_like_rglob(tmp_path, monkeypatch):
    """Regression guard, not a fix: rglob skipped a folder it had no
    permission to list, and iter_files keeps doing so."""
    an = tmp_path / "Analyzed"
    live = _touch(an / "Connectome" / "CNT01" / f"{STEM}_features.json")
    bad = an / "Connectome" / "CNT02"
    _touch(bad / "20250101_CNT0201_P1_features.json")
    _fail_listing(monkeypatch, bad, PermissionError(13, "Access is denied", str(bad)))

    assert list(iter_files(an, "*_features.json")) == [live]


# --- contract with archive.supersede ---------------------------------------
# Every walker's test builds its own Analyzed/Archive tree by hand, so none of
# them would notice if supersede wrote somewhere else. These two tie the
# skipped name to the folder supersede actually uses.

def _supersede_root(tmp_path, monkeypatch):
    from mousereach.config import Paths
    from mousereach.archive.supersede import default_archive_root
    nas = tmp_path / "nas"
    monkeypatch.setattr(Paths, "NAS_ROOT", nas)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", nas / "Analyzed")
    return default_archive_root(), Paths


def test_supersede_archive_folder_name_is_skipped_by_every_walker(tmp_path, monkeypatch):
    """Fails the moment supersede's folder is renamed ("Superseded",
    "_Archive" ...) without updating SUPERSEDED_DIR_NAMES."""
    root, _ = _supersede_root(tmp_path, monkeypatch)
    assert root.name in SUPERSEDED_DIR_NAMES


def test_importing_the_rule_does_not_load_napari():
    """Headless tools import this module (mousereach-reconcile, the census, the
    version scan). The pipeline package used to import its napari widget
    eagerly, so this import took about six seconds and loaded Qt. Checked in a
    fresh interpreter, because this test process may already have napari."""
    import subprocess
    import sys
    code = ("import sys, mousereach.pipeline.analyzed_tree; "
            "print('napari' in sys.modules, 'qtpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, env=dict(os.environ))
    assert out.stdout.split() == ["False", "False"]


def test_package_public_names_still_resolve():
    """The lazy package keeps `from mousereach.pipeline import X` working."""
    import mousereach.pipeline as pipeline
    assert "UnifiedPipelineWidget" in pipeline.__all__
    assert pipeline.UnifiedPipelineProcessor.__name__ == "UnifiedPipelineProcessor"
    with pytest.raises(AttributeError):
        pipeline.not_a_name


def test_supersede_archive_root_sits_inside_analyzed(tmp_path, monkeypatch):
    """Superseded outputs live in Analyzed/Archive, not a top-level Archive
    beside Unanalyzed/ and Processing/. The old default_archive_root returned
    <NAS_ROOT>/Archive, so root.parent was NAS_ROOT and this failed."""
    root, Paths = _supersede_root(tmp_path, monkeypatch)
    assert root.parent == Paths.ANALYZED_OUTPUT
    assert root != Paths.NAS_ROOT / "Archive"


def test_supersede_archive_root_is_none_without_analyzed(tmp_path, monkeypatch):
    """No configured Analyzed tree -> no archive root, so supersede refuses
    instead of writing into some default folder."""
    from mousereach.config import Paths
    from mousereach.archive.supersede import default_archive_root, supersede_video_outputs
    monkeypatch.setattr(Paths, "NAS_ROOT", None)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", None)
    assert default_archive_root() is None
    out = supersede_video_outputs(STEM, tmp_path)
    assert "error" in out and out["pose"] == [] and out["algo"] == []
