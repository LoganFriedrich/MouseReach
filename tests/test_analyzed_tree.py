"""Walks of Analyzed never read superseded outputs, and still find live poses.

Superseded outputs keep their original names under Analyzed/Archive, so a
plain rglob cannot tell them from live results. DLC Model folders, on the
other hand, are live pose storage and must stay visible.
"""
from pathlib import Path

from mousereach.pipeline.analyzed_tree import first_file, is_superseded_dir, iter_files

STEM = "20250101_CNT0101_P1"
POSE = f"{STEM}DLC_resnet101_MPSAOct27shuffle3_100000.h5"


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    return p


def test_only_archive_is_superseded():
    assert is_superseded_dir("Archive")
    for name in ("DLC Model 4", "DLC Model 3", "Connectome", "CNT01", "Multi-Animal", "_x", ".y"):
        assert not is_superseded_dir(name), name


def test_archived_copies_are_never_found(tmp_path):
    an = tmp_path / "Analyzed"
    live = _touch(an / "Connectome" / "CNT01" / f"{STEM}_features.json")
    _touch(an / "Archive" / "DLC Model 4.0" / "seg2.2.4_reach8.1.0_out6.1.0_asn2.1.0" / f"{STEM}_features.json")
    _touch(an / "Connectome" / "CNT01" / "Archive" / f"{STEM}_features.json")
    assert list(iter_files(an, "*_features.json")) == [live]


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
