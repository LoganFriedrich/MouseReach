"""A re-run finds the archived pose where it lives before walking the archive.

WHY (2026-10-22): every re-run walked the whole Analyzed tree for the pose --
36 s on a quiet server, about 4 minutes on a busy one -- for a re-run whose own
work takes 3 s. The pose is in the results folder or the same cohort folder of
a per-model tree, both computable from the video id; the walk stays as the
fallback when no pose from the declared model is there.
"""
import mousereach.archive.core as archive_core
import mousereach.pipeline.analyzed_tree as analyzed_tree
from mousereach.watcher import orchestrator as o

V = "20250625_CNT0106_P1"
NEW = "DLC_resnet101_MPSAOct27shuffle3_100000"
OLD = "DLC_resnet50_MPSAOct27shuffle1_100000"


def _tree(tmp_path, monkeypatch, in_results=(NEW,), in_model=((3, OLD), (4, NEW))):
    project = tmp_path / "Analyzed" / "Connectome"
    results = project / "CNT01"
    results.mkdir(parents=True)
    for scorer in in_results:
        (results / f"{V}{scorer}.h5").write_bytes(b"p")
    for n, scorer in in_model:
        d = project / f"DLC Model {n}" / "CNT01"
        d.mkdir(parents=True)
        (d / f"{V}{scorer}.h5").write_bytes(b"p")
    monkeypatch.setattr(archive_core, "get_archive_destination", lambda vid: results)
    return tmp_path / "Analyzed"


def test_the_pose_is_found_beside_the_results_without_a_walk(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)

    def no_walk(*a, **k):
        raise AssertionError("the archive was walked")

    monkeypatch.setattr(analyzed_tree, "iter_files", no_walk)
    found = o.archived_pose_candidates(V, NEW, root)
    assert sorted(f.parent.name for f in found) == ["CNT01", "CNT01", "CNT01"]
    assert sum(NEW in f.name for f in found) == 2


def test_the_archive_is_walked_when_no_declared_pose_is_near(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch, in_results=(), in_model=((3, OLD),))
    elsewhere = root / "Connectome" / "odd place"
    elsewhere.mkdir()
    (elsewhere / f"{V}{NEW}.h5").write_bytes(b"p")
    found = o.archived_pose_candidates(V, NEW, root)
    assert any(f.parent == elsewhere for f in found)
