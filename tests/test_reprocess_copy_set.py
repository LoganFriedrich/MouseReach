"""Reprocess staging: the pose tree contributes only pose artifacts.

WHY: the pose tree can hold scattered analysis jsons from runs whose pose
path pointed into it, and set-ordered copies let an ALGO segments.json from
there clobber the results dir's HUMAN one -- a reviewer's hand-set cuts were
re-segmented away exactly this way (2026-09-08).
"""
from mousereach.watcher.orchestrator import _reprocess_copy_set


def test_pose_dir_jsons_are_excluded(tmp_path):
    stem = "20240101_ABC0101_P1"
    pose = tmp_path / "model"
    res = tmp_path / "results"
    pose.mkdir(), res.mkdir()
    h5 = pose / (stem + "DLC_newmodel.h5")
    h5.write_text("x", encoding="utf-8")
    algo_seg = pose / (stem + "_segments.json")            # scattered stray
    algo_seg.write_text('{"boundary_source": null}', encoding="utf-8")
    human_seg = res / (stem + "_segments.json")
    human_seg.write_text('{"boundary_source": "human"}', encoding="utf-8")

    files = _reprocess_copy_set({pose, res}, pose, stem)
    assert h5 in files
    assert human_seg in files
    assert algo_seg not in files                           # never the clobber


def test_single_source_keeps_its_jsons(tmp_path):
    stem = "20240101_ABC0101_P1"
    pose = tmp_path / "model"
    pose.mkdir()
    h5 = pose / (stem + "DLC_newmodel.h5")
    h5.write_text("x", encoding="utf-8")
    seg = pose / (stem + "_segments.json")
    seg.write_text("{}", encoding="utf-8")
    files = _reprocess_copy_set({pose}, pose, stem)
    assert h5 in files and seg in files                    # all we have
