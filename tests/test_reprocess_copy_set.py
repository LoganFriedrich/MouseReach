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


def test_other_videos_files_are_never_statted(tmp_path, monkeypatch):
    """WHY (2026-10-22): a cohort folder holds tens of thousands of files and
    Path.is_file() on every entry cost 175 s per re-run over the share. Only
    this video's files may be asked anything beyond their name."""
    import pathlib
    stem = "20250709_CNT0206_P2"
    res = tmp_path / "CNT02"
    res.mkdir()
    for i in range(50):
        (res / f"20250709_CNT0207_P{i}_segments.json").write_text("{}")
    (res / f"{stem}.mp4").write_bytes(b"v")
    (res / f"{stem}_segments.json").write_text("{}")

    def no_stat(self):
        raise AssertionError("Path.is_file called on %s" % self.name)

    monkeypatch.setattr(pathlib.Path, "is_file", no_stat)
    files = _reprocess_copy_set({res}, res, stem)
    assert sorted(f.name for f in files) == [f"{stem}.mp4", f"{stem}_segments.json"]
