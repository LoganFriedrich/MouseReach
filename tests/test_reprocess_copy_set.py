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


def _held_layout(tmp_path):
    """A video re-posed while held in a review queue: its results folder exists
    but holds nothing of it; the new pose (and the video) sit in the posed
    folder; the original video is parked in the re-pose queue."""
    stem = "20240102_ABC0102_P2"
    posed, res = tmp_path / "Posed", tmp_path / "results"
    queue = tmp_path / "Repose_Queue"
    parked = queue / "_parked" / stem
    for d in (posed, res, parked):
        d.mkdir(parents=True)
    (res / "20240102_ABC0199_P2.mp4").write_bytes(b"another video")
    h5 = posed / (stem + "DLC_newmodel.h5")
    h5.write_text("x", encoding="utf-8")
    (posed / (stem + "_segments.json")).write_text("{}", encoding="utf-8")
    return stem, posed, res, queue, parked, h5


def test_held_repose_takes_the_video_from_the_posed_folder(tmp_path, monkeypatch):
    """WHY (2026-11-09): a held video re-run from a staged pose had no video --
    the posed folder contributes only DLC files -- and the triage bundle it
    routed had none for the reviewer to open."""
    from mousereach.watcher import orchestrator as orch
    stem, posed, res, queue, parked, h5 = _held_layout(tmp_path)
    monkeypatch.setattr(orch.Paths, "REPOSE_QUEUE", queue, raising=False)
    video = posed / (stem + ".mp4")
    video.write_bytes(b"v")
    (parked / (stem + ".mp4")).write_bytes(b"v")
    files = _reprocess_copy_set({posed, res}, posed, stem)
    assert video in files and h5 in files
    assert posed / (stem + "_segments.json") not in files     # filter still holds
    assert sum(f.name == stem + ".mp4" for f in files) == 1


def test_held_repose_falls_back_to_the_parked_video(tmp_path, monkeypatch):
    from mousereach.watcher import orchestrator as orch
    stem, posed, res, queue, parked, h5 = _held_layout(tmp_path)
    monkeypatch.setattr(orch.Paths, "REPOSE_QUEUE", queue, raising=False)
    video = parked / (stem + ".mp4")
    video.write_bytes(b"v")
    files = _reprocess_copy_set({posed, res}, posed, stem)
    assert video in files and h5 in files


def test_results_folder_video_wins(tmp_path, monkeypatch):
    """A filed video's own mp4 is used; nothing else is added beside it."""
    from mousereach.watcher import orchestrator as orch
    stem, posed, res, queue, parked, h5 = _held_layout(tmp_path)
    monkeypatch.setattr(orch.Paths, "REPOSE_QUEUE", queue, raising=False)
    filed = res / (stem + ".mp4")
    filed.write_bytes(b"v")
    (posed / (stem + ".mp4")).write_bytes(b"v")
    files = _reprocess_copy_set({posed, res}, posed, stem)
    assert [f for f in files if f.name == stem + ".mp4"] == [filed]
