"""A Pillar tray's outcome step never runs without the video beside the pose.

WHY: the outcome cascade's video checks (the CV-artifact gate and Stage 98) do
nothing when no video is found, so a displaced_sa call was committed
unchecked -- a different answer from the same pose. The watcher now fetches
the video (a copy) from wherever this node has it, and a video this node does
not have at all is 'unresolvable' (a file that is not here), never analysed
with its checks off.
"""
from test_filed_rerun_skips_checks import VID, rig  # noqa: F401  (fixture)

from mousereach.watcher import orchestrator as o


def test_no_video_anywhere_is_unresolvable_and_not_analysed(rig, monkeypatch):
    (rig.folder / f"{VID}.mp4").unlink()
    monkeypatch.setattr(o, "locate_video_file", lambda *a, **k: None)
    calls = rig.run(None)
    assert "outcome" not in calls and "gate" not in calls, calls
    assert rig.orch.db.get_video(VID)["state"] == "unresolvable"


def test_a_video_elsewhere_on_the_node_is_copied_beside_the_pose(rig, tmp_path, monkeypatch):
    (rig.folder / f"{VID}.mp4").unlink()
    filed = tmp_path / "filed"
    filed.mkdir()
    (filed / f"{VID}.mp4").write_bytes(b"x")
    monkeypatch.setattr(o, "locate_video_file", lambda *a, **k: filed / f"{VID}.mp4")
    calls = rig.run(None)
    assert "outcome" in calls, calls
    assert (rig.folder / f"{VID}.mp4").exists()
    assert (filed / f"{VID}.mp4").exists(), "the filed copy was moved"


def test_the_unresolvable_row_comes_back_by_the_posed_folder_rule(rig, monkeypatch):
    from mousereach.watcher.db import INTAKE_MISS_REASON
    (rig.folder / f"{VID}.mp4").unlink()
    monkeypatch.setattr(o, "locate_video_file", lambda *a, **k: None)
    rig.run(None)
    assert (rig.orch.db.get_video(VID)["error_message"] or "").startswith(INTAKE_MISS_REASON)
