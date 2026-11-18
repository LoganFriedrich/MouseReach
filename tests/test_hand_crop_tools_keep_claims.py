"""Hand crop/prep tools hand videos to the pose step only through the claim,
archive only finished collages, and move a bundle whole or not at all.

WHY:
  * ``mousereach-crop --queue``, ``mousereach-prep`` and the crop tab's default
    "Copy outputs to DLC Queue" copied EVERY mp4 of the singles folder straight
    into this machine's DLC queue, with no claim. The same videos still waited
    unclaimed in the shared folder, so another machine could claim and pose
    them too, and a video already posed, filed or held was queued again.
  * ``mousereach-prep --archive`` moved every collage after cropping, whatever
    its children's state.
  * the hand "move the bundle" helpers skipped a file whose name was taken at
    the destination and moved the rest, splitting a video between folders.
Video and animal ids are invented.
"""
import sys

import pytest


@pytest.fixture
def folders(tmp_path, monkeypatch):
    from mousereach.config import Paths
    roots = {
        "SINGLE_ANIMAL_OUTPUT": tmp_path / "nas" / "Unanalyzed" / "Single_Animal",
        "MULTI_ANIMAL_SOURCE": tmp_path / "nas" / "Unanalyzed" / "Multi-Animal",
        "DLC_QUEUE": tmp_path / "local" / "DLC_Queue",
        "ANALYZED_OUTPUT": tmp_path / "nas" / "Analyzed",
        "DLC_STAGING": tmp_path / "nas" / "Processing" / "Posed",
        "PROCESSING": tmp_path / "local" / "Processing",
        "TRIAGE_REVIEW": tmp_path / "nas" / "Review" / "triage",
        "DEEP_REVIEW": tmp_path / "nas" / "Review" / "deep_review",
        "NAS_ROOT": tmp_path / "nas",
    }
    for k, v in roots.items():
        if v is not None:
            v.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, k, v, raising=False)
    return roots


def test_queue_step_never_copies_the_singles_folder_into_a_dlc_queue(folders):
    from mousereach.video_prep.core import copy_to_dlc_queue
    door = folders["SINGLE_ANIMAL_OUTPUT"]
    (door / "20200101_ABC9901_P1.mp4").write_bytes(b"cut")
    (door / "20200101_ABC9902_P1.mp4").write_bytes(b"cut")
    copy_to_dlc_queue(door, folders["DLC_QUEUE"], verbose=False)
    assert list(folders["DLC_QUEUE"].iterdir()) == []
    # Still waiting where every machine's watcher claims from.
    assert (door / "20200101_ABC9901_P1.mp4").is_file()


def test_crop_queue_hands_only_what_this_run_cut_to_the_shared_folder(folders, tmp_path,
                                                                      monkeypatch):
    from mousereach.video_prep import cli
    import mousereach.video_prep.core as core
    out = tmp_path / "cut_here"
    out.mkdir()
    (out / "20200101_ABC9909_P1.mp4").write_bytes(b"an older cut, not this run's")
    collage = tmp_path / "20200101_ABC9901,ABC0001,ABC0002,ABC0003,ABC0004,ABC0005,ABC0006,ABC0007_P1.mkv"
    collage.write_bytes(b"collage")

    def fake_crop(path, output_dir, verbose=True):
        made = output_dir / "20200101_ABC9901_P1.mp4"
        made.write_bytes(b"cut")
        return [{"position": 1, "status": "success", "output_path": str(made)}]

    monkeypatch.setattr(core, "crop_collage", fake_crop)
    monkeypatch.setattr(sys, "argv", ["mousereach-crop", "-i", str(collage), "-o", str(out),
                                      "--queue", "-q"])
    cli.main_crop()
    door = folders["SINGLE_ANIMAL_OUTPUT"]
    assert list(folders["DLC_QUEUE"].iterdir()) == []
    assert (door / "20200101_ABC9901_P1.mp4").is_file()
    assert not (door / "20200101_ABC9909_P1.mp4").exists()


def test_a_video_a_machine_holds_for_pose_is_not_handed_again(folders, tmp_path):
    from mousereach.video_prep.core import hand_to_pose
    door = folders["SINGLE_ANIMAL_OUTPUT"]
    held = door / ".inflight" / "OTHERHOST" / "20200101_ABC9901_P1.mp4"
    held.parent.mkdir(parents=True)
    held.write_bytes(b"held")
    out = tmp_path / "cut_here"
    out.mkdir()
    (out / "20200101_ABC9901_P1.mp4").write_bytes(b"cut again")
    res = hand_to_pose([out / "20200101_ABC9901_P1.mp4"], verbose=False)
    assert res["not_handed"] == 1
    assert not (door / "20200101_ABC9901_P1.mp4").exists()


def test_prep_archive_keeps_a_collage_whose_children_are_not_done(folders):
    from mousereach.video_prep.core import archive_collages
    src = folders["MULTI_ANIMAL_SOURCE"]
    name = "20200101_ABC9901,ABC9902,ABC0001,ABC0002,ABC0003,ABC0004,ABC0005,ABC0006_P1.mkv"
    (src / name).write_bytes(b"collage")
    # One child cut and waiting for its pose: the collage is not done.
    (folders["SINGLE_ANIMAL_OUTPUT"] / "20200101_ABC9901_P1.mp4").write_bytes(b"cut")
    dest = folders["ANALYZED_OUTPUT"] / "Multi-Animal"
    moved = archive_collages(src, dest, verbose=False)
    assert moved == 0
    assert (src / name).is_file()
    assert not (dest / name).exists()


def _bundle(folder, vid):
    folder.mkdir(parents=True, exist_ok=True)
    files = {"video_id": vid}
    for key, name in (("video", f"{vid}.mp4"), ("segments", f"{vid}_segments.json"),
                      ("dlc_h5", f"{vid}DLC_resnet101_X_shuffle3_100000.h5")):
        p = folder / name
        p.write_text(name)
        files[key] = p
    return files


@pytest.mark.parametrize("module", ["advance", "triage"])
def test_a_bundle_is_moved_whole_or_not_at_all(tmp_path, module, monkeypatch):
    import importlib
    import mousereach.index as index_mod

    class NoIndex:                     # the pipeline index is never written here
        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(index_mod, "PipelineIndex", NoIndex)
    mod = importlib.import_module(f"mousereach.segmentation.core.{module}")
    vid = "20200101_ABC9901_P1"
    src = tmp_path / "src"
    files = _bundle(src, vid)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / f"{vid}_segments.json").write_text("another copy")
    kwargs = {"source_stage": "src"}
    if module == "triage":
        kwargs["update_index"] = False
    moved = mod.move_file_bundle(files, dest, tmp_path / "log.txt", **kwargs)
    assert moved == []
    for key in ("video", "segments", "dlc_h5"):
        assert files[key].is_file(), key          # nothing left the source
    assert not (dest / f"{vid}.mp4").exists()
    assert (dest / f"{vid}_segments.json").read_text() == "another copy"


def test_step0_quick_guide_describes_the_checkbox_the_tab_has():
    """The '?' guide must name the crop tab's real checkbox and say where the cut
    videos go now: the shared singles folder, never the DLC queue."""
    import re
    from pathlib import Path
    import mousereach.video_prep as vp
    from mousereach.ui.utils import STEP0_HELP
    widget_src = (Path(vp.__file__).parent / "widget.py").read_text(encoding="utf-8")
    label = re.search(r'copy_to_queue_cb = QCheckBox\("([^"]+)"\)', widget_src).group(1)
    assert label in STEP0_HELP
    assert "Copy to DLC Queue" not in STEP0_HELP
    assert "shared singles folder" in STEP0_HELP
    assert "Run the pipeline now" in STEP0_HELP
