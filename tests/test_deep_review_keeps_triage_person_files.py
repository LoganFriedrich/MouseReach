"""Deep review beats triage without losing what a person wrote in the triage copy.

When a video moves into deep review from somewhere other than its triage
bundle (a route from the filed results, a re-run the gate sends to deep
review), its triage bundle used to be deleted whole -- a ground-truth file or
a clear marker in it with no copy anywhere. And when the gate could not route a
run to triage because the video is in deep review, every non-input file of the
run was deleted, a person's files included. Built under tmp_path with invented
ids; no real folder or database.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher import review_gate as rg

VID = "20990101_ABC9901_P1"


@pytest.fixture
def queues(tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path / "proc_root", raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    return tmp_path, triage, deep


def _write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def test_ground_truth_in_the_triage_copy_follows_the_video_into_deep_review(queues):
    tmp_path, triage, deep = queues
    copy = triage / VID
    _write(copy, f"{VID}.mp4", "video")
    _write(copy, f"{VID}_reaches.json", "old run")
    _write(copy, f"{VID}_unified_ground_truth.json", '{"by": "person-a"}')
    _write(copy, f"{VID}_causal_review.json", '{"reviewer": "person-b"}')
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ01"
    _write(filed, f"{VID}.mp4", "video")
    _write(filed, f"{VID}_segments.json", '{"boundary_source": "algo"}')

    rg.route_to_queue(VID, filed, deep, "segmentation wrong", db=None)

    bundle = deep / VID
    assert (bundle / f"{VID}_unified_ground_truth.json").read_text(
        encoding="utf-8") == '{"by": "person-a"}'
    assert (bundle / f"{VID}_causal_review.json").read_text(
        encoding="utf-8") == '{"reviewer": "person-b"}'
    assert (tmp_path / "review_records" / "reviews" / f"{VID}_causal_review.json").is_file()
    assert not copy.exists()            # nothing left that a person needs


def test_a_clear_marker_in_the_triage_copy_is_kept_for_a_person(queues):
    tmp_path, triage, deep = queues
    copy = triage / VID
    _write(copy, f"{VID}_deep_review_cleared.json", '{"cleared_by": "person-a"}')
    _write(copy, f"{VID}_reaches.json", "old run")
    src = tmp_path / "Processing"
    _write(src, f"{VID}.mp4", "video")

    rg.route_to_queue(VID, src, deep, "segmentation_failed", db=None)

    # Not moved into deep review (it would release the video at once), and not
    # deleted: it leaves the triage queue (a video is in one place) and is
    # kept for a person, listed in the attention panel.
    kept = tmp_path / "Review" / "_Problematic" / "_kept_for_a_person" / VID
    assert not copy.exists()
    assert (kept / f"{VID}_deep_review_cleared.json").is_file()
    assert not (kept / f"{VID}_reaches.json").exists()       # machine output removed
    assert not (deep / VID / f"{VID}_deep_review_cleared.json").exists()
    items = json.loads((tmp_path / "attention" / "kept_for_a_person.json")
                       .read_text(encoding="utf-8"))["items"]
    assert [i["video_id"] for i in items] == [VID]


def test_a_refused_triage_run_sets_a_persons_file_aside(queues):
    tmp_path, triage, deep = queues
    _write(deep / VID, f"{VID}.mp4", "video")
    proc = tmp_path / "proc_root" / "work"
    _write(proc, f"{VID}_reaches.json", "this run")
    _write(proc, f"{VID}_unified_ground_truth.json", '{"by": "person-a"}')

    out = rg.discard_run_outputs(VID, proc, deep / VID)

    assert f"{VID}_reaches.json" in out["deleted"]
    assert f"{VID}_unified_ground_truth.json" not in out["deleted"]
    kept = [p for p in (tmp_path / "proc_root").rglob(f"{VID}_unified_ground_truth.json")]
    assert kept and kept[0].read_text(encoding="utf-8") == '{"by": "person-a"}'


def test_a_video_only_in_the_triage_copy_follows_the_video_into_deep_review(queues):
    """The triage copy holds the only .mp4 and the deep-review bundle has
    none: the .mp4 is moved into the deep-review bundle, not filed away."""
    tmp_path, triage, deep = queues
    copy = triage / VID
    copy.mkdir()
    (copy / f"{VID}.mp4").write_text("the only video", encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}_segments.json").write_text("cuts", encoding="utf-8")
    rg.route_to_queue(VID, src, deep, "seg failed")
    assert (deep / VID / f"{VID}.mp4").read_text(encoding="utf-8") == "the only video"
    assert not copy.exists()


def test_a_differing_video_in_the_triage_copy_is_kept_for_a_person(queues):
    tmp_path, triage, deep = queues
    copy = triage / VID
    copy.mkdir()
    (copy / f"{VID}.mp4").write_text("another video", encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}.mp4").write_text("this video", encoding="utf-8")
    rg.route_to_queue(VID, src, deep, "seg failed")
    assert not copy.exists()
    kept = tmp_path / "Review" / "_Problematic" / "_kept_for_a_person" / VID
    assert (kept / f"{VID}.mp4").read_text(encoding="utf-8") == "another video"
    listed = json.loads((tmp_path / "attention" / "kept_for_a_person.json")
                        .read_text(encoding="utf-8"))["items"]
    assert [i["video_id"] for i in listed] == [VID]
