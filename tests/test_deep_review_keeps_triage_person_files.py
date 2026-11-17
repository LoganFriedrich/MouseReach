"""Deep review beats triage without losing what a person wrote in the triage copy.

When a video moves into deep review from somewhere other than its triage
bundle (a route from the filed results, a re-run the gate sends to deep
review), its triage bundle used to be deleted whole -- a ground-truth file or
a clear marker in it with no copy anywhere. And when the gate could not route a
run to triage because the video is in deep review, every non-input file of the
run was deleted, a person's files included. Now what has a current copy
elsewhere is deleted and a person's file stays where it is, listed for a
person; nothing goes to a set-aside folder. Built under tmp_path with invented
ids; no real folder or database.
"""
import pytest

from mousereach.config import Paths
from mousereach.watcher import review_gate as rg
from mousereach.watcher.attention import read_items

# Named here, not imported, so these tests also run (and fail by what they
# assert) on code without the leftover rule.
LEFT_NOTE = "LEFT_FOR_A_PERSON.json"      # review_routing.LEFT_NOTE

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


def _listed(tmp_path, kind="left_for_a_person"):
    return [i["video_id"] for i in read_items(tmp_path) if i.get("kind") == kind]


def _nothing_set_aside(tmp_path):
    assert not (tmp_path / "Review" / "_Problematic").exists()


def _in_store(tmp_path, vid, text):
    """The durable review store already holds this review (the review window
    saves it there as well as in the bundle)."""
    d = tmp_path / "review_records" / "reviews"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{vid}_causal_review.json").write_text(text, encoding="utf-8")


def _write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def test_answers_in_the_triage_copy_are_kept_as_a_record_not_carried_into_deep_review(
        queues):
    """Re-segmentation voids every answer after it. Neither the triage copy's
    review nor its ground truth is moved into the deep-review bundle, where it
    would travel back into Processing after the re-cut and be applied to the
    new cuts; and the review is not written into the durable store either
    (the readers match the store's copy to new cuts by frame span). Both stay
    where they are, listed for a person."""
    tmp_path, triage, deep = queues
    copy = triage / VID
    _write(copy, f"{VID}.mp4", "video")
    _write(copy, f"{VID}_reaches.json", '{"run": "old"}')
    _write(copy, f"{VID}_unified_ground_truth.json", '{"by": "person-a"}')
    _write(copy, f"{VID}_causal_review.json", '{"reviewer": "person-b"}')
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ01"
    _write(filed, f"{VID}.mp4", "video")
    _write(filed, f"{VID}_segments.json", '{"boundary_source": "algo"}')

    rg.route_to_queue(VID, filed, deep, "segmentation wrong", db=None)

    bundle = deep / VID
    assert not (bundle / f"{VID}_unified_ground_truth.json").exists()
    assert not (bundle / f"{VID}_causal_review.json").exists()
    durable = tmp_path / "review_records" / "reviews" / f"{VID}_causal_review.json"
    assert not durable.exists()                     # nothing written into the store
    # Left where it is: the ground truth and the review (no copy anywhere
    # else) and the note; the machine output and the video (the deep-review
    # bundle holds it byte for byte) are deleted.
    assert {p.name for p in copy.iterdir()} == {
        LEFT_NOTE, f"{VID}_unified_ground_truth.json", f"{VID}_causal_review.json"}
    assert (copy / f"{VID}_unified_ground_truth.json").read_text(
        encoding="utf-8") == '{"by": "person-a"}'
    assert _listed(tmp_path) == [VID]
    _nothing_set_aside(tmp_path)


def test_a_review_the_store_already_holds_leaves_no_copy_in_triage(queues):
    tmp_path, triage, deep = queues
    copy = triage / VID
    _write(copy, f"{VID}_reaches.json", '{"run": "old"}')
    _write(copy, f"{VID}_causal_review.json", '{"reviewer": "person-b"}')
    _in_store(tmp_path, VID, '{"reviewer": "person-b"}')
    bundle = deep / VID
    _write(bundle, f"{VID}.mp4", "video")
    rg._drop_triage_copy(VID, bundle)
    assert not copy.exists()
    assert not (bundle / f"{VID}_causal_review.json").exists()
    durable = tmp_path / "review_records" / "reviews" / f"{VID}_causal_review.json"
    assert durable.read_text(encoding="utf-8") == '{"reviewer": "person-b"}'


def test_a_review_the_store_holds_differently_stays_and_the_store_is_untouched(queues):
    tmp_path, triage, deep = queues
    copy = triage / VID
    _write(copy, f"{VID}_causal_review.json", '{"reviewer": "person-b", "round": 1}')
    _in_store(tmp_path, VID, '{"reviewer": "person-b", "round": 2}')
    bundle = deep / VID
    _write(bundle, f"{VID}.mp4", "video")
    rg._drop_triage_copy(VID, bundle)
    assert (copy / f"{VID}_causal_review.json").read_text(
        encoding="utf-8") == '{"reviewer": "person-b", "round": 1}'
    durable = tmp_path / "review_records" / "reviews" / f"{VID}_causal_review.json"
    assert durable.read_text(encoding="utf-8") == '{"reviewer": "person-b", "round": 2}'


def test_a_clear_marker_in_the_triage_copy_is_kept_for_a_person(queues):
    tmp_path, triage, deep = queues
    copy = triage / VID
    _write(copy, f"{VID}_deep_review_cleared.json", '{"cleared_by": "person-a"}')
    _write(copy, f"{VID}_reaches.json", '{"run": "old"}')
    src = tmp_path / "Processing"
    _write(src, f"{VID}.mp4", "video")

    rg.route_to_queue(VID, src, deep, "segmentation_failed", db=None)

    # Not moved into deep review (it would release the video at once), and not
    # deleted: it stays where it is, with a note, listed in the attention
    # panel. The machine output beside it is deleted.
    assert (copy / f"{VID}_deep_review_cleared.json").is_file()
    assert (copy / LEFT_NOTE).is_file()
    assert not (copy / f"{VID}_reaches.json").exists()
    assert not (deep / VID / f"{VID}_deep_review_cleared.json").exists()
    assert _listed(tmp_path) == [VID]
    _nothing_set_aside(tmp_path)


def test_a_refused_triage_run_leaves_a_persons_file_in_place_and_lists_it(queues):
    tmp_path, triage, deep = queues
    _write(deep / VID, f"{VID}.mp4", "video")
    proc = tmp_path / "proc_root" / "work"
    _write(proc, f"{VID}_reaches.json", '{"run": "this"}')
    _write(proc, f"{VID}_unified_ground_truth.json", '{"by": "person-a"}')

    out = rg.discard_run_outputs(VID, proc, deep / VID)

    assert f"{VID}_reaches.json" in out["deleted"]
    assert out["kept"] == [f"{VID}_unified_ground_truth.json"]
    assert (proc / f"{VID}_unified_ground_truth.json").read_text(
        encoding="utf-8") == '{"by": "person-a"}'
    assert _listed(tmp_path, "person_file_left") == [VID]


def test_a_video_only_in_the_triage_copy_follows_the_video_into_deep_review(queues):
    """The triage copy holds the only .mp4 and the deep-review bundle has
    none: the .mp4 is moved into the deep-review bundle, not filed away."""
    tmp_path, triage, deep = queues
    copy = triage / VID
    copy.mkdir()
    (copy / f"{VID}.mp4").write_text("the only video", encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}_segments.json").write_text('{"cuts": "algo"}', encoding="utf-8")
    rg.route_to_queue(VID, src, deep, "seg failed")
    assert (deep / VID / f"{VID}.mp4").read_text(encoding="utf-8") == "the only video"
    assert not copy.exists()


def test_a_differing_video_in_the_triage_copy_stays_and_is_listed(queues):
    tmp_path, triage, deep = queues
    copy = triage / VID
    copy.mkdir()
    (copy / f"{VID}.mp4").write_text("another video", encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}.mp4").write_text("this video", encoding="utf-8")
    rg.route_to_queue(VID, src, deep, "seg failed")
    assert (copy / f"{VID}.mp4").read_text(encoding="utf-8") == "another video"
    assert (copy / LEFT_NOTE).is_file()
    assert _listed(tmp_path) == [VID]
    _nothing_set_aside(tmp_path)


def test_the_triage_copy_is_left_alone_while_another_process_holds_the_claim(queues):
    """Emptying the triage copy is several moves and deletes; the return scan
    of another machine acts on the same folder under the bundle claim. While
    someone else holds the claim, the copy is not touched at all (the scan
    deals with it on its next pass)."""
    import threading
    from mousereach.watcher import review_return as rr
    tmp_path, triage, deep = queues
    copy = triage / VID
    _write(copy, f"{VID}_reaches.json", '{"run": "old"}')
    _write(copy, f"{VID}_causal_review.json", '{"reviewer": "person-b"}')
    _in_store(tmp_path, VID, '{"reviewer": "person-b"}')
    bundle = deep / VID
    _write(bundle, f"{VID}.mp4", "video")
    _write(bundle, f"{VID}_segments.json", '{"boundary_source": "algo"}')
    held, done = threading.Event(), threading.Event()

    def holder():
        assert rr._claim_return(VID)
        held.set()
        done.wait(30)
        rr._release_return_claim(VID)

    t = threading.Thread(target=holder)
    t.start()
    try:
        held.wait(30)
        rg._drop_triage_copy(VID, bundle)
        assert sorted(p.name for p in copy.iterdir()) == [
            f"{VID}_causal_review.json", f"{VID}_reaches.json"]
        assert not (bundle / f"{VID}_causal_review.json").exists()
    finally:
        done.set()
        t.join()
    # Once the claim is free, the copy is emptied as usual (the review's
    # record is in the durable store; it never goes into the deep-review
    # bundle).
    rg._drop_triage_copy(VID, bundle)
    assert not copy.exists()
    assert not (bundle / f"{VID}_causal_review.json").exists()
    assert (tmp_path / "review_records" / "reviews" / f"{VID}_causal_review.json").is_file()
