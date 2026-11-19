"""The release button acts on the queue as it is when pressed, not on its list.

The Review Queues tab classifies the re-segmentation queue when it refreshes
and releases on that list after the person confirms -- which can be minutes
later. Releasing on the old list cleared videos that had since been sent
back, and re-created a bundle that had left the queue as a one-file folder
holding only a clear marker.
"""
import json

from mousereach.review.release_cli import classify_queue, release_finished

STEM = "20990101_ABC9901_P1"
MARKER = f"{STEM}_deep_review_cleared.json"


def _finished_bundle(root):
    b = root / STEM
    b.mkdir(parents=True)
    (b / f"{STEM}_routing.json").write_text(json.dumps(
        {"routed_at": "2099-01-10T09:00:00",
         "routed_reason": "segmentation: cuts in question"}), encoding="utf-8")
    (b / f"{STEM}_segments.json").write_text(json.dumps(
        {"boundaries": [100, 200, 300], "boundary_source": "human",
         "corrected_at": "2099-01-11T09:00:00",
         # every cut walked (hand_cuts.BOUNDARY_CALLS_KEY, MRR-08d)
         "boundary_calls": [{"frame": f, "action": "accepted"} for f in (100, 200, 300)]}),
        encoding="utf-8")
    return b


def test_a_bundle_that_left_the_queue_is_not_recreated(tmp_path):
    b = _finished_bundle(tmp_path)
    cls = classify_queue(tmp_path)
    assert [s for s, _ in cls["fixed_release"]] == [STEM]
    for f in b.iterdir():
        f.unlink()
    b.rmdir()
    n, failures = release_finished(tmp_path, cls)
    assert (n, failures) == (0, [])
    assert not b.exists()


def test_a_bundle_sent_back_since_the_list_was_made_is_not_released(tmp_path):
    b = _finished_bundle(tmp_path)
    cls = classify_queue(tmp_path)
    # Sent back to the queue after its cuts were saved: those cuts no longer
    # count, so it is not finished any more.
    (b / f"{STEM}_routing.json").write_text(json.dumps(
        {"routed_at": "2099-02-01T09:00:00",
         "routed_reason": "reviewer escalated from triage: bad segmentation"}),
        encoding="utf-8")
    n, _ = release_finished(tmp_path, cls)
    assert n == 0
    assert not (b / MARKER).exists()


def test_a_marker_already_there_is_kept(tmp_path):
    b = _finished_bundle(tmp_path)
    cls = classify_queue(tmp_path)
    (b / MARKER).write_text(json.dumps({"cleared_by": "someone else"}), encoding="utf-8")
    n, failures = release_finished(tmp_path, cls)
    assert (n, failures) == (0, [])
    assert json.loads((b / MARKER).read_text(encoding="utf-8")) == {"cleared_by": "someone else"}


def test_a_still_finished_bundle_is_released(tmp_path):
    b = _finished_bundle(tmp_path)
    n, failures = release_finished(tmp_path, classify_queue(tmp_path))
    assert (n, failures) == (1, [])
    doc = json.loads((b / MARKER).read_text(encoding="utf-8"))
    assert doc["type"] == "deep_review_cleared" and doc["video_stem"] == STEM
