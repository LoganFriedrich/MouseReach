"""A flagged session's other videos are not left out of the review queue index.

Flag Session records a note about the mouse+day and moves only the video on
screen. The queue-index seed still skipped every bundle of a flagged session,
so the session's other unanswered videos were hidden from review with nothing
else holding them (the review window's own copy of this exclusion was removed
earlier). Built under tmp_path; invented ids only.
"""
import json

from mousereach.review import causal_review_io as io
from mousereach.review.queue_index import QueueIndex, seed_from_folder

FLAGGED = "20990101_ABC9901_P1"
SIBLING = "20990101_ABC9901_P2"        # same mouse, same day


def test_the_sibling_of_a_flagged_video_is_seeded(tmp_path):
    pending = tmp_path / "review" / "pending"
    b = pending / SIBLING
    b.mkdir(parents=True)
    io.bundle_manifest_path(b).write_text(
        json.dumps({"video_stem": SIBLING}), encoding="utf-8")
    (b / f"{SIBLING}_pellet_outcomes.json").write_text(json.dumps(
        {"segments": [{"segment_num": 1, "outcome": "triaged"}]}), encoding="utf-8")
    (b / f"{SIBLING}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [100, 400]}), encoding="utf-8")
    io.flag_session(FLAGGED, pending.parent, reason="note")
    assert io.is_session_flagged(SIBLING, pending.parent)

    db = tmp_path / "idx.db"
    out = seed_from_folder(pending, db_path=db)
    assert out["seeded"] == 1
    assert QueueIndex(db).next_path() is not None
