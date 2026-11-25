"""A bundle with no readable routing time is still dated.

With no ``routed_at`` the re-segmentation rule used to accept ANY hand cut
ever saved as "since the video arrived" -- including the very cuts it was
sent back to have fixed. Now the arrival is the latest of the routing
record's modified time, the review manifest's staged_at and the bundle
folder's creation time.
"""
import json
import os
import time
from datetime import datetime

from mousereach.review import resegmented
from mousereach.review.resegmented import entered_deep_review_at, resegmented_since_entry

STEM = "20990101_ABC9901_P1"
OLD_CUT = "2000-01-01T09:00:00"


def _bundle(tmp_path, seg_extra=None, routing=None, manifest=None):
    b = tmp_path / STEM
    b.mkdir(parents=True)
    seg = {"boundaries": [100, 200, 300]}
    seg.update(seg_extra or {})
    # A fixer save records every cut as a person's explicit call
    # (hand_cuts.BOUNDARY_CALLS_KEY).
    if seg.get("boundary_source") == "human":
        seg.setdefault("boundary_calls", [{"frame": f, "action": "accepted"}
                                          for f in seg["boundaries"]])
    (b / f"{STEM}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    if routing is not None:
        (b / f"{STEM}_routing.json").write_text(json.dumps(routing), encoding="utf-8")
    if manifest is not None:
        (b / f"{STEM}_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return b


def test_old_hand_cuts_do_not_count_without_a_routing_record(tmp_path):
    b = _bundle(tmp_path, {"boundary_source": "human", "corrected_at": OLD_CUT})
    ok, why = resegmented_since_entry(b, STEM)
    assert not ok and "before it was sent" in why


def test_routing_record_without_routed_at_is_dated_by_its_file(tmp_path):
    b = _bundle(tmp_path, {"boundary_source": "human", "corrected_at": OLD_CUT},
                routing={"routed_reason": "segmentation: cuts in question"})
    assert not resegmented_since_entry(b, STEM)[0]


def test_manifest_staged_at_dates_the_arrival(tmp_path, monkeypatch):
    staged = "2099-06-01T09:00:00"
    monkeypatch.setattr(resegmented, "_created", lambda p: None, raising=False)
    b = _bundle(tmp_path, {"boundary_source": "human",
                           "corrected_at": "2099-05-01T09:00:00"},
                manifest={"provenance": {"staged_at": staged}})
    assert entered_deep_review_at(b, STEM) == datetime.fromisoformat(staged).isoformat()
    assert not resegmented_since_entry(b, STEM)[0]
    later = _bundle(tmp_path / "x", {"boundary_source": "human",
                                     "corrected_at": "2099-07-01T09:00:00"},
                    manifest={"provenance": {"staged_at": staged}})
    assert resegmented_since_entry(later, STEM)[0]


def test_cuts_saved_after_the_bundle_arrived_count(tmp_path):
    b = _bundle(tmp_path, {"boundary_source": "human",
                           "corrected_at": datetime.fromtimestamp(time.time() + 60).isoformat()})
    assert resegmented_since_entry(b, STEM)[0]


def test_an_undatable_bundle_cannot_leave_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(resegmented, "_created", lambda p: None, raising=False)
    b = _bundle(tmp_path, {"boundary_source": "human", "corrected_at": OLD_CUT})
    ok, why = resegmented_since_entry(b, STEM)
    assert not ok and "cannot be told" in why and "segmentation fixer" in why


def test_readable_routed_at_still_decides(tmp_path):
    b = _bundle(tmp_path, {"boundary_source": "human", "corrected_at": "2099-01-11T09:00:00"},
                routing={"routed_at": "2099-01-10T09:00:00"})
    os.utime(b / f"{STEM}_routing.json", (time.time(), time.time()))
    assert resegmented_since_entry(b, STEM)[0]
