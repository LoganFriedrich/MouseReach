"""When a person said a segment is another pellet and the cuts still disagree,
the video goes to manual segmentation -- it is not held in Processing.

A problem routes the video to the review queue of the step that has it. Here
the problem is in the cuts, so the step is manual segmentation (the
re-segmentation queue). Held in Processing, the video waited for the next
scores file, where no review tool could reach it and nothing could settle it.
"""
import json

import mousereach.review.causal_review_io as cio
import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.config import Paths
from mousereach.review.resegmented import needs_recut
from mousereach.watcher.db import WatcherDB

VID = "20990101_ABC9901_P1"


def _site(tmp_path, monkeypatch, seg_extra=None):
    f = tmp_path / "hand.csv"
    f.write_text("video_key,pellet,score\n" + "".join(
        "%s,%d,0\n" % (VID, p) for p in (1, 2, 3)), encoding="utf-8")
    f.with_suffix(".json").write_text(json.dumps({"last_successful_import": {}}),
                                      encoding="utf-8")
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": []})
    hs._CACHE.update(key=None)
    monkeypatch.setattr(cio, "durable_review_dir", lambda: None)
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())
    for attr, sub in (("NAS_ROOT", "nas"), ("DEEP_REVIEW", "Review/deep_review"),
                      ("TRIAGE_REVIEW", "Review/triage")):
        monkeypatch.setattr(Paths, attr, tmp_path / sub, raising=False)
    (tmp_path / "Review" / "deep_review").mkdir(parents=True)
    folder = tmp_path / "work"
    folder.mkdir()
    seg = {"boundaries": [100, 200, 300, 400]}
    seg.update(seg_extra or {})
    (folder / f"{VID}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    (folder / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "untouched"} for n in (1, 2, 3)]}), encoding="utf-8")
    # A person said segment 2 (frames 200-299) is really pellet 3.
    (folder / f"{VID}_causal_review.json").write_text(json.dumps({"segments": [
        {"segment_num": 2, "true_segment_num": 3, "segmentation_wrong": True,
         "segment_span": {"start": 200, "end": 299}, "human": {"outcome": None},
         "answers": {"reviewed": False}, "reviewed_at": "2099-01-01T10:00:00"}]}),
        encoding="utf-8")
    (folder / f"{VID}.mp4").write_bytes(b"x")
    return folder


def _orch(tmp_path):
    import mousereach.watcher.orchestrator as o
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processing", reason="test fixture")
    orch.hostname = "HOST"
    return orch


def test_the_video_is_sent_to_manual_segmentation_with_the_reason(tmp_path, monkeypatch):
    folder = _site(tmp_path, monkeypatch)
    orch = _orch(tmp_path)
    assert orch._hand_score_precheck(VID, folder) is False      # this run stops
    bundle = Paths.DEEP_REVIEW / VID
    assert (bundle / f"{VID}_segments.json").is_file()
    assert not (folder / f"{VID}_segments.json").exists()       # one place at a time
    routing = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert routing["routed_reason"].startswith("segmentation: a reviewer said segment(s) 2")
    assert orch.db.get_video(VID)["state"] == "deep_review"
    assert VID not in getattr(orch, "_score_waits", {})          # not held for scores
    # and the segmentation fixer lists it: its cuts were not set since it arrived
    assert needs_recut(bundle, VID)[0]


def test_a_persons_recut_after_the_record_settles_it(tmp_path, monkeypatch):
    folder = _site(tmp_path, monkeypatch, seg_extra={
        "boundary_source": "human", "corrected_at": "2099-01-02T10:00:00"})
    v = hs.check(VID, folder)
    assert v.status == hs.AGREES
