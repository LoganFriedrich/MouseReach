"""An answered pellet can be asked again on purpose ("are you sure?").

WHY (Logan, 2026-10-26): answers of "retrieved" where the frames show the
pellet lying in the scoring area afterwards -- "toss these cases back at the
review queue and be like - are you sure". The answer is kept, the pellet is
open again until a person answers it, and without --recheck an answered
pellet is still never sent back.
"""
import json

import mousereach.watcher.route_cli as rc
from mousereach.config import Paths
from mousereach.review.triage_status import resolved_segments

VID = "20250715_ABC0212_P4"


def _bundle(tmp_path):
    a = tmp_path / "Analyzed" / "Connectome" / "ABC02"
    a.mkdir(parents=True)
    segs = [{"segment_num": n, "start_frame": 100 * n, "end_frame": 100 * n + 90} for n in (1, 2)]
    (a / f"{VID}_segments.json").write_text(json.dumps({"segments": segs, "boundaries": [100, 190, 200, 290]}))
    (a / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "displaced_sa"}, {"segment_num": 2, "outcome": "untouched"}]}))
    review = {"video_stem": VID, "reviewer": "someone", "reviewed_at": "2026-07-24T16:25:55",
              "segments": [
                  {"segment_num": 1, "pellet_num": 1, "segment_span": {"start": 100, "end": 190},
                   "human": {"outcome": "retrieved", "causal_reach": {"start": 120, "end": 130}},
                   "answers": {"triage_kind": "both_uncertain", "outcome_pick": "retrieved"}},
                  {"segment_num": 2, "pellet_num": 2, "segment_span": {"start": 200, "end": 290},
                   "human": {"outcome": "untouched"}, "answers": {"outcome_pick": "untouched"}}]}
    (a / f"{VID}_causal_review.json").write_text(json.dumps(review))
    return a


def test_recheck_reopens_the_answer_keeps_it_and_backs_up_the_file(tmp_path, monkeypatch):
    a = _bundle(tmp_path)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    q = {1: "are you sure it was retrieved? the pellet seems to lie left of the pillar"}
    assert rc.reopen_answered(VID, a, [1], q) == [1]
    review = json.loads((a / f"{VID}_causal_review.json").read_text())
    rec = review["segments"][0]
    assert rec["answers"]["reviewed"] is False
    assert rec["answers"]["recheck"]["question"] == q[1]
    assert rec["answers"]["recheck"]["previous"]["human"]["outcome"] == "retrieved"
    assert resolved_segments(review) == {2}                 # 1 is open again
    backups = list((tmp_path / "Analyzed" / "Archive" / "manual_corrections").glob("*before_recheck*"))
    assert len(backups) == 1
    # asking again about a pellet already reopened changes nothing
    assert rc.reopen_answered(VID, a, [1], q) == []


def test_route_without_recheck_never_sends_an_answered_pellet_back(tmp_path, monkeypatch):
    _bundle(tmp_path)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    routed = []
    import mousereach.watcher.review_gate as rg
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    r = rc.route_video(VID, "triage", "test", [1])
    assert r["answered"] == [1] and not r["routed"] and not routed
    r = rc.route_video(VID, "triage", "test", [1], recheck=True,
                       segment_details={1: "are you sure?"})
    assert r["reopened"] == [1] and r["flagged"] == [1] and r["routed"]
    seg = json.loads((tmp_path / "Analyzed" / "Connectome" / "ABC02" /
                      f"{VID}_pellet_outcomes.json").read_text())["segments"][0]
    assert seg["flagged_for_review"] and seg["flag_reason"] == "test: are you sure?"
