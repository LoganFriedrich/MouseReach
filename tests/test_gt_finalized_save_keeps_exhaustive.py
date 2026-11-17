"""'Save as Ground Truth' keeps the person's "this part is complete"
declarations, and refuses when a part marked complete would lose items.

The finishing save built its copy without the three exhaustive flags and wrote
it over the file 'Save Progress' writes, so the finishing button erased the
declarations: a finished ground truth read as partial, the results kept
reaches the person had ruled out, and pellets came back to triage.

The widget method is driven with a stand-in for the window (no Qt window is
drawn); the file goes to tmp_path. Invented ids.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.review import ground_truth_widget as gtw
from mousereach.review.unified_gt import BoundaryGT, OutcomeGT, ReachGT, UnifiedGroundTruth

VID = "20990101_ABC0101_P1"


def _gt():
    g = UnifiedGroundTruth(video_name=VID)
    g.boundaries = [BoundaryGT(index=i, frame=100 * (i + 1), determined=True) for i in range(3)]
    g.reaches = [ReachGT(reach_id=1, segment_num=1, start_frame=120, end_frame=130,
                         start_determined=True, end_determined=True)]
    g.outcomes = [OutcomeGT(segment_num=1, outcome="retrieved", determined=True),
                  OutcomeGT(segment_num=2, outcome="untouched", determined=True)]
    for part in ("segmentation", "reaches", "outcomes"):
        setattr(g, f"{part}_exhaustive", True)
        setattr(g, f"{part}_exhaustive_determined_by", "annotator")
        setattr(g, f"{part}_exhaustive_determined_at", "2099-01-01T00:00:00")
    return g


@pytest.fixture
def messages(monkeypatch):
    seen = {"info": [], "error": []}
    monkeypatch.setattr(gtw, "show_info", lambda m: seen["info"].append(m))
    monkeypatch.setattr(gtw, "show_error", lambda m: seen["error"].append(m))
    return seen


def _window(tmp_path, gt):
    video = tmp_path / f"{VID}.mp4"
    video.write_bytes(b"x")
    return SimpleNamespace(gt=gt, video_path=video,
                           data_saved=SimpleNamespace(emit=lambda p: None)), \
        tmp_path / f"{VID}_unified_ground_truth.json"


def test_the_finishing_save_keeps_every_complete_declaration(tmp_path, messages):
    win, path = _window(tmp_path, _gt())
    gtw.GroundTruthWidget._save_ground_truth(win)
    doc = json.loads(path.read_text(encoding="utf-8"))
    for part in ("segmentation", "reaches", "outcomes"):
        assert doc[part]["exhaustive"] is True, part
        assert doc[part]["exhaustive_determined_by"] == "annotator", part
        assert doc[part]["exhaustive_determined_at"] == "2099-01-01T00:00:00", part


def test_a_complete_part_with_undetermined_items_is_refused_and_nothing_written(tmp_path, messages):
    g = _gt()
    g.reaches.append(ReachGT(reach_id=2, segment_num=2, start_frame=220, end_frame=230))
    win, path = _window(tmp_path, g)
    gtw.GroundTruthWidget._save_ground_truth(win)
    assert not path.exists()
    assert messages["error"] and "reaches" in messages["error"][0] and "1 of its 2" in messages["error"][0]


def test_without_a_complete_mark_undetermined_items_are_simply_left_out(tmp_path, messages):
    g = _gt()
    g.reaches_exhaustive = False
    g.reaches.append(ReachGT(reach_id=2, segment_num=2, start_frame=220, end_frame=230))
    win, path = _window(tmp_path, g)
    gtw.GroundTruthWidget._save_ground_truth(win)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert [r["reach_id"] for r in doc["reaches"]["reaches"]] == [1]
    assert doc["outcomes"]["exhaustive"] is True


def test_the_rule_is_a_pure_function_too():
    from mousereach.review.unified_gt import finalized_ground_truth
    out, problems = finalized_ground_truth(_gt())
    assert problems == [] and out.outcomes_exhaustive and out.reaches_exhaustive
