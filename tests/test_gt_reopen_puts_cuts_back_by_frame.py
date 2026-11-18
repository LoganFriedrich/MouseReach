"""Reopening a ground-truth file puts its saved cuts back by FRAME.

A saved boundary with index k was laid over today's boundary k by position.
When today's segmentation has a different number of cuts, the person's
determined cut landed on another cut's row while the cut really there stayed
beside it, and the next save wrote that as the person's determined
segmentation. Now a saved cut goes on the current cut within the segmenter's
merge window (30 frames); a cut a person moved further goes back only into a
stretch whose number of cuts did not change; anything else is kept, shown and
never written onto another row.

Algorithm outputs and the saved file are written under tmp_path; invented ids.
"""
import json

from mousereach.review.unified_gt import (_unified_gt_to_dict, load_or_create_unified_gt,
                                          load_unified_gt)

VID = "20990101_ABC0151_P1"


def _write(folder, name, doc):
    (folder / name).write_text(json.dumps(doc), encoding="utf-8")


def _algo(folder, boundaries):
    _write(folder, f"{VID}_segments.json", {"boundaries": boundaries})
    _write(folder, f"{VID}_reaches.json", {"video_name": VID, "segments": [
        {"segment_num": n, "reaches": []} for n in range(1, len(boundaries))]})
    _write(folder, f"{VID}_pellet_outcomes.json", {"video_name": VID, "segments": [
        {"segment_num": n, "outcome": "triaged"} for n in range(1, len(boundaries))]})


def _saved(folder, boundaries):
    """``boundaries``: [(index, frame, determined)]."""
    _write(folder, f"{VID}_unified_ground_truth.json", {
        "video_name": VID, "schema_version": "2.0",
        "segmentation": {"boundaries": [
            {"index": i, "frame": f, "determined": d,
             "determined_by": "reviewer_a" if d else None} for i, f, d in boundaries]},
        "reaches": {"reaches": []}, "outcomes": {"segments": []}})


def _open(folder):
    video = folder / f"{VID}.mp4"
    video.write_bytes(b"x")
    return load_or_create_unified_gt(video)


def _rows(gt):
    return [(b.frame, b.determined) for b in sorted(gt.boundaries, key=lambda b: b.frame)]


def test_a_determined_cut_lands_on_the_cut_with_its_frame_when_a_cut_was_added(tmp_path):
    # Saved on cuts 0/1000/2000/3000; the person determined cut 2 at 2004.
    # Today the segmenter adds a cut at 500, so today's cut 2 is at 1000.
    _algo(tmp_path, [0, 500, 1000, 2000, 3000])
    _saved(tmp_path, [(0, 0, False), (1, 1000, False), (2, 2004, True), (3, 3000, False)])
    gt = _open(tmp_path)
    assert _rows(gt) == [(0, False), (500, False), (1000, False), (2004, True), (3000, False)]
    assert gt.unmatched_boundaries == []


def test_a_determined_cut_is_not_written_onto_another_row_when_a_cut_was_removed(tmp_path):
    # The person moved cut 2 from 2000 to 2100. Today the cut at 1000 is gone;
    # by position the person's 2100 would overwrite today's cut 2 (at 3000).
    _algo(tmp_path, [0, 2000, 3000])
    _saved(tmp_path, [(0, 0, False), (1, 1000, False), (2, 2100, True), (3, 3000, False)])
    gt = _open(tmp_path)
    assert _rows(gt) == [(0, False), (2000, False), (3000, False)]
    assert [(b.frame, b.determined) for b in gt.unmatched_boundaries] == [(2100, True)]


def test_a_cut_moved_far_on_the_same_cuts_comes_back_on_its_row(tmp_path):
    # Same algorithm cuts as when saved; the person moved cut 2 by 100 frames.
    _algo(tmp_path, [0, 1000, 2000, 3000])
    _saved(tmp_path, [(0, 0, False), (1, 1000, False), (2, 2100, True), (3, 3000, False)])
    gt = _open(tmp_path)
    assert _rows(gt) == [(0, False), (1000, False), (2100, True), (3000, False)]
    assert gt.unmatched_boundaries == []


def test_a_finalized_file_with_only_determined_cuts_round_trips(tmp_path):
    # "Save as Ground Truth" writes only determined cuts: here 1 and 2, the
    # person having moved cut 2 far. Nothing else changed.
    _algo(tmp_path, [0, 1000, 2000, 3000])
    _saved(tmp_path, [(1, 1003, True), (2, 2150, True)])
    gt = _open(tmp_path)
    assert _rows(gt) == [(0, False), (1003, True), (2150, True), (3000, False)]


def test_unmatched_cuts_are_saved_back_and_read_again(tmp_path):
    _algo(tmp_path, [0, 2000, 3000])
    _saved(tmp_path, [(0, 0, False), (1, 1000, False), (2, 2100, True), (3, 3000, False)])
    gt = _open(tmp_path)
    doc = _unified_gt_to_dict(gt)
    assert [b["frame"] for b in doc["segmentation"]["unmatched"]] == [2100]
    assert 2100 not in [b["frame"] for b in doc["segmentation"]["boundaries"]]
    _write(tmp_path, f"{VID}_unified_ground_truth.json", doc)
    again = load_unified_gt(tmp_path / f"{VID}.mp4")
    assert [b.frame for b in again.unmatched_boundaries] == [2100]


def test_a_finalized_cut_is_not_put_out_of_order_after_a_new_cut(tmp_path):
    # As above, but today the segmenter adds a cut at 1800: by position the
    # person's 2150 would replace 1800 and sit after 2000. Kept, not placed.
    _algo(tmp_path, [0, 1000, 1800, 2000, 3000])
    _saved(tmp_path, [(1, 1003, True), (2, 2150, True)])
    gt = _open(tmp_path)
    assert _rows(gt) == [(0, False), (1003, True), (1800, False), (2000, False),
                         (3000, False)]
    assert [b.frame for b in gt.unmatched_boundaries] == [2150]
