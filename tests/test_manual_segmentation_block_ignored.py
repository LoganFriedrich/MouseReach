"""Cuts stored inside an old review file are history, never the cuts.

An earlier version of the review window had its own re-cut editor, which saved
cuts as a ``manual_segmentation`` block inside the review file. Every pipeline
step reads the cuts from the segments file. The window still rebuilt its
pellets from such a block when an old review had one: it showed cuts nothing
else uses, with no algorithm outcome on any pellet, and a later save from a
window that had not found the block dropped the record from every copy.

Now: the window builds its pellets from the segments file only, and a save
keeps an earlier copy's block unchanged as a record.
"""
import json
from types import SimpleNamespace

from mousereach.review import causal_review_io as io
from mousereach.review.causal_review_widget import CausalReviewWidget

STEM = "20990101_TEST0101_P1"

# The segments file: the cuts every pipeline step uses.
SEGMENTS = {"boundaries": [100, 1100, 2100, 3100], "boundary_source": "algo"}
# Cuts an old window stored in the review file: different, and one more pellet.
OLD_BLOCK = {"boundaries": [100, 900, 1700, 2500, 3100], "by": "reviewer_a",
             "at": "2099-01-01T00:00:00", "n_segments": 4}


def _old_review():
    return {"type": "causal_review", "schema_version": "1.1", "video_stem": STEM,
            "reviewer": "reviewer_a", "reviewed_at": "2099-01-01T00:00:00",
            "provenance": {}, "segments": [], "manual_segmentation": OLD_BLOCK}


def _window(tmp_path):
    """The parts of the window its load step uses, with an old review file
    holding a manual_segmentation block in the bundle folder."""
    (tmp_path / f"{STEM}_causal_review.json").write_text(json.dumps(_old_review()))
    status = SimpleNamespace(setText=lambda *_: None)
    stub = SimpleNamespace(
        _bundle_dir=tmp_path, video_path=tmp_path / f"{STEM}.mp4", _video_stem=STEM,
        _segments_data=SEGMENTS,
        _outcomes_data={"segments": [{"segment_num": n, "outcome": "retrieved"}
                                     for n in (1, 2, 3)]},
        _assignments_data={"reaches": []}, _reaches_data={"reaches": []},
        _status_label=status, _triage_only=False, _triaged_indices=[],
    )
    stub._recompute_triaged_indices = lambda: None
    return stub


def _load_segments(stub):
    """What the window's load does before showing pellets: whatever it reads
    from the review file, then building the pellet list."""
    peek = getattr(CausalReviewWidget, "_peek_manual_boundaries", None)
    if peek is not None:            # the window as it was
        stub._manual_boundaries = peek(stub)
    CausalReviewWidget._build_segment_list(stub)


def test_the_window_shows_the_segments_files_cuts_not_an_old_block(tmp_path):
    stub = _window(tmp_path)
    _load_segments(stub)
    spans = [(s["start_frame"], s["end_frame"]) for s in stub._segment_list]
    assert spans == [(100, 1099), (1100, 2099), (2100, 3099)]
    # the algorithm's outcomes are on the pellets, as on any other video
    assert [s["outcome"] for s in stub._segment_list] == ["retrieved"] * 3


def test_an_old_block_does_not_make_the_cuts_a_persons(tmp_path):
    stub = _window(tmp_path)
    _load_segments(stub)
    assert CausalReviewWidget._segmentation_is_human(stub) is False


def test_a_save_keeps_an_earlier_copys_block_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "durable_review_path", lambda stem: None)
    (tmp_path / f"{STEM}_causal_review.json").write_text(json.dumps(_old_review()))
    rec = io.build_segment_record(
        segment_num=1, pellet_num=1, algo_outcome="retrieved", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome="untouched", human_causal_reach=None,
        is_phantom=False, agreed=False, answers={}, notes="",
        segment_span={"start": 100, "end": 1099})
    out = io.save_causal_review(STEM, tmp_path, [rec], provenance={}, reviewer="reviewer_b")
    saved = json.loads(out.read_text())
    assert saved["manual_segmentation"] == OLD_BLOCK
