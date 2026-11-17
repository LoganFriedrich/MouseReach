"""The re-segmentation window shows why a video is there, whatever the reason.

The banner showed a recorded routing reason only when it named segmentation
or "qc", and told the reviewer "no routing reason was recorded" for every
other one: the quality triage that could not run, a session a reviewer
flagged (with their note), a reason typed on the routing command.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.review.causal_review_widget import CausalReviewWidget
from mousereach.watcher.review_gate import TRIAGE_FAILED_REASON

STEM = "20990101_ABC9901_P1"
NO_REASON = "no routing reason was recorded"


class _Banner:
    def __init__(self):
        self.text, self.visible = "", None

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, _):
        pass

    def setVisible(self, v):
        self.visible = v


def _banner_for(tmp_path, reason):
    b = tmp_path / STEM
    b.mkdir()
    (b / f"{STEM}_routing.json").write_text(
        json.dumps({"routed_at": "2099-01-10T09:00:00", "routed_reason": reason}),
        encoding="utf-8")
    banner = _Banner()
    stub = SimpleNamespace(_routing_banner=banner, _deep_review=True,
                           _bundle_dir=b, _video_stem=STEM)
    CausalReviewWidget._update_routing_banner(stub)
    assert banner.visible is True
    return banner.text


def _short_segment_reason():
    from mousereach.review.triage_status import short_segment_reason
    return short_segment_reason(SimpleNamespace(too_short={4: (100, 102)}), [4])


@pytest.mark.parametrize("reason", [
    TRIAGE_FAILED_REASON,                         # the gate, quality triage could not run
    "qc_needs_review",                            # the gate, quality hold
    "segmentation_failed",                        # the gate
    "bench disagreement on pellet 3",             # typed on mousereach-route-to-queue
])
def test_a_recorded_reason_is_shown(tmp_path, reason):
    text = _banner_for(tmp_path, reason)
    assert NO_REASON not in text
    assert reason in text


def test_a_short_segment_reason_is_shown(tmp_path):
    reason = _short_segment_reason()
    text = _banner_for(tmp_path, reason)
    assert NO_REASON not in text and "too short" in text


def test_a_flagged_session_shows_the_reviewers_note(tmp_path):
    text = _banner_for(tmp_path, "reviewer flagged session 20990101_ABC9901 -- "
                                 "pellet missing from the tray in every trial")
    assert NO_REASON not in text
    assert "pellet missing from the tray in every trial" in text
    assert "20990101_ABC9901" in text


def test_the_triage_failed_banner_says_what_to_check(tmp_path):
    text = _banner_for(tmp_path, TRIAGE_FAILED_REASON)
    assert "could not run" in text and "Clear" in text


def test_only_an_empty_reason_says_none_was_recorded(tmp_path):
    assert NO_REASON in _banner_for(tmp_path, "")


def test_a_reason_is_shown_as_text_not_markup(tmp_path):
    text = _banner_for(tmp_path, "bench disagreement <b>3</b>")
    assert "&lt;b&gt;3&lt;/b&gt;" in text
