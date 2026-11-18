"""A quality-check hold shows what the check found, not only "qc_needs_review".

The gate routes a quality-check hold with the bare reason ``qc_needs_review``;
what the check found is in the bundle's ``<stem>_triage.json``, which no review
tool read, so the person was told the video was flagged but not what to look
at. The re-segmentation window's banner and the re-segmentation tool now show
the critical findings. The routed reason itself stays the bare word (the return
scan and the re-segmentation tool's release match it).
Built under tmp_path; invented ids only.
"""
import json
from types import SimpleNamespace

from mousereach.review.causal_review_widget import CausalReviewWidget

STEM = "20990101_ABC9901_P1"


class _Banner:
    text, visible = "", None

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, _):
        pass

    def setVisible(self, v):
        self.visible = v


def _qc_bundle(tmp_path, flags=None):
    b = tmp_path / STEM
    b.mkdir()
    (b / f"{STEM}_routing.json").write_text(
        json.dumps({"routed_at": "2099-01-10T09:00:00",
                    "routed_reason": "qc_needs_review"}), encoding="utf-8")
    if flags is not None:
        (b / f"{STEM}_triage.json").write_text(json.dumps({
            "video_id": STEM, "verdict": "needs_review", "flags": flags}),
            encoding="utf-8")
    return b


FLAGS = [
    {"category": "dlc_coherence", "severity": "critical",
     "description": "paw tracked outside the enclosure for 40% of frames",
     "segment_num": None},
    {"category": "structural", "severity": "critical",
     "description": "reach longer than its segment", "segment_num": 7},
    {"category": "statistical", "severity": "warning",
     "description": "a warning that never holds a video", "segment_num": 2},
]


def test_the_banner_lists_the_critical_findings(tmp_path):
    b = _qc_bundle(tmp_path, FLAGS)
    banner = _Banner()
    CausalReviewWidget._update_routing_banner(SimpleNamespace(
        _routing_banner=banner, _deep_review=True, _bundle_dir=b, _video_stem=STEM))
    assert "whole video: paw tracked outside the enclosure for 40% of frames" in banner.text
    assert "segment 7: reach longer than its segment" in banner.text
    assert "never holds a video" not in banner.text


def test_the_banner_says_when_nothing_was_recorded(tmp_path):
    b = _qc_bundle(tmp_path)
    banner = _Banner()
    CausalReviewWidget._update_routing_banner(SimpleNamespace(
        _routing_banner=banner, _deep_review=True, _bundle_dir=b, _video_stem=STEM))
    assert "not recorded with this video" in banner.text


def test_the_resegmentation_tool_reason_includes_the_findings(tmp_path):
    from mousereach.review.release_cli import describe_routing_reason, qc_hold_findings
    b = _qc_bundle(tmp_path, FLAGS)
    assert qc_hold_findings(b, STEM) == [
        "whole video: paw tracked outside the enclosure for 40% of frames",
        "segment 7: reach longer than its segment"]
    text = describe_routing_reason(b, STEM)
    assert text.startswith("qc_needs_review (the quality check found: ")
    # The stored reason is unchanged.
    assert json.loads((b / f"{STEM}_routing.json").read_text())["routed_reason"] \
        == "qc_needs_review"
