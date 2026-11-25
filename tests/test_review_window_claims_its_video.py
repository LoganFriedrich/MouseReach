"""One review window per video.

Two review windows could be handed the same video: each picked by priority
and a random draw, with nothing saying another window had it open. Now a
window claims the bundle it shows with a marker in
``<review root>/.review_claims/<video>.review`` (exclusive create,
watcher/claim_marker), skips a bundle another window holds, lets its video
go when it moves on, and a claim nobody touched for an hour is broken.

Built under tmp_path with invented ids.
"""
import json
import os
import time

import pytest

from mousereach.review import review_claims as rc
from mousereach.watcher import claim_marker as cm

A, B = "20990101_ABC9901_P1", "20990101_ABC9902_P1"


def _bundle(root, stem):
    b = root / "Review" / "triage" / stem
    b.mkdir(parents=True)
    return b


def _held_by_other_window(bundle):
    """A claim written by another reviewer's window (other machine/login)."""
    d = rc.claims_dir(bundle)
    assert cm.take(d, bundle.name, "OTHERPC/reviewer-b", rc.SUFFIX, process="1 1.0")


def test_the_claim_is_a_marker_in_the_review_root(tmp_path):
    b = _bundle(tmp_path, A)
    assert rc.claim(b)
    marker = tmp_path / "Review" / ".review_claims" / f"{A}.review"
    assert marker.is_file()
    assert rc.holder(b) == rc.who()
    assert rc.claim(b)                  # the same window again: still its own
    rc.release(b)
    assert not marker.exists()


def test_another_windows_claim_is_respected(tmp_path):
    b = _bundle(tmp_path, A)
    _held_by_other_window(b)
    assert not rc.claim(b)
    rc.release(b)                       # never removes another window's claim
    assert rc.holder(b) == "OTHERPC/reviewer-b"


def test_an_idle_claim_is_broken(tmp_path, monkeypatch):
    b = _bundle(tmp_path, A)
    _held_by_other_window(b)
    marker = rc.claims_dir(b) / f"{A}.review"
    old = time.time() - rc.IDLE_AFTER_S - 600
    os.utime(marker, (old, old))
    assert rc.claim(b)
    assert rc.holder(b) == rc.who()
    rc.release(b)


def test_the_window_skips_a_video_another_window_has_open(tmp_path):
    from types import SimpleNamespace
    from mousereach.review.causal_review_widget import CausalReviewWidget as W
    a, b = _bundle(tmp_path, A), _bundle(tmp_path, B)
    _held_by_other_window(a)
    win = SimpleNamespace(_bundle_dir=None, _pool_cache=None,
                          _bundle_eligible=lambda bundle, root: True)
    picked, _ = W._pick_eligible(win, [a, b], tmp_path / "Review")
    assert picked == b
    assert rc.holder(b) == rc.who()
    # Moving on lets the video this window had go.
    win._bundle_dir = b
    c = _bundle(tmp_path, "20990101_ABC9903_P1")
    picked, _ = W._pick_eligible(win, [c], tmp_path / "Review")
    assert picked == c and rc.holder(b) is None
    rc.release(c)
    # Nobody else could be handed A while the other window holds it.
    picked, _ = W._pick_eligible(win, [a], tmp_path / "Review")
    assert picked is None
