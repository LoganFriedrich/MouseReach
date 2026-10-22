"""An exclusive claim on one item of a shared pool (watcher/claim_marker.py).

WHY: several watcher nodes read the same shared folders, and exactly one may
act on any one item. The markers this replaced were check, write, sleep, read
back -- they narrowed the race between two nodes without closing it -- and
another node's marker was removed for being old, whether or not its holder was
still at work. These tests pin the two properties that replace them: exactly
one claimant wins, and a claim is broken only when its holder has stopped
reporting.
"""
import os
import time
from multiprocessing import get_context

from mousereach.watcher import claim_marker as cm

ME, OTHER = "THIS-NODE", "OTHER-NODE"


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))
    return path


def _report(status_dir, host, age_s=0):
    status_dir.mkdir(parents=True, exist_ok=True)
    f = status_dir / f"{host}.json"
    f.write_text("{}", encoding="ascii")
    return _age(f, age_s) if age_s else f


# ---------------------------------------------------------------- the claim

def test_a_claim_is_taken_once_and_is_then_held(tmp_path):
    assert cm.take(tmp_path, "vid", ME) is True
    assert cm.holder(tmp_path, "vid") == ME
    assert cm.take(tmp_path, "vid", OTHER) is False


def test_the_holder_can_take_its_own_claim_again(tmp_path):
    """A node that stopped between claiming and releasing must not lock
    itself out of its own work."""
    assert cm.take(tmp_path, "vid", ME) is True
    assert cm.take(tmp_path, "vid", ME) is True


def test_a_node_releases_only_its_own_claim(tmp_path):
    cm.take(tmp_path, "vid", OTHER)
    assert cm.release(tmp_path, "vid", ME) is False
    assert cm.holder(tmp_path, "vid") == OTHER
    assert cm.release(tmp_path, "vid", OTHER) is True
    assert cm.holder(tmp_path, "vid") is None


def test_no_claim_when_the_claim_folder_cannot_be_made(tmp_path):
    blocker = tmp_path / "claims"
    blocker.write_text("a file where the folder goes", encoding="ascii")
    assert cm.take(blocker, "vid", ME) is False


def _race(args):
    folder, host, start, keys = args
    while time.time() < start:
        pass
    return [k for k in keys if cm.take(folder, k, host)]


def test_racing_nodes_never_both_win(tmp_path):
    """The property everything else rests on. Measured on the lab's share
    before it was relied on; here on whatever disk the tests run on."""
    keys = [f"vid{i}" for i in range(60)]
    start = time.time() + 1.5
    with get_context("spawn").Pool(3) as pool:
        won = pool.map(_race, [(tmp_path, f"NODE-{i}", start, keys) for i in range(3)])
    winners = {}
    for i, ks in enumerate(won):
        for k in ks:
            winners.setdefault(k, []).append(i)
    assert sorted(winners) == sorted(keys)               # every key won ...
    assert all(len(v) == 1 for v in winners.values())    # ... exactly once


# ---------------------------------------------------------------- liveness

def test_a_claim_whose_holder_stopped_reporting_is_broken(tmp_path):
    claims, status = tmp_path / "claims", tmp_path / "status"
    cm.take(claims, "vid", OTHER)
    _report(status, OTHER, age_s=cm.GONE_AFTER_S + 60)
    _report(status, ME)                                  # the share's "now"
    assert cm.sweep_gone(claims, ME, status) == ["vid"]
    assert cm.holder(claims, "vid") is None


def test_an_old_claim_whose_holder_is_reporting_is_kept(tmp_path):
    """A claim may be old because its work is long -- a video waiting in a
    node's queue, or held for review for weeks. Its age says nothing."""
    claims, status = tmp_path / "claims", tmp_path / "status"
    cm.take(claims, "vid", OTHER)
    _age(claims / "vid.claimed", cm.GONE_AFTER_S * 20)
    _report(status, OTHER)
    _report(status, ME)
    assert cm.sweep_gone(claims, ME, status) == []
    assert cm.holder(claims, "vid") == OTHER


def test_quiet_is_measured_against_the_newest_report_not_this_clock(tmp_path):
    """Both times come from the file server. Here every report is a day old
    (as if this machine's clock ran a day ahead of the share): the holder is
    no quieter than anyone else, so it is not gone."""
    claims, status = tmp_path / "claims", tmp_path / "status"
    cm.take(claims, "vid", OTHER)
    _report(status, OTHER, age_s=86400)
    _report(status, ME, age_s=86400)
    assert cm.sweep_gone(claims, ME, status) == []


def test_this_nodes_own_claims_are_never_swept_by_it(tmp_path):
    claims, status = tmp_path / "claims", tmp_path / "status"
    cm.take(claims, "vid", ME)
    _report(status, ME, age_s=cm.GONE_AFTER_S * 5)
    _report(status, OTHER)
    assert cm.sweep_gone(claims, ME, status) == []
    assert cm.holder(claims, "vid") == ME


def test_with_no_reports_at_all_nothing_is_broken(tmp_path):
    """No evidence that anyone is gone, so every claim stands."""
    claims = tmp_path / "claims"
    cm.take(claims, "vid", OTHER)
    _age(claims / "vid.claimed", cm.GONE_AFTER_S * 20)
    assert cm.sweep_gone(claims, ME, tmp_path / "no-status") == []


def test_a_holder_with_no_report_is_judged_by_the_claims_own_age(tmp_path):
    """An older watcher, or a host that no longer exists, never writes a
    report; its claim goes once it is older than the limit by the share."""
    claims, status = tmp_path / "claims", tmp_path / "status"
    _report(status, ME)
    cm.take(claims, "old", OTHER)
    _age(claims / "old.claimed", cm.GONE_AFTER_S + 60)
    cm.take(claims, "new", OTHER)
    assert cm.sweep_gone(claims, ME, status) == ["old"]


def test_an_empty_marker_counts_as_claimed(tmp_path):
    """A node stopped between creating the marker and writing its name: the
    item is still taken (holder unknown) until the marker is old enough."""
    claims, status = tmp_path / "claims", tmp_path / "status"
    claims.mkdir()
    (claims / "vid.claimed").write_bytes(b"")
    assert cm.holder(claims, "vid") == ""
    assert cm.take(claims, "vid", ME) is False
    _report(status, ME)
    assert cm.sweep_gone(claims, ME, status) == []
    _age(claims / "vid.claimed", cm.GONE_AFTER_S + 60)
    assert cm.sweep_gone(claims, ME, status) == ["vid"]
