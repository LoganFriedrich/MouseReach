"""The claim on a review bundle belongs to one process, not to one machine.

The watcher's return scan, a review window and mousereach-route-to-queue run as
separate processes of the machine where review happens. A claim that belonged
to the machine let one of them write into a bundle while another was moving
it out (a flag written into an outcomes file the scan was copying away, then
lost). Now another live process of the same machine is refused, a crashed
process's claim is taken over, and one thread may take its own claim again.
Built under tmp_path with an invented id.
"""
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import mousereach.watcher.review_return as rr
from mousereach.config import Paths

VID = "20990101_ABC9901_P1"

_HOLDER = textwrap.dedent("""
    import os, sys, time
    from pathlib import Path
    from mousereach.config import Paths
    Paths.REVIEW_ROOT = Path(sys.argv[1])
    Paths.NAS_ROOT = Path(sys.argv[1]).parent
    import mousereach.watcher.review_return as rr
    print("claimed" if rr._claim_return(sys.argv[2]) else "refused", flush=True)
    go = Path(sys.argv[3])
    for _ in range(600):
        if go.exists():
            break
        time.sleep(0.05)
    os._exit(0)          # ends WITHOUT releasing, as a crash would
""")


@pytest.fixture
def review(tmp_path, monkeypatch):
    root = tmp_path / "nas" / "Review"
    root.mkdir(parents=True)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", root)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    return root


def test_another_process_of_this_machine_is_refused_until_it_ends(review, tmp_path):
    go = tmp_path / "go"
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(review), VID, str(go)],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "claimed"
        assert rr._claim_return(VID) is False
        go.write_text("x")
        holder.wait(timeout=60)
        # Its process has ended without releasing: the claim is taken over.
        assert rr._claim_return(VID) is True
        rr._release_return_claim(VID)
        assert not (review / ".return_claims" / f"{VID}.claim").exists()
    finally:
        go.write_text("x")
        holder.wait(timeout=60)


def test_another_thread_of_this_process_is_refused(review):
    assert rr._claim_return(VID)
    try:
        seen = []
        t = threading.Thread(target=lambda: seen.append(rr._claim_return(VID)))
        t.start()
        t.join()
        assert seen == [False]
    finally:
        rr._release_return_claim(VID)


def test_a_nested_claim_is_released_by_the_outermost_holder_only(review):
    marker = review / ".return_claims" / f"{VID}.claim"
    assert rr._claim_return(VID)
    claimed, release = rr._claim_return_nested(VID)    # a route the scan started
    assert claimed and release
    rr._release_return_claim(VID)
    assert marker.exists()                             # the scan still holds it
    rr._release_return_claim(VID)
    assert not marker.exists()


def test_two_takers_of_an_ended_claim_never_both_win(tmp_path, monkeypatch):
    """Two live processes both see the claim of a process that has ended. The
    first takes it over; the second, which read the ended holder before the
    first acted, must not delete the first one's new marker and take it too.
    Interleaved by hand: process B checks the ended holder, and process A
    takes the claim over at exactly that moment."""
    from mousereach.watcher import claim_marker as cm
    host = "THIS-NODE"
    claims = tmp_path / "claims"
    claims.mkdir()
    (claims / f"{VID}.claim").write_text(
        f"{host}\n2099-01-01T00:00:00\nprocess dead 1.0\n", encoding="ascii")
    results = {}
    interleaved = []

    def alive(identity):
        if identity == "dead 1.0" and not interleaved:
            interleaved.append(True)
            # B has read the ended holder and decided to take over; A acts first.
            results["A"] = cm.take(claims, VID, host, ".claim", process="A 1.0")
        return identity != "dead 1.0"

    monkeypatch.setattr(cm, "process_alive", alive)
    results["B"] = cm.take(claims, VID, host, ".claim", process="B 1.0")
    assert results["A"] is True
    assert results["B"] is False
    assert cm.holder_process(claims, VID, ".claim") == "A 1.0"
    assert sorted(p.name for p in claims.iterdir()) == [f"{VID}.claim"]


def _legacy_marker(claims, key, written):
    """A marker as an older version wrote it: this machine's name and a time,
    no process."""
    import socket
    claims.mkdir(parents=True, exist_ok=True)
    (claims / f"{key}.claim").write_text(
        f"{socket.gethostname()}\n{written.isoformat()}\n", encoding="ascii")


def test_a_fresh_marker_naming_no_process_is_not_taken(tmp_path):
    """While old and new code run side by side (a review window started
    before an update), a marker of this machine naming no process may belong
    to a live process: a process-aware caller does not take it as its own."""
    import socket
    from datetime import datetime
    from mousereach.watcher import claim_marker
    claims = tmp_path / ".return_claims"
    _legacy_marker(claims, VID, datetime.now())
    assert not claim_marker.take(claims, VID, socket.gethostname(), suffix=".claim",
                                 process=claim_marker.this_process())
    assert "process" not in (claims / f"{VID}.claim").read_text(encoding="ascii")


def test_an_old_marker_naming_no_process_is_taken_over(tmp_path):
    import socket
    from datetime import datetime, timedelta
    from mousereach.watcher import claim_marker
    claims = tmp_path / ".return_claims"
    _legacy_marker(claims, VID, datetime.now() - timedelta(
        seconds=claim_marker.LEGACY_MARKER_TAKEOVER_S + 60))
    me = claim_marker.this_process()
    assert claim_marker.take(claims, VID, socket.gethostname(), suffix=".claim",
                             process=me)
    assert claim_marker.holder_process(claims, VID, ".claim") == me
