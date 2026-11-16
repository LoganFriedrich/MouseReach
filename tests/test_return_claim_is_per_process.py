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
