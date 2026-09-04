"""safe_copy under transient locks: retries, bounded give-up, raise-proof
cleanup.

WHY: a reviewer GUI (or indexer/backup pass) briefly holding a destination
file surfaces as PermissionError winerror 5/32. On 2026-09-04 one such hold
failed a reprocess work item twice over -- the copy failed, and then the
cleanup unlink hit the SAME lock and raised out of the except block. The
file was free again minutes later. These tests pin the absorbing behaviour.
"""
import pytest

import mousereach.watcher.transfer as transfer


@pytest.fixture
def fast(monkeypatch):
    """No real sleeping between retry attempts."""
    monkeypatch.setattr(transfer.time, "sleep", lambda s: None)


def _flaky_copy2(fail_times, exc_factory):
    calls = {"n": 0}

    def copy2(src, dst):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc_factory()
        # succeed: real copy so verify has sizes to compare
        with open(src, "rb") as fh_in, open(dst, "wb") as fh_out:
            fh_out.write(fh_in.read())

    return copy2, calls


def _sharing_violation():
    e = PermissionError(13, "The process cannot access the file")
    if not getattr(e, "winerror", None):        # non-Windows test hosts
        e.winerror = 32
    return e


def test_transient_lock_is_retried_to_success(tmp_path, monkeypatch, fast):
    src = tmp_path / "a.json"
    src.write_text("payload")
    dst = tmp_path / "out" / "a.json"
    copy2, calls = _flaky_copy2(2, _sharing_violation)
    monkeypatch.setattr(transfer.shutil, "copy2", copy2)
    assert transfer.safe_copy(src, dst, verify=True) is True
    assert calls["n"] == 3
    assert dst.read_text() == "payload"


def test_persistent_lock_gives_up_false_never_raises(tmp_path, monkeypatch, fast):
    src = tmp_path / "a.json"
    src.write_text("payload")
    dst = tmp_path / "out" / "a.json"
    copy2, calls = _flaky_copy2(99, _sharing_violation)
    monkeypatch.setattr(transfer.shutil, "copy2", copy2)
    assert transfer.safe_copy(src, dst, verify=True) is False
    assert calls["n"] == transfer.COPY_RETRY_ATTEMPTS


def test_non_transient_error_fails_immediately(tmp_path, monkeypatch, fast):
    src = tmp_path / "a.json"
    src.write_text("payload")
    dst = tmp_path / "out" / "a.json"
    copy2, calls = _flaky_copy2(99, lambda: OSError(28, "No space left"))
    monkeypatch.setattr(transfer.shutil, "copy2", copy2)
    assert transfer.safe_copy(src, dst, verify=True) is False
    assert calls["n"] == 1


def test_locked_destination_cleanup_never_raises(tmp_path, monkeypatch, fast):
    """The 2026-09-04 killer: cleanup unlink hitting the same lock. With an
    open handle on the destination (unlink refuses on Windows), safe_copy
    must still return False rather than raise."""
    src = tmp_path / "a.json"
    src.write_text("payload")
    dst = tmp_path / "out" / "a.json"
    dst.parent.mkdir()
    dst.write_text("old local twin")

    def boom(*a, **k):
        raise _sharing_violation()

    monkeypatch.setattr(transfer.shutil, "copy2", boom)
    with open(dst, "r"):
        assert transfer.safe_copy(src, dst, verify=True) is False
