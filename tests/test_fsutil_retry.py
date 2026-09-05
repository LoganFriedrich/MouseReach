"""fsutil: transient holds are absorbed; real failures still raise.

WHY: a stage's own output write died on a reviewer GUI's brief hold of the
file (2026-09-05, reach save_results) -- the third transient-lock casualty
in three days, each at a different unguarded site. This is the shared
primitive; these tests pin its contract: retry ONLY transient locks, and
after the retries the LAST error raises so a stuck file still fails the
stage loudly instead of stranding it silently.
"""
import json

import pytest

import mousereach.pipeline.fsutil as fs


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(fs.time, "sleep", lambda s: None)


def _sharing_violation():
    e = PermissionError(13, "The process cannot access the file")
    if not getattr(e, "winerror", None):
        e.winerror = 32
    return e


def _flaky_open(fail_times):
    calls = {"n": 0}
    real = fs._open

    def opener(path, mode="r", *a, **k):
        if "w" in mode:
            calls["n"] += 1
            if calls["n"] <= fail_times:
                raise _sharing_violation()
        return real(path, mode, *a, **k)

    return opener, calls


def test_transient_hold_is_absorbed(tmp_path, monkeypatch, fast):
    p = tmp_path / "out.json"
    opener, calls = _flaky_open(2)
    monkeypatch.setattr(fs, "_open", opener)
    fs.dump_json_with_retry(p, {"ok": 1}, indent=2)
    assert calls["n"] == 3
    assert json.loads(p.read_text()) == {"ok": 1}


def test_persistent_hold_raises_the_last_error(tmp_path, monkeypatch, fast):
    p = tmp_path / "out.json"
    opener, calls = _flaky_open(99)
    monkeypatch.setattr(fs, "_open", opener)
    with pytest.raises(PermissionError):
        fs.dump_json_with_retry(p, {"ok": 1})
    assert calls["n"] == fs.RETRY_ATTEMPTS


def test_non_transient_error_raises_immediately(tmp_path, monkeypatch, fast):
    calls = {"n": 0}

    def opener(path, mode="r", *a, **k):
        calls["n"] += 1
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fs, "_open", opener)
    with pytest.raises(OSError):
        fs.dump_json_with_retry(tmp_path / "out.json", {})
    assert calls["n"] == 1


def test_retry_transient_returns_the_value(monkeypatch, fast):
    seq = iter([_sharing_violation(), 42])

    def fn():
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    assert fs.retry_transient(fn) == 42


def test_predicate_is_single_sourced_into_transfer():
    """watcher.transfer must use THIS predicate, not its own copy -- two
    parsers of the same question is how blank detection went wrong once."""
    import mousereach.watcher.transfer as t
    assert t._is_transient_lock is fs.is_transient_lock
