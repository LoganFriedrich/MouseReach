"""The suite-wide test guard (tests/real_root_guard.py, loaded by conftest.py)
REFUSES a write into this machine's real pipeline folders -- before anything
reaches the disk -- whatever kind of write it is and however the path is
spelled; it does not only report it afterwards.

WHY: a test that wrote into the real shared version index failed, as the guard
of the time intended, but the row it wrote stayed in the live index. And the
guard of the time never saw os.open / tempfile.mkstemp creations at all (their
audit event carries int flags, not a mode string), so the temporary files of
atomic writes, claim markers and lock files went straight through.

Nothing real is touched here: a folder under tmp_path is registered as a real
root for the length of one test. tests/test_guard_holds_in_an_isolated_run.py
checks the guard in a separate, isolated Python.
"""
import os
import shutil
import sqlite3
import tempfile

import pytest

import conftest
import real_root_guard

@pytest.fixture
def root(tmp_path):
    """A folder treated as a real root by the installed guard for this test.
    Its records are removed afterwards (they are this test's own probes)."""
    real = tmp_path / "real_share"
    real.mkdir()
    (real / "existing.json").write_text("{}")
    start = len(conftest.GUARD.hits)
    conftest.GUARD.add_root(str(real))
    try:
        yield real
    finally:
        conftest.GUARD.remove_root(str(real))
        del conftest.GUARD.hits[start:]


def _refused(fn):
    with pytest.raises(PermissionError, match="test guard"):
        fn()


def _only_existing(real):
    assert sorted(p.name for p in real.iterdir()) == ["existing.json"]


@pytest.mark.parametrize("mode", ["w", "wb", "a", "x", "r+", "w+"])
def test_open_in_any_write_mode_is_refused(root, mode):
    _refused(lambda: open(root / "new.json", mode))
    _refused(lambda: open(root / "existing.json", mode))
    _only_existing(root)
    assert (root / "existing.json").read_text() == "{}"


def test_os_open_creations_are_refused(root):
    # The audit event of os.open has mode None and int flags.
    _refused(lambda: os.open(str(root / "a.tmp"), os.O_CREAT | os.O_WRONLY))
    _refused(lambda: os.open(str(root / "b.lock"), os.O_CREAT | os.O_EXCL | os.O_RDWR))
    _refused(lambda: tempfile.mkstemp(dir=str(root), suffix=".tmp"))
    _refused(lambda: tempfile.mkdtemp(dir=str(root)))
    _refused(lambda: tempfile.NamedTemporaryFile(dir=str(root)))
    _refused(lambda: (root / "touched").touch())
    _refused(lambda: (root / "text.json").write_text("x"))
    _only_existing(root)


def test_folder_creates_renames_removes_and_copies_are_refused(root, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("o")
    _refused(lambda: os.mkdir(root / "Processing"))
    _refused(lambda: os.makedirs(root / "Processing" / "Review"))
    # A create of a folder that already exists is refused too; makedirs with
    # exist_ok swallows that refusal, but the attempt is recorded (so the test
    # making it fails) -- exactly the case that once hid a real folder create.
    start = len(conftest.GUARD.hits)
    os.makedirs(root, exist_ok=True)
    assert ("os.mkdir", real_root_guard.norm(str(root))) in conftest.GUARD.hits[start:]
    _refused(lambda: os.rename(outside, root / "moved_in.json"))
    _refused(lambda: os.replace(outside, root / "existing.json"))
    _refused(lambda: os.rename(root / "existing.json", tmp_path / "moved_out.json"))
    _refused(lambda: os.remove(root / "existing.json"))
    _refused(lambda: os.utime(root / "existing.json"))
    _refused(lambda: shutil.copyfile(outside, root / "copied.json"))
    _refused(lambda: shutil.copy2(outside, root / "copied2.json"))
    (tmp_path / "src_tree").mkdir()
    _refused(lambda: shutil.copytree(tmp_path / "src_tree", root / "tree"))
    _refused(lambda: shutil.move(str(outside), str(root / "m.json")))
    _refused(lambda: shutil.rmtree(root))
    _refused(lambda: sqlite3.connect(str(root / "watcher.db")))
    _only_existing(root)
    assert outside.read_text() == "o"


def test_reading_a_real_root_is_allowed(root, tmp_path):
    assert (root / "existing.json").read_text() == "{}"
    shutil.copyfile(root / "existing.json", tmp_path / "copy_of_real.json")
    assert (tmp_path / "copy_of_real.json").read_text() == "{}"


def test_a_refused_write_fails_the_test_even_when_swallowed(root):
    start = len(conftest.GUARD.hits)
    try:
        (root / "best_effort.json").write_text("x")
    except OSError:
        pass                                   # best-effort code swallows it
    assert conftest.GUARD.hits[start:], "the attempt must still be recorded"


def test_every_spelling_of_a_real_root_is_refused(root, monkeypatch):
    s = str(root)
    upper = s.upper()
    forward = s.replace("\\", "/")
    _refused(lambda: open(os.path.join(upper, "a.json"), "w"))
    _refused(lambda: open(forward + "/b.json", "w"))
    monkeypatch.chdir(root.parent)
    _refused(lambda: open(os.path.join("real_share", "c.json"), "w"))
    _refused(lambda: open(os.path.join("real_share", "..", "real_share", "d.json"), "w"))
    if os.name == "nt":
        _refused(lambda: open("\\\\?\\" + os.path.join(s, "e.json"), "w"))
        short = real_root_guard._short_long(s)
        if short and short[0].lower() != s.lower():
            _refused(lambda: open(os.path.join(short[0], "f.json"), "w"))
    _only_existing(root)


def test_writes_elsewhere_are_not_refused(root, tmp_path):
    sibling = tmp_path / "real_share_sibling"
    sibling.mkdir()
    (sibling / "x.json").write_text("x")
    fd, name = tempfile.mkstemp(dir=str(tmp_path))
    os.close(fd)
    os.remove(name)
