"""The suite-wide guard (tests/conftest.py) refuses a write into this
machine's real pipeline folders while a test runs; it does not only report it
afterwards.

WHY: a test that wrote into the real shared version index failed, as the guard
intended, but the row it wrote stayed in the live index. Refusing the write
before it happens is the only way a failing test leaves nothing behind.

Nothing real is written here: the guard's hook is called directly with a path
inside a real root, as the interpreter would call it before a write.
"""
import os
import sys

import pytest


def _guard():
    for name, mod in sys.modules.items():
        if name.endswith("conftest") and hasattr(mod, "_audit") and hasattr(mod, "_REAL_ROOTS"):
            return mod
    pytest.skip("the suite guard is not loaded")


def test_a_write_into_a_real_root_is_refused_during_a_test():
    g = _guard()
    if not g._REAL_ROOTS:
        pytest.skip("no pipeline configured on this machine: nothing to guard")
    target = os.path.join(g._REAL_ROOTS[0], "pipeline_records", "version_index.db")
    start = len(g._hits)
    try:
        with pytest.raises(PermissionError):
            g._audit("sqlite3.connect", (target,))
    finally:
        del g._hits[start:]          # this test's own probe is not a real write


def test_a_write_elsewhere_is_not_refused(tmp_path):
    g = _guard()
    g._audit("sqlite3.connect", (str(tmp_path / "local.db"),))
