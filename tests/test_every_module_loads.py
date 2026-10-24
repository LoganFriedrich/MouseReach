"""Every module in the package imports, and none uses a name it never defines.

WHY: two review-tool faults on 2026-10-24 sat in code no test exercised -- a
plugin hook naming something napari no longer has (the whole module failed to
import) and a ``logger`` the module never defined (an error handler raised
NameError instead of reporting). Neither needed the code to run to be found:
importing every module finds the first kind, and pyflakes' "undefined name"
finds the second. The review package is also on the watcher's lazy-import
path, so a module that does not import can take watcher work down with it.
"""
import importlib
import pkgutil
from pathlib import Path

import pytest

import mousereach


def _modules():
    for m in pkgutil.walk_packages(mousereach.__path__, "mousereach."):
        # A package's __main__ IS its command: importing it runs the command
        # (mousereach.census.__main__ takes the whole census), so it is not
        # imported here.
        if m.name.rsplit(".", 1)[-1] == "__main__":
            continue
        yield m.name


def test_every_module_imports():
    failed = {}
    for name in _modules():
        try:
            importlib.import_module(name)
        except Exception as e:  # report every failure, not just the first
            failed[name] = "%s: %s" % (type(e).__name__, e)
    assert not failed, "modules that do not import:\n" + "\n".join(
        "  %s -- %s" % kv for kv in sorted(failed.items()))


def test_no_module_uses_an_undefined_name():
    api = pytest.importorskip("pyflakes.api")
    from pyflakes import messages
    from pyflakes.reporter import Reporter
    import io

    class _Collect(Reporter):
        def __init__(self):
            super().__init__(io.StringIO(), io.StringIO())
            self.found = []

        def flake(self, message):
            if isinstance(message, messages.UndefinedName):
                self.found.append(str(message))

    rep = _Collect()
    api.checkRecursive([str(Path(mousereach.__file__).parent)], rep)
    assert not rep.found, "undefined names:\n  " + "\n  ".join(rep.found)
