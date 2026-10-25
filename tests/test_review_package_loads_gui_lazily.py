"""The watcher's review imports load no GUI; the GUI names still resolve.

The watcher imports mousereach.review for causal_review_io and triage_status
(stdlib-only modules), but review/__init__.py imported every widget, so a
headless daemon loaded napari, Qt, vispy, pandas and scipy -- about 2,600
modules, 4.6 s -- the first time a cycle touched review code, and a widget that
failed to import took the watcher's review paths down with it (2026-10-24).
"""
import subprocess
import sys
from pathlib import Path

import mousereach.review as review

SRC = str(Path(review.__file__).resolve().parents[2])
GUI = ("napari", "qtpy", "PyQt5", "vispy", "pandas", "scipy", "skimage", "magicgui")


def test_the_watchers_review_imports_load_no_gui():
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from mousereach.review.causal_review_io import resolve_review_path, find_gt\n"
        "from mousereach.review.triage_status import triage_status\n"
        "import mousereach.review\n"
        "print(','.join(m for m in %r if m in sys.modules))\n" % (SRC, GUI))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "the watcher's imports loaded: " + out.stdout


def test_every_public_name_still_resolves():
    for name in review.__all__:
        assert getattr(review, name) is not None, name
    from mousereach.review import CausalReviewWidget, ComparisonPanel   # the from-import form
    assert CausalReviewWidget.__name__ == "CausalReviewWidget"
    assert ComparisonPanel.__name__ == "ComparisonPanel"


def test_an_unknown_name_is_still_an_error():
    try:
        review.NoSuchThing
    except AttributeError:
        return
    raise AssertionError("an unknown name resolved")
