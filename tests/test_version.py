#!/usr/bin/env python3
"""The package version is the pipeline's provenance record. It must be one number.

`mousereach.__version__` is stamped into every video's processing manifest by
`create_processing_manifest`, so it is what the record says produced a result.
It read "2.4.0" while pyproject.toml said "2.16.0-dev" -- every manifest written
since February named a version that did not exist.

Reading it from installed metadata instead does not help here: a stale
`mousereach.egg-info` sits in `src/` in both the Y: and the C: tree, left over
from an old setuptools install, and it shadows the real dist-info.
`importlib.metadata.version("mousereach")` returns 2.3.0 from it while pip
reports 2.16.0.dev0. So the literal stays, and this test is what keeps it honest.
"""

import re
from pathlib import Path

import mousereach

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    # the [project] table's own version, not a dependency pin
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m, "pyproject.toml has no top-level version"
    return m.group(1)


def test_the_package_and_pyproject_agree():
    assert mousereach.__version__ == _declared_version(), (
        "bump both, or the manifests will record a version that does not exist"
    )


def test_the_version_is_not_a_placeholder():
    assert mousereach.__version__ not in ("", "0.0.0", "0.0.0+unpackaged")


def test_the_manifest_stamps_this_version():
    """If this stops being true the drift stops mattering -- and so does this test."""
    import inspect
    from mousereach.pipeline import manifest
    assert "mousereach.__version__" in inspect.getsource(
        manifest.create_processing_manifest)
