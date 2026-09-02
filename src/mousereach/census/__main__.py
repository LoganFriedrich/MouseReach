"""``python -m mousereach.census`` -- the census CLI without the console shim.

Exists so integrators can invoke the package itself (``-m mousereach.census``)
rather than ``-m mousereach.census.runner``; the latter trips a runpy
RuntimeWarning because ``census/__init__`` already imports ``runner``.
"""
from .runner import main

raise SystemExit(main())
