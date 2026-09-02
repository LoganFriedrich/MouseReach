"""Pipeline census -- where every expected session is, against the full workload.

Three generic modules (no lab paths, no GUI imports, unit-testable bare):

* ``expected_sessions`` -- the denominator: every single-animal session that
  SHOULD exist, from collage expansion unioned with found artifacts.
* ``locate_sessions``   -- one element per session, the finished-but-not-landed
  invariant, and tallies with denominators.
* ``review_completeness`` -- how much of a human review is actually done
  (from ``human.outcome``, never ``answers.reviewed``).

``runner`` is the wiring: the only file here that knows the pipeline's folder
layout (through ``mousereach.config.Paths``). It adds throughput measurement
and completion estimates, and is the ``mousereach-census`` entry point --
the headless JSON interface an integrator (e.g. a database tool's GUI) calls.
"""
from .expected_sessions import (  # noqa: F401
    expected_sessions, classify_trailer, session_key, tray_of,
    collect_collages, select_sources,
)
from .locate_sessions import (  # noqa: F401
    resolve_elements, invariant_violations, tally, tray_from_stem,
    DatabaseViewUnavailable, ELEMENT_ORDER, ELEMENT_RANK, UNAVAILABLE,
)
from .review_completeness import (  # noqa: F401
    completeness_of_document, completeness_for_stem, scan_queue, summarise,
)
from .runner import run_census  # noqa: F401
