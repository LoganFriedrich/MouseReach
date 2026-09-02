"""Resolve every expected session to the element it currently occupies.

WHY THIS EXISTS
---------------
Given a denominator (see ``expected_sessions``), a census needs one element per
session, and a count per element with that denominator attached.

THE ELEMENTS, in pipeline order. These are the operator's mental model, not the
code's internal states, and the difference matters -- a person asking "where is
everything" is asking about these:

  unanalyzed    recorded, but nothing has been done to it yet
  crop_dlc      cropping or pose estimation -- RUNNING OR DONE
  mousereach    the analysis algorithms -- RUNNING OR DONE
  triage        held for a person, quick per-element questions
  deep_review   held for a person, segmentation failed or was escalated
  quarantined   held out as unprocessable (malformed name, unsupported content)
  analyzed      finished: see the THREE conditions below
  session_only  a legitimate terminal state for outcome-free trays

THE INVARIANT: nothing should sit between ``mousereach`` and ``analyzed``.
A session whose analysis has finished must be either held for a person or
finished. Anything else is in a gap it has no route out of, and the count of
those is the single most useful number this module produces -- it is an assertion
that should read zero, not a bucket that is expected to have members.

"ANALYZED" IS THREE CONDITIONS, NOT ONE
---------------------------------------
Done with the analysis, AND present in the database, AND in the final output
folder. Testing only the output files overstates it: outputs can exist for a
session that never reached the database and never will. Database membership
arrives as an injected callable so that pipeline-side code keeps no database
dependency -- the tool that owns the pipeline must not import the tool that owns
the database.

TRANSIENTS ARE NOT STATES
-------------------------
Some reasons a session sits in the gap resolve themselves as the pipeline works
through its backlog. Those must be reported as a REASON attached to a live
membership, never as their own terminal category -- otherwise the dashboard
accumulates permanent false alarms as the pipeline heals. Evaluate live; never
cache a violation.

No lab paths, hostnames or project names appear in this module.
"""
from __future__ import annotations

import re


# Order matters: later beats earlier when a session is visible in several places.
# A queue outranks everything, including the final output tree, because bundles
# are routinely staged FROM that tree -- the same session legitimately exists in
# both and the operator needs to see the hold, not the copy.
ELEMENT_RANK = {
    "unanalyzed": 0,
    "crop_dlc": 10,
    "mousereach": 20,
    "analyzed": 30,
    "session_only": 30,
    "deep_review": 40,
    "triage": 40,
    "quarantined": 40,
}

ELEMENT_ORDER = ["unanalyzed", "crop_dlc", "mousereach", "triage", "deep_review", "quarantined",
                 "analyzed", "session_only"]


class DatabaseViewUnavailable(RuntimeError):
    """Raised when something asks a question only the database can answer.

    WHY THIS IS AN EXCEPTION AND NOT A FLAG: a flag has to be checked, and display
    code drops flags. A stale database on this system answered plausibly instead of
    refusing, and it cost two wrong conclusions before anyone noticed -- see trap 11.
    A caller with no database view must not be able to render a "finished" count or
    a gap count at all, and the only way to guarantee that is to refuse structurally
    rather than to return a number with a caveat attached to it.
    """


UNAVAILABLE = None   # what a count is when it cannot be computed -- never 0


def tray_from_stem(stem, db_tray=None):
    """Effective tray letter: the database value if present, else the filename.

    WHY the fallback is not optional: on a live corpus the tray column was NULL on
    967 of 1,216 rows in one state alone, because sessions returning from a review
    queue were re-registered without their metadata. Code that trusts the column
    silently stops distinguishing trays and degrades to arbitrary ordering.
    """
    if db_tray:
        return db_tray
    m = re.search(r"_([A-Z])\d+$", stem)
    return m.group(1) if m else None


def _has_all(index, stem, required):
    return bool(required) and set(required).issubset(index.get(stem) or set())


def _has_any(index, stem, wanted):
    return bool((index.get(stem) or set()) & set(wanted))


def resolve_elements(expected, analyzed_index, locations, in_database=None,
                     outcome_free_trays=(), analysis_outputs=(),
                     session_only_outputs=(), partial_analysis_outputs=()):
    """Return {session_id: element} for every expected session.

    ``expected``        {session_id: info} from expected_sessions().
    ``analyzed_index``  {session_id: set(output suffixes present in the final tree)}.
    ``locations``       {element_name: set(session_ids)} for the non-final places.
                        Recognised keys: crop_dlc, mousereach, triage, deep_review.
    ``in_database``     callable(session_id) -> bool. REQUIRED for a session to be
                        called analyzed. When omitted, no session can reach
                        ``analyzed`` and everything finished lands in the gap --
                        which is the honest answer for a caller that cannot see the
                        database, rather than a flattering one.
    ``analysis_outputs`` suffixes that together mean the analysis finished.
    ``session_only_outputs`` suffixes that mean an outcome-free tray finished.
    ``partial_analysis_outputs`` suffixes that mean analysis has STARTED.
    """
    elements = {}
    for sid, info in expected.items():
        best, rank = "unanalyzed", -1

        for name, members in locations.items():
            if sid in members:
                r = ELEMENT_RANK.get(name, 0)
                if r > rank:
                    best, rank = name, r

        tray = (info or {}).get("tray") or tray_from_stem(sid)
        outcome_free = tray in outcome_free_trays

        finished = (_has_all(analyzed_index, sid, session_only_outputs) if outcome_free
                    else _has_all(analyzed_index, sid, analysis_outputs))

        if finished:
            in_db = bool(in_database(sid)) if in_database else False
            if in_db:
                terminal = "session_only" if outcome_free else "analyzed"
                if ELEMENT_RANK[terminal] > rank:
                    best, rank = terminal, ELEMENT_RANK[terminal]
            elif ELEMENT_RANK["mousereach"] > rank:
                # Analysis finished but the result has not landed. This is the gap.
                best, rank = "mousereach", ELEMENT_RANK["mousereach"]
        elif _has_any(analyzed_index, sid, partial_analysis_outputs) \
                and ELEMENT_RANK["mousereach"] > rank:
            best, rank = "mousereach", ELEMENT_RANK["mousereach"]

        elements[sid] = best
    return elements


def invariant_violations(expected, elements, analyzed_index, in_database=None,
                         outcome_free_trays=(), analysis_outputs=(),
                         session_only_outputs=(), reason_fn=None):
    """Sessions whose analysis has finished but which are neither held nor landed.

    Returns {session_id: reason}. THIS SHOULD BE EMPTY. It is an assertion, not a
    bucket: report it as "this number must be zero, it is N, here is which ones",
    never as another row in a table of counts.

    ``reason_fn(session_id)`` may return a short string explaining why a session is
    stuck. Supply one that distinguishes TRANSIENT causes -- ones the pipeline
    resolves on its own as it works through its backlog -- from ones needing a
    person. Do not encode a transient cause as its own element; it heals, and a
    cached violation becomes a permanent false alarm.

    RAISES DatabaseViewUnavailable when ``in_database`` is None. Without it every
    finished session looks unlanded, so the "violations" would be the entire
    finished corpus -- a number that is not merely wrong but wrong in the alarming
    direction. Refusing is the only safe answer; a caller that cannot see the
    database cannot evaluate this invariant at all.
    """
    if in_database is None:
        raise DatabaseViewUnavailable(
            "the finished-but-not-landed invariant needs a database view; "
            "pass in_database=callable(session_id)->bool, or do not report this number"
        )
    out = {}
    for sid, element in elements.items():
        if element != "mousereach":
            continue
        tray = (expected.get(sid) or {}).get("tray") or tray_from_stem(sid)
        outcome_free = tray in outcome_free_trays
        required = session_only_outputs if outcome_free else analysis_outputs
        if not _has_all(analyzed_index, sid, required):
            continue          # still running -- legitimately in the element
        if in_database and in_database(sid):
            continue          # landed; not a violation
        out[sid] = reason_fn(sid) if reason_fn else "finished but not landed"
    return out


def tally(expected, elements, project_of=None, tray_of=None, database_view=True):
    """Counts per element, and per (project, tray, element), with denominators.

    ``project_of`` and ``tray_of`` are callables so that no project name or tray
    convention appears in this module -- both are deployment facts, not code facts.

    ``database_view=False`` means the caller could not evaluate database membership.
    The ``analyzed`` and ``session_only`` counts are then set to UNAVAILABLE (None),
    NOT to zero, and a caveat is attached naming what is missing. None is chosen
    deliberately: a display that formats it prints "None" or raises, both of which a
    person notices, whereas a zero reads as a fact and is indistinguishable from
    "nothing is finished".
    """
    by_element = {}
    by_group = {}
    for sid, element in elements.items():
        by_element[element] = by_element.get(element, 0) + 1
        proj = project_of(sid) if project_of else "all"
        tray = (tray_of(sid) if tray_of
                else ((expected.get(sid) or {}).get("tray") or tray_from_stem(sid)))
        bucket = by_group.setdefault((proj, tray), {})
        bucket[element] = bucket.get(element, 0) + 1

    ordered = {k: by_element.get(k, 0) for k in ELEMENT_ORDER if by_element.get(k)}
    for k, v in by_element.items():
        ordered.setdefault(k, v)

    caveats = []
    if not database_view:
        for terminal in ("analyzed", "session_only"):
            ordered[terminal] = UNAVAILABLE
        caveats.append(
            "analyzed: UNAVAILABLE (no database view). Sessions whose analysis has "
            "finished are counted under 'mousereach' because the third condition -- "
            "present in the database -- could not be checked. Do not read this as "
            "'nothing is finished', and do not report a gap count from this run."
        )

    return {
        "total": len(elements),
        "by_element": ordered,
        "by_project_tray": {"%s|%s" % (p, t): v for (p, t), v in by_group.items()},
        "database_view": database_view,
        "caveats": caveats,
    }
