"""Which piece of work the watcher takes next: the ordering policy, from config.

WHY THIS MODULE EXISTS
----------------------
A node with a queue of work has to choose what to run next. Two facts about
that choice are local to a lab and must never be compiled into the tool:

  1. Some tray types are worth processing, but LAST. On the Easy (E) and Flat
     (F) trays the pellet divot sits level with the scoring area, so a
     displaced pellet can simply be retried and a per-pellet outcome is not a
     meaningful unit of behaviour. Those sessions are still wanted -- for
     session-level measures -- but a machine that could be running a Pillar (P)
     session must never take one on. That is not a mild preference; it is
     "only when there is nothing else at all".
  2. Which PROJECT goes first when two projects are queued. A lab running one
     project has no such rule; a lab running three may have a strict one.

So this module ships a default that is a property of the TOOL -- the tray types
it fully supports (config.FilePatterns.SUPPORTED_TRAY_TYPES) go first, the ones
it does not (UNSUPPORTED_TRAY_TYPES) wait for an idle node -- and reads
everything else from ``watcher.work_priority`` in ~/.mousereach/config.json.
No project name appears anywhere in this file, and none may be added: the
placeholders below (PROJECT_A, PROJECT_B) are examples of the SHAPE, not of
anybody's policy.

THE CONFIG SHAPE
----------------
::

    "watcher": {
      "work_priority": {
        "order": [{"project": ["PROJECT_A"]}, {"project": ["PROJECT_B"]}],
        "idle_only": [{"tray_type": ["E", "F"]}]
      }
    }

``order``
    A list of SELECTORS tried in turn. The first selector that matches at
    least one waiting item wins, and the choice is made among its matches
    only. Items that match no selector form an implicit final tier, so a
    configured order can never make work unreachable -- only later. ``order``
    is preferential; it never excludes.

``idle_only``
    A list of selectors naming work that is not TAKEN ON until the node has
    nothing else to do. It is exclusionary, and it applies to the buckets
    that ADMIT new work onto a node (see the ADMIT/DRAIN split in
    orchestrator._select_work_item). It deliberately does NOT stop a node
    finishing a video it already holds: that would strand the video, because
    a node with a backlog is never idle. Deferred work already on the disk is
    still picked last WITHIN its bucket, by ``order``.

SELECTOR SEMANTICS (exact)
--------------------------
A selector is an object mapping FIELD -> list of accepted values.

  * A selector matches an item when EVERY field it names matches.
  * A field matches when the item's resolved label set for that field shares at
    least one value with the list. Comparison is case-insensitive.
  * An item whose label set for a named field is EMPTY never matches. Unknown
    is not quietly treated as membership -- a video whose tray cannot be
    determined is not "an E tray".
  * ``{}`` matches everything. It is the honest way to write "everything else"
    as an explicit final tier.

Known fields, and how each is resolved:

  =============== =========================================================
  tray_type       DB column, else the filename (see effective_tray_type)
  project         experiment column and/or the animal id(s); matches both
                  the experiment code and the project folder name that
                  config.AnimalID maps it to, so either spelling selects the
                  same videos, and a short lettered cohort id is selected by
                  its project without anyone listing the letters
  cohort          cohort column and/or the animal id(s); matches the bare
                  cohort number, the code+number, and the archive's cohort
                  folder
  animal_id       animal_id / animal_ids column, else the filename
  state           DB column only
  reprocess_scope DB column only
  =============== =========================================================

WHY A DEFAULT AND NOT A HARD STOP
---------------------------------
This project's rule is that an unset value stops with a message naming the
command to run. That rule is about LOCATIONS -- a path has no safe substitute,
and guessing one silently writes data somewhere wrong. This key is a
PREFERENCE. Its safe substitute is exactly what the watcher did before the key
existed (Pillar first), so refusing to start over an unset preference would
stop the lab's data for no gain. The key is therefore optional, and instead of
stopping we are loud: the resolved policy is logged at startup, printed by
``--dry-run``, and every malformed entry is named in a warning that says which
file to edit. Nothing here fails silently and nothing here fails closed.
"""

import json
import logging
import os
import random
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from mousereach.config import AnimalID, FilePatterns, parse_tray_type

logger = logging.getLogger(__name__)

CONFIG_HINT = ("edit the 'watcher.work_priority' section of "
               "~/.mousereach/config.json (docs/WORK_PRIORITY.md) and restart "
               "the watcher")

# The shipped default. It reproduces exactly what the code did before this
# module existed (Pillar-first), plus the deferral this module adds. NO
# project preference is shipped: that is the lab fact, and it belongs only in
# a lab's own config file.
DEFAULT_ORDER = [{"tray_type": list(FilePatterns.SUPPORTED_TRAY_TYPES)}]
DEFAULT_IDLE_ONLY = [{"tray_type": list(FilePatterns.UNSUPPORTED_TRAY_TYPES)}]

KNOWN_FIELDS = ("tray_type", "project", "cohort", "animal_id", "state",
                "reprocess_scope")

KNOWN_SETTINGS = ("order", "idle_only")


# =============================================================================
# FIELD RESOLVERS -- the DB column is never trusted on its own
# =============================================================================

def _text(value) -> str:
    return str(value).strip().upper() if value is not None else ""


def _name_of(item: dict) -> str:
    """The filename an item is known by: a collage's filename, or a video id."""
    return item.get("filename") or item.get("video_id") or ""


def effective_tray_type(item: dict, is_collage: bool = False) -> Optional[str]:
    """Tray letter for a work item: DB column first, then its own filename.

    The column cannot be trusted on its own. A video that comes back from a
    human review queue is re-registered with no metadata at all, so its
    tray_type is NULL -- on a node with a review backlog that is most of the
    reprocess pool, not an edge case. A filter that reads only the column
    silently stops filtering on those rows and degrades to picking at random,
    which is precisely the failure this resolver exists to close.

    Order: tray_type column, then a collage's tray_suffix ('P1' -> 'P'), then
    the filename via config.parse_tray_type. Returns None when nothing names a
    tray; callers must treat None as "unknown", never as a particular tray.
    """
    col = _text(item.get("tray_type"))
    if not col:
        col = _text(item.get("tray_suffix"))[:1]
    if col in FilePatterns.TRAY_TYPES:
        return col
    name = _name_of(item)
    if name:
        # parse_tray_type takes a filename; a bare video id has no extension
        # and Path().stem would otherwise eat a trailing '.P1'-like segment.
        if not Path(name).suffix:
            name = name + ".mp4"
        parsed = parse_tray_type(name).get("tray_type")
        if parsed:
            return parsed
    # An unrecognised but non-empty column value is still what the database
    # says; return it so it can be selected on, rather than vanishing.
    return col or None


def animal_ids(item: dict) -> List[str]:
    """Every animal id on an item: column first, else the filename.

    Collages carry several (animal_ids, comma separated) because one collage is
    cropped into one video per animal.
    """
    raw = item.get("animal_id") or item.get("animal_ids") or ""
    ids = [a.strip() for a in str(raw).split(",") if a.strip()]
    if ids:
        return ids
    stem = Path(_name_of(item)).stem
    parts = stem.split("_")
    if len(parts) >= 3:  # {date}_{animal[,animal...]}_{tray}{n}
        return [a.strip() for a in parts[1].split(",") if a.strip()]
    return []


def project_labels(item: dict) -> set:
    """Every label that legitimately names this item's project.

    Both the experiment code and the project folder are returned, so a lab may
    write either one in its config, and short lettered cohort ids are covered
    by their project name without anyone listing the letters. A collage matches
    a project when ANY of its animals does: it has no single project, and it is
    cropped into per-animal videos that each get judged on their own.
    """
    labels = set()
    declared = _text(item.get("experiment"))
    if declared:
        labels.add(declared)
    for animal in animal_ids(item):
        try:
            project, cohort_folder = AnimalID.get_project_and_cohort(animal)
        except Exception:
            continue
        if project and project != "UNKNOWN":
            labels.add(_text(project))
        code = AnimalID.parse(animal).get("experiment") or ""
        if not code and cohort_folder and cohort_folder != "UNKNOWN":
            code = cohort_folder
        if code:
            labels.add(_text(code))
    return labels


def cohort_labels(item: dict) -> set:
    """Cohort labels: the bare number, the code+number, the archive folder."""
    labels = set()
    declared = _text(item.get("cohort"))
    if declared:
        labels.add(declared)
    for animal in animal_ids(item):
        parsed = AnimalID.parse(animal)
        code = _text(parsed.get("experiment"))
        num = _text(parsed.get("cohort"))
        if num:
            labels.add(num)
            if code:
                labels.add(code + num)
        try:
            _project, cohort_folder = AnimalID.get_project_and_cohort(animal)
        except Exception:
            cohort_folder = ""
        if cohort_folder and cohort_folder != "UNKNOWN":
            labels.add(_text(cohort_folder))
    return labels


def field_labels(item: dict, field: str, is_collage: bool = False) -> set:
    """The set of labels an item presents for one selector field."""
    if field == "tray_type":
        tray = effective_tray_type(item, is_collage=is_collage)
        return {tray} if tray else set()
    if field == "project":
        return project_labels(item)
    if field == "cohort":
        return cohort_labels(item)
    if field == "animal_id":
        return {_text(a) for a in animal_ids(item)}
    value = _text(item.get(field))
    return {value} if value else set()


def selector_matches(selector: dict, item: dict, is_collage: bool = False) -> bool:
    """True when every field the selector names matches this item."""
    for field, wanted in selector.items():
        labels = field_labels(item, field, is_collage=is_collage)
        if not labels:
            return False
        if not labels & {_text(w) for w in wanted}:
            return False
    return True


# =============================================================================
# POLICY
# =============================================================================

def _render(selector: dict) -> str:
    if not selector:
        return "anything"
    return " and ".join(
        "%s in [%s]" % (field, ", ".join(str(v) for v in values))
        for field, values in sorted(selector.items()))


class WorkPriority:
    """A resolved, validated ordering policy. Built once, reused every cycle."""

    def __init__(self, order, idle_only, complaints, source):
        self.order = order
        self.idle_only = idle_only
        self.complaints = complaints
        self.source = source            # "config" or "default"

    def tier(self, item: dict, is_collage: bool = False) -> int:
        """Index of the first selector in `order` matching this item.

        Items matching nothing land in an implicit LAST tier, len(order), so
        no configuration can make a piece of work unreachable -- only later.
        """
        for index, selector in enumerate(self.order):
            if selector_matches(selector, item, is_collage=is_collage):
                return index
        return len(self.order)

    def is_deferred(self, item: dict, is_collage: bool = False) -> bool:
        """True when this item is not taken ON until the node is idle.

        Deferral applies to the buckets that ADMIT work onto a node. It never
        stops a node finishing a video it already holds -- see the ADMIT/DRAIN
        split in orchestrator._select_work_item for why that distinction is
        the whole design.
        """
        return any(selector_matches(s, item, is_collage=is_collage)
                   for s in self.idle_only)

    def describe(self) -> List[str]:
        """Operator-readable summary. ASCII only -- this reaches a terminal."""
        lines = []
        if self.order:
            lines.append("order: " + " then ".join(_render(s) for s in self.order)
                         + " then everything else")
        else:
            lines.append("order: no preference")
        if self.idle_only:
            lines.append("not taken on until nothing else is waiting: "
                         + ", ".join(_render(s) for s in self.idle_only))
        else:
            lines.append("not taken on until nothing else is waiting: nothing")
        lines.append("work already on this node is always finished, in the "
                     "order above")
        lines.append("source: %s" % self.source)
        return lines


def _parse_selectors(raw, key: str, complaints: List[str]):
    """Validate one selector list. Returns None to mean "use the default"."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        complaints.append(
            "watcher.work_priority.%s must be a list of selectors, got %s -- "
            "using the shipped default" % (key, type(raw).__name__))
        return None
    parsed = []
    for i, selector in enumerate(raw):
        if not isinstance(selector, dict):
            complaints.append(
                'watcher.work_priority.%s[%d] must be an object like '
                '{"tray_type": ["P"]}, got %s -- ignoring it'
                % (key, i, type(selector).__name__))
            continue
        clean = {}
        for field, values in selector.items():
            if field not in KNOWN_FIELDS:
                complaints.append(
                    "watcher.work_priority.%s[%d]: unknown field '%s' -- "
                    "ignoring this selector. Known fields: %s"
                    % (key, i, field, ", ".join(KNOWN_FIELDS)))
                clean = None
                break
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, (list, tuple)) or not values:
                complaints.append(
                    "watcher.work_priority.%s[%d].%s must be a non-empty list "
                    "of values -- ignoring this selector" % (key, i, field))
                clean = None
                break
            clean[field] = [str(v) for v in values]
        if clean is not None:
            parsed.append(clean)
    if raw and not parsed:
        # Somebody wrote a list and every entry of it was unusable. Running
        # with NO ordering is not what they meant and is worse than the
        # default, so fall back and say so. An explicitly empty list ([]) is a
        # different statement -- "no preference" -- and is honoured above.
        complaints.append(
            "watcher.work_priority.%s: none of its %d entries could be used -- "
            "using the shipped default" % (key, len(raw)))
        return None
    return parsed


def load_policy(raw) -> WorkPriority:
    """Turn the raw config value into a policy. Never raises, never stops.

    A malformed entry is dropped and named; an unusable key falls back to the
    shipped default and says so. The complaints ride along on the returned
    object so callers can log them AND show them in --dry-run.
    """
    complaints: List[str] = []
    source = "config"
    if raw is None:
        raw = {}
        source = "default (watcher.work_priority is not set)"
    elif not isinstance(raw, dict):
        complaints.append("watcher.work_priority must be an object, got %s -- "
                          "using the shipped default" % type(raw).__name__)
        raw = {}
        source = "default (watcher.work_priority is unusable)"

    # A setting this version does not know is almost always a typo or a
    # setting that has been removed. Silently ignoring it is how somebody
    # spends a week believing a policy is in force that is not.
    for key in sorted(raw):
        if key not in KNOWN_SETTINGS:
            complaints.append(
                "watcher.work_priority.%s is not a setting this version "
                "understands -- ignoring it. Known settings: %s"
                % (key, ", ".join(KNOWN_SETTINGS)))

    order = _parse_selectors(raw.get("order"), "order", complaints)
    idle_only = _parse_selectors(raw.get("idle_only"), "idle_only", complaints)

    return WorkPriority(
        order=[dict(s) for s in DEFAULT_ORDER] if order is None else order,
        idle_only=[dict(s) for s in DEFAULT_IDLE_ONLY] if idle_only is None else idle_only,
        complaints=complaints,
        source=source,
    )


def load_and_announce(raw) -> WorkPriority:
    """load_policy, plus the one-time log every operator needs to see."""
    policy = load_policy(raw)
    for complaint in policy.complaints:
        logger.warning("%s. To fix: %s", complaint, CONFIG_HINT)
    for line in policy.describe():
        logger.info("Work priority -- %s", line)
    return policy


# =============================================================================
# THE LAB-WIDE ORDER -- one file, on the shared drive, read by every tool
# =============================================================================
#
# WHY A SHARED FILE. The selectors above are expressive but per-machine:
# they live in each node's ~/.mousereach/config.json. The queues they order
# are shared, so two machines could disagree about what mattered and nothing
# said so -- and the human review tools read no policy at all, so a reviewer
# was handed a random video while the processing node worked strictly by
# preference. One file on the shared drive, beside pipeline_versions.json,
# is what makes "set it once" true.
#
# WHY TWO LISTS RATHER THAN SELECTORS. What a lab actually wants to say is
# "this project before that one, and within it these cohorts first". Writing
# that as selectors is easy to get subtly wrong, so the file holds the two
# ordered lists and this module compiles them into selectors. Raw `order` /
# `idle_only` are still accepted for anything the lists cannot express.

LAB_PRIORITY_SCHEMA = "1.0"
LAB_PRIORITY_FILENAME = "priority_order.json"
LAB_PRIORITY_HINT = ("set the order in the MouseReach window (Watcher Control "
                     "-> Work priority), or edit priority_order.json at the "
                     "root of the shared pipeline folder")

_LAB_CACHE_TTL_S = 60.0
# What is cached is the FILE, not the resolved policy. Caching the policy was
# wrong in a way tests caught immediately: the answer also depends on the
# caller's own machine setting, so within the cache window a second caller got
# the first caller's policy no matter what it asked for. Reading the shared
# drive is the expensive part; building the policy from it is not.
_LAB_RAW: Optional[dict] = None
_LAB_RAW_AT: float = 0.0
_LAB_RAW_SEEN: bool = False


def lab_priority_path(nas_root=None) -> Optional[Path]:
    """The shared order file, or None when no shared drive is configured."""
    if nas_root is None:
        from mousereach.config import Paths
        nas_root = Paths.NAS_ROOT
    return (Path(nas_root) / LAB_PRIORITY_FILENAME) if nas_root else None


def compile_order(projects=None, cohorts=None, tray_types=None) -> List[dict]:
    """Turn "projects first, cohorts within them" into ordering selectors.

    Most specific first: each project's named cohorts, then the rest of that
    project, then the next project. A preferred tray letter is folded into
    every tier so Pillar work leads within each one, and a bare tray tier is
    appended so Pillar still outranks an unnamed project.

    Cohorts are emitted both bare ("01") and qualified ("CNT01") because a
    video's cohort resolves to both (see cohort_labels), and a lab may write
    either.
    """
    out: List[dict] = []
    seen = set()
    tray = [t for t in (tray_types or []) if str(t).strip()]

    def add(selector: dict):
        key = tuple(sorted((f, tuple(v)) for f, v in selector.items()))
        if key not in seen:
            seen.add(key)
            out.append(selector)

    for project in (projects or []):
        project = str(project).strip()
        if not project:
            continue
        for cohort in ((cohorts or {}).get(project) or []):
            cohort = str(cohort).strip()
            if not cohort:
                continue
            labels = [cohort] if cohort.upper().startswith(project.upper()) \
                else [cohort, f"{project}{cohort}"]
            selector = {"project": [project], "cohort": labels}
            if tray:
                selector["tray_type"] = list(tray)
            add(selector)
        selector = {"project": [project]}
        if tray:
            selector["tray_type"] = list(tray)
        add(selector)
    if tray:
        add({"tray_type": list(tray)})
    return out


def read_lab_priority(nas_root=None) -> Optional[dict]:
    """The shared order file's contents, or None when absent/unreadable."""
    path = lab_priority_path(nas_root)
    if not path:
        return None
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Lab priority file %s could not be read (%s); falling "
                       "back to this machine's own setting", path, e)
        return None


def save_lab_priority(projects=None, cohorts=None, tray_types=None,
                      idle_only_tray_types=None, nas_root=None,
                      updated_by=None, notes=None) -> Path:
    """Write the shared order file. Atomic, so a node reading it mid-write
    never sees half a file (the version declaration learned that the hard
    way). Returns the path written."""
    path = lab_priority_path(nas_root)
    if not path:
        raise ValueError("no shared drive configured, so there is nowhere to "
                         "put the lab-wide order")
    body = {
        "schema_version": LAB_PRIORITY_SCHEMA,
        "updated_at": datetime.now().isoformat(),
        "updated_by": updated_by or socket.gethostname(),
        "projects": [str(p).strip() for p in (projects or []) if str(p).strip()],
        "cohorts": {str(k): [str(c).strip() for c in (v or []) if str(c).strip()]
                    for k, v in (cohorts or {}).items()},
        "tray_types": [str(t).strip() for t in (tray_types or []) if str(t).strip()],
        "idle_only_tray_types": [str(t).strip() for t in (idle_only_tray_types or [])
                                 if str(t).strip()],
    }
    if notes:
        body["notes"] = str(notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    invalidate_lab_policy()
    logger.info("Lab priority order saved to %s", path)
    return path


def _policy_from_lab_file(raw_file: dict) -> WorkPriority:
    """Build a policy from the shared file, compiling the two lists unless it
    carries raw selectors."""
    if raw_file.get("order") is not None or raw_file.get("idle_only") is not None:
        policy = load_policy({k: raw_file[k] for k in ("order", "idle_only")
                              if raw_file.get(k) is not None})
        policy.source = "lab file (raw selectors)"
        return policy
    order = compile_order(raw_file.get("projects"),
                          raw_file.get("cohorts"),
                          raw_file.get("tray_types") or
                          list(FilePatterns.SUPPORTED_TRAY_TYPES))
    idle = [{"tray_type": list(raw_file["idle_only_tray_types"])}] \
        if raw_file.get("idle_only_tray_types") else None
    policy = load_policy({"order": order} if idle is None
                         else {"order": order, "idle_only": idle})
    policy.source = "lab file (%s)" % LAB_PRIORITY_FILENAME
    return policy


_UNSET = object()


def load_lab_policy(local_raw=_UNSET, nas_root=None,
                    max_age_s: Optional[float] = None) -> WorkPriority:
    """The one ordering every tool should use.

    Shared file first, then this machine's own watcher.work_priority, then
    the shipped default. Only the FILE READ is cached (a minute); pass
    max_age_s=0 to read now. Never raises.

    Passing local_raw=None means "this caller HAS no local setting", and the
    shipped default follows. That is different from not passing it at all,
    which means "go and look this machine's config up for me". Collapsing the
    two let this function answer a caller from a source the caller had already
    ruled out -- which quietly dropped the Pillar-first default.
    """
    try:
        raw_file = _cached_lab_raw(nas_root, max_age_s)
        if raw_file:
            return _policy_from_lab_file(raw_file)
        if local_raw is _UNSET:
            try:
                from mousereach.config import WatcherConfig
                local_raw = WatcherConfig.load().work_priority
            except Exception:
                local_raw = None
        policy = load_policy(local_raw)
        if local_raw:
            policy.source += " (this machine only -- no lab file yet)"
        return policy
    except Exception as e:                       # never let ordering stop work
        logger.warning("Lab priority could not be resolved (%s); using the "
                       "shipped default", e)
        return load_policy(None)


def _cached_lab_raw(nas_root=None, max_age_s: Optional[float] = None):
    """The shared file's contents, re-read at most once a minute.

    Only the READ is cached. The resolved policy is not, because it also
    depends on the caller's own machine setting -- caching that handed a
    second caller the first caller's answer.
    """
    global _LAB_RAW, _LAB_RAW_AT, _LAB_RAW_SEEN
    ttl = _LAB_CACHE_TTL_S if max_age_s is None else max_age_s
    # ttl <= 0 means "read it now"; comparing elapsed time alone would not,
    # because this clock's resolution makes two calls in one tick look
    # simultaneous.
    fresh = (_LAB_RAW_SEEN and ttl > 0
             and time.monotonic() - _LAB_RAW_AT <= ttl)
    if not fresh:
        _LAB_RAW = read_lab_priority(nas_root)
        _LAB_RAW_AT = time.monotonic()
        _LAB_RAW_SEEN = True
    return _LAB_RAW


def invalidate_lab_policy() -> None:
    """Forget the cached file (after a save, or in a test)."""
    global _LAB_RAW, _LAB_RAW_AT, _LAB_RAW_SEEN
    _LAB_RAW, _LAB_RAW_AT, _LAB_RAW_SEEN = None, 0.0, False


# --- using it on a list -----------------------------------------------------

def as_item(obj) -> dict:
    """Adapt whatever a caller holds -- a bundle folder, a video id, a DB row
    -- into the dict the resolvers above understand."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, Path):
        return {"video_id": obj.name}
    return {"video_id": str(obj)}


def order_items(items, key=None, policy=None, is_collage: bool = False) -> list:
    """``items`` sorted by preference, stable within a tier.

    Use where the order itself is the product (a queue listing, a drain that
    takes the first N). Where one item is being handed to a person, prefer
    best_tier_choice, which keeps the sampling unbiased inside a tier.
    """
    policy = policy or load_lab_policy()
    to_item = key or as_item
    return [obj for _, _, obj in sorted(
        ((policy.tier(to_item(o), is_collage=is_collage), i, o)
         for i, o in enumerate(items)),
        key=lambda t: (t[0], t[1]))]


def best_tier_choice(items, key=None, policy=None, rng=None,
                     is_collage: bool = False):
    """One item from the most-preferred non-empty tier, chosen at random.

    This is the reconciliation the review tools need: the lab's order decides
    WHICH tier a reviewer works through, and the long-standing random pick --
    there to keep the reviewed set unbiased across cohorts and days -- still
    decides which video inside that tier. Returns None for an empty list.
    """
    items = list(items)
    if not items:
        return None
    policy = policy or load_lab_policy()
    to_item = key or as_item
    tiers = [policy.tier(to_item(o), is_collage=is_collage) for o in items]
    best = min(tiers)
    candidates = [o for o, t in zip(items, tiers) if t == best]
    return (rng or random).choice(candidates)
