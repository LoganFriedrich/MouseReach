"""What a person can do, from the Pipeline Dashboard or the Review Queues
tab, about a review bundle that a route left without all its files, and
about a queue folder left for a person.

WHY THIS EXISTS
---------------
A route out of a processing folder goes on when one file cannot be moved
(most often the video, still held open) and lists it in the bundle's routing
record (review_routing.move_video_bundle). The watcher's return scan moves
such a file in by itself once it can (review_return._finish_failed_moves);
what it cannot finish is listed in the dashboard's red "Needs your attention"
panel as a ``route_unfinished`` item (review_return._write_unfinished_attention).
For those items the dashboard's "Resolve..." window offers buttons
(``actions_for``), and ``perform`` carries one out:

  * "Try to move the missing files in now" -- the watcher's own step, at once;
  * "Accept the bundle without <file>" -- for a file that is gone for good:
    crosses it off the bundle's list of missing files (the record keeps who
    accepted it, and when). Nothing is moved or deleted.

When a file is BOTH in the bundle and at its old place and the two differ,
no button is offered: which copy is right is a person's call, and the
machine never deletes or replaces a copy that may be the only good one.
The line says where both copies are.

A leftover queue folder kept for a person (``left_for_a_person``: it holds
review_routing.LEFT_NOTE, because something in it has no current copy
anywhere else) has one button:

  * "Nothing in it is needed: delete this folder" -- deletes the folder and
    everything in it (review_routing.delete_leftover_folder). The machine
    never deletes a person's work; this is a person deciding to.

A route refused because of a queue folder (``queue_folder_in_the_way``)
gets the same button when that folder is such a leftover. When the folder
is the video's LIVE bundle there is no button: finish its review in the
review window, and the route is tried again once it has returned.

The Review Queues tab lists every leftover in the queues ('Folders left for
a person'), read from the folders themselves (review_routing.
leftover_folders), with these same buttons. Other items (a return that would
have replaced a person's file) are listed with what to do and have no
buttons: each goes away by itself once the file in question is gone.

ASCII-only log text (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import logging
import socket
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (key, button label, what it does -- shown before the person confirms)
Action = Tuple[str, str, str]


class ActionRefused(RuntimeError):
    """The action cannot be carried out; the message says why and what to do.
    Nothing was changed."""


def drop_item(item: dict) -> None:
    """Take ``item`` off the attention list it was read from (``listed_in``,
    added by attention.read_items). The watchers also drop an item by
    themselves once it is resolved; this makes the panel current at once.
    Never raises."""
    from .review_routing import attention_dir
    name = item.get("listed_in")
    d = attention_dir()
    if not name or d is None:
        return
    target = d / name
    try:
        from mousereach.review.causal_review_io import _locked_update

        def _apply(doc):
            doc["items"] = [i for i in (doc.get("items") or [])
                            if not (i.get("kind") == item.get("kind")
                                    and i.get("video_id") == item.get("video_id")
                                    and i.get("folder") == item.get("folder"))]
        if target.exists():
            _locked_update(target, _apply, wait_s=10.0)
    except Exception as e:
        logger.warning("Could not update the attention list %s: %s", name, e)


def _source_not_held(bundle: Path, vid: str, db) -> str:
    """Why a file at an unfinished route's source may NOT be touched from the
    dashboard now ("" when it may): the same rule the return scan follows
    (review_return._finish_failed_moves) -- a source file is moved or deleted
    only while this machine's watcher record says the video is held in the
    bundle's queue. WHY: the source is usually the processing folder the
    video was routed from, and the video may be running there again; its
    mp4 or pose would then be taken from under the run. No watcher database
    on this computer: "" (the person pressing the button decides)."""
    own = db
    if own is None:
        from .db_location import open_watcher_db_if_present
        own, _why = open_watcher_db_if_present()
    from .review_return import _held_in_this_queue
    held, why_not = _held_in_this_queue(bundle, vid, own)
    return "" if held else why_not


def actions_for(item: dict, db=None) -> List[Action]:
    """The buttons the dashboard offers for ``item`` (empty: nothing to do
    from the GUI -- the item says what to do). An entry whose key starts with
    ``note:`` is not a button: the dialog shows its text as a line saying why
    a button is not offered. ``db``: the watcher database (opened here when
    None and one exists on this computer)."""
    if item.get("kind") in ("left_for_a_person", "queue_folder_in_the_way"):
        return _folder_actions(item)
    if item.get("kind") != "route_unfinished":
        return []
    vid = item.get("video_id") or ""
    folder = Path(str(item.get("folder") or "")) if item.get("folder") else None
    out: List[Action] = []
    try:
        if not (folder and folder.is_dir()):
            return []
        out.append(("finish_now", "Try to move the missing files in now",
                    "Runs the same step the watcher runs every ten minutes, at once."))
        from .review_routing import read_routing_manifest
        record = read_routing_manifest(folder, vid) or {}
        not_held = None
        for entry in record.get("failed_files") or []:
            name = str(entry.get("name") or "")
            if not name:
                continue
            source = entry.get("source")
            at_source = bool(source) and Path(str(source)).is_file()
            if (folder / name).is_file():
                if at_source:
                    out.append((f"note:{name}", f"{name}: no button",
                                f"{name} is in the bundle AND still at {source}, and the "
                                f"two differ. Which copy is right is a person's call: "
                                f"compare them, keep the right one in the bundle and "
                                f"delete the other by hand. The watcher crosses the file "
                                f"off by itself once only the bundle's copy is left."))
                continue
            if at_source:
                if not_held is None:
                    not_held = _source_not_held(folder, vid, db)
                if not_held:
                    out.append((f"note:{name}", f"{name}: no button",
                                f"{name} is still at {source}, and {not_held}. The "
                                f"video may be running there again, so that file is not "
                                f"moved from here. When the video is back in this queue "
                                f"the watcher finishes the move by itself."))
                continue
            out.append((f"accept_missing:{name}",
                        f"Accept the bundle without {name}",
                        f"Crosses {name} off the bundle's list of missing files (the "
                        f"record keeps who accepted it, and when). Use this only when "
                        f"the file is gone for good."))
    except OSError as e:
        logger.warning("Could not read %s for its actions: %s", folder, e)
    return out


def _folder_actions(item: dict) -> List[Action]:
    """Buttons for a queue folder left for a person, or one a route was
    refused because of (see the module docstring)."""
    from .review_routing import LEFT_NOTE, left_for_a_person
    folder = Path(str(item.get("folder") or "")) if item.get("folder") else None
    try:
        if not (folder and folder.is_dir()):
            return []
        if left_for_a_person(folder):
            names = sorted(p.name for p in folder.iterdir() if p.name != LEFT_NOTE)
            return [("delete_leftover", "Nothing in it is needed: delete this folder",
                     f"Deletes {folder} and everything in it: "
                     f"{', '.join(names) or 'nothing but its note'}. This folder is "
                     f"not the video's live copy, but what it holds has no current "
                     f"copy anywhere else, so it is gone for good once deleted. Look "
                     f"at it first; keep anything you need somewhere else.")]
    except OSError as e:
        logger.warning("Could not read %s for its actions: %s", folder, e)
        return []
    return [("note:live", "no button",
             f"{folder} is the video's live bundle in its review queue. Open that "
             f"queue's review window and finish its review; once it has returned, "
             f"the refused route is tried again by itself. No button deletes a "
             f"live bundle.")]


def perform(item: dict, key: str, db=None) -> str:
    """Carry out the action ``key`` (from actions_for) on ``item``. Returns a
    sentence saying what was done; raises ActionRefused (nothing changed)
    when it cannot be done. ``db``: the watcher database, used by
    'finish_now' to check the video is still held in the queue before a file
    at the route's source is touched (opened here when None)."""
    vid = item.get("video_id") or ""
    if key == "delete_leftover":
        from .review_routing import delete_leftover_folder
        if not vid or not item.get("folder"):
            raise ActionRefused("This item names no video folder.")
        try:
            names = delete_leftover_folder(item["folder"], vid)
        except (ValueError, RuntimeError, OSError) as e:
            raise ActionRefused(str(e))
        drop_item(item)
        return (f"{vid}: deleted {item['folder']} "
                f"({', '.join(names) or 'nothing but its note'}).")
    if item.get("kind") != "route_unfinished":
        raise ActionRefused(f"No action for items of kind {item.get('kind')!r}.")
    if not vid or not item.get("folder"):
        raise ActionRefused("This item names no video folder.")
    bundle = Path(str(item["folder"]))
    if not bundle.is_dir():
        drop_item(item)
        return f"{bundle} is no longer there; the item is removed from the list."
    from .review_return import (_claim_return, _finish_failed_moves,
                                _release_return_claim)
    from .review_routing import read_routing_manifest, _write_routing_manifest
    if key == "finish_now":
        # The same step the watcher runs, with the same check: a file at the
        # route's source is moved or deleted only while the watcher record
        # says the video is held in this queue (review_return.
        # _finish_failed_moves says why). No watcher database on this
        # computer: the person pressing the button decides.
        own = db
        if own is None:
            from .db_location import open_watcher_db_if_present
            own, _why = open_watcher_db_if_present()
        still, n = _finish_failed_moves(bundle, vid, db=own)
        if not still:
            drop_item(item)
        left = (", ".join(f"{m.get('name')} ({m.get('error') or 'not moved'})"
                          for m in still))
        return (f"{n} file(s) moved in." + (f" Still missing: {left}." if still else
                                             " The bundle is whole."))
    if not key.startswith("accept_missing:"):
        raise ActionRefused(f"Unknown action {key!r}.")
    name = key.split(":", 1)[1]
    if not _claim_return(vid):
        raise ActionRefused(
            f"{vid} is being moved by another program right now (the watcher, a "
            f"review window or the route command). Try again in a minute. Nothing "
            f"was changed.")
    try:
        record = read_routing_manifest(bundle, vid) or {}
        failed = list(record.get("failed_files") or [])
        entry = next((e for e in failed if str(e.get("name")) == name), None)
        if entry is None:
            raise ActionRefused(f"{name} is no longer listed as missing; nothing to do.")
        if (bundle / name).exists():
            raise ActionRefused(f"{name} is in the bundle; nothing to accept.")
        other = entry.get("source")
        if other and Path(str(other)).is_file():
            raise ActionRefused(f"{name} is still at {other}; it is not missing. "
                                f"Nothing was changed.")
        record["failed_files"] = [e for e in failed if e is not entry]
        resolved = list(record.get("resolved_by_a_person") or [])
        resolved.append(dict(entry, action="accept_missing",
                             at=datetime.now().isoformat(timespec="seconds"),
                             on=socket.gethostname()))
        record["resolved_by_a_person"] = resolved
        _write_routing_manifest(bundle, vid, record)
    finally:
        _release_return_claim(vid)
    if not record["failed_files"]:
        drop_item(item)
    logger.warning("Attention: %s: a person accepted the bundle without %s", vid, name)
    return f"{vid}: accepted the bundle without {name}."
