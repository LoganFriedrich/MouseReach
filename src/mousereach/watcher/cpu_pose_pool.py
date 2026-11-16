"""Pose cut videos on a machine's CPUs, many at a time, beside its watcher.

WHY THIS EXISTS
---------------
Posing is the slow step of the pipeline, and only the GPU nodes did it. A
processing server with many idle cores can add to that: one DeepLabCut process
on the CPU is slow and uses only about three cores whatever thread count it is
given, so throughput comes from running MANY small processes side by side --
up to the point where they start slowing each other. On a server with many
cores, a dozen such processes add a useful fraction to what the GPU nodes
supply.

The watcher on such a machine (ProcessingOrchestrator) analyses; it does not
pose and handles one item at a time. This pool runs BESIDE it as its own
process and does exactly what a GPU node does with a cut video, with the same
code wherever it exists:

  claim     rename the video into ``<Single_Animal>/.inflight/<holder>/``
            (watcher/single_claim.py) -- one rename, so no two machines pose
            the same video. ``holder`` is ``<hostname>-cpu``: the pool's claims
            are its own, never mixed with anything the watcher holds.
  check     never pose a video that is finished or held for a person
            (the GPU node's own rule, DLCOrchestrator._already_done_or_held).
  pose      copy it to this machine's disk and run DeepLabCut in a child
            process on the CPU (``--threads`` each, below-normal priority), up
            to ``--workers`` at once. The pose must come from the declared
            model (repose.declared_scorer), as on every node.
  hand on   copy the video and its pose files into the shared Posed folder
            under temporary names, then rename them into place, the pose
            (.h5) last -- the processing watcher keys on the .h5, so when it
            appears everything is there (DLCOrchestrator._stage_files does the
            same). Then the claimed copy is removed, only once the handed-on
            copy is confirmed (as _retire_claimed_single does).
  record    the shared record (watcher_central.db) gets dlc_running and then
            the hand-off, as a GPU node writes it, so every machine sees who
            has the video.

The work order is the lab's (watcher/work_priority.py): the most preferred
group first, random inside it, Easy/Flat-type "idle only" work last.

A FAILED pose puts the claimed video straight back in the shared folder for
any node and keeps this pool off it for an hour. On start the pool gives back
any claim a previous run left (or hands on a pose it had finished). A
finished pose whose hand-on fails keeps its claim and is retried
(_retry_hand_ons); if its claim is gone, its local copies are deleted once a
current copy of the video is confirmed elsewhere, and otherwise left where
they are -- never moved to a side folder.
A status report (``Processing/.node_status/<holder>.json``) is rewritten every
cycle: other machines read it to know the pool is alive, and break its claims
only after it has been quiet for hours (single_claim.reclaim_stale).

Run it as ``mousereach-cpu-pose`` (see --help). Stop it with Ctrl+C or by
creating ``cpu_pose_stop.flag`` in the processing root: it stops taking new
videos at once, lets the running poses finish and hands them on, then exits.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mousereach.config import Paths, config_file_path

logger = logging.getLogger("mousereach.cpu_pose")

STOP_FLAG = "cpu_pose_stop.flag"
FAILED_BACKOFF_S = 3600          # a video whose pose failed here is left alone this long
CYCLE_S = 30                     # seconds between cycles


def holder_name() -> str:
    """The name the pool claims videos and reports status under."""
    return f"{socket.gethostname()}-cpu"


def _settings(args) -> dict:
    """Workers, threads and the model, from the command line, else the
    machine's config file (``cpu_pose`` section), else the defaults. The model
    path is a fact about this machine, so it lives in configuration only."""
    try:
        raw = json.loads(Path(config_file_path()).read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    cp = raw.get("cpu_pose") or {}
    return {
        "workers": int(args.workers or cp.get("workers") or 12),
        "threads": int(args.threads or cp.get("threads") or 3),
        "dlc_config": args.dlc_config or cp.get("dlc_config_path"),
    }


# ----------------------------------------------------------------- picking

def _waiting_singles(front: Path) -> List[Path]:
    """Cut videos lying unclaimed at the top of the shared singles folder."""
    out = []
    try:
        with os.scandir(front) as it:
            for e in it:
                if e.name.lower().endswith(".mp4") and e.is_file():
                    out.append(Path(e.path))
    except OSError as e:
        logger.warning(f"could not list {front} ({e})")
    return out


def _pick(candidates: List[Path]) -> Optional[Path]:
    """The next video by the lab's work order (random inside the best group)."""
    if not candidates:
        return None
    try:
        from mousereach.watcher.work_priority import best_tier_choice
        return best_tier_choice(candidates, key=lambda p: {"video_id": p.stem})
    except Exception as e:
        logger.debug(f"work order unavailable ({e}); random pick")
        return random.choice(candidates)


def _done_or_held(video_id: str):
    """The GPU node's own rule for a video that must not be posed again
    (finished, or held for a person). It reads only the shared drive, so it
    is called here without a node object rather than copied."""
    from mousereach.watcher.orchestrator import DLCOrchestrator
    return DLCOrchestrator._already_done_or_held(None, video_id)


# ----------------------------------------------------------------- the pool

class CpuPosePool:
    def __init__(self, workers: int, threads: int, dlc_config: Path):
        from mousereach.watcher import single_claim
        self.sc = single_claim
        self.workers = workers
        self.threads = threads
        self.dlc_config = Path(dlc_config)
        self.holder = holder_name()
        self.front = single_claim.front_door()
        self.posed = Path(Paths.DLC_STAGING)
        self.work = Path(Paths.PROCESSING_ROOT) / "CPU_Pose"
        self.work.mkdir(parents=True, exist_ok=True)
        self.active: Dict[str, dict] = {}          # video_id -> {proc, local, started}
        # Finished poses whose hand-on failed; their claims are kept alive and
        # the hand-on is tried again with a growing wait (_retry_hand_ons).
        self.unhanded: set = set()
        self.backoff: Dict[str, float] = {}
        self.done_times: List[float] = []
        self.stopping = False
        self.coordinator = None
        try:
            from mousereach.watcher.coordination import PipelineCoordinator
            c = PipelineCoordinator()
            c.ensure_tables()
            self.coordinator = c
        except Exception as e:
            logger.warning(f"shared record unavailable ({e}); posing goes on, the "
                           f"record is not updated by this pool")

    # -- shared record and status --------------------------------------------
    def _record(self, video_id: str, state: str, **kw) -> None:
        if self.coordinator is None:
            return
        try:
            self.coordinator.sync_video_state(video_id, self.holder, state, **kw)
        except Exception as e:
            logger.debug(f"{video_id}: shared-record write failed ({e})")

    def _status(self) -> None:
        from mousereach.watcher import node_status
        now = time.time()
        self.done_times = [t for t in self.done_times if now - t < 3600]
        # "doing" in the shape every node reports -- one {type, id, since}, or
        # None -- because readers take it as one item (node_status.describe
        # failed on a list and lost every node's line). The full
        # list of videos in hand is "poses".
        doing = None
        if self.active:
            since = min(j["started"] for j in self.active.values())
            doing = {"type": "cpu_pose", "id": f"{len(self.active)} videos",
                     "since": datetime.fromtimestamp(since).isoformat(timespec="seconds")}
        node_status.write(
            Paths.NAS_ROOT, "stopping" if self.stopping else
            ("working" if self.active else "idle"),
            hostname=self.holder,
            extra={"role": "cpu_pose", "workers": self.workers, "doing": doing,
                   "poses": sorted(self.active), "done_last_hour": len(self.done_times)})

    def _stop_requested(self) -> bool:
        try:
            return (Path(Paths.PROCESSING_ROOT) / STOP_FLAG).exists()
        except OSError:
            return False

    # -- start-up: what a previous run left -----------------------------------
    def recover(self) -> None:
        for video_id, claimed in sorted(self.sc.host_claims(self.holder).items()):
            local = self.work / f"{video_id}.mp4"
            if self._pose_files(video_id) and local.is_file():
                logger.info(f"{video_id}: a previous run finished its pose; handing it on")
                if not self._hand_on(video_id):
                    self._unhanded().add(video_id)
                continue
            self._clear_local(video_id)
            if self.sc.release_single(claimed):
                logger.info(f"{video_id}: left claimed by a previous run; given back")
        # Finished poses a previous run left with no claim of this machine (a
        # claim that was broken or moved while the pose waited): the same
        # check as a finished pose whose claim went (_retry_hand_ons) --
        # deleted once a current copy is confirmed elsewhere, otherwise left
        # where they are and reported. WHY at start: the list of such poses
        # lives only in memory, so a restart would otherwise forget them.
        # Only when this machine's claim folder was positively READ
        # (_own_claim_ids): single_claim.host_claims answers {} for a folder it
        # cannot read, and taking that as "no claims" during a share outage
        # would put every local video on the list, a half-posed one included.
        # And only a video with a finished pose beside it: a bare local video
        # is a pose that never finished, left as it is.
        claims = self._own_claim_ids()
        if claims is None:
            logger.info("this machine's claim folder could not be read; local copies "
                        "with no claim are checked at the next start")
            return
        try:
            local_ids = sorted(p.stem for p in self.work.glob("*.mp4") if p.is_file())
        except OSError:
            local_ids = []
        for video_id in local_ids:
            if video_id not in claims and self._pose_files(video_id):
                self._unhanded().add(video_id)

    def _own_claim_ids(self) -> Optional[set]:
        """The ids of this machine's claimed videos, or None when its claim
        folder could not be read. An empty set only when the folder was read
        and is empty, or is missing from a singles folder that WAS read.

        WHY not single_claim.host_claims: that never raises and answers {}
        for an unreadable folder, which here must not be mistaken for "no
        claims" (see recover)."""
        hdir = self.sc.host_dir(self.holder)
        if hdir is None:
            return None
        try:
            if hdir.is_dir():
                return {p.stem for p in hdir.iterdir()
                        if p.is_file() and p.suffix.lower() == ".mp4"}
            os.listdir(hdir.parent.parent)    # raises when the share is unreachable
            return set()
        except OSError:
            return None

    # -- one video ------------------------------------------------------------
    def _files_of(self, video_id: str) -> List[Path]:
        """This video's files in the work folder, by exact name: the video, its
        DeepLabCut outputs (<id>DLC...) and the pose log."""
        out = []
        for p in self.work.iterdir():
            n = p.name
            if n in (f"{video_id}.mp4", f"{video_id}.log") or n.startswith(f"{video_id}DLC"):
                out.append(p)
        return out

    def _pose_files(self, video_id: str) -> List[Path]:
        return sorted(p for p in self._files_of(video_id)
                      if p.name.startswith(f"{video_id}DLC") and p.suffix.lower() == ".h5")

    def _clear_local(self, video_id: str) -> None:
        """Remove this machine's scratch copies of a video. Only scratch: the
        shared claimed copy is the video's real file until it is handed on."""
        for p in self._files_of(video_id):
            try:
                p.unlink()
            except OSError:
                pass

    def _start_one(self) -> bool:
        """Claim, check, copy and start one pose. False when nothing to start."""
        from mousereach.watcher.transfer import safe_copy
        now = time.time()
        waiting = [p for p in _waiting_singles(self.front)
                   if self.backoff.get(p.stem, 0) <= now and p.stem not in self.active]
        while waiting:
            src = _pick(waiting)
            waiting.remove(src)
            video_id = src.stem
            if _done_or_held(video_id):
                continue                       # a GPU node's business, not ours
            claimed = self.sc.claim_single(src, self.holder)
            if claimed is None:
                continue                       # another machine won it, or still being written
            verdict = _done_or_held(video_id)  # again, now that it is ours
            if verdict:
                self.sc.release_single(claimed)
                continue
            local = self.work / f"{video_id}.mp4"
            if not safe_copy(claimed, local, verify=True):
                logger.warning(f"{video_id}: could not copy to {local}; given back")
                self.sc.release_single(claimed)
                self.backoff[video_id] = now + FAILED_BACKOFF_S
                continue
            self._record(video_id, "dlc_running", source_path=str(claimed))
            self.active[video_id] = {"proc": self._spawn(local), "local": local,
                                     "started": now}
            logger.info(f"{video_id}: posing on the CPU ({len(self.active)} running)")
            return True
        return False

    def _spawn(self, local: Path) -> subprocess.Popen:
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "-1",
                    "TF_NUM_INTRAOP_THREADS": str(self.threads),
                    "TF_NUM_INTEROP_THREADS": "2",
                    "OMP_NUM_THREADS": str(self.threads),
                    "MKL_NUM_THREADS": str(self.threads)})
        with open(local.with_suffix(".log"), "w", encoding="utf-8", errors="replace") as log:
            return subprocess.Popen(
                [sys.executable, "-m", "mousereach.watcher.cpu_pose_pool", "--pose-one",
                 str(local), "--dlc-config", str(self.dlc_config)],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(self.work))

    def _reap(self) -> None:
        from mousereach.watcher import repose
        declared = repose.declared_scorer()
        for video_id in list(self.active):
            job = self.active[video_id]
            rc = job["proc"].poll()
            if rc is None:
                continue
            del self.active[video_id]
            h5 = self._pose_files(video_id)
            bad = None
            if rc != 0 or not h5:
                bad = f"the pose process ended with code {rc} and no pose file"
            elif declared and not any(repose.scorer_of(p) == declared for p in h5):
                bad = (f"the pose came from {repose.scorer_of(h5[0])}, not the declared "
                       f"model {declared}; check cpu_pose.dlc_config_path")
            if bad:
                logger.error(f"{video_id}: {bad}; given back to the shared folder "
                             f"(log kept at {job['local'].with_suffix('.log')})")
                claimed = self.sc.claimed_path(video_id, self.holder)
                for p in self._files_of(video_id):
                    if p.suffix.lower() != ".log":
                        try:
                            p.unlink()
                        except OSError:
                            pass
                if claimed is not None:
                    self.sc.release_single(claimed)
                # Not recorded as failed on the shared record: the video is back
                # in the shared folder for any node, which is all a failure here
                # means; the reason is in this pool's log and the pose log.
                self.backoff[video_id] = time.time() + FAILED_BACKOFF_S
                continue
            mins = (time.time() - job["started"]) / 60
            logger.info(f"{video_id}: pose done in {mins:.0f} min")
            if not self._hand_on(video_id):
                self._unhanded().add(video_id)

    def _unhanded(self) -> set:
        """The finished poses whose hand-on failed (created on first use)."""
        return self.__dict__.setdefault("unhanded", set())

    # Wait before the next hand-on attempt of a finished pose: doubles after
    # every failed attempt, from one cycle's worth up to this cap. WHY: each
    # attempt copies a whole video over the share; a share that is down for
    # hours must not be hammered every cycle.
    HAND_ON_RETRY_MAX_S = 30 * 60

    def _hand_on_backoff(self) -> Dict[str, tuple]:
        """video_id -> (failed attempts, earliest next attempt), created on
        first use."""
        return self.__dict__.setdefault("hand_on_backoff", {})

    def _claim_state(self, video_id: str) -> str:
        """'held' (this machine's claimed copy is there), 'gone' (the shared
        singles folder was read and the claimed copy is not in it), or
        'unknown' (the share could not be read).

        WHY three answers: Path.is_file() answers False -- it does not raise
        -- for a share that is unreachable, so "is_file() is False" alone
        would read a dropped share as a claim given back and throw away a
        finished pose. 'gone' is said only after the singles folder itself
        was listed successfully."""
        claimed = self.sc.claimed_path(video_id, self.holder)
        door = self.sc.front_door()
        if claimed is None or door is None:
            return 'unknown'
        try:
            if claimed.is_file():
                return 'held'
            os.listdir(door)                  # raises when the share is unreachable
            host = claimed.parent
            if host.is_dir():
                names = set(os.listdir(host))
            elif host.parent.is_dir():
                names = set()                 # no claim folder for this machine any more
            else:
                os.listdir(host.parent.parent)
                names = set()
            return 'held' if claimed.name in names else 'gone'
        except OSError:
            return 'unknown'

    def _current_copy_elsewhere(self, video_id: str) -> Optional[str]:
        """Where the video's current copy is, when this machine's claim on it
        is gone -- or None when no current copy could be confirmed anywhere.

        Confirmed means: the video file, the same size as this machine's
        local copy, at the top of the shared singles folder, in another
        machine's claim folder, or in the posed folder WITH a pose file
        (<id>DLC*.h5) beside it; or the GPU node's own
        finished-or-held rule (_done_or_held: filed, or in a review queue)
        says the video has moved on. WHY it must be confirmed: the local copy
        is deleted on this answer, and a local copy no other place holds may
        be the only copy of the video. A bare video in the posed folder is a
        hand-off that stopped half way -- the server does not take it in --
        so it is not a current copy (the GPU side's _queue_copy_elsewhere
        asks for the video AND its pose in the posed folder too)."""
        local = self.work / f"{video_id}.mp4"
        try:
            size = local.stat().st_size if local.is_file() else None
        except OSError:
            size = None
        candidates = []
        door = self.sc.front_door()
        if door is not None:
            candidates.append(Path(door) / f"{video_id}.mp4")
        try:
            holder = self.sc.inflight_ids().get(video_id)
        except Exception:
            holder = None
        if holder and holder != self.holder:
            other = self.sc.claimed_path(video_id, holder)
            if other is not None:
                candidates.append(Path(other))
        try:
            posed_pose = any(self.posed.glob(f"{video_id}DLC*.h5"))
        except OSError:
            posed_pose = False
        if posed_pose:
            candidates.append(self.posed / f"{video_id}.mp4")
        for c in candidates:
            try:
                if c.is_file() and (size is None or c.stat().st_size == size):
                    return str(c)
            except OSError:
                continue
        try:
            verdict = _done_or_held(video_id)
        except Exception:
            verdict = None
        if verdict:
            return str(verdict)
        return None

    def _retry_hand_ons(self) -> None:
        """Try every finished pose whose hand-on failed again, with a growing
        wait between attempts (HAND_ON_RETRY_MAX_S).

        WHY: such a pose used to wait in the work folder "until the next
        start", while its claim was no longer refreshed (only running poses
        were) -- so after a day the shared copy was returned to the folder for
        any node, which posed the video again, and this pool then handed on
        its own pose too at its next start. Now the claim is kept alive (see
        run) and the hand-on is retried here.

        If the claim is gone all the same (another machine broke it, or a
        person moved the file) -- known only when the singles folder could be
        read (_claim_state) -- the video is someone else's now, and handing
        it on would be the double this rule prevents: the pose is not handed
        on. Its local copies are machine leftovers: they are DELETED once a
        current copy of the video is confirmed elsewhere
        (_current_copy_elsewhere), and otherwise left exactly where they are
        -- never moved to a side folder -- with one warning in the log, and
        looked at again every cycle. While the share cannot be read nothing
        is decided; the pose waits for the next cycle with its claim kept."""
        backoff = self._hand_on_backoff()
        now = time.time()
        for video_id in sorted(self._unhanded()):
            if not self._pose_files(video_id):
                # No finished pose here (it was removed, or the entry is a
                # video whose pose never finished): nothing to hand on, and a
                # bare video must never reach the posed folder. Dropped from
                # the list, so its claim is no longer kept alive and goes
                # back to the shared folder by the usual stale give-back.
                logger.warning(f"{video_id}: no finished pose in {self.work}; not "
                               f"handed on, and its claim is no longer kept alive")
                self._unhanded().discard(video_id)
                backoff.pop(video_id, None)
                continue
            state = self._claim_state(video_id)
            if state == 'unknown':
                continue                     # the share is unreachable: next cycle
            if state == 'gone':
                where = self._current_copy_elsewhere(video_id)
                if where is None:
                    warned = self.__dict__.setdefault("kept_warned", set())
                    if video_id not in warned:
                        warned.add(video_id)
                        logger.warning(
                            f"{video_id}: its claim is gone and no current copy of the "
                            f"video was found elsewhere, so its pose is not handed on "
                            f"and its local copies stay in {self.work}; find where the "
                            f"video went (the local video may be its only copy)")
                    continue
                self._clear_local(video_id)
                logger.info(f"{video_id}: its claim is gone and its current copy is at "
                            f"{where}; its pose is not handed on and its local copies "
                            f"were deleted")
                self._unhanded().discard(video_id)
                self.__dict__.setdefault("kept_warned", set()).discard(video_id)
                backoff.pop(video_id, None)
                continue
            attempts, next_at = backoff.get(video_id, (0, 0.0))
            if now < next_at:
                continue
            if self._hand_on(video_id):
                self._unhanded().discard(video_id)
                backoff.pop(video_id, None)
            else:
                attempts += 1
                wait = min(self.HAND_ON_RETRY_MAX_S, 60 * (2 ** (attempts - 1)))
                backoff[video_id] = (attempts, now + wait)

    def _hand_on(self, video_id: str) -> bool:
        """Into the shared Posed folder, the GPU nodes' way: every file under a
        temporary name, then renamed into place with the pose last; local
        copies removed only after that; the claimed copy removed only once
        the handed-on video is confirmed."""
        from mousereach.watcher.transfer import safe_copy
        if not self._pose_files(video_id):
            # WHY: a bare video in the posed folder is a hand-off the server
            # never takes in; the claimed shared copy must stay the video's
            # real file until a finished pose goes with it.
            logger.error(f"{video_id}: no finished pose in {self.work}; not handed on")
            return False
        files = sorted((p for p in self._files_of(video_id)
                        if p.suffix.lower() in (".mp4", ".h5", ".csv", ".pickle")),
                       key=lambda p: (p.suffix.lower() == ".h5", p.name))
        parts = []
        try:
            self.posed.mkdir(parents=True, exist_ok=True)
            for f in files:
                dest = self.posed / f.name
                part = dest.with_name(dest.name + ".part")
                if not safe_copy(f, part, verify=True):
                    raise IOError(f"could not copy {f.name} to {self.posed}")
                parts.append((f, part, dest))
            for _, part, dest in parts:
                part.replace(dest)
        except Exception as e:
            for _, part, _ in parts:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
            logger.error(f"{video_id}: could not hand on ({e}); its pose stays in "
                         f"{self.work}, its claim is kept, and the hand-on is tried "
                         f"again, waiting longer after each failure (up to "
                         f"{self.HAND_ON_RETRY_MAX_S // 60} min)")
            return False
        for f, _, _ in parts:
            try:
                f.unlink()
            except OSError:
                pass
        try:
            (self.work / f"{video_id}.log").unlink()
        except OSError:
            pass
        staged_mp4 = self.posed / f"{video_id}.mp4"
        h5 = [d for _, _, d in parts if d.suffix.lower() == ".h5"]
        self._record(video_id, "archived", staged_at=datetime.now().isoformat(),
                     nas_path=str(self.posed),
                     dlc_completed_at=datetime.now().isoformat())
        self.done_times.append(time.time())
        claimed = self.sc.claimed_path(video_id, self.holder)
        try:
            if claimed is not None and claimed.is_file():
                same = (staged_mp4.is_file()
                        and staged_mp4.stat().st_size == claimed.stat().st_size)
                if same or any(d.name == staged_mp4.name for _, _, d in parts):
                    claimed.unlink()
        except OSError as e:
            logger.warning(f"{video_id}: handed on, but the claimed copy {claimed} could "
                           f"not be removed ({e}); remove it by hand")
        logger.info(f"{video_id}: handed on to {self.posed} ({len(parts)} files, pose "
                    f"{h5[0].name if h5 else '?'})")
        return True

    # -- main loop -------------------------------------------------------------
    def run(self) -> int:
        logger.info(f"CPU pose pool {self.holder}: up to {self.workers} poses of "
                    f"{self.threads} threads; model {self.dlc_config}")
        self.recover()
        last_sweep = 0.0
        while True:
            if not self.stopping and self._stop_requested():
                self.stopping = True
                logger.info(f"stop flag found: no new videos; finishing "
                            f"{len(self.active)} running pose(s)")
            self._reap()
            self._retry_hand_ons()
            # Keep alive the claims of running poses AND of finished poses not
            # handed on yet (_retry_hand_ons says why).
            held = list(self.active) + sorted(self._unhanded())
            if held:
                self.sc.heartbeat_claims(self.holder, only=held)
            if not self.stopping:
                while len(self.active) < self.workers and self._start_one():
                    pass
            self._status()
            if self.stopping and not self.active:
                logger.info("stopped cleanly")
                self._status()
                return 0
            if time.time() - last_sweep > 600:
                last_sweep = time.time()
                try:
                    self.sc.reclaim_stale()
                except Exception:
                    pass
            time.sleep(CYCLE_S)


# ----------------------------------------------------------------- child

def _pose_one(video: Path, dlc_config: Path) -> int:
    """Child process: one pose on the CPU, below-normal priority. The thread
    limits come from the environment the pool sets before this starts."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass
    from mousereach.dlc.core import resolve_dlc_shuffle, run_dlc_batch
    shuffle, _ = resolve_dlc_shuffle()
    results = run_dlc_batch(video_paths=[video], config_path=dlc_config,
                            output_dir=video.parent, gpu=None, save_as_csv=True,
                            shuffle=shuffle)
    ok = bool(results) and results[0].get("status") == "success"
    print("pose", "ok" if ok else "FAILED", results[0] if results else "", flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-cpu-pose",
        description="Pose cut videos on this machine's CPUs, many at once, beside "
                    "its watcher. Takes videos from the shared singles folder in the "
                    "lab's work order and hands them to Processing/Posed exactly as a "
                    "GPU node does. Stop: Ctrl+C, or create cpu_pose_stop.flag in the "
                    "processing root (running poses finish first).")
    ap.add_argument("--workers", type=int, help="poses at once (default 12, or "
                    "cpu_pose.workers in the config file)")
    ap.add_argument("--threads", type=int, help="CPU threads per pose (default 3)")
    ap.add_argument("--dlc-config", help="the DeepLabCut project's config.yaml "
                    "(default: cpu_pose.dlc_config_path in the config file)")
    ap.add_argument("--pose-one", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s")
    if args.pose_one:
        return _pose_one(Path(args.pose_one), Path(args.dlc_config))
    s = _settings(args)
    if not s["dlc_config"] or not Path(s["dlc_config"]).is_file():
        print("No DeepLabCut model configured for CPU posing. Set it once with:\n"
              "  add  \"cpu_pose\": {\"dlc_config_path\": \"<path to config.yaml>\"}  to "
              f"{config_file_path()}\n  or pass --dlc-config <path>.")
        return 2
    if not Paths.DLC_STAGING or not Paths.PROCESSING_ROOT:
        print("Processing root or shared drive not configured; run mousereach-setup.")
        return 2
    pool = CpuPosePool(s["workers"], s["threads"], Path(s["dlc_config"]))
    try:
        return pool.run()
    except KeyboardInterrupt:
        # The poses are stopped too: the next start gives their videos back
        # (recover), and it must never find a pose still running in the folder
        # it clears. The stop flag is the way to let running poses finish.
        for job in pool.active.values():
            try:
                job["proc"].kill()
            except Exception:
                pass
        logger.info(f"interrupted: {len(pool.active)} running pose(s) stopped; their "
                    f"videos are given back at the next start")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
