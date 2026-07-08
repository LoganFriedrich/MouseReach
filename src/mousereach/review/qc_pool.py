"""Routine spot-check pool: re-confirm the algo on already-passing segments.

The triage worklist (:mod:`mousereach.review.triage_queue`) surfaces the algo's
UNRESOLVED problems. This module is the complement used by the same routine
protocol: it samples segments the algo was CONFIDENT about -- a committed
outcome, and (for touched outcomes) a committed causal reach -- and asks a
reviewer to *confirm the call or flag a disagreement*. Confirmations build an
agreement-rate record over time; a flag converts the segment into a triage item
to resolve. This is how algo drift gets caught before it silently corrupts
kinematics.

Selection is a STRATIFIED ROTATING sample (by cohort + date): round-robin across
strata so drift shows up across conditions, never-checked-first so coverage
grows across sessions without re-checking the same segments until the pool is
exhausted.

State + audit live under ``<review_root>/_QC/`` (data, not code):
  - ``qc_state.json``     : which (video, segment) have been checked, when, verdict
  - ``qc_drift_log.jsonl``: append-only, one JSON line per confirm/flag verdict
"""
from __future__ import annotations

import glob
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .triage_queue import (
    ASSIGN_SUFFIX, OUTCOME_SUFFIX, REACH_SUFFIX, SEG_SUFFIX,
    TOUCHED_OUTCOMES, TriageEntry,
    _causal_segments, _load_json, _segmentation_failed,
)

# A resolved outcome the algo actually committed (i.e. worth spot-checking).
# "triaged" is excluded (that's a triage item, not a confident call); empty /
# None / "unknown" are excluded (nothing to confirm).
_CONFIDENT_OUTCOMES = ("retrieved", "displaced_sa", "displaced_outside",
                       "untouched", "abnormal")


def parse_stem(stem: str) -> Dict[str, str]:
    """Pull ``date`` / ``cohort`` / ``session`` from a video stem like
    ``20250624_CNT0101_P3``. Cohort is the two digits after ``CNT`` (per the
    project Mouse-ID convention CNT_<cohort>_<subject>). Unknown parts fall
    back to ``"?"`` so stratification never crashes on an odd name."""
    date = cohort = session = "?"
    m = re.match(r"(\d{8})_([A-Za-z]+)(\d{2})(\d{2})_P?(\d+)", stem)
    if m:
        date = m.group(1)
        cohort = f"{m.group(2)}{m.group(3)}"   # e.g. CNT01
        session = f"P{m.group(5)}"
    else:
        m2 = re.match(r"(\d{8})_", stem)
        if m2:
            date = m2.group(1)
    return {"date": date, "cohort": cohort, "session": session}


@dataclass
class QCCandidate:
    """One already-passing segment eligible for a spot-check."""
    video_name: str
    segment_num: int
    start_frame: int
    end_frame: int
    algo_outcome: str
    algo_causal_reach_id: Optional[int]
    algo_causal_start: Optional[int]
    algo_causal_end: Optional[int]
    algo_dir: Path


def iter_passing_segments(
    corpus_root: Path,
    *,
    video_filter: Optional[Sequence[str]] = None,
) -> List[QCCandidate]:
    """Enumerate confident, resolved segments across a corpus of bundles.

    A segment qualifies when: segmentation did NOT fail for the video, the
    outcome is a committed class (not ``triaged`` / empty), the segment is not
    ``flagged_for_review``, and -- for touched outcomes -- the assignment
    committed a causal reach. These are exactly the calls a spot-check can
    confirm the algo got right.
    """
    corpus_root = Path(corpus_root)
    out: List[QCCandidate] = []
    for sub in sorted(corpus_root.iterdir()):
        if not sub.is_dir():
            continue
        stem = sub.name
        if video_filter is not None and stem not in video_filter:
            continue
        op = sub / f"{stem}{OUTCOME_SUFFIX}"
        if not op.is_file():
            continue
        if _segmentation_failed(_load_json(sub / f"{stem}{SEG_SUFFIX}")):
            continue
        outcome_data = _load_json(op) or {}
        reach_data = _load_json(sub / f"{stem}{REACH_SUFFIX}") or {}
        assign_data = _load_json(sub / f"{stem}{ASSIGN_SUFFIX}")
        causal_segs = _causal_segments(assign_data)
        # Map segment_num -> committed causal reach record (for display).
        causal_reach = {}
        for r in (assign_data or {}).get("reaches", []) or []:
            if r.get("is_causal") or str(r.get("label") or "").startswith("causal"):
                causal_reach[int(r.get("segment_num"))] = r
        reach_segs = {s.get("segment_num"): s
                      for s in (reach_data.get("segments") or [])}
        for os_ in outcome_data.get("segments") or []:
            sn = os_.get("segment_num")
            if sn is None:
                continue
            outcome = os_.get("outcome")
            if outcome not in _CONFIDENT_OUTCOMES:
                continue
            if os_.get("flagged_for_review"):
                continue
            sn = int(sn)
            # Touched outcomes must have a committed causal reach to be a
            # confident call; if not, that's a triage item, not a QC candidate.
            if outcome in TOUCHED_OUTCOMES and sn not in causal_segs:
                continue
            rseg = reach_segs.get(sn, {})
            cr = causal_reach.get(sn)
            out.append(QCCandidate(
                video_name=stem,
                segment_num=sn,
                start_frame=int(rseg.get("start_frame", 0)),
                end_frame=int(rseg.get("end_frame", 0)),
                algo_outcome=outcome,
                algo_causal_reach_id=(cr.get("reach_id") if cr else None),
                algo_causal_start=(cr.get("start_frame") if cr else None),
                algo_causal_end=(cr.get("end_frame") if cr else None),
                algo_dir=sub,
            ))
    return out


@dataclass
class RoutineQCPool:
    """Stratified rotating spot-check pool over a corpus of bundles.

    Persists which (video, segment) have been checked (``qc_state.json``) and an
    append-only verdict log (``qc_drift_log.jsonl``) under ``qc_dir`` (defaults
    to ``<corpus_root>/../_QC/``).
    """
    corpus_root: Path
    qc_dir: Path
    state: dict = field(default_factory=dict)

    @classmethod
    def open(cls, corpus_root: Path, qc_dir: Optional[Path] = None) -> "RoutineQCPool":
        corpus_root = Path(corpus_root)
        qc_dir = Path(qc_dir) if qc_dir is not None else corpus_root.parent / "_QC"
        qc_dir.mkdir(parents=True, exist_ok=True)
        state = _load_json(qc_dir / "qc_state.json") or {"checked": {}}
        return cls(corpus_root=corpus_root, qc_dir=qc_dir, state=state)

    # --- paths --------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.qc_dir / "qc_state.json"

    @property
    def log_path(self) -> Path:
        return self.qc_dir / "qc_drift_log.jsonl"

    def _key(self, video_name: str, segment_num: int) -> str:
        return f"{video_name}:{segment_num}"

    def _stratum(self, video_name: str) -> str:
        p = parse_stem(video_name)
        return f"{p['cohort']}|{p['date']}"

    # --- sampling -----------------------------------------------------------

    def sample(
        self,
        n: int,
        *,
        candidates: Optional[List[QCCandidate]] = None,
        seed: Optional[int] = None,
    ) -> List[QCCandidate]:
        """Stratified rotating sample of ``n`` passing segments.

        Round-robins across (cohort, date) strata; within a stratum, prefers
        never-checked segments then least-recently-checked. Ties broken
        randomly (``seed`` for reproducibility). Pass ``candidates`` to reuse an
        already-computed enumeration (avoids a second corpus scan)."""
        if candidates is None:
            candidates = iter_passing_segments(self.corpus_root)
        if n <= 0 or not candidates:
            return []
        rng = random.Random(seed)
        checked = self.state.get("checked", {})

        def recency(c: QCCandidate):
            rec = checked.get(self._key(c.video_name, c.segment_num))
            # (checked?, last_checked_at) -- never-checked (0) sort first.
            return (1 if rec else 0, (rec or {}).get("checked_at", ""))

        strata: Dict[str, List[QCCandidate]] = defaultdict(list)
        for c in candidates:
            strata[self._stratum(c.video_name)].append(c)
        for k in strata:
            rng.shuffle(strata[k])            # randomize ties
            strata[k].sort(key=recency)       # never-checked then oldest first
        stratum_keys = list(strata.keys())
        rng.shuffle(stratum_keys)

        result: List[QCCandidate] = []
        idx = {k: 0 for k in strata}
        while len(result) < n and any(idx[k] < len(strata[k]) for k in stratum_keys):
            for k in stratum_keys:
                if idx[k] < len(strata[k]):
                    result.append(strata[k][idx[k]])
                    idx[k] += 1
                    if len(result) >= n:
                        break
        return result

    def to_entries(self, sampled: Sequence[QCCandidate]) -> List[TriageEntry]:
        """Wrap sampled candidates as ``kind="qc"`` worklist entries so the
        clearing widget can present them in the same walk as triage items."""
        entries = []
        for c in sampled:
            causal = ""
            if c.algo_causal_reach_id is not None:
                causal = (f" (causal reach {c.algo_causal_reach_id}"
                          f" [{c.algo_causal_start}-{c.algo_causal_end}])")
            reason = (f"SPOT-CHECK: algo says '{c.algo_outcome}'{causal} -- "
                      f"confirm or flag a disagreement")
            entries.append(TriageEntry(
                video_name=c.video_name,
                segment_num=c.segment_num,
                start_frame=c.start_frame,
                end_frame=c.end_frame,
                outcome_flag_reason=reason,
                algo_dir=c.algo_dir,
                kind="qc",
            ))
        return entries

    # --- verdicts + audit ---------------------------------------------------

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def record_verdict(
        self,
        video_name: str,
        segment_num: int,
        *,
        verdict: str,              # "confirmed" | "flagged"
        algo_outcome: str,
        algo_causal_reach_id: Optional[int],
        reviewer: str,
        note: str = "",
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a spot-check verdict: update the checked-state and append to
        the drift log. ``verdict="flagged"`` means the reviewer disagreed with
        the algo (the segment should then be resolved as a triage item)."""
        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        self.state.setdefault("checked", {})[self._key(video_name, segment_num)] = {
            "checked_at": ts, "verdict": verdict, "reviewer": reviewer,
        }
        self.save_state()
        rec = {
            "video_name": video_name, "segment_num": segment_num,
            "algo_outcome": algo_outcome, "algo_causal_reach_id": algo_causal_reach_id,
            "verdict": verdict, "reviewer": reviewer, "note": note, "at": ts,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def agreement_summary(self) -> dict:
        """Aggregate the drift log into a confirm/flag agreement rate."""
        n = confirmed = flagged = 0
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                if rec.get("verdict") == "confirmed":
                    confirmed += 1
                elif rec.get("verdict") == "flagged":
                    flagged += 1
        rate = (confirmed / n) if n else None
        return {"n_checks": n, "confirmed": confirmed, "flagged": flagged,
                "agreement_rate": rate}
