"""Run ONE video through every post-DLC step -- the single-video equivalent of
the watcher's automatic pipeline, for a video an operator opens by hand.

Chain: segmentation -> reach detection -> outcome detection -> assignment
(algo-4, causal reach) -> gate check -> kinematics + connectome.db sync.

Requires the video to already have a DLC pose h5 beside it (crop + DLC are
node-specific/GPU steps; if the pose is missing the driver says so and points at
the DLC tab rather than trying to run DLC on a machine that may lack a GPU).

The gate here is EVALUATED but does NOT move the bundle: a manual run reports
"would be held for review" and skips kinematics (so no un-reviewed data reaches
the database), leaving the files in place for the operator to inspect. This
preserves the project invariant -- kinematics only on clean videos -- without
side-effecting the review queues for a one-off manual run.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

STAGES = ["segmentation", "reach_detection", "outcome_detection",
          "assignment", "gate", "kinematics"]


def run_all_steps(video_path, progress: Optional[Callable[[str, str], None]] = None) -> Dict:
    """Run the full post-DLC pipeline on one video. Returns a summary dict with
    keys: video, stages (list of {stage,msg}), held (None|'triage'|'deep_review'),
    hold_reason, error (None|str), done (bool). ``progress(stage, msg)`` is called
    as each stage starts/finishes."""
    video_path = Path(video_path)
    stem = video_path.stem
    processing_dir = video_path.parent
    result: Dict = {
        "video": stem, "processing_dir": str(processing_dir),
        "stages": [], "held": None, "hold_reason": None, "error": None, "done": False,
    }

    def _p(stage: str, msg: str = ""):
        result["stages"].append({"stage": stage, "msg": msg})
        if progress:
            try:
                progress(stage, msg)
            except Exception:
                pass

    try:
        from mousereach.config import parse_tray_type
        from mousereach.segmentation.core.batch import process_single as seg_single
        from mousereach.reach.core.batch import process_single as reach_single
        from mousereach.outcomes.core.batch import process_single as outcome_single
        from mousereach.assignment.run import assign_reaches_for_video
        from mousereach.watcher.review_gate import evaluate_gate, DECISION_CLEAN
        from mousereach.pipeline.triage import triage_video

        h5s = sorted(processing_dir.glob(f"{stem}*.h5"))
        if not h5s:
            result["error"] = (
                "No DLC pose (.h5) found next to the video. Run DLC first "
                "(the '1 - DLC Analysis' tab, on a GPU machine)."
            )
            _p("blocked", result["error"])
            return result
        dlc_path = h5s[0]

        # 1. Segmentation
        _p("segmentation", "running")
        seg_single(dlc_path)
        seg_path = processing_dir / f"{stem}_segments.json"
        if not seg_path.exists():
            result["error"] = "Segmentation produced no output."
            _p("error", result["error"])
            return result

        # 2. Reach detection
        _p("reach_detection", "running")
        reach_single(dlc_path, seg_path)
        reach_path = processing_dir / f"{stem}_reaches.json"

        # 3. Outcome + 4. Assignment (skip for E/F trays -- no reliable pellet)
        tray = parse_tray_type(f"{stem}.mp4").get("tray_type", "P")
        skip_outcomes = tray in ("E", "F")
        if not skip_outcomes:
            _p("outcome_detection", "running")
            outcome_single(dlc_path, seg_path, reach_path)
            _p("assignment", "running (causal reach)")
            assign_reaches_for_video(processing_dir, stem, dlc_path)
        else:
            _p("outcome_detection", f"skipped (tray {tray})")

        # QC triage verdict (feeds the gate: critical QC -> deep review)
        qc_verdict = "auto_approved"
        try:
            tr = triage_video(video_id=stem, processing_dir=processing_dir, h5_path=dlc_path)
            qc_verdict = tr.verdict
            tr.save(processing_dir / f"{stem}_triage.json")
        except Exception as e:
            logger.warning(f"triage failed for {stem}: {e}")

        # 5. Gate -- evaluate only (do NOT move the bundle for a manual run)
        _p("gate", "checking")
        decision, reason, _st = evaluate_gate(stem, processing_dir, qc_verdict=qc_verdict)
        if decision != DECISION_CLEAN:
            result["held"] = decision
            result["hold_reason"] = reason
            _p("held", f"{decision}: {reason} -- kinematics skipped (resolve in review to finish)")
            return result

        # 6. Kinematics + DB sync (clean videos only)
        if not skip_outcomes:
            _p("kinematics", "extracting")
            from mousereach.kinematics.core.feature_extractor import FeatureExtractor
            from mousereach.review.causal_review_io import resolve_review_path
            review_path = resolve_review_path(stem, processing_dir)
            feats = FeatureExtractor().extract(
                dlc_path, reach_path,
                processing_dir / f"{stem}_pellet_outcomes.json",
                review_path=review_path,
            )
            fp = processing_dir / f"{stem}_features.json"
            fp.write_text(json.dumps(feats.to_dict(), indent=2), encoding="utf-8")
            _p("db_sync", "syncing to connectome.db")
            try:
                from mousereach.sync.database import sync_file_to_database
                sync_file_to_database(fp)
            except Exception as e:
                logger.warning(f"db sync failed for {stem}: {e}")

        result["done"] = True
        _p("complete", "all steps done")
        return result

    except Exception as e:
        logger.exception("run_all_steps failed")
        result["error"] = str(e)
        _p("error", str(e))
        return result
