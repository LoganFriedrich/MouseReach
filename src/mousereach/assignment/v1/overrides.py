"""
Hard physics overrides for the v1 reach assignment classifier.

Applied AFTER the model has predicted causal probabilities. Catches
physically-impossible predictions and either reroutes them to the
correct reach within the segment or triages the whole segment.

Each override carries a named reason so the per-reach output is
diagnostic.

Override rules
--------------
1. PRE_APEX_NOT_ON_PILLAR: a reach with pre_apex_inside_pillar_frac
   below threshold cannot be causal for an on->off transition. If the
   model picked it, downgrade to miss; if no other reach is causal,
   pick the next-best by proba; if no reach has pre_apex on pillar,
   triage the segment.
2. ONE_CAUSAL_PER_SEGMENT: only one reach per segment can be causal.
   Keep the highest-proba causal prediction; downgrade others to miss.
3. SEGMENT_OUTCOME_CONTRADICTION (optional, future): cross-check
   against v5 outcome predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

PRE_APEX_INSIDE_THRESHOLD = 0.10   # below this, pellet wasn't on pillar
TRIAGE_REASON_NO_VALID_CAUSAL = "no_valid_causal_pre_apex_on_pillar"
TRIAGE_REASON_NONE = None


@dataclass
class OverrideRow:
    """One per-reach record after override application."""
    video_id: str
    segment_num: int
    reach_id: int
    reach_start_frame: int
    reach_end_frame: int
    segment_outcome: str
    interaction_frame: int
    gt_causal: int
    pred_causal_raw: int     # what the model said
    pred_causal_final: int   # after overrides
    proba_causal: float
    override_reason: Optional[str] = None
    pre_apex_inside_pillar_frac: float = 0.0


def apply_overrides(
    fold_rows: List[Dict],
    features_df: pd.DataFrame,
    pre_apex_threshold: float = PRE_APEX_INSIDE_THRESHOLD,
) -> List[OverrideRow]:
    """Apply override rules to per-reach predictions.

    fold_rows: list from FoldResult.rows (dicts with reach metadata +
        proba_causal + pred_causal + gt_causal)
    features_df: the per-reach feature dataframe (needs
        pre_apex_inside_pillar_frac column at minimum)

    Returns list of OverrideRow with both raw and final predictions.
    """
    # Build a lookup of features by (video_id, segment_num, reach_id)
    feat_lookup: Dict[tuple, dict] = {}
    for _, r in features_df.iterrows():
        key = (r["video_id"], int(r["segment_num"]), int(r["reach_id"]))
        feat_lookup[key] = r.to_dict()

    # Group fold_rows by (video_id, segment_num)
    seg_groups: Dict[tuple, list] = {}
    for fr in fold_rows:
        key = (fr["video_id"], fr["segment_num"])
        seg_groups.setdefault(key, []).append(fr)

    out_rows: List[OverrideRow] = []

    for (vid, sn), group in seg_groups.items():
        # Sort by proba_causal desc (highest first)
        sorted_group = sorted(group, key=lambda r: -r["proba_causal"])

        # Stage 1: filter pre-apex check. Reaches with pellet not on
        # pillar pre-apex cannot be causal regardless of model output.
        eligible = []
        ineligible = []
        for r in sorted_group:
            key = (vid, sn, r["reach_id"])
            feats = feat_lookup.get(key, {})
            pre_apex = float(feats.get("pre_apex_inside_pillar_frac", 0.0))
            r["_pre_apex_inside_pillar_frac"] = pre_apex
            if pre_apex >= pre_apex_threshold:
                eligible.append(r)
            else:
                ineligible.append(r)

        # Stage 2: pick exactly one causal from the eligible set (the
        # highest-proba one). If none eligible, no causal reach -> the
        # WHOLE segment effectively triages on this dimension.
        causal_rid = None
        if eligible:
            top = max(eligible, key=lambda r: r["proba_causal"])
            causal_rid = top["reach_id"]

        # Build OverrideRow per reach
        for r in group:
            raw = int(r["pred_causal"])
            pre_apex = r["_pre_apex_inside_pillar_frac"]
            override_reason = None
            final = 0
            if r["reach_id"] == causal_rid:
                final = 1
                if raw == 0:
                    override_reason = "promoted_to_causal_after_pre_apex_filter"
            else:
                final = 0
                if raw == 1 and pre_apex < pre_apex_threshold:
                    override_reason = "downgraded_pre_apex_not_on_pillar"
                elif raw == 1 and r["reach_id"] != causal_rid:
                    override_reason = "downgraded_lower_proba_in_segment"

            out_rows.append(OverrideRow(
                video_id=r["video_id"],
                segment_num=r["segment_num"],
                reach_id=r["reach_id"],
                reach_start_frame=r["reach_start_frame"],
                reach_end_frame=r["reach_end_frame"],
                segment_outcome=r["segment_outcome"],
                interaction_frame=r["interaction_frame"],
                gt_causal=r["gt_causal"],
                pred_causal_raw=raw,
                pred_causal_final=final,
                proba_causal=r["proba_causal"],
                override_reason=override_reason,
                pre_apex_inside_pillar_frac=pre_apex,
            ))

    return out_rows


def summarize_overrides(rows: List[OverrideRow]) -> Dict:
    """Report override impact: how many predictions changed and how."""
    n_total = len(rows)
    n_changed = sum(1 for r in rows if r.pred_causal_raw != r.pred_causal_final)
    by_reason = {}
    for r in rows:
        if r.override_reason:
            by_reason[r.override_reason] = by_reason.get(r.override_reason, 0) + 1
    return {
        "n_total": n_total,
        "n_changed": n_changed,
        "by_reason": by_reason,
    }
