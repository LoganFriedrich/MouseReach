# Residual Pool Attack: Findings Report

## Summary

Investigated whether any segments in the post-Stage-27 residual pool (64 non-expected-triage cases: 47 displaced_sa, 17 retrieved) could be cleanly committed by new stages. Found one highly effective signal -- **pillar visibility transition** (Pillar bodypart likelihood rising from low to high across a reach) -- that yields two deployable stages at 100% precision. Stage 28 commits 9 retrieved cases via pillar-lk-transition + pellet-vanish guard. Stage 29 commits 6 displaced_sa cases by using pillar-lk-transition to disambiguate which of multiple high-displacement reaches is causal. Combined: residual drops from 64 to 49, retrieved yield rises from 81.3% to 91.2%, displaced_sa from 85.3% to 87.0%.

## Per-Predicate Results

| # | Predicate | Pool | Fires | TP | Wrong | Cross-class | 100% deployable? |
|---|-----------|------|-------|-----|-------|-------------|------------------|
| 1 | MULTI_HIGH_DISP: first chronological high-disp reach | 6 disp_sa | 6 | 3 | 3 | 1 on retrieved | NO (50% precision) |
| 1v | MULTI_HIGH_DISP: first == max disp | 6 disp_sa | 1 | 0 | 1 | -- | NO |
| 2 | MULTI_VANISH retrieved: first vanish reach | 0 | 0 | 0 | 0 | -- | N/A (no cases) |
| 3 | UNIQ_VANISH_GUARD_FAIL: relax Stage 26 guards | 9 retrieved | varies | -- | -- | -- | NOT INVESTIGATED (guards are load-bearing) |
| 4 | NO_VANISH retrieved: no vanish signal at all | 6 retrieved | -- | -- | -- | -- | NO (no discriminating signal) |
| 5 | LOW_DISP (5-10 px) displaced_sa | 0 | 0 | 0 | 0 | -- | N/A (no cases in range) |
| 6 | NO_QUALIFIER displaced_sa (disp < 5) | 39 disp_sa | -- | -- | -- | -- | NO (DLC can't track pellet) |
| **7** | **Pillar-lk-transition + vanish -> retrieved** | **12 ret** | **9 unique** | **9** | **0** | **0 on disp_sa** | **YES -> Stage 28** |
| **8** | **Pillar-lk-transition disambiguates multi-high-disp** | **6 disp_sa** | **6** | **6** | **0** | **0 on retrieved** | **YES -> Stage 29** |

## Stages Drafted

### Stage 28: `stage_28_retrieved_via_pillar_visibility_transition.py`

- **Path**: `src/mousereach/outcomes/v6_cascade/stage_28_retrieved_via_pillar_visibility_transition.py`
- **What it commits**: `retrieved` when exactly one reach shows a strong pillar-lk transition (delta > 0.5, pre < 0.4, post > 0.8) AND that reach also shows the pellet-vanish signal.
- **Validation**: 9/9 correct commits, 0 wrong, 0 cross-class fires on displaced_sa. All 9 pass IFR bout matching (SAME_BOUT).
- **Recommendation**: **Deploy.** The vanish guard completely eliminates cross-class risk -- all 7 displaced_sa cases that fire on pillar-lk alone are blocked by vanish=False.

### Stage 29: `stage_29_displaced_sa_via_pillar_disambiguated_multi_displacement.py`

- **Path**: `src/mousereach/outcomes/v6_cascade/stage_29_displaced_sa_via_pillar_disambiguated_multi_displacement.py`
- **What it commits**: `displaced_sa` when 2+ reaches have displacement >= 10 px (Stage 27's uniqueness requirement fails) but exactly ONE of those high-disp reaches has the pillar-lk transition. The causal reach is the one that revealed the pillar.
- **Validation**: 6/6 correct commits, 0 wrong, 0 cross-class fires on retrieved. No retrieved residual has 2+ reaches with disp >= 10.
- **Recommendation**: **Deploy.** Resolves all MULTI_HIGH_DISP residuals.

### Combined Impact (validated via test runner)

| Metric | Before (Stages 0-27) | After (+ Stages 28-29) | Delta |
|--------|---------------------|----------------------|-------|
| Retrieved yield | 74/91 (81.3%) | 83/91 (91.2%) | +9 cases, +9.9 pp |
| Displaced_sa yield | 296/347 (85.3%) | 302/347 (87.0%) | +6 cases, +1.7 pp |
| Untouched yield | 285/285 (100%) | 285/285 (100%) | unchanged |
| Non-expected residuals | 64 | 49 | -15 |
| Wrong commits (all stages) | 0 | 0 | no regression |

**Test runner**: `C:\Users\friedrichl\AppData\Local\Temp\runner_test_28_29.py` -- NOT wired into the canonical runner. Wiring is the user's call.

## Cases That Nothing Catches (irreducible manual-review pool)

**49 segments** remain after Stages 0-29 (excluding expected_triage):

### Displaced_sa: 41 residuals

- **39 NO_QUALIFIER_AT_ALL**: pellet displacement < 5 px across all reaches. DLC cannot reliably track the pellet in these segments (many reaches show `disp=None` due to pellet not being confidently detected in pre or post windows). The pellet motion is either too subtle for DLC or occurs during the reach itself (occluded by paw). No algorithmic predicate can recover these without better tracking.
- **2 VANISH_ONLY_NO_HIGH_DISP**: displaced_sa segments where vanish fires but no displacement -- these are cid=None ambiguous cases. Would be wrong commits if committed as either class.

### Retrieved: 8 residuals

- **3 pellet_still_visible_late_in_segment**: pellet stays visible after the causal reach -- contradicts the vanish signal. These are cases where DLC continues to detect "pellet" at some position even though the mouse retrieved it (likely detecting pillar tip or debris as pellet).
- **3 sustained_off_pillar_post_first_reach**: pellet detected off-pillar after the reach -- looks like displacement, not retrieval, to the cascade. These are ambiguous DLC cases.
- **1 no_candidate_reach_for_retrieval**: no reach passes Stage 9's candidate criteria.
- **1 unannotated_paw_activity_in_gap**: Stage 26 guard blocks due to uncovered paw frames.

**Recommendation**: These 49 cases should route to human review. The 39 NO_QUALIFIER displaced_sa cases are fundamentally limited by DLC tracking quality, not by cascade logic.

## Notable Gotchas / Surprises

1. **Pillar-lk-transition is the strongest residual signal.** It perfectly separates the causal reach from non-causal reaches in multi-reach segments, for both retrieved and displaced_sa. The physics is simple: whichever reach first moves the pellet off the pillar reveals the pillar tip. This signal was already used by Stage 13 (earlier in the cascade) but only for single-reach segments with different criteria. Stages 28-29 extend it to the harder multi-reach residuals.

2. **The vanish guard is essential for class separation.** Without it, pillar-lk-transition fires on both retrieved AND displaced_sa -- the pillar becomes visible regardless of whether the pellet was taken by the mouse or knocked into the SA. Vanish (pellet disappears from DLC) distinguishes retrieval from displacement.

3. **MULTI_HIGH_DISP "first chronological" fails at 50%.** The intuition that "the first reach to displace the pellet is causal" doesn't hold -- in 3 of 6 cases, an earlier reach produces a large displacement signal but the GT-causal reach is actually a later one. Pillar-lk-transition resolves this because only one reach transitions the pillar from occluded to visible; subsequent reaches find the pillar already visible.

4. **NO_QUALIFIER is 83% of remaining displaced_sa residuals (39/47).** These segments have essentially no usable DLC signal for pellet motion. The pellet is either not detected confidently enough to measure displacement, or moves < 5 px. No threshold tuning will fix this -- it's a DLC tracking limitation.

5. **S27 edge-buffer triages (4 non-expected)**: all 4 are GT-displaced_sa where the unique high-disp reach lands within 30 frames of segment start or 60 frames of segment end. These were deliberately triaged to avoid boundary noise. Stages 28-29 don't recover them (they don't have the pillar-lk signal). They remain triaged, which is the correct conservative behavior.

## Analysis Scripts (in temp)

- `C:\Users\friedrichl\AppData\Local\Temp\residual_attack_analysis.py` -- comprehensive residual characterization
- `C:\Users\friedrichl\AppData\Local\Temp\residual_pillar_lk_deep_analysis.py` -- pillar-lk threshold sweep + IFR matching
- `C:\Users\friedrichl\AppData\Local\Temp\residual_cross_class_guard.py` -- cross-class guard analysis
- `C:\Users\friedrichl\AppData\Local\Temp\runner_test_28_29.py` -- validation runner with Stages 28+29
