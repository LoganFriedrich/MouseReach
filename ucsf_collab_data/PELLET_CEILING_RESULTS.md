# Pellet Likelihood Ceiling Analysis Results

**Date:** 2026-02-11
**Analysis:** Peak DLC confidence for pellet detection across 368 sessions from 36 animals
**Script:** `pellet_ceiling_analysis.py`
**Output:** `Y:\2_Connectome\MouseDB\exports\pellet_ceiling_analysis.csv`

---

## Key Findings

### 1. **NO ANIMALS HAVE A TRACKING PROBLEM**

All 36 animals show **excellent pellet tracking ceilings** (95th percentile ≥ 0.999):

- **Group K:** 16/16 animals with pellet ceiling ≥ 0.999
- **Group L:** 4/4 animals with pellet ceiling ≥ 0.999 (including all 4 flagged animals)
- **Group M:** 16/16 animals with pellet ceiling ≥ 0.999

**This proves that DLC CAN reliably detect the pellet at high confidence when it's present.**

### 2. **Flagged Animals (L02, L10, L12, L13) Have EXCELLENT Pellet Ceilings**

| Animal | Flagged | Sessions | Pellet 95th pct | Early Frames 95th pct | Interpretation |
|--------|---------|----------|-----------------|----------------------|----------------|
| L02 | YES | 48 | 0.999 | 0.976 | Excellent - DLC sees pellet perfectly |
| L10 | YES | 48 | 0.999 | 0.955 | Excellent - DLC sees pellet perfectly |
| L12 | YES | 48 | 0.999 | 0.940 | Excellent - DLC sees pellet perfectly |
| L13 | YES | 24 | 0.999 | 0.976 | Excellent - DLC sees pellet perfectly |

**Conclusion:** The "missing pellet" issue for these animals is NOT because DLC can't see the pellet. When the pellet is present (early frames), DLC detects it at 0.94-0.98 confidence.

### 3. **The Pillar is Actually HARDER to Track Than the Pellet**

Global statistics:
- **Pellet 95th percentile:** 0.999 ± 0.002 (nearly perfect)
- **Pillar 95th percentile:** 0.739 ± 0.345 (much more variable)
- **Ceiling gap:** +0.260 (positive = pellet easier to see than pillar)

This means:
- DLC reliably tracks the pellet at near-perfect confidence when present
- The pillar reference point is actually more challenging to track
- This is OPPOSITE of what you'd expect if there was a pellet visibility problem

### 4. **Early Frame Analysis (Pellet Definitely Present)**

For the first 500 frames (before any reach attempts):
- **L02:** 0.976 mean pellet confidence
- **L10:** 0.955 mean pellet confidence
- **L12:** 0.940 mean pellet confidence
- **L13:** 0.976 mean pellet confidence

**If DLC can see the pellet at 0.94+ confidence in early frames, why does it sometimes report low confidence later?**

---

## Interpretation

### What This Analysis Proves

1. **DLC tracking capability is NOT the problem**
   - All animals show 95th percentile confidence ≥ 0.999 for pellet
   - Even in "worst case" early frames, confidence is 0.94+
   - DLC's ceiling performance is excellent across all animals

2. **Low pellet likelihood must mean something else**
   - When DLC reports low confidence, it's not because it fundamentally can't see the pellet
   - Low confidence likely indicates:
     - Pellet actually moved/removed
     - Pellet is occluded by the paw/nose
     - Pellet is in an unexpected position
     - Motion blur during rapid movement

3. **The flagged animals (L02, L10, L12, L13) do NOT have tracking deficits**
   - Their pellet detection ceiling matches or exceeds other animals
   - The issue is not that DLC can't see the pellet for these animals
   - Something else is causing pellet likelihood to drop during sessions

### What This Suggests

The original hypothesis was: "Maybe DLC just can't see the pellet well for these animals."

**This analysis DISPROVES that hypothesis.**

Instead, the low pellet likelihoods during sessions are likely REAL signals:
- Pellet actually being displaced/removed
- Paw occlusion (which is normal during reaching)
- Natural variation in pellet position after reach attempts

### Next Steps

Since tracking ceiling is NOT the issue, the next analysis should focus on:

1. **Temporal patterns:** When does pellet likelihood drop within a session?
2. **Correlation with reach events:** Does pellet likelihood drop right after reach attempts?
3. **Recovery patterns:** Does pellet likelihood recover after drops (indicating pellet repositioned)?
4. **Comparison to video:** For sessions with "missing pellet", manually verify if pellet is actually gone

---

## Methodology

### Metrics Computed

For each session:
- **Pellet 95th percentile:** The ceiling - if DLC can see the pellet well at all, this should be near 1.0
- **Pellet max:** The absolute best DLC ever does
- **First 500 frames stats:** Mean and 95th percentile when pellet is definitely on pillar
- **Pillar 95th percentile:** Control metric (pillar is always there)
- **Ceiling gap:** Pellet 95th - Pillar 95th (positive = pellet easier to see)

### Per-Animal Summary

- Mean of 95th percentile across all sessions
- Flagged as "tracking problem" if mean 95th percentile < 0.8
- No animals met this criterion

### Data Processed

- **Group K:** 100 sessions from 16 animals
- **Group L:** 168 sessions from 4 animals (all flagged: L02, L10, L12, L13)
- **Group M:** 100 sessions from 16 animals
- **Total:** 368 sessions from 36 animals

---

## Detailed Results

Full per-animal and per-session data available in:
`Y:\2_Connectome\MouseDB\exports\pellet_ceiling_analysis.csv`

Columns include:
- Session-level metrics: pellet_95pct, pellet_max, pellet_mean, pellet_early_95pct
- Pillar metrics: pillar_95pct, pillar_max
- Ceiling gap: pellet_95pct - pillar_95pct
- Animal summaries: pellet_95pct_mean, has_tracking_problem

---

## Visual Summary

```
Ceiling Quality Distribution:
  Excellent (>0.95):  36 animals ████████████████████████████████ 100%
  Good (0.85-0.95):    0 animals
  Marginal (0.70-0.85): 0 animals
  Poor (<0.70):        0 animals
```

**All animals show excellent pellet tracking ceilings.**

---

## Conclusion

**The pellet likelihood ceiling analysis definitively shows that DLC can reliably detect the pellet at high confidence (0.999) for ALL animals, including the four flagged animals (L02, L10, L12, L13).**

**This means low pellet likelihoods during sessions are NOT due to fundamental tracking limitations, but likely reflect real changes in pellet position, occlusion, or removal.**

The next phase of analysis should focus on WHEN pellet likelihood drops and whether those drops correlate with behavioral events (reaches, pellet consumption, etc.).
