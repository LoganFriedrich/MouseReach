"""
Outcome detection v5 -- pattern-discovery based per-pellet outcome
classifier.

Replaces the rules-based v4.x.x outcome detector. Operates at the
PER-PELLET (per-segment) level. For each segment, predicts:
  - outcome_label: retrieved / displaced_sa / untouched / abnormal_exception
  - interaction_frame: regression (frame within segment)
  - outcome_known_frame: regression (frame within segment)

displaced_outside is collapsed to displaced_sa per the existing
metrics standard.

Trained against full 47-video GT corpus (outcome labels are per-segment
complete in both exhaustive and supplementary kinds, verified in
Phase A). LOOCV across the 37-video train pool. 10 videos held out
for Phase E.
"""
VERSION = "5.0.0_dev"
