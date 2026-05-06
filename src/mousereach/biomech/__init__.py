"""
Biomechanical Lesion Simulator.

Virtual contusion simulator that models parameterized spinal cord injuries
along C4-T1, attenuates motor pools based on published anatomy, and predicts
kinematic trajectories via musculoskeletal modeling.

Submodules:
    nuclei_atlas    - Motor pool lookup table (segment -> muscle innervation)
    lesion_model    - Dual-channel damage model (gray matter + white matter)
    muscle_mapping  - Gilmer 21-muscle model <-> atlas crosswalk
    dlc_to_trc      - DLC .h5 to OpenSim .trc format converter
    analysis/       - Classification, bootstrap, visualization
"""
