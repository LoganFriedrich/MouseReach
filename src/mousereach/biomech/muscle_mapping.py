"""
Gilmer Model <-> Nuclei Atlas Muscle Mapping.

Maps the 21 muscles in the Gilmer et al. (2025) mouse forelimb OpenSim model
to their corresponding entries in the nuclei atlas. Provides the crosswalk
between biomechanical simulation (force generation) and neuroanatomy
(spinal innervation).

Also defines qualitative muscle -> kinematic feature mappings for the
simplified prediction pathway (Phase 3, no OpenSim needed).

Confidence tiers:
  HIGH   - Direct match to published mouse tracing data
  MEDIUM - Interpolated from related muscles or cross-species data
  LOW    - Inferred from topographic rules, no direct data
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MuscleEntry:
    """One muscle in the Gilmer model with atlas and kinematic mappings."""
    gilmer_id: int
    gilmer_name: str
    atlas_name: str           # Name in nuclei_atlas.csv
    max_iso_force_n: float    # From Gilmer Table 3
    joint: str                # 'shoulder', 'elbow', 'wrist'
    action: str               # Primary action (e.g. 'flexion', 'extension')
    confidence: str           # 'high', 'medium', 'low'


# Complete 21-muscle table from Gilmer et al. 2025 (Table 3)
GILMER_MUSCLES: List[MuscleEntry] = [
    # -- Shoulder (9 muscles) --
    MuscleEntry(1,  'Deltoid Medial',              'Deltoid Medial',              0.069, 'shoulder', 'abduction/elevation', 'high'),
    MuscleEntry(2,  'Deltoid Posterior',            'Deltoid Posterior',           0.068, 'shoulder', 'extension',           'high'),
    MuscleEntry(3,  'Infraspinatus',               'Infraspinatus',               0.065, 'shoulder', 'external rotation',   'medium'),
    MuscleEntry(4,  'Subscapularis',               'Subscapularis',               0.340, 'shoulder', 'internal rotation',   'medium'),
    MuscleEntry(5,  'Pectoralis Major Anterior',   'Pectoralis Major Anterior',   0.233, 'shoulder', 'flexion/adduction',   'medium'),
    MuscleEntry(6,  'Pectoralis Major Posterior',   'Pectoralis Major Posterior', 0.170, 'shoulder', 'extension/adduction', 'medium'),
    MuscleEntry(7,  'Pectoralis Minor Clavicular', 'Pectoralis Minor Clavicular', 0.033, 'shoulder', 'depression',          'low'),
    MuscleEntry(8,  'Latissimus Dorsi Caudal',     'Latissimus Dorsi Caudal',     0.133, 'shoulder', 'extension/adduction', 'medium'),
    MuscleEntry(9,  'Latissimus Dorsi Rostral',    'Latissimus Dorsi Rostral',    0.113, 'shoulder', 'extension/adduction', 'low'),

    # -- Elbow flexors (4 muscles) --
    MuscleEntry(10, 'Biceps Long Head',            'Biceps Long Head',            0.093, 'elbow', 'flexion',   'high'),
    MuscleEntry(11, 'Biceps Short Head',           'Biceps Short Head',           0.018, 'elbow', 'flexion',   'high'),
    MuscleEntry(12, 'Brachialis Proximal',         'Brachialis Proximal',         0.066, 'elbow', 'flexion',   'medium'),
    MuscleEntry(13, 'Brachialis Distal',           'Brachialis Distal',           0.067, 'elbow', 'flexion',   'low'),

    # -- Elbow extensors (5 muscles) --
    MuscleEntry(14, 'Triceps Long Head',           'Triceps Long Head',           0.612, 'elbow', 'extension', 'high'),
    MuscleEntry(15, 'Triceps Lateral Head',        'Triceps Lateral Head',        0.125, 'elbow', 'extension', 'high'),
    MuscleEntry(16, 'Triceps Medial Head',         'Triceps Medial Head',         0.160, 'elbow', 'extension', 'high'),
    MuscleEntry(17, 'Anconeus',                    'Anconeus',                    0.023, 'elbow', 'extension', 'medium'),
    MuscleEntry(18, 'Anconeus Short Head',         'Anconeus Short Head',         0.020, 'elbow', 'extension', 'medium'),

    # -- Forearm/wrist (3 muscles) --
    MuscleEntry(19, 'Brachioradialis',             'Brachioradialis',             0.020, 'elbow', 'flexion',   'medium'),
    MuscleEntry(20, 'Pronator Teres',              'Pronator Teres',              0.020, 'elbow', 'pronation', 'low'),
    MuscleEntry(21, 'Flexor Carpi Radialis',       'Flexor Carpi Radialis',       0.020, 'wrist', 'flexion',   'high'),
]

# Lookup by gilmer_id
GILMER_BY_ID = {m.gilmer_id: m for m in GILMER_MUSCLES}
GILMER_BY_NAME = {m.gilmer_name: m for m in GILMER_MUSCLES}


def get_muscle(name: str) -> MuscleEntry:
    """Look up a muscle by Gilmer name."""
    return GILMER_BY_NAME[name]


def get_joint_muscles(joint: str) -> List[MuscleEntry]:
    """Get all muscles acting on a joint ('shoulder', 'elbow', 'wrist')."""
    return [m for m in GILMER_MUSCLES if m.joint == joint]


def get_flexors(joint: str) -> List[MuscleEntry]:
    """Get flexor muscles for a joint."""
    return [m for m in GILMER_MUSCLES
            if m.joint == joint and 'flexion' in m.action]


def get_extensors(joint: str) -> List[MuscleEntry]:
    """Get extensor muscles for a joint."""
    return [m for m in GILMER_MUSCLES
            if m.joint == joint and 'extension' in m.action]


def total_joint_force(
    joint: str,
    action: str,
    attenuation: Dict[str, Dict[str, float]]
) -> float:
    """
    Compute total available force for a joint action after lesion attenuation.

    Args:
        joint: 'shoulder', 'elbow', or 'wrist'
        action: substring to match (e.g. 'flexion', 'extension')
        attenuation: Output of LesionModel.compute_attenuation()

    Returns:
        Total force in Newtons (attenuated max isometric forces summed)
    """
    total = 0.0
    for m in GILMER_MUSCLES:
        if m.joint == joint and action in m.action:
            scale = attenuation.get(m.atlas_name, {}).get('force_scale', 1.0)
            total += m.max_iso_force_n * scale
    return total


def intact_joint_forces() -> Dict[str, Dict[str, float]]:
    """
    Get intact (no lesion) force capacity for each joint action.

    Returns:
        Dict[joint][action] -> total force in Newtons
    """
    forces = {}
    for m in GILMER_MUSCLES:
        key = m.joint
        if key not in forces:
            forces[key] = {}
        if m.action not in forces[key]:
            forces[key][m.action] = 0.0
        forces[key][m.action] += m.max_iso_force_n
    return forces


# ---------------------------------------------------------------
# Qualitative kinematic feature mappings (Phase 3 predictions)
# Maps MouseReach kinematic features to the muscles that drive them
# ---------------------------------------------------------------

FEATURE_MUSCLE_MAP = {
    # Reach extent -- driven by shoulder elevation + elbow extension
    'max_extent_mm': {
        'positive': ['Deltoid Medial', 'Deltoid Posterior',
                     'Pectoralis Major Anterior'],
        'negative': ['Triceps Long Head', 'Triceps Lateral Head',
                     'Triceps Medial Head'],  # Braking on retraction
        'mechanism': 'Shoulder drives reach forward, triceps brakes retraction. '
                     'Loss of shoulder = shorter reach. Loss of triceps = '
                     'overshooting (longer, unbraked reach).',
    },

    # Velocity
    'peak_velocity_px_per_frame': {
        'positive': ['Deltoid Medial', 'Biceps Long Head', 'Biceps Short Head',
                     'Pectoralis Major Anterior'],
        'negative': ['Triceps Long Head'],  # Eccentric braking
        'mechanism': 'Peak velocity depends on agonist strength. Loss of '
                     'agonists = slower. Loss of antagonist braking = '
                     'potentially faster but less controlled.',
    },

    # Trajectory straightness
    'trajectory_straightness': {
        'positive': [],  # All muscles contribute to balanced straight path
        'negative': [],  # Any imbalance reduces straightness
        'mechanism': 'Straightness depends on balanced agonist/antagonist '
                     'co-contraction. Any selective muscle loss creates '
                     'deviation from straight-line path.',
        'special': 'agonist_antagonist_balance',
    },

    # Smoothness (inverse jerk)
    'trajectory_smoothness': {
        'positive': [],
        'negative': [],
        'mechanism': 'Smoothness depends on coordinated muscle activation '
                     'timing. Denervation causes jerky compensatory movements.',
        'special': 'total_force_loss',
    },

    # Duration
    'duration_frames': {
        'positive': [],  # Weakness -> slower -> longer duration
        'negative': ['Deltoid Medial', 'Biceps Long Head'],
        'mechanism': 'Weaker muscles = slower movements = longer reach duration.',
        'special': 'inverse_total_force',
    },
}


def predict_feature_shift(
    feature: str,
    attenuation: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Predict direction and magnitude of a kinematic feature shift under lesion.

    This is the simplified Phase 3 prediction -- qualitative direction and
    rough magnitude based on which muscles are affected, without running
    OpenSim.

    Args:
        feature: MouseReach feature name (e.g. 'max_extent_mm')
        attenuation: Output of LesionModel.compute_attenuation()

    Returns:
        Dict with 'direction' (-1, 0, +1), 'magnitude' (0-1),
        'explanation' (human-readable).
    """
    if feature not in FEATURE_MUSCLE_MAP:
        return {'direction': 0, 'magnitude': 0.0,
                'explanation': f'No mapping for {feature}'}

    mapping = FEATURE_MUSCLE_MAP[feature]

    # Compute force loss for positive and negative contributor groups
    pos_loss = 0.0
    pos_total = 0.0
    for name in mapping.get('positive', []):
        m = GILMER_BY_NAME.get(name)
        if m:
            scale = attenuation.get(name, {}).get('force_scale', 1.0)
            pos_loss += m.max_iso_force_n * (1.0 - scale)
            pos_total += m.max_iso_force_n

    neg_loss = 0.0
    neg_total = 0.0
    for name in mapping.get('negative', []):
        m = GILMER_BY_NAME.get(name)
        if m:
            scale = attenuation.get(name, {}).get('force_scale', 1.0)
            neg_loss += m.max_iso_force_n * (1.0 - scale)
            neg_total += m.max_iso_force_n

    # Handle special cases
    special = mapping.get('special')
    if special == 'agonist_antagonist_balance':
        # Any selective loss reduces straightness
        total_loss = sum(
            GILMER_BY_NAME[m.gilmer_name].max_iso_force_n
            * (1.0 - attenuation.get(m.atlas_name, {}).get('force_scale', 1.0))
            for m in GILMER_MUSCLES
        )
        total_capacity = sum(m.max_iso_force_n for m in GILMER_MUSCLES)
        magnitude = total_loss / total_capacity if total_capacity > 0 else 0
        return {
            'direction': -1 if magnitude > 0.05 else 0,
            'magnitude': float(magnitude),
            'explanation': 'Selective muscle loss disrupts agonist/antagonist balance',
        }

    if special == 'total_force_loss':
        total_loss = sum(
            GILMER_BY_NAME[m.gilmer_name].max_iso_force_n
            * (1.0 - attenuation.get(m.atlas_name, {}).get('force_scale', 1.0))
            for m in GILMER_MUSCLES
        )
        total_capacity = sum(m.max_iso_force_n for m in GILMER_MUSCLES)
        magnitude = total_loss / total_capacity if total_capacity > 0 else 0
        return {
            'direction': -1 if magnitude > 0.05 else 0,
            'magnitude': float(magnitude),
            'explanation': 'Overall force loss reduces movement smoothness',
        }

    if special == 'inverse_total_force':
        total_loss = sum(
            GILMER_BY_NAME[m.gilmer_name].max_iso_force_n
            * (1.0 - attenuation.get(m.atlas_name, {}).get('force_scale', 1.0))
            for m in GILMER_MUSCLES
        )
        total_capacity = sum(m.max_iso_force_n for m in GILMER_MUSCLES)
        magnitude = total_loss / total_capacity if total_capacity > 0 else 0
        return {
            'direction': +1 if magnitude > 0.05 else 0,
            'magnitude': float(magnitude),
            'explanation': 'Force loss slows movement, increasing duration',
        }

    # Standard case: positive drivers lost = feature decreases,
    # negative drivers (brakes) lost = feature increases
    pos_frac = pos_loss / pos_total if pos_total > 0 else 0
    neg_frac = neg_loss / neg_total if neg_total > 0 else 0

    # Net effect: losing positive drivers decreases feature,
    # losing negative drivers (brakes) increases it
    net = neg_frac - pos_frac
    magnitude = abs(net)

    if net > 0.05:
        direction = +1
        explanation = (f'Brake muscles lost ({neg_frac:.0%}) > '
                      f'drive muscles lost ({pos_frac:.0%}) -> feature increases')
    elif net < -0.05:
        direction = -1
        explanation = (f'Drive muscles lost ({pos_frac:.0%}) > '
                      f'brake muscles lost ({neg_frac:.0%}) -> feature decreases')
    else:
        direction = 0
        explanation = 'Balanced loss, no clear directional shift'

    return {
        'direction': direction,
        'magnitude': float(magnitude),
        'explanation': explanation,
    }
