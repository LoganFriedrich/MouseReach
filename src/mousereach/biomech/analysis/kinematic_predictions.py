"""
Simplified Kinematic Predictions (Phase 3).

Sweeps virtual lesion position across C4-T1 and predicts direction/magnitude
of change for each MouseReach kinematic feature. Compares predictions against
actual post-injury data to identify which lesion position best explains each
animal's kinematic profile.

This is the quick-win analysis -- no OpenSim needed, pure numpy/matplotlib.

Usage:
    from mousereach.biomech.analysis.kinematic_predictions import (
        LesionSweepAnalysis, plot_lesion_sweep_heatmap
    )

    analysis = LesionSweepAnalysis()
    results = analysis.run_sweep()
    plot_lesion_sweep_heatmap(results, output_path='lesion_sweep.png')
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..lesion_model import LesionModel, LesionParameters
from ..muscle_mapping import (
    GILMER_MUSCLES, GILMER_BY_NAME, FEATURE_MUSCLE_MAP,
    predict_feature_shift, total_joint_force, intact_joint_forces,
)
from ..nuclei_atlas import NucleiAtlas, SEGMENT_LABELS


# MouseReach kinematic features to predict
PREDICTION_FEATURES = [
    'max_extent_mm',
    'peak_velocity_px_per_frame',
    'mean_velocity_px_per_frame',
    'trajectory_straightness',
    'trajectory_smoothness',
    'duration_frames',
]


class LesionSweepAnalysis:
    """
    Sweep lesion center and severity, predict kinematic feature changes.
    """

    def __init__(self, atlas: Optional[NucleiAtlas] = None):
        self.atlas = atlas or NucleiAtlas()
        self.model = LesionModel(self.atlas)

    def run_sweep(
        self,
        centers: Optional[np.ndarray] = None,
        severities: Optional[List[float]] = None,
        gm_width: float = 0.75,
        wm_severity_ratio: float = 0.7,
        wm_width: float = 1.5,
    ) -> Dict:
        """
        Run full lesion sweep.

        Args:
            centers: Lesion center positions. Default: C4 to T1 in 0.25 steps.
            severities: GM severity values to sweep. Default: [0.5, 0.7, 0.9].
            gm_width: Gaussian width for gray matter damage.
            wm_severity_ratio: WM severity as fraction of GM severity.
            wm_width: Gaussian width for white matter damage.

        Returns:
            Dict with:
                'centers': array of positions
                'severities': list of severity values
                'features': list of feature names
                'predictions': dict[severity][feature] -> array of (direction, magnitude)
                    per center position
                'force_matrix': dict[severity] -> array (n_centers, n_muscles)
                'muscle_names': list of muscle names
                'joint_forces': dict[severity] -> dict[center] -> joint force breakdown
        """
        if centers is None:
            centers = np.arange(4.0, 9.25, 0.25)
        if severities is None:
            severities = [0.5, 0.7, 0.9]

        results = {
            'centers': centers,
            'severities': severities,
            'features': PREDICTION_FEATURES,
            'predictions': {},
            'force_matrix': {},
            'muscle_names': list(self.atlas.muscles),
            'joint_forces': {},
        }

        for sev in severities:
            predictions = {f: [] for f in PREDICTION_FEATURES}
            force_matrix = np.zeros((len(centers), self.atlas.n_muscles))
            joint_forces = {}

            for i, center in enumerate(centers):
                params = LesionParameters(
                    center=center,
                    gm_severity=sev,
                    gm_width=gm_width,
                    wm_severity=sev * wm_severity_ratio,
                    wm_width=wm_width,
                )
                att = self.model.compute_attenuation(params)

                # Force scales per muscle
                for j, muscle in enumerate(self.atlas.muscles):
                    force_matrix[i, j] = att[muscle]['force_scale']

                # Feature predictions
                for feat in PREDICTION_FEATURES:
                    pred = predict_feature_shift(feat, att)
                    predictions[feat].append({
                        'direction': pred['direction'],
                        'magnitude': pred['magnitude'],
                    })

                # Joint force breakdown
                jf = {}
                for joint in ['shoulder', 'elbow']:
                    jf[joint] = {}
                    for m in GILMER_MUSCLES:
                        if m.joint == joint:
                            if m.action not in jf[joint]:
                                jf[joint][m.action] = 0.0
                            scale = att.get(m.atlas_name, {}).get('force_scale', 1.0)
                            jf[joint][m.action] += m.max_iso_force_n * scale
                joint_forces[f'{center:.2f}'] = jf

            results['predictions'][sev] = predictions
            results['force_matrix'][sev] = force_matrix
            results['joint_forces'][sev] = joint_forces

        return results

    def compare_to_real_data(
        self,
        sweep_results: Dict,
        real_features: pd.DataFrame,
        baseline_features: pd.DataFrame,
        animal_col: str = 'subject_id',
    ) -> pd.DataFrame:
        """
        Compare sweep predictions to actual post-injury kinematic shifts.

        Args:
            sweep_results: Output of run_sweep()
            real_features: DataFrame of post-injury kinematic features per animal
            baseline_features: DataFrame of pre-injury baseline features per animal
            animal_col: Column name for animal ID

        Returns:
            DataFrame with per-animal best-fit lesion position and agreement score
        """
        centers = sweep_results['centers']
        # Use middle severity
        sev = sweep_results['severities'][len(sweep_results['severities']) // 2]
        predictions = sweep_results['predictions'][sev]

        animals = real_features[animal_col].unique()
        results_rows = []

        for animal in animals:
            animal_post = real_features[real_features[animal_col] == animal]
            animal_base = baseline_features[baseline_features[animal_col] == animal]

            if len(animal_post) == 0 or len(animal_base) == 0:
                continue

            # Compute observed shift direction for each feature
            observed_shifts = {}
            for feat in PREDICTION_FEATURES:
                if feat in animal_post.columns and feat in animal_base.columns:
                    post_mean = animal_post[feat].mean()
                    base_mean = animal_base[feat].mean()
                    if base_mean != 0:
                        pct_change = (post_mean - base_mean) / abs(base_mean)
                    else:
                        pct_change = 0
                    # Direction: +1 if increased, -1 if decreased
                    if abs(pct_change) > 0.05:
                        observed_shifts[feat] = +1 if pct_change > 0 else -1
                    else:
                        observed_shifts[feat] = 0

            # Score each lesion position by sign agreement
            best_score = -1
            best_center = centers[0]
            scores = []

            for i, center in enumerate(centers):
                agreement = 0
                n_compared = 0
                for feat in PREDICTION_FEATURES:
                    if feat in observed_shifts:
                        pred_dir = predictions[feat][i]['direction']
                        obs_dir = observed_shifts[feat]
                        if pred_dir != 0 and obs_dir != 0:
                            agreement += 1 if pred_dir == obs_dir else -1
                            n_compared += 1

                score = agreement / n_compared if n_compared > 0 else 0
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_center = center

            results_rows.append({
                'animal': animal,
                'best_fit_center': best_center,
                'best_fit_score': best_score,
                'observed_shifts': observed_shifts,
                'all_scores': scores,
            })

        return pd.DataFrame(results_rows)


def plot_lesion_sweep_heatmap(
    sweep_results: Dict,
    severity: Optional[float] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14, 8),
):
    """
    Plot heatmap of muscle force retention across lesion positions.

    Args:
        sweep_results: Output of LesionSweepAnalysis.run_sweep()
        severity: Which severity to plot. Default: middle value.
        output_path: Save figure to this path. If None, displays.
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if severity is None:
        severity = sweep_results['severities'][len(sweep_results['severities']) // 2]

    centers = sweep_results['centers']
    force_matrix = sweep_results['force_matrix'][severity]
    muscle_names = sweep_results['muscle_names']

    # Custom colormap: red (lost) -> white (50%) -> green (intact)
    colors = [(0.8, 0.1, 0.1), (1, 1, 1), (0.1, 0.7, 0.1)]
    cmap = LinearSegmentedColormap.from_list('damage', colors, N=256)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1],
                                     gridspec_kw={'hspace': 0.3})

    # Heatmap
    im = ax1.imshow(force_matrix.T, aspect='auto', cmap=cmap,
                     vmin=0, vmax=1, interpolation='nearest')

    # Labels
    tick_positions = range(0, len(centers), 2)
    tick_labels = [f'C{c:.0f}' if c <= 8 else f'T{c-8:.0f}'
                   for c in centers[::2]]
    ax1.set_xticks(list(tick_positions))
    ax1.set_xticklabels(tick_labels, fontsize=9)
    ax1.set_yticks(range(len(muscle_names)))
    ax1.set_yticklabels(muscle_names, fontsize=8)
    ax1.set_xlabel('Lesion Center')
    ax1.set_title(f'Muscle Force Retention by Lesion Position '
                  f'(severity={severity:.0%})', fontsize=12)

    cbar = fig.colorbar(im, ax=ax1, shrink=0.8, label='Force Retention')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Feature prediction panel
    for feat in PREDICTION_FEATURES:
        preds = sweep_results['predictions'][severity][feat]
        magnitudes = [p['magnitude'] * p['direction'] for p in preds]
        ax2.plot(range(len(centers)), magnitudes, label=feat.replace('_', ' '),
                 linewidth=1.5)

    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xticks(list(tick_positions))
    ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_xlabel('Lesion Center')
    ax2.set_ylabel('Predicted Change')
    ax2.set_title('Kinematic Feature Predictions', fontsize=11)
    ax2.legend(fontsize=7, loc='upper right', ncol=2)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_c5_vs_c6_comparison(
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 6),
):
    """
    Generate the key figure: C5 vs C6 contusion muscle/kinematic comparison.
    """
    import matplotlib.pyplot as plt

    model = LesionModel()
    c5 = LesionParameters(center=5.0, gm_severity=0.7, wm_severity=0.5)
    c6 = LesionParameters(center=6.0, gm_severity=0.7, wm_severity=0.5)

    att_c5 = model.compute_attenuation(c5)
    att_c6 = model.compute_attenuation(c6)

    muscles = list(att_c5.keys())
    c5_forces = [att_c5[m]['force_scale'] for m in muscles]
    c6_forces = [att_c6[m]['force_scale'] for m in muscles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Muscle force comparison
    y = np.arange(len(muscles))
    ax1.barh(y - 0.2, c5_forces, 0.35, label='C5 Contusion', color='#d32f2f')
    ax1.barh(y + 0.2, c6_forces, 0.35, label='C6 Contusion', color='#1976d2')
    ax1.set_yticks(y)
    ax1.set_yticklabels(muscles, fontsize=7)
    ax1.set_xlabel('Force Retention')
    ax1.set_xlim(0, 1)
    ax1.set_title('Muscle Force: C5 vs C6')
    ax1.legend(fontsize=9)
    ax1.invert_yaxis()

    # Kinematic predictions
    features = list(FEATURE_MUSCLE_MAP.keys())
    c5_preds = [predict_feature_shift(f, att_c5) for f in features]
    c6_preds = [predict_feature_shift(f, att_c6) for f in features]

    y2 = np.arange(len(features))
    c5_vals = [p['direction'] * p['magnitude'] for p in c5_preds]
    c6_vals = [p['direction'] * p['magnitude'] for p in c6_preds]

    ax2.barh(y2 - 0.2, c5_vals, 0.35, label='C5', color='#d32f2f')
    ax2.barh(y2 + 0.2, c6_vals, 0.35, label='C6', color='#1976d2')
    ax2.set_yticks(y2)
    ax2.set_yticklabels([f.replace('_', '\n') for f in features], fontsize=7)
    ax2.set_xlabel('Predicted Shift (+ = increase)')
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_title('Predicted Kinematic Changes')
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig
