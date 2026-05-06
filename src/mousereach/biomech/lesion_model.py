"""
Parameterized Spinal Cord Lesion Model.

Models cervical contusion as dual-channel damage:
  1. Gray matter (GM) - direct motor neuron loss at lesion epicenter
  2. White matter (WM) - descending drive loss affecting all motor pools
     CAUDAL to the lesion (upper motor neuron effect)

Contusion geometry is modeled as a Gaussian damage profile with adjustable
center, severity, and width. The GM and WM channels have independent
severity and spread parameters, reflecting that white matter damage extends
further rostro-caudally than gray matter (Solt et al. 2020, dorsal column
spread factor).

References:
    Forgione et al. 2022 - Graded hemicontusion parameters (mouse C5)
    Solt et al. 2020 - Contusion geometry follows anatomical tracts
    Lee et al. 2012 - IH impactor C5, 4.8mm lesion spread
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .nuclei_atlas import NucleiAtlas, SEGMENT_POSITIONS, SEGMENT_LABELS


# Forgione 2022 reference parameters (mouse C5 hemicontusion)
FORGIONE_MILD = {  # 1.2mm displacement
    'gm_spared': 0.368,   # 36.8 +/- 2.4%
    'wm_spared': 0.124,   # 12.4 +/- 2.6%
    'lesion_fraction': 0.208,  # 20.8 +/- 3.0% of cord
}
FORGIONE_SEVERE = {  # 1.5mm displacement
    'gm_spared': 0.276,   # 27.6 +/- 4.0%
    'wm_spared': 0.041,   # 4.1 +/- 2.2%
    'lesion_fraction': 0.360,  # 36.0 +/- 2.1% of cord
}

# Mouse cord dimensions (approximate)
CORD_DV_DIAMETER_MM = 1.75  # Dorso-ventral at C5 (~1.5-2.0mm)
IH_TIP_DIAMETER_MM = 1.0    # Standard IH impactor tip for mouse cervical


@dataclass
class LesionParameters:
    """Parameters defining a virtual spinal cord lesion."""

    # Position: float from 3.0 (C3) to 10.0 (T2)
    center: float = 5.0  # Default: C5

    # Gray matter damage
    gm_severity: float = 0.7   # 0.0 = intact, 1.0 = complete GM destruction
    gm_width: float = 0.75     # Gaussian sigma in segment units

    # White matter damage
    wm_severity: float = 0.5   # 0.0 = intact, 1.0 = complete WM destruction
    wm_width: float = 1.5      # WM spreads further than GM (dorsal column factor)

    # Laterality: 1.0 = fully unilateral (hemicontusion), 0.0 = bilateral
    # For unilateral, only ipsilateral motor pools are affected
    laterality: float = 0.0  # Default: bilateral (standard contusion)

    # Dorsal column spread factor: additional WM damage extension
    # rostro-caudally via dorsal columns (Solt et al. 2020)
    dorsal_column_factor: float = 1.3  # Multiplier on wm_width

    def __post_init__(self):
        """Validate parameters."""
        if not 3.0 <= self.center <= 10.0:
            raise ValueError(f"center must be 3.0-10.0, got {self.center}")
        for name, val in [
            ('gm_severity', self.gm_severity),
            ('wm_severity', self.wm_severity),
            ('laterality', self.laterality),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be 0.0-1.0, got {val}")
        for name, val in [
            ('gm_width', self.gm_width),
            ('wm_width', self.wm_width),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

    @classmethod
    def from_forgione_mild(cls, center: float = 5.0) -> 'LesionParameters':
        """Create parameters matching Forgione 2022 1.2mm displacement."""
        return cls(
            center=center,
            gm_severity=1.0 - FORGIONE_MILD['gm_spared'],
            wm_severity=1.0 - FORGIONE_MILD['wm_spared'],
            gm_width=0.75,
            wm_width=1.5,
            laterality=1.0,  # Forgione used hemicontusion
        )

    @classmethod
    def from_forgione_severe(cls, center: float = 5.0) -> 'LesionParameters':
        """Create parameters matching Forgione 2022 1.5mm displacement."""
        return cls(
            center=center,
            gm_severity=1.0 - FORGIONE_SEVERE['gm_spared'],
            wm_severity=1.0 - FORGIONE_SEVERE['wm_spared'],
            gm_width=1.0,
            wm_width=2.0,
            laterality=1.0,
        )


class LesionModel:
    """
    Computes per-muscle force attenuation from a parameterized spinal lesion.

    Two damage channels compound multiplicatively:
      force_scale = gm_scale * wm_scale

    where:
      gm_scale = 1 - sum(gm_damage * innervation_weight) across segments
      wm_scale = 1 - wm_damage at the most rostral segment overlapping
                 the muscle's motor pool (descending input loss)
    """

    def __init__(self, atlas: Optional[NucleiAtlas] = None):
        """
        Args:
            atlas: NucleiAtlas instance. If None, loads the bundled atlas.
        """
        self.atlas = atlas or NucleiAtlas()

    def compute_gm_damage_profile(self, params: LesionParameters) -> np.ndarray:
        """
        Compute gray matter damage at each segment position.

        Returns:
            Array of shape (n_segments,) with values 0.0-1.0
        """
        damage = params.gm_severity * np.exp(
            -0.5 * ((SEGMENT_POSITIONS - params.center) / params.gm_width) ** 2
        )
        return np.clip(damage, 0.0, 1.0)

    def compute_wm_damage_profile(self, params: LesionParameters) -> np.ndarray:
        """
        Compute white matter damage at each segment position.

        WM damage has a wider spread than GM (dorsal column factor) and
        its effect on motor pools is felt CAUDAL to the damage site
        (descending drive interruption).

        Returns:
            Array of shape (n_segments,) with values 0.0-1.0
        """
        effective_width = params.wm_width * params.dorsal_column_factor
        damage = params.wm_severity * np.exp(
            -0.5 * ((SEGMENT_POSITIONS - params.center) / effective_width) ** 2
        )
        return np.clip(damage, 0.0, 1.0)

    def compute_attenuation(
        self, params: LesionParameters
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-muscle force attenuation from lesion parameters.

        Args:
            params: LesionParameters defining the virtual lesion

        Returns:
            Dict mapping muscle name -> {
                'gm_scale': float,      # Gray matter surviving fraction
                'wm_scale': float,      # White matter surviving fraction
                'force_scale': float,   # Combined (gm * wm)
                'force_loss': float,    # 1.0 - force_scale (convenience)
            }
        """
        gm_damage = self.compute_gm_damage_profile(params)
        wm_damage = self.compute_wm_damage_profile(params)

        results = {}
        for i, muscle in enumerate(self.atlas.muscles):
            innervation = self.atlas.density[i]

            # GM channel: weighted damage across motor pool
            gm_loss = np.sum(gm_damage * innervation)
            gm_scale = 1.0 - np.clip(gm_loss, 0.0, 1.0)

            # WM channel: descending drive loss
            # Affected segments are those CAUDAL to (or at) the lesion center
            # The drive loss at a motor pool = WM damage at the most rostral
            # segment where that muscle has innervation
            muscle_segments = np.where(innervation > 0)[0]
            if len(muscle_segments) > 0:
                most_rostral = muscle_segments[0]
                # WM damage affects all segments caudal to it
                # The drive loss = max WM damage at or rostral to this pool
                rostral_positions = SEGMENT_POSITIONS[:most_rostral + 1]
                rostral_wm = wm_damage[:most_rostral + 1]
                if len(rostral_wm) > 0:
                    wm_loss = float(np.max(rostral_wm))
                else:
                    wm_loss = 0.0
            else:
                wm_loss = 0.0

            wm_scale = 1.0 - np.clip(wm_loss, 0.0, 1.0)

            # Combined: both channels compound
            force_scale = gm_scale * wm_scale

            results[muscle] = {
                'gm_scale': float(gm_scale),
                'wm_scale': float(wm_scale),
                'force_scale': float(force_scale),
                'force_loss': float(1.0 - force_scale),
            }

        return results

    def sweep_center(
        self,
        centers: Optional[np.ndarray] = None,
        base_params: Optional[LesionParameters] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Sweep lesion center across C4-T1, returning force scale matrix.

        Args:
            centers: Array of lesion center positions. Default: C4 to T1
                     in 0.25 steps.
            base_params: Base parameters (severity, width, etc.). Center
                        will be overridden at each sweep step.

        Returns:
            Tuple of:
            - centers: array of center positions
            - force_matrix: array of shape (n_centers, n_muscles) with force scales
            - muscle_names: list of muscle names (column order)
        """
        if centers is None:
            centers = np.arange(4.0, 9.25, 0.25)  # C4 to T1

        if base_params is None:
            base_params = LesionParameters()

        force_matrix = np.zeros((len(centers), self.atlas.n_muscles))

        for i, center in enumerate(centers):
            params = LesionParameters(
                center=center,
                gm_severity=base_params.gm_severity,
                gm_width=base_params.gm_width,
                wm_severity=base_params.wm_severity,
                wm_width=base_params.wm_width,
                laterality=base_params.laterality,
                dorsal_column_factor=base_params.dorsal_column_factor,
            )
            attenuation = self.compute_attenuation(params)
            for j, muscle in enumerate(self.atlas.muscles):
                force_matrix[i, j] = attenuation[muscle]['force_scale']

        return centers, force_matrix, list(self.atlas.muscles)

    def compare_lesions(
        self,
        params_a: LesionParameters,
        params_b: LesionParameters,
        label_a: str = "Lesion A",
        label_b: str = "Lesion B",
    ) -> str:
        """
        Compare two lesion configurations side-by-side.

        Returns:
            Human-readable comparison string.
        """
        att_a = self.compute_attenuation(params_a)
        att_b = self.compute_attenuation(params_b)

        lines = [f'{"Muscle":30s}  {label_a:>12s}  {label_b:>12s}  {"Delta":>8s}']
        lines.append('-' * 70)

        for muscle in self.atlas.muscles:
            fa = att_a[muscle]['force_scale']
            fb = att_b[muscle]['force_scale']
            delta = fb - fa
            sign = '+' if delta > 0 else ''
            lines.append(
                f'{muscle:30s}  {fa:11.1%}  {fb:11.1%}  {sign}{delta:7.1%}'
            )

        return '\n'.join(lines)
