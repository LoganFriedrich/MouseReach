"""
Motor Pool Nuclei Atlas.

Maps cervical spinal segments (C3-T2) to forelimb muscle innervation using
published retrograde tracing data. Each muscle's motor neuron density across
segments is modeled as a Gaussian centered at the peak segment, with sigma
derived from the span (sigma = span/4, placing 68% of neurons in the central
half of the span).

Sources:
    Tosolini & Morris 2013 (mouse, 9 muscles, Fluoro-Gold full MEP)
    Bacskai et al. 2012 (mouse, 11 muscle groups, Fluoro-Gold)
    Lu/Qi et al. 2022 (mouse, 14 muscles, 3D clearing + optical imaging)
    McKenna et al. 2000 (rat, 14 muscles, cross-species reference)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Segment labels and their numeric positions (0.5-segment resolution)
# C3=3.0, C3.5=3.5, C4=4.0, ... C8=8.0, T1=9.0, T2=10.0
SEGMENT_LABELS = [
    'C3', 'C3.5', 'C4', 'C4.5', 'C5', 'C5.5', 'C6', 'C6.5',
    'C7', 'C7.5', 'C8', 'C8.5', 'T1', 'T1.5', 'T2'
]

SEGMENT_POSITIONS = np.array([
    3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
    7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
])

# Approximate segment length in mm (Harrison et al. 2013)
SEGMENT_LENGTH_MM = 1.2  # ~1.0-1.4mm per segment, use midpoint


def _parse_segment(s: str) -> float:
    """Convert segment label to numeric position. E.g. 'C5' -> 5.0, 'T1' -> 9.0."""
    s = str(s).strip()
    if s.startswith('C'):
        return float(s[1:])
    elif s.startswith('T'):
        return 8.0 + float(s[1:])
    else:
        # Try parsing as plain float (e.g. '5.5' for peak_segment)
        return float(s)


class NucleiAtlas:
    """
    Motor pool atlas mapping spinal segments to muscle innervation density.

    The atlas stores a density matrix: muscles x segments, where each entry
    represents the fraction of a muscle's motor neurons located at that segment.
    Rows sum to 1.0.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Load atlas from CSV.

        Args:
            csv_path: Path to nuclei_atlas.csv. If None, uses the bundled CSV.
        """
        if csv_path is None:
            csv_path = Path(__file__).parent / 'nuclei_atlas.csv'

        self.raw = pd.read_csv(csv_path)
        self.muscles = list(self.raw['muscle'])
        self.gilmer_ids = dict(zip(self.raw['muscle'], self.raw['gilmer_id'].astype(int)))
        self.n_muscles = len(self.muscles)
        self.n_segments = len(SEGMENT_POSITIONS)

        # Build the density matrix
        self.density = self._build_density_matrix()

        # Store as DataFrame for convenient lookup
        self.density_df = pd.DataFrame(
            self.density,
            index=self.muscles,
            columns=SEGMENT_LABELS
        )

    def _build_density_matrix(self) -> np.ndarray:
        """
        Build the muscle x segment density matrix using Gaussian profiles.

        Each muscle's density is modeled as a Gaussian centered at peak_segment
        with sigma = span / 4 (68% of neurons in central half of span).
        Density is zeroed outside the muscle's documented segment range and
        renormalized so each row sums to 1.0.
        """
        density = np.zeros((self.n_muscles, self.n_segments))

        for i, row in self.raw.iterrows():
            start = _parse_segment(row['segment_start'])
            end = _parse_segment(row['segment_end'])
            peak = _parse_segment(row['peak_segment'])
            span = end - start

            # Sigma: span/4 puts 68% of density in central half
            sigma = max(span / 4.0, 0.25)  # Floor at 0.25 to avoid delta spikes

            # Compute Gaussian density at each segment position
            gauss = np.exp(-0.5 * ((SEGMENT_POSITIONS - peak) / sigma) ** 2)

            # Zero out segments outside documented range
            mask = (SEGMENT_POSITIONS >= start) & (SEGMENT_POSITIONS <= end)
            gauss *= mask

            # Normalize so this muscle's weights sum to 1.0
            total = gauss.sum()
            if total > 0:
                gauss /= total

            density[i] = gauss

        return density

    def get_innervation(self, muscle: str) -> np.ndarray:
        """
        Get innervation density vector for a muscle across all segments.

        Args:
            muscle: Muscle name (must match nuclei_atlas.csv exactly)

        Returns:
            Array of shape (n_segments,) summing to 1.0
        """
        idx = self.muscles.index(muscle)
        return self.density[idx].copy()

    def get_segment_muscles(self, segment: str) -> Dict[str, float]:
        """
        Get all muscles innervated at a given segment with their densities.

        Args:
            segment: Segment label (e.g. 'C5', 'C7.5')

        Returns:
            Dict mapping muscle name -> innervation weight at that segment
        """
        seg_idx = list(SEGMENT_LABELS).index(segment)
        result = {}
        for i, muscle in enumerate(self.muscles):
            weight = self.density[i, seg_idx]
            if weight > 0:
                result[muscle] = float(weight)
        return result

    def get_muscle_info(self, muscle: str) -> Dict:
        """
        Get full metadata for a muscle.

        Returns:
            Dict with segment_start, segment_end, peak_segment, confidence,
            source, notes, gilmer_id, innervation vector.
        """
        row = self.raw[self.raw['muscle'] == muscle].iloc[0]
        return {
            'muscle': muscle,
            'gilmer_id': int(row['gilmer_id']),
            'segment_start': row['segment_start'],
            'segment_end': row['segment_end'],
            'peak_segment': row['peak_segment'],
            'confidence': row['confidence'],
            'source': row['source'],
            'notes': row['notes'],
            'innervation': self.get_innervation(muscle),
        }

    def compute_attenuation(self, damage_profile: np.ndarray) -> Dict[str, float]:
        """
        Compute per-muscle force attenuation from a segment damage profile.

        This is the GRAY MATTER channel only (direct motor neuron loss).
        For full lesion modeling including white matter, use LesionModel.

        Args:
            damage_profile: Array of shape (n_segments,) with values 0.0 (intact)
                           to 1.0 (complete destruction) at each segment.

        Returns:
            Dict mapping muscle name -> surviving force fraction (0.0-1.0)
        """
        if len(damage_profile) != self.n_segments:
            raise ValueError(
                f"damage_profile must have {self.n_segments} entries, "
                f"got {len(damage_profile)}"
            )

        attenuation = {}
        for i, muscle in enumerate(self.muscles):
            # Weighted sum of damage across segments, weighted by innervation density
            damage = np.sum(damage_profile * self.density[i])
            surviving = 1.0 - np.clip(damage, 0.0, 1.0)
            attenuation[muscle] = float(surviving)

        return attenuation

    def summary(self) -> str:
        """Print a human-readable summary of the atlas."""
        lines = ['Motor Pool Nuclei Atlas']
        lines.append(f'  {self.n_muscles} muscles x {self.n_segments} segment bins')
        lines.append(f'  Segments: {SEGMENT_LABELS[0]} to {SEGMENT_LABELS[-1]}')
        lines.append('')

        confidence_counts = self.raw['confidence'].value_counts()
        lines.append('Confidence breakdown:')
        for conf, count in confidence_counts.items():
            lines.append(f'  {conf}: {count} muscles')
        lines.append('')

        lines.append('Muscle spans:')
        for _, row in self.raw.iterrows():
            lines.append(
                f'  {row["muscle"]:30s}  {row["segment_start"]}-{row["segment_end"]}'
                f'  peak={row["peak_segment"]}  [{row["confidence"]}]'
            )

        return '\n'.join(lines)
