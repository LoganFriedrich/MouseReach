#!/usr/bin/env python3
"""
DLC quality checking - Validate DeepLabCut tracking output

Checks:
- Reference point stability (BOXL, BOXR)
- SA anchor tracking (SABL, SABR) 
- Hand point coverage (RightHand, RHLeft, RHOut, RHRight)
- Overall likelihood scores
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json


# Expected DLC bodyparts (train your model with these labels)
BODYPARTS = [
    'Reference',
    'SATL', 'SABL', 'SABR', 'SATR',  # Scoring Area corners
    'BOXL', 'BOXR',                    # Box edges (slit)
    'Pellet', 'Pillar',                # Target objects
    'RightHand', 'RHLeft', 'RHOut', 'RHRight',  # Hand tracking
    'Nose', 'RightEar', 'LeftEar',     # Head
    'LeftFoot', 'TailBase'             # Body
]

# Critical points that must track well
CRITICAL_POINTS = ['SABL', 'SABR', 'RightHand', 'BOXL', 'BOXR']

# Reference points (should be stable/fixed)
REFERENCE_POINTS = ['BOXL', 'BOXR', 'Reference']


@dataclass
class DLCQualityReport:
    """Quality assessment for DLC tracking output."""
    video_name: str
    dlc_model: str
    total_frames: int
    
    # Reference point stability (std in pixels)
    boxl_std: float = 0.0
    boxr_std: float = 0.0
    reference_quality: str = "unknown"  # good/fair/poor
    
    # Point coverage (fraction of high-confidence frames)
    point_coverage: Dict[str, float] = field(default_factory=dict)
    mean_likelihood: float = 0.0
    
    # Critical point assessment
    critical_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    
    # Overall assessment
    overall_quality: str = "unknown"  # good/fair/poor/failed
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'DLCQualityReport':
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def load_dlc_data(h5_path: Path) -> pd.DataFrame:
    """Load DLC h5 file into DataFrame."""
    return pd.read_hdf(h5_path)


def check_dlc_quality(
    h5_path: Path,
    likelihood_threshold: float = 0.6
) -> DLCQualityReport:
    """
    Check quality of DLC tracking output.
    
    Args:
        h5_path: Path to DLC .h5 file
        likelihood_threshold: Minimum likelihood for "good" tracking (default 0.6 matches pcutoff)
        
    Returns:
        DLCQualityReport with quality metrics
    """
    h5_path = Path(h5_path)
    
    # Load data
    df = load_dlc_data(h5_path)
    
    # Get scorer (first level of column multiindex)
    scorer = df.columns.get_level_values(0)[0]
    
    # Get video name and model from filename
    # Pattern: {date}_{animal}_{tray}DLC_{network}_{model}shuffle{N}_{iters}.h5
    stem = h5_path.stem
    if 'DLC_' in stem:
        video_name = stem.split('DLC_')[0].rstrip('_')
        dlc_model = 'DLC_' + stem.split('DLC_')[1]
    else:
        video_name = stem
        dlc_model = "unknown"
    
    total_frames = len(df)
    issues = []
    
    # Get available bodyparts
    available_bodyparts = df.columns.get_level_values(1).unique().tolist()
    
    # Check reference point stability
    boxl_std = boxr_std = 0.0
    ref_quality = "unknown"
    
    if 'BOXL' in available_bodyparts and 'BOXR' in available_bodyparts:
        # Get high-confidence frames only
        boxl_conf = df[(scorer, 'BOXL', 'likelihood')] > likelihood_threshold
        boxr_conf = df[(scorer, 'BOXR', 'likelihood')] > likelihood_threshold
        
        if boxl_conf.sum() > 100:
            boxl_std = df.loc[boxl_conf, (scorer, 'BOXL', 'x')].std()
        if boxr_conf.sum() > 100:
            boxr_std = df.loc[boxr_conf, (scorer, 'BOXR', 'x')].std()
        
        if boxl_std < 3 and boxr_std < 3:
            ref_quality = "good"
        elif boxl_std < 5 and boxr_std < 5:
            ref_quality = "fair"
        else:
            ref_quality = "poor"
            issues.append(f"Reference points unstable: BOXL_std={boxl_std:.1f}px, BOXR_std={boxr_std:.1f}px")
    else:
        ref_quality = "missing"
        issues.append("Missing BOXL/BOXR reference points")
    
    # Check point coverage
    point_coverage = {}
    critical_coverage = {}
    all_likelihoods = []
    
    for bp in available_bodyparts:
        try:
            likelihood = df[(scorer, bp, 'likelihood')]
            coverage = (likelihood > likelihood_threshold).mean()
            point_coverage[bp] = round(coverage, 3)
            all_likelihoods.extend(likelihood.values)
            
            if bp in CRITICAL_POINTS:
                critical_coverage[bp] = round(coverage, 3)
                if coverage < 0.8:
                    issues.append(f"Low coverage for {bp}: {coverage:.1%}")
        except KeyError:
            continue
    
    mean_likelihood = np.mean(all_likelihoods) if all_likelihoods else 0.0
    
    # Check SA anchors specifically
    for anchor in ['SABL', 'SABR']:
        if anchor in point_coverage and point_coverage[anchor] < 0.9:
            issues.append(f"SA anchor {anchor} coverage low: {point_coverage[anchor]:.1%}")
    
    # Overall quality assessment
    if (ref_quality == "good" and 
        mean_likelihood > 0.8 and 
        len(issues) == 0 and
        all(critical_coverage.get(cp, 0) > 0.85 for cp in CRITICAL_POINTS if cp in available_bodyparts)):
        overall_quality = "good"
    elif (ref_quality != "poor" and 
          mean_likelihood > 0.7 and
          all(critical_coverage.get(cp, 0) > 0.7 for cp in CRITICAL_POINTS if cp in available_bodyparts)):
        overall_quality = "fair"
    elif len(issues) > 3 or mean_likelihood < 0.5:
        overall_quality = "failed"
    else:
        overall_quality = "poor"
    
    return DLCQualityReport(
        video_name=video_name,
        dlc_model=dlc_model,
        total_frames=total_frames,
        boxl_std=round(boxl_std, 2),
        boxr_std=round(boxr_std, 2),
        reference_quality=ref_quality,
        point_coverage=point_coverage,
        mean_likelihood=round(mean_likelihood, 3),
        critical_coverage=critical_coverage,
        issues=issues,
        overall_quality=overall_quality
    )


def check_batch(
    input_dir: Path,
    output_dir: Path = None,
    verbose: bool = True
) -> List[DLCQualityReport]:
    """Check quality of all DLC files in a directory."""
    input_dir = Path(input_dir)
    h5_files = sorted(input_dir.glob("*DLC*.h5"))
    
    if verbose:
        print(f"Checking {len(h5_files)} DLC files...")
    
    reports = []
    for h5_path in h5_files:
        report = check_dlc_quality(h5_path)
        reports.append(report)
        
        if verbose:
            status = "[OK]" if report.overall_quality == "good" else "[?]" if report.overall_quality == "fair" else "[!!]"
            print(f"  {status} {report.video_name}: {report.overall_quality.upper()} (likelihood={report.mean_likelihood:.2f})")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report.save(output_dir / f"{report.video_name}_dlc_quality.json")
    
    return reports
