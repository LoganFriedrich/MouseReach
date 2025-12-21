"""
ASPA2 DLC Utilities
===================

Functions for loading and preprocessing DeepLabCut output files.
"""

import pandas as pd
from pathlib import Path
from typing import Union


def load_dlc_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load DLC output file (.h5 or .csv).
    
    Flattens multi-index columns to single-level format:
    e.g., ('DLC_model', 'SABL', 'x') -> 'SABL_x'
    
    Args:
        filepath: Path to DLC .h5 or .csv file
    
    Returns:
        DataFrame with flattened column names
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.h5':
        df = pd.read_hdf(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    # Flatten multi-index columns
    # From ('DLC_model', 'bodypart', 'coord') to 'bodypart_coord'
    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
    
    return df


def get_bodypart_coords(df: pd.DataFrame, bodypart: str) -> pd.DataFrame:
    """
    Extract x, y, likelihood for a specific bodypart.
    
    Args:
        df: DataFrame from load_dlc_data
        bodypart: Name of bodypart (e.g., 'SABL', 'RightHand')
    
    Returns:
        DataFrame with x, y, likelihood columns
    """
    cols = [f'{bodypart}_x', f'{bodypart}_y', f'{bodypart}_likelihood']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    return df[cols].copy()


def interpolate_low_likelihood(df: pd.DataFrame, 
                                bodypart: str, 
                                threshold: float = 0.5) -> pd.DataFrame:
    """
    Interpolate positions where likelihood is below threshold.
    
    Args:
        df: DataFrame from load_dlc_data
        bodypart: Name of bodypart
        threshold: Likelihood threshold (default 0.5)
    
    Returns:
        DataFrame with interpolated values
    """
    df = df.copy()
    
    x_col = f'{bodypart}_x'
    y_col = f'{bodypart}_y'
    like_col = f'{bodypart}_likelihood'
    
    # Mask low-likelihood frames
    low_conf = df[like_col] < threshold
    df.loc[low_conf, x_col] = float('nan')
    df.loc[low_conf, y_col] = float('nan')
    
    # Interpolate
    df[x_col] = df[x_col].interpolate(method='linear', limit_direction='both')
    df[y_col] = df[y_col].interpolate(method='linear', limit_direction='both')
    
    # Fill any remaining NaNs at edges
    df[x_col] = df[x_col].ffill().bfill()
    df[y_col] = df[y_col].ffill().bfill()
    
    return df


def list_bodyparts(df: pd.DataFrame) -> list:
    """
    List all bodyparts in the DataFrame.
    
    Args:
        df: DataFrame from load_dlc_data
    
    Returns:
        List of bodypart names
    """
    bodyparts = set()
    for col in df.columns:
        # Column format: 'bodypart_coord'
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('x', 'y', 'likelihood'):
            bodyparts.add(parts[0])
    return sorted(bodyparts)
