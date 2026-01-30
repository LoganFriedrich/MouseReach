"""
Publication-ready plotting functions for MouseReach analysis.

All plots use matplotlib for maximum compatibility with journal requirements.
Figures can be exported as SVG, PDF, or high-resolution PNG.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import seaborn as sns

# Publication-ready style settings
PUBLICATION_STYLE = {
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color palettes
PALETTE_PHASE = {
    'P1': '#1f77b4',  # Blue - baseline
    'P2': '#ff7f0e',  # Orange - early
    'P3': '#2ca02c',  # Green - mid
    'P4': '#d62728',  # Red - late
}

PALETTE_OUTCOME = {
    'retrieved': '#2ecc71',      # Green
    'displaced_sa': '#f39c12',   # Orange
    'displaced_outside': '#e74c3c',  # Red
    'untouched': '#95a5a6',      # Gray
}

PALETTE_CLUSTER = sns.color_palette('Set2', 8)


def apply_publication_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update(PUBLICATION_STYLE)


def save_publication_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: List[str] = ['svg', 'pdf', 'png'],
    tight: bool = True
):
    """
    Save figure in multiple publication-ready formats.

    Args:
        fig: matplotlib Figure
        path: Output path (without extension)
        formats: List of formats to save ('svg', 'pdf', 'png')
        tight: If True, use tight_layout
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tight:
        fig.tight_layout()

    for fmt in formats:
        fig.savefig(
            path.with_suffix(f'.{fmt}'),
            format=fmt,
            bbox_inches='tight',
            transparent=True if fmt != 'png' else False
        )
        print(f"Saved: {path.with_suffix(f'.{fmt}')}")


def plot_comparison(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    metric_name: str = "Metric",
    style: str = 'box',
    show_points: bool = True,
    colors: Optional[Tuple[str, str]] = None,
    figsize: Tuple[float, float] = (3, 4),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Create comparison plot between two groups.

    Args:
        group1, group2: Data arrays for each group
        group1_name, group2_name: Labels for groups
        metric_name: Y-axis label
        style: 'box', 'violin', or 'bar'
        show_points: If True, overlay individual data points
        colors: Tuple of colors for each group
        figsize: Figure size in inches
        ax: Existing axes to plot on

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = ('#1f77b4', '#ff7f0e')

    data = [np.array(group1)[~np.isnan(group1)], np.array(group2)[~np.isnan(group2)]]
    positions = [0, 1]

    if style == 'box':
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    elif style == 'violin':
        vp = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(0.7)
    elif style == 'bar':
        means = [np.mean(d) for d in data]
        stds = [np.std(d) for d in data]
        ax.bar(positions, means, yerr=stds, color=colors, alpha=0.7, capsize=5)

    if show_points and style != 'bar':
        for i, (d, color) in enumerate(zip(data, colors)):
            jitter = np.random.uniform(-0.1, 0.1, len(d))
            ax.scatter(
                np.full(len(d), positions[i]) + jitter, d,
                c=color, alpha=0.5, s=20, edgecolor='white', linewidth=0.5
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([group1_name, group2_name])
    ax.set_ylabel(metric_name)

    return fig


def plot_pca_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict] = None,
    pc_x: int = 0,
    pc_y: int = 1,
    variance_ratio: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (5, 5),
    ax: Optional[plt.Axes] = None,
    title: str = "PCA Score Plot"
) -> plt.Figure:
    """
    Create PCA score plot (PC1 vs PC2).

    Args:
        scores: PCA scores array (n_samples, n_components)
        labels: Optional grouping labels for coloring
        label_names: Dict mapping label values to display names
        pc_x, pc_y: Which PCs to plot (0-indexed)
        variance_ratio: Variance explained per PC (for axis labels)
        figsize: Figure size
        ax: Existing axes
        title: Plot title

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if labels is None:
        ax.scatter(scores[:, pc_x], scores[:, pc_y], alpha=0.6, s=30)
    else:
        unique_labels = np.unique(labels)
        colors = PALETTE_CLUSTER[:len(unique_labels)]

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            name = label_names.get(label, str(label)) if label_names else str(label)
            ax.scatter(
                scores[mask, pc_x], scores[mask, pc_y],
                c=[color], alpha=0.6, s=30, label=name
            )
        ax.legend(loc='best', frameon=True)

    # Axis labels with variance explained
    if variance_ratio is not None:
        ax.set_xlabel(f'PC{pc_x + 1} ({variance_ratio[pc_x]*100:.1f}%)')
        ax.set_ylabel(f'PC{pc_y + 1} ({variance_ratio[pc_y]*100:.1f}%)')
    else:
        ax.set_xlabel(f'PC{pc_x + 1}')
        ax.set_ylabel(f'PC{pc_y + 1}')

    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    return fig


def plot_pca_loadings(
    loadings: np.ndarray,
    feature_names: List[str],
    pc: int = 0,
    top_n: int = 10,
    figsize: Tuple[float, float] = (5, 4),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Create loading plot showing feature contributions to a PC.

    Args:
        loadings: Loading matrix (n_components, n_features)
        feature_names: Feature names
        pc: Which PC to show (0-indexed)
        top_n: Number of top features to show
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    pc_loadings = loadings[pc]

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(pc_loadings))[::-1][:top_n]

    y_pos = np.arange(len(sorted_idx))
    values = pc_loadings[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]

    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(f'PC{pc + 1} Loading')
    ax.set_title(f'Top {top_n} Feature Contributions to PC{pc + 1}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()

    return fig


def plot_scree(
    variance_ratio: np.ndarray,
    cumulative: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (5, 4),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Create scree plot showing variance explained by each PC.

    Args:
        variance_ratio: Proportion of variance per PC
        cumulative: Cumulative variance (optional)
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_components = len(variance_ratio)
    x = np.arange(1, n_components + 1)

    # Bar plot of individual variance
    ax.bar(x, variance_ratio * 100, alpha=0.7, color='#3498db', label='Individual')

    # Line plot of cumulative variance
    if cumulative is not None:
        ax2 = ax.twinx()
        ax2.plot(x, cumulative * 100, 'ro-', markersize=5, label='Cumulative')
        ax2.set_ylabel('Cumulative Variance (%)')
        ax2.set_ylim(0, 105)
        ax2.axhline(80, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_xticks(x)
    ax.set_title('Scree Plot')

    return fig


def plot_learning_curve(
    df: pd.DataFrame,
    mouse_id: str,
    metric: str = 'success_rate',
    by_phase: bool = True,
    figsize: Tuple[float, float] = (6, 4),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot learning curve (metric over sessions) for a single mouse.

    Args:
        df: Session-level DataFrame
        mouse_id: Mouse to plot
        metric: Column name for y-axis
        by_phase: If True, color by phase
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mouse_df = df[df['mouse_id'] == mouse_id].sort_values('date')

    if by_phase and 'phase' in mouse_df.columns:
        for phase in mouse_df['phase'].unique():
            phase_df = mouse_df[mouse_df['phase'] == phase]
            color = PALETTE_PHASE.get(phase, '#333333')
            ax.plot(
                phase_df['date'], phase_df[metric],
                'o-', color=color, label=phase, markersize=6
            )
        ax.legend(title='Phase', loc='best')
    else:
        ax.plot(mouse_df['date'], mouse_df[metric], 'o-', markersize=6)

    ax.set_xlabel('Date')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{mouse_id} - Learning Curve')

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return fig


def plot_cluster_profiles(
    cluster_means: pd.DataFrame,
    figsize: Tuple[float, float] = (8, 5),
    normalize: bool = True
) -> plt.Figure:
    """
    Create radar/parallel coordinates plot showing cluster profiles.

    Args:
        cluster_means: DataFrame with cluster means (rows=clusters, cols=features)
        figsize: Figure size
        normalize: If True, normalize each feature to 0-1 range

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    df = cluster_means.copy()

    if normalize:
        df = (df - df.min()) / (df.max() - df.min())

    n_features = len(df.columns)
    x = np.arange(n_features)

    for i, (idx, row) in enumerate(df.iterrows()):
        color = PALETTE_CLUSTER[i % len(PALETTE_CLUSTER)]
        ax.plot(x, row.values, 'o-', color=color, label=idx, linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value' if normalize else 'Value')
    ax.set_title('Cluster Profiles')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return fig


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    figsize: Tuple[float, float] = (8, 6),
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Create correlation heatmap (e.g., behavior vs connectome).

    Args:
        corr_df: DataFrame with correlation values
        figsize: Figure size
        vmin, vmax: Color scale limits
        cmap: Colormap name

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr_df.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_df.index)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation (r)')

    ax.set_title('Behavior-Connectome Correlation')

    return fig


def plot_outcome_distribution(
    df: pd.DataFrame,
    group_col: str = 'phase',
    figsize: Tuple[float, float] = (6, 4)
) -> plt.Figure:
    """
    Create stacked bar plot of outcome distributions by group.

    Args:
        df: DataFrame with 'outcome' column
        group_col: Column to group by
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Count outcomes by group
    counts = df.groupby([group_col, 'outcome']).size().unstack(fill_value=0)

    # Normalize to proportions
    proportions = counts.div(counts.sum(axis=1), axis=0)

    # Plot stacked bars
    bottom = np.zeros(len(proportions))
    x = np.arange(len(proportions))

    for outcome in proportions.columns:
        color = PALETTE_OUTCOME.get(outcome, '#333333')
        ax.bar(x, proportions[outcome], bottom=bottom, color=color, label=outcome, alpha=0.8)
        bottom += proportions[outcome]

    ax.set_xticks(x)
    ax.set_xticklabels(proportions.index)
    ax.set_ylabel('Proportion')
    ax.set_xlabel(group_col.replace('_', ' ').title())
    ax.set_title('Outcome Distribution')
    ax.legend(title='Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')

    return fig


def create_figure_panel(
    plots: List[Tuple[callable, dict]],
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[float, float]] = None,
    titles: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create multi-panel figure for publication.

    Args:
        plots: List of (plot_function, kwargs) tuples
        nrows, ncols: Grid layout
        figsize: Figure size (default: auto-calculated)
        titles: Optional panel titles

    Returns:
        matplotlib Figure
    """
    apply_publication_style()

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows * ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    flat_axes = [ax for row in axes for ax in row]

    for i, (plot_func, kwargs) in enumerate(plots):
        if i < len(flat_axes):
            kwargs['ax'] = flat_axes[i]
            plot_func(**kwargs)

            if titles and i < len(titles):
                flat_axes[i].set_title(titles[i], fontweight='bold')

    # Add panel labels (A, B, C, ...)
    for i, ax in enumerate(flat_axes[:len(plots)]):
        ax.text(
            -0.1, 1.1, chr(65 + i),  # A, B, C, ...
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='top'
        )

    return fig
