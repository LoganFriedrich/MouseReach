"""
Statistical analysis functions for MouseReach data.

Includes:
- Group comparisons (t-test, Mann-Whitney, paired tests)
- Effect sizes (Cohen's d, Hedges' g)
- PCA and dimensionality reduction
- Correlation analysis (for connectome integration)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two groups."""

    metric: str
    group1_name: str
    group2_name: str

    # Descriptive stats
    group1_mean: float
    group1_std: float
    group1_n: int
    group2_mean: float
    group2_std: float
    group2_n: int

    # Test results
    test_name: str
    statistic: float
    p_value: float

    # Effect size
    effect_size: float
    effect_size_name: str  # "Cohen's d" or "Hedges' g"
    effect_interpretation: str  # "small", "medium", "large"

    def __repr__(self):
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"{self.metric}: {self.group1_name} ({self.group1_mean:.2f}±{self.group1_std:.2f}, n={self.group1_n}) "
            f"vs {self.group2_name} ({self.group2_mean:.2f}±{self.group2_std:.2f}, n={self.group2_n}) | "
            f"p={self.p_value:.4f}{sig} | {self.effect_size_name}={self.effect_size:.2f} ({self.effect_interpretation})"
        )

    def to_dict(self) -> dict:
        return {
            'metric': self.metric,
            'group1': self.group1_name,
            'group2': self.group2_name,
            'group1_mean': self.group1_mean,
            'group1_std': self.group1_std,
            'group1_n': self.group1_n,
            'group2_mean': self.group2_mean,
            'group2_std': self.group2_std,
            'group2_n': self.group2_n,
            'test': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'effect_interpretation': self.effect_interpretation,
        }


@dataclass
class PCAResult:
    """Result of PCA analysis."""

    # Components
    components: np.ndarray  # (n_components, n_features) - loadings
    explained_variance: np.ndarray  # variance explained per component
    explained_variance_ratio: np.ndarray  # proportion of variance
    cumulative_variance: np.ndarray  # cumulative proportion

    # Transformed data
    scores: np.ndarray  # (n_samples, n_components) - PC scores
    feature_names: List[str]

    # Metadata
    n_components: int
    n_samples: int
    n_features: int

    def get_loadings_df(self) -> pd.DataFrame:
        """Get loadings as a DataFrame."""
        pc_names = [f'PC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(
            self.components.T,
            index=self.feature_names,
            columns=pc_names
        )

    def get_scores_df(self, index=None) -> pd.DataFrame:
        """Get scores as a DataFrame."""
        pc_names = [f'PC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(self.scores, columns=pc_names, index=index)

    def get_variance_df(self) -> pd.DataFrame:
        """Get variance explained as a DataFrame."""
        return pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(self.n_components)],
            'Variance': self.explained_variance,
            'Proportion': self.explained_variance_ratio,
            'Cumulative': self.cumulative_variance
        })


def compute_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    method: str = 'hedges_g'
) -> Tuple[float, str]:
    """
    Compute effect size between two groups.

    Args:
        group1: First group values
        group2: Second group values
        method: 'cohens_d' or 'hedges_g' (corrected for small samples)

    Returns:
        (effect_size, interpretation) tuple
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0, "undefined"

    # Cohen's d
    d = (mean1 - mean2) / pooled_std

    if method == 'hedges_g':
        # Hedges' g correction for small sample bias
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        d = d * correction

    # Interpret effect size (Cohen's conventions)
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, interpretation


def compare_groups(
    group1: Union[np.ndarray, pd.Series],
    group2: Union[np.ndarray, pd.Series],
    metric_name: str = "metric",
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    paired: bool = False,
    parametric: bool = True
) -> ComparisonResult:
    """
    Compare two groups statistically.

    Args:
        group1: First group values
        group2: Second group values
        metric_name: Name of the metric being compared
        group1_name: Label for first group
        group2_name: Label for second group
        paired: If True, use paired test (requires equal lengths)
        parametric: If True, use t-test; if False, use Mann-Whitney/Wilcoxon

    Returns:
        ComparisonResult with test statistics and effect size
    """
    g1 = np.array(group1).flatten()
    g2 = np.array(group2).flatten()

    # Remove NaN
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    # Descriptive stats
    mean1, std1, n1 = np.mean(g1), np.std(g1, ddof=1), len(g1)
    mean2, std2, n2 = np.mean(g2), np.std(g2, ddof=1), len(g2)

    # Statistical test
    if paired:
        if len(g1) != len(g2):
            raise ValueError("Paired test requires equal group sizes")
        if parametric:
            stat, p = stats.ttest_rel(g1, g2)
            test_name = "paired t-test"
        else:
            stat, p = stats.wilcoxon(g1, g2)
            test_name = "Wilcoxon signed-rank"
    else:
        if parametric:
            stat, p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test
            test_name = "Welch's t-test"
        else:
            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            test_name = "Mann-Whitney U"

    # Effect size
    effect, interpretation = compute_effect_size(g1, g2, method='hedges_g')

    return ComparisonResult(
        metric=metric_name,
        group1_name=group1_name,
        group2_name=group2_name,
        group1_mean=mean1,
        group1_std=std1,
        group1_n=n1,
        group2_mean=mean2,
        group2_std=std2,
        group2_n=n2,
        test_name=test_name,
        statistic=stat,
        p_value=p,
        effect_size=effect,
        effect_size_name="Hedges' g",
        effect_interpretation=interpretation
    )


def compare_multiple_metrics(
    df: pd.DataFrame,
    group_col: str,
    group1_value,
    group2_value,
    metrics: List[str],
    paired: bool = False
) -> List[ComparisonResult]:
    """
    Compare two groups across multiple metrics.

    Args:
        df: DataFrame with data
        group_col: Column name for grouping
        group1_value: Value identifying first group
        group2_value: Value identifying second group
        metrics: List of column names to compare
        paired: If True, use paired tests

    Returns:
        List of ComparisonResult for each metric
    """
    results = []

    g1_mask = df[group_col] == group1_value
    g2_mask = df[group_col] == group2_value

    for metric in metrics:
        if metric not in df.columns:
            continue

        g1_data = df.loc[g1_mask, metric].dropna()
        g2_data = df.loc[g2_mask, metric].dropna()

        if len(g1_data) < 3 or len(g2_data) < 3:
            continue

        result = compare_groups(
            g1_data, g2_data,
            metric_name=metric,
            group1_name=str(group1_value),
            group2_name=str(group2_value),
            paired=paired
        )
        results.append(result)

    return results


def run_pca(
    X: np.ndarray,
    feature_names: List[str],
    n_components: Optional[int] = None,
    standardize: bool = True
) -> PCAResult:
    """
    Run PCA on feature matrix.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Names of features
        n_components: Number of components (default: all)
        standardize: If True, z-score normalize before PCA

    Returns:
        PCAResult with loadings, scores, and variance explained
    """
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    if n_components is None:
        n_components = min(X.shape)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return PCAResult(
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance=cumulative,
        scores=scores,
        feature_names=feature_names,
        n_components=n_components,
        n_samples=X.shape[0],
        n_features=X.shape[1]
    )


def correlate_with_external(
    behavior_df: pd.DataFrame,
    external_df: pd.DataFrame,
    join_col: str = 'mouse_id',
    behavior_metrics: Optional[List[str]] = None,
    external_metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Correlate behavioral metrics with external data (e.g., connectome).

    Args:
        behavior_df: DataFrame with behavioral data (one row per mouse)
        external_df: DataFrame with external data (one row per mouse)
        join_col: Column to join on (usually 'mouse_id')
        behavior_metrics: Behavioral columns to correlate (default: all numeric)
        external_metrics: External columns to correlate (default: all numeric)

    Returns:
        DataFrame with correlation coefficients and p-values
    """
    # Merge datasets
    merged = pd.merge(behavior_df, external_df, on=join_col, how='inner')

    if len(merged) < 3:
        print(f"Warning: Only {len(merged)} mice matched between datasets")
        return pd.DataFrame()

    # Get numeric columns if not specified
    if behavior_metrics is None:
        behavior_metrics = behavior_df.select_dtypes(include=[np.number]).columns.tolist()
        behavior_metrics = [c for c in behavior_metrics if c in merged.columns]

    if external_metrics is None:
        external_metrics = external_df.select_dtypes(include=[np.number]).columns.tolist()
        external_metrics = [c for c in external_metrics if c in merged.columns]

    results = []
    for b_col in behavior_metrics:
        for e_col in external_metrics:
            if b_col == e_col:
                continue

            x = merged[b_col].dropna()
            y = merged[e_col].dropna()

            # Get overlapping indices
            common_idx = x.index.intersection(y.index)
            if len(common_idx) < 3:
                continue

            x_vals = x.loc[common_idx].values
            y_vals = y.loc[common_idx].values

            # Pearson correlation
            r, p = stats.pearsonr(x_vals, y_vals)

            results.append({
                'behavior_metric': b_col,
                'external_metric': e_col,
                'correlation': r,
                'p_value': p,
                'n': len(common_idx),
                'significant': p < 0.05
            })

    return pd.DataFrame(results)


@dataclass
class ClusterResult:
    """Result of reach clustering analysis."""

    labels: np.ndarray  # Cluster assignment per reach
    n_clusters: int
    cluster_centers: Optional[np.ndarray]  # Centroids if k-means

    # Per-cluster statistics
    cluster_sizes: Dict[int, int]
    cluster_means: pd.DataFrame  # Mean features per cluster

    # Quality metrics
    silhouette_score: float
    inertia: Optional[float]  # Within-cluster sum of squares (k-means)

    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary of cluster characteristics."""
        summary = self.cluster_means.copy()
        summary['n_reaches'] = [self.cluster_sizes.get(i, 0) for i in range(self.n_clusters)]
        summary['proportion'] = summary['n_reaches'] / summary['n_reaches'].sum()
        return summary


def cluster_reaches(
    X: np.ndarray,
    feature_names: List[str],
    n_clusters: int = 3,
    method: str = 'kmeans',
    standardize: bool = True
) -> ClusterResult:
    """
    Cluster reaches by kinematic features to find reach "types".

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Names of features
        n_clusters: Number of clusters
        method: 'kmeans' or 'hierarchical'
        standardize: If True, z-score normalize before clustering

    Returns:
        ClusterResult with cluster assignments and statistics
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        centers = model.cluster_centers_
        inertia = model.inertia_
    else:  # hierarchical
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        centers = None
        inertia = None

    # Compute silhouette score
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(X_scaled, labels)
    else:
        sil_score = 0.0

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    # Mean features per cluster (in original scale)
    cluster_means = []
    for i in range(n_clusters):
        mask = labels == i
        if mask.sum() > 0:
            cluster_means.append(X[mask].mean(axis=0))
        else:
            cluster_means.append(np.zeros(X.shape[1]))

    cluster_means_df = pd.DataFrame(
        cluster_means,
        columns=feature_names,
        index=[f'Cluster {i}' for i in range(n_clusters)]
    )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        cluster_centers=centers,
        cluster_sizes=cluster_sizes,
        cluster_means=cluster_means_df,
        silhouette_score=sil_score,
        inertia=inertia
    )


def find_optimal_clusters(
    X: np.ndarray,
    max_clusters: int = 10,
    method: str = 'silhouette'
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using elbow or silhouette method.

    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to test
        method: 'silhouette' or 'elbow'

    Returns:
        (optimal_k, scores) tuple
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = []
    k_range = range(2, min(max_clusters + 1, len(X)))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        if method == 'silhouette':
            score = silhouette_score(X_scaled, labels)
        else:  # elbow
            score = kmeans.inertia_

        scores.append(score)

    if method == 'silhouette':
        optimal_k = list(k_range)[np.argmax(scores)]
    else:
        # Elbow method - find point of diminishing returns
        # Using second derivative
        if len(scores) > 2:
            diffs = np.diff(scores)
            diffs2 = np.diff(diffs)
            optimal_k = list(k_range)[np.argmax(diffs2) + 1]
        else:
            optimal_k = 2

    return optimal_k, scores


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Data array
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        (estimate, ci_lower, ci_upper) tuple
    """
    data = np.array(data)
    data = data[~np.isnan(data)]

    estimate = statistic(data)

    # Bootstrap
    rng = np.random.default_rng(42)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return estimate, ci_lower, ci_upper
