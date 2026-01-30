"""
MouseReach Analysis Dashboard

Interactive Streamlit app for behavioral data analysis.

Usage:
    streamlit run dashboard.py
    # or via CLI:
    mousereach-analyze

Features:
- Load reach data from pipeline outputs
- Filter by mouse, phase, outcome, quality flags
- Compare groups with statistical tests
- PCA analysis for dimensionality reduction
- Correlation with external data (connectome, etc.)
- Export publication-ready figures
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64

# Import our analysis modules
from mousereach.analysis.data import load_all_data, ReachDataFrame
from mousereach.analysis.stats import (
    compare_groups,
    compare_multiple_metrics,
    run_pca,
    cluster_reaches,
    find_optimal_clusters,
    correlate_with_external
)
from mousereach.analysis.plots import (
    apply_publication_style,
    save_publication_figure,
    plot_comparison,
    plot_pca_scores,
    plot_pca_loadings,
    plot_scree,
    plot_learning_curve,
    plot_cluster_profiles,
    plot_correlation_heatmap,
    plot_outcome_distribution,
)
from mousereach.config import Paths


# Page config
st.set_page_config(
    page_title="MouseReach Analysis",
    page_icon="🐭",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_download_link(fig: plt.Figure, filename: str, format: str = 'svg') -> str:
    """Create download link for matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', transparent=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    mime = 'image/svg+xml' if format == 'svg' else f'image/{format}'
    return f'<a href="data:{mime};base64,{b64}" download="{filename}.{format}">Download {format.upper()}</a>'


def main():
    st.title("🐭 MouseReach Analysis Dashboard")

    # Sidebar - Data Loading
    st.sidebar.header("📁 Data Source")

    # Default to Processing folder
    default_dir = Paths.PROCESSING if hasattr(Paths, 'PROCESSING') else Path(".")

    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=str(default_dir),
        help="Directory containing *_features.json or *_reaches.json files"
    )

    if st.sidebar.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                data = load_all_data(Path(data_dir), use_features=True, exclude_flagged=False)
                st.session_state['data'] = data
                st.session_state['data_dir'] = data_dir
                st.success(f"Loaded {len(data)} reaches")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return

    # Check if data is loaded
    if 'data' not in st.session_state:
        st.info("👆 Enter a data directory and click 'Load Data' to begin.")
        st.markdown("""
        **Expected data structure:**
        - Directory containing `*_features.json` files (preferred)
        - Or `*_reaches.json` + `*_pellet_outcomes.json` files

        **Location:**
        - Your `{PROCESSING_ROOT}/Processing/` folder (run `mousereach-setup --show` to see your path)
        """)
        return

    data: ReachDataFrame = st.session_state['data']

    # Sidebar - Filters
    st.sidebar.header("🔍 Filters")

    # Mouse filter
    all_mice = ['All'] + data.mice
    selected_mice = st.sidebar.multiselect(
        "Mouse ID",
        options=all_mice,
        default=['All'],
        help="Select one or more mice"
    )

    # Timepoint filter (experimental phase: Training, Pre-Injury, Post-Injury, Rehab)
    all_timepoints = ['All'] + data.timepoints
    selected_timepoints = st.sidebar.multiselect(
        "Timepoint",
        options=all_timepoints,
        default=['All'],
        help="Experimental phase (Training, Pre-Injury, Post-Injury, Rehab_Easy, Rehab_Flat, Rehab_Pillar)"
    )

    # Tray type filter
    all_tray_types = ['All'] + data.tray_types
    selected_tray_types = st.sidebar.multiselect(
        "Tray Type",
        options=all_tray_types,
        default=['All'],
        help="P=Pillar (standard), E=Easy (rehab), F=Flat"
    )

    # Outcome filter
    all_outcomes = ['All'] + data.outcomes
    selected_outcomes = st.sidebar.multiselect(
        "Outcome",
        options=all_outcomes,
        default=['All'],
        help="Select reach outcomes (e.g., only 'retrieved' for successful reaches)"
    )

    # Exclusion filter
    exclude_flagged = st.sidebar.checkbox(
        "Exclude flagged reaches",
        value=True,
        help="Remove reaches marked as unreliable"
    )

    # Apply filters
    filtered = data
    if 'All' not in selected_mice and selected_mice:
        filtered = filtered.filter(mouse_id=selected_mice)
    if 'All' not in selected_timepoints and selected_timepoints:
        filtered = filtered.filter(timepoint=selected_timepoints)
    if 'All' not in selected_tray_types and selected_tray_types:
        filtered = filtered.filter(tray_type=selected_tray_types)
    if 'All' not in selected_outcomes and selected_outcomes:
        filtered = filtered.filter(outcome=selected_outcomes)
    if exclude_flagged:
        filtered = filtered.filter(exclude_flagged=True)

    st.sidebar.metric("Filtered Reaches", len(filtered))

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Compare Groups",
        "🎯 PCA Analysis",
        "🔗 Correlations",
        "💾 Export"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reaches", len(filtered))
        with col2:
            st.metric("Mice", filtered.df['mouse_id'].nunique())
        with col3:
            st.metric("Sessions", filtered.df['session_id'].nunique())
        with col4:
            timepoint_col = 'timepoint' if 'timepoint' in filtered.df.columns else 'tray_type'
            n_timepoints = filtered.df[timepoint_col].nunique() if timepoint_col in filtered.df.columns else 0
            st.metric("Timepoints", n_timepoints)

        # Outcome distribution
        st.subheader("Outcome Distribution")
        if 'outcome' in filtered.df.columns:
            group_col = 'timepoint' if 'timepoint' in filtered.df.columns else 'tray_type'
            fig = plot_outcome_distribution(filtered.df, group_col=group_col)
            st.pyplot(fig)
            plt.close()

        # Summary statistics
        st.subheader("Summary Statistics")
        summary_cols = [
            'duration_frames', 'max_extent_mm', 'peak_velocity_px_per_frame',
            'trajectory_straightness', 'trajectory_smoothness'
        ]
        available_cols = [c for c in summary_cols if c in filtered.df.columns]

        if available_cols:
            summary_df = filtered.df[available_cols].describe()
            st.dataframe(summary_df.round(2))

    # Tab 2: Compare Groups
    with tab2:
        st.header("Group Comparison")

        col1, col2 = st.columns(2)

        with col1:
            compare_by = st.selectbox(
                "Compare by",
                options=['timepoint', 'tray_type', 'outcome', 'mouse_id'],
                help="Variable to split groups by"
            )

        # Get unique values for the comparison variable
        if compare_by in filtered.df.columns:
            unique_vals = sorted(filtered.df[compare_by].dropna().unique().tolist())
        else:
            unique_vals = []

        with col2:
            if len(unique_vals) >= 2:
                group1_val = st.selectbox("Group 1", options=unique_vals, index=0)
                group2_val = st.selectbox("Group 2", options=unique_vals, index=min(1, len(unique_vals)-1))
            else:
                st.warning(f"Need at least 2 unique values in '{compare_by}' to compare")
                group1_val, group2_val = None, None

        # Metric selection
        metric_options = [
            'duration_frames', 'max_extent_mm', 'max_extent_ruler',
            'peak_velocity_px_per_frame', 'mean_velocity_px_per_frame',
            'trajectory_straightness', 'trajectory_smoothness',
            'hand_angle_at_apex_deg', 'hand_rotation_total_deg'
        ]
        available_metrics = [m for m in metric_options if m in filtered.df.columns]

        selected_metric = st.selectbox("Metric to compare", options=available_metrics)

        if group1_val and group2_val and group1_val != group2_val and selected_metric:
            # Get data for each group
            g1_data = filtered.df[filtered.df[compare_by] == group1_val][selected_metric].dropna()
            g2_data = filtered.df[filtered.df[compare_by] == group2_val][selected_metric].dropna()

            if len(g1_data) >= 3 and len(g2_data) >= 3:
                # Run comparison
                result = compare_groups(
                    g1_data, g2_data,
                    metric_name=selected_metric,
                    group1_name=str(group1_val),
                    group2_name=str(group2_val)
                )

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{group1_val} (n={result.group1_n})",
                              f"{result.group1_mean:.2f} ± {result.group1_std:.2f}")
                with col2:
                    st.metric(f"{group2_val} (n={result.group2_n})",
                              f"{result.group2_mean:.2f} ± {result.group2_std:.2f}")
                with col3:
                    sig = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else "ns"
                    st.metric("p-value", f"{result.p_value:.4f} {sig}")
                    st.caption(f"Effect size: {result.effect_size_name}={result.effect_size:.2f} ({result.effect_interpretation})")

                # Plot
                fig = plot_comparison(
                    g1_data.values, g2_data.values,
                    group1_name=str(group1_val),
                    group2_name=str(group2_val),
                    metric_name=selected_metric.replace('_', ' ').title(),
                    style='box'
                )
                st.pyplot(fig)

                # Download buttons
                st.markdown(get_download_link(fig, f"comparison_{selected_metric}", 'svg'), unsafe_allow_html=True)
                plt.close()

                # Compare all metrics
                if st.checkbox("Compare all available metrics"):
                    results = compare_multiple_metrics(
                        filtered.df, compare_by, group1_val, group2_val, available_metrics
                    )

                    results_df = pd.DataFrame([r.to_dict() for r in results])
                    st.dataframe(results_df.round(4))
            else:
                st.warning(f"Need at least 3 samples per group. Got {len(g1_data)} and {len(g2_data)}.")

    # Tab 3: PCA Analysis
    with tab3:
        st.header("PCA Analysis")

        st.markdown("""
        Principal Component Analysis reduces dimensionality of kinematic features,
        revealing underlying patterns in reaching behavior.
        """)

        # Get feature matrix
        try:
            X, feature_names, index = filtered.get_feature_matrix(standardize=True)

            if len(X) < 10:
                st.warning("Need at least 10 reaches with complete kinematic data for PCA.")
            else:
                st.info(f"PCA on {len(X)} reaches with {len(feature_names)} features")

                # Number of components
                n_components = st.slider(
                    "Number of components",
                    min_value=2,
                    max_value=min(len(feature_names), 10),
                    value=min(5, len(feature_names))
                )

                # Run PCA
                pca_result = run_pca(X, feature_names, n_components=n_components, standardize=False)

                # Scree plot
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Variance Explained")
                    fig = plot_scree(pca_result.explained_variance_ratio, pca_result.cumulative_variance)
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.subheader("Variance Table")
                    st.dataframe(pca_result.get_variance_df().round(3))

                # Score plot
                st.subheader("Score Plot (PC1 vs PC2)")

                # Color by option
                color_by = st.selectbox(
                    "Color points by",
                    options=['None', 'phase', 'outcome', 'mouse_id'],
                    index=0
                )

                labels = None
                label_names = None
                if color_by != 'None' and color_by in filtered.df.columns:
                    labels = filtered.df.loc[index, color_by].values

                fig = plot_pca_scores(
                    pca_result.scores,
                    labels=labels,
                    variance_ratio=pca_result.explained_variance_ratio,
                    title="PCA Score Plot"
                )
                st.pyplot(fig)
                st.markdown(get_download_link(fig, "pca_scores", 'svg'), unsafe_allow_html=True)
                plt.close()

                # Loadings
                st.subheader("Feature Loadings")

                pc_to_show = st.selectbox("Show loadings for", options=[f"PC{i+1}" for i in range(n_components)])
                pc_idx = int(pc_to_show.replace("PC", "")) - 1

                fig = plot_pca_loadings(pca_result.components, feature_names, pc=pc_idx)
                st.pyplot(fig)
                st.markdown(get_download_link(fig, f"pca_loadings_{pc_to_show}", 'svg'), unsafe_allow_html=True)
                plt.close()

        except Exception as e:
            st.error(f"PCA Error: {e}")

    # Tab 4: Correlations
    with tab4:
        st.header("Correlation with External Data")

        st.markdown("""
        Upload external data (connectome, cell counts, etc.) to correlate with behavioral metrics.

        **Requirements:**
        - CSV or Excel file
        - Must have a `mouse_id` column matching your behavioral data
        - Numeric columns will be correlated with behavioral metrics
        """)

        uploaded_file = st.file_uploader(
            "Upload external data (CSV or Excel)",
            type=['csv', 'xlsx']
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    external_df = pd.read_csv(uploaded_file)
                else:
                    external_df = pd.read_excel(uploaded_file)

                st.write("External data preview:")
                st.dataframe(external_df.head())

                if 'mouse_id' not in external_df.columns:
                    st.error("External data must have a 'mouse_id' column")
                else:
                    # Aggregate behavioral data to mouse level
                    behavior_by_mouse = filtered.group_by_mouse()

                    # Find common mice
                    common_mice = set(behavior_by_mouse['mouse_id']) & set(external_df['mouse_id'])
                    st.info(f"Found {len(common_mice)} mice in both datasets")

                    if len(common_mice) >= 3:
                        # Run correlations
                        corr_results = correlate_with_external(
                            behavior_by_mouse,
                            external_df,
                            join_col='mouse_id'
                        )

                        if len(corr_results) > 0:
                            # Show significant correlations
                            sig_results = corr_results[corr_results['p_value'] < 0.05]
                            st.subheader("Significant Correlations (p < 0.05)")
                            st.dataframe(sig_results.round(4))

                            # Full correlation table
                            st.subheader("All Correlations")
                            st.dataframe(corr_results.round(4))

                            # Heatmap
                            pivot = corr_results.pivot(
                                index='behavior_metric',
                                columns='external_metric',
                                values='correlation'
                            )
                            if not pivot.empty:
                                fig = plot_correlation_heatmap(pivot)
                                st.pyplot(fig)
                                st.markdown(get_download_link(fig, "correlation_heatmap", 'svg'), unsafe_allow_html=True)
                                plt.close()

            except Exception as e:
                st.error(f"Error processing external data: {e}")

    # Tab 5: Export
    with tab5:
        st.header("Export Data & Figures")

        st.subheader("Export Filtered Data")

        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv_buffer = io.StringIO()
            filtered.df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name="mousereach_reaches.csv",
                mime="text/csv"
            )

        with col2:
            # Excel export
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                filtered.df.to_excel(writer, sheet_name='Reaches', index=False)
                filtered.group_by_session().to_excel(writer, sheet_name='Sessions', index=False)
                filtered.group_by_mouse().to_excel(writer, sheet_name='Mice', index=False)
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer,
                file_name="mousereach_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.subheader("Data Summary")
        st.write(f"**Reaches:** {len(filtered)}")
        st.write(f"**Sessions:** {filtered.df['session_id'].nunique()}")
        st.write(f"**Mice:** {filtered.df['mouse_id'].nunique()}")

        st.subheader("Columns in Export")
        st.write(", ".join(filtered.df.columns.tolist()))


if __name__ == "__main__":
    main()
