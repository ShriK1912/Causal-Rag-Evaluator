"""
Deep-Fetch Statistical Analyzer Page for Causal RAG Diagnostician.
Provides interactive statistical analysis of retrieval distance distributions,
comparing Top-5 baseline against Ranks 6-100 deep-fetch retrieval,
generating empirical diagnostic insights, and rendering Plotly visualizations.
"""

import sys
from pathlib import Path
from typing import Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    SYNTHETIC_DATA_DIR,
    DETAILED_RETRIEVAL_CSV,
    TOP_K_BASELINE,
    TOP_K_DEEP_FETCH,
)
from modules.chroma_connector import ChromaConnector
from modules.statistical_analyzer import (
    StatisticalAnalyzer,
    plot_top5_vs_deep_histogram,
    plot_ranked_distance_curve,
    plot_box_comparison,
    plot_category_aggregate_boxplots,
)
from modules.data_ingestion import load_csv
from modules.ui_components import render_sidebar, render_academic_disclaimer

def fmt(val: Any, decimals: int = 4) -> str:
    """Helper to format numeric metrics gracefully handling NaNs, None, and edge cases."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if np.isnan(f):
            return "N/A"
        return f"{f:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


st.set_page_config(
    page_title="Statistical Analyzer | RAG Diagnostician",
    page_icon="📈",
    layout="wide"
)

render_sidebar("Deep-Fetch Statistical Analyzer")

st.title("📈 Deep-Fetch Statistical Analyzer")
st.markdown("""
Analyze the mathematical distribution of retrieval distances and similarity scores.
Compare operational **Top-5 baseline results** against extended **Ranks 6–100 deep-fetch results**
to discover ranking shifts, distractor noise, and database coverage gaps.
""")

st.markdown("---")

# Cached ChromaDB Connector and Statistical Analyzer
@st.cache_resource(show_spinner=False)
def get_analyzer_bundle():
    conn = ChromaConnector(CHROMA_PERSIST_DIR)
    an = StatisticalAnalyzer(conn)
    return conn, an

try:
    connector, analyzer = get_analyzer_bundle()
except Exception as e:
    st.error(f"Failed to initialize ChromaDB connector: {e}")
    st.stop()

# Mode Selection
view_mode = st.radio(
    "Select Analysis View Mode:",
    ["🔍 Query-Level Deep-Fetch Analysis", "📊 Aggregate Dataset Statistics (50 Queries, 5,000 Chunks)"],
    horizontal=True
)

st.markdown("---")

# =============================================================================
# MODE 1: QUERY-LEVEL DEEP-FETCH ANALYSIS
# =============================================================================
if view_mode == "🔍 Query-Level Deep-Fetch Analysis":
    df_queries = None
    if "validated_df" in st.session_state and not st.session_state["validated_df"].empty:
        df_queries = st.session_state["validated_df"]
    else:
        sample_csv = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
        if sample_csv.exists():
            report = load_csv(sample_csv)
            df_queries = report.cleaned_df
            st.session_state["validated_df"] = df_queries

    if df_queries is None or df_queries.empty:
        st.warning("No queries available. Ingest dataset in Data Ingestion page.")
        st.stop()

    st.subheader("1. Select Query for Deep-Fetch Inspection")
    categories = ["All Categories"] + sorted(df_queries["failure_label"].dropna().unique().tolist())
    selected_cat = st.selectbox("Filter Queries by Failure Category:", categories)

    filtered_df = df_queries if selected_cat == "All Categories" else df_queries[df_queries["failure_label"] == selected_cat]

    query_options = [
        f"{row['query_id']} | [{row.get('failure_label', 'Unlabeled')}] {row['query']}"
        for _, row in filtered_df.iterrows()
    ]
    selected_query_str = st.selectbox("Choose Query:", query_options)
    selected_qid = selected_query_str.split(" | ")[0].strip()
    selected_row = df_queries[df_queries["query_id"] == selected_qid].iloc[0]

    # Interaction Context Card
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown(f"**Query ID:** `{selected_row.get('query_id')}`")
        st.info(f"❓ **Query:** {selected_row['query']}")
        st.markdown(f"**Target Failure Category:** `{selected_row.get('failure_label', 'Unlabeled')}`")
    with q_col2:
        st.markdown(f"**Expected Answer:**")
        st.success(f"🎯 {selected_row.get('expected_answer', 'Not Provided')}")
        st.markdown(f"**Generated Answer:**")
        st.warning(f"🤖 {selected_row.get('generated_answer', 'Not Provided')}")
        target_id = str(selected_row.get("target_doc_id", "NONE"))

    # Execute Retrieval & Statistical Analysis (Cached)
    @st.cache_data(show_spinner=False)
    def cached_query_analysis(query_txt: str, exp_ans: Any, gen_ans: Any, tgt_id: str):
        conn, an = get_analyzer_bundle()
        records = conn.retrieve_query(
            collection_name=DEFAULT_COLLECTION_NAME,
            query_text=query_txt,
            top_k=TOP_K_DEEP_FETCH
        )
        res = an.analyze_query_retrieval(
            retrieval_records=records,
            expected_answer=exp_ans,
            generated_answer=gen_ans,
            target_doc_id=tgt_id
        )
        return records, res

    with st.spinner("Retrieving Top-100 chunks and computing statistical distributions..."):
        retrieval_records, stats_result = cached_query_analysis(
            selected_row["query"],
            selected_row.get("expected_answer"),
            selected_row.get("generated_answer"),
            target_id
        )

    st.markdown("---")
    st.subheader("2. Statistical Distribution Highlights")

    top5_s = stats_result["top5_stats"]
    deep_s = stats_result["deep_stats"]
    all_s = stats_result["all_stats"]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Top-5 Mean Distance", fmt(top5_s['top5_mean']))
    with m2:
        st.metric("Deep-Fetch Mean (6–100)", fmt(deep_s['deep_mean']))
    with m3:
        shift_val = stats_result['mean_distance_shift']
        shift_str = f"+{fmt(shift_val)}" if shift_val is not None and not np.isnan(shift_val) and shift_val >= 0 else fmt(shift_val)
        st.metric("Distance Shift", shift_str)
    with m4:
        st.metric("Top-5 Pairwise Coherence", fmt(stats_result['top5_coherence']))
    with m5:
        target_rank = stats_result.get("target_rank")
        if target_rank:
            st.metric("Target Document Rank", f"#{target_rank}", delta="Found in Deep Fetch" if target_rank > 5 else "Inside Top-5")
        else:
            st.metric("Target Document Rank", "Not Found", delta="Corpus Gap", delta_color="inverse")

    # Advanced statistical metrics
    with st.expander("🔬 Advanced Statistical Indices (Entropy, Curvature, Drop-offs)", expanded=False):
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        with adv_col1:
            st.metric("Shannon Entropy (Bits)", fmt(stats_result['shannon_entropy']))
            st.caption("Information dispersion of distance histogram")
        with adv_col2:
            st.metric("Elbow Curvature Rank", f"Rank #{stats_result['elbow_rank']}")
            st.caption("Point of maximum distance curve curvature")
        with adv_col3:
            st.metric("Top-5 Distance Spread", fmt(top5_s['top5_spread']))
            st.caption("Max minus Min inside Top-5")
        with adv_col4:
            st.metric("Top-5 Sim. Drop-off", fmt(stats_result['top5_sim_dropoff']))
            st.caption("Similarity decay from Rank 1 to Rank 5")

    # Complete Descriptive Statistics Comparison Table
    st.markdown("#### Descriptive Statistics Comparison: Top-5 vs Deep-Fetch vs All-100")
    desc_df = pd.DataFrame({
        "Metric": [
            "Mean Distance (μ)",
            "Median Distance",
            "Variance (σ²)",
            "Standard Deviation (σ)",
            "Minimum Distance",
            "Maximum Distance",
            "Spread (Max - Min)",
            "Interquartile Range (IQR)",
            "Distribution Skewness",
            "Euclidean L2 Equivalent (Unit-Norm)"
        ],
        "Top-5 Baseline": [
            fmt(top5_s['top5_mean']),
            fmt(top5_s['top5_median']),
            fmt(top5_s['top5_var'], 6),
            fmt(top5_s['top5_std']),
            fmt(top5_s['top5_min']),
            fmt(top5_s['top5_max']),
            fmt(top5_s['top5_spread']),
            fmt(top5_s['top5_iqr']),
            fmt(top5_s['top5_skew']),
            fmt(top5_s['top5_l2_mean_equiv']),
        ],
        "Ranks 6–100 Deep Fetch": [
            fmt(deep_s['deep_mean']),
            fmt(deep_s['deep_median']),
            fmt(deep_s['deep_var'], 6),
            fmt(deep_s['deep_std']),
            fmt(deep_s['deep_min']),
            fmt(deep_s['deep_max']),
            fmt(deep_s['deep_spread']),
            fmt(deep_s['deep_iqr']),
            fmt(deep_s['deep_skew']),
            fmt(deep_s['deep_l2_mean_equiv']),
        ],
        "All Top-100 Global": [
            fmt(all_s['all_mean']),
            fmt(all_s['all_median']),
            fmt(all_s['all_var'], 6),
            fmt(all_s['all_std']),
            fmt(all_s['all_min']),
            fmt(all_s['all_max']),
            fmt(all_s['all_spread']),
            fmt(all_s['all_iqr']),
            fmt(all_s['all_skew']),
            fmt(all_s['all_l2_mean_equiv']),
        ]
    })
    st.table(desc_df)

    # 3. Diagnostic Observations Card
    st.markdown("---")
    st.subheader("3. Statistical Diagnostic Observations")
    st.caption("Empirical signals derived from distance distributions. Note: Distinguishes statistical correlation from causality.")

    insights = stats_result.get("diagnostic_insights", [])
    if insights:
        for ins in insights:
            badge_color = "red" if ins["type"] == "ERROR" else "orange" if ins["type"] == "WARNING" else "green"
            with st.container():
                st.markdown(f"**:{badge_color}[{ins['category']}]**: {ins['finding']}")
                st.markdown(f"• **Statistical Implication:** {ins['implication']}")
                st.markdown(f"• **Actionable Suggestion:** `{ins['action']}`")
                st.markdown("---")
    else:
        st.info("No significant statistical anomalies detected for this query.")

    # 4. Interactive Plotly Visualizations
    st.markdown("---")
    st.subheader("4. Interactive Statistical Visualizations")

    tab_hist, tab_curve, tab_box = st.tabs([
        "📊 Distance Distribution Histogram",
        "📈 Ranked Distance Curve",
        "📦 Comparative Box Plots"
    ])

    top5_dists = [r["distance"] for r in stats_result["top_5_records"]]
    deep_dists = [r["distance"] for r in stats_result["deep_records"]]

    with tab_hist:
        fig_hist = plot_top5_vs_deep_histogram(
            top5_dists,
            deep_dists,
            title=f"Histogram Comparison for Query '{selected_row['query'][:50]}...'"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_curve:
        fig_curve = plot_ranked_distance_curve(
            stats_result["all_records"],
            target_rank=stats_result.get("target_rank"),
            title=f"Ranked Distance Progression Curve (Cutoff at Rank 5)"
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with tab_box:
        fig_box = plot_box_comparison(
            top5_dists,
            deep_dists,
            title=f"Top-5 vs Deep-Fetch Distance Dispersion"
        )
        st.plotly_chart(fig_box, use_container_width=True)


# =============================================================================
# MODE 2: AGGREGATE DATASET STATISTICS (50 QUERIES / 5,000 CHUNKS)
# =============================================================================
else:
    st.subheader("📊 Aggregate Dataset Retrieval Analysis (50 Queries, 5,000 Chunks)")

    retrieval_csv_path = Path(DETAILED_RETRIEVAL_CSV)

    # Task 2.6: CSV Recovery Mechanism
    if not retrieval_csv_path.exists():
        st.warning("⚠️ **Detailed retrieval audit dataset (`detailed_retrieval_results.csv`) was not found on disk.**")
        st.info(
            "To perform aggregate statistical analysis across all 50 queries and 5,000 document chunks, "
            "this dataset must be generated using the local ChromaDB connector."
        )
        col_btn, _ = st.columns([2, 3])
        with col_btn:
            regen_clicked = st.button("🔄 Regenerate Detailed Retrieval Dataset (Top-100 Deep-Fetch)", type="primary")

        if regen_clicked:
            try:
                with st.spinner("Executing Top-100 deep-fetch retrieval across all queries... Please wait..."):
                    sample_csv = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
                    if not sample_csv.exists():
                        st.error(f"Source queries dataset not found at `{sample_csv}`. Please ensure Phase 0 synthetic data exists.")
                        st.stop()
                    det_df, _ = connector.ensure_or_regenerate_audit_dataset(
                        queries_csv=sample_csv,
                        output_path=DETAILED_RETRIEVAL_CSV,
                        force_regenerate=True
                    )
                    st.success(f"✅ Successfully regenerated and saved {len(det_df):,} retrieval audit records to `{DETAILED_RETRIEVAL_CSV}`.")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error during retrieval audit regeneration: {e}")
                st.stop()
        else:
            st.stop()

    # Maintenance Expander if dataset exists
    with st.expander("⚙️ Dataset Maintenance / Re-Audit Options", expanded=False):
        st.caption("You can force a re-run of the Top-100 deep-fetch retrieval across all 50 queries to refresh the dataset.")
        if st.button("🔄 Force Re-run Top-100 Deep-Fetch Audit"):
            try:
                with st.spinner("Re-executing Top-100 deep-fetch retrieval across all 50 queries..."):
                    sample_csv = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
                    det_df, _ = connector.ensure_or_regenerate_audit_dataset(
                        queries_csv=sample_csv,
                        output_path=DETAILED_RETRIEVAL_CSV,
                        force_regenerate=True
                    )
                    st.success(f"✅ Refreshed {len(det_df):,} records successfully.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error during refresh: {e}")

    @st.cache_data(show_spinner=False)
    def get_cached_aggregate_stats(csv_path_str: str):
        _, an = get_analyzer_bundle()
        return an.compute_aggregate_dataset_statistics(csv_path_str)

    agg_stats = get_cached_aggregate_stats(str(DETAILED_RETRIEVAL_CSV))

    # Aggregate KPI Cards
    ak1, ak2, ak3, ak4 = st.columns(4)
    with ak1:
        st.metric("Total Queries Audited", agg_stats["total_queries_analyzed"])
    with ak2:
        st.metric("Total Document Chunks", f"{agg_stats['total_records_processed']:,}")
    with ak3:
        st.metric("Global Top-5 Mean Distance", fmt(agg_stats['top5_mean_distance']))
    with ak4:
        shift_val = agg_stats['global_distance_shift']
        shift_str = f"+{fmt(shift_val)}" if shift_val is not None and not np.isnan(shift_val) and shift_val >= 0 else fmt(shift_val)
        st.metric("Global Deep Mean (6–100)", fmt(agg_stats['deep_mean_distance']), delta=shift_str)

    st.markdown("---")

    # Category Level Breakdown Table
    st.markdown("#### Retrieval Distance Profile by Failure Category")
    cat_stats = agg_stats.get("category_statistics", {})
    if cat_stats:
        cat_summary_rows = []
        for cat, c_m in cat_stats.items():
            s_val = c_m['distance_shift']
            s_str = f"+{fmt(s_val)}" if s_val is not None and not np.isnan(s_val) and s_val >= 0 else fmt(s_val)
            cat_summary_rows.append({
                "Failure Category": cat,
                "Query Count": c_m["query_count"],
                "Top-5 Mean Distance": fmt(c_m['top5_mean_dist']),
                "Deep-Fetch Mean (6–100)": fmt(c_m['deep_mean_dist']),
                "Depth Distance Shift": s_str
            })
        st.table(pd.DataFrame(cat_summary_rows))

    # Aggregate Box Plot across categories
    st.markdown("#### Cross-Category Top-5 Distance Distributions")
    det_df = pd.read_csv(DETAILED_RETRIEVAL_CSV)
    fig_cat_box = plot_category_aggregate_boxplots(det_df)
    st.plotly_chart(fig_cat_box, use_container_width=True)

    st.info(
        "💡 **Key Statistical Takeaways**:\n"
        "• **Retrieval Miss**: Exhibits the highest Top-5 mean distance (~0.77), indicating that no retrieved chunk is close in vector space.\n"
        "• **Contradiction**: Exhibits the lowest Top-5 mean distance (~0.59), indicating strong proximity to conflicting sources.\n"
        "• **Ranking Error**: Ground truth evidence is absent in Top-5, but present deeper in the Top-100."
    )

render_academic_disclaimer()
