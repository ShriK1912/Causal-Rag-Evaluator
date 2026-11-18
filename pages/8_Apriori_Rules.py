"""
Apriori Association Rule Mining Dashboard for Causal RAG Failure Diagnostician.

Discovers interpretable co-occurrence patterns between discretized retrieval features
and the 5 RAG failure categories using frequent itemset mining and association rules.
Provides interactive controls for support, confidence, and lift thresholds, category
filtering, interactive scatter and bar visualizations, and downloadable CSV exports.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_FEATURES_CSV,
    FAILURE_CATEGORIES,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_LIFT,
)
from modules.apriori_analyzer import (
    AprioriDiagnosticEngine,
    plot_rule_support_confidence_scatter,
    plot_rules_count_per_category,
    plot_top_rules_bar,
    explain_rule,
)
from modules.ui_components import render_sidebar, render_academic_disclaimer

st.set_page_config(
    page_title="Apriori Rules | RAG Diagnostician",
    page_icon="⛏️",
    layout="wide",
)

render_sidebar("Apriori Rule Mining")

st.title("⛏️ Apriori Association Rule Mining")
st.markdown("""
Discover interpretable, deterministic pattern rules between retrieval feature distributions 
and the **5 RAG Failure Categories**. By mining frequent itemsets across discretized metrics 
(`Low`, `Medium`, `High`), this module extracts high-confidence rules that explain *why* queries fail.
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar Hyperparameter Controls
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Apriori Hyperparameters")

min_support = st.sidebar.slider(
    "Minimum Support (min_support):",
    min_value=0.05,
    max_value=0.50,
    value=float(DEFAULT_MIN_SUPPORT),
    step=0.01,
    help="Fraction of transactions in which an itemset must appear to be considered frequent."
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence (min_confidence):",
    min_value=0.30,
    max_value=1.00,
    value=float(DEFAULT_MIN_CONFIDENCE),
    step=0.05,
    help="Conditional probability P(Failure Category | Feature Pattern)."
)

min_lift = st.sidebar.slider(
    "Minimum Lift (min_lift):",
    min_value=1.00,
    max_value=5.00,
    value=float(DEFAULT_MIN_LIFT),
    step=0.10,
    help="Ratio of observed rule confidence to expected baseline probability. Lift > 1.0 indicates positive association."
)

max_rule_len = st.sidebar.selectbox(
    "Maximum Itemset Length (max_len):",
    [2, 3],
    index=1,
    help="2 = (1 Feature -> 1 Category); 3 = (2 Features -> 1 Category)."
)

discretization_strategy = st.sidebar.radio(
    "Discretization Strategy:",
    ["uniform", "quantile"],
    index=0,
    help="Uniform = Equal-width ranges; Quantile = Equal-frequency sample percentiles."
)


# -----------------------------------------------------------------------------
# Initialize & Run Engine
# -----------------------------------------------------------------------------
if not PROCESSED_FEATURES_CSV.exists():
    st.error(f"Extracted features dataset not found at `{PROCESSED_FEATURES_CSV}`. Please run Phase 1 first.")
    st.stop()

@st.cache_data(show_spinner=False)
def run_apriori_pipeline(min_supp, min_conf, min_lft, max_l, strat):
    engine = AprioriDiagnosticEngine(
        features_csv=PROCESSED_FEATURES_CSV,
        min_support=min_supp,
        min_confidence=min_conf,
        min_lift=min_lft,
        bins=3,
        discretization_strategy=strat,
        max_len=max_l,
    )
    engine.fit()
    return engine

with st.spinner("Mining frequent itemsets and association rules with Apriori..."):
    engine = run_apriori_pipeline(min_support, min_confidence, min_lift, max_rule_len, discretization_strategy)
    summary = engine.get_summary_statistics()


# -----------------------------------------------------------------------------
# Section 1: KPI Highlights
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Transactions", f"{summary['total_samples']} queries")
with kpi2:
    st.metric("Discretized Features", f"{summary['discretized_columns']} cols")
with kpi3:
    st.metric("Frequent Itemsets", f"{summary['frequent_itemsets_count']:,}")
with kpi4:
    st.metric("Total Association Rules", f"{summary['total_rules_count']:,}")
with kpi5:
    st.metric(
        "Diagnostic Rules (Targeted)",
        f"{summary['diagnostic_rules_count']:,}",
        delta=f"{summary['diagnostic_rules_count']} actionable"
    )

st.markdown("---")


# -----------------------------------------------------------------------------
# Section 2: Visualizations (Scatter & Distribution)
# -----------------------------------------------------------------------------
st.subheader("1. Association Rule Distributions: Support, Confidence & Lift")

v_col1, v_col2 = st.columns([3, 2])

with v_col1:
    fig_scatter = plot_rule_support_confidence_scatter(engine.diagnostic_rules)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Each point represents an association rule. High-confidence rules in the top-right quadrant have strong empirical backing.")

with v_col2:
    fig_cat_count = plot_rules_count_per_category(engine.diagnostic_rules)
    st.plotly_chart(fig_cat_count, use_container_width=True)
    st.caption("Distribution of discovered rules targeting each of the 5 mutually exclusive failure modes.")


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 3: Diagnostic Rules Explorer by Failure Category
# -----------------------------------------------------------------------------
st.subheader("2. Diagnostic Association Rules Explorer")

categories = ["All Failure Categories"] + FAILURE_CATEGORIES
selected_cat = st.selectbox("Filter Rules by Failure Category:", categories)

rules_df = engine.diagnostic_rules
if selected_cat != "All Failure Categories":
    rules_df = rules_df[rules_df["target_category"] == selected_cat]

st.markdown(f"**Discovered {len(rules_df)} diagnostic rules matching filter criteria:**")

if not rules_df.empty:
    # Top 3 Rules Spotlight Cards
    st.markdown("#### 🏆 Top Diagnostic Rule Highlights")
    top_3 = rules_df.head(3)
    c_idx = 1
    for _, r in top_3.iterrows():
        ant_disp = r["antecedents_str"]
        cat_disp = r["target_category"]
        conf_disp = r["confidence"] * 100
        lift_disp = r["lift"]
        supp_disp = r["support"] * 100

        badge_color = "red" if lift_disp >= 4.0 else "orange" if lift_disp >= 2.5 else "blue"
        with st.container():
            st.markdown(
                f"**Rule #{c_idx}**: `{ant_disp}` &nbsp; ➔ &nbsp; **:{badge_color}[{cat_disp}]**"
            )
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Rule Confidence", f"{conf_disp:.1f}%")
            with col_m2:
                st.metric("Rule Lift", f"{lift_disp:.2f}x")
            with col_m3:
                st.metric("Corpus Support", f"{supp_disp:.1f}%")
            st.caption(explain_rule(r))
            st.markdown("---")
            c_idx += 1

    # Bar chart of top rules
    cat_for_plot = selected_cat if selected_cat != "All Failure Categories" else None
    fig_bar = plot_top_rules_bar(rules_df, category=cat_for_plot, top_n=10, metric="lift")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed exportable table
    st.markdown("#### 📋 Complete Filtered Association Rules Table")
    display_cols = [
        "antecedents_str", "consequents_str", "target_category",
        "support", "confidence", "lift", "leverage", "conviction"
    ]
    renamed_df = rules_df[display_cols].copy()
    renamed_df.columns = [
        "Feature Antecedents (A)", "Consequent (C)", "Target Failure",
        "Support", "Confidence", "Lift", "Leverage", "Conviction"
    ]

    # Format numbers nicely
    for col in ["Support", "Confidence", "Lift", "Leverage", "Conviction"]:
        renamed_df[col] = renamed_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    st.dataframe(renamed_df, use_container_width=True)

    # Download Button for Rules
    csv_rules = rules_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Association Rules CSV",
        data=csv_rules,
        file_name=f"apriori_rules_{selected_cat.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
else:
    st.warning("No diagnostic rules discovered for the active threshold and category filter. Try lowering Minimum Support or Confidence in the sidebar.")


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 4: Frequent Itemsets & Discretization Audit
# -----------------------------------------------------------------------------
st.subheader("3. Frequent Itemsets & Discretization Methodology Audit")

with st.expander("🔍 Inspect Frequent Itemsets Table (L1, L2, L3)", expanded=False):
    freq_df = engine.frequent_itemsets
    if freq_df is not None and not freq_df.empty:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            len_filter = st.selectbox("Filter Itemset Length:", ["All Lengths"] + sorted(freq_df["length"].unique().tolist()))
        
        filtered_freq = freq_df.copy()
        if len_filter != "All Lengths":
            filtered_freq = filtered_freq[filtered_freq["length"] == len_filter]

        st.markdown(f"**Showing {len(filtered_freq)} frequent itemsets:**")
        display_freq = filtered_freq.copy()
        display_freq["itemset_str"] = display_freq["itemsets"].apply(lambda s: ", ".join(list(s)))
        st.dataframe(
            display_freq[["length", "support", "itemset_str"]].rename(columns={
                "length": "Itemset Size",
                "support": "Support Ratio",
                "itemset_str": "Items in Frequent Pattern"
            }),
            use_container_width=True
        )

        csv_itemsets = filtered_freq.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Frequent Itemsets CSV",
            data=csv_itemsets,
            file_name="frequent_itemsets.csv",
            mime="text/csv",
        )
    else:
        st.info("No frequent itemsets available.")

with st.expander("📊 Inspect Discretized Features & Bin Edges", expanded=False):
    st.markdown("#### Feature Discretization Methodology")
    st.markdown("""
    - **Ordinal Bins**: Continuous features are discretized into 3 intervals: `Low`, `Medium`, and `High`.
    - **Uniform (Equal-Width)**: Interval width $W = \\frac{\\text{Max} - \\text{Min}}{3}$. Preserves absolute scale of retrieval distances.
    - **Quantile (Equal-Frequency)**: Bins by 33.3rd and 66.7th percentiles.
    - **Constant Filtering**: Features with 0 variance (`nunique <= 1`) are excluded from transaction encoding to prevent redundant 100% support artifacts.
    """)

    st.markdown(f"**Constant Columns Dropped ({len(engine.constant_cols_dropped)}):** `{', '.join(engine.constant_cols_dropped)}`")

    st.markdown("#### Sample Discretized Transactions (First 5 Queries)")
    st.dataframe(engine.discretized_df.head(5), use_container_width=True)


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 5: Academic & Causal Limitations Disclaimer
# -----------------------------------------------------------------------------
st.info(
    "💡 **Academic Rigor & Causality Notice**:\n\n"
    "• **Support** measures prevalence ($P(A \\cap C)$); **Confidence** measures conditional certainty ($P(C | A)$); **Lift** measures independence divergence.\n"
    "• An association rule such as `top5_dist_min=High -> Retrieval Miss (Lift=5.0)` indicates that this feature pattern co-occurs exclusively with database gap queries.\n"
    "• **Critical Viva Distinction**: Association rules prove **statistical correlation**, NOT causation. High retrieval distance does not *cause* the document to be missing; rather, both are symptoms of the underlying corpus omission."
)

render_academic_disclaimer()
