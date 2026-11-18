"""
Data Ingestion and Quality Validation Page for Causal RAG Diagnostician.
Allows users to upload custom CSVs or load the benchmark synthetic dataset,
performs schema validation, displays quality metrics, and previews data.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    FAILURE_CATEGORIES,
    get_active_corpus_config,
)
from modules.data_ingestion import load_csv, validate_rag_dataset, DataQualityReport
from modules.ui_components import render_sidebar, render_academic_disclaimer

st.set_page_config(
    page_title="Data Ingestion | RAG Diagnostician",
    page_icon="📥",
    layout="wide"
)

render_sidebar("Data Ingestion & Validation")

active_cfg = get_active_corpus_config()

st.title("📥 Data Ingestion & Quality Validation")
st.markdown(f"""
Upload a CSV dataset containing failed RAG interactions or load the controlled benchmark.
Currently Active Domain: **{active_cfg['icon']} {active_cfg['name']}** (*{active_cfg['queries_count']} queries ready*).
""")

st.markdown("---")

# Data Source Selection
st.subheader("1. Select Dataset Source")
source_option = st.radio(
    "Choose Data Source:",
    [f"Use Active Benchmark Dataset ({active_cfg['name']})", "Upload Custom CSV File"],
    horizontal=True
)

report: DataQualityReport = None
synthetic_csv_path = active_cfg["queries_csv"]

col_ctrl1, col_ctrl2 = st.columns([2, 1])

with col_ctrl1:
    if source_option.startswith("Use Active Benchmark Dataset"):
        st.info(f"Using pre-configured benchmark at `{synthetic_csv_path}`")
        if synthetic_csv_path.exists():
            if st.button("🔄 Ingest & Validate Benchmark Dataset", type="primary") or "dataset_report" not in st.session_state:
                with st.spinner("Validating dataset..."):
                    report = load_csv(synthetic_csv_path, default_provenance=f"{active_cfg['key']}_controlled_v1")
                    st.session_state["dataset_report"] = report
                    st.session_state["validated_df"] = report.cleaned_df
        else:
            st.error(f"Benchmark dataset not found at `{synthetic_csv_path}`.")

    else:
        uploaded_file = st.file_uploader(
            "Upload Failed RAG Queries CSV",
            type=["csv"],
            help="CSV containing columns: query, expected_answer, generated_answer, failure_label, etc."
        )
        provenance_tag = st.text_input("Provenance / Data Origin Tag", value="user_uploaded_v1")
        drop_dups = st.checkbox("Drop duplicate queries automatically", value=True)

        if uploaded_file is not None:
            if st.button("🚀 Ingest & Validate Uploaded CSV", type="primary"):
                with st.spinner("Parsing and validating uploaded file..."):
                    report = load_csv(
                        file_or_path=uploaded_file,
                        default_provenance=provenance_tag,
                        drop_duplicates=drop_dups
                    )
                    st.session_state["dataset_report"] = report
                    st.session_state["validated_df"] = report.cleaned_df

# Check session state for report
if "dataset_report" in st.session_state:
    report = st.session_state["dataset_report"]

if report is not None:
    st.markdown("---")
    st.subheader("2. Data Quality & Validation Report")

    # Top-level KPI cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Ingested", report.total_records_ingested)
    with m2:
        st.metric("Valid Records", report.valid_records_count, delta=None)
    with m3:
        st.metric("Invalid Records", report.invalid_records_count, delta=None if report.invalid_records_count == 0 else f"-{report.invalid_records_count}", delta_color="inverse")
    with m4:
        st.metric("Duplicates Dropped", report.duplicate_records_count)
    with m5:
        st.metric("Ground Truth Labels", "Present" if report.has_labels else "Unlabeled Mode")

    # Status alerts
    if report.is_valid_dataset:
        st.success("✅ Dataset Validation Passed: Ready for retrieval & feature extraction.")
    else:
        st.error("❌ Dataset Validation Failed: Critical mandatory columns or valid rows missing.")

    # Validation Messages, Warnings, and Errors
    if report.validation_errors:
        with st.expander("🚨 Validation Errors (Critical)", expanded=True):
            for err in report.validation_errors:
                st.error(f"• {err}")

    if report.validation_warnings:
        with st.expander("⚠️ Data Quality Warnings & Cleaning Actions", expanded=False):
            for warn in report.validation_warnings:
                st.warning(f"• {warn}")

    if report.validation_messages:
        with st.expander("ℹ️ Ingestion Messages & Schema Normalizations", expanded=False):
            for msg in report.validation_messages:
                st.info(f"• {msg}")

    # Missing Values Breakdown
    st.markdown("#### Missing Values by Column")
    if report.missing_values_by_column:
        mv_df = pd.DataFrame(
            list(report.missing_values_by_column.items()),
            columns=["Column Name", "Missing Count"]
        )
        mv_df["Missing Percentage (%)"] = (
            (mv_df["Missing Count"] / max(report.total_records_ingested, 1)) * 100
        ).round(2)
        st.dataframe(mv_df, use_container_width=True)

    # Failure Category Distribution
    if report.has_labels and report.failure_category_distribution:
        st.markdown("#### Failure Category Distribution")
        col_chart, col_table = st.columns([3, 2])
        
        cat_df = pd.DataFrame(
            list(report.failure_category_distribution.items()),
            columns=["Failure Category", "Count"]
        )
        
        with col_chart:
            fig = px.bar(
                cat_df,
                x="Failure Category",
                y="Count",
                color="Failure Category",
                title="Ground Truth Failure Class Breakdown",
                text="Count"
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_table:
            cat_df["Percentage (%)"] = ((cat_df["Count"] / cat_df["Count"].sum()) * 100).round(2)
            st.table(cat_df)

    # Valid Records Preview
    st.markdown("---")
    st.subheader(f"3. Valid Records Preview ({len(report.cleaned_df)} records)")
    st.dataframe(report.cleaned_df, use_container_width=True)

    # Invalid Records Preview (if any)
    if report.invalid_records_count > 0:
        st.markdown("---")
        st.subheader(f"⚠️ Flagged Invalid Records ({len(report.invalid_df)} records)")
        st.caption("These records violated validation rules (e.g., empty query, unrecognized failure label) and were segregated.")
        st.dataframe(report.invalid_df, use_container_width=True)

    # Export / Save clean data
    st.markdown("---")
    save_col1, save_col2 = st.columns([1, 4])
    with save_col1:
        csv_data = report.cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Cleaned CSV",
            data=csv_data,
            file_name="validated_failed_queries.csv",
            mime="text/csv",
            type="secondary"
        )
    with save_col2:
        if st.button("📁 Persist to `data/processed/validated_queries.csv`"):
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            out_path = PROCESSED_DATA_DIR / "validated_queries.csv"
            report.cleaned_df.to_csv(out_path, index=False)
            st.success(f"Saved {len(report.cleaned_df)} records to `{out_path}`.")

render_academic_disclaimer()
