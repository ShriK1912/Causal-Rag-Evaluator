"""
Causal RAG Failure Diagnostician Platform
Main Streamlit Entrypoint Application - Dashboard Overview.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import chromadb

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    SYNTHETIC_DATA_DIR,
    FAILURE_CATEGORIES,
    MODEL_NAMES,
    get_active_corpus_config,
    LEGAL_CHROMA_DIR,
    LEGAL_COLLECTION_NAME,
)
from modules.data_ingestion import load_csv, DataQualityReport
from modules.ui_components import render_sidebar, render_academic_disclaimer

st.set_page_config(
    page_title="Causal RAG Failure Diagnostician",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standardized Sidebar with Domain Switcher
render_sidebar("System Health & Dashboard")

active_cfg = get_active_corpus_config()

st.title("🔬 Causal RAG Failure Diagnostician Platform")
st.markdown("### Classical Machine Learning & Statistical Failure Analysis for RAG Systems")

st.info(f"""
💡 **Active Knowledge Base:** **{active_cfg['icon']} {active_cfg['name']}**  
*Description:* {active_cfg['domain_desc']}  
*(You can switch between the **Academic Corpus** and the external **Legal & Law Corpus** anytime using the dropdown in the left sidebar to demonstrate multi-corpus generalization!)*
""")

# Diagnostics & System Environment Overview
st.markdown("---")
st.subheader("🛠️ Phase 0 & 1: System Health & Ingestion Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

with col2:
    chroma_exists = active_cfg["chroma_dir"].exists()
    st.metric(f"Vector Store ({active_cfg['name'].split()[0]})", "Connected" if chroma_exists else "Missing")

with col3:
    csv_sample_path = active_cfg["queries_csv"]
    st.metric("Benchmark Queries", f"{active_cfg['queries_count']} Queries" if csv_sample_path.exists() else "Missing")

with col4:
    st.metric("Failure Taxonomy", f"{len(FAILURE_CATEGORIES)} Categories")

# ChromaDB & Corpus Status
st.markdown("---")
st.subheader("📚 Connected Vector Database Status")
if active_cfg["chroma_dir"].exists():
    try:
        client = chromadb.PersistentClient(path=str(active_cfg["chroma_dir"]))
        collections = client.list_collections()
        col_names = [c.name for c in collections]
        st.success(f"Connected to ChromaDB storage at `{active_cfg['chroma_dir']}`")
        st.write(f"**Available Collections in active store:** {', '.join(col_names)}")

        if active_cfg["collection_name"] in col_names:
            col = client.get_collection(active_cfg["collection_name"])
            st.info(f"Target Collection `{active_cfg['collection_name']}` contains **{col.count()} indexed documents** (Cosine Distance Space).")
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {e}")
else:
    st.warning(f"ChromaDB directory not found at `{active_cfg['chroma_dir']}`.")

# Dataset Ingestion & Validation Section
st.markdown("---")
st.subheader("📋 Active Dataset & Validation Overview")

synthetic_csv_path = active_cfg["queries_csv"]

# Auto-load synthetic if not in session state
if "dataset_report" not in st.session_state and synthetic_csv_path.exists():
    st.session_state["dataset_report"] = load_csv(synthetic_csv_path, default_provenance="synthetic_controlled_v1")
    st.session_state["validated_df"] = st.session_state["dataset_report"].cleaned_df

if "dataset_report" in st.session_state:
    report: DataQualityReport = st.session_state["dataset_report"]
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active Queries", report.valid_records_count)
    with m2:
        st.metric("Invalid Records", report.invalid_records_count)
    with m3:
        st.metric("Duplicates Filtered", report.duplicate_records_count)
    with m4:
        st.metric("Ground Truth Status", "Labeled (5 Classes)" if report.has_labels else "Unlabeled")

    st.markdown("**Sample Valid Records:**")
    st.dataframe(report.cleaned_df.head(5), use_container_width=True)

    st.markdown("**Failure Class Distribution:**")
    class_counts = pd.DataFrame(
        list(report.failure_category_distribution.items()),
        columns=["Failure Category", "Query Count"]
    )
    st.table(class_counts)
    st.info("💡 To upload a custom dataset or review the full data-quality report, navigate to **'2_Data_Ingestion'** in the sidebar.")
else:
    st.warning("No dataset loaded. Navigate to **Data Ingestion** page in the sidebar.")

# Render standardized academic disclaimer
render_academic_disclaimer()
