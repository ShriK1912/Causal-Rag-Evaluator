"""
UI Components and Visual Standardization for Causal RAG Diagnostician.
Provides unified sidebar branding, system health badges, academic disclaimers,
and consistent layout helpers across all Streamlit pages.
"""

from pathlib import Path
from typing import Optional
import streamlit as st
import pandas as pd

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    SYNTHETIC_DATA_DIR,
    PROCESSED_FEATURES_CSV,
    BEST_MODEL_PATH,
    FAILURE_CATEGORIES,
    get_active_corpus_config,
)


def render_sidebar(current_page: str = "Overview") -> None:
    """
    Renders standardized branding, active navigation badge,
    domain corpus switcher, and system status indicators in the Streamlit sidebar.
    """
    with st.sidebar:
        st.title("🔬 RAG Diagnostician")
        st.markdown("**Academic Machine Learning Platform**")
        st.caption(f"📍 Current View: **{current_page}**")
        st.markdown("---")

        # ---------------------------------------------------------------------
        # Active RAG Domain Selector (Allows demonstrating corpus independence)
        # ---------------------------------------------------------------------
        domain_options = [
            "🏛️ Academic Corpus (157 Docs)",
            "⚖️ Legal & Law Corpus (116 Docs)"
        ]
        curr_domain = st.session_state.get("active_rag_domain", domain_options[0])
        default_idx = 1 if "Legal" in str(curr_domain) else 0

        selected_domain = st.selectbox(
            "🗂️ **Active RAG Domain:**",
            domain_options,
            index=default_idx,
            help="Switch between the default Academic RAG corpus and the external Legal RAG knowledge base to verify cross-corpus generalization."
        )
        if selected_domain != st.session_state.get("active_rag_domain"):
            st.session_state["active_rag_domain"] = selected_domain
            # Clear caches that depend on dataset
            if "dataset_report" in st.session_state:
                del st.session_state["dataset_report"]
            if "validated_df" in st.session_state:
                del st.session_state["validated_df"]

        active_cfg = get_active_corpus_config()
        st.caption(f"Connected: **{active_cfg['name']}** ({active_cfg['doc_count']} docs)")
        st.markdown("---")

        # Quick System Status Expander
        with st.expander("📊 System Environment Status", expanded=False):
            # ChromaDB Status
            chroma_ok = active_cfg["chroma_dir"].exists()
            chroma_badge = f"🟢 Ready ({active_cfg['doc_count']} Docs)" if chroma_ok else "🔴 Not Initialized"
            st.markdown(f"**Vector Store:** {chroma_badge}")

            # Benchmark Dataset
            sample_csv = active_cfg["queries_csv"]
            bench_badge = f"🟢 Available ({active_cfg['queries_count']} Qs)" if sample_csv.exists() else "🔴 Missing"
            st.markdown(f"**Query Benchmark:** {bench_badge}")

            # Extracted Features
            feat_badge = "🟢 Extracted (36 Feats)" if active_cfg["features_csv"].exists() else "🔴 Pending"
            st.markdown(f"**Feature Dataset:** {feat_badge}")

            # ML Diagnostic Model
            model_badge = "🟢 Trained (6 Models)" if active_cfg["all_models_path"].exists() else "🟡 Unfitted"
            st.markdown(f"**ML Suite:** {model_badge}")

            st.markdown(f"**Taxonomy:** {len(FAILURE_CATEGORIES)} failure classes")

        st.markdown("---")
        st.caption(
            "An academic, zero-API diagnostic tool for analyzing RAG failures "
            "using distance curves, PCA, DBSCAN, Apriori mining, and 6 classical ML classifiers."
        )


def render_academic_disclaimer() -> None:
    """
    Renders standardized academic disclaimer callout regarding causality vs correlation.
    """
    st.markdown("---")
    st.markdown("""
> [!NOTE]
> **Academic Note on Statistical Diagnostic Evidence vs. True Causality**:  
> In vector-retrieval RAG architectures, empirical feature indicators (e.g., elevated cosine distances, 
> high association rule lift, or softmax probabilities) provide **diagnostic association**, not mathematical causal proof. 
> For example, an association rule linking high distance to a *Retrieval Miss* identifies an empirical symptom 
> of an underlying missing document in the corpus. The platform maintains rigorous methodological distinction 
> between predictive correlation and causal mechanisms.
""")
