"""
Latent Space & Database Gap Visualizer Page for Causal RAG Failure Diagnostician.

Visualizes the 384-dimensional latent embedding space of the ChromaDB academic corpus
in interactive 3D using Principal Component Analysis (PCA). Applies K-Means clustering
(Elbow Method, Silhouette Score) and DBSCAN density analysis to identify topic clusters,
isolated outliers, and potential knowledge coverage gaps.
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
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    RANDOM_STATE,
)
from modules.latent_space_visualizer import (
    LatentSpaceVisualizer,
    plot_pca_3d_by_domain,
    plot_pca_3d_by_kmeans,
    plot_pca_3d_by_dbscan,
    plot_elbow_curve,
    plot_silhouette_scores_bar,
    plot_dbscan_eps_sensitivity,
    plot_pca_explained_variance,
    run_kmeans_best,
    run_dbscan,
)
from modules.ui_components import render_sidebar, render_academic_disclaimer

st.set_page_config(
    page_title="Latent Space Visualizer | RAG Diagnostician",
    page_icon="🌌",
    layout="wide",
)

render_sidebar("Latent Space Visualizer")

st.title("🌌 Latent Space & Database Gap Visualizer")
st.markdown("""
Examine the high-dimensional mathematical structure of the vector database corpus.
Using **3D Principal Component Analysis (PCA)**, **K-Means Clustering**, and **DBSCAN Density Analysis**,
this module identifies semantic topic density, isolated outliers, and sparse database regions that may induce **Retrieval Misses**.
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar & Analytical Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Latent Space Configuration")

clustering_space = st.sidebar.radio(
    "Clustering Feature Space:",
    ["Full 384-Dimensional Vectors", "3D PCA Projection Coordinates"],
    index=0,
    help=(
        "Full 384D (Recommended): Clusters are computed on original embeddings without projection loss. "
        "3D PCA: Clusters are formed strictly within the 3D visual space."
    )
)
space_key = "full" if "Full" in clustering_space else "pca_3d"

standardize_pca = st.sidebar.checkbox(
    "Standardize Features before PCA (Z-Score)",
    value=False,
    help=(
        "Unchecked (Recommended): Centered PCA preserves the natural spherical geometry of unit-norm L2 embeddings. "
        "Checked: Standardizes feature variance across dimensions."
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("DBSCAN Hyperparameters")
dbscan_eps = st.sidebar.slider("Neighborhood Radius (eps):", 0.3, 1.5, 0.8, 0.05)
dbscan_min_samples = st.sidebar.slider("Min Core Samples:", 2, 8, 3, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("K-Means Explorer")
selected_k = st.sidebar.slider("Manual Cluster Count (K):", 2, 10, 5, 1)


# -----------------------------------------------------------------------------
# Initialize Engine & Pipeline Execution
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_visualizer_engine():
    return LatentSpaceVisualizer(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_dir=CHROMA_PERSIST_DIR,
        random_state=RANDOM_STATE,
    )

try:
    viz = get_visualizer_engine()
    with st.spinner("Analyzing high-dimensional embedding corpus..."):
        results = viz.run(
            clustering_space=space_key,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            standardize_pca=standardize_pca,
        )
except Exception as e:
    st.error(f"Failed to load or process corpus embeddings: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Section 1: KPI Highlights
# -----------------------------------------------------------------------------
pca_stats = results["pca_stats"]
km_results = results["kmeans_results"]
db_stats = results["dbscan_stats"]
cum_var = pca_stats["cumulative_3pc"] * 100

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Corpus Documents", f"{results['n_docs']} chunks")
with kpi2:
    st.metric("Original Embedding Dim", f"{pca_stats['total_original_dims']}D")
with kpi3:
    st.metric(
        "3D PCA Explained Variance",
        f"{cum_var:.1f}%",
        delta="Preserves Macro Structure" if cum_var >= 45 else "Multidimensional Dispersion"
    )
with kpi4:
    best_k = km_results["best_k_silhouette"]
    st.metric("Optimal K-Means (Sil)", f"K = {best_k}")
with kpi5:
    n_noise = db_stats["n_noise_points"]
    noise_pct = db_stats["noise_fraction"] * 100
    st.metric(
        "DBSCAN Noise / Outliers",
        f"{n_noise} ({noise_pct:.1f}%)",
        delta="Potential Gaps" if n_noise > 0 else "Dense Coverage",
        delta_color="inverse"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Section 2: PCA Dimensionality Reduction & Variance Comparison
# -----------------------------------------------------------------------------
st.subheader("1. Principal Component Analysis (PCA) Variance Decomposition")

col_pca_plot, col_pca_info = st.columns([3, 2])

with col_pca_plot:
    fig_pca_var = plot_pca_explained_variance(pca_stats)
    st.plotly_chart(fig_pca_var, use_container_width=True)

with col_pca_info:
    st.markdown("#### Mathematical & Scaling Findings")
    scaling_comp = results["scaling_comparison"]
    c_var = scaling_comp["centered"]["cumulative_3pc"] * 100
    s_var = scaling_comp["standardized"]["cumulative_3pc"] * 100
    diff_var = scaling_comp["variance_difference"] * 100

    st.markdown(f"""
    - **Centered PCA (Original Unit Embeddings)**: Explains **{c_var:.2f}%** variance in 3D.
    - **Standardized PCA (Z-Score Scaled)**: Explains **{s_var:.2f}%** variance in 3D.
    - **Empirical Difference**: Centered PCA retains **+{diff_var:.2f}%** more information.
    """)

    st.info(
        "💡 **Academic Precondition on Normalization**:\n\n"
        "ChromaDB's `all-MiniLM-L6-v2` embeddings are unit-norm ($L_2 = 1.0$). "
        "Standardizing unit vectors divides each feature by its empirical standard deviation, "
        "which distorts natural spherical angles and inflates low-variance noise. "
        "Mean-centering without feature scaling is mathematically superior for unit-norm vector spaces."
    )


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 3: Interactive 3D Latent Space Visualizations
# -----------------------------------------------------------------------------
st.subheader("2. Interactive 3D Latent Space Visualizations")
st.caption("Rotate, zoom, pan, and hover over individual document nodes to inspect text and metadata.")

tab_domain, tab_kmeans, tab_dbscan = st.tabs([
    "🏷️ Knowledge Domain Coloring",
    "🎯 K-Means Cluster Assignments",
    "🔍 DBSCAN Density & Sparse Noise",
])

with tab_domain:
    fig_domain = plot_pca_3d_by_domain(
        pca_coords=viz.pca_coords,
        domains=viz.domains,
        doc_ids=viz.doc_ids,
        documents=viz.documents,
        title="Corpus Embedding Space — Colored by Ground-Truth Domain Metadata",
    )
    st.plotly_chart(fig_domain, use_container_width=True)

with tab_kmeans:
    # Compute clustering on selected K
    if selected_k != km_results["best_k_silhouette"]:
        active_cluster_data = viz.embeddings if space_key == "full" else viz.pca_coords
        custom_labels, _ = run_kmeans_best(active_cluster_data, n_clusters=selected_k, random_state=RANDOM_STATE)
    else:
        custom_labels = viz.kmeans_labels_best

    fig_kmeans = plot_pca_3d_by_kmeans(
        pca_coords=viz.pca_coords,
        kmeans_labels=custom_labels,
        domains=viz.domains,
        doc_ids=viz.doc_ids,
        documents=viz.documents,
        n_clusters=selected_k,
        title="Corpus Embedding Space — K-Means Cluster Partitions",
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

with tab_dbscan:
    fig_dbscan = plot_pca_3d_by_dbscan(
        pca_coords=viz.pca_coords,
        dbscan_labels=viz.dbscan_labels,
        domains=viz.domains,
        doc_ids=viz.doc_ids,
        documents=viz.documents,
        title=f"Corpus Latent Space — DBSCAN Density Clusters & Sparse Outliers (eps={dbscan_eps}, min_samples={dbscan_min_samples})",
    )
    st.plotly_chart(fig_dbscan, use_container_width=True)
    st.caption("Red diamond markers denote DBSCAN noise points (label = -1). These represent semantically sparse vector regions.")


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 4: Cluster Quality & Hyperparameter Sweeps
# -----------------------------------------------------------------------------
st.subheader("3. Cluster Evaluation: Elbow Method & Silhouette Comparison")

col_elbow, col_sil = st.columns(2)

with col_elbow:
    fig_elbow = plot_elbow_curve(
        k_values=km_results["k_values"],
        inertias=km_results["inertias"],
        best_k=selected_k,
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.caption("Within-Cluster Sum of Squares (Inertia) diminishes as K increases. The elbow indicates optimal partition efficiency.")

with col_sil:
    fig_sil = plot_silhouette_scores_bar(
        k_values=km_results["k_values"],
        silhouette_scores=km_results["silhouette_scores"],
        best_k=best_k,
    )
    st.plotly_chart(fig_sil, use_container_width=True)
    st.caption("Mean Silhouette Score balances cluster compactness against inter-cluster separation. Gold indicates the global maximum.")

# DBSCAN sensitivity curve
st.markdown("#### DBSCAN Parameter Sensitivity Sweep")
fig_sens = plot_dbscan_eps_sensitivity(results["dbscan_eps_sweep"])
st.plotly_chart(fig_sens, use_container_width=True)


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 5: Diagnostic Insights & Academic Interpretations
# -----------------------------------------------------------------------------
st.subheader("4. Diagnostic Insights: Corpus Structure & Potential Gaps")

for ins in results["insights"]:
    color = "red" if ins["level"] == "WARNING" else "blue"
    with st.container():
        st.markdown(f"**:{color}[{ins['category']}]**: {ins['observation']}")
        st.markdown(f"• **Engineering Implication:** {ins['implication']}")
        st.markdown(f"• **Academic Note:** _{ins['academic_note']}_")
        st.markdown("---")


# -----------------------------------------------------------------------------
# Section 6: Interactive Document Inspector
# -----------------------------------------------------------------------------
st.subheader("5. Interactive Document Inspector")
st.markdown("Inspect the raw text, metadata, and mathematical projections of any document chunk in the corpus.")

df_docs = viz.get_document_dataframe()

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    domain_filter = st.selectbox("Filter by Knowledge Domain:", ["All Domains"] + sorted(df_docs["domain"].unique()))
with filter_col2:
    kmeans_filter = st.selectbox("Filter by K-Means Cluster:", ["All Clusters"] + [f"Cluster {c}" for c in sorted(df_docs["kmeans_best"].unique())])
with filter_col3:
    dbscan_filter = st.selectbox("Filter by DBSCAN Status:", ["All Documents", "⚠️ Noise Points Only (label = -1)", "Dense Clusters Only"])

# Apply filters
filtered_docs = df_docs.copy()
if domain_filter != "All Domains":
    filtered_docs = filtered_docs[filtered_docs["domain"] == domain_filter]
if kmeans_filter != "All Clusters":
    k_num = int(kmeans_filter.replace("Cluster ", ""))
    filtered_docs = filtered_docs[filtered_docs["kmeans_best"] == k_num]
if dbscan_filter == "⚠️ Noise Points Only (label = -1)":
    filtered_docs = filtered_docs[filtered_docs["dbscan_label"] == -1]
elif dbscan_filter == "Dense Clusters Only":
    filtered_docs = filtered_docs[filtered_docs["dbscan_label"] != -1]

st.markdown(f"**Showing {len(filtered_docs)} of {len(df_docs)} corpus documents:**")

if not filtered_docs.empty:
    doc_options = [
        f"{row['doc_id']} | [{row['domain']}] | K-Means #{row['kmeans_best']} | DBSCAN #{row['dbscan_label']}"
        for _, row in filtered_docs.iterrows()
    ]
    selected_doc_option = st.selectbox("Select Document Chunk to Inspect:", doc_options)
    selected_doc_id = selected_doc_option.split(" | ")[0].strip()
    doc_row = df_docs[df_docs["doc_id"] == selected_doc_id].iloc[0]

    d_col1, d_col2 = st.columns([1, 2])
    with d_col1:
        st.markdown(f"**Document ID:** `{doc_row['doc_id']}`")
        st.markdown(f"**Domain:** `{doc_row['domain']}`")
        st.markdown(f"**Character Length:** `{doc_row['char_length']}` chars")
        st.markdown(f"**K-Means Assignment:** `Cluster {doc_row['kmeans_best']}`")
        if doc_row["dbscan_label"] == -1:
            st.error("**DBSCAN Status:** ⚠️ Sparse Outlier / Noise")
        else:
            st.success(f"**DBSCAN Status:** Dense Cluster #{doc_row['dbscan_label']}")
        st.markdown(f"**3D PCA Coordinates:**")
        st.code(f"PC1: {doc_row['PC1']:+.4f}\nPC2: {doc_row['PC2']:+.4f}\nPC3: {doc_row['PC3']:+.4f}")

    with d_col2:
        st.markdown("**Document Content:**")
        st.text_area("Chunk Text Content", doc_row["document_text"], height=160, disabled=True)

    with st.expander("📋 View Filtered Document Table (Exportable)", expanded=False):
        st.dataframe(
            filtered_docs[["doc_id", "domain", "char_length", "PC1", "PC2", "PC3", "kmeans_best", "dbscan_label"]],
            use_container_width=True
        )
else:
    st.warning("No documents match the active filter criteria.")

render_academic_disclaimer()
