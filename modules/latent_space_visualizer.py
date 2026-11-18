"""
Latent Space & Database Gap Visualizer for Causal RAG Failure Diagnostician Platform.

Retrieves all document embeddings from ChromaDB and applies dimensionality reduction
(PCA) and clustering (K-Means, DBSCAN) to visualize the latent structure of the
knowledge corpus. Identifies potential database gaps by evaluating embedding density
and outlier patterns in vector space.

Key Technical Considerations:
  - Embeddings in ChromaDB are 384-dimensional unit-norm L2 vectors (all-MiniLM-L6-v2).
  - Centered PCA (mean-centering without feature-variance scaling) preserves the spherical
    geometry of normalized embeddings. Standardizing unit vectors distorts angular distances
    and inflates noise dimensions. Both centered and standardized PCA are compared.
  - Clustering is performed on the full 384-dimensional embedding space to prevent
    information loss, while 3D PCA coordinates are utilized strictly for spatial visualization.
  - K-Means evaluates k = 2 to 10 using WCSS (Inertia) and Silhouette Scores.
  - DBSCAN identifies dense topic regions and isolated noise points (label = -1).
    Academic Caveat: Noise points indicate low embedding density, but do not automatically
    prove functional RAG failure unless user queries specifically target those sparse regions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    RANDOM_STATE,
    FAILURE_CATEGORIES,
)
from modules.chroma_connector import ChromaConnector

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Loading (Task 3.1)
# ---------------------------------------------------------------------------

def load_corpus_embeddings(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_dir: Union[str, Path] = CHROMA_PERSIST_DIR,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], List[str], List[str]]:
    """
    Retrieves all stored document embeddings, IDs, metadatas, and text from ChromaDB.

    Returns:
        embeddings_matrix : np.ndarray of shape (n_docs, embedding_dim)
        doc_ids           : List of document IDs in ChromaDB order
        metadatas         : List of metadata dicts (keys: 'domain', 'char_length')
        domains           : List of domain strings parallel to embeddings_matrix rows
        documents         : List of raw document chunk texts
    """
    connector = ChromaConnector(persist_dir=persist_dir)
    col = connector.get_collection(collection_name)

    result = col.get(include=["embeddings", "documents", "metadatas"])

    raw_embeddings = result.get("embeddings")
    doc_ids: List[str] = result.get("ids") or []
    metadatas: List[Dict[str, Any]] = result.get("metadatas") or [{} for _ in doc_ids]
    documents: List[str] = result.get("documents") or ["" for _ in doc_ids]

    # Robust validation handling numpy arrays and lists safely
    if raw_embeddings is None or len(raw_embeddings) == 0:
        raise ValueError("No embeddings found in ChromaDB collection.")

    embeddings_matrix = np.array(raw_embeddings, dtype=np.float64)
    if embeddings_matrix.ndim != 2 or embeddings_matrix.shape[0] == 0:
        raise ValueError(f"Invalid embeddings matrix shape: {embeddings_matrix.shape}")

    domains = [m.get("domain", "Unknown") for m in metadatas]

    return embeddings_matrix, doc_ids, metadatas, domains, documents


# ---------------------------------------------------------------------------
# PCA Dimensionality Reduction (Task 3.2)
# ---------------------------------------------------------------------------

def apply_pca_3d(
    embeddings: np.ndarray,
    random_state: int = RANDOM_STATE,
    standardize: bool = False,
) -> Tuple[np.ndarray, PCA, Optional[StandardScaler]]:
    """
    Applies 3-component PCA to project high-dimensional embeddings into 3D space.

    Mathematical Choice regarding Standardization:
      - When standardize=False (Recommended for unit-norm embeddings):
        PCA mean-centers the data (X_centered = X - mean(X)) without dividing by feature
        variances. Since all embeddings have unit Euclidean norm (||x||_2 = 1.0), each
        dimension is already on an identical natural scale. Feature standardization
        artificially inflates low-variance noise dimensions and warps spherical geometry.
      - When standardize=True:
        Applies StandardScaler (z-score scaling). Provided for comparative ablation.

    Args:
        embeddings   : ndarray of shape (n_docs, n_features)
        random_state : Reproducibility seed
        standardize  : If True, apply StandardScaler before PCA; if False, mean-center only

    Returns:
        pca_coords   : ndarray of shape (n_docs, 3)
        pca_model    : Fitted sklearn PCA object
        scaler       : Fitted StandardScaler if standardize else None
    """
    if embeddings.shape[0] < 3:
        raise ValueError("PCA requires at least 3 samples to compute 3 principal components.")

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(embeddings)
    else:
        scaler = None
        X_proc = embeddings

    pca = PCA(n_components=3, random_state=random_state)
    pca_coords = pca.fit_transform(X_proc)

    return pca_coords, pca, scaler


def get_pca_explained_variance(pca_model: PCA) -> Dict[str, Any]:
    """
    Extracts explained variance statistics from a fitted PCA model.

    Returns a dict with per-component and cumulative explained variance ratios.
    """
    ratios = pca_model.explained_variance_ratio_
    return {
        "pc1_var": float(ratios[0]),
        "pc2_var": float(ratios[1]),
        "pc3_var": float(ratios[2]),
        "cumulative_3pc": float(ratios[:3].sum()),
        "n_components": 3,
        "total_original_dims": pca_model.n_features_in_,
    }


def compare_pca_scaling_strategies(
    embeddings: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Compares Centered PCA (on original unit embeddings) against Standardized PCA.
    Empirically demonstrates why centered PCA preserves more variance on normalized vectors.
    """
    _, pca_centered, _ = apply_pca_3d(embeddings, random_state=random_state, standardize=False)
    _, pca_standardized, _ = apply_pca_3d(embeddings, random_state=random_state, standardize=True)

    centered_stats = get_pca_explained_variance(pca_centered)
    standardized_stats = get_pca_explained_variance(pca_standardized)

    return {
        "centered": centered_stats,
        "standardized": standardized_stats,
        "variance_difference": centered_stats["cumulative_3pc"] - standardized_stats["cumulative_3pc"],
    }


# ---------------------------------------------------------------------------
# K-Means Clustering & Evaluation (Task 3.3)
# ---------------------------------------------------------------------------

def run_kmeans_sweep(
    data_matrix: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Runs K-Means clustering across a range of k values (default 2 to 10).
    For each k, records:
      - Inertia (Within-Cluster Sum of Squares, WCSS) — for Elbow Method
      - Silhouette Score — cluster cohesion and separation
      - Cluster size distribution

    Validation Safeguard:
      Silhouette score is computed only when at least 2 distinct clusters exist.

    Args:
        data_matrix : ndarray of shape (n_docs, n_features). Can be full 384D embeddings
                      or 3D PCA coordinates.
        k_range     : (min_k, max_k) inclusive range
        random_state: Reproducibility seed

    Returns:
        dict containing k_values, inertias, silhouette_scores, best_k_silhouette,
        cluster_sizes_per_k, labels_per_k, and models_per_k.
    """
    n_samples = data_matrix.shape[0]
    max_k = min(k_range[1], n_samples - 1)
    min_k = min(k_range[0], max_k)

    if n_samples < 3 or min_k < 2:
        return {
            "k_values": [],
            "inertias": [],
            "silhouette_scores": [],
            "best_k_silhouette": 2,
            "cluster_sizes_per_k": {},
            "labels_per_k": {},
            "models_per_k": {},
        }

    k_values = list(range(min_k, max_k + 1))
    inertias: List[float] = []
    silhouette_scores_list: List[float] = []
    cluster_sizes_per_k: Dict[int, Dict[int, int]] = {}
    labels_per_k: Dict[int, np.ndarray] = {}
    models_per_k: Dict[int, KMeans] = {}

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(data_matrix)
        inertias.append(float(km.inertia_))

        # Check validity for silhouette calculation
        unique_labels = np.unique(labels)
        if len(unique_labels) >= 2 and len(unique_labels) < n_samples:
            sil = float(silhouette_score(data_matrix, labels))
        else:
            sil = np.nan
        silhouette_scores_list.append(sil)

        # Record cluster size counts
        counts = {int(c): int(np.sum(labels == c)) for c in unique_labels}
        cluster_sizes_per_k[k] = counts
        labels_per_k[k] = labels
        models_per_k[k] = km

    # Determine best k by valid maximum silhouette score
    valid_sils = [s if not np.isnan(s) else -1.0 for s in silhouette_scores_list]
    best_idx = int(np.argmax(valid_sils)) if valid_sils else 0
    best_k = k_values[best_idx] if k_values else 2

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "best_k_silhouette": best_k,
        "cluster_sizes_per_k": cluster_sizes_per_k,
        "labels_per_k": labels_per_k,
        "models_per_k": models_per_k,
    }


def run_kmeans_best(
    data_matrix: np.ndarray,
    n_clusters: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, KMeans]:
    """Fits K-Means with the specified number of clusters."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(data_matrix)
    return labels, km


# ---------------------------------------------------------------------------
# DBSCAN Density-Based Clustering (Task 3.4)
# ---------------------------------------------------------------------------

def run_dbscan(
    data_matrix: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies DBSCAN clustering to identify dense semantic neighborhoods and sparse noise points.

    Mathematical Properties:
      - Points with >= min_samples within radius eps are Core Points.
      - Points within eps of a core point but < min_samples neighbors are Border Points.
      - Points with no core point in their eps-neighborhood are Noise Points (label = -1).

    Validation Safeguard:
      Handles cases where all points are classified as noise (0 clusters) or where
      the entire dataset collapses into a single cluster.

    Returns:
        labels : np.ndarray of cluster assignments (-1 = noise)
        stats  : dict containing cluster counts, noise counts, sizes, and silhouette score
    """
    n_samples = data_matrix.shape[0]
    if n_samples < min_samples:
        return np.full(n_samples, -1), {
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters_found": 0,
            "n_noise_points": n_samples,
            "noise_fraction": 1.0,
            "cluster_sizes": {},
            "silhouette": np.nan,
        }

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data_matrix)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    cluster_sizes = {
        int(lbl): int(np.sum(labels == lbl))
        for lbl in sorted(unique_labels)
        if lbl != -1
    }

    # Silhouette calculation safe check
    # Need at least 2 non-noise clusters or 1 cluster + noise with valid points
    non_noise_mask = (labels != -1)
    if n_clusters >= 2 and np.sum(non_noise_mask) > n_clusters:
        try:
            sil = float(silhouette_score(data_matrix[non_noise_mask], labels[non_noise_mask]))
        except Exception:
            sil = np.nan
    else:
        sil = np.nan

    stats = {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters_found": n_clusters,
        "n_noise_points": n_noise,
        "noise_fraction": float(n_noise / n_samples) if n_samples > 0 else 0.0,
        "cluster_sizes": cluster_sizes,
        "silhouette": sil,
    }
    return labels, stats


def sweep_dbscan_eps(
    data_matrix: np.ndarray,
    eps_values: Optional[List[float]] = None,
    min_samples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Runs DBSCAN across multiple eps radii to evaluate cluster stability and noise decay.
    """
    if eps_values is None:
        eps_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    results = []
    for eps in eps_values:
        _, stats = run_dbscan(data_matrix, eps=eps, min_samples=min_samples)
        results.append(stats)
    return results


# ---------------------------------------------------------------------------
# Plotly Visualizations (Tasks 3.2, 3.3, 3.4)
# ---------------------------------------------------------------------------

def _build_hover_text(doc_ids: List[str], domains: List[str], snippets: List[str], extra_label: Optional[List[str]] = None) -> List[str]:
    """Formats rich hover HTML string for 3D Plotly markers."""
    hover_lines = []
    for i in range(len(doc_ids)):
        line = f"<b>Doc ID:</b> {doc_ids[i]}<br><b>Domain:</b> {domains[i]}"
        if extra_label and i < len(extra_label):
            line += f"<br><b>{extra_label[i]}</b>"
        snippet = snippets[i].replace("<", "&lt;").replace(">", "&gt;") if i < len(snippets) else ""
        line += f"<br><br><i>{snippet}</i>"
        hover_lines.append(line)
    return hover_lines


def plot_pca_3d_by_domain(
    pca_coords: np.ndarray,
    domains: List[str],
    doc_ids: List[str],
    documents: List[str],
    title: str = "Corpus Latent Space — PCA 3D (Colored by Knowledge Domain)",
) -> go.Figure:
    """
    Interactive 3D scatter plot of corpus documents colored by domain metadata.
    Includes text snippets on hover for direct document inspection.
    """
    snippets = [d[:110] + "..." if len(d) > 110 else d for d in documents]
    hover_texts = _build_hover_text(doc_ids, domains, snippets)

    df_plot = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "PC3": pca_coords[:, 2],
        "Domain": domains,
        "doc_id": doc_ids,
        "hover": hover_texts,
    })

    fig = px.scatter_3d(
        df_plot,
        x="PC1", y="PC2", z="PC3",
        color="Domain",
        title=title,
        opacity=0.85,
    )
    fig.update_traces(
        marker=dict(size=4),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=df_plot["hover"]
    )
    fig.update_layout(
        legend_title_text="Knowledge Domain",
        margin=dict(l=0, r=0, b=0, t=50),
        height=620,
    )
    return fig


def plot_pca_3d_by_kmeans(
    pca_coords: np.ndarray,
    kmeans_labels: np.ndarray,
    domains: List[str],
    doc_ids: List[str],
    documents: List[str],
    n_clusters: int,
    title: str = "Corpus Latent Space — K-Means Cluster Assignments",
) -> go.Figure:
    """
    Interactive 3D scatter plot colored by K-Means cluster assignments.
    """
    snippets = [d[:110] + "..." if len(d) > 110 else d for d in documents]
    cluster_tags = [f"Cluster {lbl}" for lbl in kmeans_labels]
    hover_texts = _build_hover_text(doc_ids, domains, snippets, extra_label=cluster_tags)

    df_plot = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "PC3": pca_coords[:, 2],
        "Cluster": cluster_tags,
        "Domain": domains,
        "doc_id": doc_ids,
        "hover": hover_texts,
    })

    fig = px.scatter_3d(
        df_plot,
        x="PC1", y="PC2", z="PC3",
        color="Cluster",
        symbol="Domain",
        title=f"{title} (K = {n_clusters})",
        opacity=0.85,
    )
    fig.update_traces(
        marker=dict(size=4),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=df_plot["hover"]
    )
    fig.update_layout(
        legend_title_text="K-Means Cluster",
        margin=dict(l=0, r=0, b=0, t=50),
        height=620,
    )
    return fig


def plot_pca_3d_by_dbscan(
    pca_coords: np.ndarray,
    dbscan_labels: np.ndarray,
    domains: List[str],
    doc_ids: List[str],
    documents: List[str],
    title: str = "Corpus Latent Space — DBSCAN Density Clusters & Sparse Noise",
) -> go.Figure:
    """
    Interactive 3D scatter plot with distinct visual markers for DBSCAN Noise points.
    Core and border clusters are shown as solid spheres, while noise points (label = -1)
    are rendered with high-contrast diamond markers, highlighting sparse database regions.
    """
    snippets = [d[:110] + "..." if len(d) > 110 else d for d in documents]
    dbscan_tags = [
        "⚠️ Noise / Sparse Point (Outlier)" if lbl == -1 else f"Dense Cluster {lbl}"
        for lbl in dbscan_labels
    ]
    hover_texts = _build_hover_text(doc_ids, domains, snippets, extra_label=dbscan_tags)

    fig = go.Figure()

    # Trace 1: Dense Clusters (points with label >= 0)
    dense_mask = (dbscan_labels != -1)
    if np.any(dense_mask):
        fig.add_trace(go.Scatter3d(
            x=pca_coords[dense_mask, 0],
            y=pca_coords[dense_mask, 1],
            z=pca_coords[dense_mask, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=dbscan_labels[dense_mask],
                colorscale="Viridis",
                opacity=0.85,
                colorbar=dict(title="Cluster ID", x=1.02),
            ),
            name="Dense Cluster Member",
            customdata=[hover_texts[i] for i in range(len(hover_texts)) if dense_mask[i]],
            hovertemplate="%{customdata}<extra></extra>",
        ))

    # Trace 2: Noise / Outlier Points (label == -1) — distinct diamond styling
    noise_mask = (dbscan_labels == -1)
    if np.any(noise_mask):
        fig.add_trace(go.Scatter3d(
            x=pca_coords[noise_mask, 0],
            y=pca_coords[noise_mask, 1],
            z=pca_coords[noise_mask, 2],
            mode="markers",
            marker=dict(
                size=7,
                color="#E74C3C",  # Vibrant red
                symbol="diamond",
                opacity=0.95,
                line=dict(color="black", width=1),
            ),
            name="⚠️ Sparse Noise / Outlier (Potential Gap)",
            customdata=[hover_texts[i] for i in range(len(hover_texts)) if noise_mask[i]],
            hovertemplate="%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        height=620,
        legend=dict(x=0.0, y=1.0, orientation="v"),
    )
    return fig


def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    best_k: Optional[int] = None,
) -> go.Figure:
    """
    Plots Within-Cluster Sum of Squares (Inertia / WCSS) vs K.
    The 'elbow' represents the diminishing returns threshold for additional clusters.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=inertias,
        mode="lines+markers",
        name="Inertia (WCSS)",
        line=dict(color="#E74C3C", width=2.5),
        marker=dict(size=8),
    ))

    if best_k is not None and best_k in k_values:
        fig.add_vline(
            x=best_k,
            line_width=2,
            line_dash="dot",
            line_color="#2980B9",
            annotation_text=f"Selected K={best_k}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="K-Means: Elbow Method (Inertia vs. Number of Clusters K)",
        xaxis=dict(title="Number of Clusters (K)", tickmode="array", tickvals=k_values),
        yaxis=dict(title="Within-Cluster Sum of Squares (WCSS)"),
        template="plotly_white",
        height=420,
    )
    return fig


def plot_silhouette_scores_bar(
    k_values: List[int],
    silhouette_scores: List[float],
    best_k: int,
) -> go.Figure:
    """
    Dedicated bar chart of Silhouette Scores across K = 2 to 10.
    Highlights the optimal cluster count with a gold accent.
    """
    colors = ["#F39C12" if k == best_k else "#3498DB" for k in k_values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"K={k}" for k in k_values],
        y=silhouette_scores,
        marker_color=colors,
        text=[f"{s:.3f}" if not np.isnan(s) else "N/A" for s in silhouette_scores],
        textposition="outside",
        name="Silhouette Score",
    ))

    fig.update_layout(
        title="K-Means: Mean Silhouette Score by Cluster Count (Higher is Better)",
        xaxis_title="Cluster Count (K)",
        yaxis_title="Silhouette Score (Cohesion & Separation)",
        yaxis=dict(range=[0, max([s for s in silhouette_scores if not np.isnan(s)] + [0.5]) * 1.25]),
        template="plotly_white",
        height=420,
    )
    return fig


def plot_dbscan_eps_sensitivity(eps_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Plots DBSCAN sensitivity: Clusters Found and Noise Percentage vs eps radius.
    """
    eps_vals = [r["eps"] for r in eps_results]
    n_clusters = [r["n_clusters_found"] for r in eps_results]
    noise_pct = [r["noise_fraction"] * 100 for r in eps_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eps_vals, y=n_clusters,
        mode="lines+markers",
        name="Dense Clusters Found",
        line=dict(color="#8E44AD", width=2.5),
        marker=dict(size=8),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=eps_vals, y=noise_pct,
        mode="lines+markers",
        name="Noise / Outlier %",
        line=dict(color="#E67E22", width=2, dash="dash"),
        marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        title="DBSCAN Parameter Sensitivity: Cluster Count and Noise % vs Neighborhood Radius (eps)",
        xaxis=dict(title="Epsilon (eps) Radius"),
        yaxis=dict(title="Number of Dense Clusters", title_font=dict(color="#8E44AD")),
        yaxis2=dict(
            title="Noise Percentage (%)",
            title_font=dict(color="#E67E22"),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.5, y=1.12, orientation="h"),
        template="plotly_white",
        height=420,
    )
    return fig


def plot_pca_explained_variance(pca_stats: Dict[str, Any]) -> go.Figure:
    """
    Bar chart of explained variance per PCA component with cumulative step line.
    """
    components = ["PC1", "PC2", "PC3"]
    variances = [pca_stats["pc1_var"] * 100, pca_stats["pc2_var"] * 100, pca_stats["pc3_var"] * 100]
    cumulative = np.cumsum(variances).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=components, y=variances,
        name="Individual Variance %",
        marker_color="#2980B9",
        text=[f"{v:.1f}%" for v in variances],
        textposition="outside",
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=components, y=cumulative,
        mode="lines+markers",
        name="Cumulative Explained %",
        line=dict(color="#C0392B", width=2.5),
        marker=dict(size=10),
        yaxis="y1",
    ))
    fig.update_layout(
        title=f"PCA Explained Variance Ratio (3D Projection from {pca_stats['total_original_dims']} Dimensions)",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        yaxis=dict(range=[0, max(cumulative) * 1.30]),
        height=400,
        template="plotly_white",
        legend=dict(x=0.65, y=0.95),
    )
    return fig


# ---------------------------------------------------------------------------
# Interpretations and Diagnostic Insights (Task 3.4 & 3.5)
# ---------------------------------------------------------------------------

def generate_latent_space_insights(
    pca_stats: Dict[str, Any],
    kmeans_results: Dict[str, Any],
    dbscan_stats: Dict[str, Any],
    domain_distribution: Dict[str, int],
) -> List[Dict[str, str]]:
    """
    Generates rule-based diagnostic observations from latent-space structure.

    ACADEMIC DISCLAIMER:
    These observations describe geometric and statistical properties of the embedding
    distribution. They do NOT constitute causal proof of RAG failures. Outlier status
    indicates semantic isolation, but a query only fails if it targets an uncovered concept.
    """
    insights: List[Dict[str, str]] = []

    # 1. PCA Information Retention
    cumvar = pca_stats["cumulative_3pc"] * 100
    if cumvar >= 35.0:
        insights.append({
            "level": "INFO",
            "category": "Dimensionality Reduction Retention",
            "observation": f"The 3D PCA projection captures {cumvar:.1f}% of total corpus variance.",
            "implication": (
                "A significant portion of embedding geometry is retained in 3D space. "
                "Spatial clustering visually reflects macro-level domain separation."
            ),
            "academic_note": "65% of residual variance remains in higher dimensions (PC4–PC384). Clustering is executed on full 384D vectors to prevent projection bias."
        })
    else:
        insights.append({
            "level": "WARNING",
            "category": "High Information Loss in 3D",
            "observation": f"3D PCA explains only {cumvar:.1f}% of embedding variance.",
            "implication": (
                "The corpus is highly multidimensional. Visual distances in 3D may appear closer "
                "or further than their true 384-dimensional cosine proximity."
            ),
            "academic_note": "Evaluations rely on full-dimensional distances, not visual 3D coordinates."
        })

    # 2. K-Means Optimal K vs Failure Taxonomy
    best_k = kmeans_results.get("best_k_silhouette", 5)
    best_sil = max([s for s in kmeans_results.get("silhouette_scores", [0.0]) if not np.isnan(s)] or [0.0])
    insights.append({
        "level": "INFO",
        "category": "Natural Semantic Partitions (K-Means)",
        "observation": f"K-Means identifies K={best_k} as the optimal partition (Mean Silhouette = {best_sil:.3f}).",
        "implication": (
            f"The embedding space partitions naturally into {best_k} semantic clusters. "
            f"{'This aligns with the 5 failure classes.' if best_k == 5 else 'This indicates corpus topic clusters dominate over failure modes.'}"
        ),
        "academic_note": "K-Means clusters group documents by semantic content, not ground-truth failure labels."
    })

    # 3. DBSCAN Noise & Potential Database Gaps
    n_noise = dbscan_stats.get("n_noise_points", 0)
    noise_pct = dbscan_stats.get("noise_fraction", 0.0) * 100
    n_clusters = dbscan_stats.get("n_clusters_found", 0)

    if noise_pct > 15.0:
        insights.append({
            "level": "WARNING",
            "category": "Potential Database Coverage Gap (High Noise)",
            "observation": f"DBSCAN flags {n_noise} documents ({noise_pct:.1f}% of corpus) as noise/outliers at eps={dbscan_stats['eps']}.",
            "implication": (
                "These documents do not belong to dense semantic clusters and lack close neighbors. "
                "Queries targeting topics in these sparse neighborhoods face high risk of Retrieval Miss."
            ),
            "academic_note": "Noise points are statistical density outliers, not necessarily corrupted text."
        })
    else:
        insights.append({
            "level": "INFO",
            "category": "Dense Corpus Topic Cohesion",
            "observation": f"DBSCAN identifies {n_clusters} dense clusters with low noise ({noise_pct:.1f}%).",
            "implication": "Most documents possess close semantic neighbors within their respective domain clusters.",
            "academic_note": "High density reduces vector sparsity but increases risk of polysemous context contamination."
        })

    # 4. Domain Distribution Balance
    if len(domain_distribution) > 1:
        counts = list(domain_distribution.values())
        min_c, max_c = min(counts), max(counts)
        ratio = max_c / min_c if min_c > 0 else float("inf")
        if ratio > 2.5:
            insights.append({
                "level": "WARNING",
                "category": "Domain Imbalance in Corpus",
                "observation": f"Corpus domain representation is skewed (max/min ratio: {ratio:.1f}x: {max_c} vs {min_c} docs).",
                "implication": "Heavily indexed domains may bias Top-K retrieval against underrepresented subjects.",
                "academic_note": "Vector search similarity scales with localized density in embedding space."
            })

    return insights


# ---------------------------------------------------------------------------
# Main Orchestrator Class
# ---------------------------------------------------------------------------

class LatentSpaceVisualizer:
    """
    High-level orchestrator for Phase 3 Latent Space and Database Gap Analysis.
    Coordinates embedding loading, centered PCA, K-Means sweeps, DBSCAN runs,
    insight generation, and data export.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_dir: Union[str, Path] = CHROMA_PERSIST_DIR,
        random_state: int = RANDOM_STATE,
    ):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.random_state = random_state

        # Corpus Data
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: Optional[List[str]] = None
        self.metadatas: Optional[List[Dict[str, Any]]] = None
        self.domains: Optional[List[str]] = None
        self.documents: Optional[List[str]] = None

        # PCA Results
        self.pca_coords: Optional[np.ndarray] = None
        self.pca_model: Optional[PCA] = None
        self.pca_stats: Optional[Dict[str, Any]] = None
        self.scaling_comparison: Optional[Dict[str, Any]] = None

        # K-Means Results
        self.kmeans_sweep_results: Optional[Dict[str, Any]] = None
        self.kmeans_labels_best: Optional[np.ndarray] = None
        self.kmeans_labels_k5: Optional[np.ndarray] = None

        # DBSCAN Results
        self.dbscan_labels: Optional[np.ndarray] = None
        self.dbscan_stats: Optional[Dict[str, Any]] = None
        self.dbscan_eps_sweep: Optional[List[Dict[str, Any]]] = None

        # Interpretations
        self.insights: Optional[List[Dict[str, str]]] = None

    def run(
        self,
        clustering_space: str = "full",  # 'full' (384D) or 'pca_3d' (3D)
        dbscan_eps: float = 0.8,
        dbscan_min_samples: int = 3,
        standardize_pca: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end latent space visualizer pipeline.

        Args:
            clustering_space   : 'full' runs K-Means & DBSCAN on full 384D embeddings;
                                 'pca_3d' runs clustering on the 3D projection.
            dbscan_eps         : Radius parameter for DBSCAN
            dbscan_min_samples : Core point neighborhood threshold
            standardize_pca    : If False (recommended), mean-centers unit-norm vectors;
                                 if True, standardizes features.
        """
        # 1. Load corpus embeddings & texts
        self.embeddings, self.doc_ids, self.metadatas, self.domains, self.documents = (
            load_corpus_embeddings(self.collection_name, self.persist_dir)
        )

        # 2. PCA 3D projection
        self.pca_coords, self.pca_model, _ = apply_pca_3d(
            self.embeddings, random_state=self.random_state, standardize=standardize_pca
        )
        self.pca_stats = get_pca_explained_variance(self.pca_model)
        self.scaling_comparison = compare_pca_scaling_strategies(
            self.embeddings, random_state=self.random_state
        )

        # 3. Determine clustering representation
        cluster_data = self.embeddings if clustering_space == "full" else self.pca_coords

        # 4. K-Means sweep (K = 2 to 10)
        self.kmeans_sweep_results = run_kmeans_sweep(
            cluster_data, k_range=(2, 10), random_state=self.random_state
        )

        # 5. K-Means at optimal K and K=5
        best_k = self.kmeans_sweep_results["best_k_silhouette"]
        self.kmeans_labels_best, _ = run_kmeans_best(
            cluster_data, n_clusters=best_k, random_state=self.random_state
        )
        self.kmeans_labels_k5, _ = run_kmeans_best(
            cluster_data, n_clusters=5, random_state=self.random_state
        )

        # 6. DBSCAN epsilon sweep
        self.dbscan_eps_sweep = sweep_dbscan_eps(
            cluster_data, min_samples=dbscan_min_samples
        )

        # 7. DBSCAN final run
        self.dbscan_labels, self.dbscan_stats = run_dbscan(
            cluster_data, eps=dbscan_eps, min_samples=dbscan_min_samples
        )

        # 8. Domain counts
        from collections import Counter
        domain_dist = dict(Counter(self.domains))

        # 9. Diagnostic insights
        self.insights = generate_latent_space_insights(
            self.pca_stats,
            self.kmeans_sweep_results,
            self.dbscan_stats,
            domain_dist,
        )

        return {
            "n_docs": len(self.doc_ids),
            "pca_stats": self.pca_stats,
            "scaling_comparison": self.scaling_comparison,
            "kmeans_results": self.kmeans_sweep_results,
            "dbscan_stats": self.dbscan_stats,
            "dbscan_eps_sweep": self.dbscan_eps_sweep,
            "insights": self.insights,
            "domain_distribution": domain_dist,
            "clustering_space": clustering_space,
        }

    def get_document_dataframe(self) -> pd.DataFrame:
        """
        Returns a tabular DataFrame containing document IDs, domain, full text,
        PCA 3D coordinates, and cluster assignments.
        """
        if self.pca_coords is None:
            raise RuntimeError("Execute .run() prior to accessing document dataframe.")
        return pd.DataFrame({
            "doc_id": self.doc_ids,
            "domain": self.domains,
            "char_length": [len(d) for d in self.documents],
            "PC1": self.pca_coords[:, 0],
            "PC2": self.pca_coords[:, 1],
            "PC3": self.pca_coords[:, 2],
            "kmeans_best": self.kmeans_labels_best,
            "kmeans_k5": self.kmeans_labels_k5,
            "dbscan_label": self.dbscan_labels,
            "document_text": self.documents,
        })
