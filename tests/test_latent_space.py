"""
Unit tests for Latent Space and Database Gap Visualizer (Phase 3).
Verifies PCA 3D dimensionality reduction, centered vs standardized variance,
K-Means inertia/silhouette sweeps, DBSCAN noise detection, edge-case handling,
and document text preservation.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CHROMA_PERSIST_DIR, DEFAULT_COLLECTION_NAME, RANDOM_STATE
from modules.latent_space_visualizer import (
    load_corpus_embeddings,
    apply_pca_3d,
    get_pca_explained_variance,
    compare_pca_scaling_strategies,
    run_kmeans_sweep,
    run_kmeans_best,
    run_dbscan,
    sweep_dbscan_eps,
    LatentSpaceVisualizer,
)


def test_load_corpus_embeddings_integrity():
    """Verify loading of embeddings, IDs, metadatas, domains, and full text from ChromaDB."""
    embs, doc_ids, metadatas, domains, documents = load_corpus_embeddings(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_dir=CHROMA_PERSIST_DIR
    )

    assert isinstance(embs, np.ndarray)
    assert embs.ndim == 2
    assert embs.shape[0] == 157
    assert embs.shape[1] == 384
    assert len(doc_ids) == 157
    assert len(metadatas) == 157
    assert len(domains) == 157
    assert len(documents) == 157

    # Check non-empty document text
    assert all(len(d) > 0 for d in documents)

    # Check unit-norm property of ChromaDB all-MiniLM-L6-v2 embeddings
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_pca_3d_output_dimensions_and_variance():
    """Verify PCA projects 384D to 3D and computes valid explained variance ratios."""
    embs, _, _, _, _ = load_corpus_embeddings()

    # 1. Centered PCA
    coords, pca_model, scaler = apply_pca_3d(embs, random_state=RANDOM_STATE, standardize=False)
    assert coords.shape == (157, 3)
    assert scaler is None

    stats = get_pca_explained_variance(pca_model)
    assert 0.0 < stats["pc1_var"] < 1.0
    assert 0.0 < stats["pc2_var"] < 1.0
    assert 0.0 < stats["pc3_var"] < 1.0
    assert stats["pc1_var"] >= stats["pc2_var"] >= stats["pc3_var"]
    assert 0.0 < stats["cumulative_3pc"] <= 1.0
    assert stats["total_original_dims"] == 384


def test_compare_pca_scaling_strategies():
    """Verify that centered PCA preserves more variance than standardized PCA on unit vectors."""
    embs, _, _, _, _ = load_corpus_embeddings()
    comp = compare_pca_scaling_strategies(embs, random_state=RANDOM_STATE)

    assert "centered" in comp
    assert "standardized" in comp
    # On unit-normalized vectors, centered PCA should retain higher or equal cumulative variance
    assert comp["centered"]["cumulative_3pc"] >= comp["standardized"]["cumulative_3pc"]


def test_kmeans_sweep_inertia_monotonicity_and_silhouette():
    """Verify K-Means sweep from K=2 to K=10, inertia reduction, and silhouette score validity."""
    embs, _, _, _, _ = load_corpus_embeddings()
    coords, _, _ = apply_pca_3d(embs, random_state=RANDOM_STATE)

    sweep = run_kmeans_sweep(coords, k_range=(2, 10), random_state=RANDOM_STATE)

    assert sweep["k_values"] == list(range(2, 11))
    assert len(sweep["inertias"]) == 9
    assert len(sweep["silhouette_scores"]) == 9

    # Verify inertia is monotonically non-increasing as K grows
    for i in range(len(sweep["inertias"]) - 1):
        assert sweep["inertias"][i] >= sweep["inertias"][i + 1] - 1e-4

    # Verify silhouette scores are in valid range [-1.0, 1.0]
    for sil in sweep["silhouette_scores"]:
        if not np.isnan(sil):
            assert -1.0 <= sil <= 1.0

    assert 2 <= sweep["best_k_silhouette"] <= 10

    # Verify cluster sizes sum to 157
    for k, sizes in sweep["cluster_sizes_per_k"].items():
        assert sum(sizes.values()) == 157


def test_kmeans_best_and_k5_assignments():
    """Verify K-Means clustering at K=5 and label validity."""
    embs, _, _, _, _ = load_corpus_embeddings()
    labels, km = run_kmeans_best(embs, n_clusters=5, random_state=RANDOM_STATE)

    assert len(labels) == 157
    assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
    assert km.n_clusters == 5


def test_dbscan_clustering_and_noise_detection():
    """Verify DBSCAN identifies clusters, detects noise points (-1), and handles sensitivity sweeps."""
    embs, _, _, _, _ = load_corpus_embeddings()

    # Test with standard eps=0.8
    labels, stats = run_dbscan(embs, eps=0.8, min_samples=3)
    assert len(labels) == 157
    assert stats["n_clusters_found"] >= 1
    assert stats["n_noise_points"] >= 0
    assert 0.0 <= stats["noise_fraction"] <= 1.0

    # Test sensitivity sweep
    sweep_results = sweep_dbscan_eps(embs, eps_values=[0.5, 0.8, 1.2], min_samples=3)
    assert len(sweep_results) == 3
    # As eps expands, noise points should decrease or remain constant
    assert sweep_results[0]["n_noise_points"] >= sweep_results[2]["n_noise_points"]


def test_dbscan_edge_cases_all_noise_and_single_cluster():
    """Verify DBSCAN handles extreme parameter settings without exceptions."""
    # Extremely small eps -> all points become noise
    tiny_data = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    labels_noise, stats_noise = run_dbscan(tiny_data, eps=0.001, min_samples=2)
    assert all(l == -1 for l in labels_noise)
    assert stats_noise["n_clusters_found"] == 0
    assert stats_noise["noise_fraction"] == 1.0
    assert np.isnan(stats_noise["silhouette"])

    # Insufficient points (< min_samples)
    labels_few, stats_few = run_dbscan(tiny_data, eps=5.0, min_samples=10)
    assert all(l == -1 for l in labels_few)
    assert stats_few["n_clusters_found"] == 0


def test_pca_edge_case_insufficient_samples():
    """Verify apply_pca_3d raises ValueError when sample count < 3."""
    small_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="at least 3 samples"):
        apply_pca_3d(small_data)


def test_latent_space_visualizer_orchestrator_end_to_end():
    """Verify LatentSpaceVisualizer class executes full pipeline and produces complete export DataFrame."""
    viz = LatentSpaceVisualizer(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_dir=CHROMA_PERSIST_DIR,
        random_state=RANDOM_STATE
    )

    results = viz.run(clustering_space="full", dbscan_eps=0.8, dbscan_min_samples=3)

    assert results["n_docs"] == 157
    assert "pca_stats" in results
    assert "kmeans_results" in results
    assert "dbscan_stats" in results
    assert "insights" in results
    assert len(results["insights"]) >= 3

    # Document DataFrame check
    df_docs = viz.get_document_dataframe()
    assert len(df_docs) == 157
    assert "doc_id" in df_docs.columns
    assert "domain" in df_docs.columns
    assert "document_text" in df_docs.columns
    assert "PC1" in df_docs.columns
    assert "kmeans_best" in df_docs.columns
    assert "dbscan_label" in df_docs.columns
