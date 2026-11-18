"""
Unit tests for Deep-Fetch Statistical Analyzer Module (Phase 2).
Tests distribution metrics calculations, Top-5 coherence, Shannon entropy,
elbow curvature detection, query-level statistical breakdown, and aggregate metrics.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CHROMA_PERSIST_DIR, DETAILED_RETRIEVAL_CSV
from modules.statistical_analyzer import (
    compute_distribution_metrics,
    compute_pairwise_top5_coherence,
    compute_shannon_entropy,
    compute_elbow_rank,
    StatisticalAnalyzer,
)
from modules.chroma_connector import ChromaConnector


def test_compute_distribution_metrics_known_values():
    """Verify parametric and non-parametric statistics on known numerical inputs."""
    dists = [0.2, 0.4, 0.6, 0.8, 1.0]
    metrics = compute_distribution_metrics(dists, prefix="test")

    assert pytest.approx(metrics["test_mean"], 1e-4) == 0.6
    assert pytest.approx(metrics["test_median"], 1e-4) == 0.6
    assert pytest.approx(metrics["test_min"], 1e-4) == 0.2
    assert pytest.approx(metrics["test_max"], 1e-4) == 1.0
    assert pytest.approx(metrics["test_spread"], 1e-4) == 0.8
    assert metrics["test_var"] > 0.0
    assert metrics["test_std"] > 0.0
    assert metrics["test_iqr"] > 0.0
    assert metrics["test_l2_mean_equiv"] > 0.0


def test_compute_distribution_metrics_empty():
    """Verify graceful NaN handling for empty distance arrays."""
    metrics = compute_distribution_metrics([], prefix="empty")
    assert np.isnan(metrics["empty_mean"])
    assert np.isnan(metrics["empty_median"])
    assert np.isnan(metrics["empty_spread"])


def test_top5_coherence():
    """Verify pairwise text coherence calculation."""
    # Identical documents -> Coherence 1.0
    identical_docs = [
        "Supervised machine learning algorithms",
        "Supervised machine learning algorithms",
        "Supervised machine learning algorithms",
    ]
    assert compute_pairwise_top5_coherence(identical_docs) == 1.0

    # Disjoint documents -> Coherence 0.0
    disjoint_docs = [
        "Astrophysics stellar evolution nucleosynthesis",
        "Neanderthal pleistocene anthropological fossil",
        "Corporate equity derivatives financial banking",
    ]
    assert compute_pairwise_top5_coherence(disjoint_docs) == 0.0

    # Single document -> Coherence 0.0
    assert compute_pairwise_top5_coherence(["Only one document"]) == 0.0


def test_shannon_entropy():
    """Verify Shannon entropy calculation on distances."""
    dists = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    entropy = compute_shannon_entropy(dists, n_bins=5)
    assert entropy >= 0.0
    assert compute_shannon_entropy([0.5]) == 0.0


def test_elbow_curvature_rank():
    """Verify elbow detection on a sharp bend curve."""
    # Curve starts low, bends sharply at rank 3, then plateaus
    curve = [0.1, 0.15, 0.70, 0.72, 0.74, 0.75, 0.76]
    elbow = compute_elbow_rank(curve)
    assert elbow in [2, 3], f"Expected elbow near bend (rank 2 or 3), got {elbow}"
    assert compute_elbow_rank([0.1, 0.2]) == 1


def test_analyzer_query_breakdown():
    """Verify end-to-end query statistical analysis with diagnostic insights."""
    connector = ChromaConnector(CHROMA_PERSIST_DIR)
    analyzer = StatisticalAnalyzer(connector)

    mock_records = [
        {"rank": i, "doc_id": f"doc_{i}", "document": f"Sample content {i}", "distance": 0.4 + i*0.005, "similarity_score": 0.6 - i*0.005}
        for i in range(1, 21)
    ]

    res = analyzer.analyze_query_retrieval(
        retrieval_records=mock_records,
        expected_answer="Sample expected answer",
        generated_answer="Sample generated answer",
        target_doc_id="doc_8"  # Gold target at rank 8 (Demotion)
    )

    assert res["num_retrieved"] == 20
    assert "top5_stats" in res
    assert "deep_stats" in res
    assert "mean_distance_shift" in res
    assert res["target_rank"] == 8
    assert res["target_in_deep_fetch"] is True

    # Check diagnostic insights
    insights = res["diagnostic_insights"]
    assert len(insights) > 0
    assert any("Ranking Demotion Pattern" in ins["category"] for ins in insights)


def test_aggregate_dataset_statistics():
    """Verify aggregate statistics on the 5,000 detailed retrieval records."""
    assert Path(DETAILED_RETRIEVAL_CSV).exists()
    connector = ChromaConnector(CHROMA_PERSIST_DIR)
    analyzer = StatisticalAnalyzer(connector)

    agg = analyzer.compute_aggregate_dataset_statistics(DETAILED_RETRIEVAL_CSV)
    assert agg["total_queries_analyzed"] == 50
    assert agg["total_records_processed"] == 5000
    assert 0.0 < agg["top5_mean_distance"] < 1.0
    assert 0.0 < agg["deep_mean_distance"] < 1.0
    assert agg["global_distance_shift"] > 0.0  # Deep fetch distances must be higher than top 5
    assert len(agg["category_statistics"]) == 5


def test_ensure_or_regenerate_audit_dataset(tmp_path):
    """Verify Task 2.6 CSV recovery mechanism generates and saves missing audit dataset."""
    from config.config import SYNTHETIC_DATA_DIR
    sample_csv = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
    assert sample_csv.exists()

    connector = ChromaConnector(CHROMA_PERSIST_DIR)
    temp_out_csv = tmp_path / "recovered_detailed_retrieval.csv"

    assert not temp_out_csv.exists()

    # Run recovery
    det_df, was_regenerated = connector.ensure_or_regenerate_audit_dataset(
        queries_csv=sample_csv,
        output_path=temp_out_csv,
        top_k=10  # small k for rapid test execution
    )

    assert was_regenerated is True
    assert temp_out_csv.exists()
    assert len(det_df) == 50 * 10
    assert "distance" in det_df.columns
    assert "is_top_5" in det_df.columns

    # Calling again without force should load existing without regenerating
    det_df2, was_regenerated2 = connector.ensure_or_regenerate_audit_dataset(
        queries_csv=sample_csv,
        output_path=temp_out_csv,
        force_regenerate=False
    )
    assert was_regenerated2 is False
    assert len(det_df2) == len(det_df)


def test_l2_euclidean_unit_norm_identity():
    """Verify Euclidean L2 equivalent calculation under unit-norm assumption."""
    # Cosine distance = 0.0 -> L2 = 0.0
    m0 = compute_distribution_metrics([0.0], prefix="t0")
    assert pytest.approx(m0["t0_l2_mean_equiv"], 1e-5) == 0.0

    # Cosine distance = 0.5 -> L2 = sqrt(2 * 0.5) = 1.0
    m1 = compute_distribution_metrics([0.5], prefix="t1")
    assert pytest.approx(m1["t1_l2_mean_equiv"], 1e-5) == 1.0

    # Cosine distance = 2.0 (diametrically opposite unit vectors) -> L2 = sqrt(2 * 2.0) = 2.0
    m2 = compute_distribution_metrics([2.0], prefix="t2")
    assert pytest.approx(m2["t2_l2_mean_equiv"], 1e-5) == 2.0


def test_edge_case_fewer_than_five_docs():
    """Verify graceful handling when retrieved document count is less than Top-5."""
    connector = ChromaConnector(CHROMA_PERSIST_DIR)
    analyzer = StatisticalAnalyzer(connector)

    mock_records = [
        {"rank": 1, "doc_id": "d1", "document": "Alpha content", "distance": 0.25, "similarity_score": 0.75},
        {"rank": 2, "doc_id": "d2", "document": "Beta content", "distance": 0.35, "similarity_score": 0.65},
    ]

    res = analyzer.analyze_query_retrieval(mock_records, target_doc_id="d1")
    assert res["num_retrieved"] == 2
    assert len(res["top_5_records"]) == 2
    assert len(res["deep_records"]) == 0
    assert np.isnan(res["deep_stats"]["deep_mean"])
    assert np.isnan(res["mean_distance_shift"])
    assert res["target_in_top5"] is True

