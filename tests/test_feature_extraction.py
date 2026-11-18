"""
Unit tests for Feature Extraction Module (Tasks 1.3, 1.4, 1.5).
Tests Jaccard similarity, ROUGE-L, descriptive statistics calculations,
single-query feature extraction, and end-to-end feature pipeline execution.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.feature_extraction import (
    compute_jaccard_similarity,
    compute_rouge_l,
    calculate_distribution_stats,
    extract_features_for_single_query,
    extract_features_pipeline,
    tokenize_text,
)


def test_tokenize_text():
    """Verify lowercasing and punctuation stripping."""
    tokens = tokenize_text("Hello, World! Machine-Learning 2026.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "machine" in tokens
    assert "learning" in tokens
    assert "2026" in tokens
    assert tokenize_text("") == set()
    assert tokenize_text(None) == set()


def test_jaccard_similarity():
    """Verify Jaccard index calculations across edge cases."""
    # Identical
    assert compute_jaccard_similarity("Support Vector Machine", "support vector machine") == 1.0
    # Disjoint
    assert compute_jaccard_similarity("Apple Inc", "Quantum physics") == 0.0
    # Overlap: {hello, world} and {hello, there} -> intersection {hello} (1), union {hello, world, there} (3)
    assert pytest.approx(compute_jaccard_similarity("hello world", "hello there"), 0.01) == 0.333
    # Empty
    assert compute_jaccard_similarity("", "test") == 0.0
    assert compute_jaccard_similarity(None, None) == 0.0


def test_rouge_l():
    """Verify ROUGE-L LCS calculation."""
    # Identical
    assert pytest.approx(compute_rouge_l("Supervised learning algorithms", "Supervised learning algorithms"), 0.01) == 1.0
    # Disjoint
    assert compute_rouge_l("Planetary orbit astronomy", "Eukaryotic cellular biology") == 0.0
    # Empty
    assert compute_rouge_l("", "content") == 0.0
    assert compute_rouge_l(None, "content") == 0.0


def test_distribution_stats():
    """Verify descriptive statistics calculation on known values."""
    dists = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = calculate_distribution_stats(dists, "test")
    assert stats["test_mean"] == 3.0
    assert stats["test_min"] == 1.0
    assert stats["test_max"] == 5.0
    assert stats["test_spread"] == 4.0
    assert pytest.approx(stats["test_var"], 0.01) == 2.0
    assert pytest.approx(stats["test_std"], 0.01) == np.sqrt(2.0)

    # Empty handling
    empty_stats = calculate_distribution_stats([], "empty")
    assert np.isnan(empty_stats["empty_mean"])


def test_single_query_feature_extraction():
    """Verify complete feature extraction dictionary generation for a query."""
    mock_query_data = {
        "query_id": "TEST_001",
        "query": "What is PCA?",
        "expected_answer": "PCA transforms correlated features into principal components.",
        "generated_answer": "PCA reduces feature space dimensions.",
        "failure_label": "Hallucination",
        "provenance": "synthetic_test",
        "target_doc_id": "doc_001",
        "target_rank": 2,
        "top_5": [
            {"rank": 1, "doc_id": "doc_002", "document": "PCA computes linear components.", "distance": 0.40, "similarity_score": 0.60},
            {"rank": 2, "doc_id": "doc_001", "document": "Principal component analysis transforms correlated features.", "distance": 0.45, "similarity_score": 0.55},
            {"rank": 3, "doc_id": "doc_003", "document": "K-Means clusters data into Voronoi cells.", "distance": 0.60, "similarity_score": 0.40},
            {"rank": 4, "doc_id": "doc_004", "document": "DBSCAN clusters spatial density.", "distance": 0.70, "similarity_score": 0.30},
            {"rank": 5, "doc_id": "doc_005", "document": "Linear regression models linear relationships.", "distance": 0.75, "similarity_score": 0.25},
        ],
        "ranks_6_100": [
            {"rank": 6, "doc_id": "doc_006", "document": "Deep fetch doc 6", "distance": 0.80, "similarity_score": 0.20},
            {"rank": 7, "doc_id": "doc_007", "document": "Deep fetch doc 7", "distance": 0.85, "similarity_score": 0.15},
        ],
        "all_results": [
            {"rank": i, "doc_id": f"doc_{i:03d}", "document": f"Text {i}", "distance": 0.4 + i*0.05, "similarity_score": 0.6 - i*0.05}
            for i in range(1, 8)
        ]
    }

    feats = extract_features_for_single_query(mock_query_data)
    assert feats["query_id"] == "TEST_001"
    assert feats["failure_label"] == "Hallucination"
    assert feats["target_in_top5"] == 1
    assert feats["target_in_deep_fetch"] == 0
    assert feats["target_doc_rank"] == 2
    assert "top5_dist_mean" in feats
    assert "deep_dist_mean" in feats
    assert "jaccard_expected_vs_top5" in feats
    assert "rougel_expected_vs_top5" in feats
    assert feats["jaccard_expected_vs_top5"] > 0.0
    assert feats["rougel_expected_vs_top5"] > 0.0


def test_extracted_features_csv_exists():
    """Verify that extracted_features.csv is populated and conforms to schema."""
    from config.config import PROCESSED_FEATURES_CSV
    assert PROCESSED_FEATURES_CSV.exists(), "extracted_features.csv must exist."
    df = pd.read_csv(PROCESSED_FEATURES_CSV)
    assert len(df) == 50, "Must contain 50 rows for the 50 failed queries."
    assert "failure_label" in df.columns
    assert "top5_dist_mean" in df.columns
    assert "jaccard_generated_vs_top5" in df.columns
    assert "rougel_generated_vs_top5" in df.columns
    assert df["failure_label"].nunique() == 5
