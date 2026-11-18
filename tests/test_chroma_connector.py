"""
Unit tests for ChromaDB Local Connector and Retrieval Pipeline (Task 1.2).
Tests database connection, collection inspection, Top-100 deep-fetch retrieval,
rank order verification, Top-5 vs ranks 6-100 partitioning, and batch retrieval.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    SYNTHETIC_DATA_DIR,
    TOP_K_BASELINE,
    TOP_K_DEEP_FETCH,
)
from modules.chroma_connector import ChromaConnector


@pytest.fixture(scope="module")
def connector():
    """Initializes a shared ChromaConnector instance."""
    return ChromaConnector(persist_dir=CHROMA_PERSIST_DIR)


def test_chroma_connection(connector):
    """Verify persistent client connects and detects academic_rag_corpus."""
    collections = connector.list_collections()
    assert DEFAULT_COLLECTION_NAME in collections, f"'{DEFAULT_COLLECTION_NAME}' must exist."


def test_collection_metadata(connector):
    """Verify collection statistics, count (157 docs), and cosine distance metric."""
    info = connector.get_collection_info(DEFAULT_COLLECTION_NAME)
    assert info["name"] == DEFAULT_COLLECTION_NAME
    assert info["count"] >= 120, "Corpus should contain at least 120 documents for Top-100 retrieval."
    assert info["distance_metric"] == "cosine", "Collection should be configured with cosine distance."


def test_single_query_top_100_retrieval(connector):
    """
    Verify Top-100 retrieval returns strictly ranked, monotonically increasing distances,
    and valid similarity scores.
    """
    records = connector.retrieve_query(
        collection_name=DEFAULT_COLLECTION_NAME,
        query_text="What is supervised learning in machine learning?",
        top_k=TOP_K_DEEP_FETCH
    )

    assert len(records) == TOP_K_DEEP_FETCH, f"Must return exactly {TOP_K_DEEP_FETCH} documents."
    
    # Check sequential ranks
    ranks = [r["rank"] for r in records]
    assert ranks == list(range(1, TOP_K_DEEP_FETCH + 1))

    # Check monotonicity of distance (ascending order)
    distances = [r["distance"] for r in records]
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1] + 1e-6, "Retrieval distances must be sorted in ascending order."

    # Check similarity scores in [0, 1]
    similarities = [r["similarity_score"] for r in records]
    for s in similarities:
        assert 0.0 <= s <= 1.0, "Cosine similarity score must be bounded between 0 and 1."


def test_top_5_vs_deep_fetch_partitioning(connector):
    """Verify clean partition between Top-5 baseline and ranks 6-100."""
    records = connector.retrieve_query(
        collection_name=DEFAULT_COLLECTION_NAME,
        query_text="How does PCA transform correlated features?",
        top_k=TOP_K_DEEP_FETCH
    )

    top_5, ranks_6_100, all_res = connector.partition_retrieval_results(records, baseline_k=TOP_K_BASELINE)
    assert len(top_5) == 5
    assert len(ranks_6_100) == TOP_K_DEEP_FETCH - 5
    assert len(all_res) == TOP_K_DEEP_FETCH
    assert all(r["is_top_5"] is True for r in top_5)
    assert all(r["is_top_5"] is False for r in ranks_6_100)


def test_batch_retrieval_for_dataset(connector):
    """Verify batch retrieval across queries and target document rank detection."""
    sample_csv = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
    assert sample_csv.exists()
    df = pd.read_csv(sample_csv).head(5)  # Test with 5 queries for fast unit test

    detailed_df, query_map = connector.retrieve_for_dataset(
        df_queries=df,
        collection_name=DEFAULT_COLLECTION_NAME,
        top_k=TOP_K_DEEP_FETCH
    )

    assert len(detailed_df) == 5 * TOP_K_DEEP_FETCH
    assert len(query_map) == 5

    # Check columns
    required_cols = [
        "query_id", "query", "expected_answer", "generated_answer",
        "failure_label", "rank", "doc_id", "document", "distance",
        "distance_metric", "similarity_score", "is_top_5", "is_target_doc"
    ]
    for col in required_cols:
        assert col in detailed_df.columns, f"Column '{col}' must be present in detailed results."


def test_invalid_collection_error(connector):
    """Verify that requesting a non-existent collection raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        connector.get_collection_info("non_existent_collection_xyz")
    assert "not found" in str(excinfo.value)
