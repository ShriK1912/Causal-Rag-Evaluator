"""
Unit tests for Data Ingestion and Validation Module (Task 1.1).
Tests schema validation, missing columns, missing values, duplicates,
unrecognized categories, and synthetic dataset loading.
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

from config.config import SYNTHETIC_DATA_DIR, FAILURE_CATEGORIES
from modules.data_ingestion import (
    load_csv,
    validate_rag_dataset,
    standardize_column_names,
    DataQualityReport,
)


def test_synthetic_dataset_ingestion():
    """Verify that the baseline synthetic dataset ingests with 100% validity."""
    csv_path = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
    assert csv_path.exists(), "Sample failed queries CSV must exist."

    report = load_csv(csv_path)
    assert report.is_valid_dataset is True
    assert report.total_records_ingested == 50
    assert report.valid_records_count == 50
    assert report.invalid_records_count == 0
    assert report.duplicate_records_count == 0
    assert report.has_labels is True

    # Check all 5 categories have exactly 10 samples
    for cat in FAILURE_CATEGORIES:
        assert report.failure_category_distribution.get(cat, 0) == 10

    # Check provenance preserved
    assert (report.cleaned_df["provenance"] == "synthetic_controlled_v1").all()


def test_missing_mandatory_query_column():
    """Verify dataset rejection when mandatory 'query' column is absent."""
    df_missing_query = pd.DataFrame({
        "expected_answer": ["Ground truth 1", "Ground truth 2"],
        "generated_answer": ["Output 1", "Output 2"]
    })

    report = validate_rag_dataset(df_missing_query)
    assert report.is_valid_dataset is False
    assert any("Mandatory column 'query' is missing" in err for err in report.validation_errors)


def test_column_aliasing():
    """Verify that common column aliases (e.g. prompt, reference_answer) are standardized."""
    df_aliased = pd.DataFrame({
        "prompt": ["What is PCA?"],
        "reference_answer": ["PCA is dimensionality reduction."],
        "rag_response": ["PCA reduces dimensions."],
        "category": ["Hallucination"],
        "id": ["Q_CUSTOM_1"]
    })

    report = validate_rag_dataset(df_aliased)
    assert report.is_valid_dataset is True
    assert "query" in report.cleaned_df.columns
    assert "expected_answer" in report.cleaned_df.columns
    assert "generated_answer" in report.cleaned_df.columns
    assert "failure_label" in report.cleaned_df.columns
    assert "query_id" in report.cleaned_df.columns
    assert report.cleaned_df.iloc[0]["query"] == "What is PCA?"
    assert report.cleaned_df.iloc[0]["failure_label"] == "Hallucination"


def test_empty_and_whitespace_query_handling():
    """Verify that empty, NaN, and whitespace-only queries are filtered to invalid_df."""
    df_with_empties = pd.DataFrame({
        "query": ["Valid query 1", "", "   ", np.nan, "Valid query 2"],
        "expected_answer": ["Ans 1", "Ans 2", "Ans 3", "Ans 4", "Ans 5"],
        "generated_answer": ["Gen 1", "Gen 2", "Gen 3", "Gen 4", "Gen 5"]
    })

    report = validate_rag_dataset(df_with_empties)
    assert report.total_records_ingested == 5
    assert report.valid_records_count == 2
    assert report.invalid_records_count == 3
    assert len(report.invalid_df) == 3
    assert all("Empty or null query" in r for r in report.invalid_df["rejection_reason"])


def test_duplicate_query_removal():
    """Verify that duplicate queries are removed and recorded in the quality report."""
    df_with_duplicates = pd.DataFrame({
        "query": ["Repeated Query", "Unique Query", "Repeated Query"],
        "expected_answer": ["Ans A", "Ans B", "Ans C"],
        "generated_answer": ["Gen A", "Gen B", "Gen C"]
    })

    report = validate_rag_dataset(df_with_duplicates, drop_duplicates=True)
    assert report.total_records_ingested == 3
    assert report.valid_records_count == 2
    assert report.duplicate_records_count == 1
    assert len(report.cleaned_df) == 2
    assert "Unique Query" in report.cleaned_df["query"].values


def test_invalid_failure_category():
    """Verify that invalid failure categories are flagged and segregated."""
    df_invalid_category = pd.DataFrame({
        "query": ["What is SVM?", "What is DBSCAN?"],
        "generated_answer": ["Ans 1", "Ans 2"],
        "failure_label": ["Retrieval Miss", "Completely Fake Glitch Category"]
    })

    report = validate_rag_dataset(df_invalid_category)
    assert report.valid_records_count == 1
    assert report.invalid_records_count == 1
    assert report.cleaned_df.iloc[0]["failure_label"] == "Retrieval Miss"
    assert "Invalid failure_label" in report.invalid_df.iloc[0]["rejection_reason"]


def test_unlabeled_mode():
    """Verify that missing failure_label column allows unlabeled inference mode."""
    df_unlabeled = pd.DataFrame({
        "query": ["How does gradient descent work?"],
        "generated_answer": ["It optimizes weights."]
    })

    report = validate_rag_dataset(df_unlabeled)
    assert report.is_valid_dataset is True
    assert report.has_labels is False
    assert report.valid_records_count == 1
    assert "failure_label" not in report.cleaned_df.columns
