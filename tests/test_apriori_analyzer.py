"""
Unit tests for Apriori Association Rule Mining Engine (Phase 5).
Verifies feature discretization, constant column handling, one-hot transaction encoding,
frequent itemset mining, association rule calculations (support, confidence, lift, leverage, conviction),
category-specific rule extraction across all 5 failure modes, edge cases, and invalid parameters.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import PROCESSED_FEATURES_CSV, FAILURE_CATEGORIES
from modules.apriori_analyzer import (
    discretize_features,
    encode_transactions,
    mine_frequent_itemsets,
    generate_diagnostic_rules,
    filter_rules_by_category,
    explain_rule,
    AprioriDiagnosticEngine,
)


def test_discretize_features_correctness():
    """Verify that continuous features are partitioned into Low, Medium, High ordinal bins."""
    assert PROCESSED_FEATURES_CSV.exists()
    df = pd.read_csv(PROCESSED_FEATURES_CSV)

    binned_df, dropped_cols, edges = discretize_features(
        df, bins=3, strategy="uniform", target_col="failure_label"
    )

    assert not binned_df.empty
    assert len(binned_df) == 50
    assert "failure_label" in binned_df.columns

    # Verify identifier columns were ignored
    for col in ["query_id", "query", "provenance"]:
        assert col not in binned_df.columns

    # Verify constant columns were dropped
    assert len(dropped_cols) >= 4
    for dc in dropped_cols:
        assert dc not in binned_df.columns

    # Verify discrete values are within allowed labels
    for c in binned_df.columns:
        if c != "failure_label":
            unique_vals = set(binned_df[c].unique())
            assert unique_vals.issubset({"Low", "Medium", "High"})


def test_discretize_handles_missing_and_constant_columns():
    """Verify safe handling of synthetic DataFrames with NaNs and zero-variance constant columns."""
    df_synthetic = pd.DataFrame({
        "query_id": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "metric_var": [0.1, np.nan, 0.5, 0.9, 0.2],
        "metric_constant": [5.0, 5.0, 5.0, 5.0, 5.0],
        "failure_label": ["Retrieval Miss", "Ranking Error", "Contradiction", "Hallucination", "Context Contamination"]
    })

    binned, dropped, _ = discretize_features(df_synthetic, bins=3, strategy="uniform")

    assert "metric_constant" in dropped
    assert "metric_constant" not in binned.columns
    assert "metric_var" in binned.columns
    assert not binned["metric_var"].isnull().any()  # NaN must be imputed


def test_encode_transactions_boolean_matrix():
    """Verify one-hot transaction encoding produces valid boolean DataFrame."""
    df = pd.read_csv(PROCESSED_FEATURES_CSV)
    binned_df, _, _ = discretize_features(df, bins=3, target_col="failure_label")

    one_hot = encode_transactions(binned_df, target_col="failure_label")

    assert not one_hot.empty
    assert one_hot.shape[0] == 50
    # Must be boolean dtype
    assert all(one_hot[c].dtype == bool for c in one_hot.columns)

    # Must contain encoded failure category columns
    cat_cols = [c for c in one_hot.columns if c.startswith("failure_category=")]
    assert len(cat_cols) == 5


def test_mine_frequent_itemsets_properties():
    """Verify frequent itemset discovery satisfies support bounds and itemset length constraints."""
    df = pd.read_csv(PROCESSED_FEATURES_CSV)
    binned_df, _, _ = discretize_features(df, bins=3, target_col="failure_label")
    one_hot = encode_transactions(binned_df, target_col="failure_label")

    min_supp = 0.15
    max_l = 3
    freq = mine_frequent_itemsets(one_hot, min_support=min_supp, max_len=max_l)

    assert not freq.empty
    assert "support" in freq.columns
    assert "itemsets" in freq.columns
    assert "length" in freq.columns

    # Verify support constraint
    assert (freq["support"] >= min_supp).all()
    assert (freq["support"] <= 1.0).all()
    assert (freq["length"] <= max_l).all()


def test_generate_diagnostic_rules_metrics():
    """Verify calculation of Support, Confidence, Lift, Leverage, and Conviction."""
    df = pd.read_csv(PROCESSED_FEATURES_CSV)
    binned_df, _, _ = discretize_features(df, bins=3, target_col="failure_label")
    one_hot = encode_transactions(binned_df, target_col="failure_label")

    freq = mine_frequent_itemsets(one_hot, min_support=0.10, max_len=3)
    rules = generate_diagnostic_rules(freq, metric="confidence", min_threshold=0.50, min_lift=1.10)

    assert not rules.empty
    for metric_col in ["support", "confidence", "lift", "leverage", "conviction"]:
        assert metric_col in rules.columns

    # Assert metric boundaries
    assert (rules["confidence"] >= 0.50).all()
    assert (rules["confidence"] <= 1.0).all()
    assert (rules["lift"] >= 1.10).all()
    assert (rules["support"] >= 0.10).all()

    # Leverage in [-1, 1]
    assert (rules["leverage"] >= -1.0).all()
    assert (rules["leverage"] <= 1.0).all()


def test_category_specific_rules_all_five_categories():
    """Verify that diagnostic rules are discovered across all 5 failure categories."""
    engine = AprioriDiagnosticEngine(
        features_csv=PROCESSED_FEATURES_CSV,
        min_support=0.08,
        min_confidence=0.50,
        min_lift=1.10,
        max_len=3,
    )
    engine.fit()

    summary = engine.get_summary_statistics()
    cat_counts = summary["category_rule_counts"]

    # All 5 failure categories must have discovered rules
    for cat in FAILURE_CATEGORIES:
        assert cat in cat_counts
        assert cat_counts[cat] > 0, f"Expected rules for category '{cat}', but found {cat_counts[cat]}"

    # Verify top rule exists for each
    top_rules = summary["top_rule_per_category"]
    for cat in FAILURE_CATEGORIES:
        assert cat in top_rules
        assert len(top_rules[cat]) > 0


def test_apriori_invalid_parameters():
    """Verify that invalid support and empty inputs raise appropriate exceptions."""
    one_hot_dummy = pd.DataFrame({"item_A": [True, False], "item_B": [True, True]})

    with pytest.raises(ValueError, match="min_support"):
        mine_frequent_itemsets(one_hot_dummy, min_support=0.0)

    with pytest.raises(ValueError, match="min_support"):
        mine_frequent_itemsets(one_hot_dummy, min_support=1.5)

    with pytest.raises(ValueError, match="empty"):
        discretize_features(pd.DataFrame())


def test_apriori_empty_and_insufficient_datasets():
    """Verify graceful handling when dataset is too small to mine frequent itemsets."""
    tiny_one_hot = pd.DataFrame({"A": [True]})
    freq = mine_frequent_itemsets(tiny_one_hot, min_support=0.5)
    assert freq.empty

    rules = generate_diagnostic_rules(freq)
    assert rules.empty


def test_explain_rule_formatting():
    """Verify human-readable diagnostic explanation formatting."""
    mock_rule = pd.Series({
        "antecedents_str": "top5_dist_min=High, all_dist_spread=Low",
        "consequents_str": "failure_category=Retrieval Miss",
        "target_category": "Retrieval Miss",
        "support": 0.16,
        "confidence": 1.00,
        "lift": 5.00,
    })

    explanation = explain_rule(mock_rule)
    assert "top5_dist_min=High" in explanation
    assert "Retrieval Miss" in explanation
    assert "100.0% confidence" in explanation
    assert "5.00x" in explanation


def test_apriori_engine_export_methods():
    """Verify AprioriDiagnosticEngine export methods produce clean DataFrames."""
    engine = AprioriDiagnosticEngine(
        features_csv=PROCESSED_FEATURES_CSV,
        min_support=0.10,
        min_confidence=0.60,
        max_len=3,
    ).fit()

    df_export = engine.export_rules_table(category="Retrieval Miss")
    assert isinstance(df_export, pd.DataFrame)
    assert not df_export.empty
    assert "Antecedent Pattern (Features)" in df_export.columns
    assert "Confidence" in df_export.columns
    assert "Lift" in df_export.columns
    assert (df_export["Failure Category"] == "Retrieval Miss").all()
