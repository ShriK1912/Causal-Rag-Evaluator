"""
Unit tests for User's Supervised Diagnostic Model (Multinomial Logistic Regression).
Tests data preparation, stratified splitting, scaling experiment, feature selection,
model pipeline training, evaluation metrics, and Joblib artifact persistence.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import PROCESSED_FEATURES_CSV, BEST_MODEL_PATH
from modules.feature_selection import FilterFeatureSelector, EmbeddedL1FeatureSelector
from modules.diagnostic_models import (
    LogisticRegressionDiagnosticEngine,
    LinearSVMPlaceholder,
    RBFSVMPlaceholder,
    DecisionTreePlaceholder,
    RandomForestPlaceholder,
    KNNPlaceholder,
)


@pytest.fixture(scope="module")
def engine():
    """Initializes and prepares data for LogisticRegressionDiagnosticEngine."""
    eng = LogisticRegressionDiagnosticEngine(random_state=42, test_size=0.25, cv_folds=5)
    eng.prepare_data(PROCESSED_FEATURES_CSV)
    return eng


def test_stratified_split_balance(engine):
    """Verify that stratified train-test split preserves all 5 failure categories."""
    assert engine.X_train is not None
    assert engine.X_test is not None
    assert engine.y_train is not None
    assert engine.y_test is not None

    assert len(engine.X_train) == 37
    assert len(engine.X_test) == 13
    assert engine.y_train.nunique() == 5
    assert engine.y_test.nunique() == 5


def test_scaling_comparison(engine):
    """Verify that scaling comparison produces valid cross-validation metrics."""
    res = engine.run_scaling_experiment()
    assert "StandardScaler" in res
    assert "MinMaxScaler" in res
    assert 0.0 <= res["StandardScaler"]["mean_accuracy"] <= 1.0
    assert 0.0 <= res["MinMaxScaler"]["mean_accuracy"] <= 1.0


def test_filter_feature_selector(engine):
    """Verify ANOVA F-statistic filter selection."""
    selector = FilterFeatureSelector(k=10)
    X_sub = selector.fit_transform(engine.X_train, engine.y_train)
    assert X_sub.shape[1] == 10
    summary = selector.get_ranked_feature_summary()
    assert len(summary) == engine.X_train.shape[1]
    assert summary["f_statistic"].iloc[0] >= summary["f_statistic"].iloc[-1]


def test_embedded_l1_feature_selector(engine):
    """Verify L1 sparse feature selection."""
    selector = EmbeddedL1FeatureSelector(C=1.0, random_state=42)
    X_sub = selector.fit_transform(engine.X_train, engine.y_train)
    assert X_sub.shape[1] > 0
    assert len(selector.selected_features_) <= engine.X_train.shape[1]


def test_pipeline_fit_and_prediction(engine):
    """Verify that the final pipeline fits, predicts, and emits valid probabilities."""
    pipeline = engine.train_best_model(
        scaler_type="StandardScaler",
        feature_subset=engine.feature_names_[:15],
        C_param=1.0,
        penalty="l2"
    )
    assert pipeline is not None

    probs = engine.predict_proba(engine.X_test)
    assert probs.shape == (13, 5)
    # Check probabilities sum to 1.0
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    preds = engine.predict(engine.X_test)
    assert len(preds) == 13
    assert all(p in engine.classes_ for p in preds)


def test_model_evaluation_metrics(engine):
    """Verify calculation of test metrics and confusion matrix."""
    metrics = engine.evaluate_on_test_set()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["precision_macro"] <= 1.0
    assert 0.0 <= metrics["recall_macro"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 5
    assert len(metrics["confusion_matrix"][0]) == 5


def test_joblib_serialization_and_reload(engine, tmp_path):
    """Verify serialization to disk and deserialization."""
    test_path = tmp_path / "test_diagnostic_model.pkl"
    saved_path = engine.save_model(test_path)
    assert Path(saved_path).exists()

    artifact = joblib.load(saved_path)
    assert "pipeline" in artifact
    assert "best_config" in artifact
    assert "test_metrics" in artifact

    # Test prediction with reloaded pipeline
    reloaded_pipeline = artifact["pipeline"]
    feats = artifact["feature_names"]
    reloaded_preds = reloaded_pipeline.predict(engine.X_test[feats])
    assert len(reloaded_preds) == 13


def test_teammate_placeholders():
    """Verify that teammate placeholders raise NotImplementedError cleanly."""
    df_dummy = pd.DataFrame({"x": [1, 2]})
    s_dummy = pd.Series(["A", "B"])
    with pytest.raises(NotImplementedError): LinearSVMPlaceholder().fit(df_dummy, s_dummy)
    with pytest.raises(NotImplementedError): RBFSVMPlaceholder().fit(df_dummy, s_dummy)
    with pytest.raises(NotImplementedError): DecisionTreePlaceholder().fit(df_dummy, s_dummy)
    with pytest.raises(NotImplementedError): RandomForestPlaceholder().fit(df_dummy, s_dummy)
    with pytest.raises(NotImplementedError): KNNPlaceholder().fit(df_dummy, s_dummy)
