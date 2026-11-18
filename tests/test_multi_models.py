"""
Unit tests for All 6 Supervised Diagnostic Models & Multi-Model Shootout Suite.
Verifies fitting, probability calibration, metrics calculation, and artifact persistence for:
1. Multinomial Logistic Regression
2. Linear Support Vector Machine
3. RBF Kernel Support Vector Machine
4. Decision Tree Classifier
5. Random Forest Ensemble
6. K-Nearest Neighbors (KNN)
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

from config.config import PROCESSED_FEATURES_CSV
from modules.diagnostic_models import (
    LogisticRegressionDiagnosticEngine,
    LinearSVMDiagnosticEngine,
    RBFSVMDiagnosticEngine,
    DecisionTreeDiagnosticEngine,
    RandomForestDiagnosticEngine,
    KNNDiagnosticEngine,
    MultiModelDiagnosticSuite,
)


@pytest.fixture(scope="module")
def prepared_features():
    df = pd.read_csv(PROCESSED_FEATURES_CSV)
    return df


def test_linear_svm_engine(prepared_features):
    engine = LinearSVMDiagnosticEngine(random_state=42, test_size=0.25)
    engine.prepare_data(PROCESSED_FEATURES_CSV)
    pipe = engine.train_model(feature_subset=engine.feature_names_[:15])
    assert pipe is not None
    assert engine.coefficients_df_ is not None
    assert engine.coefficients_df_.shape == (5, 15)

    probs = engine.predict_proba(engine.X_test)
    assert probs.shape == (len(engine.X_test), 5)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-4)

    metrics = engine.evaluate_on_test_set()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_rbf_svm_engine(prepared_features):
    engine = RBFSVMDiagnosticEngine(random_state=42, test_size=0.25)
    engine.prepare_data(PROCESSED_FEATURES_CSV)
    pipe = engine.train_model(feature_subset=engine.feature_names_[:15])
    assert pipe is not None
    assert engine.support_vectors_count_ > 0

    probs = engine.predict_proba(engine.X_test)
    assert probs.shape == (len(engine.X_test), 5)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-4)

    metrics = engine.evaluate_on_test_set()
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_decision_tree_engine(prepared_features):
    engine = DecisionTreeDiagnosticEngine(max_depth=5, random_state=42, test_size=0.25)
    engine.prepare_data(PROCESSED_FEATURES_CSV)
    pipe = engine.train_model(feature_subset=engine.feature_names_[:15])
    assert pipe is not None
    assert engine.feature_importances_df_ is not None
    assert len(engine.feature_importances_df_) == 15

    probs = engine.predict_proba(engine.X_test)
    assert probs.shape == (len(engine.X_test), 5)

    metrics = engine.evaluate_on_test_set()
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_random_forest_engine(prepared_features):
    engine = RandomForestDiagnosticEngine(n_estimators=50, max_depth=5, random_state=42, test_size=0.25)
    engine.prepare_data(PROCESSED_FEATURES_CSV)
    pipe = engine.train_model(feature_subset=engine.feature_names_[:15])
    assert pipe is not None
    assert engine.feature_importances_df_ is not None

    probs = engine.predict_proba(engine.X_test)
    assert probs.shape == (len(engine.X_test), 5)

    metrics = engine.evaluate_on_test_set()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["f1_macro"] >= 0.5


def test_knn_engine(prepared_features):
    engine = KNNDiagnosticEngine(n_neighbors=5, random_state=42, test_size=0.25)
    engine.prepare_data(PROCESSED_FEATURES_CSV)
    pipe = engine.train_model(feature_subset=engine.feature_names_[:15])
    assert pipe is not None

    probs = engine.predict_proba(engine.X_test)
    assert probs.shape == (len(engine.X_test), 5)

    metrics = engine.evaluate_on_test_set()
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_multi_model_shootout_suite(tmp_path):
    suite = MultiModelDiagnosticSuite(random_state=42, test_size=0.25)
    comp_df = suite.train_and_evaluate_all(PROCESSED_FEATURES_CSV)

    assert len(comp_df) == 6
    assert suite.best_model_name_ in comp_df["Model"].values
    assert suite.best_engine_ is not None

    # Test serialization
    test_suite_path = tmp_path / "test_all_models.pkl"
    test_best_path = tmp_path / "test_best_model.pkl"
    suite.save_suite(test_suite_path, test_best_path)

    assert test_suite_path.exists()
    assert test_best_path.exists()

    loaded_suite = joblib.load(test_suite_path)
    assert "comparison_df" in loaded_suite
    assert "engines" in loaded_suite
    assert len(loaded_suite["engines"]) == 6

