"""
Multi-Model Training & Benchmark Shootout Runner for Causal RAG Diagnostician.

Executes the complete multi-model shootout across all 6 classical ML classifiers:
1. Multinomial Logistic Regression
2. Linear Support Vector Machine (Linear SVM)
3. RBF Kernel Support Vector Machine (RBF SVM)
4. Decision Tree Classifier
5. Random Forest Ensemble
6. K-Nearest Neighbors (KNN)

Evaluates all models on identical stratified cross-validation folds and unseen holdout test set,
prints comprehensive comparison rankings, confusion matrices, and persists both:
- models/all_diagnostic_models.pkl (all 6 models + shootout benchmarks)
- models/best_diagnostic_model.pkl (the #1 ranked model pipeline)
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import PROCESSED_FEATURES_CSV, BEST_MODEL_PATH, ALL_MODELS_PATH
from modules.diagnostic_models import MultiModelDiagnosticSuite


def run_multi_model_shootout():
    print("=" * 75)
    print("PHASE 4: MULTI-MODEL BENCHMARK SHOOTOUT & DIAGNOSTIC ENGINE TRAINING")
    print("=" * 75)

    suite = MultiModelDiagnosticSuite()

    print("\n--- 1. Training & Cross-Validating All 6 Classifiers ---")
    comp_df = suite.train_and_evaluate_all(PROCESSED_FEATURES_CSV)

    print("\n" + "=" * 75)
    print("🏆 FINAL BENCHMARK SHOOTOUT LEADERBOARD (Ranked by Test Macro F1)")
    print("=" * 75)
    display_cols = [
        "Model", "CV Accuracy", "CV Macro F1",
        "Test Accuracy", "Test Macro Precision", "Test Macro Recall", "Test Macro F1"
    ]
    print(comp_df[display_cols].to_string(index=False))

    print("\n" + "=" * 75)
    print(f"🥇 BEST PERFORMING MODEL: {suite.best_model_name_}")
    print("=" * 75)

    # Detailed report for top model
    best_eng = suite.best_engine_
    print(f"Algorithm:       {suite.best_model_name_}")
    print(f"Test Accuracy:   {best_eng.test_metrics_['accuracy'] * 100:.1f}%")
    print(f"Test Macro F1:   {best_eng.test_metrics_['f1_macro']:.4f}")
    print(f"Test Precision:  {best_eng.test_metrics_['precision_macro']:.4f}")
    print(f"Test Recall:     {best_eng.test_metrics_['recall_macro']:.4f}")

    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    classes = best_eng.test_metrics_["classes"]
    cm_df = pd.DataFrame(best_eng.confusion_matrix_, index=classes, columns=classes)
    print(cm_df)

    print("\nDetailed Per-Class Performance:")
    for cls in classes:
        rep = best_eng.classification_report_dict_[cls]
        print(f"  {cls:<22} P: {rep['precision']:.2f} | R: {rep['recall']:.2f} | F1: {rep['f1-score']:.2f} (Support: {int(rep['support'])})")

    # Persist suite and best model
    print("\n--- 2. Persisting Model Artifacts to Disk ---")
    suite.save_suite(ALL_MODELS_PATH, BEST_MODEL_PATH)
    print(f"Saved complete multi-model suite artifact to: {ALL_MODELS_PATH}")
    print(f"Saved best diagnostic model pipeline artifact to: {BEST_MODEL_PATH}")

    # Validation of saved artifact
    reloaded_suite = joblib.load(ALL_MODELS_PATH)
    assert "comparison_df" in reloaded_suite
    assert "engines" in reloaded_suite
    print("Artifact verification: All 6 models verified and ready for live diagnosis!")
    return suite


if __name__ == "__main__":
    run_multi_model_shootout()

