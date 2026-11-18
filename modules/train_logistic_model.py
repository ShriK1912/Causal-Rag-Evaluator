"""
Standalone training and evaluation runner for User's Logistic Regression Model.
Executes:
1. Data loading and stratified train-test splitting.
2. Experiment 1: StandardScaler vs MinMaxScaler.
3. Experiment 2: Filter-based vs Embedded L1 feature selection.
4. Model training on best configuration.
5. Evaluation on held-out test set.
6. Joblib model serialization to models/best_diagnostic_model.pkl.
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

from config.config import PROCESSED_FEATURES_CSV, BEST_MODEL_PATH
from modules.diagnostic_models import LogisticRegressionDiagnosticEngine


def run_logistic_training_pipeline():
    print("=" * 70)
    print("PHASE 4: USER'S PERSONAL LOGISTIC REGRESSION DIAGNOSTIC ENGINE")
    print("=" * 70)

    engine = LogisticRegressionDiagnosticEngine(random_state=42, test_size=0.25, cv_folds=5)

    # 1. Prepare Data & Split
    print("\n--- 1. Data Ingestion & Stratified Split ---")
    X_train, X_test, y_train, y_test = engine.prepare_data(PROCESSED_FEATURES_CSV)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} numerical features")
    print(f"Test holdout: {X_test.shape[0]} samples (stratified)")
    print("Training class balance:")
    print(y_train.value_counts().to_dict())

    # 2. Experiment 1: Scaling Comparison
    print("\n--- 2. Experiment 1: Feature Scaling Comparison (5-Fold CV) ---")
    scaling_res = engine.run_scaling_experiment()
    for scaler_name, metrics in scaling_res.items():
        print(f"[{scaler_name}] CV Accuracy: {metrics['mean_accuracy']:.4f} (+/- {metrics['std_accuracy']:.4f}) | "
              f"CV Macro F1: {metrics['mean_f1_macro']:.4f} (+/- {metrics['std_f1_macro']:.4f})")

    best_scaler = "StandardScaler" if scaling_res["StandardScaler"]["mean_f1_macro"] >= scaling_res["MinMaxScaler"]["mean_f1_macro"] else "MinMaxScaler"
    print(f">> Selected Scaler for Pipeline: {best_scaler}")

    # 3. Experiment 2: Feature Selection Comparison
    print("\n--- 3. Experiment 2: Feature Selection Comparison (5-Fold CV) ---")
    feat_res = engine.run_feature_selection_experiment(best_scaler_name=best_scaler)
    configs = feat_res["configurations"]
    for config_name, metrics in configs.items():
        print(f"[{config_name}] CV Accuracy: {metrics['mean_accuracy']:.4f} (+/- {metrics['std_accuracy']:.4f}) | "
              f"CV Macro F1: {metrics['mean_f1_macro']:.4f} (+/- {metrics['std_f1_macro']:.4f})")

    # Display Top Filter Features
    print("\nTop 8 Features by ANOVA F-Statistic (Filter Selection):")
    filter_summary = feat_res["filter_ranked_summary"]
    print(filter_summary.head(8)[["feature", "f_statistic", "p_value"]].to_string(index=False))

    # Display Top Embedded L1 Features
    print("\nTop 8 Features by L1 Mean Absolute Coefficient (Embedded Selection):")
    embedded_summary = feat_res["embedded_coefficient_summary"]
    print(embedded_summary.head(8)[["feature", "mean_abs_coefficient", "is_selected"]].to_string(index=False))

    # Determine best feature configuration based on CV Macro F1
    best_config_name = max(configs.keys(), key=lambda k: configs[k]["mean_f1_macro"])
    selected_features = configs[best_config_name]["features_list"]
    print(f"\n>> Selected Feature Configuration: '{best_config_name}' ({len(selected_features)} features)")

    # 4. Train Final Best Model Pipeline
    print("\n--- 4. Training Final Multinomial Logistic Regression Pipeline ---")
    engine.train_best_model(
        scaler_type=best_scaler,
        feature_subset=selected_features,
        C_param=1.0,
        penalty="l2"
    )
    print("Fitted Pipeline Architecture:")
    print(engine.pipeline)

    # 5. Evaluate on Held-out Test Set
    print("\n--- 5. Evaluation on Unseen Test Holdout Set (Strict No-Leakage) ---")
    test_metrics = engine.evaluate_on_test_set()
    print(f"Test Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Test Macro Precision:   {test_metrics['precision_macro']:.4f}")
    print(f"Test Macro Recall:      {test_metrics['recall_macro']:.4f}")
    print(f"Test Macro F1-Score:    {test_metrics['f1_macro']:.4f}")
    print(f"Test Weighted F1-Score: {test_metrics['f1_weighted']:.4f}")
    print(f"Test Multiclass Loss:   {test_metrics['log_loss']:.4f}")

    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    classes = test_metrics["classes"]
    cm_df = pd.DataFrame(engine.confusion_matrix_, index=classes, columns=classes)
    print(cm_df)

    print("\nDetailed Classification Report:")
    for cls in classes:
        cls_rep = engine.classification_report_dict_[cls]
        print(f"  {cls:<22} Precision: {cls_rep['precision']:.2f} | Recall: {cls_rep['recall']:.2f} | F1: {cls_rep['f1-score']:.2f} (Support: {int(cls_rep['support'])})")

    # 6. Coefficient Interpretability
    print("\n--- 6. Feature Coefficient Interpretability Analysis ---")
    coef_df = engine.coefficients_df_
    for cls in classes:
        print(f"\nTop Positive Drivers for [{cls}]:")
        top_pos = coef_df.loc[cls].sort_values(ascending=False).head(3)
        for feat, val in top_pos.items():
            print(f"  + {feat:<30}: {val:+.4f}")
        print(f"Top Negative Indicators for [{cls}]:")
        top_neg = coef_df.loc[cls].sort_values(ascending=True).head(3)
        for feat, val in top_neg.items():
            print(f"  - {feat:<30}: {val:+.4f}")

    # 7. Model Serialization
    print("\n--- 7. Serializing Trained Pipeline to Disk ---")
    saved_path = engine.save_model(BEST_MODEL_PATH)
    print(f"Saved complete diagnostic pipeline artifact to: {saved_path}")

    # Verify reload
    reloaded = joblib.load(saved_path)
    assert "pipeline" in reloaded
    assert reloaded["pipeline"] is not None
    print("Artifact verification: Model reloaded and validated successfully!")
    print("\nPhase 4 Logistic Regression Workflow Completed Successfully!")
    return engine


if __name__ == "__main__":
    run_logistic_training_pipeline()
