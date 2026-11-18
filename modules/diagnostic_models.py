"""
Supervised Machine Learning Diagnostic Models Module.

User Responsibility:
- Model 1: Multinomial Logistic Regression Pipeline
  (Preprocessing, StandardScaler vs MinMaxScaler, Feature Selection,
   Hyperparameter Optimization, Evaluation, and Joblib Serialization).

Teammate Responsibility:
- Models 2 to 6 (Linear SVM, RBF SVM, Decision Tree, Random Forest, KNN)
  are reserved modular interface slots to be integrated independently.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_FEATURES_CSV,
    BEST_MODEL_PATH,
    ALL_MODELS_PATH,
    RANDOM_STATE,
    TEST_SPLIT_SIZE,
    N_CROSS_VALIDATION_FOLDS,
    FAILURE_CATEGORIES,
    MODEL_NAMES,
)
from modules.feature_selection import FilterFeatureSelector, EmbeddedL1FeatureSelector


class BaseDiagnosticModel(ABC):
    """Abstract interface ensuring pluggable model integration across team members."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseDiagnosticModel":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass


# -----------------------------------------------------------------------------
# Teammate Model Placeholders (Reserved Modular Slots)
# -----------------------------------------------------------------------------
class LinearSVMPlaceholder(BaseDiagnosticModel):
    """Reserved slot for teammate's Linear SVM implementation."""
    def fit(self, X, y): raise NotImplementedError("Linear SVM is reserved for teammate implementation.")
    def predict(self, X): raise NotImplementedError("Linear SVM is reserved for teammate implementation.")
    def predict_proba(self, X): raise NotImplementedError("Linear SVM is reserved for teammate implementation.")


class RBFSVMPlaceholder(BaseDiagnosticModel):
    """Reserved slot for teammate's RBF Kernel SVM implementation."""
    def fit(self, X, y): raise NotImplementedError("RBF SVM is reserved for teammate implementation.")
    def predict(self, X): raise NotImplementedError("RBF SVM is reserved for teammate implementation.")
    def predict_proba(self, X): raise NotImplementedError("RBF SVM is reserved for teammate implementation.")


class DecisionTreePlaceholder(BaseDiagnosticModel):
    """Reserved slot for teammate's Decision Tree implementation."""
    def fit(self, X, y): raise NotImplementedError("Decision Tree is reserved for teammate implementation.")
    def predict(self, X): raise NotImplementedError("Decision Tree is reserved for teammate implementation.")
    def predict_proba(self, X): raise NotImplementedError("Decision Tree is reserved for teammate implementation.")


class RandomForestPlaceholder(BaseDiagnosticModel):
    """Reserved slot for teammate's Random Forest implementation."""
    def fit(self, X, y): raise NotImplementedError("Random Forest is reserved for teammate implementation.")
    def predict(self, X): raise NotImplementedError("Random Forest is reserved for teammate implementation.")
    def predict_proba(self, X): raise NotImplementedError("Random Forest is reserved for teammate implementation.")


class KNNPlaceholder(BaseDiagnosticModel):
    """Reserved slot for teammate's K-Nearest Neighbors implementation."""
    def fit(self, X, y): raise NotImplementedError("KNN is reserved for teammate implementation.")
    def predict(self, X): raise NotImplementedError("KNN is reserved for teammate implementation.")
    def predict_proba(self, X): raise NotImplementedError("KNN is reserved for teammate implementation.")


# -----------------------------------------------------------------------------
# User Implementation: Multinomial Logistic Regression Diagnostic Engine
# -----------------------------------------------------------------------------
class LogisticRegressionDiagnosticEngine(BaseDiagnosticModel):
    """
    Multinomial Logistic Regression Diagnostic System.
    Handles data splitting, scaling comparison, feature selection,
    training, multiclass evaluation, interpretability analysis, and serialization.
    """

    METADATA_COLUMNS = ["query_id", "query", "provenance", "failure_label"]

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        test_size: float = TEST_SPLIT_SIZE,
        cv_folds: int = N_CROSS_VALIDATION_FOLDS
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        # Datasets
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names_: List[str] = []

        # Experiment Results
        self.scaling_experiment_results: Dict[str, Dict[str, float]] = {}
        self.feature_selection_results: Dict[str, Any] = {}
        self.tuning_results: List[Dict[str, Any]] = []

        # Final Fitted Pipeline & Artifacts
        self.pipeline: Optional[Pipeline] = None
        self.best_config_: Dict[str, Any] = {}
        self.test_metrics_: Dict[str, Any] = {}
        self.confusion_matrix_: Optional[np.ndarray] = None
        self.classification_report_dict_: Dict[str, Any] = {}
        self.classes_: List[str] = []
        self.coefficients_df_: Optional[pd.DataFrame] = None

    def prepare_data(
        self,
        features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads feature CSV, isolates numerical predictive columns, and applies
        stratified train-test split.
        """
        df = pd.read_csv(features_csv)

        if "failure_label" not in df.columns:
            raise ValueError("Target column 'failure_label' is missing from feature matrix.")

        # Filter rows without valid labels
        valid_df = df[df["failure_label"].isin(FAILURE_CATEGORIES)].copy()
        if len(valid_df) == 0:
            raise ValueError("No records with valid failure categories found.")

        # Separate feature columns from metadata
        feature_cols = [c for c in valid_df.columns if c not in self.METADATA_COLUMNS]
        # Keep only numerical columns
        numeric_cols = valid_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        # Clean infs/nans
        X = valid_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = valid_df["failure_label"]

        self.feature_names_ = numeric_cols

        # Stratified train-test split to preserve exact class ratios
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def run_scaling_experiment(self) -> Dict[str, Dict[str, float]]:
        """
        Experiment 1: Compares StandardScaler vs MinMaxScaler using 5-fold Stratified CV
        on the training set strictly to prevent data leakage.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data must be prepared before running scaling experiment.")

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
        }

        results = {}
        for name, scaler in scalers.items():
            pipe = Pipeline([
                ("scaler", scaler),
                ("clf", LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    random_state=self.random_state,
                    max_iter=1000
                ))
            ])
            cv_res = cross_validate(pipe, self.X_train, self.y_train, cv=cv, scoring=scoring)
            results[name] = {
                "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
                "std_accuracy": float(np.std(cv_res["test_accuracy"])),
                "mean_f1_macro": float(np.mean(cv_res["test_f1_macro"])),
                "std_f1_macro": float(np.std(cv_res["test_f1_macro"])),
                "mean_precision_macro": float(np.mean(cv_res["test_precision_macro"])),
                "mean_recall_macro": float(np.mean(cv_res["test_recall_macro"])),
            }

        self.scaling_experiment_results = results
        return results

    def run_feature_selection_experiment(
        self,
        best_scaler_name: str = "StandardScaler"
    ) -> Dict[str, Any]:
        """
        Experiment 2: Evaluates three feature configurations:
        1. All numerical features (baseline)
        2. Filter-based selection (ANOVA F-statistic top 15 features)
        3. Embedded L1 selection (L1-penalized sparse Logistic Regression)
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data must be prepared first.")

        scaler_cls = StandardScaler if best_scaler_name == "StandardScaler" else MinMaxScaler
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # 1. Fit filter selector on training set only
        filter_sel = FilterFeatureSelector(k=15).fit(self.X_train, self.y_train)
        filter_feats = filter_sel.selected_features_

        # 2. Fit embedded L1 selector on training set only
        embedded_sel = EmbeddedL1FeatureSelector(C=1.0, random_state=self.random_state).fit(self.X_train, self.y_train)
        embedded_feats = embedded_sel.selected_features_

        feature_sets = {
            "All Features (36)": self.feature_names_,
            f"Filter Selected ({len(filter_feats)})": filter_feats,
            f"Embedded L1 Selected ({len(embedded_feats)})": embedded_feats,
        }

        results = {}
        for config_name, feats in feature_sets.items():
            X_sub = self.X_train[feats]
            pipe = Pipeline([
                ("scaler", scaler_cls()),
                ("clf", LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    random_state=self.random_state,
                    max_iter=1000
                ))
            ])
            cv_res = cross_validate(pipe, X_sub, self.y_train, cv=cv, scoring=["accuracy", "f1_macro"])
            results[config_name] = {
                "num_features": len(feats),
                "features_list": feats,
                "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
                "std_accuracy": float(np.std(cv_res["test_accuracy"])),
                "mean_f1_macro": float(np.mean(cv_res["test_f1_macro"])),
                "std_f1_macro": float(np.std(cv_res["test_f1_macro"])),
            }

        self.feature_selection_results = {
            "configurations": results,
            "filter_ranked_summary": filter_sel.get_ranked_feature_summary(),
            "embedded_coefficient_summary": embedded_sel.get_coefficient_summary(),
            "filter_features": filter_feats,
            "embedded_features": embedded_feats,
        }
        return self.feature_selection_results

    def train_best_model(
        self,
        scaler_type: str = "StandardScaler",
        feature_subset: Optional[List[str]] = None,
        C_param: float = 1.0,
        penalty: str = "l2"
    ) -> Pipeline:
        """
        Trains the final, production-ready Multinomial Logistic Regression Pipeline
        on the entire training set.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data must be prepared first.")

        scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
        selected_features = feature_subset if feature_subset is not None else self.feature_names_

        X_train_selected = self.X_train[selected_features]

        solver = "saga" if penalty == "l1" else "lbfgs"

        clf = LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=C_param,
            random_state=self.random_state,
            max_iter=2000
        )

        pipeline = Pipeline([
            ("scaler", scaler),
            ("clf", clf)
        ])

        pipeline.fit(X_train_selected, self.y_train)
        self.pipeline = pipeline
        self.classes_ = list(clf.classes_)

        # Extract coefficient interpretability matrix
        coefs = clf.coef_
        self.coefficients_df_ = pd.DataFrame(
            coefs,
            index=self.classes_,
            columns=selected_features
        )

        self.best_config_ = {
            "scaler": scaler_type,
            "feature_subset": selected_features,
            "num_features": len(selected_features),
            "C": C_param,
            "penalty": penalty,
            "solver": solver,
        }

        return pipeline

    def evaluate_on_test_set(self) -> Dict[str, Any]:
        """
        Evaluates the final fitted model pipeline on the unseen holdout test set.
        Computes Accuracy, Macro Precision, Macro Recall, Macro F1,
        Confusion Matrix, and full Classification Report.
        """
        if self.pipeline is None or self.X_test is None or self.y_test is None:
            raise RuntimeError("Model must be trained and test set available before evaluation.")

        selected_features = self.best_config_["feature_subset"]
        X_test_selected = self.X_test[selected_features]

        y_pred = self.pipeline.predict(X_test_selected)
        y_prob = self.pipeline.predict_proba(X_test_selected)

        acc = float(accuracy_score(self.y_test, y_pred))
        prec_macro = float(precision_score(self.y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        self.confusion_matrix_ = cm

        # Detailed per-class report dictionary
        report_dict = classification_report(
            self.y_test,
            y_pred,
            labels=self.classes_,
            output_dict=True,
            zero_division=0
        )
        self.classification_report_dict_ = report_dict

        # Multiclass log loss
        try:
            test_loss = float(log_loss(self.y_test, y_prob, labels=self.classes_))
        except Exception:
            test_loss = np.nan

        self.test_metrics_ = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "log_loss": test_loss,
            "test_sample_count": len(self.y_test),
            "classes": self.classes_,
            "confusion_matrix": cm.tolist(),
        }

        return self.test_metrics_

    def save_model(self, model_path: Union[str, Path] = BEST_MODEL_PATH) -> Path:
        """
        Serializes the complete fitted preprocessing and classification pipeline
        along with all configuration and evaluation metadata using Joblib.
        """
        if self.pipeline is None:
            raise RuntimeError("No trained pipeline available to save.")

        out_path = Path(model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "pipeline": self.pipeline,
            "best_config": self.best_config_,
            "feature_names": self.best_config_["feature_subset"],
            "classes": self.classes_,
            "test_metrics": self.test_metrics_,
            "confusion_matrix": self.confusion_matrix_,
            "coefficients_df": self.coefficients_df_,
            "scaling_experiment": self.scaling_experiment_results,
            "feature_selection_experiment": self.feature_selection_results,
        }

        joblib.dump(artifact, out_path)
        return out_path

    # Implement BaseDiagnosticModel abstract methods
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionDiagnosticEngine":
        self.prepare_data()
        self.train_best_model()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict(X[selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict_proba(X[selected])


# -----------------------------------------------------------------------------
# Concrete Supervised Diagnostic Models (5 Additional ML Syllabus Classifiers)
# -----------------------------------------------------------------------------

class LinearSVMDiagnosticEngine(BaseDiagnosticModel):
    """
    Linear Support Vector Machine (SVM) Diagnostic Engine.
    Uses linear hyperplanes with L2 penalty and Calibrated Probability output
    via Platt scaling (SVC probability=True). Provides interpretable linear weights.
    """
    METADATA_COLUMNS = ["query_id", "query", "provenance", "failure_label"]

    def __init__(self, C: float = 1.0, random_state: int = RANDOM_STATE, test_size: float = TEST_SPLIT_SIZE, cv_folds: int = N_CROSS_VALIDATION_FOLDS):
        self.C = C
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names_ = []

        self.pipeline = None
        self.best_config_ = {}
        self.test_metrics_ = {}
        self.cv_metrics_ = {}
        self.confusion_matrix_ = None
        self.classification_report_dict_ = {}
        self.classes_ = []
        self.coefficients_df_ = None

    def prepare_data(self, features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV):
        df = pd.read_csv(features_csv)
        valid_df = df[df["failure_label"].isin(FAILURE_CATEGORIES)].copy()
        feature_cols = [c for c in valid_df.columns if c not in self.METADATA_COLUMNS]
        numeric_cols = valid_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = valid_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = valid_df["failure_label"]

        self.feature_names_ = numeric_cols
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, feature_subset: Optional[List[str]] = None, C_param: Optional[float] = None) -> Pipeline:
        if self.X_train is None:
            self.prepare_data()
        c_val = C_param if C_param is not None else self.C
        selected = feature_subset if feature_subset is not None else self.feature_names_

        clf = SVC(kernel="linear", C=c_val, probability=True, random_state=self.random_state)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # Run Stratified CV on training data
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_res = cross_validate(pipe, self.X_train[selected], self.y_train, cv=cv, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
        self.cv_metrics_ = {
            "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
            "std_accuracy": float(np.std(cv_res["test_accuracy"])),
            "mean_f1_macro": float(np.mean(cv_res["test_f1_macro"])),
            "std_f1_macro": float(np.std(cv_res["test_f1_macro"])),
            "mean_precision_macro": float(np.mean(cv_res["test_precision_macro"])),
            "mean_recall_macro": float(np.mean(cv_res["test_recall_macro"])),
        }

        pipe.fit(self.X_train[selected], self.y_train)
        self.pipeline = pipe
        self.classes_ = list(clf.classes_)

        # Extract linear hyperplanes: SVC multi-class uses one-vs-one (OVO) producing n_classes*(n_classes-1)/2 rows
        n_classes = len(self.classes_)
        pair_labels = [f"{self.classes_[i]} vs {self.classes_[j]}" for i in range(n_classes) for j in range(i + 1, n_classes)]
        self.pairwise_coefficients_df_ = pd.DataFrame(clf.coef_, index=pair_labels, columns=selected)

        # Aggregate one-vs-one coefficients to obtain effective per-class directional feature weights
        per_class_coef = np.zeros((n_classes, len(selected)))
        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                per_class_coef[i] += clf.coef_[k]
                per_class_coef[j] -= clf.coef_[k]
                k += 1
        per_class_coef /= (n_classes - 1)
        self.coefficients_df_ = pd.DataFrame(per_class_coef, index=self.classes_, columns=selected)
        self.best_config_ = {
            "algorithm": "Linear SVM",
            "kernel": "linear",
            "C": c_val,
            "scaler": "StandardScaler",
            "feature_subset": selected,
            "num_features": len(selected),
        }
        return pipe

    def evaluate_on_test_set(self) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Model must be trained before evaluation.")
        selected = self.best_config_["feature_subset"]
        X_test_sub = self.X_test[selected]

        y_pred = self.pipeline.predict(X_test_sub)
        y_prob = self.pipeline.predict_proba(X_test_sub)

        acc = float(accuracy_score(self.y_test, y_pred))
        prec = float(precision_score(self.y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_m = float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_w = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        self.confusion_matrix_ = cm
        self.classification_report_dict_ = classification_report(self.y_test, y_pred, labels=self.classes_, output_dict=True, zero_division=0)

        self.test_metrics_ = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "classes": self.classes_,
            "confusion_matrix": cm.tolist(),
            "test_sample_count": len(self.y_test),
        }
        return self.test_metrics_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearSVMDiagnosticEngine":
        self.prepare_data()
        self.train_model()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict(X[selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict_proba(X[selected])


class RBFSVMDiagnosticEngine(BaseDiagnosticModel):
    """
    RBF Kernel Support Vector Machine Diagnostic Engine.
    Maps non-linear failure distributions into high-dimensional Hilbert spaces
    via Gaussian Radial Basis Function (RBF) kernel with Platt probability calibration.
    """
    METADATA_COLUMNS = ["query_id", "query", "provenance", "failure_label"]

    def __init__(self, C: float = 1.0, gamma: str = "scale", random_state: int = RANDOM_STATE, test_size: float = TEST_SPLIT_SIZE, cv_folds: int = N_CROSS_VALIDATION_FOLDS):
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names_ = []

        self.pipeline = None
        self.best_config_ = {}
        self.test_metrics_ = {}
        self.cv_metrics_ = {}
        self.confusion_matrix_ = None
        self.classification_report_dict_ = {}
        self.classes_ = []
        self.support_vectors_count_ = 0

    def prepare_data(self, features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV):
        df = pd.read_csv(features_csv)
        valid_df = df[df["failure_label"].isin(FAILURE_CATEGORIES)].copy()
        feature_cols = [c for c in valid_df.columns if c not in self.METADATA_COLUMNS]
        numeric_cols = valid_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = valid_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = valid_df["failure_label"]

        self.feature_names_ = numeric_cols
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, feature_subset: Optional[List[str]] = None, C_param: Optional[float] = None, gamma_param: Optional[str] = None) -> Pipeline:
        if self.X_train is None:
            self.prepare_data()
        c_val = C_param if C_param is not None else self.C
        g_val = gamma_param if gamma_param is not None else self.gamma
        selected = feature_subset if feature_subset is not None else self.feature_names_

        clf = SVC(kernel="rbf", C=c_val, gamma=g_val, probability=True, random_state=self.random_state)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_res = cross_validate(pipe, self.X_train[selected], self.y_train, cv=cv, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
        self.cv_metrics_ = {
            "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
            "std_accuracy": float(np.std(cv_res["test_accuracy"])),
            "mean_f1_macro": float(np.mean(cv_res["test_f1_macro"])),
            "std_f1_macro": float(np.std(cv_res["test_f1_macro"])),
            "mean_precision_macro": float(np.mean(cv_res["test_precision_macro"])),
            "mean_recall_macro": float(np.mean(cv_res["test_recall_macro"])),
        }

        pipe.fit(self.X_train[selected], self.y_train)
        self.pipeline = pipe
        self.classes_ = list(clf.classes_)
        self.support_vectors_count_ = int(len(clf.support_))
        self.best_config_ = {
            "algorithm": "RBF Kernel SVM",
            "kernel": "rbf",
            "C": c_val,
            "gamma": g_val,
            "scaler": "StandardScaler",
            "support_vectors": self.support_vectors_count_,
            "feature_subset": selected,
            "num_features": len(selected),
        }
        return pipe

    def evaluate_on_test_set(self) -> Dict[str, Any]:
        if self.pipeline is None: raise RuntimeError("Model must be trained before evaluation.")
        selected = self.best_config_["feature_subset"]
        X_test_sub = self.X_test[selected]

        y_pred = self.pipeline.predict(X_test_sub)
        y_prob = self.pipeline.predict_proba(X_test_sub)

        acc = float(accuracy_score(self.y_test, y_pred))
        prec = float(precision_score(self.y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_m = float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_w = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        self.confusion_matrix_ = cm
        self.classification_report_dict_ = classification_report(self.y_test, y_pred, labels=self.classes_, output_dict=True, zero_division=0)

        self.test_metrics_ = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "classes": self.classes_,
            "confusion_matrix": cm.tolist(),
            "test_sample_count": len(self.y_test),
        }
        return self.test_metrics_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RBFSVMDiagnosticEngine":
        self.prepare_data()
        self.train_model()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict(X[selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict_proba(X[selected])


class DecisionTreeDiagnosticEngine(BaseDiagnosticModel):
    """
    Decision Tree Learning Diagnostic Engine.
    Constructs hierarchical, recursive partitioning decision boundaries using
    Gini Impurity / Information Gain. Extracts interpretable feature importances.
    """
    METADATA_COLUMNS = ["query_id", "query", "provenance", "failure_label"]

    def __init__(self, max_depth: Optional[int] = 5, criterion: str = "gini", min_samples_split: int = 3, random_state: int = RANDOM_STATE, test_size: float = TEST_SPLIT_SIZE, cv_folds: int = N_CROSS_VALIDATION_FOLDS):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names_ = []

        self.pipeline = None
        self.best_config_ = {}
        self.test_metrics_ = {}
        self.cv_metrics_ = {}
        self.confusion_matrix_ = None
        self.classification_report_dict_ = {}
        self.classes_ = []
        self.feature_importances_df_ = None

    def prepare_data(self, features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV):
        df = pd.read_csv(features_csv)
        valid_df = df[df["failure_label"].isin(FAILURE_CATEGORIES)].copy()
        feature_cols = [c for c in valid_df.columns if c not in self.METADATA_COLUMNS]
        numeric_cols = valid_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = valid_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = valid_df["failure_label"]

        self.feature_names_ = numeric_cols
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, feature_subset: Optional[List[str]] = None, max_depth: Optional[int] = None) -> Pipeline:
        if self.X_train is None:
            self.prepare_data()
        depth = max_depth if max_depth is not None else self.max_depth
        selected = feature_subset if feature_subset is not None else self.feature_names_

        clf = DecisionTreeClassifier(
            max_depth=depth,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_res = cross_validate(pipe, self.X_train[selected], self.y_train, cv=cv, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
        self.cv_metrics_ = {
            "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
            "std_accuracy": float(np.std(cv_res["test_accuracy"])),
            "mean_f1_macro": float(np.mean(cv_res["test_f1_macro"])),
            "std_f1_macro": float(np.std(cv_res["test_f1_macro"])),
            "mean_precision_macro": float(np.mean(cv_res["test_precision_macro"])),
            "mean_recall_macro": float(np.mean(cv_res["test_recall_macro"])),
        }

        pipe.fit(self.X_train[selected], self.y_train)
        self.pipeline = pipe
        self.classes_ = list(clf.classes_)

        self.feature_importances_df_ = pd.DataFrame({
            "feature": selected,
            "importance": clf.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        self.best_config_ = {
            "algorithm": "Decision Tree",
            "max_depth": depth,
            "criterion": self.criterion,
            "min_samples_split": self.min_samples_split,
            "feature_subset": selected,
            "num_features": len(selected),
        }
        return pipe

    def evaluate_on_test_set(self) -> Dict[str, Any]:
        if self.pipeline is None: raise RuntimeError("Model must be trained before evaluation.")
        selected = self.best_config_["feature_subset"]
        X_test_sub = self.X_test[selected]

        y_pred = self.pipeline.predict(X_test_sub)
        y_prob = self.pipeline.predict_proba(X_test_sub)

        acc = float(accuracy_score(self.y_test, y_pred))
        prec = float(precision_score(self.y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_m = float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_w = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        self.confusion_matrix_ = cm
        self.classification_report_dict_ = classification_report(self.y_test, y_pred, labels=self.classes_, output_dict=True, zero_division=0)

        self.test_metrics_ = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "classes": self.classes_,
            "confusion_matrix": cm.tolist(),
            "test_sample_count": len(self.y_test),
        }
        return self.test_metrics_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DecisionTreeDiagnosticEngine":
        self.prepare_data()
        self.train_model()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict(X[selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict_proba(X[selected])


class RandomForestDiagnosticEngine(BaseDiagnosticModel):
    """
    Random Forest Ensemble Diagnostic Engine.
    Aggregates randomized decorrelated decision trees via Bagging (Bootstrap Aggregation).
    Reduces diagnostic variance and mitigates overfitting on complex multimodal failure spaces.
    """
    METADATA_COLUMNS = ["query_id", "query", "provenance", "failure_label"]

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 6, criterion: str = "gini", random_state: int = RANDOM_STATE, test_size: float = TEST_SPLIT_SIZE, cv_folds: int = N_CROSS_VALIDATION_FOLDS):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names_ = []

        self.pipeline = None
        self.best_config_ = {}
        self.test_metrics_ = {}
        self.cv_metrics_ = {}
        self.confusion_matrix_ = None
        self.classification_report_dict_ = {}
        self.classes_ = []
        self.feature_importances_df_ = None

    def prepare_data(self, features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV):
        df = pd.read_csv(features_csv)
        valid_df = df[df["failure_label"].isin(FAILURE_CATEGORIES)].copy()
        feature_cols = [c for c in valid_df.columns if c not in self.METADATA_COLUMNS]
        numeric_cols = valid_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = valid_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = valid_df["failure_label"]

        self.feature_names_ = numeric_cols
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, feature_subset: Optional[List[str]] = None, n_estimators: Optional[int] = None, max_depth: Optional[int] = None) -> Pipeline:
        if self.X_train is None:
            self.prepare_data()
        n_est = n_estimators if n_estimators is not None else self.n_estimators
        depth = max_depth if max_depth is not None else self.max_depth
        selected = feature_subset if feature_subset is not None else self.feature_names_

        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            criterion=self.criterion,
            random_state=self.random_state
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_res = cross_validate(pipe, self.X_train[selected], self.y_train, cv=cv, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
        self.cv_metrics_ = {
            "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
            "std_accuracy": float(np.std(cv_res["test_accuracy"])),
            "mean_f1_macro": float(np.mean(cv_res["test_f1_macro"])),
            "std_f1_macro": float(np.std(cv_res["test_f1_macro"])),
            "mean_precision_macro": float(np.mean(cv_res["test_precision_macro"])),
            "mean_recall_macro": float(np.mean(cv_res["test_recall_macro"])),
        }

        pipe.fit(self.X_train[selected], self.y_train)
        self.pipeline = pipe
        self.classes_ = list(clf.classes_)

        self.feature_importances_df_ = pd.DataFrame({
            "feature": selected,
            "importance": clf.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        self.best_config_ = {
            "algorithm": "Random Forest",
            "n_estimators": n_est,
            "max_depth": depth,
            "criterion": self.criterion,
            "feature_subset": selected,
            "num_features": len(selected),
        }
        return pipe

    def evaluate_on_test_set(self) -> Dict[str, Any]:
        if self.pipeline is None: raise RuntimeError("Model must be trained before evaluation.")
        selected = self.best_config_["feature_subset"]
        X_test_sub = self.X_test[selected]

        y_pred = self.pipeline.predict(X_test_sub)
        y_prob = self.pipeline.predict_proba(X_test_sub)

        acc = float(accuracy_score(self.y_test, y_pred))
        prec = float(precision_score(self.y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_m = float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_w = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        self.confusion_matrix_ = cm
        self.classification_report_dict_ = classification_report(self.y_test, y_pred, labels=self.classes_, output_dict=True, zero_division=0)

        self.test_metrics_ = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "classes": self.classes_,
            "confusion_matrix": cm.tolist(),
            "test_sample_count": len(self.y_test),
        }
        return self.test_metrics_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestDiagnosticEngine":
        self.prepare_data()
        self.train_model()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict(X[selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict_proba(X[selected])


class KNNDiagnosticEngine(BaseDiagnosticModel):
    """
    K-Nearest Neighbors (KNN) Diagnostic Engine.
    Instance-based non-parametric classifier using distance-weighted Minkowski/Euclidean metric.
    Particularly sensitive to feature scaling and proximity in retrieval metric space.
    """
    METADATA_COLUMNS = ["query_id", "query", "provenance", "failure_label"]

    def __init__(self, n_neighbors: int = 5, weights: str = "distance", metric: str = "minkowski", p: int = 2, random_state: int = RANDOM_STATE, test_size: float = TEST_SPLIT_SIZE, cv_folds: int = N_CROSS_VALIDATION_FOLDS):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names_ = []

        self.pipeline = None
        self.best_config_ = {}
        self.test_metrics_ = {}
        self.cv_metrics_ = {}
        self.confusion_matrix_ = None
        self.classification_report_dict_ = {}
        self.classes_ = []

    def prepare_data(self, features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV):
        df = pd.read_csv(features_csv)
        valid_df = df[df["failure_label"].isin(FAILURE_CATEGORIES)].copy()
        feature_cols = [c for c in valid_df.columns if c not in self.METADATA_COLUMNS]
        numeric_cols = valid_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = valid_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = valid_df["failure_label"]

        self.feature_names_ = numeric_cols
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, feature_subset: Optional[List[str]] = None, n_neighbors: Optional[int] = None) -> Pipeline:
        if self.X_train is None:
            self.prepare_data()
        k_val = n_neighbors if n_neighbors is not None else self.n_neighbors
        selected = feature_subset if feature_subset is not None else self.feature_names_

        clf = KNeighborsClassifier(
            n_neighbors=k_val,
            weights=self.weights,
            metric=self.metric,
            p=self.p
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_res = cross_validate(pipe, self.X_train[selected], self.y_train, cv=cv, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
        self.cv_metrics_ = {
            "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
            "std_accuracy": float(np.std(cv_res["test_accuracy"])),
            "mean_f1_macro": float(np.mean(cv_res["test_f1_macro"])),
            "std_f1_macro": float(np.std(cv_res["test_f1_macro"])),
            "mean_precision_macro": float(np.mean(cv_res["test_precision_macro"])),
            "mean_recall_macro": float(np.mean(cv_res["test_recall_macro"])),
        }

        pipe.fit(self.X_train[selected], self.y_train)
        self.pipeline = pipe
        self.classes_ = list(clf.classes_)

        self.best_config_ = {
            "algorithm": "K-Nearest Neighbors",
            "n_neighbors": k_val,
            "weights": self.weights,
            "metric": self.metric,
            "feature_subset": selected,
            "num_features": len(selected),
        }
        return pipe

    def evaluate_on_test_set(self) -> Dict[str, Any]:
        if self.pipeline is None: raise RuntimeError("Model must be trained before evaluation.")
        selected = self.best_config_["feature_subset"]
        X_test_sub = self.X_test[selected]

        y_pred = self.pipeline.predict(X_test_sub)
        y_prob = self.pipeline.predict_proba(X_test_sub)

        acc = float(accuracy_score(self.y_test, y_pred))
        prec = float(precision_score(self.y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_m = float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))
        f1_w = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        self.confusion_matrix_ = cm
        self.classification_report_dict_ = classification_report(self.y_test, y_pred, labels=self.classes_, output_dict=True, zero_division=0)

        self.test_metrics_ = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "classes": self.classes_,
            "confusion_matrix": cm.tolist(),
            "test_sample_count": len(self.y_test),
        }
        return self.test_metrics_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KNNDiagnosticEngine":
        self.prepare_data()
        self.train_model()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict(X[selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None: raise RuntimeError("Model is not fitted.")
        selected = self.best_config_.get("feature_subset", self.feature_names_)
        return self.pipeline.predict_proba(X[selected])


# -----------------------------------------------------------------------------
# Multi-Model Shootout & Diagnostic Suite (Coordinating All 6 Classifiers)
# -----------------------------------------------------------------------------

class MultiModelDiagnosticSuite:
    """
    Coordinates end-to-end multi-model benchmarking shootout across all 6 classical ML classifiers:
    1. Multinomial Logistic Regression
    2. Linear Support Vector Machine (SVM)
    3. RBF Kernel Support Vector Machine
    4. Decision Tree Classifier
    5. Random Forest Ensemble
    6. K-Nearest Neighbors (KNN)

    Ensures identical train/test split, identical feature matrix, and rigorous comparative metrics.
    """

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        test_size: float = TEST_SPLIT_SIZE,
        cv_folds: int = N_CROSS_VALIDATION_FOLDS
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        self.engines: Dict[str, Any] = {
            "Logistic Regression": LogisticRegressionDiagnosticEngine(random_state=random_state, test_size=test_size, cv_folds=cv_folds),
            "Linear SVM": LinearSVMDiagnosticEngine(random_state=random_state, test_size=test_size, cv_folds=cv_folds),
            "RBF Kernel SVM": RBFSVMDiagnosticEngine(random_state=random_state, test_size=test_size, cv_folds=cv_folds),
            "Decision Tree": DecisionTreeDiagnosticEngine(random_state=random_state, test_size=test_size, cv_folds=cv_folds),
            "Random Forest": RandomForestDiagnosticEngine(random_state=random_state, test_size=test_size, cv_folds=cv_folds),
            "K-Nearest Neighbors": KNNDiagnosticEngine(random_state=random_state, test_size=test_size, cv_folds=cv_folds),
        }

        self.comparison_df_: Optional[pd.DataFrame] = None
        self.shootout_summary_: Dict[str, Dict[str, Any]] = {}
        self.best_model_name_: Optional[str] = None
        self.best_engine_: Optional[Any] = None
        self.selected_features_: List[str] = []

    def train_and_evaluate_all(
        self,
        features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV,
        feature_subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Trains and cross-validates all 6 models on the training set,
        evaluates on the holdout test set, and builds the benchmark ranking table.
        """
        # Determine best feature set using Logistic Regression selector if none provided
        log_engine = self.engines["Logistic Regression"]
        log_engine.prepare_data(features_csv)
        log_engine.run_scaling_experiment()
        feat_res = log_engine.run_feature_selection_experiment()

        if feature_subset is None:
            # Use top Embedded L1 features (or Filter if fallback)
            best_cfg_key = max(feat_res["configurations"].keys(), key=lambda k: feat_res["configurations"][k]["mean_f1_macro"])
            self.selected_features_ = feat_res["configurations"][best_cfg_key]["features_list"]
        else:
            self.selected_features_ = feature_subset

        records = []

        for name, engine in self.engines.items():
            engine.prepare_data(features_csv)
            if name == "Logistic Regression":
                engine.train_best_model(scaler_type="StandardScaler", feature_subset=self.selected_features_, C_param=1.0)
            else:
                engine.train_model(feature_subset=self.selected_features_)

            t_metrics = engine.evaluate_on_test_set()

            # Retrieve CV metrics
            if hasattr(engine, "cv_metrics_") and engine.cv_metrics_:
                cv_acc = engine.cv_metrics_["mean_accuracy"]
                cv_acc_std = engine.cv_metrics_["std_accuracy"]
                cv_f1 = engine.cv_metrics_["mean_f1_macro"]
                cv_f1_std = engine.cv_metrics_["std_f1_macro"]
            else:
                # Fallback to scaling experiment for log reg
                cv_acc = log_engine.scaling_experiment_results["StandardScaler"]["mean_accuracy"]
                cv_acc_std = log_engine.scaling_experiment_results["StandardScaler"]["std_accuracy"]
                cv_f1 = log_engine.scaling_experiment_results["StandardScaler"]["mean_f1_macro"]
                cv_f1_std = log_engine.scaling_experiment_results["StandardScaler"]["std_f1_macro"]

            rec = {
                "Model": name,
                "CV Accuracy": f"{cv_acc * 100:.1f}% (±{cv_acc_std * 100:.1f}%)",
                "CV Macro F1": f"{cv_f1:.4f} (±{cv_f1_std:.4f})",
                "Test Accuracy": f"{t_metrics['accuracy'] * 100:.1f}%",
                "Test Macro Precision": f"{t_metrics['precision_macro']:.4f}",
                "Test Macro Recall": f"{t_metrics['recall_macro']:.4f}",
                "Test Macro F1": f"{t_metrics['f1_macro']:.4f}",
                "Test Weighted F1": f"{t_metrics['f1_weighted']:.4f}",
                "raw_test_acc": t_metrics["accuracy"],
                "raw_test_f1": t_metrics["f1_macro"],
                "raw_cv_f1": cv_f1,
            }
            records.append(rec)
            self.shootout_summary_[name] = {
                "test_metrics": t_metrics,
                "best_config": engine.best_config_,
                "confusion_matrix": engine.confusion_matrix_.tolist(),
            }

        comp_df = pd.DataFrame(records).sort_values(by=["raw_test_f1", "raw_test_acc"], ascending=False).reset_index(drop=True)
        self.comparison_df_ = comp_df

        self.best_model_name_ = comp_df.iloc[0]["Model"]
        self.best_engine_ = self.engines[self.best_model_name_]

        return comp_df

    def save_suite(
        self,
        suite_path: Union[str, Path] = ALL_MODELS_PATH,
        best_model_path: Union[str, Path] = BEST_MODEL_PATH
    ):
        """Serializes the multi-model comparison suite and updates the best diagnostic model artifact."""
        out_suite = Path(suite_path)
        out_suite.parent.mkdir(parents=True, exist_ok=True)

        suite_payload = {
            "comparison_df": self.comparison_df_,
            "shootout_summary": self.shootout_summary_,
            "best_model_name": self.best_model_name_,
            "selected_features": self.selected_features_,
            "engines": self.engines,
        }
        joblib.dump(suite_payload, out_suite)

        # Also persist best model for backward compatibility with existing single-model pages
        if self.best_engine_ is not None:
            # If best engine is LogisticRegressionDiagnosticEngine, use its save_model
            if hasattr(self.best_engine_, "save_model"):
                self.best_engine_.save_model(best_model_path)
            else:
                # Pack standardized artifact
                art = {
                    "pipeline": self.best_engine_.pipeline,
                    "best_config": self.best_engine_.best_config_,
                    "feature_names": self.selected_features_,
                    "classes": self.best_engine_.classes_,
                    "test_metrics": self.best_engine_.test_metrics_,
                    "confusion_matrix": self.best_engine_.confusion_matrix_,
                    "coefficients_df": getattr(self.best_engine_, "coefficients_df_", None),
                    "feature_importances_df": getattr(self.best_engine_, "feature_importances_df_", None),
                    "algorithm": self.best_model_name_,
                    "scaling_experiment": self.engines["Logistic Regression"].scaling_experiment_results,
                    "feature_selection_experiment": self.engines["Logistic Regression"].feature_selection_results,
                }
                joblib.dump(art, Path(best_model_path))

        return out_suite
