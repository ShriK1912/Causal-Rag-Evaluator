"""
Feature Selection Module for Causal RAG Failure Diagnostician.

Implements two classical feature selection strategies:
1. Filter-Based Selection: Uses ANOVA F-value (f_classif) to measure linear dependency
   between individual numerical features and target failure classes.
2. Embedded L1 Selection: Uses L1-penalized (Lasso) Multinomial Logistic Regression
   with SAGA solver to induce coefficient sparsity, eliminating non-informative features.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class FilterFeatureSelector:
    """
    Filter-based feature selection using ANOVA F-statistic (f_classif).
    Ranks features by univariate statistical significance without model training.
    """

    def __init__(self, k: int = 15):
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}
        self.feature_pvalues_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FilterFeatureSelector":
        """Fits SelectKBest on training data."""
        # Replace infs and drop constant columns for statistical stability
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        self.selector.fit(X_clean, y)
        support = self.selector.get_support()
        
        self.selected_features_ = X.columns[support].tolist()
        for idx, col in enumerate(X.columns):
            self.feature_scores_[col] = float(self.selector.scores_[idx]) if not np.isnan(self.selector.scores_[idx]) else 0.0
            self.feature_pvalues_[col] = float(self.selector.pvalues_[idx]) if not np.isnan(self.selector.pvalues_[idx]) else 1.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Projects dataframe to selected k features."""
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fits and transforms."""
        return self.fit(X, y).transform(X)

    def get_ranked_feature_summary(self) -> pd.DataFrame:
        """Returns dataframe of all candidate features ranked by ANOVA F-score."""
        summary_df = pd.DataFrame({
            "feature": list(self.feature_scores_.keys()),
            "f_statistic": list(self.feature_scores_.values()),
            "p_value": list(self.feature_pvalues_.values()),
        }).sort_values(by="f_statistic", ascending=False).reset_index(drop=True)
        summary_df["is_selected"] = summary_df["feature"].isin(self.selected_features_)
        return summary_df


class EmbeddedL1FeatureSelector:
    """
    Embedded feature selection using L1-penalized (Lasso) Multinomial Logistic Regression.
    Enforces sparsity by driving coefficients of irrelevant features to exactly 0.0.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42, max_iter: int = 3000):
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.model: Optional[LogisticRegression] = None
        self.selected_features_: List[str] = []
        self.coef_matrix_: Optional[pd.DataFrame] = None
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EmbeddedL1FeatureSelector":
        """
        Fits L1-penalized Logistic Regression on standardized data.
        Features where the L1-norm across all class coefficients is > 1e-5 are retained.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.replace([np.inf, -np.inf], np.nan).fillna(0.0))

        # Use SAGA solver for multinomial L1 regularization
        self.model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=1e-4
        )
        self.model.fit(X_scaled, y)

        # Coefficients shape: (n_classes, n_features)
        coef = self.model.coef_
        classes = self.model.classes_

        # Calculate mean absolute coefficient magnitude per feature across all classes
        mean_abs_coef = np.mean(np.abs(coef), axis=0)

        selected_mask = mean_abs_coef > 1e-4
        # Fallback if too aggressive: retain at least top 5 features
        if not np.any(selected_mask):
            selected_indices = np.argsort(mean_abs_coef)[-5:]
            selected_mask = np.zeros(len(mean_abs_coef), dtype=bool)
            selected_mask[selected_indices] = True

        self.selected_features_ = X.columns[selected_mask].tolist()

        for idx, col in enumerate(X.columns):
            self.feature_importances_[col] = float(mean_abs_coef[idx])

        # Store coefficient matrix
        self.coef_matrix_ = pd.DataFrame(
            coef,
            index=classes,
            columns=X.columns
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Projects dataframe to selected L1 non-zero features."""
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fits and transforms."""
        return self.fit(X, y).transform(X)

    def get_coefficient_summary(self) -> pd.DataFrame:
        """Returns dataframe showing feature coefficients per class and mean magnitude."""
        summary = pd.DataFrame({
            "feature": list(self.feature_importances_.keys()),
            "mean_abs_coefficient": list(self.feature_importances_.values()),
        }).sort_values(by="mean_abs_coefficient", ascending=False).reset_index(drop=True)
        summary["is_selected"] = summary["feature"].isin(self.selected_features_)
        return summary
