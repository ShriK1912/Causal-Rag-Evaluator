"""
Apriori Association Rule Mining Engine for Causal RAG Failure Diagnostician Platform.

Discovers empirical co-occurrence patterns between discretized retrieval feature metrics
and RAG failure categories using the classical Apriori frequent itemset algorithm
and association rule mining (via mlxtend).

Key Capabilities:
  - Discretizes continuous retrieval features into categorical bins ('Low', 'Medium', 'High').
  - Safely handles constant columns, missing values, and non-feature identifiers.
  - Generates frequent itemsets with configurable minimum support.
  - Extracts association rules with configurable confidence, lift, leverage, and conviction.
  - Filters rules specific to each of the 5 failure categories:
      1. Retrieval Miss
      2. Ranking Error
      3. Context Contamination
      4. Hallucination
      5. Contradiction
  - Generates human-interpretable diagnostic rule explanations.

CRITICAL ACADEMIC DISCLAIMER:
Association rules describe observed statistical co-occurrence in transaction data
(e.g., Support, Confidence, Lift). They do NOT constitute causal proof of failure.
An association rule identifies feature patterns that correlate with failure modes;
causal attribution requires domain validation and intervention.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_FEATURES_CSV,
    FAILURE_CATEGORIES,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_LIFT,
    NUMERICAL_DISCRETIZATION_BINS,
    DISCRETIZATION_LABELS,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature Discretization (Section 3)
# ---------------------------------------------------------------------------

def discretize_features(
    df: pd.DataFrame,
    bins: int = NUMERICAL_DISCRETIZATION_BINS,
    strategy: str = "uniform",
    labels: Optional[List[str]] = None,
    target_col: str = "failure_label",
    ignore_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Converts continuous numerical features into discrete ordinal bins (e.g. Low, Medium, High).

    Methodology:
      - 'uniform' (Equal-Width, Default): Divides [min, max] range into B equal intervals.
        Preserves the absolute physical distance / similarity scale.
      - 'quantile' (Equal-Frequency): Uses empirical percentiles to place equal sample counts
        per bin. Duplicates in bin edges are handled with 'drop'.

    Safeguards:
      - Ignores query identifiers, raw prompt text, and provenance metadata.
      - Detects and drops constant numeric columns (nunique <= 1) to prevent trivial 100% support rules.
      - Imputes missing numeric values with column medians prior to binning.

    Returns:
        discretized_df       : DataFrame containing discretized categorical columns + target_col.
        constant_cols_dropped: List of dropped column names.
        bin_edges_dict       : Dict mapping feature names to their bin edge boundaries.
    """
    if df.empty:
        raise ValueError("Cannot discretize an empty DataFrame.")

    if labels is None:
        labels = DISCRETIZATION_LABELS[:bins]

    if ignore_cols is None:
        ignore_cols = ["query_id", "query", "provenance"]

    # Filter out identifier columns
    cols_to_process = [c for c in df.columns if c not in ignore_cols and c != target_col]

    discretized = pd.DataFrame(index=df.index)
    constant_cols_dropped: List[str] = []
    bin_edges_dict: Dict[str, Any] = {}

    for col in cols_to_process:
        series = df[col]

        # Handle numeric features
        if pd.api.types.is_numeric_dtype(series):
            # Impute NaNs with median if any exist
            if series.isnull().any():
                series = series.fillna(series.median())

            # Check for constant columns
            if series.nunique() <= 1:
                constant_cols_dropped.append(col)
                continue

            try:
                if strategy == "quantile":
                    binned, edges = pd.qcut(
                        series, q=bins, labels=labels, retbins=True, duplicates="drop"
                    )
                else:  # 'uniform'
                    binned, edges = pd.cut(
                        series, bins=bins, labels=labels, retbins=True, include_lowest=True
                    )

                discretized[col] = binned.astype(str)
                bin_edges_dict[col] = edges.tolist()
            except Exception:
                # Fallback to uniform if quantile edges collapse
                binned, edges = pd.cut(
                    series, bins=bins, labels=labels, retbins=True, include_lowest=True
                )
                discretized[col] = binned.astype(str)
                bin_edges_dict[col] = edges.tolist()
        else:
            # Categorical feature: retain as-is
            discretized[col] = series.astype(str)

    # Attach target column if present
    if target_col in df.columns:
        discretized[target_col] = df[target_col].astype(str)

    return discretized, constant_cols_dropped, bin_edges_dict


def encode_transactions(
    discretized_df: pd.DataFrame,
    target_col: str = "failure_label",
) -> pd.DataFrame:
    """
    One-hot encodes discretized features into a boolean transaction matrix.
    Renames target column values to explicit prefix 'failure_category=<Category>'.
    """
    df_enc = discretized_df.copy()

    # Reformat target column explicitly for clear rule interpretation
    if target_col in df_enc.columns:
        df_enc["failure_category"] = df_enc[target_col].apply(lambda x: f"failure_category={x}")
        df_enc = df_enc.drop(columns=[target_col])

    # Convert all columns to dummy booleans
    one_hot = pd.get_dummies(df_enc, prefix_sep="=", dtype=bool)

    # Rename dummy column for target to clean format if needed
    rename_map = {}
    for c in one_hot.columns:
        if c.startswith("failure_category="):
            # Clean format e.g. failure_category=failure_category=Retrieval Miss -> failure_category=Retrieval Miss
            clean_name = c.replace("failure_category=failure_category=", "failure_category=")
            rename_map[c] = clean_name
    if rename_map:
        one_hot = one_hot.rename(columns=rename_map)

    return one_hot


# ---------------------------------------------------------------------------
# Apriori Frequent Itemsets & Association Rules (Section 4)
# ---------------------------------------------------------------------------

def mine_frequent_itemsets(
    one_hot_df: pd.DataFrame,
    min_support: float = DEFAULT_MIN_SUPPORT,
    max_len: int = 3,
) -> pd.DataFrame:
    """
    Discovers frequent itemsets using the Apriori algorithm via mlxtend.

    Mathematical Definition:
      Support(I) = count(I) / N >= min_support
    """
    if not (0.0 < min_support <= 1.0):
        raise ValueError(f"min_support must be in (0.0, 1.0], got {min_support}")

    if one_hot_df.empty or len(one_hot_df) < 2:
        return pd.DataFrame(columns=["support", "itemsets"])

    frequent_itemsets = apriori(
        one_hot_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)
    return frequent_itemsets


def generate_diagnostic_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = DEFAULT_MIN_CONFIDENCE,
    min_lift: float = DEFAULT_MIN_LIFT,
) -> pd.DataFrame:
    """
    Extracts association rules from frequent itemsets and calculates:
      - Support: P(A ∩ C)
      - Confidence: P(C | A) = Support(A ∩ C) / Support(A)
      - Lift: Confidence(A -> C) / Support(C)
      - Leverage: Support(A ∩ C) - (Support(A) * Support(C))
      - Conviction: (1 - Support(C)) / (1 - Confidence(A -> C))

    Filters rules specific to RAG failure categories:
      Antecedent: [feature_pattern_1, ...]
      Consequent: [failure_category = Category]
    """
    if frequent_itemsets.empty or len(frequent_itemsets) < 2:
        return pd.DataFrame(columns=[
            "antecedents", "consequents", "antecedent support", "consequent support",
            "support", "confidence", "lift", "leverage", "conviction",
            "antecedents_str", "consequents_str", "target_category", "is_diagnostic_rule"
        ])

    try:
        rules = association_rules(
            frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
        )
    except Exception:
        return pd.DataFrame()

    if rules.empty:
        return pd.DataFrame()

    # Apply minimum lift constraint
    if min_lift > 0.0:
        rules = rules[rules["lift"] >= min_lift]

    if rules.empty:
        return pd.DataFrame()

    # Add readable string representations
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    # Identify diagnostic rules (rules whose consequent is a failure category)
    def extract_target_cat(consequents_set):
        con_list = list(consequents_set)
        if len(con_list) == 1 and con_list[0].startswith("failure_category="):
            return con_list[0].replace("failure_category=", "").strip()
        return None

    rules["target_category"] = rules["consequents"].apply(extract_target_cat)
    rules["is_diagnostic_rule"] = rules["target_category"].notnull()

    # Sort primarily by Lift, then Confidence, then Support
    rules = rules.sort_values(by=["lift", "confidence", "support"], ascending=[False, False, False]).reset_index(drop=True)
    return rules


def filter_rules_by_category(
    rules_df: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    """Filters association rules targeting a specific failure category."""
    if rules_df.empty or "target_category" not in rules_df.columns:
        return pd.DataFrame()
    return rules_df[rules_df["target_category"] == category].copy()


def explain_rule(rule_row: pd.Series) -> str:
    """
    Constructs a plain-language, academic diagnostic interpretation of an association rule.
    Separates statistical association from causality.
    """
    ant_str = rule_row["antecedents_str"]
    target = rule_row["target_category"] or rule_row["consequents_str"]
    supp = rule_row["support"]
    conf = rule_row["confidence"]
    lift = rule_row["lift"]

    text = (
        f"**Diagnostic Pattern**: When a query exhibits `[{ant_str}]`, "
        f"it results in `{target}` with **{conf*100:.1f}% confidence** "
        f"({supp*100:.1f}% corpus support). "
        f"This pattern is **{lift:.2f}x more likely** than baseline chance."
    )
    return text


# ---------------------------------------------------------------------------
# Plotly Visualizations (Section 6)
# ---------------------------------------------------------------------------

def plot_rule_support_confidence_scatter(rules_df: pd.DataFrame) -> go.Figure:
    """
    Interactive scatter plot of Support vs Confidence colored by Lift.
    Highlights high-confidence, high-lift diagnostic failure rules.
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Association Rules Available to Plot", template="plotly_white")
        return fig

    plot_df = rules_df.copy()
    plot_df["Category"] = plot_df["target_category"].fillna("General Pattern")
    plot_df["Hover"] = plot_df.apply(
        lambda r: f"<b>Rule:</b> {r['antecedents_str']} &rarr; {r['consequents_str']}<br>"
                  f"<b>Support:</b> {r['support']:.3f}<br>"
                  f"<b>Confidence:</b> {r['confidence']:.3f}<br>"
                  f"<b>Lift:</b> {r['lift']:.2f}",
        axis=1
    )

    fig = px.scatter(
        plot_df,
        x="support",
        y="confidence",
        size="lift",
        color="Category",
        hover_data=["Hover"],
        title="Association Rules: Support vs. Confidence (Size = Lift)",
        labels={"support": "Support (P(A ∩ C))", "confidence": "Confidence (P(C | A))"},
        template="plotly_white",
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(sizemin=6, sizemode="diameter")
    )
    fig.update_layout(height=480, legend_title_text="Target Category")
    return fig


def plot_rules_count_per_category(rules_df: pd.DataFrame) -> go.Figure:
    """Bar chart showing number of discovered association rules per failure category."""
    if rules_df.empty or "target_category" not in rules_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Category Rules Discovered", template="plotly_white")
        return fig

    cat_counts = rules_df["target_category"].value_counts().reset_index()
    cat_counts.columns = ["Failure Category", "Rule Count"]

    fig = px.bar(
        cat_counts,
        x="Failure Category",
        y="Rule Count",
        color="Failure Category",
        text="Rule Count",
        title="Discovered Diagnostic Rules by RAG Failure Category",
        template="plotly_white",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, showlegend=False)
    return fig


def plot_top_rules_bar(
    rules_df: pd.DataFrame,
    category: Optional[str] = None,
    top_n: int = 10,
    metric: str = "lift",
) -> go.Figure:
    """
    Horizontal bar chart of top diagnostic rules ranked by Lift or Confidence.
    """
    filtered = rules_df.copy()
    if category:
        filtered = filtered[filtered["target_category"] == category]

    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No Rules Match Filter Criteria", template="plotly_white")
        return fig

    top_rules = filtered.sort_values(by=metric, ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=top_rules[metric],
        y=[f"{a[:45]}..." if len(a) > 45 else a for a in top_rules["antecedents_str"]],
        orientation="h",
        marker_color="#2980B9",
        text=[f"{val:.2f}" for val in top_rules[metric]],
        textposition="outside",
    ))

    title_str = f"Top {top_n} Rules for '{category}' (by {metric.capitalize()})" if category else f"Top {top_n} Diagnostic Rules (by {metric.capitalize()})"
    fig.update_layout(
        title=title_str,
        xaxis_title=metric.capitalize(),
        yaxis_title="Feature Antecedents",
        template="plotly_white",
        height=450,
        margin=dict(l=150),
    )
    return fig


# ---------------------------------------------------------------------------
# Orchestrator Engine Class
# ---------------------------------------------------------------------------

class AprioriDiagnosticEngine:
    """
    High-level orchestrator for Phase 5 Association Rule Mining.
    Executes feature discretization, one-hot transaction encoding, Apriori
    frequent itemset mining, and diagnostic rule extraction.
    """

    def __init__(
        self,
        features_csv: Union[str, Path] = PROCESSED_FEATURES_CSV,
        min_support: float = DEFAULT_MIN_SUPPORT,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_lift: float = DEFAULT_MIN_LIFT,
        bins: int = NUMERICAL_DISCRETIZATION_BINS,
        discretization_strategy: str = "uniform",
        max_len: int = 3,
    ):
        self.features_csv = Path(features_csv)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.bins = bins
        self.discretization_strategy = discretization_strategy
        self.max_len = max_len

        # Cached artifacts
        self.df_raw: Optional[pd.DataFrame] = None
        self.discretized_df: Optional[pd.DataFrame] = None
        self.constant_cols_dropped: List[str] = []
        self.bin_edges: Dict[str, Any] = {}
        self.one_hot_df: Optional[pd.DataFrame] = None
        self.frequent_itemsets: Optional[pd.DataFrame] = None
        self.rules: Optional[pd.DataFrame] = None
        self.diagnostic_rules: Optional[pd.DataFrame] = None

    def fit(self, df_input: Optional[pd.DataFrame] = None) -> "AprioriDiagnosticEngine":
        """Executes the full discretization and rule-mining pipeline."""
        if df_input is not None:
            self.df_raw = df_input.copy()
        else:
            if not self.features_csv.exists():
                raise FileNotFoundError(f"Extracted features CSV not found at: {self.features_csv}")
            self.df_raw = pd.read_csv(self.features_csv)

        # 1. Discretize features
        self.discretized_df, self.constant_cols_dropped, self.bin_edges = discretize_features(
            self.df_raw,
            bins=self.bins,
            strategy=self.discretization_strategy,
            labels=DISCRETIZATION_LABELS[:self.bins],
            target_col="failure_label",
        )

        # 2. One-hot encode transactions
        self.one_hot_df = encode_transactions(self.discretized_df, target_col="failure_label")

        # 3. Mine frequent itemsets
        self.frequent_itemsets = mine_frequent_itemsets(
            self.one_hot_df,
            min_support=self.min_support,
            max_len=self.max_len,
        )

        # 4. Generate association rules
        self.rules = generate_diagnostic_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
            min_lift=self.min_lift,
        )

        # 5. Extract category-specific diagnostic rules
        if not self.rules.empty:
            self.diagnostic_rules = self.rules[self.rules["is_diagnostic_rule"] == True].copy()
        else:
            self.diagnostic_rules = pd.DataFrame()

        return self

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns KPI summary of itemset and rule counts."""
        n_itemsets = len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0
        n_rules = len(self.rules) if self.rules is not None else 0
        n_diag_rules = len(self.diagnostic_rules) if self.diagnostic_rules is not None else 0

        # Per category counts
        cat_counts = {}
        top_rule_per_cat = {}
        for cat in FAILURE_CATEGORIES:
            cat_r = filter_rules_by_category(self.rules, cat) if self.rules is not None else pd.DataFrame()
            cat_counts[cat] = len(cat_r)
            if not cat_r.empty:
                top_rule_per_cat[cat] = cat_r.iloc[0]["antecedents_str"]

        return {
            "total_samples": len(self.df_raw) if self.df_raw is not None else 0,
            "discretized_columns": len(self.discretized_df.columns) if self.discretized_df is not None else 0,
            "constant_columns_dropped": len(self.constant_cols_dropped),
            "one_hot_features": len(self.one_hot_df.columns) if self.one_hot_df is not None else 0,
            "frequent_itemsets_count": n_itemsets,
            "total_rules_count": n_rules,
            "diagnostic_rules_count": n_diag_rules,
            "category_rule_counts": cat_counts,
            "top_rule_per_category": top_rule_per_cat,
        }

    def export_rules_table(self, category: Optional[str] = None) -> pd.DataFrame:
        """Returns clean exportable DataFrame of diagnostic association rules."""
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()

        df_out = self.rules.copy()
        if category:
            df_out = df_out[df_out["target_category"] == category]

        return df_out[[
            "antecedents_str", "consequents_str", "target_category",
            "support", "confidence", "lift", "leverage", "conviction"
        ]].rename(columns={
            "antecedents_str": "Antecedent Pattern (Features)",
            "consequents_str": "Consequent (Outcome)",
            "target_category": "Failure Category",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
            "leverage": "Leverage",
            "conviction": "Conviction",
        })
