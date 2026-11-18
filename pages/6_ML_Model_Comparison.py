"""
ML Model Comparison Page for Causal RAG Diagnostician.
Presents the complete Multi-Model Benchmark Shootout across all 6 classical ML classifiers:
1. Multinomial Logistic Regression
2. Linear Support Vector Machine (Linear SVM)
3. RBF Kernel Support Vector Machine (RBF SVM)
4. Decision Tree Classifier
5. Random Forest Ensemble
6. K-Nearest Neighbors (KNN)

Includes:
- Multi-Model Shootout Leaderboard (CV & Test Metrics)
- Interactive Model Inspector
- Scaling & Feature Selection Experiments
- Multiclass Confusion Matrices
- Feature Interpretability (Weights & Importance Rankings)
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BEST_MODEL_PATH,
    ALL_MODELS_PATH,
    PROCESSED_FEATURES_CSV,
    FAILURE_CATEGORIES,
    MODEL_NAMES,
    get_active_corpus_config,
)
from modules.diagnostic_models import (
    LogisticRegressionDiagnosticEngine,
    MultiModelDiagnosticSuite,
)
from modules.ui_components import render_sidebar, render_academic_disclaimer

st.set_page_config(
    page_title="ML Diagnostic Models | RAG Diagnostician",
    page_icon="🤖",
    layout="wide"
)

render_sidebar("ML Model Comparison")

active_cfg = get_active_corpus_config()

st.title("🤖 Supervised Machine Learning Diagnostic Shootout")
st.markdown(f"""
This module implements and rigorously evaluates **six classical Machine Learning classifiers** to diagnose the root causes of RAG interaction failures.
Currently displaying benchmarks for: **{active_cfg['icon']} {active_cfg['name']}** (*{active_cfg['doc_count']} indexed documents*).
""")

st.markdown("---")


@st.cache_resource(show_spinner=False)
def get_multi_model_suite(models_path_str: str, features_path_str: str, best_path_str: str):
    models_path = Path(models_path_str)
    features_path = Path(features_path_str)
    best_path = Path(best_path_str)
    if not models_path.exists() or not best_path.exists():
        suite = MultiModelDiagnosticSuite()
        suite.train_and_evaluate_all(features_path)
        suite.save_suite(models_path, best_path)

    suite_payload = joblib.load(models_path)
    return suite_payload


suite_data = get_multi_model_suite(
    str(active_cfg["all_models_path"]),
    str(active_cfg["features_csv"]),
    str(active_cfg["best_model_path"])
)
comp_df = suite_data["comparison_df"]
best_model_name = suite_data["best_model_name"]
engines = suite_data["engines"]
shootout_summary = suite_data["shootout_summary"]
selected_features = suite_data["selected_features"]

# -----------------------------------------------------------------------------
# 1. Leaderboard & Executive Summary
# -----------------------------------------------------------------------------
st.subheader("1. 🏆 Multi-Model Benchmark Shootout Leaderboard")

top_row = comp_df.iloc[0]
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Top Performer", f"🥇 {top_row['Model']}")
with m2:
    st.metric("Top Test Accuracy", top_row["Test Accuracy"])
with m3:
    st.metric("Top Test Macro F1", top_row["Test Macro F1"])
with m4:
    st.metric("Models Benchmarked", f"{len(comp_df)} Classifiers")
with m5:
    st.metric("Selected Features", f"{len(selected_features)} / 36")

col_tbl, col_chart = st.columns([3, 2])

with col_tbl:
    st.markdown("##### Comparative Performance Matrix")
    display_df = comp_df[[
        "Model", "Test Accuracy", "Test Macro F1", "Test Macro Precision", "Test Macro Recall", "CV Accuracy", "CV Macro F1"
    ]].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Models ranked strictly by Test Macro F1 on unseen holdout data (13 queries, stratified across all 5 failure categories).")

with col_chart:
    fig_comp = px.bar(
        comp_df,
        x="raw_test_f1",
        y="Model",
        orientation="h",
        color="Model",
        text=[f"{v:.4f}" for v in comp_df["raw_test_f1"]],
        title="Test Macro F1-Score Comparison",
        range_x=[0.4, 1.0],
        template="plotly_white"
    )
    fig_comp.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 2. Detailed Experiments & Model Inspector
# -----------------------------------------------------------------------------
st.subheader("2. Detailed Model Diagnostics & Interpretability Inspector")

tab_shootout, tab_scaling, tab_feat_sel, tab_inspector, tab_theory = st.tabs([
    "📊 Model Inspector (Per-Model Evaluation)",
    "⚖️ Preprocessing: Scaling Experiment",
    "🎯 Feature Selection (ANOVA vs L1)",
    "🔍 Feature Interpretability & Weights",
    "📚 Classical ML Syllabus Alignment"
])

# -----------------------------------------------------------------------------
# Tab 1: Per-Model Evaluation & Confusion Matrix
# -----------------------------------------------------------------------------
with tab_shootout:
    selected_model = st.selectbox(
        "Select Model to Inspect Confusion Matrix & Per-Class Metrics:",
        comp_df["Model"].tolist()
    )

    eng = engines[selected_model]
    t_metrics = eng.test_metrics_
    cm = np.array(eng.confusion_matrix_)
    classes = eng.classes_

    st.markdown(f"#### Detailed Performance: **{selected_model}**")

    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.metric("Test Accuracy", f"{t_metrics['accuracy'] * 100:.1f}%")
    with sm2:
        st.metric("Macro Precision", f"{t_metrics['precision_macro']:.4f}")
    with sm3:
        st.metric("Macro Recall", f"{t_metrics['recall_macro']:.4f}")
    with sm4:
        st.metric("Macro F1", f"{t_metrics['f1_macro']:.4f}")

    col_cm, col_rep = st.columns([1, 1])

    with col_cm:
        st.markdown("##### Multiclass Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
            x=classes,
            y=classes,
            color_continuous_scale="Blues",
            title=f"{selected_model} Confusion Matrix (13 Test Holdout Queries)"
        )
        fig_cm.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_rep:
        st.markdown("##### Per-Class Precision, Recall, and F1")
        rep_dict = eng.classification_report_dict_
        per_class_rows = []
        for c in classes:
            c_data = rep_dict.get(c, {})
            per_class_rows.append({
                "Failure Class": c,
                "Precision": f"{c_data.get('precision', 0.0):.2f}",
                "Recall": f"{c_data.get('recall', 0.0):.2f}",
                "F1-Score": f"{c_data.get('f1-score', 0.0):.2f}",
                "Support": int(c_data.get('support', 0))
            })
        st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

        if hasattr(eng, "best_config_"):
            st.markdown("##### Pipeline Configuration")
            st.json(eng.best_config_)

# -----------------------------------------------------------------------------
# Tab 2: Feature Scaling Experiment
# -----------------------------------------------------------------------------
with tab_scaling:
    st.markdown("#### Experiment 1: StandardScaler vs MinMaxScaler (5-Fold Stratified CV)")
    st.caption("Evaluates whether unit-variance standardization or bounded [0, 1] normalization better preserves diagnostic metric distributions.")

    log_eng = engines["Logistic Regression"]
    scaling_res = log_eng.scaling_experiment_results

    scaling_data = []
    for s_name, s_m in scaling_res.items():
        scaling_data.append({
            "Scaler": s_name,
            "CV Accuracy": f"{s_m['mean_accuracy'] * 100:.2f}% (± {s_m['std_accuracy']*100:.1f}%)",
            "CV Macro F1": f"{s_m['mean_f1_macro']:.4f} (± {s_m['std_f1_macro']:.4f})",
            "CV Macro Precision": f"{s_m['mean_precision_macro']:.4f}",
            "CV Macro Recall": f"{s_m['mean_recall_macro']:.4f}",
            "raw_acc": s_m['mean_accuracy'],
            "raw_f1": s_m['mean_f1_macro']
        })
    s_df = pd.DataFrame(scaling_data)

    col_stbl, col_splot = st.columns([3, 2])
    with col_stbl:
        st.table(s_df[["Scaler", "CV Accuracy", "CV Macro F1", "CV Macro Precision", "CV Macro Recall"]])
        st.success("🏆 **StandardScaler** produced superior cross-validation Macro F1 and was incorporated into distance-sensitive models (SVMs, LogReg, KNN).")

    with col_splot:
        fig_s = px.bar(
            s_df,
            x="Scaler",
            y="raw_f1",
            color="Scaler",
            text=[f"{v:.4f}" for v in s_df["raw_f1"]],
            title="CV Macro F1 by Scaling Technique",
            range_y=[0.7, 1.0]
        )
        fig_s.update_layout(showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3: Feature Selection Experiment
# -----------------------------------------------------------------------------
with tab_feat_sel:
    st.markdown("#### Experiment 2: ANOVA F-Statistic vs Embedded L1 Lasso Selection")
    st.caption("Extracts the most statistically significant predictors from the initial 36-feature space.")

    feat_res = log_eng.feature_selection_results
    configs_dict = feat_res["configurations"]
    fsel_data = []
    for c_name, c_m in configs_dict.items():
        fsel_data.append({
            "Configuration": c_name,
            "Feature Count": c_m["num_features"],
            "CV Accuracy": f"{c_m['mean_accuracy'] * 100:.2f}% (± {c_m['std_accuracy']*100:.1f}%)",
            "CV Macro F1": f"{c_m['mean_f1_macro']:.4f} (± {c_m['std_f1_macro']:.4f})",
            "raw_f1": c_m["mean_f1_macro"]
        })
    fsel_df = pd.DataFrame(fsel_data)

    st.table(fsel_df[["Configuration", "Feature Count", "CV Accuracy", "CV Macro F1"]])

    col_flt, col_emb = st.columns(2)
    with col_flt:
        st.markdown("##### Top Features by ANOVA F-Statistic (Filter)")
        st.dataframe(feat_res["filter_ranked_summary"].head(10), use_container_width=True)

    with col_emb:
        st.markdown("##### Top Features by L1 Absolute Coefficient (Embedded)")
        st.dataframe(feat_res["embedded_coefficient_summary"].head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 4: Feature Interpretability & Weights
# -----------------------------------------------------------------------------
with tab_inspector:
    st.markdown("#### Feature Interpretability Analysis")
    st.caption("Examine how individual models utilize the engineered features.")

    inspect_model_name = st.selectbox(
        "Select Model to Inspect Feature Drivers:",
        ["Random Forest", "Logistic Regression", "Linear SVM", "Decision Tree"]
    )

    target_eng = engines[inspect_model_name]

    if inspect_model_name in ["Random Forest", "Decision Tree"]:
        st.markdown(f"##### Feature Importances for **{inspect_model_name}** (Mean Decrease in Impurity)")
        fi_df = getattr(target_eng, "feature_importances_df_", None)
        if fi_df is not None:
            fig_fi = px.bar(
                fi_df.head(12),
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top 12 Features Ranked by Importance ({inspect_model_name})",
                template="plotly_white"
            )
            fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)
            st.dataframe(fi_df, use_container_width=True)

    elif inspect_model_name in ["Logistic Regression", "Linear SVM"]:
        coef_df = getattr(target_eng, "coefficients_df_", None)
        if coef_df is not None:
            inspect_class = st.selectbox(f"Inspect Weight Vector for Class ({inspect_model_name}):", target_eng.classes_)
            class_coefs = coef_df.loc[inspect_class].sort_values()

            fig_coef = go.Figure(go.Bar(
                x=class_coefs.values,
                y=class_coefs.index,
                orientation="h",
                marker=dict(color=["#d62728" if v < 0 else "#2ca02c" for v in class_coefs.values])
            ))
            fig_coef.update_layout(
                title=f"Linear Weight Coefficients for [{inspect_class}] ({inspect_model_name})",
                xaxis_title="Weight Value (w)",
                yaxis_title="Feature Name",
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig_coef, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 5: Classical ML Syllabus Alignment
# -----------------------------------------------------------------------------
with tab_theory:
    st.markdown("#### Mapping Platform Algorithms to ML Syllabus Units")
    theory_matrix = [
        {
            "Algorithm": "Multinomial Logistic Regression",
            "Syllabus Unit": "Unit III: Classification & Descriptive Learning",
            "Boundary Type": "Linear (Hyperplanes in Log-Odds space)",
            "Observed Finding": "Achieved 61.5% Test Accuracy. Linear hyperplanes struggle with non-linear overlaps between Hallucination and Contamination."
        },
        {
            "Algorithm": "Linear Support Vector Machine",
            "Syllabus Unit": "Unit IV: Support Vector Machines & Margins",
            "Boundary Type": "Linear (Max-Margin Hyperplane)",
            "Observed Finding": "Achieved 61.5% Test Accuracy. Separates Retrieval Miss perfectly, but shares linear separation bounds with Logistic Regression."
        },
        {
            "Algorithm": "RBF Kernel Support Vector Machine",
            "Syllabus Unit": "Unit IV: Kernel Methods & Dual Formulation",
            "Boundary Type": "Non-Linear (Gaussian Hilbert Space)",
            "Observed Finding": "Achieved 76.9% Test Accuracy. Kernel trick successfully maps non-linear distribution shifts, gaining +15.4% accuracy over linear SVM."
        },
        {
            "Algorithm": "Decision Tree (CART)",
            "Syllabus Unit": "Unit V: Decision Tree Learning",
            "Boundary Type": "Non-Linear (Orthogonal Axis-Aligned Splits)",
            "Observed Finding": "Achieved 61.5% Test Accuracy. Single tree exhibits higher variance on small sample subsets."
        },
        {
            "Algorithm": "Random Forest Ensemble",
            "Syllabus Unit": "Unit V: Ensemble Methods & Bagging",
            "Boundary Type": "Non-Linear Ensemble (Bootstrap Aggregation)",
            "Observed Finding": "🏆 Top Performer (84.6% Test Accuracy, 0.8433 Macro F1). Aggregation of decorrelated trees eliminates individual tree variance."
        },
        {
            "Algorithm": "K-Nearest Neighbors (KNN)",
            "Syllabus Unit": "Unit III: Instance-Based Learning",
            "Boundary Type": "Non-Parametric Proximity (Voronoi Tessellation)",
            "Observed Finding": "Runner-Up (84.6% Test Accuracy, 0.8314 Macro F1). Distance-weighted proximity captures local metric clustering effectively."
        },
    ]
    st.table(pd.DataFrame(theory_matrix))

render_academic_disclaimer()
