"""
Failure Diagnosis Page for Causal RAG Diagnostician.
Applies trained supervised diagnostic models to diagnose RAG interaction queries.
Supports selecting any of the 6 benchmarked classifiers or an Ensemble Voting consensus,
reports calibrated confidence distributions, compares multi-model consensus, and delivers
actionable corrective recommendations.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BEST_MODEL_PATH,
    ALL_MODELS_PATH,
    PROCESSED_FEATURES_CSV,
    FAILURE_CATEGORIES,
    get_active_corpus_config,
)
from modules.ui_components import render_sidebar, render_academic_disclaimer

st.set_page_config(
    page_title="Failure Diagnosis | RAG Diagnostician",
    page_icon="🩺",
    layout="wide"
)

render_sidebar("Interactive Failure Diagnosis")

active_cfg = get_active_corpus_config()

st.title("🩺 Interactive Multi-Model Failure Diagnosis Console")
st.markdown(f"""
Diagnose why an interaction failed using your choice of **six supervised Machine Learning models** or an **Ensemble Consensus**.
Active Domain: **{active_cfg['icon']} {active_cfg['name']}** (*{active_cfg['doc_count']} indexed documents*).
""")

st.markdown("---")

all_models_file = active_cfg["all_models_path"]
features_file = active_cfg["features_csv"]
best_model_file = active_cfg["best_model_path"]

if not best_model_file.exists() or not features_file.exists():
    st.error(f"Model artifact or feature dataset missing for {active_cfg['name']}. Please run pipeline first.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_all_models_artifact(path_str: str):
    p = Path(path_str)
    if p.exists():
        return joblib.load(p)
    return None


@st.cache_resource(show_spinner=False)
def load_best_model_artifact(path_str: str):
    return joblib.load(Path(path_str))


@st.cache_data(show_spinner=False)
def load_features_dataset(path_str: str):
    return pd.read_csv(Path(path_str))


all_models_data = load_all_models_artifact(str(all_models_file))
best_artifact = load_best_model_artifact(str(best_model_file))
features_df = load_features_dataset(str(features_file))

# Available models
if all_models_data and "engines" in all_models_data:
    engines = all_models_data["engines"]
    model_choices = list(engines.keys()) + ["⚡ Multi-Model Ensemble Consensus"]
    default_model = all_models_data.get("best_model_name", model_choices[0])
    selected_features = all_models_data["selected_features"]
    classes = FAILURE_CATEGORIES
else:
    engines = {"Active Model": best_artifact["pipeline"]}
    model_choices = ["Active Model"]
    default_model = "Active Model"
    selected_features = best_artifact["feature_names"]
    classes = best_artifact["classes"]

# -----------------------------------------------------------------------------
# 1. Query & Model Selection
# -----------------------------------------------------------------------------
st.subheader("1. Select Query & Diagnostic Model")

col_q, col_m = st.columns([3, 2])

with col_q:
    query_labels = [
        f"{row['query_id']} | [{row['failure_label']}] {row['query']}"
        for _, row in features_df.iterrows()
    ]
    selected_query_str = st.selectbox("Choose a Query from Extracted Feature Dataset:", query_labels)
    selected_qid = selected_query_str.split(" | ")[0].strip()
    query_row = features_df[features_df["query_id"] == selected_qid].iloc[0]

with col_m:
    selected_model_name = st.selectbox(
        "Choose Diagnostic Model for Evaluation:",
        model_choices,
        index=model_choices.index(default_model) if default_model in model_choices else 0
    )

# -----------------------------------------------------------------------------
# 2. Query Metadata & Retrieval Metrics
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. Query Metadata & Retrieval Context")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Query ID:** `{query_row['query_id']}`")
    st.info(f"❓ **Query:** {query_row['query']}")
    st.markdown(f"**Ground Truth Label:** `{query_row['failure_label']}`")
with c2:
    st.markdown(f"**Top-5 Mean Distance:** `{query_row['top5_dist_mean']:.4f}`")
    st.markdown(f"**Depth Distance Shift:** `{query_row['dist_diff_deep_vs_top5']:+.4f}`")
    st.markdown(f"**Generated vs Top-5 Jaccard:** `{query_row['jaccard_generated_vs_top5']:.4f}`")

# -----------------------------------------------------------------------------
# 3. Model Prediction & Probabilities
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. Model Prediction & Diagnostic Confidence")

X_input = pd.DataFrame([query_row[selected_features]])

# Compute predictions for all available models
model_predictions = {}
model_probabilities = {}

if all_models_data and "engines" in all_models_data:
    for m_name, m_eng in engines.items():
        try:
            m_pred = m_eng.predict(X_input)[0]
            m_prob = m_eng.predict_proba(X_input)[0]
            model_predictions[m_name] = m_pred
            model_probabilities[m_name] = m_prob
        except Exception:
            pass

if selected_model_name == "⚡ Multi-Model Ensemble Consensus":
    # Average probabilities across all 6 models
    avg_probs = np.mean(list(model_probabilities.values()), axis=0)
    pred_class = classes[int(np.argmax(avg_probs))]
    probs = avg_probs
else:
    active_engine = engines[selected_model_name]
    pred_class = active_engine.predict(X_input)[0]
    probs = active_engine.predict_proba(X_input)[0]

prob_df = pd.DataFrame({
    "Failure Category": classes,
    "Probability": probs,
    "Confidence (%)": (probs * 100).round(2)
}).sort_values(by="Probability", ascending=False).reset_index(drop=True)

col_pred, col_chart = st.columns([2, 3])

with col_pred:
    is_correct = (pred_class == query_row["failure_label"])
    badge = "✅ MATCHES GROUND TRUTH" if is_correct else "⚠️ DISCREPANCY"
    st.success(f"### Predicted Diagnosis: **{pred_class}**\n*{badge}*")

    top_prob = prob_df.iloc[0]["Confidence (%)"]
    st.metric("Model Confidence", f"{top_prob:.1f}%")
    st.caption(f"Evaluated using: **{selected_model_name}**")

with col_chart:
    fig_prob = px.bar(
        prob_df,
        x="Probability",
        y="Failure Category",
        orientation="h",
        color="Failure Category",
        text=[f"{v:.1f}%" for v in prob_df["Confidence (%)"]],
        title=f"Probability Distribution ({selected_model_name})",
        range_x=[0, 1.0],
        template="plotly_white"
    )
    fig_prob.update_layout(showlegend=False)
    st.plotly_chart(fig_prob, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Multi-Model Consensus Grid
# -----------------------------------------------------------------------------
if model_predictions:
    st.markdown("---")
    st.subheader("4. 🗳️ Multi-Model Consensus Grid Across All 6 Classifiers")
    st.caption("Compare how all six algorithms diagnose this exact query:")

    consensus_rows = []
    for m_name, m_pred in model_predictions.items():
        m_prob_max = np.max(model_probabilities[m_name]) * 100
        matches = "✅ Correct" if m_pred == query_row["failure_label"] else "❌ Differed"
        consensus_rows.append({
            "Classifier Model": m_name,
            "Predicted Failure Mode": m_pred,
            "Confidence": f"{m_prob_max:.1f}%",
            "Ground Truth Check": matches
        })
    st.dataframe(pd.DataFrame(consensus_rows), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# 5. Actionable Corrective Recommendations
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. 🔧 Actionable Engineering Recommendations")

RECOMMENDATIONS = {
    "Retrieval Miss": [
        "**Expand Document Corpus**: The needed evidence is absent from the indexed database.",
        "**Implement Web Search / Live Fallback**: Integrate hybrid web-search fallbacks when distance exceeds 0.75.",
        "**Audit Index Ingestion**: Check whether entire document partitions were omitted during data indexing."
    ],
    "Ranking Error": [
        "**Hybrid Search**: Integrate sparse BM25 lexical search with dense vector embeddings to elevate exact keyword matches.",
        "**Deploy Cross-Encoder Reranker**: Use a reranker (e.g. BGE-Reranker or Cohere) on the Top-50 candidates before trimming to Top-5.",
        "**Tune Retrieval Horizon**: Expand operational context window from Top-5 to Top-10 chunks."
    ],
    "Context Contamination": [
        "**Adjust Chunk Size & Overlap**: Overly large chunks dilute semantic density with peripheral topics.",
        "**Semantic Filtering / Deduplication**: Eliminate distractor documents with divergent domain metadata before prompt assembly.",
        "**Context Compression**: Extract only relevant query-conditioned sentences rather than whole chunks."
    ],
    "Hallucination": [
        "**Constrain Generation Prompting**: Enforce strict system instructions: 'Answer strictly using retrieved context only.'",
        "**Lower Temperature**: Set generation temperature $\le 0.2$ to reduce ungrounded stochastic completions.",
        "**Faithfulness Verification Filter**: Check generated answer against retrieved chunks before serving response."
    ],
    "Contradiction": [
        "**Corpus Reconciliation**: Resolve opposing claims in conflicting documents (e.g. outdated vs updated revisions).",
        "**Source Authority Metadata**: Weight or filter retrieval by publication date, version timestamp, or publisher authority.",
        "**Multi-Perspective Synthesis**: Instruct model: 'When retrieved sources conflict, state both perspectives explicitly.'"
    ]
}

recs = RECOMMENDATIONS.get(pred_class, ["Inspect retrieval pipeline configuration."])
for rec in recs:
    st.markdown(f"• {rec}")

render_academic_disclaimer()
