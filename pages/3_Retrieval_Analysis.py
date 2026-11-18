"""
Retrieval Analysis Page for Causal RAG Failure Diagnostician.
Allows interactive inspection of Top-5 vs Top-100 retrieved chunks,
distance curves, document ranks, and expected vs generated answers.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
    DETAILED_RETRIEVAL_CSV,
    TOP_K_BASELINE,
    TOP_K_DEEP_FETCH,
)
from modules.chroma_connector import ChromaConnector
from modules.data_ingestion import load_csv
from modules.ui_components import render_sidebar, render_academic_disclaimer

st.set_page_config(
    page_title="Retrieval Analysis | RAG Diagnostician",
    page_icon="🔍",
    layout="wide"
)

render_sidebar("Retrieval Deep-Fetch Inspector")

st.title("🔍 Retrieval Analysis & Deep-Fetch Inspector")
st.markdown("""
Inspect the vector retrieval behavior for any failed RAG query. Compare the operational 
**Top-5 baseline chunks** against the extended **Top-100 deep-fetch retrieval** to diagnose 
ranking demotions, distractor noise, and database coverage gaps.
""")

st.markdown("---")

# Load queries dataset
df_queries = None
if "validated_df" in st.session_state and not st.session_state["validated_df"].empty:
    df_queries = st.session_state["validated_df"]
else:
    sample_csv = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
    if sample_csv.exists():
        report = load_csv(sample_csv)
        df_queries = report.cleaned_df
        st.session_state["validated_df"] = df_queries

if df_queries is None or df_queries.empty:
    st.warning("No queries available. Please visit the **Data Ingestion** page first.")
    st.stop()

# Cached ChromaDB Connector
@st.cache_resource(show_spinner=False)
def get_chroma_connector():
    return ChromaConnector(persist_dir=CHROMA_PERSIST_DIR)

try:
    connector = get_chroma_connector()
except Exception as e:
    st.error(f"Failed to connect to ChromaDB: {e}")
    st.stop()

# 1. Query Selection
st.subheader("1. Select Failed Query for Analysis")

# Filter by failure category
categories = ["All Categories"] + sorted(df_queries["failure_label"].dropna().unique().tolist())
selected_cat = st.selectbox("Filter by Failure Category:", categories)

filtered_df = df_queries if selected_cat == "All Categories" else df_queries[df_queries["failure_label"] == selected_cat]

query_options = [
    f"{row['query_id']} | [{row.get('failure_label', 'Unlabeled')}] {row['query']}"
    for _, row in filtered_df.iterrows()
]

selected_query_str = st.selectbox("Choose Query:", query_options)
selected_qid = selected_query_str.split(" | ")[0].strip()
selected_row = df_queries[df_queries["query_id"] == selected_qid].iloc[0]

# Display Query Details Card
st.markdown("---")
st.subheader("2. Interaction Context & Ground Truth")

q_col1, q_col2 = st.columns(2)

with q_col1:
    st.markdown(f"**Query ID:** `{selected_row.get('query_id', 'N/A')}`")
    st.markdown(f"**Query Text:**")
    st.info(f"❓ {selected_row['query']}")
    st.markdown(f"**Failure Category:** `{selected_row.get('failure_label', 'Unlabeled')}`")
    if "notes" in selected_row and pd.notna(selected_row["notes"]):
        st.caption(f"**Annotator Diagnosis:** {selected_row['notes']}")

with q_col2:
    st.markdown(f"**Expected Answer (Ground Truth):**")
    st.success(f"🎯 {selected_row.get('expected_answer', 'Not Provided')}")
    st.markdown(f"**Generated Answer (RAG Model Output):**")
    st.warning(f"🤖 {selected_row.get('generated_answer', 'Not Provided')}")
    target_id = str(selected_row.get("target_doc_id", "NONE"))
    st.caption(f"**Target Corpus Document ID:** `{target_id}`")

# 2. Execute Retrieval (Cached by query and depth)
@st.cache_data(show_spinner=False)
def fetch_retrieval_records(query_text: str, top_k: int):
    conn = get_chroma_connector()
    return conn.retrieve_query(
        collection_name=DEFAULT_COLLECTION_NAME,
        query_text=query_text,
        top_k=top_k
    )

with st.spinner("Executing Top-100 deep-fetch retrieval from ChromaDB..."):
    retrieval_records = fetch_retrieval_records(selected_row["query"], TOP_K_DEEP_FETCH)

if not retrieval_records:
    st.error("No documents retrieved from ChromaDB.")
    st.stop()

top_5, ranks_6_100, all_results = connector.partition_retrieval_results(retrieval_records, TOP_K_BASELINE)

# Check target document rank
target_rank = None
if target_id and target_id != "NONE":
    for r in all_results:
        if r["doc_id"] == target_id:
            target_rank = r["rank"]
            break

# Metric highlights
st.markdown("---")
st.subheader("3. Retrieval Metrics Overview")

top_5_dists = [r["distance"] for r in top_5]
deep_dists = [r["distance"] for r in ranks_6_100]

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Top-5 Mean Distance", f"{np.mean(top_5_dists):.4f}")
with m2:
    st.metric("Top-5 Min Distance (Rank 1)", f"{np.min(top_5_dists):.4f}")
with m3:
    st.metric("Ranks 6–100 Mean Distance", f"{np.mean(deep_dists):.4f}" if deep_dists else "N/A")
with m4:
    diff_val = (np.mean(deep_dists) - np.mean(top_5_dists)) if deep_dists else 0.0
    st.metric("Depth Distance Shift", f"+{diff_val:.4f}")
with m5:
    if target_rank is not None:
        rank_badge = f"Rank #{target_rank}"
        if target_rank <= 5:
            st.metric("Target Document Rank", rank_badge, delta="Inside Top-5", delta_color="normal")
        else:
            st.metric("Target Document Rank", rank_badge, delta="Ranking Error (>5)", delta_color="inverse")
    else:
        st.metric("Target Document Rank", "Not Found (Miss)", delta="Absent from Corpus", delta_color="inverse")

# Diagnostic Alert based on retrieval
if target_rank is not None and target_rank > 5:
    st.warning(
        f"🚨 **Ranking Error Detected**: The target evidence (`{target_id}`) was found in the database at **Rank #{target_rank}**, "
        f"but was missed because the baseline retrieval cut off at Top-5!"
    )
elif target_id != "NONE" and target_rank is None:
    st.error(
        f"🚨 **Retrieval Miss / Database Gap**: The target evidence was not found even after deep-fetching 100 documents. "
        f"The necessary information is absent from the indexed corpus."
    )
elif target_id != "NONE" and target_rank <= 5:
    if selected_row.get("failure_label") == "Hallucination":
        st.info("💡 **Retrieved Context is Present**: Evidence was retrieved in Top-5, but the generation model hallucinated unbacked claims.")
    elif selected_row.get("failure_label") == "Contradiction":
        st.info("💡 **Contradiction in Context**: Top-5 chunks contain mutually conflicting source claims.")
    elif selected_row.get("failure_label") == "Context Contamination":
        st.info("💡 **Context Contamination**: Top-5 chunks contain distracting polysemous terms.")

# 4. Interactive Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 Top-5 Baseline Chunks",
    "📚 Deep-Fetch Chunks (Ranks 6–100)",
    "📈 Ranked Distance Curve",
    "📊 Complete Tabular Data"
])

with tab1:
    st.markdown("#### Operational Top-5 Chunks Supplied to Generator")
    for r in top_5:
        is_target = (r["doc_id"] == target_id)
        border_color = "🟢" if is_target else "📄"
        with st.container():
            st.markdown(
                f"**{border_color} Rank #{r['rank']} | Document ID: `{r['doc_id']}`** | "
                f"Cosine Distance: `{r['distance']:.4f}` | Cosine Similarity: `{r['similarity_score']:.4f}` | "
                f"Domain: `{r['metadata'].get('domain', 'General')}`"
            )
            st.info(r["document"])

with tab2:
    st.markdown("#### Deep-Fetch Retrieval (Ranks 6 to 100)")
    st.caption("Inspect documents retrieved deeper in the index. Use the slider to browse chunks.")
    
    rank_range = st.slider("Select Rank Window:", min_value=6, max_value=len(all_results), value=(6, 25))
    selected_deep_chunks = [r for r in ranks_6_100 if rank_range[0] <= r["rank"] <= rank_range[1]]

    for r in selected_deep_chunks:
        is_target = (r["doc_id"] == target_id)
        if is_target:
            st.success(
                f"🎯 **TARGET DOCUMENT FOUND AT RANK #{r['rank']} | ID: `{r['doc_id']}`** | "
                f"Cosine Distance: `{r['distance']:.4f}` | Domain: `{r['metadata'].get('domain', 'General')}`"
            )
            st.markdown(f"**Content:** {r['document']}")
        else:
            st.markdown(
                f"**Rank #{r['rank']} | ID: `{r['doc_id']}`** | Distance: `{r['distance']:.4f}` | Domain: `{r['metadata'].get('domain', 'General')}`"
            )
            st.text(r["document"])
        st.markdown("---")

with tab3:
    st.markdown("#### Ranked Distance Curve (Top-1 to Top-100)")
    st.caption("Visualizes distance progression across retrieval depth. Notice distance plateauing or sudden drop-offs.")
    
    curve_df = pd.DataFrame(all_results)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["rank"],
        y=curve_df["distance"],
        mode="lines+markers",
        name="Cosine Distance",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4)
    ))
    
    # Add vertical line for Top-5 cutoff
    fig.add_vline(x=5.5, line_dash="dash", line_color="red", annotation_text="Top-5 Baseline Cutoff", annotation_position="top right")
    
    # Highlight target document if found
    if target_rank is not None:
        target_dist = curve_df.loc[curve_df["rank"] == target_rank, "distance"].values[0]
        fig.add_trace(go.Scatter(
            x=[target_rank],
            y=[target_dist],
            mode="markers+text",
            name="Target Document",
            text=[f"Target (Rank #{target_rank})"],
            textposition="top center",
            marker=dict(size=14, color="gold", symbol="star")
        ))

    fig.update_layout(
        title=f"Retrieval Distance vs Rank Curve for Query: '{selected_row['query'][:60]}...'",
        xaxis_title="Retrieval Rank (1 to 100)",
        yaxis_title="Cosine Distance",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("#### Full Retrieval Table (100 Chunks)")
    full_table_df = pd.DataFrame(all_results)
    st.dataframe(full_table_df, use_container_width=True)
    
    csv_bytes = full_table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Query Retrieval Chunks (CSV)",
        data=csv_bytes,
        file_name=f"retrieval_{selected_qid}.csv",
        mime="text/csv"
    )

render_academic_disclaimer()
