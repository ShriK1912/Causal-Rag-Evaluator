"""
Deep-Fetch Statistical Analyzer Module for Causal RAG Failure Diagnostician.

Performs rigorous statistical analysis on retrieval distance distributions,
comparing Top-5 operational retrieval against Ranks 6-100 deep-fetch retrieval.
Calculates Mean, Median, Variance, Standard Deviation, Min, Max, Spread, IQR,
and distance drop-offs. Generates interpretable diagnostic insights with strict
distinction between statistical association and causality.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TOP_K_BASELINE,
    TOP_K_DEEP_FETCH,
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    DETAILED_RETRIEVAL_CSV,
)
from modules.chroma_connector import ChromaConnector
from modules.feature_extraction import (
    compute_jaccard_similarity,
    compute_rouge_l,
    tokenize_text,
)


def compute_distribution_metrics(distances: List[float], prefix: str = "dist") -> Dict[str, float]:
    """
    Computes parametric and non-parametric descriptive statistics for a distance list:
    Mean, Median, Variance, Standard Deviation, Minimum, Maximum, Spread,
    Interquartile Range (IQR), Skewness, and Euclidean L2 equivalents.

    Mathematical Verification & Preconditions:
      - Parametric:
          Mean (mu) = (1/n) * sum(x_i)
          Variance (sigma^2) = (1/n) * sum((x_i - mu)^2)
          Standard Deviation (sigma) = sqrt(sigma^2)
      - Non-Parametric (Robust):
          Median = 50th percentile (robust to extreme retrieval outliers)
          Spread = max(x) - min(x)
          IQR = Q3 - Q1 (75th percentile - 25th percentile)
          Skewness = Fisher-Pearson coefficient g1 = m3 / m2^(3/2)
      - Euclidean L2 Equivalent:
          Under the geometric identity for unit-normalized vectors (||u||_2 = ||v||_2 = 1):
              ||u - v||_2^2 = ||u||_2^2 + ||v||_2^2 - 2(u . v)
                            = 1 + 1 - 2*cos(theta)
                            = 2 * (1 - cos(theta))
                            = 2 * d_cosine(u, v)
              ||u - v||_2 = sqrt(2 * d_cosine(u, v))
          IMPORTANT: This formula strictly holds when embeddings are L2-normalized
          (such as ChromaDB's default sentence-transformers 'all-MiniLM-L6-v2' embeddings).
          For arbitrary unnormalized embeddings, this relationship does not directly hold
          without scaling by ||u||_2 * ||v||_2.
    """
    if not distances:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_var": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_spread": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_l2_mean_equiv": np.nan,
        }

    arr = np.array(distances, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "median", "var", "std", "min", "max", "spread", "iqr", "skew", "l2_mean_equiv"]}

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    var_val = float(np.var(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    spread_val = max_val - min_val

    q75, q25 = np.percentile(arr, [75, 25])
    iqr_val = float(q75 - q25)

    skew_val = float(stats.skew(arr)) if len(arr) >= 3 else 0.0
    l2_equiv = float(np.sqrt(max(0.0, 2.0 * mean_val)))

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_median": median_val,
        f"{prefix}_var": var_val,
        f"{prefix}_std": std_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_spread": spread_val,
        f"{prefix}_iqr": iqr_val,
        f"{prefix}_skew": skew_val,
        f"{prefix}_l2_mean_equiv": l2_equiv,
    }


def compute_pairwise_top5_coherence(top5_docs: List[str]) -> float:
    """
    Calculates the average pairwise Jaccard token overlap among the Top-5 retrieved chunks.
    High coherence (e.g. > 0.3) indicates semantic consensus.
    Low coherence (e.g. < 0.1) indicates semantic distraction or context contamination.
    """
    if len(top5_docs) < 2:
        return 0.0

    similarities = []
    for i in range(len(top5_docs)):
        for j in range(i + 1, len(top5_docs)):
            sim = compute_jaccard_similarity(top5_docs[i], top5_docs[j])
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def compute_shannon_entropy(distances: List[float], n_bins: int = 10) -> float:
    """
    Optional Advanced Feature: Calculates Shannon Entropy of the distance histogram.
    Measures dispersion/uniformity across retrieval tiers.
    """
    if len(distances) < 2:
        return 0.0
    counts, _ = np.histogram(distances, bins=n_bins, density=True)
    counts = counts[counts > 0]
    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log2(probs)))


def compute_elbow_rank(distances: List[float]) -> int:
    """
    Optional Advanced Feature: Detects the 'elbow' rank (point of maximum curvature)
    along the ranked distance curve using the perpendicular distance method.
    """
    if len(distances) < 3:
        return 1

    n_points = len(distances)
    coords = np.vstack((range(1, n_points + 1), distances)).T
    first_pt = coords[0]
    line_vec = coords[-1] - first_pt
    line_norm = np.linalg.norm(line_vec)
    if line_norm == 0:
        return 1

    unit_line = line_vec / line_norm
    vec_from_first = coords - first_pt
    scalar_proj = np.dot(vec_from_first, unit_line)
    proj_vec = np.outer(scalar_proj, unit_line)
    perp_vec = vec_from_first - proj_vec
    perp_dist = np.linalg.norm(perp_vec, axis=1)

    elbow_idx = int(np.argmax(perp_dist))
    return elbow_idx + 1


class StatisticalAnalyzer:
    """
    Analytical engine comparing operational Top-5 retrieval against Top-100 deep-fetch.
    Extracts statistical features, generates diagnostic insights, and builds interactive charts.
    """

    def __init__(self, connector: Optional[ChromaConnector] = None):
        self.connector = connector or ChromaConnector(CHROMA_PERSIST_DIR)

    def analyze_query_retrieval(
        self,
        retrieval_records: List[Dict[str, Any]],
        expected_answer: Optional[str] = None,
        generated_answer: Optional[str] = None,
        target_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete statistical breakdown for a query's retrieval list.

        Parameters:
            retrieval_records: List of document records from ChromaConnector.retrieve_query().
            expected_answer: Optional ground truth string.
            generated_answer: Optional model output string.
            target_doc_id: Optional target document ID.

        Returns:
            Structured dictionary of statistical metrics, drop-offs, and observations.
        """
        if not retrieval_records:
            return {"error": "Empty retrieval records."}

        top_5, ranks_6_100, all_res = self.connector.partition_retrieval_results(retrieval_records, TOP_K_BASELINE)

        top_5_dists = [r["distance"] for r in top_5]
        deep_dists = [r["distance"] for r in ranks_6_100]
        all_dists = [r["distance"] for r in all_res]

        top_5_sims = [r["similarity_score"] for r in top_5]
        deep_sims = [r["similarity_score"] for r in ranks_6_100]

        top5_stats = compute_distribution_metrics(top_5_dists, prefix="top5")
        deep_stats = compute_distribution_metrics(deep_dists, prefix="deep")
        all_stats = compute_distribution_metrics(all_dists, prefix="all")

        # Shifts and Drop-offs
        mean_shift = deep_stats["deep_mean"] - top5_stats["top5_mean"]
        median_shift = deep_stats["deep_median"] - top5_stats["top5_median"]
        
        # Distance dropoff: how much distance grows within Top-5 relative to total
        total_spread = all_stats["all_spread"]
        top5_spread = top5_stats["top5_spread"]
        dropoff_ratio = (top5_spread / max(1e-6, total_spread)) if total_spread > 0 else 0.0

        # Similarity drop-off
        top5_sim_dropoff = (top_5_sims[0] - top_5_sims[-1]) if len(top_5_sims) > 1 else 0.0
        deep_sim_dropoff = (deep_sims[0] - deep_sims[-1]) if len(deep_sims) > 1 else 0.0

        # Top-5 Pairwise Text Coherence
        top5_docs = [r["document"] for r in top_5]
        coherence = compute_pairwise_top5_coherence(top5_docs)

        # Advanced features
        entropy = compute_shannon_entropy(all_dists)
        elbow = compute_elbow_rank(all_dists)

        # Target Document Rank Tracking
        target_rank = None
        if target_doc_id and target_doc_id != "NONE":
            for r in all_res:
                if r["doc_id"] == target_doc_id:
                    target_rank = r["rank"]
                    break

        # Lexical similarities with expected and generated answers if provided
        top1_text = top5_docs[0] if top5_docs else ""
        top5_concat = " ".join(top5_docs)

        sim_metrics = {}
        if expected_answer and str(expected_answer).strip() != "":
            sim_metrics["jaccard_expected_vs_top1"] = compute_jaccard_similarity(expected_answer, top1_text)
            sim_metrics["jaccard_expected_vs_top5"] = compute_jaccard_similarity(expected_answer, top5_concat)
            sim_metrics["rougel_expected_vs_top1"] = compute_rouge_l(expected_answer, top1_text)
            sim_metrics["rougel_expected_vs_top5"] = compute_rouge_l(expected_answer, top5_concat)

        if generated_answer and str(generated_answer).strip() != "":
            sim_metrics["jaccard_generated_vs_top5"] = compute_jaccard_similarity(generated_answer, top5_concat)
            sim_metrics["rougel_generated_vs_top5"] = compute_rouge_l(generated_answer, top5_concat)
            if expected_answer and str(expected_answer).strip() != "":
                sim_metrics["jaccard_expected_vs_generated"] = compute_jaccard_similarity(expected_answer, generated_answer)
                sim_metrics["rougel_expected_vs_generated"] = compute_rouge_l(expected_answer, generated_answer)

        # Assemble summary
        result = {
            "num_retrieved": len(all_res),
            "top5_stats": top5_stats,
            "deep_stats": deep_stats,
            "all_stats": all_stats,
            "mean_distance_shift": mean_shift,
            "median_distance_shift": median_shift,
            "distance_dropoff_ratio": dropoff_ratio,
            "top5_sim_dropoff": top5_sim_dropoff,
            "deep_sim_dropoff": deep_sim_dropoff,
            "top5_coherence": coherence,
            "shannon_entropy": entropy,
            "elbow_rank": elbow,
            "target_doc_id": target_doc_id,
            "target_rank": target_rank,
            "target_in_top5": (target_rank is not None and target_rank <= TOP_K_BASELINE),
            "target_in_deep_fetch": (target_rank is not None and target_rank > TOP_K_BASELINE),
            "similarity_metrics": sim_metrics,
            "top_5_records": top_5,
            "deep_records": ranks_6_100,
            "all_records": all_res,
        }

        # Generate diagnostic insights
        insights = self.generate_diagnostic_insights(result)
        result["diagnostic_insights"] = insights

        return result

    def generate_diagnostic_insights(self, analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Evaluates statistical metrics against empirical thresholds to produce
        transparent, interpretable diagnostic observations.
        Distinguishes observed statistical patterns from proven causality.
        """
        insights = []
        top5_mean = analysis_result["top5_stats"]["top5_mean"]
        top5_var = analysis_result["top5_stats"]["top5_var"]
        mean_shift = analysis_result["mean_distance_shift"]
        coherence = analysis_result["top5_coherence"]
        target_rank = analysis_result.get("target_rank")
        target_id = analysis_result.get("target_doc_id", "NONE")
        sims = analysis_result.get("similarity_metrics", {})

        # 1. Unusually High Retrieval Distance Observation
        if top5_mean > 0.72:
            insights.append({
                "type": "WARNING",
                "category": "Retrieval Miss / Database Gap Pattern",
                "finding": f"Top-5 Mean Distance is exceptionally high ({top5_mean:.4f} > 0.72 threshold).",
                "implication": "Even the closest retrieved documents are semantically distant from the query. Suggests the database may lack coverage on this topic.",
                "action": "Consider checking corpus indexing or enabling fallback search."
            })
        elif top5_mean < 0.60:
            insights.append({
                "type": "SUCCESS",
                "category": "Strong Context Proximity",
                "finding": f"Top-5 Mean Distance is low ({top5_mean:.4f} < 0.60 threshold).",
                "implication": "Retrieved documents possess high semantic vector affinity with the query text.",
                "action": "If the generated response is still incorrect, inspect generator faithfulness or contradiction."
            })

        # 2. Ranking Error Observation
        if target_id != "NONE" and target_rank is not None and target_rank > TOP_K_BASELINE:
            insights.append({
                "type": "ERROR",
                "category": "Ranking Demotion Pattern",
                "finding": f"Ground-truth document (`{target_id}`) found in Deep-Fetch at Rank #{target_rank}.",
                "implication": "The relevant evidence exists in the vector index, but was missed due to operational Top-5 cutoff.",
                "action": "Deploy a cross-encoder reranker on Top-50 or increase retrieval horizon to Top-10."
            })
        elif target_id != "NONE" and target_rank is None:
            insights.append({
                "type": "ERROR",
                "category": "Complete Database Gap Pattern",
                "finding": f"Target document (`{target_id}`) was not found even in Top-100 deep-fetch retrieval.",
                "implication": "Statistical evidence indicates the target information is absent from the indexed collection.",
                "action": "Ingest missing domain documents into ChromaDB."
            })

        # 3. Context Contamination / High Distractor Diversity
        if top5_var > 0.005 and coherence < 0.15:
            insights.append({
                "type": "WARNING",
                "category": "Context Contamination Pattern",
                "finding": f"High Top-5 distance variance ({top5_var:.5f}) with low pairwise coherence ({coherence:.3f}).",
                "implication": "Top-5 chunks diverge into conflicting or disjoint sub-topics (e.g. polysemous distractors).",
                "action": "Apply semantic clustering deduplication or reduce chunk size."
            })

        # 4. Hallucination Signal (Context Groundedness)
        gen_sim = sims.get("jaccard_generated_vs_top5", None)
        if gen_sim is not None and gen_sim < 0.10 and top5_mean < 0.68:
            insights.append({
                "type": "WARNING",
                "category": "Ungrounded Generation Pattern",
                "finding": f"Low lexical overlap between generated answer and retrieved chunks ({gen_sim:.4f} < 0.10).",
                "implication": "The generator produced statements not supported by the retrieved context.",
                "action": "Enforce strict prompt grounding constraints or reduce generation temperature."
            })

        # 5. Plateaued Retrieval Horizon
        if mean_shift < 0.04:
            insights.append({
                "type": "INFO",
                "category": "Flat Distance Profile",
                "finding": f"Minimal distance shift between Top-5 and deeper ranks ({mean_shift:.4f} < 0.04).",
                "implication": "Retrieval distances plateau quickly, providing weak discriminatory contrast across ranks.",
                "action": "Inspect embedding model density resolution or apply BM25 hybrid ranking."
            })

        return insights

    def compute_aggregate_dataset_statistics(
        self,
        detailed_csv: Union[str, Path] = DETAILED_RETRIEVAL_CSV
    ) -> Dict[str, Any]:
        """
        Aggregates statistical metrics across all 5,000 detailed retrieval records
        in the processed dataset.
        """
        df = pd.read_csv(detailed_csv)

        top_5_df = df[df["is_top_5"] == True]
        deep_df = df[df["is_top_5"] == False]

        top5_mean_dist = float(top_5_df["distance"].mean())
        deep_mean_dist = float(deep_df["distance"].mean())
        all_mean_dist = float(df["distance"].mean())

        # Category level breakdown if failure_label exists
        cat_stats = {}
        if "failure_label" in df.columns:
            for cat, group in df.groupby("failure_label"):
                top5_grp = group[group["is_top_5"] == True]
                deep_grp = group[group["is_top_5"] == False]
                cat_stats[str(cat)] = {
                    "top5_mean_dist": float(top5_grp["distance"].mean()),
                    "deep_mean_dist": float(deep_grp["distance"].mean()),
                    "distance_shift": float(deep_grp["distance"].mean() - top5_grp["distance"].mean()),
                    "query_count": int(group["query_id"].nunique()),
                }

        return {
            "total_queries_analyzed": int(df["query_id"].nunique()),
            "total_records_processed": len(df),
            "top5_mean_distance": top5_mean_dist,
            "deep_mean_distance": deep_mean_dist,
            "all_mean_distance": all_mean_dist,
            "global_distance_shift": deep_mean_dist - top5_mean_dist,
            "category_statistics": cat_stats,
        }


# -----------------------------------------------------------------------------
# Visualization Functions (Interactive Plotly)
# -----------------------------------------------------------------------------
def plot_top5_vs_deep_histogram(top5_dists: List[float], deep_dists: List[float], title: str = "") -> go.Figure:
    """Generates an interactive Plotly overlapping histogram / distribution chart."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=top5_dists,
        name="Top-5 Operational Chunks",
        marker_color="#1f77b4",
        opacity=0.75,
        nbinsx=15
    ))
    fig.add_trace(go.Histogram(
        x=deep_dists,
        name="Ranks 6–100 Deep Fetch",
        marker_color="#ff7f0e",
        opacity=0.6,
        nbinsx=30
    ))
    fig.update_layout(
        barmode="overlay",
        title=title or "Distance Distribution: Top-5 vs Deep-Fetch (Ranks 6–100)",
        xaxis_title="Cosine Distance",
        yaxis_title="Chunk Count",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig


def plot_ranked_distance_curve(all_records: List[Dict[str, Any]], target_rank: Optional[int] = None, title: str = "") -> go.Figure:
    """Generates an interactive ranked distance progression curve with Top-5 cutoff line."""
    ranks = [r["rank"] for r in all_records]
    distances = [r["distance"] for r in all_records]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks,
        y=distances,
        mode="lines+markers",
        name="Cosine Distance",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=4)
    ))

    # Top-5 cutoff line
    fig.add_vline(
        x=5.5,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="Top-5 Baseline Cutoff",
        annotation_position="top right"
    )

    # Highlight target document if found
    if target_rank is not None and 1 <= target_rank <= len(distances):
        target_dist = distances[target_rank - 1]
        fig.add_trace(go.Scatter(
            x=[target_rank],
            y=[target_dist],
            mode="markers+text",
            name="Gold Target Document",
            text=[f"Target #{target_rank}"],
            textposition="top center",
            marker=dict(size=14, color="gold", symbol="star", line=dict(color="black", width=1))
        ))

    fig.update_layout(
        title=title or "Ranked Distance Curve (Ranks 1 to 100)",
        xaxis_title="Retrieval Rank",
        yaxis_title="Cosine Distance",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def plot_box_comparison(top5_dists: List[float], deep_dists: List[float], title: str = "") -> go.Figure:
    """Generates comparative box plots for Top-5 vs Deep-Fetch retrieval scores."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=top5_dists, name="Top-5 Baseline", marker_color="#1f77b4", boxmean=True))
    fig.add_trace(go.Box(y=deep_dists, name="Ranks 6–100 Deep Fetch", marker_color="#ff7f0e", boxmean=True))
    fig.update_layout(
        title=title or "Retrieval Distance Spread Comparison",
        yaxis_title="Cosine Distance",
        template="plotly_white"
    )
    return fig


def plot_category_aggregate_boxplots(df_detailed: pd.DataFrame) -> go.Figure:
    """Generates box plot comparing Top-5 distances across failure categories."""
    top5_df = df_detailed[df_detailed["is_top_5"] == True]
    fig = px.box(
        top5_df,
        x="failure_label",
        y="distance",
        color="failure_label",
        title="Operational Top-5 Retrieval Distance Distribution Across Failure Categories",
        labels={"failure_label": "Failure Category", "distance": "Top-5 Cosine Distance"}
    )
    fig.update_layout(template="plotly_white", showlegend=False, xaxis_tickangle=-20)
    return fig
