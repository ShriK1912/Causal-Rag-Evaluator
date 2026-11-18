"""
Feature Extraction Pipeline for Causal RAG Failure Diagnostician.

Extracts numerical, statistical, lexical, and retrieval-based features from
ChromaDB retrieval results to form the tabular dataset for ML diagnosis.

Features Extracted:
A. Distance-Based Features (Cosine & Euclidean L2, Top-5 vs Deep-Fetch Ranks 6-100)
B. Text Similarity Features (Jaccard Token Overlap & ROUGE-L LCS)
C. Retrieval-Based Features (Target Ranks, Coverage, Drop-off)
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    PROCESSED_FEATURES_CSV,
    SYNTHETIC_DATA_DIR,
    TOP_K_BASELINE,
    TOP_K_DEEP_FETCH,
)
from modules.chroma_connector import ChromaConnector


# Initialize global ROUGE scorer once for efficiency
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def tokenize_text(text: str) -> set:
    """Extracts lowercase alphanumeric word tokens as a unique set."""
    if not text or pd.isna(text):
        return set()
    return set(re.findall(r"\b\w+\b", str(text).lower()))


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Computes Jaccard Similarity between two texts based on unique word tokens:
    J(A, B) = |A ∩ B| / |A ∪ B|
    Returns float in range [0.0, 1.0].
    """
    tokens_a = tokenize_text(text_a)
    tokens_b = tokenize_text(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    if not union:
        return 0.0
    return float(len(intersection) / len(union))


def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    Computes ROUGE-L F1 score based on Longest Common Subsequence (LCS).
    Returns float in range [0.0, 1.0].
    """
    if not reference or not candidate or pd.isna(reference) or pd.isna(candidate):
        return 0.0
    ref_str = str(reference).strip()
    cand_str = str(candidate).strip()
    if not ref_str or not cand_str:
        return 0.0
    scores = _ROUGE_SCORER.score(ref_str, cand_str)
    return float(scores["rougeL"].fmeasure)


def calculate_distribution_stats(distances: List[float], prefix: str) -> Dict[str, float]:
    """
    Calculates descriptive statistics for a list of distance values:
    Mean, Variance, Standard Deviation, Minimum, Maximum, and Spread.
    """
    if not distances:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_var": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_spread": np.nan,
        }

    arr = np.array(distances, dtype=float)
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    spread_val = max_val - min_val

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_var": var_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_spread": spread_val,
    }


def extract_features_for_single_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts all numerical and categorical features for a single RAG query.

    Parameters:
        query_data: Dictionary produced by ChromaConnector.retrieve_for_dataset()
                    containing query strings, target metadata, top_5, ranks_6_100, all_results.

    Returns:
        Flat dictionary of features for one row in the ML feature matrix.
    """
    qid = query_data["query_id"]
    query_text = query_data["query"]
    expected_ans = query_data.get("expected_answer", "")
    generated_ans = query_data.get("generated_answer", "")
    failure_label = query_data.get("failure_label", "Unlabeled")
    provenance = query_data.get("provenance", "unknown")
    target_id = query_data.get("target_doc_id", "NONE")
    target_rank = query_data.get("target_rank", None)

    top_5_records = query_data.get("top_5", [])
    deep_records = query_data.get("ranks_6_100", [])
    all_records = query_data.get("all_results", [])

    # Distances
    top_5_dists = [r["distance"] for r in top_5_records if not np.isnan(r.get("distance", np.nan))]
    deep_dists = [r["distance"] for r in deep_records if not np.isnan(r.get("distance", np.nan))]
    all_dists = [r["distance"] for r in all_records if not np.isnan(r.get("distance", np.nan))]

    features: Dict[str, Any] = {
        "query_id": qid,
        "query": query_text,
        "provenance": provenance,
        "failure_label": failure_label,
    }

    # -------------------------------------------------------------------------
    # A. Distance-Based Features
    # -------------------------------------------------------------------------
    top5_stats = calculate_distribution_stats(top_5_dists, "top5_dist")
    deep_stats = calculate_distribution_stats(deep_dists, "deep_dist")
    all_stats = calculate_distribution_stats(all_dists, "all_dist")

    features.update(top5_stats)
    features.update(deep_stats)
    features.update(all_stats)

    # Relative Distance Shift between Deep-Fetch and Top-5
    features["dist_diff_deep_vs_top5"] = deep_stats["deep_dist_mean"] - top5_stats["top5_dist_mean"]

    # Distance Drop-off / Steepness Ratio
    # Measures whether distances plateau rapidly or expand gradually
    total_spread = all_stats["all_dist_spread"]
    top5_spread = top5_stats["top5_dist_spread"]
    features["dist_dropoff_ratio"] = (top5_spread / max(1e-6, total_spread)) if total_spread > 0 else 0.0

    # Euclidean L2 equivalent distance (derived from Cosine Distance: d_L2 = sqrt(2 * d_cosine))
    features["top5_l2_mean_equiv"] = np.sqrt(max(0.0, 2.0 * top5_stats["top5_dist_mean"]))
    features["all_l2_mean_equiv"] = np.sqrt(max(0.0, 2.0 * all_stats["all_dist_mean"]))

    # -------------------------------------------------------------------------
    # B. Text Similarity Features (Lexical Jaccard & Syntactic ROUGE-L)
    # -------------------------------------------------------------------------
    # Prepare text blocks
    top1_doc_text = top_5_records[0]["document"] if len(top_5_records) > 0 else ""
    top5_concat_text = " ".join([r["document"] for r in top_5_records])

    # 1. Expected Answer vs Retrieved Context (Retrieval Relevance Signal)
    if expected_ans and str(expected_ans).strip() != "":
        features["jaccard_expected_vs_top1"] = compute_jaccard_similarity(expected_ans, top1_doc_text)
        features["jaccard_expected_vs_top5"] = compute_jaccard_similarity(expected_ans, top5_concat_text)
        features["rougel_expected_vs_top1"] = compute_rouge_l(expected_ans, top1_doc_text)
        features["rougel_expected_vs_top5"] = compute_rouge_l(expected_ans, top5_concat_text)
        features["jaccard_expected_vs_generated"] = compute_jaccard_similarity(expected_ans, generated_ans)
        features["rougel_expected_vs_generated"] = compute_rouge_l(expected_ans, generated_ans)
    else:
        features["jaccard_expected_vs_top1"] = 0.0
        features["jaccard_expected_vs_top5"] = 0.0
        features["rougel_expected_vs_top1"] = 0.0
        features["rougel_expected_vs_top5"] = 0.0
        features["jaccard_expected_vs_generated"] = 0.0
        features["rougel_expected_vs_generated"] = 0.0

    # 2. Generated Answer vs Retrieved Context (Hallucination & Faithfulness Signal)
    # Low overlap with retrieved context indicates generated answer fabricated claims ungrounded in context!
    if generated_ans and str(generated_ans).strip() != "":
        features["jaccard_generated_vs_top5"] = compute_jaccard_similarity(generated_ans, top5_concat_text)
        features["rougel_generated_vs_top5"] = compute_rouge_l(generated_ans, top5_concat_text)
    else:
        features["jaccard_generated_vs_top5"] = 0.0
        features["rougel_generated_vs_top5"] = 0.0

    # -------------------------------------------------------------------------
    # C. Retrieval-Based Coverage Features
    # -------------------------------------------------------------------------
    features["num_retrieved_docs"] = len(all_records)
    features["target_doc_rank"] = int(target_rank) if target_rank is not None else -1
    features["target_in_top5"] = 1 if (target_rank is not None and target_rank <= TOP_K_BASELINE) else 0
    features["target_in_deep_fetch"] = 1 if (target_rank is not None and target_rank > TOP_K_BASELINE) else 0
    features["is_database_gap"] = 1 if (target_id != "NONE" and target_rank is None) else 0
    features["retrieval_overlap_top5_top100"] = float(len(top_5_records) / max(1, len(all_records)))

    return features


def extract_features_pipeline(
    df_queries: pd.DataFrame,
    connector: Optional[ChromaConnector] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = TOP_K_DEEP_FETCH
) -> pd.DataFrame:
    """
    Main pipeline function that ingests validated queries, retrieves Top-100 chunks,
    extracts all feature categories, and returns a clean feature DataFrame.
    """
    if connector is None:
        connector = ChromaConnector(persist_dir=CHROMA_PERSIST_DIR)

    # 1. Execute Top-100 batch retrieval
    _, query_map = connector.retrieve_for_dataset(
        df_queries=df_queries,
        collection_name=collection_name,
        top_k=top_k
    )

    # 2. Extract features per query
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["query_id"])
        if qid in query_map:
            feat_dict = extract_features_for_single_query(query_map[qid])
            rows.append(feat_dict)

    feature_df = pd.DataFrame(rows)
    return feature_df


def save_extracted_features(
    feature_df: pd.DataFrame,
    output_path: Union[str, Path] = PROCESSED_FEATURES_CSV
) -> Path:
    """Saves the extracted feature dataset to disk as CSV."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print("--- Running Feature Extraction Pipeline ---")
    sample_csv = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
    if sample_csv.exists():
        df_in = pd.read_csv(sample_csv)
        conn = ChromaConnector()
        feats = extract_features_pipeline(df_in, conn)
        out = save_extracted_features(feats)
        print(f"Extracted {len(feats)} rows and {len(feats.columns)} features.")
        print(f"Saved successfully to: {out}")
        print("\nFeature Columns:")
        print(list(feats.columns))
    else:
        print(f"Sample CSV not found at {sample_csv}")
