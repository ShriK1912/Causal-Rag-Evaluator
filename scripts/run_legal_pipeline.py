"""
Full Pipeline Runner for Legal RAG Demo Application.
1. Connects to Legal ChromaDB (data/legal_rag/chroma_db)
2. Extracts all 36 features across Top-5 and Top-100 Deep Fetch for the 50 legal queries
3. Saves data/legal_rag/legal_extracted_features.csv
4. Runs MultiModelDiagnosticSuite across all 6 classical ML classifiers
5. Saves data/legal_rag/models/all_diagnostic_models.pkl and best_diagnostic_model.pkl
6. Tests on unseen legal test set and reports benchmark leaderboard
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

from modules.chroma_connector import ChromaConnector
from modules.feature_extraction import extract_features_for_single_query
from modules.diagnostic_models import MultiModelDiagnosticSuite

LEGAL_DIR = PROJECT_ROOT / "data" / "legal_rag"
LEGAL_CHROMA = LEGAL_DIR / "chroma_db"
LEGAL_CSV = LEGAL_DIR / "legal_failed_queries.csv"
LEGAL_FEATURES_CSV = LEGAL_DIR / "legal_extracted_features.csv"
LEGAL_MODELS_DIR = LEGAL_DIR / "models"
LEGAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LEGAL_ALL_MODELS = LEGAL_MODELS_DIR / "all_diagnostic_models.pkl"
LEGAL_BEST_MODEL = LEGAL_MODELS_DIR / "best_diagnostic_model.pkl"
LEGAL_COLLECTION = "legal_knowledge_base"


def main():
    print("=" * 75)
    print("LEGAL RAG PIPELINE: FEATURE EXTRACTION & MULTI-MODEL BENCHMARK")
    print("=" * 75)

    print("\n[Step 1] Connecting to Legal ChromaDB collection...")
    connector = ChromaConnector(persist_dir=LEGAL_CHROMA)
    col = connector.get_collection(LEGAL_COLLECTION)
    print(f"Connected to '{LEGAL_COLLECTION}' with {col.count()} documents.")

    print("\n[Step 2] Loading legal queries dataset...")
    df = pd.read_csv(LEGAL_CSV)
    print(f"Loaded {len(df)} queries across 5 failure categories.")

    print("\n[Step 3] Extracting 36 numerical retrieval & text features via ChromaDB...")
    all_features = []
    for i, row in df.iterrows():
        qid = row["query_id"]
        records = connector.retrieve_query(LEGAL_COLLECTION, str(row["query"]), top_k=100)
        top5, deep, all_r = connector.partition_retrieval_results(records)

        target_id = str(row.get("target_doc_id", "NONE"))
        target_rank = None
        if target_id and target_id != "NONE":
            for r in records:
                if r["doc_id"] == target_id:
                    target_rank = r["rank"]
                    break

        query_data = {
            "query_id": qid,
            "query": str(row["query"]),
            "expected_answer": str(row.get("expected_answer", "")),
            "generated_answer": str(row.get("generated_answer", "")),
            "failure_label": str(row.get("failure_label", "Unlabeled")),
            "provenance": str(row.get("provenance", "legal_rag_v1")),
            "target_doc_id": target_id,
            "target_rank": target_rank,
            "top_5": top5,
            "ranks_6_100": deep,
            "all_results": all_r,
        }
        feats = extract_features_for_single_query(query_data)
        all_features.append(feats)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(df)} queries...")

    features_df = pd.DataFrame(all_features)
    features_df.to_csv(LEGAL_FEATURES_CSV, index=False)
    print(f"\nSaved {len(features_df)} feature rows to: {LEGAL_FEATURES_CSV}")

    print("\n[Step 4] Training and benchmarking all 6 classical ML classifiers on Legal RAG...")
    suite = MultiModelDiagnosticSuite()
    comp_df = suite.train_and_evaluate_all(LEGAL_FEATURES_CSV)

    suite.save_suite(LEGAL_ALL_MODELS, LEGAL_BEST_MODEL)
    print(f"Saved legal model suite to: {LEGAL_ALL_MODELS}")

    print("\n" + "=" * 75)
    print("🏆 LEGAL RAG MULTI-MODEL BENCHMARK SHOOTOUT RESULTS")
    print("=" * 75)
    display_cols = [
        "Model",
        "CV Accuracy",
        "CV Macro F1",
        "Test Accuracy",
        "Test Macro Precision",
        "Test Macro Recall",
        "Test Macro F1",
    ]
    print(comp_df[display_cols].to_string(index=False))

    print("\n" + "=" * 75)
    print(f"🥇 BEST LEGAL DIAGNOSTIC MODEL: {suite.best_model_name_}")
    best_eng = suite.best_engine_
    acc = best_eng.test_metrics_["accuracy"] * 100
    f1 = best_eng.test_metrics_["f1_macro"]
    print(f"   Test Accuracy : {acc:.1f}%")
    print(f"   Test Macro F1 : {f1:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
