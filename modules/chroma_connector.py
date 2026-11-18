"""
ChromaDB Local Connector and Deep-Fetch Retrieval Pipeline.

Responsible for:
1. Connecting directly to local persistent ChromaDB without external APIs or middleware.
2. Auditing collection statistics, counts, and HNSW distance metrics.
3. Executing Top-100 deep-fetch retrieval for failed queries.
4. Preserving ranks (1 to 100), document IDs, text, metadata, and distance metrics.
5. Partitioning Top-5 baseline results from ranks 6-100 deep-fetch results.
6. Generating detailed retrieval audit datasets (detailed_retrieval_results.csv).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import chromadb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    TOP_K_BASELINE,
    TOP_K_DEEP_FETCH,
    DETAILED_RETRIEVAL_CSV,
)


class ChromaConnector:
    """
    Direct interface to local ChromaDB vector collections.
    Provides collection inspection, batch Top-100 deep-fetch retrieval,
    and rank-distance tracking.
    """

    def __init__(self, persist_dir: Union[str, Path] = CHROMA_PERSIST_DIR):
        self.persist_dir = Path(persist_dir)
        self.client: Optional[chromadb.PersistentClient] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initializes the persistent ChromaDB client."""
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"ChromaDB persist directory not found at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    def list_collections(self) -> List[str]:
        """Returns names of all collections present in the local database."""
        if self.client is None:
            return []
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieves detailed statistics and configuration for a collection.
        Returns: document count, HNSW space (distance metric), metadata dict.
        """
        if self.client is None:
            raise RuntimeError("Chroma client is not initialized.")

        available = self.list_collections()
        if collection_name not in available:
            raise ValueError(f"Collection '{collection_name}' not found. Available: {available}")

        col = self.client.get_collection(collection_name)
        col_metadata = col.metadata or {}
        distance_metric = col_metadata.get("hnsw:space", "cosine").lower()

        return {
            "name": collection_name,
            "count": col.count(),
            "distance_metric": distance_metric,
            "metadata": col_metadata,
            "persist_dir": str(self.persist_dir),
        }

    def get_collection(self, collection_name: str = DEFAULT_COLLECTION_NAME):
        """Returns the Chroma Collection object."""
        if self.client is None:
            raise RuntimeError("Chroma client is not initialized.")
        return self.client.get_collection(collection_name)

    def retrieve_query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = TOP_K_DEEP_FETCH
    ) -> List[Dict[str, Any]]:
        """
        Executes vector nearest-neighbor search for a single query text.
        Returns list of structured records with ranks, distances, and text.
        """
        col = self.get_collection(collection_name)
        total_docs = col.count()
        if total_docs == 0:
            return []

        effective_k = min(top_k, total_docs)
        distance_metric = (col.metadata or {}).get("hnsw:space", "cosine").lower()

        results = col.query(
            query_texts=[query_text],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"]
        )

        records = []
        ids = results["ids"][0] if results.get("ids") else []
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        dists = results["distances"][0] if results.get("distances") else []

        for idx in range(len(ids)):
            rank = idx + 1
            distance_val = float(dists[idx]) if idx < len(dists) else np.nan

            # Calculate appropriate similarity score respecting Chroma distance metric
            if distance_metric == "cosine":
                # For cosine distance, similarity is 1 - distance (clamped to [0, 1])
                similarity_score = max(0.0, 1.0 - distance_val)
            elif distance_metric == "l2":
                # For L2 distance, similarity can be derived as 1 / (1 + distance)
                similarity_score = 1.0 / (1.0 + distance_val) if distance_val >= 0 else 0.0
            elif distance_metric == "ip":
                # Inner product: higher is closer
                similarity_score = distance_val
            else:
                similarity_score = max(0.0, 1.0 - distance_val)

            records.append({
                "rank": rank,
                "doc_id": ids[idx],
                "document": docs[idx] if idx < len(docs) else "",
                "metadata": metas[idx] if idx < len(metas) else {},
                "distance": distance_val,
                "distance_metric": distance_metric,
                "similarity_score": similarity_score,
                "is_top_5": rank <= TOP_K_BASELINE,
            })

        return records

    def partition_retrieval_results(
        self,
        retrieval_results: List[Dict[str, Any]],
        baseline_k: int = TOP_K_BASELINE
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits retrieval results into:
        1. Top-5 baseline results (ranks 1 to 5).
        2. Deep-fetch results (ranks 6 to 100).
        3. All retrieved results (ranks 1 to 100).
        """
        top_5 = [r for r in retrieval_results if r["rank"] <= baseline_k]
        ranks_6_to_100 = [r for r in retrieval_results if r["rank"] > baseline_k]
        return top_5, ranks_6_to_100, retrieval_results

    def retrieve_for_dataset(
        self,
        df_queries: pd.DataFrame,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = TOP_K_DEEP_FETCH
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Executes Top-K retrieval across an entire dataset of queries.
        Produces:
        1. A comprehensive long-format DataFrame with rank-level retrieval records.
        2. A structured dictionary indexed by query_id for downstream feature extraction.
        """
        col = self.get_collection(collection_name)
        total_docs = col.count()
        effective_k = min(top_k, total_docs)

        detailed_rows = []
        query_retrieval_map: Dict[str, Dict[str, Any]] = {}

        for _, row in df_queries.iterrows():
            qid = str(row.get("query_id", f"Q_{len(query_retrieval_map)+1:03d}"))
            query_text = str(row["query"]).strip()
            exp_ans = str(row.get("expected_answer", ""))
            gen_ans = str(row.get("generated_answer", ""))
            fail_label = str(row.get("failure_label", "Unlabeled"))
            target_id = str(row.get("target_doc_id", "NONE"))
            provenance = str(row.get("provenance", "unknown"))

            # Execute retrieval
            records = self.retrieve_query(collection_name, query_text, top_k=effective_k)
            top_5, ranks_6_100, all_res = self.partition_retrieval_results(records, TOP_K_BASELINE)

            # Check target document rank (if target document was specified)
            target_rank: Optional[int] = None
            if target_id and target_id != "NONE":
                for r in records:
                    if r["doc_id"] == target_id:
                        target_rank = r["rank"]
                        break

            # Store in map for statistical & ML feature extraction
            query_retrieval_map[qid] = {
                "query_id": qid,
                "query": query_text,
                "expected_answer": exp_ans,
                "generated_answer": gen_ans,
                "failure_label": fail_label,
                "provenance": provenance,
                "target_doc_id": target_id,
                "target_rank": target_rank,
                "top_5": top_5,
                "ranks_6_100": ranks_6_100,
                "all_results": all_res,
                "total_retrieved": len(all_res),
            }

            # Build row-level records for detailed audit CSV
            for r in records:
                detailed_rows.append({
                    "query_id": qid,
                    "query": query_text,
                    "expected_answer": exp_ans,
                    "generated_answer": gen_ans,
                    "failure_label": fail_label,
                    "provenance": provenance,
                    "target_doc_id": target_id,
                    "rank": r["rank"],
                    "doc_id": r["doc_id"],
                    "document": r["document"],
                    "distance": r["distance"],
                    "distance_metric": r["distance_metric"],
                    "similarity_score": r["similarity_score"],
                    "is_top_5": r["is_top_5"],
                    "is_target_doc": (r["doc_id"] == target_id) if target_id != "NONE" else False,
                    "target_rank": target_rank,
                })

        detailed_df = pd.DataFrame(detailed_rows)
        return detailed_df, query_retrieval_map

    def save_detailed_results(
        self,
        detailed_df: pd.DataFrame,
        output_path: Union[str, Path] = DETAILED_RETRIEVAL_CSV
    ) -> Path:
        """Persists the detailed retrieval records to CSV for auditing and inspection."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(out_path, index=False)
        return out_path

    def ensure_or_regenerate_audit_dataset(
        self,
        queries_csv: Union[str, Path],
        output_path: Union[str, Path] = DETAILED_RETRIEVAL_CSV,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = TOP_K_DEEP_FETCH,
        force_regenerate: bool = False
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Self-healing CSV recovery mechanism:
        Verifies existence of detailed_retrieval_results.csv. If missing or forced,
        executes Top-100 deep-fetch retrieval across all queries using local ChromaDB,
        persists the output CSV, and returns (df, was_regenerated).
        """
        out_path = Path(output_path)
        if out_path.exists() and not force_regenerate:
            return pd.read_csv(out_path), False

        src_path = Path(queries_csv)
        if not src_path.exists():
            raise FileNotFoundError(f"Source queries dataset not found at: {src_path}")

        df_queries = pd.read_csv(src_path)
        detailed_df, _ = self.retrieve_for_dataset(
            df_queries=df_queries,
            collection_name=collection_name,
            top_k=top_k
        )
        self.save_detailed_results(detailed_df, out_path)
        return detailed_df, True
