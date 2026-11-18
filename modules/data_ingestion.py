"""
Data Ingestion and Data Quality Validation Module for Causal RAG Diagnostician.

Responsible for:
1. Flexible CSV ingestion (file path or file-like object).
2. Schema standardization and column aliasing.
3. Mandatory column validation.
4. Handling missing values, empty queries, whitespace anomalies, and duplicate records.
5. Strict validation of RAG failure categories against academic taxonomy.
6. Provenance preservation and query_id generation.
7. Generating structured DataQualityReport for Streamlit presentation and auditing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io
import pandas as pd
import numpy as np

import sys
# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import FAILURE_CATEGORIES


# Column name aliases for flexible schema handling
COLUMN_ALIASES: Dict[str, List[str]] = {
    "query": ["query", "prompt", "question", "user_query", "input_query", "user_prompt"],
    "expected_answer": ["expected_answer", "ground_truth", "reference_answer", "target_answer", "gold_answer", "true_answer"],
    "generated_answer": ["generated_answer", "model_answer", "rag_response", "response", "output_answer", "model_output"],
    "failure_label": ["failure_label", "failure_category", "label", "category", "root_cause", "failure_type"],
    "query_id": ["query_id", "id", "qid", "sample_id", "interaction_id"],
    "provenance": ["provenance", "source", "dataset_source", "origin", "data_source"],
    "target_doc_id": ["target_doc_id", "relevant_doc_id", "doc_id", "gold_doc_id", "target_chunk_id"],
    "notes": ["notes", "explanation", "comment", "description", "annotator_notes"]
}

MANDATORY_COLUMNS = ["query"]
RECOMMENDED_COLUMNS = ["generated_answer", "expected_answer", "failure_label"]


@dataclass
class DataQualityReport:
    """Structured container holding all validation metrics and dataframes."""
    total_records_ingested: int = 0
    valid_records_count: int = 0
    invalid_records_count: int = 0
    duplicate_records_count: int = 0
    missing_values_by_column: Dict[str, int] = field(default_factory=dict)
    failure_category_distribution: Dict[str, int] = field(default_factory=dict)
    validation_messages: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    missing_recommended_columns: List[str] = field(default_factory=list)
    is_valid_dataset: bool = False
    has_labels: bool = False
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    invalid_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary_dict(self) -> Dict[str, Union[int, bool, Dict[str, int]]]:
        """Returns a high-level dictionary summary suitable for reporting."""
        return {
            "Total Ingested": self.total_records_ingested,
            "Valid Records": self.valid_records_count,
            "Invalid Records": self.invalid_records_count,
            "Duplicates Removed": self.duplicate_records_count,
            "Has Ground Truth Labels": self.has_labels,
            "Is Dataset Usable": self.is_valid_dataset,
            "Missing Values": self.missing_values_by_column,
            "Failure Category Breakdown": self.failure_category_distribution
        }


def standardize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Maps user-provided column variants to standardized internal schema names.
    Preserves any extra user columns not in the canonical list.
    """
    mapped_columns = {}
    df_columns_lower = {col.strip().lower(): col for col in df.columns}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in df_columns_lower:
                original_col = df_columns_lower[alias.lower()]
                mapped_columns[original_col] = canonical_name
                break

    renamed_df = df.rename(columns=mapped_columns)
    return renamed_df, mapped_columns


def validate_rag_dataset(
    df_raw: pd.DataFrame,
    default_provenance: str = "user_uploaded",
    drop_duplicates: bool = True
) -> DataQualityReport:
    """
    Comprehensive data validation and cleaning pipeline for failed RAG interaction datasets.

    Parameters:
        df_raw: Input raw pandas DataFrame.
        default_provenance: Provenance label to assign if column is absent.
        drop_duplicates: Whether to remove duplicate queries.

    Returns:
        DataQualityReport containing validation statistics, cleaned data, and invalid records.
    """
    report = DataQualityReport()
    report.total_records_ingested = len(df_raw)

    if df_raw.empty:
        report.validation_errors.append("Uploaded dataset is empty (0 records).")
        report.is_valid_dataset = False
        return report

    # 1. Standardize column names using alias mapping
    df_standard, mapping = standardize_column_names(df_raw)
    report.validation_messages.append(
        f"Standardized {len(mapping)} columns: {', '.join([f'{k} -> {v}' for k, v in mapping.items()])}"
    )

    # 2. Check mandatory columns
    for mand_col in MANDATORY_COLUMNS:
        if mand_col not in df_standard.columns:
            err_msg = f"Mandatory column '{mand_col}' is missing from dataset. Ingestion aborted."
            report.validation_errors.append(err_msg)
            report.is_valid_dataset = False
            return report

    # 3. Check recommended columns
    for rec_col in RECOMMENDED_COLUMNS:
        if rec_col not in df_standard.columns:
            report.missing_recommended_columns.append(rec_col)
            report.validation_warnings.append(
                f"Optional column '{rec_col}' is missing. Downstream features relying on this may be disabled."
            )

    # 4. Remove completely empty rows
    initial_count = len(df_standard)
    df_work = df_standard.dropna(how="all").copy()
    empty_rows_dropped = initial_count - len(df_work)
    if empty_rows_dropped > 0:
        report.validation_warnings.append(f"Dropped {empty_rows_dropped} completely empty row(s).")

    # 5. String strip & missing values profiling
    # Clean text columns
    text_cols = [c for c in ["query", "expected_answer", "generated_answer", "failure_label", "notes"] if c in df_work.columns]
    for c in text_cols:
        df_work[c] = df_work[c].astype(str).str.strip()
        # Convert "nan", "None", "" back to proper np.nan for numerical and missing calculations
        df_work[c] = df_work[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Record missing values per column
    for col in df_work.columns:
        null_count = int(df_work[col].isna().sum())
        report.missing_values_by_column[col] = null_count

    # 6. Separate valid queries from invalid queries (empty or null queries)
    invalid_rows = []
    
    # Query must not be null or empty
    empty_query_mask = df_work["query"].isna() | (df_work["query"].str.len() == 0)
    if empty_query_mask.any():
        empty_queries_df = df_work[empty_query_mask].copy()
        empty_queries_df["rejection_reason"] = "Empty or null query string"
        invalid_rows.append(empty_queries_df)
        df_work = df_work[~empty_query_mask].copy()
        report.validation_warnings.append(f"Flagged {len(empty_queries_df)} record(s) with empty or missing query text.")

    # 7. Failure Label Validation
    if "failure_label" in df_work.columns:
        non_null_labels = df_work["failure_label"].dropna()
        if len(non_null_labels) > 0:
            report.has_labels = True
            
            # Check validity against canonical academic categories
            invalid_category_mask = ~df_work["failure_label"].isna() & ~df_work["failure_label"].isin(FAILURE_CATEGORIES)
            
            if invalid_category_mask.any():
                invalid_cat_df = df_work[invalid_category_mask].copy()
                invalid_values = invalid_cat_df["failure_label"].unique().tolist()
                invalid_cat_df["rejection_reason"] = invalid_cat_df["failure_label"].apply(
                    lambda v: f"Invalid failure_label '{v}'. Must be one of: {FAILURE_CATEGORIES}"
                )
                invalid_rows.append(invalid_cat_df)
                df_work = df_work[~invalid_category_mask].copy()
                report.validation_warnings.append(
                    f"Found {len(invalid_cat_df)} record(s) with unrecognized failure label(s): {invalid_values}."
                )

            # Record class distribution
            counts = df_work["failure_label"].value_counts().to_dict()
            report.failure_category_distribution = {k: int(v) for k, v in counts.items()}
        else:
            report.has_labels = False
            report.validation_messages.append("Dataset does not contain failure labels (Unlabeled Query Ingestion Mode).")
    else:
        report.has_labels = False
        report.validation_messages.append("No 'failure_label' column found. Operating in Inference / Unlabeled diagnostic mode.")

    # 8. Duplicate Handling
    if drop_duplicates and not df_work.empty:
        # Check duplicates based on query text
        duplicate_mask = df_work.duplicated(subset=["query"], keep="first")
        num_duplicates = int(duplicate_mask.sum())
        report.duplicate_records_count = num_duplicates
        if num_duplicates > 0:
            dup_df = df_work[duplicate_mask].copy()
            dup_df["rejection_reason"] = "Duplicate query text"
            invalid_rows.append(dup_df)
            df_work = df_work[~duplicate_mask].copy()
            report.validation_warnings.append(f"Removed {num_duplicates} duplicate query record(s).")
    else:
        report.duplicate_records_count = 0

    # 9. Handle Provenance
    if "provenance" not in df_work.columns:
        df_work["provenance"] = default_provenance
    else:
        df_work["provenance"] = df_work["provenance"].fillna(default_provenance)

    # 10. Handle Query ID
    if "query_id" not in df_work.columns:
        df_work["query_id"] = [f"Q_{i+1:03d}" for i in range(len(df_work))]
    else:
        # Fill missing query_ids
        missing_qid = df_work["query_id"].isna()
        if missing_qid.any():
            generated_ids = [f"Q_GEN_{i+1:03d}" for i in range(int(missing_qid.sum()))]
            df_work.loc[missing_qid, "query_id"] = generated_ids

    # Assemble cleaned and invalid DataFrames
    report.cleaned_df = df_work.reset_index(drop=True)
    report.valid_records_count = len(report.cleaned_df)

    if invalid_rows:
        report.invalid_df = pd.concat(invalid_rows, ignore_index=True)
        report.invalid_records_count = len(report.invalid_df)
    else:
        report.invalid_df = pd.DataFrame()
        report.invalid_records_count = 0

    report.is_valid_dataset = report.valid_records_count > 0
    if report.is_valid_dataset:
        report.validation_messages.append(
            f"Successfully validated dataset: {report.valid_records_count} valid records ready for feature extraction."
        )
    else:
        report.validation_errors.append("Dataset validation failed: No valid records remaining after filtering.")

    return report


def load_csv(
    file_or_path: Union[str, Path, io.BytesIO, io.StringIO],
    default_provenance: str = "user_uploaded",
    drop_duplicates: bool = True
) -> DataQualityReport:
    """
    Ingests CSV from either a local file path or an in-memory uploaded file buffer,
    then executes the full validation pipeline.

    Parameters:
        file_or_path: Path to CSV or file buffer from Streamlit.
        default_provenance: Provenance tag for audit tracking.
        drop_duplicates: Whether duplicate queries should be eliminated.

    Returns:
        DataQualityReport object.
    """
    try:
        df = pd.read_csv(file_or_path)
    except Exception as e:
        report = DataQualityReport()
        report.validation_errors.append(f"Failed to parse CSV file: {str(e)}")
        report.is_valid_dataset = False
        return report

    return validate_rag_dataset(
        df_raw=df,
        default_provenance=default_provenance,
        drop_duplicates=drop_duplicates
    )
