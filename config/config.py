"""
Configuration module for Causal RAG Failure Diagnostician Platform.
Maintains directory paths, constant parameters, failure categories,
and reproducibility seeds.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Output artifact paths
PROCESSED_FEATURES_CSV = PROCESSED_DATA_DIR / "extracted_features.csv"
DETAILED_RETRIEVAL_CSV = PROCESSED_DATA_DIR / "detailed_retrieval_results.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_diagnostic_model.pkl"
ALL_MODELS_PATH = MODELS_DIR / "all_diagnostic_models.pkl"

# ---------------------------------------------------------------------------
# Legal & Law Domain Knowledge Base (Demo RAG Application #2)
# ---------------------------------------------------------------------------
LEGAL_DATA_DIR = DATA_DIR / "legal_rag"
LEGAL_CHROMA_DIR = LEGAL_DATA_DIR / "chroma_db"
LEGAL_COLLECTION_NAME = "legal_knowledge_base"
LEGAL_QUERIES_CSV = LEGAL_DATA_DIR / "legal_failed_queries.csv"
LEGAL_FEATURES_CSV = LEGAL_DATA_DIR / "legal_extracted_features.csv"
LEGAL_MODELS_DIR = LEGAL_DATA_DIR / "models"
LEGAL_ALL_MODELS_PATH = LEGAL_MODELS_DIR / "all_diagnostic_models.pkl"
LEGAL_BEST_MODEL_PATH = LEGAL_MODELS_DIR / "best_diagnostic_model.pkl"


def get_active_corpus_config():
    """
    Returns active corpus paths dynamically based on Streamlit session state.
    Defaults to Academic Multidisciplinary Corpus if not explicitly switched.
    """
    domain = "Academic"
    try:
        import streamlit as st
        domain = st.session_state.get("active_rag_domain", "Academic")
    except Exception:
        pass

    if "Legal" in str(domain):
        return {
            "key": "legal",
            "name": "Legal & Law Knowledge Base",
            "icon": "⚖️",
            "domain_desc": "Indian Constitutional, Criminal (IPC/BNS), Contract, Tort, Company & International Law",
            "chroma_dir": LEGAL_CHROMA_DIR,
            "collection_name": LEGAL_COLLECTION_NAME,
            "queries_csv": LEGAL_QUERIES_CSV,
            "features_csv": LEGAL_FEATURES_CSV,
            "all_models_path": LEGAL_ALL_MODELS_PATH,
            "best_model_path": LEGAL_BEST_MODEL_PATH,
            "doc_count": 116,
            "queries_count": 50,
        }

    return {
        "key": "academic",
        "name": "Academic Multidisciplinary Corpus",
        "icon": "🏛️",
        "domain_desc": "Machine Learning, Astrophysics, Chemistry, CS History, Genetics, Medicine",
        "chroma_dir": CHROMA_PERSIST_DIR,
        "collection_name": DEFAULT_COLLECTION_NAME,
        "queries_csv": SYNTHETIC_DATA_DIR / "sample_failed_queries.csv",
        "features_csv": PROCESSED_FEATURES_CSV,
        "all_models_path": ALL_MODELS_PATH,
        "best_model_path": BEST_MODEL_PATH,
        "doc_count": 157,
        "queries_count": 50,
    }


# ---------------------------------------------------------------------------
# RAG Failure Categories (Academic Multi-Class Taxonomy)
# ---------------------------------------------------------------------------
FAILURE_CATEGORIES = [
    "Retrieval Miss",        # Ground truth evidence not present in DB
    "Ranking Error",         # Evidence present in DB (> rank 5), missed in Top-5
    "Context Contamination", # Irrelevant/distracting documents retrieved in Top-5
    "Hallucination",         # Answer fabricates information unbacked by context
    "Contradiction",         # Retrieved sources contain conflicting claims
]

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
TOP_K_BASELINE = 5
TOP_K_DEEP_FETCH = 100
DEFAULT_COLLECTION_NAME = "academic_rag_corpus"

# ---------------------------------------------------------------------------
# Reproducibility & ML Defaults
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.25
N_CROSS_VALIDATION_FOLDS = 5

# Supported Classification Algorithms
MODEL_NAMES = [
    "Logistic Regression",
    "Linear SVM",
    "RBF Kernel SVM",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
]

# ---------------------------------------------------------------------------
# Association Rule Mining (Apriori) Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_SUPPORT = 0.10
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MIN_LIFT = 1.10
NUMERICAL_DISCRETIZATION_BINS = 3  # Low, Medium, High
DISCRETIZATION_LABELS = ["Low", "Medium", "High"]
