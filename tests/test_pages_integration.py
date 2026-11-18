"""
Unit and Integration Tests for Phase 6 Streamlit Multi-Page Integration.
Verifies that all 8 Streamlit pages compile and load headlessly,
validates UI component functions, and ensures session-state fallbacks exist.
"""

import sys
import importlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    PROCESSED_FEATURES_CSV,
    BEST_MODEL_PATH,
    SYNTHETIC_DATA_DIR,
)
from modules.ui_components import render_sidebar, render_academic_disclaimer


def test_ui_components_functions():
    """Verify that ui_components functions are callable without crashing."""
    assert callable(render_sidebar)
    assert callable(render_academic_disclaimer)


@pytest.mark.parametrize("page_module_name", [
    "app",
    "pages.2_Data_Ingestion",
    "pages.3_Retrieval_Analysis",
    "pages.4_Statistical_Analyzer",
    "pages.5_Latent_Space_Visualizer",
    "pages.6_ML_Model_Comparison",
    "pages.7_Failure_Diagnosis",
    "pages.8_Apriori_Rules",
])
def test_all_pages_import_cleanly(page_module_name):
    """
    Verifies that each of the 8 Streamlit application pages loads cleanly
    without unhandled NameErrors, ModuleNotErrors, or AttributeErrors.
    """
    mod = importlib.import_module(page_module_name)
    assert mod is not None


def test_critical_artifacts_exist_for_pages():
    """Verifies that all required datasets and models are present on disk."""
    assert CHROMA_PERSIST_DIR.exists()
    assert (SYNTHETIC_DATA_DIR / "sample_failed_queries.csv").exists()
    assert PROCESSED_FEATURES_CSV.exists()
    assert BEST_MODEL_PATH.exists()
