# Causal RAG Failure Diagnostician Platform - Development Log

This document serves as the chronological, task-by-task engineering log for the Causal RAG Failure Diagnostician Platform. Every configuration, module, function, and debugging decision is documented in full detail for academic viva review and software reproducibility.

---

## Log Entry 001: Phase 0 - Workspace Inspection, Architecture Formulation, and Environment Setup

- **Task Number**: Task-0.1
- **Task Name**: Environment Inspection & Dependency Resolution
- **What Was Implemented**:
  1. Inspected existing workspace directory `c:\Users\kriti\Desktop\ml` (verified it was initially empty).
  2. Identified Python runtime version: `Python 3.11.9 (64-bit)` on Windows OS.
  3. Audited installed packages; discovered pre-installed `chromadb 1.5.9`, `scikit-learn`, `streamlit 1.52.0`, `plotly 7.1.0`.
  4. Installed missing libraries `mlxtend` (version 0.25.0) and `rouge-score` (version 0.1.2).
  5. Resolved dependency conflict: `mlxtend` upgraded `pandas` to `3.1.1`, which broke `streamlit 1.52.0` (which requires `pandas < 3`). Pinned and installed `pandas 2.3.3` to ensure harmonious compatibility across all components.
  6. Verified local ChromaDB onnx embedding model download and zero-API local caching (`all-MiniLM-L6-v2`, 79.3MB).
- **Why Required**: A machine learning platform with vector storage, statistical analysis, interactive dashboards, and association rules requires strict package compatibility without external API vulnerabilities or breaking version mismatches.
- **Problem Solved**: Eliminated version clashes between Streamlit, Pandas, and Scikit-Learn; ensured vector search runs 100% offline without paid APIs.
- **Libraries Used**: `python 3.11.9`, `pip`, `chromadb`, `streamlit`, `pandas`, `scikit-learn`, `mlxtend`, `rouge-score`, `nltk`.
- **Files Created**:
  - `requirements.txt`: Master project dependency manifest.
  - `.gitignore`: Production-grade ignore rules for Python, virtual environments, cache, and Streamlit secrets.
- **Input & Output**:
  - *Input*: Shell commands `python -m pip install mlxtend rouge-score`, `python -m pip install "pandas<3"`.
  - *Actual Output*: Verified with `python -c "import streamlit, pandas, chromadb, sklearn, mlxtend, rouge_score; print('All core packages successfully imported!')"`. Exited with code 0.
- **Errors Encountered & Resolutions**:
  - *Error*: Streamlit reported conflict with `pandas 3.1.1`.
  - *Resolution*: Explicitly installed `pandas<3` (`pandas 2.3.3`), restoring full compatibility.

---

## Log Entry 002: Phase 0 - Directory Structure and Central Configuration

- **Task Number**: Task-0.2
- **Task Name**: Project Directory Scaffolding & Configuration Architecture
- **What Was Implemented**:
  1. Established clean modular architecture: `config/`, `data/` (`raw/`, `processed/`, `synthetic/`, `chroma_db/`), `models/`, `modules/`, `pages/`, `tests/`, `docs/`.
  2. Implemented `config/config.py` declaring global paths, RAG failure taxonomy, retrieval hyperparameters, ML classification settings, and Apriori rule defaults.
- **Why Required**: Separation of concerns is essential. Hardcoded paths, magic numbers, and arbitrary class definitions across multiple scripts create technical debt, data leakage risks, and make academic review difficult.
- **Algorithms & Concepts**:
  - Multi-class academic taxonomy: 5 mutually distinguishable failure classes:
    1. *Retrieval Miss* (Corpus gap; evidence absent).
    2. *Ranking Error* (Evidence present in corpus, but ranked outside Top-5; captured by Top-100 deep-fetch).
    3. *Context Contamination* (Top-5 chunks contain distracting polysemous or irrelevant noise).
    4. *Hallucination* (Top-5 chunks contain valid evidence, but generator synthesizes unsupported claims).
    5. *Contradiction* (Top-5 chunks contain mutually conflicting claims, causing conflicting output).
  - Reproducibility seed (`RANDOM_STATE = 42`).
- **Files Created**:
  - `config/config.py`
  - `modules/__init__.py`
  - `tests/__init__.py`
- **Important Constants**:
  - `TOP_K_BASELINE = 5`
  - `TOP_K_DEEP_FETCH = 100`
  - `RANDOM_STATE = 42`
  - `DEFAULT_COLLECTION_NAME = "academic_rag_corpus"`

---

## Log Entry 003: Phase 0 - Synthetic Corpus Generation & Controlled Failure Dataset

- **Task Number**: Task-0.3
- **Task Name**: Controlled Synthetic Corpus & Failure Ground Truth Dataset Generation
- **What Was Implemented**:
  1. Created `data/synthetic/generate_synthetic_data.py`.
  2. Built an academic multidisciplinary corpus with 157 documents across Machine Learning, Astrophysics, Genetics/Biology, Quantum Physics, Climatology, World History, and targeted distractor/contradiction pairs.
  3. Populated a local persistent ChromaDB vector collection (`academic_rag_corpus`) configured with `hnsw:space = "cosine"`.
  4. Constructed a controlled benchmark dataset of 50 failed RAG queries (10 per class) with exact ground-truth failure annotations, expected answers, generated answers, target document IDs, provenance tags, and explanatory notes.
- **Why Required**: Real-world public benchmarks (MS MARCO, HaluEval, RAGTruth) rarely provide unified, mutually distinguishable root-cause labels for all five failure modes. Controlled synthetic data with known provenance allows deterministic verification of feature extraction, statistical analyzers, clustering, and diagnostic classifiers before ingesting external datasets.
- **Problem Solved**: Solved the "cold start" and ground-truth ambiguity problem. Every failure has an unambiguous, provable root cause.
- **Libraries Used**: `chromadb`, `pandas`, `pathlib`.
- **Execution & Output**:
  - Command: `python data/synthetic/generate_synthetic_data.py`
  - Result:
    - ChromaDB collection `academic_rag_corpus` populated with 157 documents.
    - `data/synthetic/sample_failed_queries.csv` created with 50 rows (exactly 10 per category).
- **Provenance Rule Enforced**: Every record carries `provenance: "synthetic_controlled_v1"`, preventing mixing of synthetic and benchmark data without audit trails.

---

## Log Entry 004: Phase 0 - Streamlit Dashboard Entrypoint & Verification

- **Task Number**: Task-0.4
- **Task Name**: Interactive Entrypoint Application (`app.py`)
- **What Was Implemented**:
  1. Developed `app.py` as the main Streamlit application entrypoint.
  2. Integrated environment status dashboard displaying Python version, directory validation, ChromaDB collection status, document counts, and synthetic dataset previews.
  3. Added academic disclaimer regarding empirical correlation vs. mathematical causality.
- **Why Required**: Provides immediate interactive verification of Phase 0 status and forms the foundation for multi-page navigation across subsequent phases.
- **Files Created**: `app.py`
- **Verification**: Verified using `python -m py_compile app.py config/config.py data/synthetic/generate_synthetic_data.py` (Exit code 0).

---

## Log Entry 005: Phase 1 - Task 1.1: CSV Ingestion and Data Quality Validation

- **Task Number**: Task-1.1
- **Task Name**: CSV Ingestion and Data Quality Validation Engine
- **What Was Implemented**:
  1. Developed `modules/data_ingestion.py` providing flexible CSV loading from either file paths or Streamlit in-memory uploaded buffers.
  2. Implemented column alias normalization mapping variant headers (`prompt`, `reference_answer`, `rag_response`, `category`, `id`) to canonical schema keys (`query`, `expected_answer`, `generated_answer`, `failure_label`, `query_id`).
  3. Enforced mandatory field verification: guaranteed rejection if `query` is missing.
  4. Handled missing and dirty data: whitespace stripping, converting empty strings/pseudo-nulls to proper `np.nan`, dropping completely blank rows.
  5. Implemented query text validity check: queries that are empty or whitespace-only are filtered out and logged with explicit rejection reasons.
  6. Implemented duplicate detection: flags and removes duplicate queries to prevent data leakage during subsequent cross-validation and feature extraction.
  7. Validated failure categories: validated present labels strictly against canonical 5-class taxonomy (`Retrieval Miss`, `Ranking Error`, `Context Contamination`, `Hallucination`, `Contradiction`). Unrecognized labels are segregated into `invalid_df` rather than silently ignored or coerced.
  8. Maintained unlabeled inference support: datasets lacking ground-truth labels are handled gracefully in inference mode without fabricating artificial labels.
  9. Preserved dataset provenance tags (`provenance`), auto-assigning `"user_uploaded"` only when missing.
  10. Auto-generated unique query identifiers (`query_id`) where missing (`Q_001`, `Q_002`, ...).
  11. Generated structured `DataQualityReport` dataclass encapsulating total, valid, invalid, duplicate counts, missing values by column, and class distribution.
  12. Built dedicated Streamlit page `pages/2_Data_Ingestion.py` featuring interactive CSV uploading, one-click synthetic benchmark loading, KPI metric cards, data quality alerts, missing values tables, Plotly category distribution bar charts, cleaned data preview, and CSV export.
  13. Updated `app.py` dashboard to automatically integrate with session state and display live ingestion metrics.
  14. Wrote unit test suite `tests/test_preprocessing.py` with 7 comprehensive test cases.
- **Why Required**: Real-world RAG interaction logs are messy, inconsistent, and often contain duplicate or invalid queries. Feature extraction and ML classifiers require clean, validated, and normalized tabular inputs. Without rigorous data quality validation, models suffer from garbage-in, garbage-out errors and data leakage.
- **Algorithms and Concepts Used**:
  - Schema mapping and normalization via dictionary lookups.
  - Data hygiene: whitespace stripping, type casting, null normalization.
  - Duplicate record elimination via hashing on normalized text (`pd.DataFrame.duplicated`).
  - Set-based membership testing for taxonomy integrity (`isin(FAILURE_CATEGORIES)`).
  - Segregation pattern: separating valid records from invalid records with audit trails.
- **Files Created or Modified**:
  - `modules/data_ingestion.py`: Core ingestion, standardization, and validation pipeline.
  - `pages/2_Data_Ingestion.py`: Interactive Streamlit ingestion and quality auditing interface.
  - `app.py`: Updated to integrate session state and display active dataset overview.
  - `tests/test_preprocessing.py`: Pytest suite covering all validation rules and edge cases.
- **Input and Output**:
  - *Input*: CSV files (e.g. `data/synthetic/sample_failed_queries.csv` or user uploads).
  - *Output*: `DataQualityReport` containing `cleaned_df` (50 rows, 10 per class), `invalid_df` (0 rows for synthetic, N rows for dirty data), summary statistics dictionary.
- **Testing Results**:
  - Executed `python -m pytest tests/test_preprocessing.py -v`.
  - Result: 7 passed out of 7 tests in 0.79 seconds:
    - `test_synthetic_dataset_ingestion`: PASSED
    - `test_missing_mandatory_query_column`: PASSED
    - `test_column_aliasing`: PASSED
    - `test_empty_and_whitespace_query_handling`: PASSED
    - `test_duplicate_query_removal`: PASSED
    - `test_invalid_failure_category`: PASSED
    - `test_unlabeled_mode`: PASSED
  - Executed `python -m py_compile app.py pages/2_Data_Ingestion.py modules/data_ingestion.py tests/test_preprocessing.py`: Exit code 0.
- **Errors Encountered and Resolutions**:
  - *Issue*: PowerShell did not find `pytest` binary directly on PATH (`CommandNotFoundException`).
  - *Resolution*: Invoked pytest via Python module runner `python -m pytest tests/test_preprocessing.py -v`, which resolved reliably with exit code 0.

---

## Log Entry 006: Phase 1 - Task 1.2: ChromaDB Local Connector & Top-100 Deep-Fetch Retrieval Pipeline

- **Task Number**: Task-1.2
- **Task Name**: ChromaDB Vector DB Integration & Top-100 Retrieval Pipeline
- **What Was Implemented**:
  1. Built `modules/chroma_connector.py` implementing `ChromaConnector` class connecting directly to local ChromaDB persistent vector storage without middleware or external APIs.
  2. Implemented collection inspection: lists available collections, retrieves document counts, extracts collection metadata, and audits HNSW indexing space (`hnsw:space = "cosine"`).
  3. Implemented `retrieve_query()`: Executes vector nearest-neighbor search retrieving the Top-100 documents (or collection max) per query.
  4. Preserved retrieval attributes for each retrieved document:
     - 1-based sequential rank ($1 \le \text{rank} \le 100$)
     - Document unique identifier (`doc_id`)
     - Document raw text (`document`)
     - Metadata dictionary (domain, char length)
     - Distance value (`distance`)
     - Distance metric (`distance_metric = "cosine"`)
     - Validated similarity score (`similarity_score = max(0.0, 1.0 - distance)` bounded in $[0, 1]$)
     - `is_top_5` boolean flag.
  5. Implemented `partition_retrieval_results()`: cleanly segregates operational Top-5 baseline chunks from ranks 6–100 deep-fetch chunks.
  6. Implemented `retrieve_for_dataset()`: Executes batch retrieval across all queries in a DataFrame, computes target document rank positions (`target_rank`), and generates a comprehensive 5,000-row detailed retrieval dataset (`data/processed/detailed_retrieval_results.csv`).
  7. Built interactive Streamlit analysis interface `pages/3_Retrieval_Analysis.py`:
     - Query selector with failure category filtering.
     - Interaction context card (Query, Expected Answer, Generated Answer, Target Doc ID).
     - Metric cards: Top-5 Mean Distance, Min Distance, Ranks 6–100 Mean Distance, Depth Distance Shift, Target Document Rank.
     - Automated diagnostic alerts: identifies ranking errors (target found at rank > 5), retrieval misses (target absent in Top-100), and generation-phase issues.
     - 4 interactive tabs: Top-5 Baseline Chunks, Deep-Fetch Chunks with rank slider, Ranked Distance Curve with Plotly interactive graph highlighting Top-5 cutoff and target document, and Complete Tabular Data with CSV download.
  8. Created Pytest suite `tests/test_chroma_connector.py` with 6 automated tests.
- **Why Required**: RAG failure diagnosis requires inspecting not merely what was returned in the Top-5, but what was missed deeper in the index (ranks 6–100). This provides the empirical basis for distinguishing between true database gaps (Retrieval Miss) and ranking demotions (Ranking Error).
- **Algorithms and Concepts Used**:
  - HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search ($O(\log N)$ query time).
  - Cosine distance vs similarity transformation: $S_{\text{cosine}} = 1 - D_{\text{cosine}}$.
  - Monotonicity verification: ensuring retrieval distances are sorted in non-decreasing order.
  - Deep-fetch comparison: comparing distribution metrics between $k \in [1, 5]$ and $k \in [6, 100]$.
- **Files Created or Modified**:
  - `modules/chroma_connector.py`: Core ChromaDB persistent connector and retrieval engine.
  - `pages/3_Retrieval_Analysis.py`: Interactive Streamlit page for inspecting retrieval chunks and distance curves.
  - `tests/test_chroma_connector.py`: Automated Pytest verification suite.
  - `data/processed/detailed_retrieval_results.csv`: 5,000-row detailed retrieval artifact.
- **Testing Results**:
  - Pytest: `6 passed in 28.65s` (all connector and retrieval tests passed).
  - Total test suite (`python -m pytest tests/ -v`): `13 passed in 22.85s` (100% pass rate).
  - Compilation check: `python -m py_compile` passed with Exit Code 0.

---

## Log Entry 007: Phase 1 - Tasks 1.3, 1.4, 1.5: Feature Extraction Pipeline & Feature Matrix Assembly

- **Task Numbers**: Task-1.3, Task-1.4, Task-1.5
- **Task Names**: Distance Feature Extraction, Text Similarity (Jaccard & ROUGE-L) Extraction, and Feature Matrix Assembly (`extracted_features.csv`)
- **Shared Foundation vs Personal Assignment**:
  - *Shared Foundation*: This feature engineering pipeline extracts the unified numerical feature representation required by ALL models in the platform.
  - *Team Task Allocation Update*:
    - **User's Personal Role**: Assigned exclusively to implement, scale, train, evaluate, and serialize **Model 1: Logistic Regression**, including feature selection, train-test splitting, metrics calculation, and viva documentation.
    - **Teammate's Role**: The remaining 5 classification algorithms (Linear SVM, RBF SVM, Decision Tree, Random Forest, KNN) will be developed and integrated independently by the teammate.
- **What Was Implemented**:
  1. Built `modules/feature_extraction.py` implementing comprehensive feature extraction across three domains:
     - **Distance Features**: `top5_dist_mean`, `top5_dist_std`, `top5_dist_var`, `top5_dist_min`, `top5_dist_max`, `top5_dist_spread`, `deep_dist_mean`, `deep_dist_std`, `deep_dist_var`, `deep_dist_min`, `deep_dist_max`, `deep_dist_spread`, `all_dist_mean`, `all_dist_std`, `all_dist_var`, `all_dist_min`, `all_dist_max`, `all_dist_spread`, `dist_diff_deep_vs_top5`, `dist_dropoff_ratio`, and Euclidean L2 distance equivalents (`top5_l2_mean_equiv`, `all_l2_mean_equiv`).
     - **Text Similarity Features**: Lexical Jaccard token overlap (`jaccard_expected_vs_top1`, `jaccard_expected_vs_top5`, `jaccard_expected_vs_generated`, `jaccard_generated_vs_top5`) and syntactic ROUGE-L LCS F1 scores (`rougel_expected_vs_top1`, `rougel_expected_vs_top5`, `rougel_expected_vs_generated`, `rougel_generated_vs_top5`).
     - **Retrieval Coverage Features**: `num_retrieved_docs`, `target_doc_rank`, `target_in_top5`, `target_in_deep_fetch`, `is_database_gap`, `retrieval_overlap_top5_top100`.
  2. Orchestrated batch pipeline generating the unified feature matrix for all 50 queries:
     - Persisted to: `data/processed/extracted_features.csv`.
     - Matrix shape: 50 rows $\times$ 40 feature columns.
     - Class balance preserved: exactly 10 samples per failure category.
  3. Created Pytest suite `tests/test_feature_extraction.py` testing tokenization, Jaccard math, ROUGE-L LCS calculation, distribution statistics, single query feature dict generation, and CSV integrity.
- **Why Required**: Machine learning models require fixed-dimensional, tabular numerical feature vectors. Raw text chunks and vector distances from ChromaDB must be distilled into statistical indicators that allow classification algorithms to establish decision boundaries.
- **Algorithms and Concepts Used**:
  - Descriptive statistics (Mean, Variance, Bessel's correction, Min, Max, Range).
  - Jaccard Similarity Coefficient: $J(A, B) = |A \cap B| / |A \cup B|$.
  - ROUGE-L Longest Common Subsequence (LCS) F1 measure.
  - Euclidean L2 projection from cosine distance: $d_{\text{L2}} = \sqrt{2 \cdot d_{\text{cosine}}}$.
- **Files Created or Modified**:
  - `modules/feature_extraction.py`: Modular feature extractor.
  - `tests/test_feature_extraction.py`: Pytest suite (6 tests).
  - `data/processed/extracted_features.csv`: 50 $\times$ 40 dataset artifact.
- **Testing Results**:
  - Pytest: `6 passed in 4.59s`.
  - Full suite (`pytest tests/ -v`): `19 passed in 6.83s` (100% pass rate).

---

## Log Entry 008: Phase 4 - User's Personal Assignment: Multinomial Logistic Regression Diagnostic Engine

- **Task Number**: Task-4.1
- **Task Name**: Multinomial Logistic Regression Pipeline, Scaling & Selection Experiments, and Evaluation
- **Assigned Developer**: User (Sole Model Assignment)
- **Teammate Scope**: Linear SVM, RBF SVM, Decision Tree, Random Forest, and KNN are reserved modular interface slots (`BaseDiagnosticModel`).
- **What Was Implemented**:
  1. Developed `modules/diagnostic_models.py` with `BaseDiagnosticModel` abstract base class and `LogisticRegressionDiagnosticEngine`.
  2. Implemented stratified train-test split (`test_size=0.25`, `random_state=42`, `stratify=y`):
     - Training set: 37 samples across 36 numerical features (balanced ~7-8 per category).
     - Test set: 13 samples held out strictly for final evaluation.
     - Enforced strict no-leakage discipline: all preprocessing and feature selection fitted strictly on training data.
  3. **Experiment 1 (Feature Scaling Comparison via 5-Fold Stratified CV on Training Data)**:
     - `StandardScaler`: CV Accuracy = **88.93%** (± 5.58%), CV Macro F1 = **0.8773** (± 0.0765).
     - `MinMaxScaler`: CV Accuracy = **83.57%** (± 6.12%), CV Macro F1 = **0.8227** (± 0.0786).
     - *Outcome*: StandardScaler demonstrated superior stability and F1-score (+5.5% improvement) and was selected for the production pipeline.
  4. **Experiment 2 (Feature Selection Comparison via 5-Fold Stratified CV)**:
     - Implemented `FilterFeatureSelector` using ANOVA F-value (`f_classif`) selecting top 15 features: CV Accuracy = 81.43%, CV Macro F1 = 0.7853.
     - Implemented `EmbeddedL1FeatureSelector` using L1-penalized sparse Logistic Regression with SAGA solver: CV Accuracy = **88.93%**, CV Macro F1 = **0.8800**.
     - Baseline (All 36 Features): CV Accuracy = 88.93%, CV Macro F1 = 0.8773.
     - *Outcome*: Embedded L1 selection reduced feature dimensionality by 58% (from 36 down to 15) while slightly boosting Macro F1 to 0.8800.
     - *Top Predictive Features Identified*:
       - `jaccard_generated_vs_top5` (mean |coef| = 0.4710) - Primary Hallucination discriminator.
       - `target_in_top5` (mean |coef| = 0.4405) - Primary Retrieval Miss / Ranking Error discriminator.
       - `top5_l2_mean_equiv` (mean |coef| = 0.3308) - Global proximity indicator.
       - `jaccard_expected_vs_top5` (mean |coef| = 0.3090) - Gold evidence presence.
       - `rougel_expected_vs_top1` (mean |coef| = 0.3060) - Rank 1 syntactic quality.
  5. **Model Training & Evaluation on Unseen Test Set (13 Holdout Samples)**:
     - Trained final `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs'))])` on the 15 selected features.
     - Actual Held-Out Test Set Performance:
       - **Test Accuracy**: **61.54%** (8 / 13 correct)
       - **Test Macro Precision**: **75.00%**
       - **Test Macro Recall**: **63.33%**
       - **Test Macro F1-Score**: **66.48%**
       - **Test Weighted F1-Score**: **63.88%**
       - **Test Multiclass Log-Loss**: **0.8089**
     - Per-Class Test Breakdown:
       - `Retrieval Miss`: Precision **1.00**, Recall **1.00**, F1 **1.00** (Perfect isolation)
       - `Ranking Error`: Precision **1.00**, Recall **0.67**, F1 **0.80**
       - `Contradiction`: Precision **1.00**, Recall **0.50**, F1 **0.67**
       - `Context Contamination`: Precision **0.50**, Recall **0.67**, F1 **0.57**
       - `Hallucination`: Precision **0.25**, Recall **0.33**, F1 **0.29**
  6. **Model Serialization**:
     - Serialized fitted pipeline and experiment dictionary using Joblib to `models/best_diagnostic_model.pkl`.
     - Tested deserialization: reloaded artifact verified with 100% prediction match.
  7. **Streamlit UI Integration**:
     - Built `pages/6_ML_Model_Comparison.py`: Interactive tabs for Scaling, Feature Selection, Confusion Matrix heatmap, Coefficient Bar Charts, and Teammate allocation table.
     - Built `pages/7_Failure_Diagnosis.py`: Real-time query diagnosis with Softmax probability distributions and actionable corrective engineering recommendations.
  8. **Pytest Verification**:
     - Added 8 unit tests in `tests/test_diagnostic_models.py`. Full project test suite passed: **27 passed in 7.12s**.
- **Important Academic Caveat on Evaluation**:
  - The dataset consists of 50 controlled synthetic records. Consequently, the 25% test split comprises only 13 queries (~2 to 3 queries per class). While Retrieval Miss and Ranking Error are separated decisively, small sample sizes mean that single-sample variations in Hallucination and Contamination have a high percentage impact. We report these empirical results without fabricating artificial 100% metrics.

---

## Log Entry 009: Phase 2 - Deep-Fetch Statistical Analyzer & Diagnostic Observations

- **Task Numbers**: Task-2.1, Task-2.2, Task-2.3, Task-2.4
- **Task Name**: Top-5 vs Top-100 Deep-Fetch Statistical Analysis, Interactive Plotly Visualizations, and Diagnostic Rules
- **Scope**: Shared Analytical Foundation (reused by all downstream modules)
- **What Was Implemented**:
  1. Developed `modules/statistical_analyzer.py` providing:
     - `compute_distribution_metrics()`: Computes Mean, Median, Variance, Standard Deviation, Min, Max, Spread, Interquartile Range (IQR), Skewness, and Euclidean L2 equivalents.
     - `compute_pairwise_top5_coherence()`: Measures pairwise Jaccard token overlap among Top-5 retrieved chunks to detect context contamination.
     - Advanced pluggable features: `compute_shannon_entropy()` (information dispersion across retrieval tiers) and `compute_elbow_rank()` (perpendicular distance curvature method to identify distance transition points).
     - `analyze_query_retrieval()`: Executes query-level statistical profiling comparing operational Top-5 against deep-fetch ranks 6–100, deriving distance shifts ($\mu_{\text{deep}} - \mu_{\text{top5}}$) and drop-off ratios.
     - `generate_diagnostic_insights()`: Transparent rule engine outputting empirical observations (High Distance Warning, Ranking Demotion Alert, Context Contamination Alert, Ungrounded Generation Signal, Flat Horizon Notice) with explicit disclaimers separating correlation from causality.
     - `compute_aggregate_dataset_statistics()`: Aggregates retrieval distance metrics across all 50 queries and 5,000 document records in `detailed_retrieval_results.csv`.
     - Reusable Plotly visualization functions: overlapping histograms, ranked distance curves with Top-5 cutoffs, comparative box plots, and cross-category box plots.
  2. Built interactive Streamlit dashboard `pages/4_Statistical_Analyzer.py`:
     - Toggle between **Query-Level Deep-Fetch Analysis** and **Aggregate Dataset Statistics (5,000 Chunks)**.
     - Comprehensive descriptive statistics comparison table (Top-5 vs Ranks 6–100 vs Global Top-100).
     - Advanced indices expander (Shannon Entropy, Elbow Curvature Rank, Drop-off).
     - Statistical diagnostic findings card with actionable suggestions.
     - 3 interactive Plotly tabs: Overlapping Histogram, Ranked Distance Curve, Comparative Box Plots.
     - Aggregate dataset analysis: Global distance shift, category-level distance profile table, and interactive cross-category box plot.
  3. Created Pytest suite `tests/test_statistical_analyzer.py` with 7 unit tests.
- **Why Required**: Operational Top-5 retrieval masks structural retrieval demotions and corpus gaps. Statistical depth profiling provides empirical mathematical evidence enabling developers and examiners to diagnose whether failure stemmed from ranking demotion, distractor noise, or complete database absence.
- **Algorithms and Mathematical Concepts Used**:
  - Parametric statistics (Mean $\mu$, Variance $\sigma^2$, Standard Deviation $\sigma$, Skewness).
  - Non-parametric statistics (Median, Interquartile Range $Q_3 - Q_1$, Range Spread).
  - Pairwise token coherence (Average Jaccard across Top-5 chunk pairs).
  - Shannon Entropy of continuous distance distributions: $H = -\sum p_i \log_2 p_i$.
  - Perpendicular distance elbow curvature detection.
  - Euclidean L2 equivalent distance projection: $d_{\text{L2}} = \sqrt{2 \cdot d_{\text{cosine}}}$.
- **Files Created or Modified**:
  - `modules/statistical_analyzer.py`: Core statistical calculations and Plotly generators.
  - `pages/4_Statistical_Analyzer.py`: Interactive Streamlit statistical analysis dashboard.
  - `tests/test_statistical_analyzer.py`: Automated Pytest verification suite (7 tests).
- **Testing Results**:
  - Pytest: `7 passed in 5.82s`.
  - Full test suite (`python -m pytest tests/ -v`): **34 passed in 7.36s (100% pass rate)**.
  - Python compilation check: `python -m py_compile` passed with Exit Code 0.

---

## Log Entry 010: Phase 2 - Deep-Fetch Statistical Analyzer Refinement, Mathematical Verification & Formal Sign-off

- **Task Numbers**: Task-2.5, Task-2.6, Task-2.7, Task-2.8, Task-2.9
- **Task Name**: End-to-End Validation, CSV Recovery Mechanism, Mathematical Derivations & Formal Sign-off
- **Scope**: Shared Analytical Foundation (Production Hardening)
- **What Was Implemented**:
  1. **Task 2.5: End-to-End Partition & Edge-Case Validation**:
     - Inspected `modules/chroma_connector.py`, `modules/statistical_analyzer.py`, and `pages/4_Statistical_Analyzer.py`.
     - Confirmed that operational Top-5 (ranks 1 to 5) and deep-fetch tiers (ranks 6 to 100) are strictly disjoint and indexed correctly ($1 \dots 5$ and $6 \dots 100$).
     - Verified graceful handling of edge cases where retrieval returns fewer than 5 documents (deep-fetch metrics yield `np.nan` and UI formats as `"N/A"` without runtime crashes).
  2. **Task 2.6: Self-Healing CSV Recovery Mechanism**:
     - Implemented `ChromaConnector.ensure_or_regenerate_audit_dataset(queries_csv, output_path, top_k, force_regenerate)`.
     - In `pages/4_Statistical_Analyzer.py` Mode 2: Added detection of missing `detailed_retrieval_results.csv`, accompanied by a prominent warning alert and interactive recovery button (`🔄 Regenerate Detailed Retrieval Dataset`).
     - Added an administrative dataset maintenance expander enabling developers to force-refresh the 5,000-chunk retrieval audit dataset at any time.
  3. **Task 2.7: Mathematical Verification**:
     - Verified formulas for parametric ($\mu, \sigma^2, \sigma$), non-parametric ($\text{Median}, \text{Spread}, \text{IQR}, \text{Skewness}$), and information-theoretic metrics ($H(X), \text{Elbow Rank}, \text{Coherence}$).
     - Documented and tested the Euclidean $L_2$ equivalent formula under unit-norm embeddings:
       $$\|u - v\|_2 = \sqrt{2 \cdot d_{\text{cosine}}}$$
       Precondition: Strictly valid when $\|u\|_2 = \|v\|_2 = 1$ (default for ChromaDB `all-MiniLM-L6-v2` embeddings). Documented boundary conditions ($d_{\text{cosine}} = 0 \implies L_2 = 0$; $d_{\text{cosine}} = 0.5 \implies L_2 = 1.0$; $d_{\text{cosine}} = 2.0 \implies L_2 = 2.0$).
  4. **Task 2.8: UI Resilience & Formatting**:
     - Implemented `fmt(val, decimals=4)` helper function across all metric displays and summary tables in `pages/4_Statistical_Analyzer.py`, preventing unhandled exceptions when encountering NaNs or empty lists.
     - Verified interactive Plotly histograms, ranked progression curves, and cross-category box plots in both query-level and aggregate dataset modes.
  5. **Task 2.9: Automated Testing Expansion**:
     - Added 3 new unit tests to `tests/test_statistical_analyzer.py`:
       - `test_ensure_or_regenerate_audit_dataset`: Verifies recovery when destination CSV is missing or forced.
       - `test_l2_euclidean_unit_norm_identity`: Verifies geometric identity and boundary conditions.
       - `test_edge_case_fewer_than_five_docs`: Verifies handling of queries with retrieval count $< 5$.
     - Ran full compilation check across 22 Python files (all passed).
     - Executed full test suite: **37 passed, 0 failed (100% pass rate)**.
- **Why Required**: RAG diagnostic platforms must maintain strict mathematical integrity, numerical stability, and self-healing resilience during live examiner demonstrations.
- **Files Modified**:
  - `modules/statistical_analyzer.py`: Added detailed mathematical precondition docstrings for Euclidean-cosine conversion.
  - `modules/chroma_connector.py`: Added `ensure_or_regenerate_audit_dataset()` recovery method.
  - `pages/4_Statistical_Analyzer.py`: Added recovery UI warning/button, maintenance expander, and `fmt()` formatting helper.
  - `tests/test_statistical_analyzer.py`: Added 3 unit tests covering recovery, mathematical identities, and edge cases.
- **Testing Results**:
  - `python -m pytest tests/ -v`: **37 passed, 0 failed** in 22.79s.
  - Python compilation check: 22/22 files compiled successfully with exit code 0.
- **Sign-off Status**: **Phase 2 is formally completed, verified, and signed off.**

---

## Log Entry 011: Phase 3 - Latent Space & Database Gap Visualizer (PCA 3D, K-Means, DBSCAN)

- **Task Numbers**: Task-3.1, Task-3.2, Task-3.3, Task-3.4, Task-3.5, Task-3.6
- **Task Name**: Latent Space 3D Projection, Dual PCA Scaling Analysis, K-Means Multi-K Sweep, DBSCAN Density Outlier Gap Detection, Multi-Page Dashboard, and Unit Testing
- **Scope**: Shared Analytical Foundation (Dimensionality Reduction & Unsupervised Clustering)
- **What Was Implemented**:
  1. **Task 3.1: Corpus Embeddings & Text Ingestion (`modules/latent_space_visualizer.py`)**:
     - Connected directly to local ChromaDB collection `academic_rag_corpus` (157 documents, 384 dimensions).
     - Fixed NumPy array truthiness evaluation bug (`len(raw_embeddings) == 0`).
     - Preserved document IDs, metadata, domain categories, and raw chunk text for interactive tooltips and document inspection.
     - Verified $L_2$-normalization of vectors ($\|v\|_2 = 1.0 \pm 1e-3$).
  2. **Task 3.2: 3D PCA & Scaling Strategy Analysis**:
     - Implemented `apply_pca_3d()` reducing 384 dimensions to 3 principal components.
     - Addressed unit-norm scaling geometry: demonstrated that Centered PCA (mean-centering without feature scaling) preserves **53.44%** cumulative variance in 3D, compared to **49.52%** for Standardized PCA (+3.92% variance preserved by preserving natural spherical angles).
     - Computed individual variance ratios: PC1 = 19.84%, PC2 = 17.26%, PC3 = 16.35%.
  3. **Task 3.3: K-Means Clustering & Sweep ($K = 2 \dots 10$)**:
     - Swept $K \in [2, 10]$ recording WCSS (Inertia) and Mean Silhouette Scores.
     - Enforced mathematical safeguard: validated that Silhouette score is computed only when at least 2 distinct clusters exist.
     - Added dual evaluation: Elbow curve (WCSS vs K) and dedicated Silhouette score bar chart highlighting optimal K.
     - Supported clustering on either the full 384-dimensional space (avoiding projection loss) or 3D PCA coordinates.
  4. **Task 3.4: DBSCAN Density & Sparse Gap Detection**:
     - Implemented `run_dbscan()` with configurable $\epsilon$ and $\text{MinPts}$.
     - Generated sensitivity curves across $\epsilon \in [0.4, 1.5]$ analyzing cluster formation vs noise decay.
     - Styled noise points (label = -1) with distinct red diamond markers in Plotly 3D scatter plots to visually highlight potential database coverage gaps.
     - Documented academic disclaimer: density outliers indicate semantic sparsity, not definitive functional failure.
  5. **Task 3.5: Streamlit Interactive Dashboard (`pages/5_Latent_Space_Visualizer.py`)**:
     - Built comprehensive dashboard with KPI highlights, PCA variance decomposition chart, and scaling comparison.
     - Implemented 3 interactive 3D Plotly tabs: Domain Coloring, K-Means Partitions, and DBSCAN Clusters/Outliers with rich hover text snippets.
     - Added side-by-side Elbow & Silhouette graphs and DBSCAN sensitivity plot.
     - Created interactive Document Inspector with domain, K-Means, and DBSCAN status filtering, displaying raw text, metadata, and 3D PCA coordinates.
  6. **Task 3.6: Automated Testing (`tests/test_latent_space.py`)**:
     - Created 9 comprehensive unit tests covering: embedding loading integrity, PCA shapes and variance, scaling comparison, K-Means inertia monotonicity and silhouette validity, K=5 partitions, DBSCAN noise detection and sensitivity sweeps, extreme edge cases (all noise, insufficient samples), and end-to-end orchestrator execution.
- **Why Required**: Dense retrieval models operate in high-dimensional vector space where human intuition fails. Dimensionality reduction and unsupervised clustering provide visual and statistical transparency into corpus topology, domain imbalances, and sparse regions where queries are vulnerable to retrieval gaps.
- **Algorithms and Mathematical Concepts Used**:
  - Sample Covariance Matrix Eigen-Decomposition ($\Sigma = \frac{1}{n-1} X^T X, \Sigma v = \lambda v$).
  - Explained Variance Ratio ($\text{EVR}_k = \lambda_k / \sum \lambda_j$).
  - K-Means Lloyd's Algorithm minimizing Within-Cluster Sum of Squares ($J = \sum_{j=1}^k \sum_{x \in S_j} \|x - \mu_j\|^2$).
  - Silhouette Coefficient ($s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$).
  - DBSCAN Core, Border, and Noise Point Partitioning ($N_\epsilon(p)$ with threshold $\text{MinPts}$).
- **Files Created or Modified**:
  - `modules/latent_space_visualizer.py`: Core PCA, K-Means, DBSCAN engine, and 7 Plotly visualization generators.
  - `pages/5_Latent_Space_Visualizer.py`: Interactive Streamlit dashboard.
  - `tests/test_latent_space.py`: 9 automated Pytest unit tests.
- **Testing Results**:
  - `python -m pytest tests/test_latent_space.py -v`: **9 passed in 10.57s (100% pass rate)**.
  - Complete project regression suite (`python -m pytest tests/ -v`): **46 passed in 25.79s (100% pass rate)**.
  - Python compilation check: 24/24 files compiled successfully with exit code 0.
- **Sign-off Status**: **Phase 3 is formally completed, verified, and signed off.**

---

## Log Entry 012: Phase 5 - Apriori Association Rule Mining Engine

- **Task Numbers**: Task-5.1, Task-5.2, Task-5.3, Task-5.4, Task-5.5
- **Task Name**: Feature Discretization, Frequent Itemset Mining, Association Rules Extraction, Multi-Page Dashboard, and Automated Pytest Verification
- **Scope**: Shared Analytical Foundation (Association Rule Mining & Diagnostic Pattern Discovery)
- **What Was Implemented**:
  1. **Task 5.1: Feature Discretization & Transaction Encoding (`modules/apriori_analyzer.py`)**:
     - Loaded `data/processed/extracted_features.csv` (50 samples, 40 features).
     - Filtered out non-feature identifier columns (`query_id`, `query`, `provenance`).
     - Preserved `failure_label` as the target consequent column, encoded as `failure_category=<Category>`.
     - Detected and dropped 4 constant numeric columns (`num_retrieved_docs`, `target_in_deep_fetch`, `is_database_gap`, `retrieval_overlap_top5_top100`) to prevent trivial 100% support artifacts.
     - Implemented uniform equal-width discretization (default) and quantile equal-frequency binning into ordinal bins (`Low`, `Medium`, `High`).
     - One-hot encoded discretized features into a 100-column boolean transaction matrix.
  2. **Task 5.2: Apriori Frequent Itemset Mining**:
     - Integrated `mlxtend.frequent_patterns.apriori` with configurable `min_support` and `max_len`.
     - At `min_support = 0.10, max_len = 3`, mined **23,597 frequent itemsets**.
  3. **Task 5.3: Association Rule Mining Across All 5 Failure Categories**:
     - Integrated `mlxtend.frequent_patterns.association_rules` computing Support, Confidence, Lift, Leverage, and Conviction.
     - Extracted **610 category-specific diagnostic rules** across all 5 failure categories:
       - **Retrieval Miss (366 rules)**: e.g. `top5_dist_min=High -> Retrieval Miss` (Conf: 1.0, Lift: 5.0).
       - **Contradiction (125 rules)**: e.g. `top5_dist_var=High -> Contradiction` (Conf: 1.0, Lift: 5.0).
       - **Ranking Error (76 rules)**: e.g. `jaccard_expected_vs_top5=High -> Ranking Error` (Conf: 1.0, Lift: 5.0).
       - **Context Contamination (40 rules)**: e.g. `top5_dist_mean=Medium, jaccard_generated_vs_top5=High -> Context Contamination` (Conf: 1.0, Lift: 5.0).
       - **Hallucination (3 rules)**: e.g. `jaccard_expected_vs_top1=High, rougel_generated_vs_top5=Medium -> Hallucination` (Conf: 0.83, Lift: 4.17).
     - Built human-interpretable plain-language rule explanation formatter (`explain_rule()`) emphasizing statistical association over causality.
  4. **Task 5.4: Streamlit Interactive Dashboard (`pages/8_Apriori_Rules.py`)**:
     - Hyperparameter controls: Support slider (0.05–0.50), Confidence slider (0.30–1.00), Lift slider (1.00–5.00), itemset length selector, and discretization strategy toggle.
     - KPI header displaying total transactions, features, frequent itemsets, and diagnostic rules.
     - Interactive Plotly visualizations: Support vs Confidence scatter (bubble size = Lift), rule counts by category bar chart, and horizontal top rules ranking bar chart.
     - Rule explorer with top-3 spotlight cards, complete sortable data table, and one-click CSV download.
     - Frequent itemsets inspector (L1, L2, L3) with length filters and CSV export.
     - Feature discretization audit showing bin methodology, dropped constant columns, and sample transactions.
  5. **Task 5.5: Automated Testing Suite (`tests/test_apriori_analyzer.py`)**:
     - Implemented 10 comprehensive unit tests covering: discretization correctness, constant/missing value handling, one-hot boolean encoding, frequent itemset constraints, metric calculations (confidence, lift, leverage, conviction), rule discovery across all 5 categories, parameter validation, empty dataset edge cases, explanation formatting, and table export methods.
- **Why Required**: While black-box ML models predict failure probabilities, examiners and engineers demand interpretable, rule-based diagnostic patterns that explicitly describe which combination of retrieval distance and lexical coherence triggers each failure mode.
- **Algorithms and Mathematical Concepts Used**:
  - Apriori downward-closure property (all non-empty subsets of a frequent itemset must also be frequent).
  - Rule Support: $P(A \cap C) = \sigma(A \cup C) / N$.
  - Rule Confidence: $P(C \mid A) = \sigma(A \cup C) / \sigma(A)$.
  - Rule Lift: $P(C \mid A) / P(C) = \text{Confidence} / \text{Support}(C)$.
  - Rule Leverage: $\text{Support}(A \cap C) - \text{Support}(A) \cdot \text{Support}(C)$.
  - Rule Conviction: $(1 - \text{Support}(C)) / (1 - \text{Confidence})$.
- **Files Created**:
  - `modules/apriori_analyzer.py`: Discretization, transaction encoder, Apriori engine, and Plotly visualizers.
  - `pages/8_Apriori_Rules.py`: Interactive Streamlit dashboard.
  - `tests/test_apriori_analyzer.py`: 10 automated unit tests.
- **Testing Results**:
  - `python -m pytest tests/test_apriori_analyzer.py -v`: **10 passed in 3.84s (100% pass rate)**.
  - Complete project regression suite (`python -m pytest tests/ -v`): **56 passed in 49.74s (100% pass rate)**.
  - Python compilation check: 27/27 files compiled successfully with exit code 0.
- **Sign-off Status**: **Phase 5 is formally completed, verified, and signed off.**


---

## Log Entry 014: Phase 6 Multi-Page Streamlit Integration & System Polish
- **Date**: 2026-11-07
- **Phase**: Phase 6 – Streamlit Multi-Page Integration & System Polish
- **Status**: Completed & Verified

### 1. Goals & Accomplishments
1. **End-to-End Application Audit**:
   - Inspected and verified all 8 application pages: `app.py`, `pages/2_Data_Ingestion.py`, `pages/3_Retrieval_Analysis.py`, `pages/4_Statistical_Analyzer.py`, `pages/5_Latent_Space_Visualizer.py`, `pages/6_ML_Model_Comparison.py`, `pages/7_Failure_Diagnosis.py`, and `pages/8_Apriori_Rules.py`.
   - Fixed runtime `NameError: name 'Any' is not defined` in `pages/4_Statistical_Analyzer.py`.
   - Fixed `KeyError: 'configurations'` on `pages/6_ML_Model_Comparison.py` by restoring full cross-validation and feature selection metadata in `models/best_diagnostic_model.pkl`.
   - Prevented test pollution in `tests/test_diagnostic_models.py` by isolating model serialization tests using pytest's `tmp_path`.
2. **Standardized UI Component Architecture**:
   - Created `modules/ui_components.py` featuring:
     - `render_sidebar(current_page)`: Standardized title, academic subtitle, active view badge, and environment health cards (ChromaDB connection, dataset status, feature count, model status).
     - `render_academic_disclaimer()`: Prominent academic callout distinguishing statistical correlation from true causal mechanisms.
   - Integrated `render_sidebar()` and `render_academic_disclaimer()` across all 8 pages.
   - Enforced `layout="wide"` and Plotly `plotly_white` theme consistency across all views.
3. **Responsive Caching Optimization**:
   - Implemented `@st.cache_resource(show_spinner=False)` for heavy singleton instances:
     - ChromaDB connector (`get_chroma_connector()`, `get_analyzer_bundle()`)
     - Model artifact deserialization (`get_trained_model_artifact()`, `load_diagnostic_artifact()`)
     - PCA & Latent Space visualizer engine (`get_visualizer_engine()`)
   - Implemented `@st.cache_data(show_spinner=False)` for repetitive computations:
     - Deep-fetch query retrieval (`fetch_retrieval_records()`)
     - Query-level statistical analysis (`cached_query_analysis()`)
     - Aggregate dataset statistics calculation (`get_cached_aggregate_stats()`)
     - Feature CSV loading (`load_features_dataset()`)
     - Apriori frequent itemset and association rule mining (`run_apriori_pipeline()`)
4. **Automated Integration Testing**:
   - Created `tests/test_pages_integration.py` containing 10 automated test cases verifying that UI component functions execute cleanly, all 8 pages import and execute headlessly without tracebacks, and required disk artifacts exist.
   - Full test suite passes: 66 total tests across 8 test suites.
