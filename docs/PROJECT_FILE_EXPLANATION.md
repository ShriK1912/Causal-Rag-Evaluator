# Project File-by-File Technical Guide

This document provides a comprehensive, file-by-file explanation of every script, configuration file, and module in the **Causal RAG Failure Diagnostician Platform**. It is designed to prepare you for college project evaluations, code reviews, and technical viva examinations.

---

## 1. `requirements.txt`

### Purpose and Responsibility
Defines the exact software dependencies required to run the project. Ensures that any examiner, peer, or deployment server can replicate the exact Python environment.

### Key Libraries Specified:
- `numpy`: Fast multidimensional vector and matrix operations, vector distances, and numerical statistics.
- `pandas (<3.0.0)`: Tabular data handling, dataset ingestion, CSV preprocessing, and feature discretization. Pinned below version 3 to ensure compatibility with Streamlit.
- `scipy`: Statistical computations, probability distributions, variance calculations.
- `chromadb`: Direct local vector database for storing document embeddings, metadata, and executing Top-K semantic similarity searches via HNSW indexing.
- `scikit-learn`: Implementation of classical ML algorithms (Logistic Regression, Linear/RBF SVM, Decision Tree, Random Forest, KNN), PCA dimensionality reduction, preprocessing pipelines, and evaluation metrics.
- `mlxtend`: Apriori association rule mining algorithm for extracting interpretable failure diagnostic patterns.
- `rouge-score` & `nltk`: Pure statistical/lexical NLP metrics (ROUGE-L Longest Common Subsequence, tokenization) without requiring external LLMs.
- `streamlit`: Web dashboard framework for interactive visual analytics.
- `plotly`: Interactive 2D/3D visualizations (e.g. 3D PCA embedding latent space, ranked distance curves).
- `joblib`: Model serialization and persistence to disk.

---

## 2. `.gitignore`

### Purpose and Responsibility
Prevents temporary runtime artifacts, bytecode caches (`__pycache__`), virtual environment directories (`.venv/`, `env/`), operating system metadata (`.DS_Store`, `Thumbs.db`), and IDE configuration directories from being committed to Git version control.

---

## 3. `config/config.py`

### Purpose and Responsibility
Acts as the **Single Source of Truth** for global constants, filesystem paths, hyperparameter defaults, and category taxonomies. Centralizing configuration prevents magic strings, hardcoded directories, and accidental mismatch between modules.

### Detailed Breakdown:
- **`PROJECT_ROOT`**: Computes the absolute path to the project root directory dynamically using `Path(__file__).resolve().parent.parent`. This guarantees cross-platform path resolution whether running on Windows, Linux, or macOS.
- **Directory Paths**:
  - `DATA_DIR`: Root data storage folder.
  - `RAW_DATA_DIR`: Destination for raw user-uploaded CSV files.
  - `PROCESSED_DATA_DIR`: Destination for extracted feature datasets (`extracted_features.csv`).
  - `SYNTHETIC_DATA_DIR`: Storage for controlled synthetic benchmark datasets.
  - `CHROMA_PERSIST_DIR`: Local persistent directory storing the ChromaDB vector database.
  - `MODELS_DIR`: Directory where trained Scikit-Learn pipelines are saved (`best_diagnostic_model.pkl`).
- **`FAILURE_CATEGORIES`**:
  Defines the 5 mutually distinguishable failure classes:
  1. `Retrieval Miss`: The vector database does not contain the necessary ground-truth information.
  2. `Ranking Error`: Ground-truth information exists in the database but ranks below the baseline retrieval cutoff (e.g. ranks 6–100 instead of Top-5).
  3. `Context Contamination`: The retrieved Top-5 chunks contain irrelevant or noisy distractor text that misleads the generator.
  4. `Hallucination`: The retrieved Top-5 chunks contain the necessary factual evidence, but the generation model fabricates ungrounded information.
  5. `Contradiction`: The retrieved Top-5 chunks contain mutually conflicting factual assertions, causing confusion or contradiction.
- **Retrieval Defaults**:
  - `TOP_K_BASELINE = 5`: Simulates the standard operational retrieval budget in common RAG systems.
  - `TOP_K_DEEP_FETCH = 100`: Extended retrieval depth used to inspect database gaps and ranking demotions.
- **Reproducibility**:
  - `RANDOM_STATE = 42`: Ensures deterministic dataset splitting, cross-validation, and model weight initialization.

---

## 4. `data/synthetic/generate_synthetic_data.py`

### Purpose and Responsibility
Generates a controlled, self-contained testing environment consisting of:
1. A local persistent ChromaDB collection (`academic_rag_corpus`) with 157 documents across 6 academic disciplines.
2. A controlled CSV dataset (`sample_failed_queries.csv`) of 50 failed queries with exact ground-truth failure labels (10 queries per category).

### Key Functions and Workflow:

#### `build_corpus_documents()`
- **Responsibility**: Constructs a list of document dictionaries containing `id`, `text`, and `domain`.
- **Content Structure**:
  - *Domain 1: Machine Learning & AI*: 15 foundational concepts (Supervised/Unsupervised learning, SVM, PCA, RAG, Cosine similarity, etc.).
  - *Targeted Ranking Error Documents*: 5 specialized ground-truth documents (e.g. Kepler launch date, Haber-Bosch process, Ada Lovelace algorithm).
  - *Targeted Contradiction Pairs*: 6 documents containing deliberate factual conflicts (e.g., Neanderthal extinction date Alpha [40,000 BP] vs. Beta [28,000 BP]; Coster 1439 vs. Gutenberg 1440; Drug Z9 safety reports).
  - *Distractor Context Contamination Pairs*: 6 documents with polysemous terms (e.g. Python language vs. Python snake; Apple company vs. Apple fruit; Java software vs. Java island).
  - *Domains 2–6*: 125 documents spanning Astrophysics, Molecular Biology, Quantum Physics, Climatology, and Economic History.
- **Returns**: List of 157 document dictionaries.

#### `populate_chromadb()`
- **Responsibility**: Connects to ChromaDB via `chromadb.PersistentClient`, initializes the `academic_rag_corpus` collection with cosine distance metric (`{"hnsw:space": "cosine"}`), and inserts all 157 documents in batches of 50.
- **Why Batches**: Prevents memory spikes and simulates standard vector database bulk indexing.

#### `generate_synthetic_queries()`
- **Responsibility**: Creates 50 queries across the 5 failure classes.
- **Schema Generated**:
  - `query_id`: Unique identifier (e.g., `FAIL_001` to `FAIL_050`).
  - `query`: The user prompt.
  - `expected_answer`: Ground-truth reference text.
  - `generated_answer`: The faulty output produced by the RAG model.
  - `failure_label`: Target class label.
  - `provenance`: Origin indicator (`synthetic_controlled_v1`).
  - `target_doc_id`: ID of the relevant document in ChromaDB (`NONE` for Retrieval Miss).
  - `notes`: Detailed causal explanation for review.

---

## 5. `app.py`

### Purpose and Responsibility
The main user interface script launched via `streamlit run app.py`. Acts as the central dashboard hub.

### Key Components:
- **Page Configuration**: Configures page title, icon, and wide responsive layout.
- **System Verification Dashboard**: Checks Python version, ChromaDB persistence status, synthetic dataset availability, and number of diagnostic classes.
- **Live Vector DB Inspection**: Connects to ChromaDB, lists available collections, and displays document counts.
- **Dataset Preview**: Loads and displays `sample_failed_queries.csv` with class distribution tables.
- **Academic Disclaimer**: Clearly displays the distinction between empirical correlation and mathematical causality.

---

## 6. `modules/data_ingestion.py`

### Purpose and Responsibility
Serves as the data ingestion gatekeeper for the platform. It accepts raw CSV data from local disk or in-memory upload buffers, standardizes column nomenclature, validates schema integrity against mandatory and recommended fields, sanitizes string data, detects and removes duplicates, validates target failure labels against the 5-class taxonomy, and produces an audited `DataQualityReport`.

### Major Classes and Functions:

#### `DataQualityReport` (Dataclass)
- **Responsibility**: Strongly-typed container encapsulating all data-quality metadata and resulting datasets.
- **Attributes**:
  - `total_records_ingested: int`: Raw count of rows parsed from CSV.
  - `valid_records_count: int`: Number of rows that passed all quality checks.
  - `invalid_records_count: int`: Number of rows segregated due to errors.
  - `duplicate_records_count: int`: Number of duplicate queries detected and dropped.
  - `missing_values_by_column: Dict[str, int]`: Dictionary mapping each column name to its missing value count.
  - `failure_category_distribution: Dict[str, int]`: Class frequency dictionary.
  - `validation_messages: List[str]`: Informational logs (e.g. schema renames).
  - `validation_warnings: List[str]`: Non-critical warnings (e.g. missing optional fields).
  - `validation_errors: List[str]`: Critical errors (e.g. missing `query` column).
  - `is_valid_dataset: bool`: Flag indicating whether the dataset has >=1 valid records.
  - `has_labels: bool`: Flag indicating whether ground-truth failure labels are available.
  - `cleaned_df: pd.DataFrame`: Validated, cleaned tabular dataset ready for feature extraction.
  - `invalid_df: pd.DataFrame`: Segregated invalid rows annotated with specific `rejection_reason`.

#### `standardize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]`
- **Responsibility**: Resolves column naming inconsistencies across diverse RAG benchmarks. Maps user headers (`prompt`, `reference_answer`, `rag_response`, etc.) to internal canonical keys (`query`, `expected_answer`, `generated_answer`, etc.).
- **Returns**: Renamed dataframe and dictionary mapping of renamed columns.

#### `validate_rag_dataset(df_raw, default_provenance, drop_duplicates) -> DataQualityReport`
- **Responsibility**: Orchestrates the 10-step cleaning and validation pipeline:
  1. Standardizes columns.
  2. Verifies mandatory columns (`query`).
  3. Audits recommended columns (`expected_answer`, `generated_answer`, `failure_label`).
  4. Removes all-NaN rows.
  5. Strips leading/trailing whitespace and normalizes empty strings to `np.nan`.
  6. Filters out empty or missing query strings into `invalid_df`.
  7. Audits `failure_label` against `FAILURE_CATEGORIES`; filters unrecognized labels into `invalid_df`.
  8. Detects duplicate query strings and drops duplicates if configured.
  9. Preserves or assigns provenance information (`default_provenance`).
  10. Ensures unique, formatted `query_id` values.

#### `load_csv(file_or_path, default_provenance, drop_duplicates) -> DataQualityReport`
- **Responsibility**: Polymorphic loader accepting file paths or file buffers from Streamlit's `file_uploader`, wraps Pandas reading in error handling, and delegates to `validate_rag_dataset`.

---

## 7. `pages/2_Data_Ingestion.py`

### Purpose and Responsibility
Interactive Streamlit page dedicated to data ingestion and validation reporting.

### Key Components:
- **Data Source Selector**: Radio button allowing users to switch between the controlled synthetic benchmark and custom user CSV uploads.
- **Top-Level KPI Metric Cards**: Displays Ingested Count, Valid Count, Invalid Count, Duplicates Dropped, and Label Status.
- **Validation Audit Panels**: Expandable panels displaying critical validation errors, data quality warnings, and schema normalization messages.
- **Missing Values Audit Table**: Displays exact missing value counts and percentage per column.
- **Plotly Class Distribution Chart**: Interactive bar chart displaying the frequency and percentage breakdown of ground-truth failure categories.
- **Valid and Invalid Records Dataframes**: Interactive tables allowing inspection of cleaned data and rejected rows with rejection reasons.
- **Data Export & Persistence**: One-click download button and local file persistence to `data/processed/validated_queries.csv`.

---

## 8. `tests/test_preprocessing.py`

### Purpose and Responsibility
Automated Pytest verification suite testing data validation rules and edge cases.

### Test Cases Covered:
1. `test_synthetic_dataset_ingestion`: Verifies `sample_failed_queries.csv` loads with 100% validity, exactly 50 records, 10 per category, and intact provenance.
2. `test_missing_mandatory_query_column`: Verifies rejection when `query` is missing.
3. `test_column_aliasing`: Confirms aliases (`prompt`, `reference_answer`, `rag_response`, `category`, `id`) are normalized properly.
4. `test_empty_and_whitespace_query_handling`: Confirms blank and whitespace queries are segregated with rejection reasons.
5. `test_duplicate_query_removal`: Confirms duplicates are identified and eliminated from the clean dataset.
6. `test_invalid_failure_category`: Confirms non-taxonomy labels are quarantined.
7. `test_unlabeled_mode`: Confirms datasets without labels function correctly in inference mode.

---

## 9. `modules/chroma_connector.py`

### Purpose and Responsibility
Provides direct, offline connectivity to local ChromaDB persistent vector storage without middleware or paid external APIs. It enables collection health auditing, batch Top-100 deep-fetch retrieval, document rank preservation, distance metric extraction, and Top-5 vs Deep-Fetch result partitioning.

### Major Functions and Methods:

#### `ChromaConnector.__init__(persist_dir)`
- **Responsibility**: Verifies directory existence and establishes a `chromadb.PersistentClient`. Raises `FileNotFoundError` if the directory does not exist.

#### `list_collections() -> List[str]`
- **Responsibility**: Queries the vector database for all active collection names.

#### `get_collection_info(collection_name) -> Dict[str, Any]`
- **Responsibility**: Retrieves collection statistics including document count, HNSW indexing space (`cosine`, `l2`, `ip`), and user metadata.

#### `retrieve_query(collection_name, query_text, top_k) -> List[Dict[str, Any]]`
- **Responsibility**: Executes vector nearest-neighbor search for a single query text. Retrieves up to `top_k` documents (bounded by collection size). For every document, constructs a record with:
  - `rank`: 1 to $K$ (monotonically non-decreasing order of distance).
  - `doc_id`: Unique document ID.
  - `document`: Raw text content.
  - `metadata`: Domain and document length metadata.
  - `distance`: Float distance metric from Chroma.
  - `distance_metric`: Metric used (`cosine`, `l2`, `ip`).
  - `similarity_score`: Properly converted similarity value respecting the metric ($1 - D$ for cosine, clamped to $[0, 1]$).
  - `is_top_5`: Boolean flag ($rank \le 5$).

#### `partition_retrieval_results(retrieval_results, baseline_k) -> Tuple[List, List, List]`
- **Responsibility**: Splits retrieval results into operational baseline (`top_5`), extended retrieval (`ranks_6_to_100`), and combined list (`all_results`).

#### `retrieve_for_dataset(df_queries, collection_name, top_k) -> Tuple[pd.DataFrame, Dict]`
- **Responsibility**: Executes batch retrieval for every query in a DataFrame. Identifies target document ranks (`target_rank`) to detect ranking errors. Returns a 5,000-row detailed DataFrame and a query-indexed dictionary for feature extraction.

#### `save_detailed_results(detailed_df, output_path) -> Path`
- **Responsibility**: Persists detailed retrieval records to CSV (`data/processed/detailed_retrieval_results.csv`).

#### `ensure_or_regenerate_audit_dataset(queries_csv, output_path, top_k, force_regenerate) -> Tuple[pd.DataFrame, bool]`
- **Responsibility**: Self-healing recovery mechanism that checks whether `detailed_retrieval_results.csv` exists on disk. If missing or forced, executes batch Top-100 deep-fetch retrieval across all queries using local ChromaDB and serializes the resulting audit DataFrame. Returns `(df, was_regenerated)`.

---

## 10. `pages/3_Retrieval_Analysis.py`

### Purpose and Responsibility
Interactive Streamlit UI for inspecting vector retrieval behavior on a query-by-query basis.

### Key Components:
- **Query Filter and Selector**: Filter queries by failure category and select specific interactions.
- **Interaction Context Card**: Displays query prompt, expected ground truth, generated model response, and target document ID.
- **Retrieval Metric Overview**: Displays Top-5 Mean Distance, Top-1 Min Distance, Ranks 6–100 Mean Distance, Depth Distance Shift, and Target Document Rank.
- **Diagnostic Alert Engine**: Surfaces real-time warnings (e.g. Ranking Error alerts when target document is found at rank > 5; Retrieval Miss alerts when target is absent from Top-100).
- **Tab 1 (Top-5 Baseline Chunks)**: Displays the operational Top-5 chunks supplied to the RAG generator with cosine distance and similarity metrics.
- **Tab 2 (Deep-Fetch Chunks)**: Interactive rank window slider (e.g. ranks 6–25) with golden badge highlighting target documents found in deep fetch.
- **Tab 3 (Ranked Distance Curve)**: Plotly interactive curve showing distance vs rank, Top-5 vertical cutoff line, and target document star marker.
- **Tab 4 (Full Tabular Data)**: Complete 100-chunk table with CSV export.

---

## 11. `tests/test_chroma_connector.py`

### Purpose and Responsibility
Pytest test suite verifying ChromaDB connector functionality and retrieval integrity.

### Test Cases Covered:
1. `test_chroma_connection`: Verifies persistent client connection and discovery of `academic_rag_corpus`.
2. `test_collection_metadata`: Verifies document count ($\ge 120$) and cosine distance metric configuration.
3. `test_single_query_top_100_retrieval`: Verifies 100 documents returned, strict 1-to-100 sequential ranking, distance monotonicity, and bounded similarity scores $[0, 1]$.
4. `test_top_5_vs_deep_fetch_partitioning`: Verifies exact partitioning into 5 baseline and 95 deep-fetch chunks.
5. `test_batch_retrieval_for_dataset`: Verifies batch execution across queries and column completeness.
6. `test_invalid_collection_error`: Verifies proper `ValueError` handling for non-existent collections.

---

## 12. `modules/feature_extraction.py`

### Purpose and Responsibility
Orchestrates the conversion of raw vector distances and textual chunks retrieved from ChromaDB into a clean, 40-column numerical and categorical feature matrix. This feature dataset serves as the shared input for all downstream analytical modules, including the user's Logistic Regression classifier, the statistical depth analyzer, and the teammate's upcoming classifiers.

### Major Functions:

#### `tokenize_text(text: str) -> set`
- **Responsibility**: Normalizes text into lowercase alphanumeric word tokens using regex `\b\w+\b`, eliminating punctuation bias and returning a set of unique words.

#### `compute_jaccard_similarity(text_a: str, text_b: str) -> float`
- **Responsibility**: Computes the set-theoretic intersection over union of word tokens:
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
  Returns a float in $[0.0, 1.0]$. Returns $0.0$ gracefully if either string is empty or null.

#### `compute_rouge_l(reference: str, candidate: str) -> float`
- **Responsibility**: Uses `rouge_score.rouge_scorer.RougeScorer` to evaluate the Longest Common Subsequence (LCS) F1 score. Evaluates sentence-level structural preservation without requiring consecutive n-gram matches.

#### `calculate_distribution_stats(distances: List[float], prefix: str) -> Dict[str, float]`
- **Responsibility**: Computes descriptive statistics over any array of retrieval distances: Mean ($\mu$), Variance ($\sigma^2$), Standard Deviation ($\sigma$), Minimum, Maximum, and Spread ($Max - Min$). Returns a formatted dictionary with the designated feature prefix.

#### `extract_features_for_single_query(query_data: Dict[str, Any]) -> Dict[str, Any]`
- **Responsibility**: Aggregates all distance features, text similarity features, and retrieval coverage indicators for a single query into a flat dictionary.
- **Key Features Extracted**:
  - Distance: `top5_dist_mean`, `top5_dist_std`, `top5_dist_var`, `top5_dist_min`, `top5_dist_max`, `top5_dist_spread`, `deep_dist_mean`, `deep_dist_std`, `deep_dist_var`, `deep_dist_min`, `deep_dist_max`, `deep_dist_spread`, `all_dist_mean`, `all_dist_std`, `all_dist_var`, `all_dist_min`, `all_dist_max`, `all_dist_spread`, `dist_diff_deep_vs_top5`, `dist_dropoff_ratio`, `top5_l2_mean_equiv`, `all_l2_mean_equiv`.
  - Similarity: `jaccard_expected_vs_top1`, `jaccard_expected_vs_top5`, `jaccard_expected_vs_generated`, `jaccard_generated_vs_top5`, `rougel_expected_vs_top1`, `rougel_expected_vs_top5`, `rougel_expected_vs_generated`, `rougel_generated_vs_top5`.
  - Coverage: `num_retrieved_docs`, `target_doc_rank`, `target_in_top5`, `target_in_deep_fetch`, `is_database_gap`, `retrieval_overlap_top5_top100`.

#### `extract_features_pipeline(df_queries, connector, collection_name, top_k) -> pd.DataFrame`
- **Responsibility**: Batch orchestrator executing retrieval and feature extraction across all rows of an input DataFrame.

#### `save_extracted_features(feature_df, output_path) -> Path`
- **Responsibility**: Persists the feature DataFrame to `data/processed/extracted_features.csv`.

---

## 13. `tests/test_feature_extraction.py`

### Purpose and Responsibility
Pytest suite verifying numerical accuracy, edge-case safety, and data format compliance of the feature extraction pipeline.

### Test Cases Covered:
1. `test_tokenize_text`: Tests tokenization, lowercasing, and empty string handling.
2. `test_jaccard_similarity`: Tests identity ($1.0$), disjointness ($0.0$), partial overlap ($0.333$), and null resilience.
3. `test_rouge_l`: Tests identical strings ($1.0$), disjoint strings ($0.0$), and null resilience.
4. `test_distribution_stats`: Tests arithmetic mean, variance, standard deviation, min, max, spread on known values, and empty array handling.
5. `test_single_query_feature_extraction`: Tests dictionary generation on mock retrieval data.
6. `test_extracted_features_csv_exists`: Confirms `extracted_features.csv` has 50 rows, 40 columns, and 5 balanced failure categories.

---

## 14. `modules/feature_selection.py`

### Purpose and Responsibility
Provides feature selection strategies to reduce feature dimensionality from 36 numerical features down to the most discriminative predictors, preventing overfitting and multicollinearity in the user's Logistic Regression model.

### Key Classes:
- **`FilterFeatureSelector(k=15)`**:
  - Computes univariate ANOVA F-statistics ($F = \text{MSB} / \text{MSW}$) between each continuous feature and the categorical failure labels.
  - Selects the top $k$ features based on statistical significance without relying on model weights.
  - Generates ranked summary table (`get_ranked_feature_summary()`).
- **`EmbeddedL1FeatureSelector(C=1.0)`**:
  - Fits an $L_1$-regularized (Lasso) sparse Multinomial Logistic Regression model using the SAGA solver on standardized training data.
  - Drives non-informative coefficients to exactly zero.
  - Features with mean absolute coefficient magnitude $> 10^{-4}$ across all classes are selected.
  - Produces coefficient summary dataframe (`get_coefficient_summary()`).

---

## 15. `modules/diagnostic_models.py`

### Purpose and Responsibility
The core machine learning module for supervised RAG failure classification. Implements the user's personal assignment: **Multinomial Logistic Regression**, along with modular placeholder classes for the teammate's 5 models.

### Key Components:
- **`BaseDiagnosticModel`**: Abstract base class declaring the standard Scikit-Learn API: `fit(X, y)`, `predict(X)`, `predict_proba(X)`.
- **Teammate Model Placeholders**: `LinearSVMPlaceholder`, `RBFSVMPlaceholder`, `DecisionTreePlaceholder`, `RandomForestPlaceholder`, `KNNPlaceholder` (raise `NotImplementedError` with clear team allocation messages).
- **`LogisticRegressionDiagnosticEngine`**:
  - `prepare_data()`: Splits the 50 queries into 37 training samples and 13 stratified test samples.
  - `run_scaling_experiment()`: Executes 5-fold Stratified CV comparing `StandardScaler` vs `MinMaxScaler` on training data.
  - `run_feature_selection_experiment()`: Compares All 36 features vs Filter Top 15 vs Embedded L1 Top 15 using 5-fold Stratified CV.
  - `train_best_model()`: Fits the final `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])` on the training set using the best feature subset.
  - `evaluate_on_test_set()`: Evaluates on the unseen 13-sample holdout test set; computes Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Multiclass Log Loss, and Confusion Matrix.
  - `save_model()`: Serializes the fitted pipeline and metadata dictionary via Joblib to `models/best_diagnostic_model.pkl`.
  - `predict()` and `predict_proba()`: Infers failure class and calibrated Softmax probabilities for new queries.

---

## 16. `modules/train_logistic_model.py`

### Purpose and Responsibility
Standalone command-line runner script executing the entire Phase 4 Logistic Regression pipeline, printing detailed progress logs, experimental comparison tables, confusion matrices, top feature coefficients per class, and serializing the model.

---

## 17. `pages/6_ML_Model_Comparison.py`

### Purpose and Responsibility
Interactive Streamlit page visualizing the user's machine learning results:
- Top-level KPI cards (Accuracy, Macro F1, Selected Features).
- Tab 1: Scaling Comparison (StandardScaler vs MinMaxScaler table and bar chart).
- Tab 2: Feature Selection (All vs Filter vs Embedded L1 table and ranked feature lists).
- Tab 3: Test Evaluation & Confusion Matrix Heatmap (Plotly).
- Tab 4: Feature Coefficient Interpretability (interactive horizontal bar chart showing positive and negative drivers per failure class).
- Tab 5: Team Allocation & Pluggable Architecture (table displaying user's model vs teammate's reserved slots).

---

## 18. `pages/7_Failure_Diagnosis.py`

### Purpose and Responsibility
Interactive diagnostic console allowing users to select any failed query and receive:
- Predicted failure category.
- Calibrated Softmax probability distribution bar chart.
- Key retrieval and distance statistics.
- Tailored, actionable engineering recommendations (e.g. BM25 hybrid search for Ranking Errors, corpus expansion for Retrieval Misses, prompt constraints for Hallucinations).

---

## 19. `tests/test_diagnostic_models.py`

### Purpose and Responsibility
Pytest suite covering:
1. Stratified train-test split balance across all 5 classes.
2. Scaling comparison execution.
3. Filter feature selection functionality.
4. Embedded L1 feature selection functionality.
5. Final pipeline fitting and probability normalization ($\sum P = 1.0$).
6. Test evaluation metrics and confusion matrix dimensions.
7. Joblib model serialization and deserialization.
8. Teammate model placeholder `NotImplementedError` verification.

---

## 20. `modules/statistical_analyzer.py`

### Purpose and Responsibility
Serves as the deep-fetch mathematical analysis engine. It processes distance distributions from ChromaDB Top-100 retrievals, isolates the Top-5 operational tier from deeper ranks (6–100), computes parametric and non-parametric descriptive statistics, measures pairwise lexical coherence, computes entropy and curve curvature, and generates empirical diagnostic observations.

### Major Functions and Classes:

#### `compute_distribution_metrics(distances, prefix) -> Dict[str, float]`
- **Responsibility**: Computes Mean ($\mu$), Median ($Q_2$), Variance ($\sigma^2$), Standard Deviation ($\sigma$), Minimum, Maximum, Spread ($Max - Min$), Interquartile Range ($\text{IQR} = Q_3 - Q_1$), Fisher-Pearson Skewness, and Euclidean L2 equivalents. Handles empty arrays gracefully by returning NaNs.

#### `compute_pairwise_top5_coherence(top5_docs) -> float`
- **Responsibility**: Computes the mean Jaccard token similarity across all $\binom{5}{2} = 10$ pairs of Top-5 retrieved chunks. Low coherence ($< 0.15$) with high distance variance flags Context Contamination.

#### `compute_shannon_entropy(distances, n_bins) -> float`
- **Responsibility**: Bins continuous retrieval distances into discrete intervals and evaluates Shannon information entropy:
  $$H = -\sum_{i=1}^B p_i \log_2(p_i)$$
  Measures the uncertainty and dispersion of distance scores across the retrieval horizon.

#### `compute_elbow_rank(distances) -> int`
- **Responsibility**: Applies perpendicular distance optimization between the curve endpoints and all intermediate rank coordinates to identify the rank where retrieval distance growth bends or plateaus.

#### `StatisticalAnalyzer.analyze_query_retrieval(...) -> Dict[str, Any]`
- **Responsibility**: Main query analysis pipeline: computes Top-5 vs Deep-Fetch statistics, distance shifts, drop-off ratios, pairwise coherence, target document rank positions, and lexical similarity metrics.

#### `StatisticalAnalyzer.generate_diagnostic_insights(...) -> List[Dict[str, str]]`
- **Responsibility**: Evaluates calculated metrics against empirical thresholds to generate actionable engineering alerts (e.g. Ranking Demotion, High Distance Outlier, Context Contamination, Ungrounded Generation).

#### `StatisticalAnalyzer.compute_aggregate_dataset_statistics(...) -> Dict[str, Any]`
- **Responsibility**: Audits global retrieval behavior across all 5,000 document records in `detailed_retrieval_results.csv`, computing global distance averages and category-specific profiles.

#### Visualization Functions:
- `plot_top5_vs_deep_histogram()`: Overlapping Plotly histogram comparing Top-5 and deep-fetch distance distributions.
- `plot_ranked_distance_curve()`: Plotly line chart of rank vs distance with Top-5 cutoff line and target document star marker.
- `plot_box_comparison()`: Comparative box plots showing Top-5 vs Deep-Fetch dispersion.
- `plot_category_aggregate_boxplots()`: Cross-category box plot of Top-5 distances across failure modes.

---

## 21. `pages/4_Statistical_Analyzer.py`

### Purpose and Responsibility
Interactive Streamlit dashboard allowing users to explore statistical depth metrics at both the individual query level and the aggregate dataset level.

### Key Features:
- **Mode Toggle**: Switch between Query-Level Deep-Fetch Analysis and Aggregate Dataset Statistics (50 Queries, 5,000 Chunks).
- **Self-Healing CSV Fallback**: Automatically detects missing `detailed_retrieval_results.csv`, renders an informative warning banner, and provides a prominent Streamlit button to regenerate the 5,000-chunk retrieval audit dataset directly from ChromaDB without crashing.
- **Maintenance Expander**: Allows manual force-refresh of the 5,000-record retrieval audit dataset when needed.
- **Robust Formatting Engine**: Employs `fmt()` helper across all tables and metrics, guaranteeing zero runtime crashes on NaN values, missing labels, or edge cases with fewer than 5 documents.
- **Descriptive Statistics Comparison Table**: Side-by-side comparison of Top-5 Baseline vs Ranks 6–100 Deep Fetch vs Global Top-100.
- **Advanced Indices Expander**: Displays Shannon Entropy, Elbow Curvature Rank, Top-5 Spread, and Similarity Drop-off.
- **Statistical Diagnostic Observations Card**: Surfaces findings, implications, and corrective actions.
- **Interactive Plotly Visualizations**:
  - Distance Distribution Histogram (Overlapping bars).
  - Ranked Distance Progression Curve (Cutoff at Rank 5 with Target Document badge).
  - Comparative Box Plots.
- **Aggregate Dataset View**: Global KPIs, category-level distance breakdown table, and cross-category box plots.

---

## 22. `tests/test_statistical_analyzer.py`

### Purpose and Responsibility
Pytest test suite verifying numerical correctness, mathematical identities, edge cases, and robustness of the statistical analyzer.

### Test Cases Covered:
1. `test_compute_distribution_metrics_known_values`: Tests exact values on known array `[0.2, 0.4, 0.6, 0.8, 1.0]`.
2. `test_compute_distribution_metrics_empty`: Tests NaN resilience on empty distance arrays.
3. `test_top5_coherence`: Tests pairwise coherence on identical ($1.0$), disjoint ($0.0$), and single-element inputs.
4. `test_shannon_entropy`: Tests entropy non-negativity and single-value behavior.
5. `test_elbow_curvature_rank`: Tests curvature rank detection on sharp-bend curves.
6. `test_analyzer_query_breakdown`: Tests end-to-end query analysis and Ranking Demotion insight generation.
7. `test_aggregate_dataset_statistics`: Tests dataset-level aggregation on 5,000 detailed records.
8. `test_ensure_or_regenerate_audit_dataset`: Tests self-healing CSV regeneration on temporary path and verifies caching.
9. `test_l2_euclidean_unit_norm_identity`: Tests geometric identity $\|u - v\|_2 = \sqrt{2 \cdot d_{\text{cosine}}}$ on boundary cosine distances $0.0$, $0.5$, and $2.0$.
10. `test_edge_case_fewer_than_five_docs`: Tests graceful degradation and NaN handling when retrieval contains $< 5$ chunks.

---

## 23. `modules/latent_space_visualizer.py`

### Purpose and Responsibility
High-dimensional dimensionality reduction and clustering engine for analyzing corpus structure and vector gaps. Ingests all 157 documents from local ChromaDB with 384-dimensional dense vectors, applies 3D PCA, evaluates K-Means partitions ($K=2 \dots 10$), and fits DBSCAN density clusters.

### Major Functions and Classes:

#### `load_corpus_embeddings(...) -> Tuple[np.ndarray, List, List, List, List]`
- **Responsibility**: Retrieves 157 dense 384D embeddings, document IDs, metadata, domain labels, and raw chunk texts from ChromaDB. Handles NumPy arrays safely and validates matrix dimensions.

#### `apply_pca_3d(embeddings, random_state, standardize) -> Tuple[np.ndarray, PCA, Optional[StandardScaler]]`
- **Responsibility**: Projects 384D vectors to 3 principal components. Supports Centered PCA (mean-centering without feature scaling, recommended for unit-norm embeddings) and Standardized PCA.

#### `compare_pca_scaling_strategies(embeddings) -> Dict[str, Any]`
- **Responsibility**: Compares variance retention of Centered PCA vs Standardized PCA, proving that Centered PCA retains +3.92% more variance on unit-normalized vectors.

#### `run_kmeans_sweep(data_matrix, k_range, random_state) -> Dict[str, Any]`
- **Responsibility**: Sweeps $K \in [2, 10]$ evaluating Within-Cluster Sum of Squares (Inertia) and Mean Silhouette Scores. Includes safe check requiring $\ge 2$ clusters for silhouette evaluation.

#### `run_dbscan(data_matrix, eps, min_samples) -> Tuple[np.ndarray, Dict[str, Any]]`
- **Responsibility**: Identifies dense semantic neighborhoods and flags sparse noise points (`label = -1`) representing potential database coverage gaps.

#### `LatentSpaceVisualizer.run(...) -> Dict[str, Any]`
- **Responsibility**: High-level orchestrator executing the full latent space pipeline, generating diagnostic insights, and producing exportable document DataFrames.

#### Visualization Generators:
- `plot_pca_3d_by_domain()`: 3D scatter plot colored by knowledge domain with rich hover text snippets.
- `plot_pca_3d_by_kmeans()`: 3D scatter plot colored by K-Means cluster partitions.
- `plot_pca_3d_by_dbscan()`: 3D scatter plot with prominent red diamond markers highlighting sparse noise points.
- `plot_elbow_curve()`: Inertia vs K with annotated optimal K threshold.
- `plot_silhouette_scores_bar()`: Bar chart of mean silhouette scores with gold accent on optimal K.
- `plot_dbscan_eps_sensitivity()`: Dual-axis sensitivity curve (Cluster count and Noise % vs $\epsilon$).
- `plot_pca_explained_variance()`: Component variance bar chart with cumulative percentage line.

---

## 24. `pages/5_Latent_Space_Visualizer.py`

### Purpose and Responsibility
Interactive multi-page Streamlit dashboard visualizing the 384-dimensional corpus embedding space.

### Key Features:
- **Configuration Controls**: Choose clustering space (Full 384D vs 3D PCA coordinates), toggle PCA standardization, adjust DBSCAN $\epsilon$ and $\text{MinPts}$, and manually select K-Means K.
- **KPI Summary Cards**: Total documents (157), original dimensions (384D), 3D explained variance (53.4%), optimal K, and DBSCAN noise count.
- **PCA Variance Decomposition**: Interactive bar chart and mathematical explanation of unit-norm geometric preservation.
- **3D Latent Space Viewer**: 3 interactive tabs (Domain Coloring, K-Means Clusters, DBSCAN Outliers) with 3D rotation, zoom, and rich snippet hover tooltips.
- **Cluster Diagnostics**: Side-by-side Elbow Method and Silhouette Score bar charts alongside DBSCAN parameter sensitivity curves.
- **Empirical Insights Card**: Rule-based alerts distinguishing geometric sparsity from functional RAG failure.
- **Interactive Document Inspector**: Filter documents by domain, cluster, or noise status; inspect raw text, char length, and 3D PCA coordinates.

---

## 25. `tests/test_latent_space.py`

### Purpose and Responsibility
Pytest test suite verifying numerical correctness, dimensionality preservation, and edge-case handling of the latent space visualizer.

### Test Cases Covered:
1. `test_load_corpus_embeddings_integrity`: Validates 157 documents, 384 dimensions, text presence, and $L_2 = 1.0$ unit norms.
2. `test_pca_3d_output_dimensions_and_variance`: Verifies 3D output shape and descending variance ratios ($PC1 \ge PC2 \ge PC3$).
3. `test_compare_pca_scaling_strategies`: Validates that centered PCA preserves more variance than standardized PCA.
4. `test_kmeans_sweep_inertia_monotonicity_and_silhouette`: Verifies monotonic inertia decrease and valid silhouette bounds $[-1, 1]$.
5. `test_kmeans_best_and_k5_assignments`: Validates K=5 partitions and label ranges.
6. `test_dbscan_clustering_and_noise_detection`: Tests cluster discovery and noise point detection.
7. `test_dbscan_edge_cases_all_noise_and_single_cluster`: Verifies extreme parameter handling without exceptions.
8. `test_pca_edge_case_insufficient_samples`: Tests ValueError on $<3$ samples.
9. `test_latent_space_visualizer_orchestrator_end_to_end`: Validates end-to-end execution and export DataFrame format.

---

## 26. `modules/apriori_analyzer.py`

### Purpose and Responsibility
Core analytical engine for association rule mining. Discretizes the 40 continuous retrieval features into categorical intervals (`Low`, `Medium`, `High`), formats transactions into one-hot boolean matrices, executes frequent itemset mining via Apriori, computes association rule metrics (Support, Confidence, Lift, Leverage, Conviction), extracts category-specific failure diagnostic rules, and produces interactive Plotly charts.

### Major Functions and Classes:

#### `discretize_features(...) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]`
- **Responsibility**: Bins continuous features into ordinal intervals (`Low`, `Medium`, `High`). Supports equal-width (uniform) and equal-frequency (quantile) strategies. Automatically identifies and drops constant numeric columns (`nunique <= 1`) and imputes missing values. Excludes identifier columns (`query_id`, `query`, `provenance`).

#### `encode_transactions(...) -> pd.DataFrame`
- **Responsibility**: Converts discretized features into a boolean transaction matrix. Encodes target failure labels as explicit categorical tokens (`failure_category=<Category>`).

#### `mine_frequent_itemsets(...) -> pd.DataFrame`
- **Responsibility**: Applies `mlxtend.frequent_patterns.apriori` using the downward-closure (Apriori) property to discover itemsets with support $\ge \text{min\_support}$.

#### `generate_diagnostic_rules(...) -> pd.DataFrame`
- **Responsibility**: Generates association rules from frequent itemsets. Calculates Support, Confidence, Lift, Leverage, and Conviction. Filters rules specific to RAG failure categories (where consequent is a single failure category).

#### `filter_rules_by_category(rules_df, category) -> pd.DataFrame`
- **Responsibility**: Returns subset of association rules targeting a specific failure category (Retrieval Miss, Ranking Error, Context Contamination, Hallucination, Contradiction).

#### `explain_rule(rule_row) -> str`
- **Responsibility**: Translates mathematical rule metrics into an academic, plain-language diagnostic explanation while emphasizing the distinction between statistical co-occurrence and causality.

#### `AprioriDiagnosticEngine`
- **Responsibility**: Orchestrator class executing the entire discretization and rule-mining pipeline via `.fit()`. Provides summary KPIs, category rule breakdowns, and clean exportable DataFrames.

#### Visualization Generators:
- `plot_rule_support_confidence_scatter()`: Scatter plot of Support vs Confidence with bubble size representing Lift.
- `plot_rules_count_per_category()`: Categorical bar chart of rules discovered per failure mode.
- `plot_top_rules_bar()`: Horizontal bar chart of top diagnostic rules ranked by Lift.

---

## 27. `pages/8_Apriori_Rules.py`

### Purpose and Responsibility
Interactive multi-page Streamlit dashboard for exploring discovered association rules and frequent itemset patterns.

### Key Features:
- **Interactive Hyperparameter Sidebar**: Sliders for minimum support (0.05–0.50), minimum confidence (0.30–1.00), minimum lift (1.00–5.00), maximum rule length (2 or 3), and discretization strategy toggle.
- **KPI Summary Header**: Total queries analyzed (50), discretized features (33), frequent itemsets mined, total association rules, and targeted diagnostic rules count.
- **Interactive Visualizations**: Support vs Confidence scatter plot, category rule count distribution, and top rules ranking bar chart.
- **Diagnostic Rules Explorer**: Category filter dropdown, top-3 spotlight cards with plain-language explanations, complete sortable rules table, and CSV download.
- **Frequent Itemsets Inspector**: Filter itemsets by length (L1, L2, L3) with support ratios and one-click CSV export.
- **Discretization Methodology Audit**: Details bin boundary equations, lists dropped constant columns, and displays sample discretized transactions.

---

## 28. `tests/test_apriori_analyzer.py`

### Purpose and Responsibility
Pytest test suite verifying feature discretization, transaction encoding, Apriori frequent itemset constraints, metric calculations, and edge-case handling.

### Test Cases Covered:
1. `test_discretize_features_correctness`: Verifies ordinal Low/Med/High binning, column preservation, and identifier exclusion.
2. `test_discretize_handles_missing_and_constant_columns`: Validates constant column dropping and median imputation on NaNs.
3. `test_encode_transactions_boolean_matrix`: Verifies boolean matrix generation and target category formatting.
4. `test_mine_frequent_itemsets_properties`: Tests support thresholds ($\ge \text{min\_support}$) and itemset length constraints ($\le \text{max\_len}$).
5. `test_generate_diagnostic_rules_metrics`: Verifies Support, Confidence, Lift, Leverage, and Conviction boundary properties.
6. `test_category_specific_rules_all_five_categories`: Proves rules are discovered across all 5 failure categories.
7. `test_apriori_invalid_parameters`: Validates ValueError on invalid support bounds ($0.0$, $>1.0$) and empty DataFrames.
8. `test_apriori_empty_and_insufficient_datasets`: Tests graceful fallback on single-row and empty inputs.
9. `test_explain_rule_formatting`: Validates plain-language explanation generation.
10. `test_apriori_engine_export_methods`: Verifies clean export DataFrame generation.


---

## 29. `modules/ui_components.py`

### Purpose and Responsibility
Centralized UI styling and standard branding module for all Streamlit pages. Provides standard visual identity, active page status tracking, system environment health indicators, and academic methodology disclaimers.

### Major Functions:
- `render_sidebar(current_page: str)`: Renders unified platform logo, subtitle, active view badge, and an expandable system environment health card (ChromaDB connection status, synthetic benchmark query count, feature dataset status, and trained model status).
- `render_academic_disclaimer()`: Renders a standardized academic callout emphasizing the epistemological distinction between predictive statistical association and true causal proof in retrieval-augmented systems.

---

## 30. `tests/test_pages_integration.py`

### Purpose and Responsibility
Automated integration test suite for the Phase 6 Streamlit multi-page platform. Verifies headless execution, UI function reliability, and disk artifact persistence.

### Test Cases Covered:
1. `test_ui_components_functions`: Verifies that `render_sidebar` and `render_academic_disclaimer` are callable without errors.
2. `test_all_pages_import_cleanly`: Parametrized test executing top-level import and initialization across all 8 pages (`app`, `pages.2_Data_Ingestion`, `pages.3_Retrieval_Analysis`, `pages.4_Statistical_Analyzer`, `pages.5_Latent_Space_Visualizer`, `pages.6_ML_Model_Comparison`, `pages.7_Failure_Diagnosis`, `pages.8_Apriori_Rules`).
3. `test_critical_artifacts_exist_for_pages`: Validates that ChromaDB directory, synthetic queries CSV, extracted features CSV, and Joblib model artifacts exist on disk.
