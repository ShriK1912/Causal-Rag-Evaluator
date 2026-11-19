# 📑 10-Slide Master Presentation Deck Outline
## Causal RAG Failure Diagnostician Platform
**Course:** ML3003 — Machine Learning  
**Architecture:** Deterministic Classical Statistical Machine Learning & Dimensionality Reduction  
**Team:** 4 Members (Member 1: Data & Vector Geometry | Member 2: Feature Engineering & Selection | Member 3: Supervised Shootout | Member 4: Unsupervised & Cross-Domain)

---

## SLIDE 1: Title Slide (Project Introduction)
- **Header:** Causal RAG Failure Diagnostician Platform
- **Sub-header:** Classical Machine Learning & Statistical Failure Analysis for Retrieval-Augmented Generation
- **Presenter:** Member 1 (Lead Presenter)
- **Visual Layout:**
  - Left side: High-level architectural banner showing ChromaDB $\to$ Feature Extraction $\to$ 6 ML Classifiers $\to$ Root Cause Diagnosis.
  - Right side: Course & Team credentials card.
- **Slide Content:**
  - **Course:** ML3003 — Machine Learning
  - **Core Thesis:** Replacing opaque, circular "LLM-as-a-judge" evaluation with deterministic, statistical ML classifiers ($p=15$ features, 6 supervised models, 3D PCA, and Apriori rule mining).
  - **Team Members & Role Allocation:**
    - *Member 1:* Data Architecture & Retrieval Geometry (Units I & II)
    - *Member 2:* Feature Engineering & Statistical Selection (Units II & V)
    - *Member 3:* Supervised 6-Model Benchmark Shootout (Units III, IV & V)
    - *Member 4:* Unsupervised Space & Domain Generalization (Units V & VI)
- **Speaker Notes (Member 1 - 30s):**
  > "Good morning, respected examiners. Today our team presents the Causal RAG Failure Diagnostician Platform. Retrieval-Augmented Generation has become the industry standard for knowledge-grounded AI, but diagnosing why a RAG pipeline fails remains a massive bottleneck. The common practice of using 'LLM-as-a-judge' is non-deterministic, cost-prohibitive, and conceptually circular. Our project proves that classical statistical Machine Learning can deterministically diagnose RAG failures in milliseconds based on vector distance geometry and lexical overlap."

---

## SLIDE 2: Problem Statement & Motivation
- **Header:** The Fallacy of "LLM-as-a-Judge" in RAG Diagnostics
- **Presenter:** Member 1
- **Visual Layout:**
  - 2-Column Comparison Table (Current Industry Standard vs. Our Classical ML Approach).
  - Bottom Callout Box: The 3 Core Failures of LLM-as-a-Judge.
- **Slide Content:**
  - **The Industry Dilemma:**
    - *Circularity:* Using an LLM to evaluate an LLM creates an ungrounded feedback loop.
    - *Stochastic Variance:* Identical queries yield different diagnostic evaluations depending on temperature and prompt drift.
    - *High Latency & Cost:* API calls cost \$0.01–\$0.03 per diagnostic query with 1.5–3.0 second latency.
  - **Our Solution:** Treat RAG as a **Physical Data-Generating Mechanism**:
    - Query $\to$ Vector Store $\to$ 100-Neighbor Retrieval Horizon $\to$ Mathematical Distance & Lexical Signals $\to$ Deterministic Classical Classifier.
    - Latency: **$<5\text{ ms}$** | Cost: **\$0.00** | Reproducibility: **$100\%$ Deterministic**.
- **Speaker Notes (Member 1 - 30s):**
  > "When a RAG system answers incorrectly, developers usually prompt GPT-4 to ask: 'Did this fail?' This is circular logic. If the generator can hallucinate, the evaluator LLM can hallucinate its evaluation. Furthermore, it treats failure as a binary pass/fail without pinpointing the stage of breakdown. We treat the RAG pipeline as an empirical data generator. By analyzing the vector distance distributions and token overlaps, classical ML classifiers can accurately diagnose the exact failure stage without making a single LLM call."

---

## SLIDE 3: The 5-Class Failure Taxonomy & End-to-End Pipeline
- **Header:** Academic Multi-Class Taxonomy & System Architecture
- **Presenter:** Member 1
- **Visual Layout:**
  - Top: Horizontal pipeline flow diagram ($50 \times 8$ Query Log $\to$ ChromaDB HNSW $\to$ 36 Features $\to$ L1 Selection $\to$ 6 Models $\to$ Actionable Remediation).
  - Bottom: Grid of 5 failure categories with mathematical symptoms.
- **Slide Content:**
  - **Closed-World Supervised Taxonomy ($K=5$):**
    1. **Retrieval Miss:** Target knowledge missing from database; all cosine distances $>0.70$.
    2. **Ranking Error:** Evidence exists in corpus but was demoted past Top-5 ($d_{\text{diff}} = d_{\text{deep}} - d_{\text{top5}}$ dropoff).
    3. **Context Contamination:** Top-5 context polluted with irrelevant, high-variance distractor chunks.
    4. **Hallucination:** LLM invents facts; lexical overlap with retrieved context collapses ($J_{\text{gen, top5}} \approx 0$).
    5. **Contradiction:** Retrieved sources present conflicting claims; top-5 distance variance $\sigma^2_{\text{top5}}$ spikes by $2.7\times$.
  - **Dataset Balance:** 50 controlled benchmark queries (10 per class) partitioned into 37 training and 13 holdout test samples under Stratified sampling.
- **Speaker Notes (Member 1 - 35s):**
  > "To build a rigorous supervised diagnostic engine, we established a 5-class taxonomy covering the complete RAG failure lifecycle. Retrieval Miss and Ranking Error represent retrieval-stage failures; Context Contamination represents context assembly failure; while Hallucination and Contradiction represent synthesis failures. Notice our 50-query dataset is perfectly balanced with 10 queries per category, allowing unbiased Stratified Cross-Validation across all 5 classes."

---

## SLIDE 4: Vector Retrieval Geometry & Deep-Fetch Analysis
- **Header:** Phase 0 & 1: ChromaDB Indexing & $K=100$ Deep Fetch
- **Presenter:** Member 1 $\to$ Hand-off to Member 2
- **Visual Layout:**
  - 2 Comparative Retrieval Curves (Plotly charts from `pages/3_Retrieval_Analysis.py`):
    - Left: Flat High-Distance Curve (Retrieval Miss).
    - Right: Steep Elbow Curve (Ranking Error).
- **Slide Content:**
  - **Corpus Construction:**
    - 157 multi-disciplinary academic documents indexed in **ChromaDB** using 384-dimensional dense embeddings (`all-MiniLM-L6-v2`) in Cosine Space.
    - HNSW Graph Traversal Complexity: $\mathcal{O}(\log N)$.
  - **Why $K=100$ Deep Fetch is Mandatory:**
    - *Top-5 baseline* reveals only what entered the LLM prompt.
    - *Ranks 6 to 100* reveal the retriever's suppression tail.
    - If gold document is found at rank 18: **Ranking Error** (solvable by reranker).
    - If gold document is absent across all 100 ranks: **Database Gap** (requires corpus expansion).
  - **Production Scalability:** Deep fetch requires transmitting only 100 floating-point numbers ($400\text{ bytes}$), incurring $<4\text{ ms}$ overhead without hydrating raw text payloads.
- **Speaker Notes (Member 1 - 35s):**
  > "Here is the key architectural insight of our retrieval engine. In production RAG, systems only retrieve the Top-5 chunks. But if a query fails, looking only at Top-5 cannot tell you why. By performing a $K=100$ deep fetch, we extract the distance decay curve. If the curve is completely flat above 0.75, it's a confirmed Database Gap. If the curve displays an immediate elbow where distances drop, the knowledge exists but was demoted. Now, Member 2 will explain how we convert these curves into numerical features."

---

## SLIDE 5: Mathematical Feature Engineering & Preprocessing
- **Header:** Phase 2: 36 Extracted Features & Hypersphere Geometry
- **Presenter:** Member 2
- **Visual Layout:**
  - 3 Category Cards: Distance Metrics (18) | Lexical Overlaps (8) | Horizon Indicators (10).
  - Callout Box with mathematical equations for Cosine-to-$L_2$ equivalence and Jaccard.
- **Slide Content:**
  - **Distance Distribution Features (18):**
    - Mean, Variance ($\sigma^2$), Standard Deviation, Spread ($d_{\max} - d_{\min}$), Min, Max computed across Top-5, Deep Fetch (6–100), and All (1–100).
    - Relative shift: $d_{\text{diff}} = \bar{d}_{\text{deep}} - \bar{d}_{\text{top5}}$.
  - **Cosine-to-Euclidean ($L_2$) Hypersphere Equivalence:**
    - Because embeddings are $L_2$-normalized ($\|\mathbf{u}\| = 1$), Euclidean distance is strictly monotonic with Cosine distance:
      $$\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{2(1 - \cos\theta)} = \sqrt{2 \cdot d_{\cos}}$$
  - **Lexical & Syntactic Overlap Metrics (8):**
    - *Jaccard Set Overlap:* $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$ (Vocabulary presence).
    - *ROUGE-L LCS:* Longest Common Subsequence F1-score (Syntactic phrasing preservation).
  - **No-Leakage Preprocessing Pipeline:**
    - `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) fitted strictly on $\mathcal{D}_{\text{train}}$ inside scikit-learn `Pipeline` to prevent validation leakage.
- **Speaker Notes (Member 2 - 35s):**
  > "Member 1 showed the raw retrieval results; my role was to engineer a 36-dimensional continuous feature matrix from those interactions. We compute distribution statistics across both retrieval horizons. Crucially, because transformer embeddings lie on a unit hypersphere, we derive the exact Euclidean distance directly from cosine distance without re-computing dot products. We also extract Jaccard set overlap and ROUGE-L sequence alignment between the generated answer, expected answer, and retrieved chunks. Everything is wrapped in a strict scikit-learn Pipeline so that scaling parameters never leak into test folds."

---

## SLIDE 6: Feature Selection & Collinearity Elimination
- **Header:** Phase 3: Filter ANOVA & Embedded L1 Lasso Sparsity
- **Presenter:** Member 2
- **Visual Layout:**
  - Left: Bar chart of ANOVA F-statistic rankings (`rougel_expected_vs_top1`, `top5_dist_var` at top).
  - Right: L1 Lasso coefficient shrinkage diagram showing 21 features driven to $0.0$.
- **Slide Content:**
  - **Filter Selection (ANOVA F-statistic):**
    - Computes univariate ratio of between-class variance to within-class variance:
      $$F = \frac{\sum_c N_c(\bar{x}_c - \bar{x})^2 / (K - 1)}{\sum_c \sum_i (x_{ci} - \bar{x}_c)^2 / (N - K)}$$
    - Top discriminative predictors: `rougel_expected_vs_top1` ($F=42.1$, $p<10^{-6}$), `top5_dist_var` ($F=28.4$, $p<10^{-5}$).
  - **Embedded Selection (L1 Lasso Regularization):**
    - SAGA solver minimizes multinomial loss with $L_1$ penalty:
      $$\min_{\mathbf{W}} \mathcal{L}(\mathbf{W}) + \lambda \sum_{k=1}^K \sum_{j=1}^p |w_{kj}|$$
    - Diamond polytope corners force redundant collinear features strictly to **0.0**.
  - **Pruning Results:**
    - Condenses feature space from **$36 \to 15$ non-collinear features**.
    - Pruned redundant collinear pairs: `rougel_expected_vs_top5` zeroed ($\beta=0.0$) in favor of `jaccard_expected_vs_top5` ($\beta=0.309$).
    - Improves sample-to-feature ratio from $37 / 36 \approx 1.02 \to 37 / 15 \approx \mathbf{2.47}$.
- **Speaker Notes (Member 2 - 35s):**
  > "Having 36 features for 37 training samples would cause severe overfitting. To solve this, we implemented Unit V feature selection. First, an ANOVA F-test verified that our distance variance and lexical overlap metrics possess statistically significant class separation with p-values well below 0.001. Then, we applied L1 Lasso Regularization using the SAGA solver. The geometric diamond constraints of L1 drove 21 collinear features to absolute zero—for instance, eliminating redundant ROUGE-L variants while keeping Jaccard. This gave us 15 independent predictors, perfectly conditioning the data for Member 3's supervised models."

---

## SLIDE 7: Unsupervised Latent Space & Association Rule Mining
- **Header:** Phase 4: 3D PCA, Clustering & Apriori Diagnostics
- **Presenter:** Member 4
- **Visual Layout:**
  - Left: Interactive 3D PCA snapshot of the 384-dimensional corpus colored by domain.
  - Right: Apriori association rules table with Support, Confidence, and Lift.
- **Slide Content:**
  - **Principal Component Analysis (PCA on Unnormalized Manifold):**
    - Eigendecomposition of covariance matrix $\boldsymbol{\Sigma} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ projects 384-dim embeddings to 3D.
    - **Key Mathematical Decision:** PCA was applied to raw directional unit embeddings *without* Z-score standardization, preserving spherical cosine angles.
    - First 3 components explain **$>55\%$ of total corpus variance**.
  - **Unsupervised Clustering:**
    - *K-Means:* Optimal $K=4\text{--}5$ validated via Elbow Inertia inflection and Silhouette score ($s=0.38$).
    - *DBSCAN ($\epsilon=0.35$, MinPts=3, metric="cosine"):* Discovers density-connected clusters; isolates unindexed queries as noise ($\text{label} = -1$) to flag database coverage gaps.
  - **Apriori Association Rule Mining:**
    - Continuous features discretized into 3 quantile bins: `Low`, `Medium`, `High`.
    - Discovers interpretable diagnostic rules ($S \ge 0.10, C \ge 0.60, L \ge 1.10$):
      $$\{\text{top5\_dist\_spread}=\text{High}, \text{jaccard\_gen\_vs\_top5}=\text{Low}\} \implies \text{Contradiction} \quad (\text{Conf: } 80\%, \text{Lift: } \mathbf{3.2})$$
- **Speaker Notes (Member 4 - 35s):**
  > "Before moving to supervised models, we explored unsupervised discovery under Unit VI. We projected ChromaDB's 384-dimensional embedding manifold into 3D using PCA. Crucially, we did not standardize the vectors prior to PCA, because Z-scoring unit-norm embeddings distorts their spherical geometry. DBSCAN with cosine metric identifies spatial noise points, automatically flagging out-of-distribution queries without labels. Furthermore, our Apriori mining extracted deterministic rules, proving with a Lift of 3.2 that high retrieval dispersion coupled with low lexical generation overlap implies a Contradiction failure."

---

## SLIDE 8: Supervised 6-Model Benchmark Shootout
- **Header:** Phase 5: Multi-Model Evaluation & Leaderboard
- **Presenter:** Member 3
- **Visual Layout:**
  - Top: Benchmark Shootout Leaderboard table ranking all 6 models.
  - Bottom Left: Confusion Matrix Heatmap (5x5).
  - Bottom Right: Feature Importance Bar Chart (Random Forest Gini importances).
- **Slide Content:**
  - **Multi-Model Benchmark Shootout (Identical 5-Fold Stratified CV & Test Split):**
    | Classifier Model | Algorithm Family | 5-Fold CV Accuracy | Test Accuracy | Test Macro F1 |
    | :--- | :--- | :---: | :---: | :---: |
    | **🥇 Random Forest** | Bagging Ensemble (100 Trees) | **89.6% (±9.5%)** | **69.2%** | **0.6743** |
    | **🥈 K-Nearest Neighbors** | Instance-Based Lazy ($k=5$) | 81.1% (±10.2%) | 69.2% | 0.6533 |
    | **🥉 Linear SVM** | Maximum Margin Hyperplane | 83.8% (±8.4%) | 61.5% | 0.5867 |
    | **RBF Kernel SVM** | Non-linear Hilbert Space | 81.1% (±11.4%) | 61.5% | 0.5867 |
    | **Logistic Regression** | Multinomial Softmax GLM | 78.4% (±12.1%) | 61.5% | 0.5600 |
    | **Decision Tree** | Recursive Partitioning | 78.4% (±14.8%) | 53.8% | 0.4933 |
  - **Why Random Forest Outperformed Others:**
    - 100 decorrelated trees with random subspace feature sampling ($m = \sqrt{15} = 3$) eliminate variance on complex multi-modal failure boundaries.
    - Capacity constraints (`max_depth=6`, `min_samples_split=3`) prevented sample memorization.
- **Speaker Notes (Member 3 - 40s):**
  > "Here is our core supervised shootout on Page 6 of the platform. We benchmarked all six classical classifiers required by the syllabus under strict no-leakage conditions: identical 5-fold Stratified CV splits and a held-out 13-query test set. Random Forest took first place on the Academic corpus with an 89.6% Cross-Validation accuracy. The confusion matrix shows that Retrieval Miss and Ranking Error are separated with near 100% precision. The hardest boundary is Hallucination versus Contradiction, which Random Forest resolves through orthogonal splits on distance variance."

---

## SLIDE 9: Cross-Domain Generalization & Live Multi-Model Consensus
- **Header:** Live Diagnostic Inference & Dual-Domain Benchmark
- **Presenter:** Member 3 & Member 4
- **Visual Layout:**
  - Left: Snapshot of `pages/7_Failure_Diagnosis.py` showing Live Multi-Model Consensus Grid and Probability Distribution.
  - Right: Dual-Domain comparison card (Academic Corpus vs. Legal Corpus).
- **Slide Content:**
  - **Live Inference Engine (`pages/7_Failure_Diagnosis.py`):**
    - Takes any query, extracts features, and runs a **Soft-Voting Ensemble Consensus**:
      $$\hat{y}_{\text{consensus}} = \arg\max_{c} \frac{1}{6} \sum_{m=1}^6 P_m(Y = c \mid \mathbf{x})$$
    - Outputs calibrated class probability bars and automated engineering recommendations (e.g., *Deploy Cross-Encoder Reranker* for Ranking Errors; *Lower Temperature $\le 0.2$* for Hallucinations).
  - **Dual-Domain Generalization Proof (Legal & Law Corpus):**
    - To prove the platform is not hardcoded to one dataset, we built a 2nd RAG app: **116 legal documents** (Constitution, IPC/BNS, Contracts, Torts).
    - When retrained on the Legal feature space, **K-Nearest Neighbors (KNN) took #1 with 84.6% Test Accuracy and 0.8200 Macro F1**!
    - Proves the architecture generalizes across completely disparate knowledge domains.
- **Speaker Notes (Member 3 & 4 - 35s):**
  > "Member 3: On Page 7, our live diagnostic console runs multi-model consensus voting, displaying confidence distributions and engineering remediation.  
  Member 4: Crucially, examiners often ask if ML models are corpus-dependent. To prove generalization, we built a second, completely separate RAG app covering Indian Constitutional and Criminal Law with 116 documents. Switching the sidebar dropdown re-targets the entire platform to Legal Law. On this domain, K-Nearest Neighbors achieved 84.6% Test Accuracy, demonstrating that when retrieval features are recalibrated, classical ML generalizes across domains."

---

## SLIDE 10: Conclusion & ML3003 Academic Mapping
- **Header:** Summary of Contributions & Syllabus Alignment
- **Presenter:** All Members (Concluding summary by Member 1)
- **Visual Layout:**
  - Grid mapping all 6 syllabus units directly to project code modules.
  - Key takeaway bullet cards.
- **Slide Content:**
  - **Syllabus Unit Mapping Matrix:**
    - *Unit I (ML Foundations):* 5-Fold Stratified CV, Macro F1, Precision/Recall, Bias-Variance control (`diagnostic_models.py`).
    - *Unit II (Supervised Preprocessing):* Jaccard/ROUGE-L math, Cosine-to-$L_2$ equivalence, StandardScaler pipelines (`feature_extraction.py`).
    - *Unit III (Statistical Learning):* Softmax Logistic Regression, K-Nearest Neighbors ($k=5$) (`diagnostic_models.py`).
    - *Unit IV (Support Vector Machines):* Linear SVM (OVO hyperplanes), Non-linear RBF Kernel SVM (`diagnostic_models.py`).
    - *Unit V (Trees & Ensembles):* Decision Trees (`max_depth=5`), Random Forest (100 trees), Filter ANOVA, Embedded L1 Lasso, 3D PCA (`feature_selection.py`, `latent_space_visualizer.py`).
    - *Unit VI (Unsupervised & Rules):* K-Means Elbow/Silhouette, DBSCAN density noise, Apriori Association Rules (`apriori_analyzer.py`).
  - **Final Takeaway:** 
    - Proven feasibility of **zero-LLM deterministic RAG failure diagnostics**.
    - Fully open-source, tested (72/72 passing tests), and deployed with dual-domain verification.
  - **Thank You — Questions & Viva Defense Open.**
- **Speaker Notes (Member 1 - 30s):**
  > "In conclusion, our platform successfully implements every major concept from our ML3003 syllabus—from preprocessing pipelines and L1 regularization to SVMs, Random Forests, PCA, and Apriori mining. We have replaced the circularity of LLM-as-a-judge with a deterministic, mathematically grounded, and sub-millisecond diagnostic platform. All 72 unit tests pass, and the entire codebase is open-sourced on GitHub. Thank you, and we now welcome your questions."

---

## 📋 Quick Slide-to-Speaker Reference Card

| Slide # | Slide Title | Primary Presenter | Key Screen / Code to Show |
| :---: | :--- | :---: | :--- |
| **1** | Title & Architecture Overview | **Member 1** | App Home Page banner |
| **2** | Motivation: Fallacy of LLM-as-a-Judge | **Member 1** | Problem comparison table |
| **3** | 5-Class Taxonomy & Pipeline | **Member 1** | `pages/2_Data_Ingestion.py` |
| **4** | ChromaDB & $K=100$ Deep Fetch | **Member 1** | `pages/3_Retrieval_Analysis.py` |
| **5** | Mathematical Feature Extraction | **Member 2** | `modules/feature_extraction.py` |
| **6** | Filter ANOVA & Embedded L1 Lasso | **Member 2** | `pages/4_Statistical_Analyzer.py` |
| **7** | 3D PCA, Clustering & Apriori Rules | **Member 4** | `pages/5_Latent_Space_Visualizer.py` & `8_Apriori_Rules.py` |
| **8** | Supervised 6-Model Shootout | **Member 3** | `pages/6_ML_Model_Comparison.py` |
| **9** | Live Diagnosis & Dual-Domain Demo | **Member 3 & 4** | `pages/7_Failure_Diagnosis.py` (Legal Domain Toggle) |
| **10** | Conclusion & Syllabus Alignment | **Member 1** | Final ML3003 Matrix |
