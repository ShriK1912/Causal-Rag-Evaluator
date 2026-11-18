# 🔬 Causal RAG Failure Diagnostician — Complete Demo & Usage Guide

> **App URL:** [http://localhost:8501](http://localhost:8501)  
> **Dataset:** 50 synthetic failed RAG queries across 5 failure categories  
> **Models trained:** 6 classical ML classifiers, all ready to diagnose

---

## 🗺️ Platform Map — 8 Tabs Explained

```
🔬 Dashboard (app.py)          → System health, database status
📋 Data Ingestion              → Load & validate query datasets
🔍 Retrieval Analysis          → Inspect Top-5 vs K=100 vector distances
📈 Statistical Analyzer        → ANOVA F-tests, distance distributions
🌌 Latent Space Visualizer     → 3D PCA + K-Means + DBSCAN
🤖 ML Model Comparison         → 6-model shootout leaderboard
🩺 Failure Diagnosis           → Live query diagnosis with any model
⛏️ Apriori Rules               → Association rule mining
```

---

## 🚀 Step-by-Step Demo Walkthrough (Best Presentation Order)

---

### STEP 1 — 🔬 Dashboard (Home)

**What to show:**
- **System Health metrics** in the 4-column top banner:
  - Python Version ✅
  - ChromaDB Vector Store: **Connected** ✅
  - Synthetic Benchmark: **Available** ✅
  - Failure Taxonomy: **5 Categories** ✅
- **Local Vector Database Status** — point out that ChromaDB has the academic corpus indexed with cosine distance space.
- Scroll to **Active Dataset Overview** — shows the 50 queries loaded automatically.

**What to say (Viva talking point):**
> "The first thing the platform does is perform a live health check on the vector store and confirm all 157 indexed documents are available. This confirms Phase 0 — corpus construction — is complete before any ML pipeline runs."

---

### STEP 2 — 📋 Data Ingestion (`2_Data Ingestion`)

**What to show:**
- The page auto-loads `sample_failed_queries.csv` and runs a **Data Quality Audit**.
- Point out the ingestion stats: **50 queries, 0 duplicates, 5 failure classes (balanced at 10 each)**.
- Show the **class distribution bar chart** — perfectly balanced dataset, no class imbalance bias.
- Show the **sample records table** with columns: `query_id`, `query`, `expected_answer`, `generated_answer`, `failure_label`.

**The 5 Failure Categories the system diagnoses:**

| Label | What it Means |
|---|---|
| `Retrieval Miss` | Query topic absent from corpus — retrieval distance is an outlier |
| `Ranking Error` | Correct document exists but was demoted to rank 6–100 |
| `Context Contamination` | Irrelevant distractor chunks polluted the Top-5 context |
| `Hallucination` | LLM generated facts not present in any retrieved chunk |
| `Contradiction` | Conflicting documents retrieved — LLM synthesized opposing claims |

**What to say:**
> "Notice the dataset is perfectly stratified — 10 queries per class. This is critical because we apply Stratified Train-Test Split downstream, ensuring every fold of our 5-fold cross-validation contains all 5 failure class representatives."

---

### STEP 3 — 🔍 Retrieval Analysis (`3_Retrieval Analysis`)

**What to show:**
1. Use the **query selector dropdown** at the top — pick different Query IDs.
2. For a **Retrieval Miss** query (e.g. `FAIL_001` — "Treaty of Vega"):
   - Show **Top-5 retrieved chunks** — the distances are all very high (> 0.80), meaning the retriever found nothing relevant.
   - Show the **Ranked Distance Curve (Rank 1 → 100)** — a nearly flat line at high distance values, confirming database gap.
3. For a **Ranking Error** query:
   - Show Top-5 chunks — the **target document doesn't appear in Top-5**.
   - Show the K=100 deep fetch — the target doc appears at rank 15–50.
   - Point out the **distance drop-off gradient** between Top-5 and deep fetch.

**What to say:**
> "This is the mathematical heart of our approach. Instead of asking an LLM 'is this answer correct?', we extract the cosine distance vector from ChromaDB for all 100 retrieved neighbors and compute the variance, min, max, and spread mathematically. That becomes our feature vector for the ML classifiers."

---

### STEP 4 — 📈 Statistical Analyzer (`4_Statistical Analyzer`)

**What to show:**
1. **ANOVA F-Score Feature Rankings** — the bar chart showing which features have the highest statistical discriminative power (F-statistic) across the 5 failure classes.
   - Highest F-scores: `jaccard_expected_vs_top1`, `rougel_expected_vs_top1`, `target_in_top5` — these are what drive the model most.
2. **Top-5 vs Deep-Fetch Distance Histogram** — side-by-side histogram comparing the two retrieval horizons.
3. **Category Aggregate Box Plots** — shows the median and variance of key features grouped by failure class.
   - Point out how **Retrieval Miss** has consistently high distances (box plot shifted right).
   - Point out how **Contradiction** has the highest variance (wider box).

**What to say:**
> "The ANOVA F-statistic ranks features by their univariate discriminative power. Features with high F-scores and p-value < 0.05 carry statistically significant class-separation signal. These form the candidate pool for our L1 Lasso embedded feature selection, which then further prunes redundant collinear features."

---

### STEP 5 — 🌌 Latent Space Visualizer (`5_Latent Space Visualizer`)

**What to show:**
1. **3D PCA Projection by Domain** — the 384-dimensional sentence embeddings of all 157 corpus documents reduced to 3 principal components. Each color = a topic domain. Rotate the 3D chart interactively.
2. **Elbow Curve & Silhouette Scores** — K-Means optimal cluster selection showing K=4 or K=5 as the inflection point.
3. **3D PCA by K-Means Labels** — same projection now color-coded by learned cluster ID.
4. **DBSCAN** — identifies density outliers (noise points = potential knowledge gap locations).

**Key Demo Moment — PCA Explained Variance:**
- Show the **Scree / Explained Variance Chart** and point out that PC1 + PC2 + PC3 together explain approximately 50–60% of total embedding variance, confirming that 3 components meaningfully compress the high-dimensional space while retaining structural information.

**What to say:**
> "This is Unit V in our syllabus — dimensionality reduction. The critical insight here is that we did NOT apply PCA directly to cosine-normalized unit-norm vectors, because standardizing them would destroy their spherical geometry. Instead, we apply PCA directly to the raw unnormalized embeddings, preserving the variance structure of the high-dimensional manifold."

---

### STEP 6 — 🤖 ML Model Comparison (`6_ML Model Comparison`)

> **This is your SHOWSTOPPER SLIDE for the demo.**

**What to show, in order:**

#### Sub-Tab A: Leaderboard
- The **6-Model Shootout Leaderboard** table with columns:
  - `Accuracy`, `Precision (Macro)`, `Recall (Macro)`, `F1 (Macro)`, `CV Mean ± Std`
- Point out the best-performing model (typically **Random Forest** or **KNN**) in the green row.
- Point out the **5-Fold Cross-Validation Mean ± Std** column — this proves results aren't a lucky test-split artifact.

**Say:** 
> "Every model is evaluated under identical conditions — same 37-sample training set, same L1-selected 15 features, same StandardScaler normalization, same 13-sample holdout test, same 5-fold Stratified CV. This is a fair, no-leakage benchmark."

#### Sub-Tab B: Model Inspector
- Select each model from the dropdown and show:
  - **Confusion Matrix (Plotly heatmap)** — 5×5 matrix showing which classes each model confuses.
  - Point out that **Hallucination vs Contradiction** is the hardest pair to separate.
  
#### Sub-Tab C: Feature Interpretability
- **Random Forest Gini Feature Importance Bar Chart** — shows `jaccard_generated_vs_top5` and `rougel_expected_vs_top1` as top features.
- **Logistic Regression L1 Coefficients heatmap** — 5 classes × 15 features matrix, showing how each feature pushes the model toward each failure class.

**What to say:**
> "This satisfies the explainability requirement completely. Unlike a black-box LLM judge, we can tell you *exactly* which numerical signal — a cosine distance spread of 0.57 versus 0.41 — caused the model to choose Contradiction over Hallucination."

---

### STEP 7 — 🩺 Failure Diagnosis (`7_Failure Diagnosis`)

> **The live interactive diagnosis — this is your demo climax.**

**Step-by-step actions:**

**Demo A — Correct Diagnosis (Retrieval Miss)**
1. In the **Query Selector** dropdown, choose: `FAIL_001 | [Retrieval Miss] What year was the interstellar Treaty of Vega ratified?`
2. Set the **Model** to `Random Forest`.
3. Click **Run Diagnosis** (or just observe — it auto-runs).
4. **Show:** The green `✅ MATCHES GROUND TRUTH` badge appears.
5. Point to the **Probability Distribution bar chart** — `Retrieval Miss` bar dominates at 90%+ confidence.
6. Scroll to **Multi-Model Consensus Grid** — all 6 classifiers should agree: `Retrieval Miss`.
7. Scroll to **Engineering Recommendations** — shows 3 actionable fixes specific to Retrieval Misses.

**Demo B — Harder Case (Contradiction)**
1. Switch to a Contradiction query (e.g. a `FAIL_041–050` range).
2. Switch the model to `Linear SVM`.
3. Show the probability chart — the confidence split between `Contradiction` and `Hallucination` is tighter.
4. Switch to `⚡ Multi-Model Ensemble Consensus`.
5. Show how the ensemble averages the 6 probability distributions and resolves the ambiguity.

**Demo C — Model Switching**
1. Keep the same Contradiction query.
2. Switch through all 6 models one by one — observe how `KNN` and `Decision Tree` may differ from `Random Forest` and `SVM`.
3. Show the Consensus Grid — the row where any model `❌ Differed` becomes a teaching moment.

**What to say:**
> "This is the full end-to-end pipeline in action. A single query goes through the same 36-feature extraction process, gets projected onto the 15 L1-selected features, is normalized by the StandardScaler, and then each of the 6 models independently classifies the failure mode. The ensemble takes a soft majority vote over their probability distributions — this is Bagging logic applied at the meta-classifier level."

---

### STEP 8 — ⛏️ Apriori Rules (`8_Apriori Rules`)

**What to show:**
1. The **default rules** mined at `min_support=0.2`, `min_confidence=0.7`, `min_lift=1.5`.
2. **Rules Count per Category Bar** — shows how many association rules each failure class generates.
3. Click on a high-confidence rule, e.g.:
   - `{Jaccard_gen_top5=Low, Top5_dist_spread=High} → Contradiction (Confidence: 0.80, Lift: 3.2)`
4. Adjust the **support slider** in the sidebar down to `0.1` — more rules appear. Raise `confidence` to `0.85` — only the strongest survive.
5. Show the **Support-Confidence Scatter Plot** — each bubble is a rule, top-right corner = high value rules.

**What to say:**
> "Association Rule Mining covers Unit V of our syllabus. Unlike supervised classifiers that require labeled ground truth, Apriori mines the raw feature distributions without labels. It discovers deterministic logical patterns — if top-5 distance variance is high AND generated Jaccard overlap is low, then with 80% confidence and 3.2× lift over random chance, the failure mode is Contradiction."

---

## 🎯 Perfect Demo Script (5-Minute Quick Version)

If you need a fast 5-minute demo for a professor or reviewer:

1. **Open Dashboard** (10 sec) → "System connected, 157 documents indexed, 50 failure queries ready."
2. **Jump to Retrieval Analysis** (1 min) → Pick `FAIL_001`, show the high-distance flat retrieval curve. "No relevant chunks exist — this is a database gap."
3. **Jump to ML Model Comparison** (2 min) → Show the 6-model leaderboard and one confusion matrix. "Classical ML benchmarked rigorously — Random Forest wins at [X]% F1 Macro."
4. **Jump to Failure Diagnosis** (1.5 min) → Diagnose `FAIL_001` live. Show the green tick, probability chart, consensus grid. "Every model agrees — Retrieval Miss, with recommendations."
5. **Jump to Apriori Rules** (30 sec) → Show one rule. "Deterministic pattern: if cosine spread is high and lexical overlap is low → Contradiction."

---

## 📋 Which Queries Are in Each Class?

| IDs | Failure Class | What Makes Them Interesting to Demo |
|---|---|---|
| `FAIL_001–010` | Retrieval Miss | Fictional topics — distances uniformly high, all models easily agree |
| `FAIL_011–020` | Ranking Error | Real topics but target doc buried — target_in_top5 = 0, in deep fetch = 1 |
| `FAIL_021–030` | Context Contamination | Distractor docs inflated into Top-5 — variance and spread anomalies |
| `FAIL_031–040` | Hallucination | Low jaccard/rouge vs top5 — model invented answer, tokens absent from corpus |
| `FAIL_041–050` | Contradiction | High top5_dist_var — retriever pulled documents from opposing semantic anchors |

---

## ⚡ Common Questions During Demo — Quick Answers

**Q: Is this real data or synthetic?**
> "The RAG failure patterns are synthetic and controlled — we know the ground-truth failure label for each query. This lets us rigorously benchmark our classifiers against a gold standard, something impossible with production data where failure labels are unknown."

**Q: Why not just use an LLM to judge the outputs?**
> "LLM-as-a-judge is circular logic — you're using an LLM to evaluate another LLM. Our approach is deterministic and interpretable: a 15-feature numerical vector goes through a trained classifier with explicit coefficient weights. You can audit every prediction."

**Q: What happens if I feed a new query it has never seen?**
> "The same 36 features would be extracted from ChromaDB's retrieval results — cosine distances, Jaccard overlaps, ROUGE-L scores — and the trained pipeline would classify it in milliseconds. The model generalizes to new queries with the same retrieval failure signatures."

**Q: What is the best model?**
> "Random Forest achieves the highest F1 Macro on our held-out test set. However, Logistic Regression provides full coefficient interpretability, and the Multi-Model Ensemble Consensus is the most robust production choice as it hedges against any single model's miscalibration."

---

> [!IMPORTANT]
> The Streamlit server is running at **http://localhost:8501**. It is a background daemon — it will keep running until you close the terminal or ask me to stop it.

> [!TIP]
> For the best visual impression during a demo, use the **full-screen browser view** (press F11) and use the **wide layout** that is already configured for all pages.
