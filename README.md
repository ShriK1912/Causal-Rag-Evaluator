# 🔬 Causal RAG Failure Diagnostician Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, classical statistical Machine Learning platform for diagnosing Retrieval-Augmented Generation (RAG) failures without relying on circular, non-deterministic "LLM-as-a-judge" architectures.

Developed as a capstone project for **ML3003 Machine Learning**, directly mapping classical ML algorithms to failure analysis in vector-retrieval systems.

---

## 📌 Architectural Overview

Traditional RAG evaluations use LLM-as-a-judge, which is cost-prohibitive, non-deterministic, and conceptually circular. This platform reframes RAG evaluation as a **data-generating mechanism**:
1. Queries are executed against an indexed vector database (**ChromaDB** with HNSW indexing).
2. Deep-fetch retrieval is performed up to $K=100$.
3. **36 continuous numerical and lexical features** are extracted (Cosine & Euclidean $L_2$ distance curves, variance, spread, Jaccard token set overlap, ROUGE-L LCS structural coherence).
4. Feature selection via **Filter ANOVA F-tests** and **Embedded L1 Lasso Regularization** condenses the space from 36 to 15 non-collinear predictors.
5. Six classical supervised ML classifiers benchmark and diagnose the failure category:
   - **Multinomial Logistic Regression** (Softmax GLM with L-BFGS)
   - **Linear Support Vector Machine** (Maximum-margin hyperplanes with Platt scaling)
   - **RBF Kernel Support Vector Machine** (Non-linear Gaussian projection)
   - **Decision Tree Classifier** (Constrained recursive partitioning)
   - **Random Forest Ensemble** (100 decorrelated bagged decision trees)
   - **K-Nearest Neighbors (KNN)** (Instance-based lazy learning in Euclidean space)
6. Unsupervised space analysis via **3D Principal Component Analysis (PCA)**, **K-Means clustering**, **DBSCAN outlier detection**, and **Apriori Association Rule Mining**.

---

## 🏷️ The 5-Class Failure Taxonomy

| Failure Mode | Mathematical & Empirical Definition |
| :--- | :--- |
| **Retrieval Miss** | Target document absent from database; retrieval distances are uniformly high outliers ($d_{\cos} > 0.70$). |
| **Ranking Error** | Gold document present in database but demoted past rank 5 ($d_{\text{diff}} = d_{\text{deep}} - d_{\text{top5}}$ dropoff). |
| **Context Contamination** | Top-5 chunks polluted with high-dispersion distractor documents from conflicting domains. |
| **Hallucination** | Generated answer fabricates ungrounded facts; Jaccard overlap with retrieved chunks collapses to near zero. |
| **Contradiction** | Retrieved sources present opposing claims; top-5 distance variance $\sigma^2_{\text{top5}}$ spikes by $2.7\times$. |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/causal-rag-diagnostician.git
cd causal-rag-diagnostician
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Test Suite
```bash
pytest tests/
```
All 72 tests execute under strict no-leakage validation.

---

## 📊 Multi-Domain Generalization (Dual Knowledge Bases)

The platform includes two fully functional, switchable RAG knowledge bases:

1. **🏛️ Academic Multidisciplinary Corpus (157 Documents):**
   - Topics: Machine Learning, Astrophysics, Chemistry, CS History, Genetics, Medicine.
   - **Top Classifier:** Random Forest (89.6% CV Accuracy).
2. **⚖️ Legal & Constitutional Knowledge Base (116 Documents):**
   - Topics: Indian Constitution (Articles 14, 19, 21, 32, 356), IPC / BNS Criminal Law, Contract Act, Torts, Property Law.
   - **Top Classifier:** K-Nearest Neighbors (84.6% Test Accuracy, 0.8200 Macro F1).

Users can toggle between both knowledge bases live from the sidebar dropdown.

---

## 📂 Project Structure

```
ml/
├── app.py                             # Main Streamlit Dashboard entrypoint
├── config/
│   └── config.py                      # Global configuration, constants & corpus paths
├── data/
│   ├── chroma_db/                     # Persistent ChromaDB vector store (Academic)
│   ├── legal_rag/                     # Legal RAG knowledge base & retrained models
│   ├── processed/                     # Extracted 36-feature matrices
│   └── synthetic/                     # 50-query controlled failure benchmarks
├── docs/
│   ├── MASTER_VIVA_DEFENSE_GUIDE.md   # Exhaustive 49KB ML3003 syllabus defense manual
│   ├── DEMO_AND_PRESENTATION_GUIDE.md # 5-minute presentation script & demo guide
│   ├── ML_CONCEPTS_AND_ALGORITHMS.md  # Mathematical theory for all algorithms
│   └── PROJECT_PROGRESS.md            # Implementation roadmap & audit trail
├── models/
│   ├── all_diagnostic_models.pkl      # Pickled 6-model benchmark suite
│   └── best_diagnostic_model.pkl      # Serialized #1 ranked model pipeline
├── modules/
│   ├── apriori_analyzer.py            # Apriori frequent itemset & association rules
│   ├── chroma_connector.py            # ChromaDB HNSW nearest-neighbor retrieval
│   ├── data_ingestion.py              # Schema validation & quality audit
│   ├── diagnostic_models.py           # 6 Supervised Classifiers & Evaluation Suite
│   ├── feature_extraction.py          # 36-feature extraction engine
│   ├── feature_selection.py           # Filter ANOVA & Embedded L1 Lasso
│   ├── latent_space_visualizer.py     # 3D PCA, K-Means sweep & DBSCAN
│   ├── statistical_analyzer.py        # Descriptive statistics & ANOVA tests
│   └── ui_components.py               # Sidebar branding & domain switcher
├── pages/
│   ├── 2_Data_Ingestion.py            # Interactive dataset upload & quality audit
│   ├── 3_Retrieval_Analysis.py        # Top-5 vs Deep-Fetch chunk inspection
│   ├── 4_Statistical_Analyzer.py      # ANOVA F-score rankings & distributions
│   ├── 5_Latent_Space_Visualizer.py   # Interactive 3D PCA & cluster analysis
│   ├── 6_ML_Model_Comparison.py       # 6-Model Shootout Leaderboard & Inspector
│   ├── 7_Failure_Diagnosis.py         # Live query diagnosis & consensus voting
│   └── 8_Apriori_Rules.py             # Discretized association rule mining
├── scripts/
│   └── run_legal_pipeline.py          # Automated pipeline runner for Legal RAG
├── tests/                             # 72 unit & integration tests
├── pytest.ini                         # Test runner configuration
└── requirements.txt                   # Production dependencies
```

---

## 🎓 Academic Alignment (ML3003 Syllabus)

- **Unit I: ML Foundations:** Types of Learning, Bias-Variance Tradeoff, 5-Fold Stratified Cross-Validation, Macro F1, Precision/Recall.
- **Unit II: Supervised Learning & Preprocessing:** Cosine/Euclidean equivalence, Jaccard/ROUGE-L lexical math, StandardScaler no-leakage pipelines.
- **Unit III: Statistical Learning:** Multinomial Logistic Regression (Softmax GLM with L-BFGS), K-Nearest Neighbors ($k=5$, Euclidean distance).
- **Unit IV: Support Vector Machines:** Linear SVM (OVO hyperplanes, Platt probability scaling), Non-linear RBF Kernel SVM ($K(\mathbf{x},\mathbf{z}) = \exp(-\gamma \|\mathbf{x}-\mathbf{z}\|^2)$).
- **Unit V: Trees, Ensembles & Dimensionality Reduction:** Decision Trees (`max_depth=5`, Gini index), Random Forest (100 trees, bootstrap aggregation, $m=\sqrt{p}$), Filter ANOVA F-test, Embedded L1 Lasso (SAGA solver), Principal Component Analysis (3D PCA on unnormalized manifold).
- **Unit VI: Unsupervised Learning & Rules:** K-Means clustering (Elbow curve, Silhouette analysis), DBSCAN density clustering ($\epsilon$, MinPts, cosine distance), Apriori Association Rule Mining (Support, Confidence, Lift).

For complete formula derivations, line-by-line code breakdowns, and Viva questions, see **[`docs/MASTER_VIVA_DEFENSE_GUIDE.md`](docs/MASTER_VIVA_DEFENSE_GUIDE.md)**.

---

## 📜 License
MIT License. Developed for academic and research purposes.
