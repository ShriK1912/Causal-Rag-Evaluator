# Machine Learning Concepts and Algorithms Notebook

This document provides mathematical intuition, formal definitions, step-by-step algorithms, formulas, and interpretations for every statistical, machine learning, and data mining technique implemented in the **Causal RAG Failure Diagnostician Platform**.

---

## 1. Data Preprocessing & Validation Theory

### Concept 1.1: Data Quality Profiling & Schema Normalization
- **Definition**: The systematic auditing of structured data prior to downstream ingestion into statistical and machine learning pipelines.
- **Mathematical / Logical Foundation**:
  Given an input dataset $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$ with schema attributes $\mathcal{A} = \{a_1, a_2, \dots, a_M\}$:
  1. *Schema Mapping*: $\phi: \mathcal{A}_{\text{raw}} \to \mathcal{A}_{\text{canonical}}$ maps lexical variations (e.g. `prompt` $\to$ `query`).
  2. *Null Representation*: Ensures string null representations ("nan", "None", "") satisfy:
     $$\mathbf{x}_{i, j} \in \{\text{valid text}\} \cup \{\text{NaN}\}$$
  3. *Validity Condition*: A record $\mathbf{x}_i$ is valid if and only if:
     $$\text{len}(\text{strip}(\mathbf{x}_{i, \text{query}})) > 0 \quad \text{and} \quad \mathbf{x}_{i, \text{label}} \in \mathcal{C} \cup \{\text{NaN}\}$$
     where $\mathcal{C}$ is the canonical set of 5 failure categories.

### Concept 1.2: Duplicate Detection & Data Leakage Prevention
- **Why it matters in Machine Learning**:
  In supervised machine learning, data leakage occurs when information from outside the training dataset is used to create the model.
  If identical or near-identical queries $q_a = q_b$ appear multiple times in $\mathcal{D}$:
  $$\mathbb{P}(q_b \in \mathcal{D}_{\text{test}} \mid q_a \in \mathcal{D}_{\text{train}}) > 0$$
  This causes the model to memorize specific queries rather than learning generalizable retrieval distance patterns, falsely inflating validation metrics (accuracy, F1-score).
- **De-duplication Procedure**:
  We apply cryptographic hashing / hash-table lookups on normalized query strings:
  $$h(q_i) = \text{hash}(\text{lowercase}(\text{strip}(q_i)))$$
  Keeping the first occurrence and recording duplicate frequency:
  $$\Delta_{\text{dup}} = N_{\text{total}} - |\text{unique}(\{h(q_i)\})|$$

---

## 2. Distance Metrics in Vector Space (Preview for Task 1.2 / 1.3)

### Concept 2.1: Cosine Distance
- **Definition**: A metric measuring the angular difference between two non-zero vectors in an inner product space, invariant to vector magnitude.
- **Formula**:
  Given query embedding $\mathbf{u} \in \mathbb{R}^d$ and document embedding $\mathbf{v} \in \mathbb{R}^d$:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$
  $$\text{Cosine Distance}(\mathbf{u}, \mathbf{v}) = 1 - \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$$
- **Range**:
  For normalized vectors, cosine similarity $\in [-1, 1]$, and cosine distance $\in [0, 2]$ (typically $[0, 1]$ for non-negative embeddings).
- **Interpretation in RAG**:
  - Distance near $0.0$: Document is semantically almost identical to query.
  - Distance $> 0.7$: Document is semantically distant; high likelihood of a retrieval miss.

### Concept 2.2: Euclidean (L2) Distance
- **Formula**:
  $$d_{\text{L2}}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
- **Relationship to Cosine Distance**:
  For $L_2$-normalized vectors where $\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$:
  $$d_{\text{L2}}^2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 2 - 2 \cos(\mathbf{u}, \mathbf{v}) = 2 \cdot d_{\text{cosine}}(\mathbf{u}, \mathbf{v})$$
  $$d_{\text{L2}}(\mathbf{u}, \mathbf{v}) = \sqrt{2 \cdot d_{\text{cosine}}(\mathbf{u}, \mathbf{v})}$$

---

## 3. Text Similarity Metrics (Preview for Task 1.4)

### Concept 3.1: Jaccard Similarity (Lexical Token Overlap)
- **Formula**:
  Let $A$ be the token set of string 1, and $B$ be the token set of string 2:
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- **Characteristics**: Fast, word-order agnostic, sensitive to exact token matching. Useful for catching lexical keyword distractors.

### Concept 3.2: ROUGE-L (Longest Common Subsequence)
- **Formula**:
  Let $X$ of length $m$ be the reference text, and $Y$ of length $n$ be the generated text. Let $\text{LCS}(X, Y)$ be the length of the longest common subsequence:
  $$R_{\text{LCS}} = \frac{\text{LCS}(X, Y)}{m}, \quad P_{\text{LCS}} = \frac{\text{LCS}(X, Y)}{n}$$
  $$F_{\text{LCS}} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$
- **Characteristics**: Preserves sequence ordering without requiring consecutive n-gram matches. Directly captures whether the generator incorporated phrases from retrieved chunks.

---

## 4. Approximate Nearest Neighbors & Graph Indexing

### Concept 4.1: Hierarchical Navigable Small World (HNSW) Graphs
- **Purpose**: Fast, scalable approximate nearest-neighbor search in high-dimensional vector spaces without exhaustive linear scanning.
- **Mathematical / Algorithmic Mechanism**:
  1. *Multi-Layer Hierarchy*: A set of nested graphs $\mathcal{G}_0, \mathcal{G}_1, \dots, \mathcal{G}_{L}$ where layer $0$ contains all vectors, and higher layers contain exponentially fewer vectors (governed by decay parameter $m_L = 1 / \ln(M)$).
  2. *Skip-List Analogy*: Top layers serve as highway express lanes for long-range navigation across vector clusters; bottom layers perform local clustering refinement.
  3. *Greedy Search*:
     At layer $l$, search iterates through neighbors of current node $v_c$, selecting neighbor $v_{\text{next}}$ that minimizes distance to query vector $\mathbf{q}$:
     $$v_{\text{next}} = \arg\min_{v \in \mathcal{N}(v_c)} d(\mathbf{q}, v)$$
     When local minimum is reached, search drops down to layer $l-1$, using the local minimum as entry point.
  4. *Query Complexity*: Expected search time is $O(\log N)$ with high recall ($>98\%$).

### Concept 4.2: Distance Monotonicity in Top-K Heaps
- **Mathematical Property**:
  Let $\mathcal{R}(q) = \langle (d_{(1)}, \delta_{(1)}), (d_{(2)}, \delta_{(2)}), \dots, (d_{(K)}, \delta_{(K)}) \rangle$ be the ranked retrieval list for query $q$, where $\delta_{(i)}$ is the distance between $q$ and retrieved document $d_{(i)}$.
  The list satisfies strict distance monotonicity:
  $$\delta_{(1)} \le \delta_{(2)} \le \delta_{(3)} \le \dots \le \delta_{(K)}$$
  where rank $r = 1$ is the nearest neighbor, and rank $r = K$ is the furthest neighbor in the retrieved set.

---

## 5. Multinomial Logistic Regression (User's Assigned ML Model)

### Concept 5.1: Mathematical Formulation
- **Definition**: A generalized linear model for multi-class classification that models the posterior class probabilities $\mathbb{P}(Y = k \mid \mathbf{x})$ using the Softmax (normalized exponential) function.
- **Hypothesis Function (Softmax)**:
  Given input feature vector $\mathbf{x} \in \mathbb{R}^p$ and $K$ classes (where $K = 5$ in our failure taxonomy):
  $$\mathbb{P}(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
  where $\mathbf{w}_k \in \mathbb{R}^p$ is the weight vector and $b_k$ is the bias for class $k$.

- **Log-Odds (Logit Transformation)**:
  The relative log-odds between any class $k$ and reference class $K$ is strictly linear in the features:
  $$\ln\left(\frac{\mathbb{P}(Y = k \mid \mathbf{x})}{\mathbb{P}(Y = K \mid \mathbf{x})}\right) = (\mathbf{w}_k - \mathbf{w}_K)^T \mathbf{x} + (b_k - b_K)$$
  This linear decision boundary property makes Logistic Regression highly interpretable: the sign and magnitude of weight $w_{k, j}$ directly reveal whether feature $j$ increases or decreases the log-odds of failure mode $k$.

### Concept 5.2: Optimization Objective & Loss Function
- **Multinomial Cross-Entropy Loss (Log-Loss)**:
  For a training set of $N$ observations with one-hot encoded targets $y_{i, k} \in \{0, 1\}$:
  $$\mathcal{L}_{\text{CE}}(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i, k} \ln \mathbb{P}(Y = k \mid \mathbf{x}_i)$$

- **Regularization ($L_2$ Ridge Penalty)**:
  To prevent overfitting in the presence of correlated distance features, we add an $L_2$ weight penalty:
  $$\mathcal{J}(\mathbf{W}) = \mathcal{L}_{\text{CE}}(\mathbf{W}) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$
  where $C > 0$ is the inverse regularization strength (smaller $C$ means stronger regularization).

### Concept 5.3: Why Feature Scaling is Mandatory for Logistic Regression
1. **Penalization Invariance**: The $L_2$ regularization penalty $\sum w_j^2$ treats all weights equally. If feature $x_1$ has a scale of $[0, 1]$ (e.g. Jaccard similarity) and feature $x_2$ has a scale of $[0, 100]$ (e.g. rank), $w_2$ will naturally be much smaller than $w_1$ to achieve the same effect. Consequently, $L_2$ penalization unfairly penalizes $w_1$ while barely affecting $w_2$.
2. **Gradient Optimization Convergence**: Solvers such as **L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)** compute curvature approximations. Unscaled features create elongated, elliptical loss contours, causing gradient trajectories to oscillate and converge slowly. Standardizing features to zero mean and unit variance produces circular contours, enabling rapid, stable convergence.

### Concept 5.4: Multiclass Evaluation Metrics
- **Confusion Matrix**: $K \times K$ matrix where entry $C_{i, j}$ represents the count of instances truly in class $i$ predicted as class $j$.
- **Per-Class Precision, Recall, F1**:
  $$\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}, \quad \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \quad \text{F1}_k = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
- **Macro-Averaged F1 vs Weighted F1**:
  $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k, \quad \text{Weighted F1} = \sum_{k=1}^K \frac{N_k}{N} \text{F1}_k$$
  In our balanced academic dataset ($N_k = 10$), Macro and Weighted F1 are identical, evaluating performance equally across all five failure categories.

---

## 6. Statistical Depth Analysis & Curvature Modeling (Phase 2)

### Concept 6.1: Distance Distribution Statistics
Given a sorted sequence of retrieval distances $\mathcal{D} = \langle d_{(1)}, d_{(2)}, \dots, d_{(K)} \rangle$:
1. **Arithmetic Mean ($\mu$)**:
   $$\mu = \frac{1}{K} \sum_{i=1}^K d_{(i)}$$
   Reflects the central tendency of vector proximity.

2. **Median ($Q_2$)**:
   $$\text{Median} = \begin{cases} d_{((K+1)/2)} & \text{if } K \text{ is odd} \\ \frac{1}{2}\left(d_{(K/2)} + d_{(K/2 + 1)}\right) & \text{if } K \text{ is even} \end{cases}$$
   Non-parametric central tendency resistant to extreme distance outliers.

3. **Variance ($\sigma^2$) and Standard Deviation ($\sigma$)**:
   $$\sigma^2 = \frac{1}{K} \sum_{i=1}^K (d_{(i)} - \mu)^2, \quad \sigma = \sqrt{\sigma^2}$$
   Quantifies the dispersion of retrieved candidates. In Top-5 retrieval, high $\sigma^2$ indicates cluster divergence (**Context Contamination**).

4. **Interquartile Range (IQR)**:
   $$\text{IQR} = Q_3 - Q_1$$
   Measures the spread of the middle 50% of retrieval distances, robust to extreme corpus outliers.

5. **Fisher-Pearson Skewness ($g_1$)**:
   $$g_1 = \frac{\frac{1}{K} \sum_{i=1}^K (d_{(i)} - \mu)^3}{\sigma^3}$$
   Measures distribution asymmetry. Positive skew indicates a tight cluster of close documents with a long tail of distant documents.

---

### Concept 6.2: Depth Contrast & Drop-Off Metrics
1. **Depth Distance Shift ($\Delta_\mu$)**:
   $$\Delta_\mu = \mu_{\text{deep}} - \mu_{\text{top5}} = \frac{1}{95} \sum_{i=6}^{100} d_{(i)} - \frac{1}{5} \sum_{i=1}^5 d_{(i)}$$
   - High $\Delta_\mu$ ($> 0.10$): Clear semantic boundary separating relevant top results from the background corpus.
   - Low $\Delta_\mu$ ($< 0.04$): Flat retrieval horizon; indicates that Top-5 candidates are no closer than random background documents (**Retrieval Miss / Database Gap**).

2. **Distance Drop-Off Ratio ($R_{\text{drop}}$)**:
   $$R_{\text{drop}} = \frac{d_{(5)} - d_{(1)}}{\max(\epsilon, d_{(100)} - d_{(1)})}$$
   Measures the fraction of the total distance dynamic range consumed within the first 5 ranks.

---

### Concept 6.3: Top-5 Pairwise Lexical Coherence
- **Definition**: The arithmetic mean of pairwise Jaccard similarities between all distinct pairs of Top-5 retrieved chunks:
  $$\text{Coherence}(\text{Top5}) = \frac{1}{\binom{5}{2}} \sum_{1 \le i < j \le 5} J(\text{chunk}_i, \text{chunk}_j)$$
  where $\binom{5}{2} = 10$.
- **Interpretation**:
  - High ($> 0.25$): Chunks share common terminology; consensus on topic.
  - Low ($< 0.10$): Chunks discuss disparate topics; indicative of polysemous distractors polluting the context (**Context Contamination**).

---

### Concept 6.4: Shannon Entropy of Retrieval Distance
- **Formula**:
  By partitioning continuous distances into $B$ discrete histogram bins with normalized frequencies $p_1, p_2, \dots, p_B$:
  $$H = -\sum_{i=1}^B p_i \log_2(p_i) \quad (\text{bits})$$
- **Interpretation**: Measures the information uncertainty of retrieval scores. Maximum entropy corresponds to a completely uniform distance spread across all retrieved ranks.

---

### Concept 6.5: Elbow Curvature Detection
- **Method**: Computes perpendicular distance from each ranked point $(r, d_{(r)})$ to the secant line connecting the first point $(1, d_{(1)})$ and last point $(K, d_{(K)})$:
  $$r_{\text{elbow}} = \arg\max_{r \in [1, K]} \frac{|(d_{(K)} - d_{(1)}) r - (K - 1) d_{(r)} + K d_{(1)} - d_{(K)}|}{\sqrt{(d_{(K)} - d_{(1)})^2 + (K - 1)^2}}$$
- **Significance**: Identifies the mathematical rank where retrieval proximity transitions from specific relevance to generic corpus plateauing.

---

### Concept 6.6: Mathematical Derivation & Boundary Preconditions of Cosine-to-Euclidean Distance Conversion
- **Mathematical Identity**:
  Let $u, v \in \mathbb{R}^d$ denote two embedding vectors. The Euclidean distance squared is:
  $$\|u - v\|_2^2 = (u - v) \cdot (u - v) = \|u\|_2^2 + \|v\|_2^2 - 2(u \cdot v)$$
  Cosine similarity is defined as:
  $$\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} \implies u \cdot v = \|u\|_2 \|v\|_2 \cos(u, v)$$
  Substituting this into the Euclidean expansion:
  $$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2 \|u\|_2 \|v\|_2 \cos(u, v)$$
  
- **Unit-Norm Special Case**:
  **If and only if** the vectors are $L_2$-normalized to unit hyperspherical length ($\|u\|_2 = \|v\|_2 = 1$):
  $$\|u - v\|_2^2 = 1^2 + 1^2 - 2(1)(1)\cos(u, v) = 2 - 2\cos(u, v) = 2(1 - \cos(u, v))$$
  Since cosine distance is defined as $d_{\text{cosine}}(u, v) = 1 - \cos(u, v)$:
  $$\|u - v\|_2^2 = 2 \cdot d_{\text{cosine}}(u, v) \implies \|u - v\|_2 = \sqrt{2 \cdot d_{\text{cosine}}(u, v)}$$

- **Boundary Conditions & Validity Range**:
  1. $d_{\text{cosine}} = 0.0$ (identical vectors, $\theta = 0^\circ$):
     $$\|u - v\|_2 = \sqrt{2(0)} = 0.0$$
  2. $d_{\text{cosine}} = 0.5$ (orthogonal components with $\cos\theta = 0.5, \theta = 60^\circ$):
     $$\|u - v\|_2 = \sqrt{2(0.5)} = 1.0$$
  3. $d_{\text{cosine}} = 1.0$ (orthogonal vectors, $\theta = 90^\circ$):
     $$\|u - v\|_2 = \sqrt{2(1.0)} = \sqrt{2} \approx 1.4142$$
  4. $d_{\text{cosine}} = 2.0$ (diametrically opposite vectors, $\theta = 180^\circ$):
     $$\|u - v\|_2 = \sqrt{2(2.0)} = 2.0$$

- **Critical Viva Disclaimer**:
  This identity is **strictly invalid** for unnormalized vectors. If $\|u\|_2 \neq 1$ or $\|v\|_2 \neq 1$, the Euclidean distance depends explicitly on vector magnitudes. In our platform, ChromaDB's default ONNX embedding function (`all-MiniLM-L6-v2`) outputs strictly $L_2$-normalized vectors, ensuring mathematical consistency.

---

## 7. Latent Space Dimensionality Reduction & Unsupervised Clustering (Phase 3)

### Concept 7.1: Principal Component Analysis (PCA)
- **Objective**: Finds an orthogonal linear transformation that projects high-dimensional data $X \in \mathbb{R}^{n \times d}$ into a lower-dimensional subspace $\mathbb{R}^{n \times p}$ ($p=3$) while maximizing the variance of the projected coordinates.
- **Mathematical Derivation**:
  1. Mean-center the data matrix: $\tilde{X} = X - \mathbf{1} \mu^T$, where $\mu = \frac{1}{n} \sum_{i=1}^n x_i$.
  2. Compute the sample covariance matrix:
     $$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$$
  3. Perform spectral decomposition (or Singular Value Decomposition $\tilde{X} = U S V^T$):
     $$\Sigma v_j = \lambda_j v_j, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$$
     where $v_j$ are orthogonal eigenvectors (principal axes) and $\lambda_j$ are eigenvalues proportional to variance.
  4. Form projection matrix $W_p = [v_1, v_2, \dots, v_p] \in \mathbb{R}^{d \times p}$.
  5. Projected 3D coordinates: $Z = \tilde{X} W_3 \in \mathbb{R}^{n \times 3}$.
- **Explained Variance Ratio**:
  $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^d \lambda_j}, \quad \text{Cumulative}(p) = \sum_{k=1}^p \text{EVR}_k$$
- **Centered vs Standardized PCA for Normalized Embeddings**:
  When embeddings are already $L_2$-normalized ($\|x_i\|_2 = 1.0$), each vector resides on the surface of a unit hypersphere $S^{d-1}$.
  - **Centered PCA (Recommended)**: Preserves the angular/spherical metric. In our experiments, centered PCA retains **53.44%** variance in 3D.
  - **Standardized PCA**: Divides by feature standard deviation $\sigma_j$, which warps the sphere into an ellipsoid and amplifies low-variance noise dimensions, retaining only **49.52%** variance (-3.92% loss).
- **Time Complexity**: $O(n d^2 + d^3)$ via covariance eigen-decomposition (or $O(n d p)$ via randomized SVD).
- **Space Complexity**: $O(n d + d^2)$.

---

### Concept 7.2: K-Means Clustering & Lloyd's Objective Function
- **Objective**: Partitions $n$ document embeddings into $K$ disjoint clusters $S = \{S_1, S_2, \dots, S_K\}$ to minimize Within-Cluster Sum of Squares (Inertia):
  $$J(S) = \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|_2^2$$
  where $\mu_k = \frac{1}{|S_k|} \sum_{x \in S_k} x$ is the centroid of cluster $S_k$.
- **Lloyd's Iterative Optimization**:
  1. **Assignment Step**: Assign each document $x_i$ to the nearest centroid:
     $$S_k^{(t)} = \left\{ x_i : \|x_i - \mu_k^{(t)}\|^2 \le \|x_i - \mu_j^{(t)}\|^2, \forall j \in [1, K] \right\}$$
  2. **Update Step**: Recompute centroids as the mean of assigned vectors:
     $$\mu_k^{(t+1)} = \frac{1}{|S_k^{(t)}|} \sum_{x \in S_k^{(t)}} x$$
  3. Converges when assignments stabilize or iterations exceed threshold.
- **Silhouette Coefficient**:
  For each document $i$:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  where:
  - $a(i) = \frac{1}{|S_k| - 1} \sum_{j \in S_k, j \neq i} \|x_i - x_j\|_2$ (mean intra-cluster distance).
  - $b(i) = \min_{l \neq k} \frac{1}{|S_l|} \sum_{j \in S_l} \|x_i - x_j\|_2$ (mean nearest-cluster distance).
  - Interpretation: $s(i) \approx 1$ indicates excellent cluster fit; $s(i) \approx 0$ indicates boundary ambiguity; $s(i) < 0$ indicates misclustering.
- **Time Complexity**: $O(I \cdot K \cdot n \cdot d)$ where $I$ is iteration count.
- **Space Complexity**: $O(n \cdot d + K \cdot d)$.

---

### Concept 7.3: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Objective**: Discovers non-linear, density-connected clusters and identifies isolated noise points (label = -1).
- **Definitions**:
  Given radius $\epsilon > 0$ and minimum neighbor count $\text{MinPts}$:
  - $\epsilon$-Neighborhood: $N_\epsilon(p) = \{q \in D : \text{dist}(p, q) \le \epsilon\}$.
  - **Core Point**: Point $p$ where $|N_\epsilon(p)| \ge \text{MinPts}$.
  - **Directly Density-Reachable**: $q$ is directly reachable from $p$ if $p$ is a core point and $q \in N_\epsilon(p)$.
  - **Density-Reachable**: A chain of core points connects $p$ to $q$.
  - **Border Point**: A point that is density-reachable from a core point but has $< \text{MinPts}$ neighbors itself.
  - **Noise Point**: A point that is not density-reachable from any core point ($\text{label} = -1$).
- **Application to RAG Gap Diagnosis**:
  In a vector database, documents labeled as DBSCAN noise represent isolated semantic concepts lacking dense topical support in the corpus. If user queries target concepts in these sparse regions, the retriever has no close candidates, resulting in **Retrieval Misses**.
- **Academic Caveat**: Noise points are geometric outliers; they indicate vector sparsity, not guaranteed functional failure unless queries demand that specific knowledge.
- **Time Complexity**: $O(n^2 \cdot d)$ (or $O(n \log n \cdot d)$ with spatial trees).
- **Space Complexity**: $O(n^2)$ distance matrix.

---

## 8. Apriori Association Rule Mining & Pattern Diagnostics (Phase 5)

### Concept 8.1: Feature Discretization Strategies
Continuous retrieval metrics must be mapped into finite ordinal categories for transaction rule mining.
1. **Equal-Width (Uniform) Discretization**:
   Given continuous feature $x$ with minimum $x_{\min}$ and maximum $x_{\max}$, the range is divided into $B = 3$ equal intervals of width $W = \frac{x_{\max} - x_{\min}}{3}$:
   - $\text{Low}: [x_{\min}, x_{\min} + W)$
   - $\text{Medium}: [x_{\min} + W, x_{\min} + 2W)$
   - $\text{High}: [x_{\min} + 2W, x_{\max}]$
   - *Advantage*: Preserves the absolute physical distance / similarity scale of retrieval metrics.
2. **Equal-Frequency (Quantile) Discretization**:
   Divides $x$ into intervals containing approximately equal sample counts ($\approx 33.3\%$ per bin) using empirical 33.3rd and 66.7th percentiles.
   - *Limitation*: When metrics have repeated values (e.g. zero values in lexical similarity), quantile bin edges collapse and require boundary deduplication.

### Concept 8.2: Frequent Itemset Mining & The Apriori Downward-Closure Property
- **Definitions**:
  Let $I = \{i_1, i_2, \dots, i_m\}$ be the set of all discretized feature-bin items and target categories.
  A transaction database $D = \{t_1, t_2, \dots, t_N\}$ consists of $N = 50$ query transactions, where each $t_k \subseteq I$.
- **Support of an Itemset $X \subseteq I$**:
  $$\text{Support}(X) = \frac{|\{t \in D : X \subseteq t\}|}{N} = P(X)$$
  An itemset $X$ is frequent if $\text{Support}(X) \ge \text{min\_support}$.
- **The Apriori Principle (Downward-Closure)**:
  *If an itemset $X$ is frequent, then all of its non-empty subsets $Y \subset X$ must also be frequent.*
  *Contrapositive*: If an itemset $Y$ is infrequent ($\text{Support}(Y) < \text{min\_support}$), then any superset $X \supset Y$ is guaranteed to be infrequent and can be pruned immediately without scanning the database.
- **Level-Wise Candidate Generation ($L_k \to C_{k+1} \to L_{k+1}$)**:
  1. Generate candidate 1-itemsets $C_1$, count support to form $L_1$.
  2. Join step: Join $L_k$ with $L_k$ where the first $k-1$ items match to form candidate $(k+1)$-itemsets $C_{k+1}$.
  3. Prune step: Remove any candidate in $C_{k+1}$ containing an infrequent $k$-subset.
  4. Database scan: Compute empirical support for candidates in $C_{k+1}$ and retain those meeting $\text{min\_support}$ to form $L_{k+1}$.
  5. Terminate when $L_{k+1} = \emptyset$ or $k = \text{max\_len}$.

### Concept 8.3: Association Rule Quality Metrics
An association rule takes the form $A \implies C$, where $A$ (antecedent feature pattern) and $C$ (consequent failure category) are disjoint itemsets ($A \cap C = \emptyset$):

1. **Rule Support**:
   $$\text{Support}(A \implies C) = P(A \cup C) = \frac{\sigma(A \cup C)}{N}$$
   Measures the fraction of all queries in the corpus that exhibit both the feature pattern $A$ and the failure mode $C$.

2. **Rule Confidence**:
   $$\text{Confidence}(A \implies C) = P(C \mid A) = \frac{P(A \cup C)}{P(A)} = \frac{\sigma(A \cup C)}{\sigma(A)}$$
   Measures the conditional probability of failure mode $C$ given that feature pattern $A$ has occurred. Confidence reflects diagnostic certainty.

3. **Rule Lift**:
   $$\text{Lift}(A \implies C) = \frac{P(A \cup C)}{P(A) \cdot P(C)} = \frac{\text{Confidence}(A \implies C)}{\text{Support}(C)}$$
   - $\text{Lift} = 1.0$: $A$ and $C$ are statistically independent.
   - $\text{Lift} > 1.0$: $A$ and $C$ are positively associated (the presence of feature pattern $A$ increases the likelihood of failure mode $C$).
   - In a balanced 5-class dataset where $P(C) = 0.20$, a perfect rule with $\text{Confidence} = 1.0$ achieves a maximum theoretical $\text{Lift} = \frac{1.0}{0.20} = 5.0$.

4. **Rule Leverage**:
   $$\text{Leverage}(A \implies C) = P(A \cup C) - P(A) \cdot P(C)$$
   Measures the difference between the observed frequency of co-occurrence and the frequency expected under mutual independence.

5. **Rule Conviction**:
   $$\text{Conviction}(A \implies C) = \frac{1 - P(C)}{1 - \text{Confidence}(A \implies C)} = \frac{P(A) \cdot P(\neg C)}{P(A \cup \neg C)}$$
   Measures the degree to which rule $A \implies C$ would be incorrect if $A$ and $C$ were independent. An infinite conviction indicates that the rule has zero counterexamples (100% confidence).

### Concept 8.4: Complexity Analysis
- **Time Complexity**:
  Worst-case itemset generation is $O(2^{|I|} \cdot N)$, where $|I|$ is the number of distinct discretized items. With pruning via Apriori downward closure and bounding itemset length to $\text{max\_len} = 3$, runtime reduces to $O(\sum_{k=1}^3 \binom{|I|}{k} \cdot N)$, executing in $< 3$ seconds on our 50-query dataset.
- **Space Complexity**:
  $O(|L| \cdot \text{max\_len})$ where $|L|$ is the number of retained frequent itemsets in memory.


---

# 9. Multi-Page Architecture, Caching Patterns & Empirical Causality

### 9.1 Streamlit Execution Model & Caching Semantics
In Streamlit applications, each user interaction (e.g. clicking a button, moving a slider, or switching pages) triggers a top-to-bottom re-execution of the active script. Without appropriate caching, heavy operations—such as vector retrieval from ChromaDB, PCA projections, or Joblib model deserialization—would re-run unnecessarily, introducing latency and UI freezing.

To achieve responsive performance, the platform implements a dual-tier caching strategy:
1. **`@st.cache_resource` (Singleton Object Persistence)**:
   - Used for global, non-serializable, or connection-heavy objects that should persist across reruns and user sessions.
   - Applied to: `get_chroma_connector()`, `get_analyzer_bundle()`, `get_trained_model_artifact()`, and `get_visualizer_engine()`.
   - Prevents reopening ChromaDB SQLite/HNSW index files or reloading serialized weights on every widget interaction.
2. **`@st.cache_data` (Computation & Data Frame Memoization)**:
   - Used for deterministic transformations and queries returning serializable data (DataFrames, dictionaries, lists).
   - Applied to: `fetch_retrieval_records(query, top_k)`, `cached_query_analysis(...)`, `get_cached_aggregate_stats(...)`, `load_features_dataset()`, and `run_apriori_pipeline(...)`.
   - Uses input parameters as cache keys: if a user adjusts Plotly chart sliders without altering the underlying query or thresholds, the cached result is returned instantaneously ($O(1)$ lookup).

### 9.2 Session State Bridging vs Disk Fallbacks
To decouple page navigation, every page follows a **Session State with Disk Fallback** pattern:
1. If the user navigates sequentially from Data Ingestion (`pages/2_Data_Ingestion.py`), data is retrieved directly from `st.session_state["validated_df"]`.
2. If the user directly accesses an analytical page (e.g., `pages/7_Failure_Diagnosis.py`), the platform automatically detects missing session variables and loads pre-computed benchmark artifacts from disk (`data/synthetic/sample_failed_queries.csv`, `data/processed/extracted_features.csv`, `models/best_diagnostic_model.pkl`).
3. This architecture prevents crash loops and ensures bookmarkable, non-linear navigation.

### 9.3 Statistical Association vs. True Causality in RAG Diagnostics
A foundational methodological principle emphasized across the platform is the distinction between **empirical association** and **causal mechanisms**:
* **Empirical Diagnostic Signal**: An ML model or Apriori rule identifies that high Top-5 distance ($d > 0.75$) strongly predicts a *Retrieval Miss* ($\text{Lift} = 5.0$).
* **Statistical Meaning**: $P(\text{Retrieval Miss} \mid d > 0.75) \approx 1.0$. The feature is a reliable diagnostic indicator.
* **Causal Reality**: The high distance does *not* cause the document to be missing. Rather, both the high distance and the retrieval miss are joint causal consequences of a common upstream root cause: **the omission of relevant knowledge chunks during corpus ingestion**.
* The platform provides diagnostic evidence to guide engineering remediation, strictly avoiding false claims of mathematical causality from correlational observations.
