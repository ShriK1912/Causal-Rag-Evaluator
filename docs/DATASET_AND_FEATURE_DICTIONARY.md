# Dataset and Feature Dictionary

This document serves as the formal specification for all raw datasets, validation rules, taxonomies, and features processed by the **Causal RAG Failure Diagnostician Platform**.

---

## 1. Raw Dataset Specification

### Ingestion Schema

| Column Name | Internal Canonical Name | Data Type | Requirement Level | Description |
|---|---|---|---|---|
| `query` / `prompt` / `question` | `query` | String | **Mandatory** | The natural language question submitted by the user to the RAG system. Must be non-empty. |
| `expected_answer` / `ground_truth` / `reference_answer` | `expected_answer` | String | **Recommended** | The factually verified ground-truth response. Used for lexical overlap (ROUGE-L, Jaccard) comparisons. |
| `generated_answer` / `rag_response` / `model_answer` | `generated_answer` | String | **Recommended** | The response produced by the RAG generation component. Used to identify hallucinations and contradictions. |
| `failure_label` / `category` / `root_cause` | `failure_label` | String | **Optional** | Supervised ground-truth failure category. If provided, must match one of the 5 canonical taxonomy classes. |
| `query_id` / `id` / `qid` | `query_id` | String | **Optional** | Unique query identifier (e.g. `FAIL_001`, `Q_001`). Auto-generated if absent. |
| `provenance` / `source` | `provenance` | String | **Optional** | Audit trail recording dataset origin (e.g. `synthetic_controlled_v1`, `ms_marco_v2`, `user_uploaded`). |
| `target_doc_id` / `relevant_doc_id` | `target_doc_id` | String | **Optional** | Corpus document ID containing target factual evidence (`NONE` for Retrieval Miss). |
| `notes` / `explanation` | `notes` | String | **Optional** | Annotator notes or causal breakdown of the interaction failure. |

---

## 2. Failure Category Taxonomy

The platform strictly categorizes failures into five mutually distinguishable, academically grounded classes:

```
                  ┌───────────────────────────────────────────────┐
                  │           RAG Interaction Failures            │
                  └───────────────────────┬───────────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────┐                                   ┌───────────────────────┐
│   Retrieval-Phase     │                                   │   Generation-Phase    │
│       Failures        │                                   │       Failures        │
└───────────┬───────────┘                                   └───────────┬───────────┘
            │                                                           │
   ┌────────┴────────┐                                         ┌────────┴────────┐
   ▼                 ▼                                         ▼                 ▼
Retrieval         Ranking         Context                 Hallucination     Contradiction
  Miss             Error        Contamination
(Corpus Gap)  (Demoted Chunk)  (Distractor Noise)         (Ungrounded)    (Conflicting)
```

### Detailed Class Definitions:

1. **Retrieval Miss (Database Gap)**:
   - *Definition*: The necessary factual evidence to answer the query does not exist anywhere in the vector database corpus.
   - *Retrieval Symptom*: Both Top-5 and Top-100 searches yield high cosine distance outliers with negligible lexical overlap.
   - *Target Document ID*: `NONE`.

2. **Ranking Error**:
   - *Definition*: The target evidence exists in the vector corpus, but due to lexical mismatch or sub-optimal embedding proximity, it is ranked below the operational retrieval cutoff (e.g., ranked at rank 15 or 42 rather than Top-5).
   - *Retrieval Symptom*: Top-5 results miss the evidence, but deep-fetch Top-100 contains the target document with low distance.
   - *Target Document ID*: Present in corpus.

3. **Context Contamination**:
   - *Definition*: Top-5 retrieval returns irrelevant, polysemous, or misleading distractor documents that confuse the generator into generating an erroneous response.
   - *Retrieval Symptom*: High variance across Top-5 retrieved chunks; semantic divergence between retrieved chunks and target topic.

4. **Hallucination**:
   - *Definition*: Top-5 retrieval successfully supplies the correct factual evidence, but the generator synthesizes claims unsupported by the context.
   - *Retrieval Symptom*: Top-5 retrieval distance is low and context relevance is high, but similarity between `generated_answer` and retrieved chunks is low.

5. **Contradiction**:
   - *Definition*: Retrieved Top-5 chunks contain mutually conflicting factual assertions (e.g., opposing dates, differing statistics), causing the generator to emit conflicting or arbitrary statements.
   - *Retrieval Symptom*: High pairwise semantic distance or opposing factual statements between Top-5 retrieved documents.

---

## 3. Data Cleaning & Validation Rules

| Rule | Condition | Action Taken | Logging |
|---|---|---|---|
| **Mandatory Query Check** | `query` column absent | Ingestion aborted | Critical Error in `validation_errors` |
| **Empty Query Filter** | `query` is NaN, `""`, or whitespace | Row removed from `cleaned_df`, moved to `invalid_df` | Logged in `validation_warnings` with reason |
| **All-NaN Row Removal** | Entire row is empty | Row dropped | Logged in `validation_warnings` |
| **Duplicate Query Filter** | Query string repeats earlier entry | Duplicate dropped from `cleaned_df`, moved to `invalid_df` | Count logged in `duplicate_records_count` |
| **Taxonomy Validation** | `failure_label` not in 5 classes | Row moved to `invalid_df` | Specific unrecognized label reported |
| **Unlabeled Handling** | `failure_label` missing/NaN | Dataset marked valid for Unlabeled Inference Mode | Noted in `has_labels = False` |
| **Missing Recommended** | `expected_answer` missing | Row kept valid; flag logged | Warning in `missing_recommended_columns` |
| **Provenance Tracking** | `provenance` column missing | Auto-assigned default (`"user_uploaded"`) | Logged in `validation_messages` |

---

## 4. Extracted Feature Dictionary (`extracted_features.csv`)

The feature dataset contains 50 rows (one per query interaction) and 40 columns. Below is the comprehensive technical dictionary for every feature:

### Identity & Target Columns
1. `query_id` (String): Unique identifier (e.g. `FAIL_001`).
2. `query` (String): Natural language query string.
3. `provenance` (String): Provenance audit tag (`synthetic_controlled_v1`).
4. `failure_label` (String): Ground-truth target diagnostic class (5 classes: `Retrieval Miss`, `Ranking Error`, `Context Contamination`, `Hallucination`, `Contradiction`).

---

### Distance-Based Features (Top-5 Baseline vs Deep-Fetch Ranks 6–100)

| Feature Name | Type | Formula / Definition | Expected Range | Diagnostic Relevance |
|---|---|---|---|---|
| `top5_dist_mean` | Float | $\frac{1}{5} \sum_{i=1}^5 d_i$ | $[0.0, 2.0]$ | High in **Retrieval Miss** (all distant chunks); lower in **Contradiction** and **Hallucination**. |
| `top5_dist_std` | Float | $\sqrt{\frac{1}{5} \sum_{i=1}^5 (d_i - \mu_5)^2}$ | $\ge 0.0$ | High in **Context Contamination** (cluster divergence between target and distractor). |
| `top5_dist_var` | Float | $\sigma_5^2$ | $\ge 0.0$ | Variance of Top-5 distances. |
| `top5_dist_min` | Float | $\min(d_1, \dots, d_5) = d_1$ | $[0.0, 2.0]$ | Distance of nearest retrieved document. Near $0.0$ indicates strong semantic match. |
| `top5_dist_max` | Float | $\max(d_1, \dots, d_5) = d_5$ | $[0.0, 2.0]$ | Boundary distance of operational retrieval set. |
| `top5_dist_spread` | Float | $d_5 - d_1$ | $\ge 0.0$ | Distance breadth across the top-5 candidate set. |
| `deep_dist_mean` | Float | $\frac{1}{95} \sum_{i=6}^{100} d_i$ | $[0.0, 2.0]$ | Baseline distance profile of the broader corpus. |
| `deep_dist_std` | Float | $\sigma_{6-100}$ | $\ge 0.0$ | Dispersion across deeper retrieval tiers. |
| `deep_dist_var` | Float | $\sigma_{6-100}^2$ | $\ge 0.0$ | Variance of ranks 6 to 100. |
| `deep_dist_min` | Float | $d_6$ | $[0.0, 2.0]$ | Distance at start of deep fetch tier. |
| `deep_dist_max` | Float | $d_{100}$ | $[0.0, 2.0]$ | Furthest distance in Top-100. |
| `deep_dist_spread` | Float | $d_{100} - d_6$ | $\ge 0.0$ | Spread across deep fetch tier. |
| `all_dist_mean` | Float | $\frac{1}{100} \sum_{i=1}^{100} d_i$ | $[0.0, 2.0]$ | Global retrieval distance average. |
| `all_dist_std` | Float | $\sigma_{100}$ | $\ge 0.0$ | Global standard deviation. |
| `all_dist_var` | Float | $\sigma_{100}^2$ | $\ge 0.0$ | Global variance. |
| `all_dist_min` | Float | $d_1$ | $[0.0, 2.0]$ | Global minimum distance. |
| `all_dist_max` | Float | $d_{100}$ | $[0.0, 2.0]$ | Global maximum distance. |
| `all_dist_spread` | Float | $d_{100} - d_1$ | $\ge 0.0$ | Total distance dynamic range. |
| `dist_diff_deep_vs_top5` | Float | $\mu_{6-100} - \mu_5$ | $[-2.0, 2.0]$ | Semantic contrast between Top-5 and deeper corpus. Low contrast signifies uniform irrelevance (**Retrieval Miss**). |
| `dist_dropoff_ratio` | Float | $\frac{d_5 - d_1}{\max(1e-6, d_{100} - d_1)}$ | $[0.0, 1.0]$ | Proportion of total distance drop occurring inside Top-5. |
| `top5_l2_mean_equiv` | Float | $\sqrt{2 \cdot \mu_5}$ | $\ge 0.0$ | Euclidean L2 equivalent distance for Top-5. |
| `all_l2_mean_equiv` | Float | $\sqrt{2 \cdot \mu_{100}}$ | $\ge 0.0$ | Euclidean L2 equivalent distance for Top-100. |

---

### Text Similarity & Lexical Overlap Features

| Feature Name | Type | Formula / Definition | Expected Range | Diagnostic Relevance |
|---|---|---|---|---|
| `jaccard_expected_vs_top1` | Float | $\frac{\|T(\text{exp}) \cap T(\text{top1})\|}{\|T(\text{exp}) \cup T(\text{top1})\|}$ | $[0.0, 1.0]$ | Token overlap between ground truth and rank 1 chunk. |
| `jaccard_expected_vs_top5` | Float | $\frac{\|T(\text{exp}) \cap T(\text{top5})\|}{\|T(\text{exp}) \cup T(\text{top5})\|}$ | $[0.0, 1.0]$ | Overlap between expected answer and concatenated Top-5 context. Low in **Retrieval Miss** and **Ranking Error**. |
| `rougel_expected_vs_top1` | Float | ROUGE-L F1 (expected, top1) | $[0.0, 1.0]$ | Syntactic LCS preservation between ground truth and rank 1. |
| `rougel_expected_vs_top5` | Float | ROUGE-L F1 (expected, top5) | $[0.0, 1.0]$ | Syntactic coverage of ground truth inside Top-5. |
| `jaccard_expected_vs_generated` | Float | Jaccard(expected, generated) | $[0.0, 1.0]$ | Accuracy of the generated answer against ground truth. |
| `rougel_expected_vs_generated` | Float | ROUGE-L F1(expected, generated)| $[0.0, 1.0]$ | Sequential syntactic overlap of generated answer with ground truth. |
| `jaccard_generated_vs_top5` | Float | Jaccard(generated, top5) | $[0.0, 1.0]$ | **Hallucination Detector**: If Top-5 contains evidence but generated answer has near-zero overlap with Top-5, claims were fabricated! |
| `rougel_generated_vs_top5` | Float | ROUGE-L F1(generated, top5) | $[0.0, 1.0]$ | **Hallucination Detector**: Syntactic ungroundedness score. |

---

### Retrieval Coverage & Demotion Features

| Feature Name | Type | Definition | Range | Diagnostic Relevance |
|---|---|---|---|---|
| `num_retrieved_docs` | Integer | Total documents retrieved (100) | $\ge 1$ | Retrieval budget depth. |
| `target_doc_rank` | Integer | Rank of gold target document (-1 if absent) | $[-1, 100]$ | Crucial: Values $> 5$ pinpoint **Ranking Error** demotions. |
| `target_in_top5` | Binary | 1 if $1 \le \text{rank} \le 5$ else 0 | $\{0, 1\}$ | Indicates whether target evidence was present in operational prompt. |
| `target_in_deep_fetch` | Binary | 1 if $6 \le \text{rank} \le 100$ else 0 | $\{0, 1\}$ | **Ranking Error Flag**: Evidence was found by deep fetch but missed by Top-5. |
| `is_database_gap` | Binary | 1 if target not in Top-100 else 0 | $\{0, 1\}$ | **Retrieval Miss Flag**: Empirical evidence of missing corpus information. |
| `retrieval_overlap_top5_top100` | Float | $\frac{5}{\min(100, N)}$ | $[0.0, 1.0]$ | Baseline to deep fetch ratio ($0.05$). |
