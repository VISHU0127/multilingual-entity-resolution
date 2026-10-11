# System Architecture & Technical Specifications

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Multi-Source Ingestion & Normalization"]
        S1["Source 1: Enterprise Reference"]
        S2["Source 2: Supplier Stream"]
        S3["Source 3: Corporate Registries"]
        NORM["Linguistic Normalization Engine<br/>• NFKD Unicode Decomposing<br/>• Legal Suffix Stripping (LLC, Pvt Ltd, SARL)<br/>• Street & Postal Extraction"]
        S1 & S2 & S3 --> NORM
    end

    subgraph BLOCKING["2. Scalable Candidate Generation (99.999% Space Reduction)"]
        IDX1["Inverted Index: Token IDF Weighting"]
        IDX2["Phonetic Hashing: Soundex & Metaphone"]
        IDX3["Locality Indexing: Street Numbers & Pincodes"]
        FUSION["Country Partitioning & Fusion Capping (Top-60)"]
        NORM --> IDX1 & IDX2 & IDX3 --> FUSION
    end

    subgraph FEATURES["3. Dense Semantics & Pairwise Feature Engine"]
        FEAT["27 Pair-wise Features<br/>• Edit Distances: Jaro-Winkler, Levenshtein<br/>• Sub-word n-grams: Character 3-gram Jaccard<br/>• Legal Suffix Equality Flags<br/>• Street Number Tolerance Gating"]
        VEC["Multilingual Transformer (MiniLM-L12-v2)<br/>• 384-dimensional dense semantic vectors<br/>• Cross-script alignment (English vs Devanagari)"]
        FUSION --> FEAT & VEC
    end

    subgraph ML_DECISION["4. Supervised Ranking & Calibrated Decision Layer"]
        LGBM["LightGBM GBDT Classifier<br/>• Class Imbalance Weighting: scale_pos_weight = 4.05<br/>• ROC-AUC = 0.9993, PR-AUC = 0.9971"]
        CALIB["Isotonic Probability Calibration<br/>• Calibrated Brier Score = 0.0077<br/>• True Posterior P(match|x)"]
        GUARD["Guardrail Filters & Disambiguation<br/>• Co-location Guardrail (blocks shared malls)<br/>• Global 1:1 Bipartite Uniqueness"]
        FEAT & VEC --> LGBM --> CALIB --> GUARD
    end

    subgraph PRODUCTION_LAYER["5. Enterprise Delivery & Governance Layer"]
        VDB["Modern Vector Databases<br/>• Meta AI FAISS (HNSW Index)<br/>• Qdrant (Embedded Vector DB)<br/>• PostgreSQL pgvector Blueprint"]
        SHAP["TreeSHAP Feature Attribution<br/>• Exact additive logit waterfall decomposition"]
        LLM["LLM-as-a-Judge Regulatory Auditor<br/>• Gemini 3.8 Flash & Llama 3 compliance memos"]
        UI["Streamlit Interactive Console<br/>• Confidence Speedometer (tau = 0.40)<br/>• Human-in-the-Loop Review Queue"]
        GUARD --> VDB & SHAP & LLM --> UI
    end
```

---

## 2. Algorithmic Complexity Analysis

| Stage | Brute-Force Baseline | Our Engineered Architecture | Computational Speedup |
| :--- | :--- | :--- | :--- |
| **Candidate Generation** | $O(N_1 \times (N_2 + N_3)) \approx 2.2 \times 10^{13}$ | Inverted Index Lookup + Cap $K=60 \approx 1.3 \times 10^8$ | **$\sim 169,000 \times$ faster** |
| **Vector Similarity** | Flat scan $O(N \cdot D)$ | HNSW Hierarchical Graph Traversal $O(\log N)$ | **$< 2\text{ ms}$ query latency** |
| **Pair-wise Classification** | Deep Neural Cross-Encoder $O(L^2)$ | Tabular GBDT over 27 features | **$> 550\text{ pairs/sec}$ on CPU** |

---

## 3. Mathematical Foundations

### 3.1 Probability Calibration via Isotonic Regression
Gradient boosted decision trees trained with loss penalties (`scale_pos_weight`) output distorted ranking logits rather than true posterior probabilities. We solve:
$$\min_{m} \sum_{i=1}^n \left( y_i - m(\hat{p}_i) \right)^2 \quad \text{subject to } m(\hat{p}_i) \le m(\hat{p}_j) \text{ whenever } \hat{p}_i \le \hat{p}_j$$
This monotonic transformation reduces Brier score to **$0.0077$**, allowing an optimal decision threshold at $\tau = 0.40$.

### 3.2 Additive TreeSHAP Logit Decomposition
For each pair feature vector $x$:
$$\ln \frac{P(\text{match})}{1 - P(\text{match})} = \phi_0 + \sum_{j=1}^{27} \phi_j(x)$$
Where $\phi_0$ is the base expected prior log-odds, and $\phi_j$ is the exact marginal contribution of feature $j$.
