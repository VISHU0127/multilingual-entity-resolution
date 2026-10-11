# Interactive Entity Resolution Dashboard — Guide & Viva Preparation

An interactive full-stack dashboard for the **Multilingual Business Entity Resolution & Record Linkage Engine**, built with Streamlit, Plotly, LightGBM, and SentenceTransformers.

---

## 🚀 Quickstart

To launch the dashboard locally:

```bash
streamlit run app.py
```

Open your browser at: **`http://localhost:8501`**

---

## 🌟 Core Features Implemented

### 1. 🔍 Live Entity Lookup & Cluster Disambiguation
- **Query Input**: Select from pre-loaded enterprise benchmark scenarios (US corporate records, Devanagari cross-script, French zero-shot records, Co-location traps) or type any custom company name, address, and country.
- **Candidate Retrieval**: Real-time evaluation of candidate target records across Source 2 (supplier feeds) and Source 3 (state registries).
- **Cluster Visualization**: Renders resolved matches with status badges (`Confirmed Match`, `Borderline (Needs Review)`, or `Non-match`), calibrated probabilities, and similarity metrics.
- **Co-Location Guardrail**: Automatically suppresses unrelated independent businesses that share identical commercial suites or shopping mall addresses.

### 2. 📊 Match Explanation & TreeSHAP Attribution
- **Speedometer Confidence Gauge**: Visualizes calibrated match probability ($0\%$ to $100\%$) with color-coded safety bands (Red $<35\%$, Amber $35-45\%$, Green $>45\%$) and the operational decision threshold at $\tau = 0.40$.
- **TreeSHAP Waterfall Chart**: Leverages exact tree-path additivity from LightGBM to show which features pushed the log-odds up (+logits in green) and which pulled it down (-logits in red) from the base prior value.
- **Key Feature Breakdown**:
  - Address Token Overlap Ratio (e.g. $92\%$)
  - Name Jaro-Winkler Similarity (e.g. $88\%$)
  - Multilingual Semantic Vector Cosine (MiniLM) (e.g. $0.84$)
  - Street Number Match Status (Exact Match / Mismatch / Missing)
  - Legal Suffix Normalized Equality (Match / Suffix dropped)
  - Full 27-feature inspection table with individual SHAP impacts.

### 3. 🛡️ Human-in-the-Loop (HITL) Queue & Governance
- **Automated Queue Routing**: Any candidate pair with ambiguous or borderline confidence ($0.35 \le p \le 0.45$) is automatically flagged for human supervisor review.
- **Auditor Console**:
  - Side-by-side comparison of Source 1 vs Candidate target.
  - Interactive **✅ Approve Match** and **❌ Reject Match** workflow.
  - Auditor note taking / justification log (e.g., *"Verified legal trade name via business registry"*).
- **Persistent Audit Trail**: Records are saved to `artifacts/hitl_review_log.json` and downloadable as a compliance-ready CSV.

### 4. ⚡ Vector Search & Database Integration (FAISS / Qdrant / pgvector)
- **FAISS (Meta AI)**: In-memory Hierarchical Navigable Small World (`IndexHNSWFlat`, $m=32$) graph index for sub-millisecond Approximate Nearest Neighbor (ANN) search over 384-dimensional dense vectors.
- **Qdrant (Embedded Vector DB)**: High-performance embedded collection with native single-stage **payload metadata filtering** (e.g. filtering candidates by `country = 'India'` or `source = 'Source 2'`) and Cosine distance metric.
- **PostgreSQL + pgvector Blueprint**: Complete enterprise DDL script creating HNSW vector indexes with `vector_cosine_ops` and sample asynchronous Python queries (`psycopg3`/`SQLAlchemy`).
- **Real-Time Latency Benchmark**: Decomposes query vectorization latency (~15ms) vs HNSW graph traversal latency (<2ms).
- **Architectural Trade-offs Matrix**: Direct comparison between Lexical Inverted Indexing, FAISS, Qdrant, and pgvector.

### 5. 🤖 LLM-as-a-Judge / Explanation Layer (GenAI Touch)
- **Multi-Engine Support**: Supports **Google Gemini 3.8 Flash** (`gemini-3.8-flash` via official `google-genai` SDK), **Meta Llama 3** (via local Ollama REST API), and a **Deterministic Regulatory Compliance Engine** (zero-dependency offline fallback).
- **Compliance Justification Format**: Generates concise, audit-ready single-sentence explanations for regulators (Dodd-Frank, Basel III, AML/KYC):
  > *"Entities merged because 'Clairvoyant Récord Pvt Ltd' and 'Clairvoyant Record' share matching street identifiers and geographical locality (US) despite legal suffix abbreviations and minor spelling/diacritic variations (Calibrated Match Confidence: 100.0%)."*
- **Evidentiary Breakdown**: Synthesizes address token overlap, Jaro-Winkler string similarity, street number unit checks, and multilingual embedding alignment.
- **Export Audit Memorandum**: Generates downloadable Markdown compliance reports with date, auditor signature, and risk rating.

### 6. 📑 System Architecture & Model Cards
- Complete pipeline diagram, academic performance metrics (Local Holdout Macro $F_{0.5} = 0.9662$, Singleton Accuracy $95.1\%$, Brier Score $0.0077$), and defense talking points.

---

## 🎯 Viva & Interview Defense Cheat-Sheet

| Question | Strong Answer |
| :--- | :--- |
| **Why not just compare all records directly?** | Comparing 2.2M Source 1 records against 10.3M targets requires $>2.2 \times 10^{13}$ pairwise comparisons ($O(N \times M)$), taking weeks. Our 3-strategy inverted index blocking reduces search space by **99.999%** with **96%+ recall**. |
| **Why use an LLM-as-a-Judge in an ML pipeline?** | In regulated domains (banking KYC, healthcare, fraud prevention), raw machine learning probabilities ($p = 0.999$) are insufficient for compliance audits. An **LLM-as-a-Judge** translates complex high-dimensional feature vectors and TreeSHAP attributions into clear, human-readable legal justifications explaining *why* entities were merged or blocked. |
| **How does your LLM layer avoid hallucinations?** | The LLM prompt is strictly grounded with deterministic telemetry: raw text strings, exact Jaro-Winkler scores, token overlap percentages, unit number checks, and TreeSHAP feature attributions. The model is constrained to output strict JSON schemas and acts as an auditor rather than an unconstrained generator. |
| **Why add Vector DBs (FAISS/Qdrant) alongside Inverted Index?** | Inverted indices fail on cross-script transliteration (e.g., Hindi Devanagari vs English) or severe typos where no tokens overlap. Dense vector search via SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`) maps cross-script equivalents to the same vector space, enabling **hybrid lexical + semantic retrieval**. |
| **What is the difference between FAISS and Qdrant in your architecture?** | **FAISS** provides maximum raw C++ search throughput for pure vector similarity (<1ms), but lacks native metadata filtering. **Qdrant** provides single-stage payload filtering (e.g., restricting search to specific country partitions or supplier sources) without losing recall. |
| **How does pgvector fit into enterprise Master Data Management?** | Organizations using PostgreSQL can use `pgvector` with HNSW indexes (`m=32, ef_construction=128`) to store company embeddings inside existing relational tables with ACID transactions, eliminating the need to maintain a separate external vector database. |
| **Why LightGBM instead of pure Deep Learning?** | Tabular pair-wise matching with string distances and regex features runs orders of magnitude faster on CPUs ($>550$ pairs/sec), requires $<1.5$ GB RAM, and provides exact TreeSHAP explainability for regulatory audits. |
| **How did you prevent co-located false merges?** | In dense commercial centers, different businesses share the exact building address. We designed a **Co-location Guardrail** and name-similarity floor that prevents merges when name similarity is negligible despite high address overlap. |
| **Why do you calibrate probabilities with Isotonic Regression?** | Raw GBDT scores with `scale_pos_weight = 4.05` are uncalibrated rankings, not true probabilities. Isotonic regression maps raw scores to true posterior probabilities $P(\text{match} \mid x)$, enabling meaningful confidence thresholds and borderline queuing. |
