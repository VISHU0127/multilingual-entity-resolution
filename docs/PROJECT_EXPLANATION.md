# End-to-End Project Explanation

## 1. Problem statement

Organizations often maintain several records for the same business:

- a master or reference catalog,
- supplier or partner feeds,
- corporate registries,
- regional systems, or
- historical exports.

The same organization may appear with abbreviations, punctuation changes, legal-suffix changes, transliteration, missing fields, typos, or a different address format. A simple exact join misses valid matches, while an unrestricted fuzzy join is expensive and can create dangerous false merges.

This project implements a controlled record-linkage pipeline. It combines fast retrieval with richer pairwise evidence, then places the final decision behind explicit probability calibration and business guardrails.

## 2. End-to-end data flow

```text
Raw business records
        |
        v
Normalize names, addresses, countries, suffixes, postal codes, street numbers
        |
        v
Generate a small candidate set with lexical, phonetic, address, and country blocking
        |
        v
Build 27 features for every source/candidate pair
        |
        v
LightGBM match score
        |
        v
Probability calibration
        |
        v
Street/name/co-location/uniqueness guardrails
        |
        +--> Confirmed match
        +--> Borderline case --> human review queue
        +--> Rejected or blocked pair
        |
        v
TreeSHAP explanation and audit memorandum
```

The dashboard uses the same runtime engine for interactive scoring. Persisted artifacts make the demo reproducible without retraining on every startup.

The repository also includes a visual walkthrough in the README:

- [Live entity lookup](screenshots/live-entity-lookup.png)
- [TreeSHAP explanation](screenshots/treeshap-explanation.png)
- [Regulatory audit](screenshots/regulatory-audit.png)

## 3. Stage 1 — normalization

Normalization is deliberately conservative: it removes formatting noise while preserving evidence needed by later gates.

### Business names

`src/normalize.py`:

- decomposes Unicode text with NFKD,
- removes diacritic marks,
- lowercases and strips punctuation,
- collapses whitespace,
- removes URL/domain noise where applicable,
- canonicalizes legal suffixes such as LLC, Ltd, Pvt Ltd, and SARL,
- preserves a separate legal-suffix representation for feature extraction.

For example, `"Clairvoyant Récord Private Ltd"` becomes a comparable normalized form while the presence and canonical form of the legal suffix remain available as features.

### Addresses

Addresses are cleaned into comparable tokens. The pipeline separately extracts:

- postal codes,
- street numbers,
- address tokens,
- empty-address indicators.

Street numbers are retained because two businesses in the same building may share many address tokens but still represent different units or entities.

### Countries

Country values are normalized before partitioning and pairwise comparison. The blocking implementation does not hard-code a finite list of countries; it works with the values present in the input data.

## 4. Stage 2 — candidate generation

Comparing every Source 1 record with every target record is an `O(N × M)` operation. The blocking engine reduces this search space before expensive feature computation.

`src/blocking.py` builds country-specific indexes using three complementary strategies:

1. **Name-token inverted index**  
   Normalized name tokens point to target records. IDF weighting gives more influence to distinctive tokens and limits work for extremely common terms.
2. **Phonetic and prefix keys**  
   Soundex and short name prefixes recover likely candidates with spelling variation.
3. **Address and street-number keys**  
   Distinctive address tokens and extracted street numbers provide locality evidence and help recover records with weak or abbreviated names.

The candidate lists are score-fused and capped, with a default top-60 budget per source record. Country partitioning is attempted first, with fallback behavior available when the country is unknown or produces no candidates.

Blocking is a recall-oriented stage. It should not make the final merge decision; it only decides which pairs deserve detailed evaluation.

## 5. Stage 3 — pairwise feature engineering

For each candidate pair, `src/features.py` creates 27 numeric features in a fixed order. The order is persisted in the trained model’s expected input contract.

### Name evidence

- Jaro-Winkler similarity
- multilingual name embedding cosine similarity
- normalized Levenshtein similarity
- token Jaccard and token-overlap ratios
- character 3-gram Jaccard similarity
- first-token Soundex and Metaphone agreement
- acronym agreement
- exact normalized/raw-name equality
- normalized name-length difference

### Legal-suffix evidence

- canonical suffix equality
- suffix presence on the Source 1 record
- suffix presence on the target record

Suffixes are not treated as the business identity by themselves. They are supporting evidence that helps distinguish legal-form variation from a genuinely different name.

### Address evidence

- address token Jaccard and overlap ratios
- normalized address Levenshtein similarity
- address embedding cosine similarity
- postal-code equality or missingness handling
- street-number match
- empty target-address indicator

### Context and interaction evidence

- country equality
- target source indicator (`S2` or `S3`)
- blocking score
- blocking rank
- name/address similarity product

The combination is intentionally hybrid: lexical features are fast and interpretable, phonetic features tolerate spelling variation, embeddings help with cross-script or semantic similarity, and address features prevent name-only overmatching.

## 6. Stage 4 — model scoring and calibration

`src/train.py` trains a LightGBM gradient-boosted decision-tree classifier on pairwise feature vectors. This model is a practical choice for tabular matching because it:

- handles heterogeneous numeric features well,
- runs efficiently on CPU,
- supports class-imbalance weighting,
- exposes native TreeSHAP contributions,
- avoids the latency and explainability cost of a cross-encoder for every candidate pair.

The raw model output is a ranking signal, not automatically a trustworthy probability. Because the training configuration uses positive-class weighting, `src/train.py` evaluates calibration methods and persists the selected calibrator in `artifacts/calibrator.joblib`.

At runtime, `dashboard_engine.py`:

1. extracts a feature vector,
2. obtains the LightGBM positive-class score,
3. applies the persisted calibrator,
4. loads the operational threshold and guardrails from `artifacts/decision_params.json`.

This makes a displayed value such as `0.92` meaningful as an estimated match probability under the calibration data, rather than just an arbitrary model score.

## 7. Stage 5 — guardrails and decision policy

The final decision is not based on probability alone. `src/decision.py` applies policy checks before a pair is accepted.

### Street-number gate

If both records have numeric street numbers, a difference larger than the configured tolerance blocks the pair. This reduces false merges between nearby but distinct businesses.

### Name floor

The engine requires sufficient name evidence. Indic-script records use a dedicated Jaro-Winkler floor; other records use token-overlap and Levenshtein conditions, with an alternative high-Levenshtein path.

### Co-location guard

A high address overlap is not enough when names are unrelated. The guard detects cases with strong shared-address evidence but weak name and embedding evidence, such as independent businesses in the same office tower or mall.

### Borderline review interval

The dashboard identifies probabilities in the configured borderline interval (currently represented in the runtime as `0.35 <= p <= 0.45`) for human review. A pair can also be blocked by gates even when its probability is above the nominal threshold.

### Global uniqueness and budget caps

For batch resolution, accepted pairs are ordered by probability. The resolver can enforce:

- one target record claimed by at most one source record,
- a per-source budget,
- a total candidate budget,
- a confidence-gap rule when the top candidates are too close.

These controls prevent a single ambiguous target from being assigned repeatedly.

## 8. Stage 6 — explainability

LightGBM native TreeSHAP output provides one contribution for each of the 27 features plus a base value. The dashboard sorts contributions by absolute impact and displays a waterfall chart.

Interpretation:

- positive contributions push the pair toward match,
- negative contributions push it toward non-match,
- the base value is the model’s prior contribution,
- the sum of contributions and base value is the model logit.

The explanation should be read alongside the guardrail result. A high model contribution does not override a failed safety gate.

## 9. Stage 7 — dense vector retrieval

`src/vector_search.py` provides two retrieval backends over 384-dimensional multilingual sentence embeddings:

### FAISS

FAISS stores normalized vectors in an HNSW index using inner product, which is equivalent to cosine similarity after normalization. It is optimized for fast nearest-neighbor search and stores metadata in a companion JSON file.

### Qdrant

Qdrant runs in embedded mode and stores vectors with payload metadata. It supports native country filtering during the query, which is useful when retrieval must respect an operational partition.

`VectorDatabaseManager` loads the persisted FAISS index, initializes Qdrant storage, and lazily loads the SentenceTransformer model only when a new query needs embedding.

Dense retrieval complements, rather than replaces, lexical blocking. Embeddings help with transliteration and severe spelling variation, while lexical and address signals remain important for precision and auditability.

## 10. Stage 8 — audit explanations

`src/llm_judge.py` turns machine evidence into a concise compliance-style explanation.

The prompt includes:

- both records and countries,
- calibrated probability and decision status,
- name, address, and embedding metrics,
- street-number and suffix evidence,
- co-location state.

Three provider modes are supported:

1. **Deterministic**: offline and repeatable; always available as the fallback.
2. **Google Gemini**: enabled with `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
3. **Local Llama 3 through Ollama**: uses the local Ollama HTTP endpoint.

The generative providers are constrained to a JSON-shaped audit response, but the deterministic explanation remains the source of truth when external services are unavailable or unsuitable for sensitive data.

## 11. Stage 9 — human-in-the-loop operations

Borderline and selected audit cases can be routed to the HITL queue. The dashboard supports:

- pending-review inspection,
- side-by-side record comparison,
- approve or reject actions,
- reviewer notes,
- historical decision review,
- audit-log download.

Records are persisted to `artifacts/hitl_review_log.json`. In a production deployment, this file should be replaced with a transactional, access-controlled store and should include authenticated reviewer identity, immutable timestamps, and retention controls.

## 12. How the dashboard starts

`app.py` initializes three cached resources:

1. `EntityResolutionEngine`, which loads the model, calibrator, decision parameters, and demo artifacts.
2. Demo scenarios from `artifacts/demo_data.json`.
3. `VectorDatabaseManager`, which loads the FAISS index and hydrates embedded Qdrant when required.

Streamlit caching prevents heavyweight models and indexes from being rebuilt on every interaction. The dashboard then exposes lookup, explanations, retrieval comparisons, audit generation, and review management in one workflow.

## 13. Reproducing and extending the project

### Run the existing implementation

```bash
pip install -r requirements.txt
streamlit run app.py
python -m unittest discover -s tests -v
```

### Add a new normalization rule

Implement it in `src/normalize.py`, then add a focused test in `tests/test_pipeline.py`. Re-run feature and inference tests because normalization changes affect every downstream feature.

### Add a new pair feature

1. Add the feature name to `FEATURE_NAMES` in `src/features.py`.
2. Compute it in `extract_pair_features`.
3. Regenerate training matrices.
4. Retrain and validate the model.
5. Update tests and any explanation/UI tables that assume 27 features.

Changing feature order or count without retraining is not compatible with the checked-in model.

### Change the decision policy

Edit the generated or reviewed values in `artifacts/decision_params.json`, then test positive, borderline, co-location, and street-number cases. Keep policy changes separate from model changes so their effects can be measured independently.

## 14. Operational limitations

- Raw training datasets are not included in the repository; the checked-in model is an artifact, not a complete retraining dataset.
- The embedding model is downloaded by SentenceTransformers on first use if it is not cached locally.
- Embedded Qdrant is suitable for the demo and local evaluation, not a multi-user production deployment.
- A local JSON HITL log does not provide database transactions, access control, or immutable audit guarantees.
- External LLM providers may have availability, cost, privacy, and data-residency implications.
- Thresholds and reported metrics must be revalidated on representative data before automated merges are enabled.

## 15. Recommended production hardening

Before production use, add authenticated APIs and reviewers, durable storage, dataset/version lineage, monitoring for precision and recall drift, threshold approval workflows, PII handling controls, retry and timeout policies for external services, and an offline evaluation set that includes hard negatives such as co-located businesses and cross-script records.
