# Multilingual Entity Resolution & Deduplication Engine

An explainable, multilingual record-linkage system for deciding whether business records from different sources refer to the same real-world organization.

The project combines linguistic normalization, scalable candidate blocking, pairwise machine-learning features, calibrated LightGBM scoring, deterministic guardrails, dense vector retrieval, TreeSHAP explanations, and a Streamlit review console.

## What this project does

Given a reference business record and one or more candidate records, the engine:

1. Normalizes names, addresses, countries, legal suffixes, postal codes, and street numbers.
2. Retrieves a small candidate set instead of comparing every possible pair.
3. Computes 27 lexical, phonetic, address, semantic, and metadata features.
4. Scores each pair with a trained LightGBM classifier.
5. Converts the model score into a calibrated match probability.
6. Applies safety gates such as street-number checks, name floors, co-location protection, and uniqueness constraints.
7. Returns a decision with feature values and TreeSHAP contributions.
8. Routes ambiguous cases to a persistent human-in-the-loop queue.
9. Generates deterministic or LLM-assisted audit explanations.

The included Streamlit dashboard provides an interactive way to inspect these decisions.

## Key capabilities

- **Multilingual normalization**: Unicode decomposition, punctuation cleanup, legal-suffix canonicalization, country normalization, and address extraction.
- **Candidate blocking**: country partitions, IDF-weighted token indexing, phonetic keys, prefixes, address tokens, and street numbers.
- **Pairwise matching**: Jaro-Winkler, Levenshtein, token overlap, character 3-grams, Soundex, Metaphone, acronym checks, postal/street checks, and multilingual embeddings.
- **Calibrated decisions**: LightGBM ranking/classification followed by the persisted calibration model and an operational threshold.
- **False-positive guardrails**: protects against co-located businesses, implausible street-number mismatches, weak names, uncertain top-candidate gaps, and duplicate target assignments.
- **Semantic retrieval**: FAISS HNSW and embedded Qdrant backends using `paraphrase-multilingual-MiniLM-L12-v2` embeddings.
- **Explainability**: exact native TreeSHAP contributions for the 27 model features.
- **Human review**: pending and historical HITL records persisted in `artifacts/hitl_review_log.json`.
- **Audit layer**: deterministic offline explanations plus optional Google Gemini or local Ollama/Llama 3 integrations.

## Quick start

### 1. Create an environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the dashboard

The repository includes model, calibration, demo, and vector-search artifacts, so the dashboard can be started without the original training datasets:

```bash
streamlit run app.py
```

Open <http://localhost:8501>.

### 3. Run the tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers normalization, model inference, calibration, co-location protection, FAISS/Qdrant retrieval, and deterministic audit explanations.

## Dashboard workflow

The application is organized around these operational tasks:

1. **Entity lookup**: choose a supplied demo scenario or enter a custom Source 1 record.
2. **Candidate evaluation**: inspect resolved candidates, calibrated probabilities, decision status, and key similarity metrics.
3. **TreeSHAP explanation**: see which of the 27 features increased or decreased the model score.
4. **Dense retrieval benchmark**: compare FAISS and Qdrant candidate retrieval, with optional country filtering.
5. **Regulatory audit**: produce a deterministic, Gemini, or local Llama 3 explanation.
6. **HITL review queue**: approve or reject borderline cases and record reviewer notes.

## Repository layout

```text
.
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── src/
│   ├── normalize.py               # Text, address, suffix, and country normalization
│   ├── blocking.py                # Candidate generation and blocking indexes
│   ├── features.py                # 27 pairwise matching features
│   ├── train.py                   # LightGBM training and calibration workflow
│   ├── decision.py                # Thresholds, gates, uniqueness, and caps
│   ├── dashboard_engine.py        # Runtime scoring, SHAP, and HITL persistence
│   ├── vector_search.py           # FAISS and Qdrant retrieval
│   ├── llm_judge.py               # Audit explanation providers
│   └── config.py                  # Project paths and shared configuration
├── artifacts/
│   ├── model.joblib               # Trained LightGBM model
│   ├── calibrator.joblib          # Persisted probability calibrator
│   ├── decision_params.json       # Runtime thresholds and guardrails
│   ├── demo_data.json             # Dashboard demo scenarios
│   ├── faiss_index.bin            # Persisted FAISS index
│   ├── faiss_meta.json            # FAISS record metadata
│   ├── feature_importances.json   # Model feature importance report
│   └── hitl_review_log.json       # Review queue and audit history
├── docs/
│   ├── ARCHITECTURE.md            # Architecture and technical specifications
│   ├── PROJECT_EXPLANATION.md     # End-to-end project explanation
│   └── VIVA_PRESENTATION_GUIDE.md # Demo and presentation guide
└── tests/
    └── test_pipeline.py           # Unit and integration tests
```

## Model decision semantics

The runtime distinguishes three outcomes:

- **Confirmed Match**: calibrated probability meets the configured threshold and all gates pass.
- **Borderline (Needs Review)**: probability falls in the configured review interval.
- **Rejected / Non-Match**: the pair does not meet the probability threshold or a guardrail blocks it.

The authoritative values are loaded from [`artifacts/decision_params.json`](artifacts/decision_params.json), rather than duplicated in the UI. This keeps the dashboard and batch decision logic aligned.

## Optional audit providers

The deterministic auditor works offline and is the default safe fallback. To use Gemini, provide an API key through the environment:

```bash
export GEMINI_API_KEY="your-key"
streamlit run app.py
```

The Llama 3 option expects a local Ollama server at `http://localhost:11434/api/generate`. Do not commit API keys or other secrets; `.streamlit/secrets.toml` is ignored by Git.

## Training and data notes

The checked-in `artifacts/` directory contains the runtime assets used by the demo. The raw training datasets are intentionally excluded by `.gitignore`. The training script expects generated feature matrices at:

```text
artifacts/train_features.npz
artifacts/val_features.npz
```

To retrain the model, provide the project’s training data and feature-generation step, generate those matrices, then run:

```bash
python src/train.py
```

Retraining replaces model and calibration artifacts, so validate the resulting metrics and decision behavior before deployment.

## Documentation

- [End-to-end project explanation](docs/PROJECT_EXPLANATION.md)
- [System architecture and technical specifications](docs/ARCHITECTURE.md)
- [Viva and presentation guide](docs/VIVA_PRESENTATION_GUIDE.md)

## Responsible use

Entity resolution can affect financial, compliance, customer, and operational records. Treat model output as decision support, retain audit evidence, review borderline cases, monitor false merges and missed matches, and validate thresholds against the deployment domain before using automated merges in production.
