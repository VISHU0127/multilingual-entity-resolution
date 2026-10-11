#!/usr/bin/env python3
"""
Dashboard Engine for Business Entity Resolution
Provides:
- Real-time pair-wise feature extraction
- SentenceTransformer semantic embedding computation (cached)
- TreeSHAP feature attribution & contribution breakdown
- Calibrated probability inference (Isotonic Regression)
- Guardrail checks (Co-location guard, name floor, street number gate)
- Persistent Human-in-the-Loop (HITL) Queue & Audit Trail management
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import joblib

# Ensure path to src modules
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from features import extract_pair_features, FEATURE_NAMES
from decision import (
    load_decision_params,
    calibrated_probabilities,
    passes_pair_gates,
)

HITL_LOG_PATH = Path("artifacts/hitl_review_log.json")

class EntityResolutionEngine:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.calibrator = None
        self.params = None
        self.embedder = None
        self.load_models()

    def load_models(self):
        """Load LightGBM classifier, calibrator, and decision parameters."""
        model_path = self.artifacts_dir / "model.joblib"
        calibrator_path = self.artifacts_dir / "calibrator.joblib"
        params_path = self.artifacts_dir / "decision_params.json"

        if model_path.exists():
            self.model = joblib.load(model_path)
        if calibrator_path.exists():
            self.calibrator = joblib.load(calibrator_path)
        if params_path.exists():
            self.params = load_decision_params(params_path)
        else:
            self.params = {
                "optimal_threshold": 0.40,
                "confidence_gap": 0.05,
                "confidence_ceiling": 0.85,
                "colocation_guard": True,
                "global_uniqueness": True,
                "street_number_gate": {"enabled": True, "max_difference": 1},
                "name_floor": {
                    "enabled": True,
                    "indic_pattern": r"[\u0900-\u0D7F]",
                    "indic_jaro_winkler_min": 0.3,
                    "token_overlap_min": 0.5,
                    "levenshtein_min": 0.5,
                    "levenshtein_alternative_min": 0.75,
                },
                "budget_caps": {"S2": 5, "S3": 6, "total": 11},
            }

    def get_embedder(self):
        """Lazy load multilingual sentence transformer."""
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer: {e}")
                self.embedder = False
        return self.embedder

    def compute_embedding_cosine(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two text strings."""
        if not text1.strip() or not text2.strip():
            return 0.0
        embedder = self.get_embedder()
        if not embedder:
            # Fallback heuristic if embedding model not loaded
            return 0.5
        try:
            embeddings = embedder.encode([text1, text2], normalize_embeddings=True)
            return float(np.dot(embeddings[0], embeddings[1]))
        except Exception:
            return 0.5

    def evaluate_pair(
        self,
        s1_name: str,
        s1_addr: str,
        s1_country: str,
        t_name: str,
        t_addr: str,
        t_country: str,
        target_id: str = "S2-000000",
        s1_id: str = "S1-000000",
        blocking_score: float = 1.0,
        blocking_rank: int = 1,
    ) -> Dict[str, Any]:
        """
        Extracts features, computes TreeSHAP attributions, predicts calibrated probability,
        and evaluates guardrails.
        """
        # Semantic vector similarity
        name_emb_cosine = self.compute_embedding_cosine(s1_name, t_name)
        addr_emb_cosine = self.compute_embedding_cosine(s1_addr, t_addr)

        # 27 Discriminative Pair Features
        feats = extract_pair_features(
            s1_name_raw=s1_name,
            s1_addr_raw=s1_addr,
            s1_country_raw=s1_country,
            t_name_raw=t_name,
            t_addr_raw=t_addr,
            t_country_raw=t_country,
            target_id=target_id,
            blocking_score=blocking_score,
            blocking_rank=blocking_rank,
            name_emb_cosine=name_emb_cosine,
            addr_emb_cosine=addr_emb_cosine,
        )

        feature_vector = np.array([[feats[col] for col in FEATURE_NAMES]])

        # Model Inference
        raw_prob = float(self.model.predict_proba(feature_vector)[0, 1])
        cal_prob = float(calibrated_probabilities(self.model, self.calibrator, feature_vector)[0])

        # Exact Native TreeSHAP Explanations
        # predict(..., pred_contrib=True) returns [contrib_col_0, ..., contrib_col_26, base_value]
        shap_contribs = self.model.booster_.predict(feature_vector, pred_contrib=True)[0]
        base_value = float(shap_contribs[-1])
        feature_shap = {col: float(shap_contribs[i]) for i, col in enumerate(FEATURE_NAMES)}

        # Sort features by absolute SHAP impact
        sorted_shap = sorted(
            [{"feature": k, "shap_value": v, "feature_value": feats[k]} for k, v in feature_shap.items()],
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        # Guardrails check
        from normalize import extract_street_number
        source_meta = {"raw_name": s1_name, "street_num": extract_street_number(s1_addr)}
        target_meta = {"raw_name": t_name, "street_num": extract_street_number(t_addr)}
        passed_gates = passes_pair_gates(feats, source_meta, target_meta, self.params, cal_prob)

        # Status Determination
        # Borderline interval: [0.35, 0.45]
        is_borderline = 0.35 <= cal_prob <= 0.45
        is_match = (cal_prob >= 0.40) and passed_gates

        if is_match:
            status = "CONFIRMED_MATCH"
            status_label = "Confirmed Match"
            badge_color = "#10b981"  # Emerald green
        elif is_borderline:
            status = "BORDERLINE_REVIEW"
            status_label = "Borderline (Needs Review)"
            badge_color = "#f59e0b"  # Amber yellow
        else:
            status = "REJECTED"
            status_label = "Rejected / Non-Match"
            badge_color = "#ef4444"  # Red

        # Check if co-location trap was triggered
        colocation_triggered = False
        if feats["addr_token_overlap_ratio"] >= 0.70 and feats["name_jaro_winkler"] < 0.40 and feats["name_embedding_cosine"] < 0.40 and feats["name_token_jaccard"] == 0:
            colocation_triggered = True

        return {
            "s1_id": s1_id,
            "target_id": target_id,
            "raw_prob": raw_prob,
            "calibrated_prob": cal_prob,
            "passed_gates": passed_gates,
            "status": status,
            "status_label": status_label,
            "badge_color": badge_color,
            "is_borderline": is_borderline,
            "colocation_triggered": colocation_triggered,
            "features": feats,
            "base_value": base_value,
            "sorted_shap": sorted_shap,
        }


# ── Human-in-the-Loop (HITL) Persistence ──

def init_hitl_store():
    """Initializes default HITL storage if not existing."""
    if not HITL_LOG_PATH.exists():
        # Populate with a couple realistic pending queue items for rich initial experience
        initial_records = [
            {
                "review_id": "REV-2026-001",
                "timestamp": "2026-10-12 02:30:15",
                "s1_id": "S1-229104819",
                "target_id": "S2-661902847",
                "s1_name": "Cascade Mountain Outdoor Gear & Apparel LLC",
                "s1_addr": "1420 NW Gilman Blvd, Issaquah, WA",
                "t_name": "Cascade Outfitters",
                "t_addr": "Gilman Blvd, Issaquah, Washington",
                "country": "US",
                "calibrated_prob": 0.412,
                "status": "PENDING",
                "reviewer_notes": "",
                "reviewed_at": None,
                "reason": "Truncated trade name with missing street number component"
            },
            {
                "review_id": "REV-2026-002",
                "timestamp": "2026-10-12 02:35:40",
                "s1_id": "S1-889102451",
                "target_id": "S3-551029384",
                "s1_name": "Shree Ganesh Logistics & Freight Ltd",
                "s1_addr": "Plot 45, MIDC Industrial Area, Andheri East, Mumbai 400093",
                "t_name": "Ganesh Cargo Movers Pvt Ltd",
                "t_addr": "45 MIDC Indl Estate, Andheri (E), Mumbai",
                "country": "India",
                "calibrated_prob": 0.384,
                "status": "APPROVED",
                "reviewer_notes": "Verified business branch rebranding through MCA registry record #40291.",
                "reviewed_at": "2026-10-12 02:40:02",
                "reason": "Corporate branch name alteration with matching MIDC plot location"
            },
            {
                "review_id": "REV-2026-003",
                "timestamp": "2026-10-12 02:42:11",
                "s1_id": "S1-771928405",
                "target_id": "S2-110928374",
                "s1_name": "Royal Cafe & Bistro",
                "s1_addr": "12 Connaught Place, Block B, New Delhi",
                "t_name": "Royal Books & Stationary Emporium",
                "t_addr": "12 B-Block, CP, New Delhi 110001",
                "country": "India",
                "calibrated_prob": 0.362,
                "status": "REJECTED",
                "reviewer_notes": "Distinct businesses operating adjacent units at 12 CP. Cafe vs Bookstore.",
                "reviewed_at": "2026-10-12 02:45:18",
                "reason": "Identical commercial address building with different business operations"
            }
        ]
        HITL_LOG_PATH.parent.mkdir(exist_ok=True)
        with open(HITL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(initial_records, f, indent=2)

def get_hitl_records() -> List[Dict[str, Any]]:
    init_hitl_store()
    try:
        with open(HITL_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def queue_hitl_case(
    s1_id: str,
    target_id: str,
    s1_name: str,
    s1_addr: str,
    t_name: str,
    t_addr: str,
    country: str,
    calibrated_prob: float,
    reason: str = "Borderline confidence score"
) -> str:
    """Adds a new borderline candidate pair into the review queue."""
    records = get_hitl_records()
    # Check if pair already exists in queue
    for r in records:
        if r["s1_id"] == s1_id and r["target_id"] == target_id:
            return r["review_id"]

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_id = f"REV-{datetime.datetime.now().strftime('%Y%m%d')}-{len(records) + 1:03d}"
    records.insert(0, {
        "review_id": new_id,
        "timestamp": now_str,
        "s1_id": s1_id,
        "target_id": target_id,
        "s1_name": s1_name,
        "s1_addr": s1_addr,
        "t_name": t_name,
        "t_addr": t_addr,
        "country": country,
        "calibrated_prob": round(float(calibrated_prob), 4),
        "status": "PENDING",
        "reviewer_notes": "",
        "reviewed_at": None,
        "reason": reason
    })

    with open(HITL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return new_id

def update_hitl_decision(review_id: str, decision: str, notes: str = "") -> bool:
    """Updates a review case with APPROVED or REJECTED status and auditor notes."""
    records = get_hitl_records()
    found = False
    for r in records:
        if r["review_id"] == review_id:
            r["status"] = decision
            r["reviewer_notes"] = notes
            r["reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break
    if found:
        with open(HITL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    return found
