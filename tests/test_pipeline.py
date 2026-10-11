#!/usr/bin/env python3
"""
Unit and integration test suite for Multilingual Entity Resolution Engine.
Tests:
- Linguistic normalization & legal suffix extraction
- 27-Feature extraction vector calculation
- Model prediction & Isotonic Calibration
- Meta AI FAISS dense HNSW vector search
- Qdrant embedded vector DB retrieval with payload filter
- LLM Regulatory Auditor explanation generation
"""

import os
import sys
import unittest
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from normalize import normalize_name, normalize_address, extract_legal_suffixes, extract_street_number
from dashboard_engine import EntityResolutionEngine, get_hitl_records
from vector_search import VectorDatabaseManager
from llm_judge import LLMJudge


class TestEntityResolutionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EntityResolutionEngine(artifacts_dir=str(REPO_ROOT / "artifacts"))
        cls.vector_mgr = VectorDatabaseManager(artifacts_dir=str(REPO_ROOT / "artifacts"))
        cls.judge = LLMJudge()

    def test_01_normalization(self):
        """Verify cross-lingual normalization and legal suffix stripping."""
        name_clean = normalize_name("Clairvoyant Récord Private Ltd")
        self.assertEqual(name_clean, "clairvoyant record")
        
        suffixes = extract_legal_suffixes("Acme Industries Private Limited")
        self.assertIn("pvt_ltd", suffixes)

        street_num = extract_street_number("630 45th Terrace, Kansas City, MO")
        self.assertEqual(street_num, "630")

    def test_02_model_inference_and_calibration(self):
        """Verify model prediction and Isotonic calibration on benchmark pair."""
        eval_res = self.engine.evaluate_pair(
            s1_name="Dahlia Power Reliable Scientific LLC",
            s1_addr="630 45th Terrace, Kansas City, MO",
            s1_country="US",
            t_name="Dahlia Power Reliable Scientific",
            t_addr="45ND TERRACE, null, KANSAS CITY, MO",
            t_country="US",
            target_id="S2-479876582"
        )
        self.assertEqual(eval_res["status"], "CONFIRMED_MATCH")
        self.assertGreater(eval_res["calibrated_prob"], 0.80)
        self.assertIn("sorted_shap", eval_res)
        self.assertEqual(len(eval_res["sorted_shap"]), 27)

    def test_03_co_location_guardrail(self):
        """Verify that co-located independent stores in same building are suppressed."""
        eval_res = self.engine.evaluate_pair(
            s1_name="Apex Diagnostic & Pathology Center",
            s1_addr="Tower B, Cyber Gateway, Hitech City, Hyderabad",
            s1_country="India",
            t_name="Blue Star Cloud Consultancy Services",
            t_addr="Tower B, Cyber Gateway, Hitech City, Hyderabad",
            t_country="India",
            target_id="S3-881920394"
        )
        self.assertTrue(eval_res["colocation_triggered"] or eval_res["status"] == "REJECTED")

    def test_04_vector_search_faiss_and_qdrant(self):
        """Verify dense vector search on FAISS and Qdrant with metadata filtering."""
        faiss_res = self.vector_mgr.query("Dahlia Scientific Kansas", top_k=1, backend="faiss")
        self.assertGreaterEqual(len(faiss_res["results"]), 1)
        self.assertGreater(faiss_res["results"][0]["similarity_score"], 0.40)

        qdrant_res = self.vector_mgr.query("Dahlia Scientific Kansas", top_k=1, country_filter="US", backend="qdrant")
        self.assertGreaterEqual(len(qdrant_res["results"]), 1)
        self.assertEqual(qdrant_res["results"][0]["country"], "US")

    def test_05_llm_regulatory_judge(self):
        """Verify deterministic regulatory audit explanation generation."""
        mock_eval = {
            "calibrated_prob": 0.999,
            "status": "CONFIRMED_MATCH",
            "status_label": "Confirmed Match",
            "colocation_triggered": False,
            "features": {
                "name_jaro_winkler": 0.95,
                "addr_token_overlap_ratio": 0.92,
                "name_embedding_cosine": 0.90,
                "suffix_normalized_equality": 0.0,
                "name_exact_match": 0.0,
                "addr_street_num_match": 1.0
            }
        }
        audit = self.judge.generate_audit_explanation(
            s1_name="Clairvoyant Récord Pvt Ltd",
            s1_addr="88 Olive Circle, Lebanon, TN",
            s1_country="US",
            t_name="Clairvoyant Record",
            t_addr="88 Olive Cir, Lebanon, TN",
            t_country="US",
            eval_result=mock_eval,
            backend="deterministic"
        )
        self.assertEqual(audit["verdict"], "MERGE CONFIRMED")
        self.assertIn("Entities merged because", audit["audit_justification"])


if __name__ == "__main__":
    unittest.main()
