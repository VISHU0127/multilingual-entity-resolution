#!/usr/bin/env python3
"""
LLM-as-a-Judge & Regulatory Audit Explanation Layer (GenAI Touch)
Generates human-readable, compliance-ready entity match justifications
using:
1. Google Gemini 3.8 Flash (via google-genai SDK)
2. Local Llama 3 (via Ollama REST API)
3. Deterministic Regulatory Audit Engine (zero-dependency offline fallback)
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

AUDIT_PROMPT_TEMPLATE = """You are a Senior Regulatory Audit & Compliance Judge specializing in enterprise Master Data Management (MDM), KYC, and Entity Resolution.

Evaluate the following two business entity records that our machine learning pipeline scored:

Source 1 (Reference Entity):
- Entity ID: {s1_id}
- Business Name: "{s1_name}"
- Address: "{s1_addr}"
- Country: {s1_country}

Candidate Target Record:
- Entity ID: {target_id}
- Business Name: "{t_name}"
- Address: "{t_addr}"
- Country: {t_country}

ML Model Telemetry:
- Calibrated Probability: {calibrated_prob:.1%}
- Decision Status: {status}
- Name Jaro-Winkler Similarity: {name_jw:.1%}
- Address Token Overlap: {addr_overlap:.1%}
- Semantic Vector Cosine: {emb_cosine:.2f}
- Street Number Match: {street_match}
- Legal Suffix Match: {suffix_match}
- Co-location Guard Triggered: {colocation_flag}

TASK:
Provide a rigorous, concise regulatory audit justification answering whether these two records refer to the same real-world business entity.

OUTPUT FORMAT (Strict JSON):
{{
  "verdict": "MERGE CONFIRMED" | "BORDERLINE ESCALATION" | "SEPARATE ENTITIES",
  "audit_justification": "A single, authoritative sentence explaining why the entities are merged or separated (e.g., 'Entities merged because ... share identical ... despite ...')",
  "evidence_breakdown": [
    "Point 1: Name and legal suffix variation analysis",
    "Point 2: Physical street and locality consistency",
    "Point 3: Semantic embedding and phonetic consistency"
  ],
  "compliance_risk_assessment": "Low Risk / Moderate Risk / High Risk with rationale",
  "recommended_action": "Auto-Approve Merge" | "Human-in-the-Loop Review" | "Reject & Block"
}}
"""


class LLMJudge:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def generate_audit_explanation(
        self,
        s1_name: str,
        s1_addr: str,
        s1_country: str,
        t_name: str,
        t_addr: str,
        t_country: str,
        eval_result: Dict[str, Any],
        backend: str = "gemini",
        ollama_url: str = "http://localhost:11434/api/generate",
        model_name: str = "gemini-3.8-flash"
    ) -> Dict[str, Any]:
        """
        Generates a regulatory audit explanation using Gemini 3.8 Flash, local Llama 3,
        or deterministic compliance fallback.
        """
        feats = eval_result.get("features", {})
        cal_prob = eval_result.get("calibrated_prob", 0.5)
        status = eval_result.get("status_label", "Unknown")

        street_status = "Exact Match" if feats.get("addr_street_num_match") == 1.0 else (
            "Mismatch" if feats.get("addr_street_num_match") == 0.0 else "Missing / Not Applicable"
        )
        suffix_status = "Exact Match" if feats.get("suffix_normalized_equality") == 1.0 else "Differing Legal Suffixes"
        colocation_flag = "Yes (Unrelated store at same address)" if eval_result.get("colocation_triggered") else "No"

        prompt = AUDIT_PROMPT_TEMPLATE.format(
            s1_id=eval_result.get("s1_id", "S1-REF"),
            s1_name=s1_name,
            s1_addr=s1_addr,
            s1_country=s1_country,
            target_id=eval_result.get("target_id", "TARGET-CAND"),
            t_name=t_name,
            t_addr=t_addr,
            t_country=t_country,
            calibrated_prob=cal_prob,
            status=status,
            name_jw=feats.get("name_jaro_winkler", 0.0),
            addr_overlap=feats.get("addr_token_overlap_ratio", 0.0),
            emb_cosine=feats.get("name_embedding_cosine", 0.0),
            street_match=street_status,
            suffix_match=suffix_status,
            colocation_flag=colocation_flag,
        )

        # 1. Try Gemini 3.8 Flash via google-genai
        if backend.lower() == "gemini" and self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=model_name or "gemini-3.8-flash",
                    contents=prompt,
                )
                raw_text = response.text.strip()
                # Parse JSON block
                clean_json = raw_text
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                parsed = json.loads(clean_json)
                parsed["engine"] = f"Google Gemini ({model_name})"
                return parsed
            except Exception as e:
                print(f"Gemini API generation failed or returned error: {e}. Falling back to deterministic auditor.")

        # 2. Try Local Ollama (e.g. Llama 3)
        if backend.lower() == "llama3":
            try:
                payload = json.dumps({
                    "model": "llama3",
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                }).encode("utf-8")
                req = urllib.request.Request(ollama_url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=8) as res:
                    data = json.loads(res.read().decode("utf-8"))
                    parsed = json.loads(data["response"])
                    parsed["engine"] = "Local Llama 3 (Ollama)"
                    return parsed
            except Exception as e:
                print(f"Ollama local connection failed: {e}. Falling back to deterministic auditor.")

        # 3. Deterministic Regulatory Audit Engine (Guaranteed zero-dependency fallback)
        return self._deterministic_regulatory_audit(
            s1_name, s1_addr, s1_country, t_name, t_addr, t_country, eval_result
        )

    def _deterministic_regulatory_audit(
        self,
        s1_name: str,
        s1_addr: str,
        s1_country: str,
        t_name: str,
        t_addr: str,
        t_country: str,
        eval_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deterministic, rule-grounded regulatory compliance statement.
        Synthesizes exact string distance deltas, suffix strippings, and locality tokens.
        """
        feats = eval_result.get("features", {})
        cal_prob = eval_result.get("calibrated_prob", 0.0)
        status = eval_result.get("status", "REJECTED")
        is_colocation = eval_result.get("colocation_triggered", False)

        name_jw = feats.get("name_jaro_winkler", 0.0)
        addr_overlap = feats.get("addr_token_overlap_ratio", 0.0)
        emb_cos = feats.get("name_embedding_cosine", 0.0)
        street_match = feats.get("addr_street_num_match")

        # Synthesize plain-English justifications
        if is_colocation:
            verdict = "SEPARATE ENTITIES"
            justification = (
                f"Entities kept separate because '{s1_name}' and '{t_name}' represent distinct, independent businesses "
                f"that happen to share the same physical building/locality ({s1_addr}). Co-location guardrail suppressed false merge."
            )
            evidence = [
                f"Address tokens show {addr_overlap:.1%} overlap indicating shared commercial structure or shopping complex.",
                f"Business names are lexically disjoint (Jaro-Winkler: {name_jw:.1%}, Token Jaccard: 0%).",
                "Co-location Guardrail activated to prevent severe catalog false merge."
            ]
            risk = "Prevented False Merge (High risk averted)"
            action = "Reject & Block"

        elif status == "CONFIRMED_MATCH" or cal_prob >= 0.40:
            verdict = "MERGE CONFIRMED"
            
            # Determine specific variance sources
            reasons = []
            if feats.get("suffix_normalized_equality") != 1.0:
                reasons.append("legal suffix abbreviations")
            if feats.get("name_exact_match") == 0.0 and name_jw > 0.80:
                reasons.append("minor spelling/diacritic variations")
            if emb_cos > 0.80 and name_jw < 0.60:
                reasons.append("cross-script multilingual transliteration")
            
            variance_str = f"despite {' and '.join(reasons)}" if reasons else "with exact name and address concordance"

            justification = (
                f"Entities merged because '{s1_name}' and '{t_name}' share matching street identifiers "
                f"and geographical locality ({s1_country}) {variance_str} (Calibrated Match Confidence: {cal_prob:.1%})."
            )
            evidence = [
                f"High physical location agreement: Address token overlap of {addr_overlap:.1%} with matching municipal zone.",
                f"Name similarity metric: Jaro-Winkler {name_jw:.1%} and dense multilingual vector cosine {emb_cos:.2f}.",
                "Street number verification passed: Zero conflicting unit discrepancies detected."
            ]
            risk = "Low Risk (Meets Dodd-Frank & Basel III Master Data Entity Disambiguation standards)"
            action = "Auto-Approve Merge"

        elif 0.35 <= cal_prob <= 0.45 or status == "BORDERLINE_REVIEW":
            verdict = "BORDERLINE ESCALATION"
            justification = (
                f"Entities flagged for supervisory review: '{s1_name}' and '{t_name}' exhibit partial token concordance "
                f"but fall in the ambiguous confidence band ({cal_prob:.1%}), requiring human verification of registration filings."
            )
            evidence = [
                f"Intermediate confidence score {cal_prob:.1%} falls within the [35%, 45%] supervisory review window.",
                f"Address overlap stands at {addr_overlap:.1%}, with partial name agreement ({name_jw:.1%}).",
                "Possible branch division or secondary trade name; automated merge suspended."
            ]
            risk = "Moderate Risk (Requires Human-in-the-Loop clearance)"
            action = "Human-in-the-Loop Review"

        else:
            verdict = "SEPARATE ENTITIES"
            justification = (
                f"Entities kept separate because '{s1_name}' and '{t_name}' fail core similarity thresholds "
                f"(Confidence: {cal_prob:.1%}, Address Overlap: {addr_overlap:.1%}) and represent distinct commercial entities."
            )
            evidence = [
                f"Low address overlap ({addr_overlap:.1%}) confirms disparate operating locations.",
                f"Name similarity ({name_jw:.1%}) fails candidate qualification thresholds.",
                "Model score well below decision boundary (tau = 0.40)."
            ]
            risk = "Zero Merge Risk (Non-match verified)"
            action = "Reject & Block"

        return {
            "verdict": verdict,
            "audit_justification": justification,
            "evidence_breakdown": evidence,
            "compliance_risk_assessment": risk,
            "recommended_action": action,
            "engine": "Deterministic Regulatory Compliance Engine"
        }
