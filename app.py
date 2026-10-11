#!/usr/bin/env python3
"""
Interactive Entity Resolution & Record Linkage Dashboard
Streamlit-based system featuring:
1. Live Entity Lookup & Cluster Matching across multiple sources
2. Exact TreeSHAP Match Explanation & Plotly Waterfall Charts
3. Calibrated Confidence Speedometer Gauge (p >= 0.40)
4. Human-in-the-Loop (HITL) Queue with persistent approval/rejection audit trail
"""

import sys
import os
import json
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Ensure module pathing
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard_engine import (
    EntityResolutionEngine,
    get_hitl_records,
    queue_hitl_case,
    update_hitl_decision,
)
from vector_search import VectorDatabaseManager, FAISSVectorStore
from llm_judge import LLMJudge

# Page Configuration
st.set_page_config(
    page_title="EntityResolvers — Multilingual Entity Resolution",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-match {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-borderline {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-rejected {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Cached initialization of model and pipeline engine."""
    return EntityResolutionEngine(artifacts_dir="artifacts")


@st.cache_data
def load_demo_scenarios():
    """Load benchmark scenarios."""
    path = Path("artifacts/demo_data.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_resource
def load_vector_manager():
    """Cached initialization of FAISS and Qdrant vector databases."""
    v_mgr = VectorDatabaseManager(artifacts_dir="artifacts")
    faiss_bin = Path("artifacts/faiss_index.bin")
    faiss_meta = Path("artifacts/faiss_meta.json")
    if faiss_bin.exists() and faiss_meta.exists():
        v_mgr.faiss_store = FAISSVectorStore.load(str(faiss_bin), str(faiss_meta))
    return v_mgr


engine = load_engine()
demo_scenarios = load_demo_scenarios()
vector_mgr = load_vector_manager()


# ── Helper: Gauge Chart ──
def plot_confidence_gauge(prob: float, threshold: float = 0.40):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
        title={"text": "Calibrated Match Probability", "font": {"size": 16, "color": "#334155"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
            "bar": {"color": "#0284c7", "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#cbd5e1",
            "steps": [
                {"range": [0, 35], "color": "#fee2e2"},    # Red
                {"range": [35, 45], "color": "#fef3c7"},   # Yellow / Borderline
                {"range": [45, 100], "color": "#d1fae5"},  # Green
            ],
            "threshold": {
                "line": {"color": "#dc2626", "width": 4},
                "thickness": 0.8,
                "value": threshold * 100,
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif")
    )
    return fig


# ── Helper: TreeSHAP Waterfall Chart ──
def plot_shap_waterfall(base_value: float, sorted_shap: list, top_k: int = 8):
    """Generates an intuitive TreeSHAP waterfall chart using Plotly."""
    selected_shap = sorted_shap[:top_k]
    
    # Calculate cumulative values for waterfall
    features = [item["feature"] for item in selected_shap]
    values = [item["shap_value"] for item in selected_shap]
    
    measure = ["relative"] * len(features)
    x_labels = features + ["Net Logit Output"]
    y_values = values + [0]  # Total will be calculated
    measure.append("total")
    
    # Prettier feature names
    clean_labels = []
    for f in features:
        lbl = f.replace("_", " ").title()
        if len(lbl) > 22:
            lbl = lbl[:20] + "…"
        clean_labels.append(lbl)
    clean_labels.append("Final Model Logit")

    fig = go.Figure(go.Waterfall(
        name="TreeSHAP",
        orientation="v",
        measure=measure,
        x=clean_labels,
        textposition="outside",
        text=[f"{v:+.2f}" if m == "relative" else f"{sum(values) + base_value:.2f}" for v, m in zip(y_values, measure)],
        y=values + [sum(values) + base_value],
        connector={"line": {"color": "#94a3b8"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#0284c7"}}
    ))
    
    fig.update_layout(
        title=f"TreeSHAP Feature Contributions (Base Prior: {base_value:.2f})",
        title_font=dict(size=15, color="#1e293b"),
        showlegend=False,
        height=360,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        yaxis=dict(title="Log-Odds Impact (Logit)", gridcolor="#e2e8f0"),
        xaxis=dict(tickangle=-25)
    )
    return fig


# ── Sidebar Navigation ──
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/network.png", width=64)
    st.title("EntityResolvers")
    st.markdown("**Enterprise Record Linkage Engine**")
    st.caption("Multilingual Candidate Generation & Bipartite Disambiguation")
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "🔍 Live Entity Lookup",
            "📊 Feature Breakdown & TreeSHAP",
            "⚡ Vector Search & DB (FAISS / Qdrant)",
            "🤖 LLM-as-a-Judge (GenAI Auditor)",
            "🛡️ Human-in-the-Loop (HITL) Queue",
            "📑 Model Card & Architecture",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### ⚙️ Production Telemetry")
    st.markdown("- **Model:** LightGBM GBDT (Scale 4.05)")
    st.markdown("- **Calibrator:** Isotonic Regression")
    st.markdown("- **Holdout $F_{0.5}$:** **0.9662**")
    st.markdown("- **Singleton Accuracy:** **95.1%**")
    st.markdown("- **Operating $\\tau$:** **0.40**")
    
    hitl_items = get_hitl_records()
    pending_count = sum(1 for r in hitl_items if r["status"] == "PENDING")
    st.markdown(f"- **HITL Pending:** `{pending_count} cases`")

    st.divider()
    st.caption("Antigravity 2.0 • ML Challenge 2026")


# ══════════════════════════════════════════════════════════════════
# 1. LIVE ENTITY LOOKUP
# ══════════════════════════════════════════════════════════════════
if menu == "🔍 Live Entity Lookup":
    st.markdown('<div class="main-header">Live Entity Lookup & Cluster Disambiguation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Query any business identity fragment across noisy multi-source datasets with automated blocking, candidate scoring, and guardrail enforcement.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Select or Enter Query Entity (Source 1)")
        input_mode = st.radio(
            "Input Mode:",
            ["Preset Benchmark Scenarios", "Custom Free-form Input"],
            horizontal=True
        )

        if input_mode == "Preset Benchmark Scenarios":
            scenario_titles = [s["title"] for s in demo_scenarios]
            selected_title = st.selectbox("Choose a benchmark case:", scenario_titles)
            selected_scenario = next(s for s in demo_scenarios if s["title"] == selected_title)
            
            s1_id = selected_scenario["source1"]["entity_id"]
            s1_name = selected_scenario["source1"]["business_name"]
            s1_addr = selected_scenario["source1"]["business_address"]
            s1_country = selected_scenario["source1"]["country"]
            
            st.info(f"**Scenario Context:** {selected_scenario['description']}")
        else:
            selected_scenario = None
            s1_id = st.text_input("Source 1 Entity ID:", value="S1-CUSTOM-99")
            s1_name = st.text_input("Business Name:", value="Unique & Sons Private Limited")
            s1_addr = st.text_input("Business Address:", value="88 Olive Circle, Lebanon, TN 37087")
            s1_country = st.selectbox("Country:", ["US", "India", "France"])

    with col2:
        st.subheader("Reference Entity Card")
        st.markdown(f"""
        <div style="background:#f1f5f9; padding:18px; border-radius:8px; border-left: 5px solid #0284c7;">
            <p style="margin:0; font-size:0.85rem; color:#64748b;"><strong>ID:</strong> {s1_id} &nbsp;|&nbsp; <strong>Country:</strong> {s1_country}</p>
            <h3 style="margin:4px 0 8px 0; color:#0f172a;">{s1_name}</h3>
            <p style="margin:0; color:#334155;">📍 {s1_addr}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Candidate Pool Sources:")
        st.caption("Matches are resolved against Source 2 (supplier feed) and Source 3 (state registry).")

    st.divider()

    # Targets to evaluate
    if selected_scenario:
        targets_to_evaluate = selected_scenario["candidates"]
    else:
        # Free-form target candidate builder
        st.subheader("2. Target Candidate to Match Against")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            t_id = st.text_input("Target ID:", value="S2-CUSTOM-01")
            t_name = st.text_input("Candidate Business Name:", value="Unique and Sons Pvt Ltd")
        with t_col2:
            t_addr = st.text_input("Candidate Business Address:", value="88 Olive Cir, Lebanon, Tennessee")
            t_country = st.selectbox("Candidate Country:", ["US", "India", "France"], index=["US", "India", "France"].index(s1_country))
        
        targets_to_evaluate = [{
            "entity_id": t_id,
            "source": "Source 2",
            "business_name": t_name,
            "business_address": t_addr,
            "country": t_country,
            "notes": "Custom user entered candidate record"
        }]

    # Run Resolution
    st.subheader(f"3. Resolved Cluster Matches ({len(targets_to_evaluate)} Candidates Evaluated)")
    
    results = []
    for cand in targets_to_evaluate:
        eval_res = engine.evaluate_pair(
            s1_name=s1_name,
            s1_addr=s1_addr,
            s1_country=s1_country,
            t_name=cand["business_name"],
            t_addr=cand["business_address"],
            t_country=cand["country"],
            target_id=cand["entity_id"],
            s1_id=s1_id
        )
        cand_dict = dict(cand)
        cand_dict.update(eval_res)
        results.append(cand_dict)

    # Render Candidate Cards / Table
    for idx, res in enumerate(results):
        prob = res["calibrated_prob"]
        status = res["status"]
        
        with st.container():
            c_header, c_prob, c_badge = st.columns([3, 1.2, 1.5])
            with c_header:
                st.markdown(f"**{res['entity_id']}** ({res.get('source', 'Target')}) — **{res['business_name']}**")
                st.caption(f"📍 {res['business_address']} [{res['country']}]")
                if res.get("notes"):
                    st.caption(f"ℹ️ *{res['notes']}*")
            with c_prob:
                st.metric("Calibrated Prob", f"{prob*100:.1f}%", delta=f"{prob - 0.40:+.1%}")
            with c_badge:
                if status == "CONFIRMED_MATCH":
                    st.markdown('<span class="badge-match">🟢 Confirmed Match</span>', unsafe_allow_html=True)
                elif status == "BORDERLINE_REVIEW":
                    st.markdown('<span class="badge-borderline">🟡 Borderline (HITL)</span>', unsafe_allow_html=True)
                    # Quick button to push to queue
                    if st.button("➕ Queue to HITL", key=f"q_{idx}"):
                        new_rev_id = queue_hitl_case(
                            s1_id=s1_id,
                            target_id=res["entity_id"],
                            s1_name=s1_name,
                            s1_addr=s1_addr,
                            t_name=res["business_name"],
                            t_addr=res["business_address"],
                            country=res["country"],
                            calibrated_prob=prob,
                            reason=res.get("notes", "Borderline score")
                        )
                        st.success(f"Added to Review Queue ({new_rev_id})!")
                else:
                    st.markdown('<span class="badge-rejected">🔴 Non-Match</span>', unsafe_allow_html=True)
            
            # Key feature metrics inline
            feats = res["features"]
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("Name Jaro-Winkler", f"{feats['name_jaro_winkler']*100:.1f}%")
            f2.metric("Address Overlap", f"{feats['addr_token_overlap_ratio']*100:.1f}%")
            f3.metric("Semantic MiniLM Cosine", f"{feats['name_embedding_cosine']:.2f}")
            f4.metric("Street Num Match", "Match" if feats["addr_street_num_match"] == 1.0 else ("Mismatch" if feats["addr_street_num_match"] == 0.0 else "N/A"))
            f5.metric("Legal Suffix Match", "Match" if feats["suffix_normalized_equality"] == 1.0 else "Diff")

            if res["colocation_triggered"]:
                st.warning("⚠️ **Co-location Guardrail Activated:** This candidate shares the exact address/building but represents an unrelated business entity. Suppressed to avoid false merge.")

            st.divider()


# ══════════════════════════════════════════════════════════════════
# 2. FEATURE BREAKDOWN & TreeSHAP WATERFALL
# ══════════════════════════════════════════════════════════════════
elif menu == "📊 Feature Breakdown & TreeSHAP":
    st.markdown('<div class="main-header">Match Explanation & TreeSHAP Attribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explain why two multi-source entity fragments were linked or separated using exact TreeSHAP feature additivity.</div>', unsafe_allow_html=True)

    # Let user select a pair from benchmark or custom
    col_s1, col_t = st.columns(2)
    
    with col_s1:
        st.markdown("#### Source 1 Reference")
        s1_name_in = st.text_input("S1 Name:", value="Dahlia Power Reliable Scientific LLC")
        s1_addr_in = st.text_input("S1 Address:", value="630 45th Terrace, Kansas City, MO")
        s1_country_in = st.selectbox("S1 Country:", ["US", "India", "France"], index=0)

    with col_t:
        st.markdown("#### Candidate Target Record")
        t_name_in = st.text_input("Target Name:", value="Dahlia Power Reliable Scientific")
        t_addr_in = st.text_input("Target Address:", value="45ND TERRACE, null, KANSAS CITY, MO")
        t_country_in = st.selectbox("Target Country:", ["US", "India", "France"], index=0)

    eval_result = engine.evaluate_pair(
        s1_name=s1_name_in,
        s1_addr=s1_addr_in,
        s1_country=s1_country_in,
        t_name=t_name_in,
        t_addr=t_addr_in,
        t_country=t_country_in,
        target_id="S2-DEMO-01",
        s1_id="S1-DEMO-01"
    )

    prob = eval_result["calibrated_prob"]
    raw_prob = eval_result["raw_prob"]
    feats = eval_result["features"]
    sorted_shap = eval_result["sorted_shap"]
    base_val = eval_result["base_value"]

    st.divider()

    # Gauges & High-level Verdict
    g_col, stat_col = st.columns([1.2, 1.8])
    with g_col:
        gauge_fig = plot_confidence_gauge(prob, threshold=0.40)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with stat_col:
        st.markdown("### Match Decision Summary")
        if eval_result["status"] == "CONFIRMED_MATCH":
            st.success(f"### 🟢 Confirmed Entity Match (Confidence: {prob*100:.2f}%)")
            st.markdown(f"The candidate clears the calibrated decision threshold ($\\tau = 0.40$) and satisfies all pair gates (Street Number Gate & Name Lexical Floor).")
        elif eval_result["status"] == "BORDERLINE_REVIEW":
            st.warning(f"### 🟡 Borderline Ambiguity (Confidence: {prob*100:.2f}%)")
            st.markdown(f"The candidate falls within the human-in-the-loop review band ($0.35 \\le p \\le 0.45$). Recommended for manual supervisory review.")
        else:
            st.error(f"### 🔴 Disjoint Entity / Non-Match (Confidence: {prob*100:.2f}%)")
            st.markdown(f"The candidate is rejected by the decision layer. Score is below the confidence boundary.")

        # Quick summary stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw LightGBM Prob", f"{raw_prob*100:.1f}%")
        c2.metric("Isotonic Calibrated Prob", f"{prob*100:.1f}%")
        c3.metric("Pair Gates Check", "PASSED" if eval_result["passed_gates"] else "FAILED")

    st.divider()

    # TreeSHAP Waterfall Chart
    st.subheader("TreeSHAP Feature Attribution Waterfall")
    st.caption("Shows how each pair-wise feature pushes the classification log-odds UP (green: evidence of true match) or pulls it DOWN (red: divergence penalty) from the base prior.")
    
    top_k = st.slider("Number of top features in waterfall:", min_value=5, max_value=15, value=8)
    waterfall_fig = plot_shap_waterfall(base_val, sorted_shap, top_k=top_k)
    st.plotly_chart(waterfall_fig, use_container_width=True)

    # Detailed 27 Features Grid
    with st.expander("🔍 Deep Dive: Complete 27-Feature Breakdown"):
        feature_df = pd.DataFrame([
            {
                "Feature Name": item["feature"],
                "Feature Value": f"{item['feature_value']:.4f}" if isinstance(item['feature_value'], float) else item['feature_value'],
                "TreeSHAP Impact (Logit)": f"{item['shap_value']:+.4f}",
                "Influence": "Positive Evidence" if item['shap_value'] > 0 else "Penalty / Dissimilar"
            }
            for item in sorted_shap
        ])
        st.dataframe(feature_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 3. VECTOR SEARCH & DATABASE INTEGRATION (FAISS / QDRANT / PGVECTOR)
# ══════════════════════════════════════════════════════════════════
elif menu == "⚡ Vector Search & DB (FAISS / Qdrant)":
    st.markdown('<div class="main-header">Vector Search & Modern Vector Database Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scalable Dense Semantic Candidate Retrieval using Meta AI FAISS (HNSW), Qdrant (Embedded Vector DB), and PostgreSQL pgvector Blueprints.</div>', unsafe_allow_html=True)

    vec_tab1, vec_tab2, vec_tab3 = st.tabs([
        "🔎 Live Semantic Vector Search",
        "⚖️ Architectural Comparison (Lexical vs FAISS vs Qdrant vs pgvector)",
        "🐘 PostgreSQL + pgvector Production Blueprint"
    ])

    with vec_tab1:
        st.subheader("Dense Embedding Retrieval Benchmark")
        st.caption("Search across pre-indexed multi-source company records using 384-dimensional multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).")

        q_col1, q_col2 = st.columns([2.5, 1.5])
        with q_col1:
            query_input = st.text_input(
                "Enter Company Name or Semantic Query:",
                value="Dahlia Scientific Instruments Kansas City",
                help="Type any company name or keyword to retrieve nearest dense vector neighbors."
            )
            # Quick suggestion buttons
            st.markdown("<small><strong>Quick Examples:</strong></small>", unsafe_allow_html=True)
            ex_cols = st.columns(4)
            if ex_cols[0].button("🔬 Dahlia (US)"):
                query_input = "Dahlia Power Reliable Scientific LLC"
            if ex_cols[1].button("🇮🇳 Supreme IT (India)"):
                query_input = "Supreme IT Solutions Private Limited Pune"
            if ex_cols[2].button("🥖 Thermal (France)"):
                query_input = "Thermal & Fils SASU Dunkerque"
            if ex_cols[3].button("🚚 Ganesh Logistics"):
                query_input = "Shree Ganesh Logistics MIDC Mumbai"

        with q_col2:
            backend_choice = st.selectbox(
                "Vector DB Engine:",
                ["Qdrant (Embedded HNSW + Payload Filter)", "FAISS (Meta AI In-Memory HNSW)"]
            )
            country_filter = st.selectbox(
                "Payload Country Filter:",
                ["All", "US", "India", "France"]
            )
            top_k_vec = st.slider("Top Candidates (K):", min_value=1, max_value=15, value=5)

        backend_key = "faiss" if "FAISS" in backend_choice else "qdrant"
        filter_val = None if country_filter == "All" else country_filter

        # Run Vector Search Query
        with st.spinner("Executing Vector Query..."):
            query_result = vector_mgr.query(
                query_text=query_input,
                top_k=top_k_vec,
                country_filter=filter_val,
                backend=backend_key
            )

        st.divider()

        # Telemetry Latency KPI Cards
        lat1, lat2, lat3, lat4 = st.columns(4)
        lat1.metric("Query Vectorization", f"{query_result['encode_latency_ms']:.2f} ms", help="SentenceTransformers MiniLM embedding generation")
        lat2.metric("Index Search Latency", f"{query_result['search_latency_ms']:.3f} ms", help="HNSW graph traversal latency")
        lat3.metric("Total End-to-End Latency", f"{query_result['total_latency_ms']:.2f} ms")
        lat4.metric("Indexed Candidates Pool", "260 Records", help="Sampled multi-source benchmark pool")

        st.markdown(f"#### Top {len(query_result['results'])} Nearest Vector Neighbors:")
        if not query_result["results"]:
            st.info("No candidates matched the specified metadata filter.")
        else:
            for rank, item in enumerate(query_result["results"], 1):
                sim_pct = item["similarity_score"] * 100
                with st.container():
                    r_col1, r_col2, r_col3 = st.columns([3, 1.2, 1.2])
                    with r_col1:
                        st.markdown(f"**#{rank}. {item['business_name']}** (`{item['entity_id']}`)")
                        st.caption(f"📍 {item['business_address']} &nbsp;|&nbsp; 🌍 **{item['country']}** &nbsp;|&nbsp; Source: **{item['source']}**")
                    with r_col2:
                        st.metric("Cosine Similarity", f"{sim_pct:.1f}%")
                    with r_col3:
                        st.markdown(f"<span class='badge-match'>{item['backend']}</span>", unsafe_allow_html=True)
                    st.divider()

    with vec_tab2:
        st.subheader("Architectural Comparison: Candidate Retrieval Engines")
        st.markdown("""
        In production Entity Resolution pipelines, candidate generation must balance **recall**, **query throughput**, and **metadata filtering capability**.
        """)
        
        comparison_data = [
            {
                "Dimension": "Core Retrieval Mechanism",
                "Inverted Index (Stage 2)": "Token Inverted Index + IDF weights",
                "FAISS (Meta AI)": "Dense Vector HNSW / IVF graph",
                "Qdrant (Vector DB)": "Vector HNSW + Native Payload Filter",
                "PostgreSQL pgvector": "Relational Tables + HNSW Vector Ops"
            },
            {
                "Dimension": "Primary Strength",
                "Inverted Index (Stage 2)": "Exact lexical & typo matching",
                "FAISS (Meta AI)": "Ultra-fast raw C++ similarity search",
                "Qdrant (Vector DB)": "Integrated metadata filtering (country/source)",
                "PostgreSQL pgvector": "ACID compliance & unified SQL joins"
            },
            {
                "Dimension": "Query Latency (1M Vectors)",
                "Inverted Index (Stage 2)": "~1 - 3 ms (Token lookup)",
                "FAISS (Meta AI)": "< 1 ms (Sub-millisecond)",
                "Qdrant (Vector DB)": "~1.5 - 4 ms (Embedded)",
                "PostgreSQL pgvector": "~3 - 8 ms (SQL overhead)"
            },
            {
                "Dimension": "Metadata Filtering",
                "Inverted Index (Stage 2)": "Manual partition by country",
                "FAISS (Meta AI)": "Post-filtering or ID mapping",
                "Qdrant (Vector DB)": "Native pre/single-stage payload filter",
                "PostgreSQL pgvector": "Full SQL WHERE clause & JOINs"
            },
            {
                "Dimension": "Multilingual Cross-Script",
                "Inverted Index (Stage 2)": "Requires explicit transliteration",
                "FAISS (Meta AI)": "Native semantic cross-script alignment",
                "Qdrant (Vector DB)": "Native semantic cross-script alignment",
                "PostgreSQL pgvector": "Native semantic cross-script alignment"
            },
            {
                "Dimension": "Deployment Mode",
                "Inverted Index (Stage 2)": "In-Memory Python dict",
                "FAISS (Meta AI)": "In-Memory C++ binary",
                "Qdrant (Vector DB)": "Embedded or Distributed Container",
                "PostgreSQL pgvector": "Managed Cloud Database (AWS/GCP)"
            }
        ]
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        st.info(r"💡 **Enterprise Hybrid Recommendation:** Combine the **Inverted Index** (for high-precision acronym and postal matches) with **Qdrant / FAISS** (for multilingual cross-script and synonym retrieval) to reach $98\%+$ recall before LightGBM classification.")

    with vec_tab3:
        st.subheader("Production PostgreSQL + pgvector Deployment Blueprint")
        st.markdown("For enterprise stacks requiring unified relational ACID transactions with vector search, use the following production-hardened SQL schema:")

        sql_code = VectorDatabaseManager.get_pgvector_sql_blueprint()
        st.code(sql_code, language="sql")

        st.markdown("#### Python Async Query Implementation (`asyncpg` / `psycopg3`):")
        st.code("""
import psycopg
from pgvector.psycopg import register_vector
import numpy as np

# Connect to cloud database
conn = psycopg.connect("postgresql://postgres:password@localhost:5432/entity_resolution")
register_vector(conn)

query_embedding = embedder.encode("Dahlia Power Scientific", normalize_embeddings=True)

with conn.cursor() as cur:
    cur.execute(\"\"\"
        SELECT entity_id, business_name, country, 1 - (embedding <=> %s) AS score
        FROM company_entities
        WHERE country = 'US'
        ORDER BY embedding <=> %s
        LIMIT 5;
    \"\"\", (query_embedding, query_embedding))
    
    candidates = cur.fetchall()
    print("Retrieved pgvector candidates:", candidates)
        """, language="python")


# ══════════════════════════════════════════════════════════════════
# 4. LLM-AS-A-JUDGE / EXPLANATION LAYER (GENAI AUDITOR)
# ══════════════════════════════════════════════════════════════════
elif menu == "🤖 LLM-as-a-Judge (GenAI Auditor)":
    st.markdown('<div class="main-header">LLM-as-a-Judge & Regulatory Audit Explanation Layer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated generation of human-readable, compliance-ready entity match justifications for KYC, AML, and master data regulatory audits.</div>', unsafe_allow_html=True)

    # Top Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.8, 1.2])
    with ctrl_col1:
        judge_engine = st.selectbox(
            "Select LLM Judge Engine:",
            [
                "Google Gemini 3.8 Flash (google-genai)",
                "Meta Llama 3 (Local Ollama)",
                "Deterministic Regulatory Compliance Engine"
            ],
            index=0
        )
    with ctrl_col2:
        api_key_input = st.text_input(
            "Gemini API Key (Optional):",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Leave empty to use automatic deterministic auditor or environment variable GEMINI_API_KEY."
        )
    with ctrl_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if api_key_input:
            st.success("API Key Active")
        else:
            st.info("Offline Fallback Ready")

    st.divider()

    # Preset Regulatory Benchmark Cases
    st.subheader("1. Select Regulatory Audit Case or Custom Input")
    audit_presets = {
        "Clairvoyant Record (Diacritic & Legal Suffix)": {
            "s1_name": "Clairvoyant Récord Pvt Ltd",
            "s1_addr": "88 Olive Circle, Lebanon, TN",
            "s1_country": "US",
            "t_name": "Clairvoyant Record",
            "t_addr": "88 Olive Cir, Lebanon, TN",
            "t_country": "US",
            "desc": "Legal suffix 'Pvt Ltd' dropped, diacritic 'é' normalized to 'e', matching street number 88."
        },
        "Dahlia Power (Address Component Reordering)": {
            "s1_name": "Dahlia Power Reliable Scientific LLC",
            "s1_addr": "630 45th Terrace, Kansas City, MO",
            "s1_country": "US",
            "t_name": "Dahlia Power Reliable Scientific",
            "t_addr": "45ND TERRACE, null, KANSAS CITY, MO",
            "t_country": "US",
            "desc": "Word-order swap in address, suffix 'LLC' stripped, uppercase formatting."
        },
        "Co-Location Mall Trap (Unrelated Stores Sharing Address)": {
            "s1_name": "Apex Diagnostic & Pathology Center",
            "s1_addr": "Tower B, Level 3, Cyber Gateway, Hitech City, Hyderabad, Telangana",
            "s1_country": "India",
            "t_name": "Blue Star Cloud Consultancy Services",
            "t_addr": "Tower B, Level 3, Cyber Gateway, Hitech City, Hyderabad, Telangana",
            "t_country": "India",
            "desc": "Identical commercial address building but completely disjoint business domains. Must be rejected."
        },
        "Cross-Script Indic Transliteration (English ↔ Devanagari)": {
            "s1_name": "Supreme IT Solutions Private Limited",
            "s1_addr": "Shivaji Nagar, Pune, Maharashtra 411005",
            "s1_country": "India",
            "t_name": "सुप्रीम आईटी सॉल्यूशंस प्राइवेट लिमिटेड",
            "t_addr": "शिवाजी नगर, पुणे, महाराष्ट्र 411005",
            "t_country": "India",
            "desc": "Exact transliteration from English Latin script to Hindi Devanagari script."
        },
        "Cascade Mountain (Borderline Ambiguity Case)": {
            "s1_name": "Cascade Mountain Outdoor Gear & Apparel LLC",
            "s1_addr": "1420 NW Gilman Blvd, Issaquah, WA",
            "s1_country": "US",
            "t_name": "Cascade Outfitters",
            "t_addr": "Gilman Blvd, Issaquah, Washington",
            "t_country": "US",
            "desc": "Divergent brand name with partial address omission. Triggers human supervisory escalation."
        }
    }

    selected_case_name = st.selectbox("Choose Audit Scenario:", list(audit_presets.keys()))
    case_info = audit_presets[selected_case_name]

    c_s1, c_t = st.columns(2)
    with c_s1:
        st.markdown("**Source 1 (Reference Entity):**")
        s1_n = st.text_input("Name:", value=case_info["s1_name"], key="aud_s1_name")
        s1_a = st.text_input("Address:", value=case_info["s1_addr"], key="aud_s1_addr")
        s1_c = st.text_input("Country:", value=case_info["s1_country"], key="aud_s1_c")

    with c_t:
        st.markdown("**Candidate Target Record:**")
        t_n = st.text_input("Name:", value=case_info["t_name"], key="aud_t_name")
        t_a = st.text_input("Address:", value=case_info["t_addr"], key="aud_t_addr")
        t_c = st.text_input("Country:", value=case_info["t_country"], key="aud_t_c")

    st.caption(f"ℹ️ **Scenario Background:** {case_info['desc']}")

    st.divider()

    # Trigger Audit Button
    if st.button("⚖️ Generate Regulatory Audit Justification", type="primary", use_container_width=True):
        with st.spinner("Evaluating pair telemetry & generating regulatory justification..."):
            # 1. Pipeline ML Evaluation
            eval_res = engine.evaluate_pair(
                s1_name=s1_n,
                s1_addr=s1_a,
                s1_country=s1_c,
                t_name=t_n,
                t_addr=t_a,
                t_country=t_c,
                target_id="S2-AUDIT-01",
                s1_id="S1-AUDIT-01"
            )

            # 2. LLM Judge Generation
            judge = LLMJudge(api_key=api_key_input)
            backend_param = "gemini" if "Gemini" in judge_engine else ("llama3" if "Llama" in judge_engine else "deterministic")
            
            audit_result = judge.generate_audit_explanation(
                s1_name=s1_n,
                s1_addr=s1_a,
                s1_country=s1_c,
                t_name=t_n,
                t_addr=t_a,
                t_country=t_c,
                eval_result=eval_res,
                backend=backend_param
            )

            # Store in session state for display
            st.session_state["current_audit"] = audit_result
            st.session_state["current_eval"] = eval_res

    # Render Audit Memorandum if available
    if "current_audit" in st.session_state:
        audit = st.session_state["current_audit"]
        eval_data = st.session_state["current_eval"]
        verdict = audit.get("verdict", "UNKNOWN")

        st.subheader("2. Official Regulatory Audit Memorandum")

        # Verdict Callout Banner
        if "MERGE CONFIRMED" in verdict:
            st.success(f"### 🏛️ OFFICIAL VERDICT: {verdict}")
        elif "BORDERLINE" in verdict:
            st.warning(f"### ⚠️ OFFICIAL VERDICT: {verdict}")
        else:
            st.error(f"### 🚫 OFFICIAL VERDICT: {verdict}")

        # Highlighted Executive Quote Box
        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 6px solid #0284c7; padding: 18px 24px; border-radius: 8px; margin: 15px 0; font-size: 1.15rem; color: #0f172a; line-height: 1.6; font-style: italic; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            "{audit.get('audit_justification', '')}"
        </div>
        """, unsafe_allow_html=True)

        # Supporting Evidentiary Findings
        ev_col, risk_col = st.columns([1.8, 1.2])
        with ev_col:
            st.markdown("#### 📋 Evidentiary Findings:")
            for pt in audit.get("evidence_breakdown", []):
                st.markdown(f"- {pt}")

            st.caption(f"**Audit Provenance:** `{audit.get('engine', 'Compliance Engine')}`")

        with risk_col:
            st.markdown("#### 🛡️ Compliance & Governance:")
            st.markdown(f"**Risk Level:** `{audit.get('compliance_risk_assessment', 'Low')}`")
            st.markdown(f"**Recommended Action:** **{audit.get('recommended_action', 'Review')}**")
            st.metric("Calibrated Match Confidence", f"{eval_data['calibrated_prob']*100:.2f}%")

        # Download Report
        report_md = f"""# Regulatory Entity Resolution Audit Memorandum
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Auditor: Antigravity Autonomous Entity Judge ({audit.get('engine', 'Compliance Engine')})

## Entity Comparison
- Source 1: {s1_n} | {s1_a} [{s1_c}]
- Target:   {t_n} | {t_a} [{t_c}]

## Verdict: {verdict}
Recommended Action: {audit.get('recommended_action')}

## Justification
"{audit.get('audit_justification')}"

## Detailed Evidence
{chr(10).join('- ' + pt for pt in audit.get('evidence_breakdown', []))}

## Risk Assessment
{audit.get('compliance_risk_assessment')}
"""
        st.download_button(
            label="📥 Download Audit Memorandum (Markdown)",
            data=report_md,
            file_name="entity_audit_memorandum.md",
            mime="text/markdown"
        )


# ══════════════════════════════════════════════════════════════════
# 5. HUMAN-IN-THE-LOOP (HITL) QUEUE
# ══════════════════════════════════════════════════════════════════
elif menu == "🛡️ Human-in-the-Loop (HITL) Queue":
    st.markdown('<div class="main-header">Human-in-the-Loop (HITL) Review Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supervisory audit console for borderline identity pairs (0.35 ≤ p ≤ 0.45). Enables manual verification, override decisions, and audit trail compliance.</div>', unsafe_allow_html=True)

    records = get_hitl_records()

    # Top KPI Metrics
    total_cases = len(records)
    pending_cases = sum(1 for r in records if r["status"] == "PENDING")
    approved_cases = sum(1 for r in records if r["status"] == "APPROVED")
    rejected_cases = sum(1 for r in records if r["status"] == "REJECTED")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Queued", total_cases)
    m2.metric("Pending Review", pending_cases, delta=f"{pending_cases} active" if pending_cases else "Clear", delta_color="inverse")
    m3.metric("Approved Matches", approved_cases)
    m4.metric("Rejected Matches", rejected_cases)

    st.divider()

    tab_pending, tab_history = st.tabs(["📋 Pending Review Queue", "📜 Historical Audit Trail"])

    with tab_pending:
        pending_list = [r for r in records if r["status"] == "PENDING"]
        if not pending_list:
            st.success("🎉 No pending review items! All borderline entity cases have been evaluated.")
        else:
            for item in pending_list:
                with st.expander(f"Review Item: **{item['review_id']}** — {item['s1_name']} ↔ {item['t_name']} (Prob: {item['calibrated_prob']*100:.1f}%)", expanded=True):
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown("**Source 1 Entity (Reference):**")
                        st.info(f"**{item['s1_name']}**\n\n📍 {item['s1_addr']}\n\nID: `{item['s1_id']}` | Country: `{item['country']}`")
                    with c_right:
                        st.markdown("**Target Candidate Record:**")
                        st.warning(f"**{item['t_name']}**\n\n📍 {item['t_addr']}\n\nID: `{item['target_id']}` | Country: `{item['country']}`")

                    st.markdown(f"**Flag Reason:** *{item['reason']}* &nbsp;|&nbsp; **Queued At:** `{item['timestamp']}`")

                    # Auditor Notes and Decision Buttons
                    notes = st.text_area(f"Auditor Justification / Notes ({item['review_id']}):", placeholder="Enter notes justifying decision (e.g. verified through company registration number)...", key=f"note_{item['review_id']}")
                    
                    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
                    with btn_col1:
                        if st.button("✅ Approve Match", key=f"app_{item['review_id']}", type="primary"):
                            update_hitl_decision(item['review_id'], "APPROVED", notes)
                            st.success(f"Case {item['review_id']} APPROVED!")
                            st.rerun()
                    with btn_col2:
                        if st.button("❌ Reject Match", key=f"rej_{item['review_id']}"):
                            update_hitl_decision(item['review_id'], "REJECTED", notes)
                            st.info(f"Case {item['review_id']} REJECTED!")
                            st.rerun()

    with tab_history:
        st.subheader("Audit Trail & Decisions Log")
        hist_df = pd.DataFrame(records)
        if not hist_df.empty:
            display_cols = ["review_id", "timestamp", "s1_name", "t_name", "calibrated_prob", "status", "reviewer_notes", "reviewed_at"]
            st.dataframe(hist_df[display_cols], use_container_width=True)

            csv_data = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Audit Trail CSV",
                data=csv_data,
                file_name="hitl_audit_trail.csv",
                mime="text/csv",
            )
        else:
            st.write("No audit history available.")


# ══════════════════════════════════════════════════════════════════
# 4. MODEL CARD & SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
elif menu == "📑 Model Card & Architecture":
    st.markdown('<div class="main-header">System Architecture & Model Card</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive architectural blueprint, data lineage, and academic validation report for final year defense and presentations.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏗️ End-to-End Pipeline Pipeline Stages

    ```
    ┌─────────────────────────┐
    │ Raw Multi-Source Data   │  Source 1 (Reference) + Source 2 (Supplier) + Source 3 (Registry)
    └───────────┬─────────────┘  ~12.5M train records / ~11.7M test records across US, India, France
                ▼
    ┌─────────────────────────┐
    │ Linguistic Normalizer   │  Linguistic cleaning, legal suffix extraction (LLC, Pvt Ltd, SARL, SAS),
    └───────────┬─────────────┘  Devanagari script support, address canonicalization
                ▼
    ┌─────────────────────────┐
    │ 3-Strategy Blocking     │  1. IDF-weighted Token Inverted Index
    │ Candidate Generation    │  2. Metaphone / Soundex Phonetic & Prefix Hashing
    └───────────┬─────────────┘  3. Locality & Street Number Indexing  ==> 99.999% Reduction Ratio!
                ▼
    ┌─────────────────────────┐
    │ Pair-wise Features      │  27 Discriminative Features: String edit distances (Jaro-Winkler, Levenshtein),
    └───────────┬─────────────┘  Multilingual MiniLM Transformer Embeddings, Suffix Equality, Street Checks
                ▼
    ┌─────────────────────────┐
    │ LightGBM Classifier     │  Gradient Boosted Decision Trees trained with scale_pos_weight = 4.0506
    └───────────┬─────────────┘  ROC-AUC = 0.9993, PR-AUC = 0.9971
                ▼
    ┌─────────────────────────┐
    │ Isotonic Calibration    │  Maps raw logits to well-calibrated posterior probabilities P(match|x)
    └───────────┬─────────────┘  Brier Score = 0.0077
                ▼
    ┌─────────────────────────┐
    │ Decision & Guardrails   │  - Co-location Guardrail (blocks co-located independent stores)
    │ Layer (Optimal τ = 0.40)│  - Global 1:1 Bipartite Uniqueness Constraint
    └───────────┬─────────────┘  - Street Number Gate & Name Similarity Floor
                ▼
    ┌─────────────────────────┐
    │ Human-in-the-Loop Queue │  Borderline cases [0.35, 0.45] routed to governance review dashboard
    └─────────────────────────┘
    ```
    """)

    st.divider()

    st.subheader("Key Academic Benchmarks")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Local Holdout Macro F0.5", "0.9662", "+11.6% above baseline")
    b2.metric("Singleton Accuracy", "95.10%", "Prevents false merges")
    b3.metric("Reduction Ratio", "99.9994%", "Top-60 Candidate Cap")
    b4.metric("Calibrated Brier Score", "0.0077", "Threshold < 0.10 DoD")

    st.divider()

    st.subheader("Defense Presentation Talking Points")
    st.markdown(r"""
    - **Combinatorial Challenge:** Without blocking, linking 2.2M Source 1 entities against 10.3M Source 2/3 records requires $>2.2 \times 10^{13}$ pairwise comparisons ($O(N \cdot M)$). Our inverted indexing with token-IDF weights reduces this by **99.999%** while preserving **96%+ candidate recall**.
    - **Multilingual Generalization:** Evaluated zero-shot on unseen country data (`France`) with French legal forms (`SARL`, `SASU`) and cross-script Indian languages (`English` vs `Devanagari`) powered by `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
    - **Explainability:** Fully compliant with financial & enterprise regulatory auditability using exact TreeSHAP attributions rather than black-box embeddings.
    """)
