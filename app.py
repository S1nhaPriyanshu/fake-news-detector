import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np

# Import predictor and scraper
from src.predictor import FakeNewsClassifier
from src.url_scraper import extract_article_from_url

# Page Configuration
st.set_page_config(
    page_title="VERITAS // News Verification Engine",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CARI-Inspired Cyber-Editorial & Neo-Precision UI Stylesheet
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700;800&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0A0D12;
        color: #E2E8F0;
    }

    /* Hide Streamlit default chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display: none;}

    /* Container constraints */
    .block-container {
        max-width: 860px;
        padding-top: 3rem;
        padding-bottom: 3.5rem;
    }

    /* Brand Header */
    .brand-container {
        text-align: center;
        margin-bottom: 2.5rem;
        padding-bottom: 1.8rem;
        border-bottom: 1px solid #1E2638;
    }
    .brand-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 5px 14px;
        border-radius: 999px;
        display: inline-block;
        margin-bottom: 0.9rem;
    }
    .brand-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.1;
        background: linear-gradient(180deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-top: 0.7rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }

    /* URL Input Box */
    .stTextInput > div > div > input {
        background-color: #0F1623;
        border: 1px solid #233149;
        border-radius: 12px;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        padding: 16px 20px;
        transition: all 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38BDF8;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25);
        background-color: #131C2D;
    }

    /* Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        border: 1px solid #38BDF8;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
        border-radius: 10px;
        padding: 14px 28px;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.45);
        transform: translateY(-1px);
    }

    /* Verdict Card Real */
    .verdict-box-real {
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.02) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.08);
        border-radius: 14px;
        padding: 26px;
        margin-top: 26px;
        text-align: center;
    }
    .verdict-header-real {
        font-family: 'JetBrains Mono', monospace;
        color: #10B981;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    /* Verdict Card Fake */
    .verdict-box-fake {
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.09) 0%, rgba(239, 68, 68, 0.02) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.1);
        border-radius: 14px;
        padding: 26px;
        margin-top: 26px;
        text-align: center;
    }
    .verdict-header-fake {
        font-family: 'JetBrains Mono', monospace;
        color: #EF4444;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    /* Verdict Card Satire */
    .verdict-box-satire {
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.09) 0%, rgba(245, 158, 11, 0.02) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.1);
        border-radius: 14px;
        padding: 26px;
        margin-top: 26px;
        text-align: center;
    }
    .verdict-header-satire {
        font-family: 'JetBrains Mono', monospace;
        color: #F59E0B;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    /* Telemetry HUD Cards */
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 20px;
    }
    .telemetry-card {
        background-color: #0E1420;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .telemetry-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 6px;
    }
    .telemetry-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F1F5F9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Article Dossier */
    .dossier-box {
        background-color: #0C1019;
        border: 1px solid #1A2333;
        border-radius: 12px;
        padding: 22px;
        margin-top: 22px;
    }
    .dossier-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94A3B8;
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid #1E293B;
    }
    .dossier-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.35;
        margin-bottom: 10px;
    }
    .dossier-snippet {
        font-size: 0.94rem;
        color: #94A3B8;
        line-height: 1.65;
        margin: 0;
    }

    /* Footer */
    .system-footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 1.8rem;
        border-top: 1px solid #141C2B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #475569;
        letter-spacing: 0.1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine():
    """Load the high-accuracy trained model engine."""
    return FakeNewsClassifier()


engine = load_engine()

# Brand Header
st.markdown(
    """
    <div class="brand-container">
        <div class="brand-tag">⚡ VERITAS // INTELLIGENCE SYSTEM</div>
        <h1 class="brand-title">Fake News Detection Engine</h1>
        <div class="brand-subtitle">Paste any digital news URL to autonomously scrape, analyze, and verify its authenticity.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Clean, Single-Purpose URL Input Field
target_url = st.text_input(
    label="Target URL",
    placeholder="Paste news article link here (e.g. https://www.bbc.com/news/articles/...)",
    label_visibility="collapsed",
)

col_run, col_reset = st.columns([3, 1])
with col_run:
    execute_scan = st.button("RUN VERIFICATION SCAN ⚡", use_container_width=True)
with col_reset:
    clear_clicked = st.button("CLEAR", use_container_width=True)

if clear_clicked:
    st.session_state["last_processed"] = ""
    st.rerun()

# Verification Execution
if execute_scan or (target_url and target_url.strip() and target_url != st.session_state.get("last_processed")):
    st.session_state["last_processed"] = target_url.strip()

    if not target_url.strip():
        st.warning("Please provide a valid news link to analyze.")
    else:
        with st.spinner("Scraping webpage, verifying publisher authority, and computing linguistic vectors..."):
            result = engine.predict_from_url(target_url.strip(), explain=True)

        label = result.get("label", "UNKNOWN")
        conf = result.get("confidence", 0.5)
        conf_band = result.get("confidence_band", "MODERATE")
        conf_source = result.get("confidence_source", "Analysis Engine")
        probs = result.get("probabilities", {"REAL": 0.5, "FAKE": 0.5})
        domain = result.get("domain", "Unknown Domain")
        cred = result.get("credibility", {})
        title = result.get("article_title", "Headline Extracted")
        snippet = result.get("article_content_snippet", "")
        word_count = result.get("word_count", 0)
        sens_score = result.get("sensationalism_score", 0.0)
        sens_keywords = result.get("sensational_keywords", [])
        top_features = result.get("top_features", [])
        is_cached = result.get("cached", False)
        lime_data = result.get("explanation_features")

        # Cache Indicator
        if is_cached:
            st.markdown(
                '<div style="text-align: right; font-family: \'JetBrains Mono\', monospace; font-size: 0.75rem; color: #38BDF8; margin-top: 8px;">⚡ INSTANT CACHE HIT (SERVED FROM LOCAL MEMORY)</div>',
                unsafe_allow_html=True,
            )

        # Render Core Verdict Banner
        if "REAL" in label:
            st.markdown(
                f"""
                <div class="verdict-box-real">
                    <div class="verdict-header-real">✓ VERIFIED AUTHENTIC NEWS</div>
                    <div style="color: #6EE7B7; font-size: 1.02rem; margin-top: 6px;">
                        Article content correlates with validated wire journalism patterns and publisher credibility standards.
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; color: #10B981; font-weight: 700; margin-top: 10px;">
                        CONFIDENCE: {conf * 100:.1f}% &nbsp;
                        <span style="font-size: 0.8rem; padding: 3px 8px; border-radius: 4px; background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981;">{conf_band} CALIBRATION</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                        BASIS: {conf_source}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif "Satire" in label:
            st.markdown(
                f"""
                <div class="verdict-box-satire">
                    <div class="verdict-header-satire">⚠ SATIRE / PARODY PUBLICATION</div>
                    <div style="color: #FCD34D; font-size: 1.02rem; margin-top: 6px;">
                        Published by a documented satire outlet. Content is humorous or hyperbolic, not factual reportage.
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; color: #F59E0B; font-weight: 700; margin-top: 10px;">
                        SATIRE PROBABILITY: {conf * 100:.1f}% &nbsp;
                        <span style="font-size: 0.8rem; padding: 3px 8px; border-radius: 4px; background: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B;">{conf_band} CALIBRATION</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                        BASIS: {conf_source}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-box-fake">
                    <div class="verdict-header-fake">⚠ FLAGGED MISINFORMATION / HOAX</div>
                    <div style="color: #FCA5A5; font-size: 1.02rem; margin-top: 6px;">
                        High density of unverified linguistic markers, sensationalist cues, or domain irregularities.
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; color: #EF4444; font-weight: 700; margin-top: 10px;">
                        MISINFORMATION RISK: {conf * 100:.1f}% &nbsp;
                        <span style="font-size: 0.8rem; padding: 3px 8px; border-radius: 4px; background: rgba(239, 68, 68, 0.2); border: 1px solid #EF4444;">{conf_band} CALIBRATION</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                        BASIS: {conf_source}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Telemetry Metrics HUD
        st.markdown(
            f"""
            <div class="telemetry-grid">
                <div class="telemetry-card">
                    <div class="telemetry-title">Authenticity Index</div>
                    <div class="telemetry-val" style="color: {'#10B981' if probs['REAL'] > 0.5 else '#64748B'};">
                        {probs['REAL'] * 100:.1f}%
                    </div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-title">Misinformation Risk</div>
                    <div class="telemetry-val" style="color: {'#EF4444' if probs['FAKE'] > 0.5 else '#64748B'};">
                        {probs['FAKE'] * 100:.1f}%
                    </div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-title">Sensationalism Score</div>
                    <div class="telemetry-val" style="color: {'#F59E0B' if sens_score > 0.35 else '#38BDF8'};">
                        {sens_score * 100:.0f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Scraped Article Dossier
        st.markdown(
            f"""
            <div class="dossier-box">
                <div class="dossier-meta">
                    <span>SOURCE // {domain.upper()}</span>
                    <span>CREDIBILITY // {cred.get('label', 'UNVERIFIED').upper()}</span>
                    <span>WORDS // {word_count}</span>
                </div>
                <div class="dossier-title">{title}</div>
                <p class="dossier-snippet">{snippet or "Body paragraphs extracted and tokenized."}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if sens_keywords:
            st.warning(f"Sensationalist buzzwords detected in copy: {', '.join(sens_keywords)}")

        # LIME Explainability Visualization
        if lime_data and lime_data.get("available") and lime_data.get("weights"):
            with st.expander("LIME EXPLAINABILITY // WORD-LEVEL INFLUENCE BREAKDOWN", expanded=False):
                st.markdown(
                    "<p style='color: #94A3B8; font-size: 0.85rem;'>Local Interpretable Model-agnostic Explanations (LIME) isolates individual words that push the model toward FAKE (red) or REAL (green).</p>",
                    unsafe_allow_html=True,
                )
                lime_rows = []
                for w_item in lime_data.get("weights", []):
                    w = w_item["word"]
                    score = w_item["weight"]
                    direction = "PUSHES FAKE (RISK)" if score > 0 else "PUSHES REAL (AUTHENTIC)"
                    lime_rows.append({"Word Token": w, "Direction": direction, "Weight": abs(score)})
                lime_df = pd.DataFrame(lime_rows)
                st.dataframe(lime_df, use_container_width=True, hide_index=True)

        # Salient Lexical Features
        if top_features:
            with st.expander("LEXICAL TELEMETRY // TOP TF-IDF FEATURES"):
                f_df = pd.DataFrame(top_features, columns=["Stemmed Feature", "TF-IDF Weight"])
                st.dataframe(f_df, use_container_width=True, hide_index=True)

# System Status Footer
st.markdown(
    """
    <div class="system-footer">
        ENGINE: KAGGLE CLOUD CALIBRATED LINEARSVC (99.30% ACCURACY) // 10,000 N-GRAM TF-IDF // REAL-TIME SCRAPER
    </div>
    """,
    unsafe_allow_html=True,
)
