"""
app/pages/4_explainability.py
=============================
SHAP word-level explainability page.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
from spam_detector.config import cfg
from spam_detector.db import init_db
from app.components.ui_theme import apply_theme

st.set_page_config(page_title="Explainability — SpamShield", page_icon="💡", layout="wide")
apply_theme()
init_db()

from app.components.auth_ui import require_auth
current_user = require_auth()


st.title("💡 SHAP Explainability")
st.markdown(
    """
    Understand **why** the model classified an email as SPAM or HAM.  
    SHAP (SHapley Additive exPlanations) assigns each word an importance score:
    - 🔴 **Red tokens** → pushed the prediction toward **SPAM**
    - 🟢 **Green tokens** → pushed the prediction toward **HAM**
    """
)

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    from spam_detector.predict import Predictor
    return Predictor()

model_dir = cfg.model_dir
if not model_dir.exists():
    st.error("Model not found. Please train first.")
    st.stop()

predictor = load_predictor()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
email_text = st.text_area(
    "📧 Email to explain",
    placeholder="Paste the email here...",
    height=200,
)

max_evals = st.slider(
    "SHAP max evaluations (higher = more accurate, slower)",
    min_value=100, max_value=1000, value=300, step=50,
)

if st.button("💡 Explain Prediction", type="primary", disabled=not email_text.strip()):
    with st.spinner("Running SHAP analysis... (this may take 10–30s on CPU)"):
        # First run normal prediction for the verdict
        result = predictor.predict(email_text)

        # Then run SHAP
        from spam_detector.explain import explain_prediction, shap_to_html
        token_importance = explain_prediction(
            text=email_text,
            predictor_pipeline=predictor._pipeline,
            max_evals=max_evals,
        )

    st.divider()

    # Prediction result
    if result.is_spam:
        st.error(f"🚨 **SPAM** — Confidence: {result.confidence:.1%}")
    else:
        st.success(f"✅ **HAM** — Confidence: {result.confidence:.1%}")

    st.markdown("### Token Importance Heatmap")
    html = shap_to_html(token_importance)
    st.markdown(html, unsafe_allow_html=True)

    # Bar chart of top tokens
    st.markdown("### Top Influential Tokens")
    tokens = [t for t, _ in token_importance]
    values = [v for _, v in token_importance]
    colors = ["#EF4444" if v > 0 else "#22C55E" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=tokens,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Values (positive → SPAM, negative → HAM)",
        xaxis_title="SHAP Value",
        yaxis_title="Token",
        height=max(300, len(tokens) * 35),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "ℹ️ SHAP values are approximate (Partition SHAP algorithm). "
        "They show relative token importance, not absolute probabilities."
    )
