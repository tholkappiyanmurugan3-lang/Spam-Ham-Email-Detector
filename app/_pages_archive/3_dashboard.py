"""
app/pages/3_dashboard.py
========================
Model performance dashboard and prediction history.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd
from spam_detector.config import cfg
from spam_detector.db import fetch_all_predictions, get_prediction_stats, init_db
from spam_detector.evaluate import load_test_results
from app.components.ui_theme import apply_theme

st.set_page_config(page_title="Dashboard — SpamShield", page_icon="📊", layout="wide")
apply_theme()
init_db()

from app.components.auth_ui import require_auth
current_user = require_auth()


st.title("📊 Model Performance Dashboard")

# ---------------------------------------------------------------------------
# Test set metrics (from training)
# ---------------------------------------------------------------------------
st.subheader("🎯 Held-out Test Set Metrics")
test_results = load_test_results()

if test_results:
    cols = st.columns(5)
    metric_map = {
        "eval_accuracy": ("Accuracy", "🎯"),
        "eval_f1": ("F1 Score", "📐"),
        "eval_precision": ("Precision", "🔬"),  # if available
        "eval_recall": ("Recall", "📡"),          # if available
        "eval_loss": ("Test Loss", "📉"),
    }
    shown = 0
    for key, (label, icon) in metric_map.items():
        if key in test_results:
            cols[shown % 5].metric(f"{icon} {label}", f"{test_results[key]:.4f}")
            shown += 1
    if shown == 0:
        st.info("Train the model first to see test metrics here.")
else:
    st.info(
        "No test results found. Run training to populate this section:\n\n"
        "```bash\npython -m spam_detector.train --csv data/raw/spam.csv\n```"
    )

st.divider()

# ---------------------------------------------------------------------------
# Live prediction statistics
# ---------------------------------------------------------------------------
st.subheader("🔴 Live Prediction History")
stats = get_prediction_stats()

if stats["total"] == 0:
    st.info("No predictions logged yet. Make some predictions first!")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", stats["total"])
    c2.metric("🚨 SPAM", stats["spam_count"])
    c3.metric("✅ HAM", stats["ham_count"])
    c4.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")

    df = fetch_all_predictions()

    # Pie chart — spam vs ham
    col_pie, col_hist = st.columns(2)
    with col_pie:
        pie_data = pd.DataFrame({
            "Label": ["SPAM", "HAM"],
            "Count": [stats["spam_count"], stats["ham_count"]],
        })
        fig_pie = px.pie(pie_data, values="Count", names="Label",
                         color="Label",
                         color_discrete_map={"SPAM": "#EF4444", "HAM": "#22C55E"},
                         title="SPAM vs HAM Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_hist:
        fig_hist = px.histogram(
            df, x="confidence", color="label", nbins=20,
            color_discrete_map={"SPAM": "#EF4444", "HAM": "#22C55E"},
            title="Confidence Score Distribution",
            labels={"confidence": "Confidence", "label": "Label"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Timeline
    df["created_at"] = pd.to_datetime(df["created_at"])
    df_timeline = df.set_index("created_at").resample("1h")["id"].count().reset_index()
    df_timeline.columns = ["Time", "Predictions"]
    fig_line = px.line(df_timeline, x="Time", y="Predictions",
                       title="Predictions Over Time (hourly)", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

    # Raw log table
    with st.expander("🗃️ Raw Prediction Log"):
        st.dataframe(
            df[["id", "created_at", "label", "confidence", "source", "raw_text"]],
            use_container_width=True,
        )
