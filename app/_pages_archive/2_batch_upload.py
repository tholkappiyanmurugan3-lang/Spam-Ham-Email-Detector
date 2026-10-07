"""
app/pages/2_batch_upload.py
===========================
Batch CSV upload and bulk prediction page.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import io
import pandas as pd
import streamlit as st
from spam_detector.config import cfg
from spam_detector.db import log_batch, init_db
from app.components.ui_theme import apply_theme

st.set_page_config(page_title="Batch Upload — SpamShield", page_icon="📂", layout="wide")
apply_theme()
init_db()

from app.components.auth_ui import require_auth
current_user = require_auth()


st.title("📂 Batch Email Classification")
st.markdown(
    "Upload a **CSV file** with an `email` or `text` column. "
    "The model will classify every row and you can download the annotated results."
)

# ---------------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_predictor():
    from spam_detector.predict import Predictor
    return Predictor()


model_dir = cfg.model_dir
if not model_dir.exists():
    st.error("Model not found. Please train first: `python -m spam_detector.train --csv data/raw/spam.csv`")
    st.stop()

predictor = load_predictor()

# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="CSV must have a column named `text`, `email`, or `body`.",
)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded, encoding="latin-1")
    st.write(f"**Loaded:** {len(df_raw)} rows — Columns: `{list(df_raw.columns)}`")

    # Auto-detect text column
    text_col = None
    for candidate in ["text", "email", "body", "message", "v2"]:
        if candidate in df_raw.columns:
            text_col = candidate
            break

    if text_col is None:
        st.error(
            "Could not find a text column. "
            "Rename your column to `text`, `email`, `body`, or `message`."
        )
        st.stop()

    st.info(f"Using column: **`{text_col}`**")
    st.dataframe(df_raw[[text_col]].head(5), use_container_width=True)

    if st.button("🚀 Run Classification", type="primary"):
        texts = df_raw[text_col].fillna("").tolist()

        progress_bar = st.progress(0, text="Classifying emails...")
        results = []
        chunk = 32
        for i in range(0, len(texts), chunk):
            batch_results = predictor.predict_batch(texts[i : i + chunk])
            results.extend(batch_results)
            progress_bar.progress(min((i + chunk) / len(texts), 1.0), text=f"Processing {min(i+chunk, len(texts))}/{len(texts)}...")

        progress_bar.empty()

        # Log to DB
        log_batch(results)

        # Build result DataFrame
        df_out = df_raw.copy()
        df_out["predicted_label"] = [r.label for r in results]
        df_out["spam_probability"] = [round(r.spam_prob, 4) for r in results]
        df_out["ham_probability"] = [round(r.ham_prob, 4) for r in results]
        df_out["confidence"] = [round(r.confidence, 4) for r in results]
        df_out["uncertain"] = [r.is_uncertain for r in results]

        st.success(f"✅ Done! Classified {len(results)} emails.")

        # Summary
        spam_n = sum(1 for r in results if r.is_spam)
        ham_n = len(results) - spam_n
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(results))
        c2.metric("🚨 SPAM", spam_n)
        c3.metric("✅ HAM", ham_n)

        st.dataframe(df_out, use_container_width=True)

        # Download button
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_bytes,
            file_name="spam_predictions.csv",
            mime="text/csv",
            type="primary",
        )
