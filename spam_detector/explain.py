"""
spam_detector.explain
=====================
SHAP-based explainability for the spam detector.

Returns token-level importance scores so users can see *which words*
drove the SPAM / HAM decision.

Design decisions
----------------
* ``shap.Explainer`` with a ``transformers`` pipeline is used — this
  leverages the Partition SHAP algorithm which works well for text.
* Results are returned as a list of (token, shap_value) tuples sorted
  by absolute importance, making them easy to render in Streamlit.
* A ``max_evals`` cap is enforced from config to keep inference time
  acceptable (< 10 s on CPU for typical email lengths).

Trade-off: Partition SHAP is an approximation — exact Shapley values
for transformers are computationally intractable.  The approximation is
accurate enough for explanatory purposes.
"""

from __future__ import annotations

from typing import Optional

import shap
from loguru import logger
from transformers import pipeline as hf_pipeline

from spam_detector.config import cfg
from spam_detector.preprocess import clean_email


def get_explainer(predictor_pipeline) -> shap.Explainer:
    """
    Build a SHAP Explainer around the HuggingFace text-classification pipeline.

    Parameters
    ----------
    predictor_pipeline : transformers.Pipeline
        The ``text-classification`` pipeline from ``Predictor._pipeline``.

    Returns
    -------
    shap.Explainer
        Ready-to-use SHAP explainer.
    """
    masker = shap.maskers.Text(tokenizer=r"\W+")  # Split on non-word chars
    explainer = shap.Explainer(predictor_pipeline, masker=masker)
    return explainer


def explain_prediction(
    text: str,
    predictor_pipeline,
    max_evals: Optional[int] = None,
    num_top_tokens: Optional[int] = None,
    clean: bool = True,
) -> list[tuple[str, float]]:
    """
    Compute SHAP token importance for a single email.

    Parameters
    ----------
    text : str
        Raw email body.
    predictor_pipeline : transformers.Pipeline
        Pipeline from ``Predictor._pipeline``.
    max_evals : int, optional
        Maximum SHAP evaluations (controls speed vs. accuracy).
    num_top_tokens : int, optional
        Number of top-importance tokens to return.
    clean : bool
        If True, preprocess the text before explaining.

    Returns
    -------
    list[tuple[str, float]]
        List of (token, shap_value) pairs, sorted by |shap_value| descending.
        Positive values push toward SPAM, negative toward HAM.
    """
    shap_cfg = cfg.shap_cfg
    max_evals = max_evals or shap_cfg.get("max_evals", 500)
    num_top_tokens = num_top_tokens or shap_cfg.get("num_top_tokens", 15)

    processed = clean_email(text) if clean else text

    logger.info(f"Running SHAP explanation (max_evals={max_evals})...")
    explainer = get_explainer(predictor_pipeline)

    shap_values = explainer([processed], max_evals=max_evals)

    # shap_values.values shape: (n_samples, n_tokens, n_classes)
    # We take class index 1 (SPAM) values for the first sample
    tokens = shap_values.data[0]          # list of token strings
    values = shap_values.values[0]        # shape: (n_tokens, n_classes)

    # Extract SPAM-class SHAP values (index 1)
    spam_shap = values[:, 1].tolist()

    token_importance = list(zip(tokens, spam_shap))

    # Sort by absolute SHAP value (most influential first)
    token_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    logger.info(f"SHAP explanation complete. Top token: {token_importance[0] if token_importance else 'N/A'}")
    return token_importance[:num_top_tokens]


def shap_to_html(token_importance: list[tuple[str, float]]) -> str:
    """
    Convert SHAP token importance into an HTML colour-coded string.

    Tokens with positive SHAP (→ SPAM) are highlighted red,
    negative SHAP (→ HAM) in green.  Intensity scales with |shap_value|.

    Parameters
    ----------
    token_importance : list[tuple[str, float]]
        Output of :func:`explain_prediction`.

    Returns
    -------
    str
        HTML string suitable for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    if not token_importance:
        return "<p>No explanation available.</p>"

    max_val = max(abs(v) for _, v in token_importance) or 1.0

    parts = []
    for token, val in sorted(token_importance, key=lambda x: x[0]):  # alphabetical for display
        intensity = int(200 * abs(val) / max_val)
        if val > 0:
            color = f"rgba(255, {255 - intensity}, {255 - intensity}, 0.8)"
        else:
            color = f"rgba({255 - intensity}, 255, {255 - intensity}, 0.8)"
        parts.append(
            f'<span style="background-color:{color}; padding:2px 4px; '
            f'border-radius:3px; margin:2px; display:inline-block;">'
            f'{token} <small>({val:+.3f})</small></span>'
        )

    return "<p>" + " ".join(parts) + "</p>"
