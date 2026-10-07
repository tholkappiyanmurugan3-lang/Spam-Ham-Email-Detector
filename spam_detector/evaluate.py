"""
spam_detector.evaluate
======================
Model evaluation utilities — metrics, confusion matrix, ROC curve.

All plot functions return Plotly figures so they can be rendered
inside Streamlit with ``st.plotly_chart()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from spam_detector.config import cfg


def compute_all_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: Optional[list[float]] = None,
) -> dict[str, float]:
    """
    Compute a full suite of classification metrics.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth labels (0=HAM, 1=SPAM).
    y_pred : list[int]
        Model predicted labels.
    y_prob : list[float], optional
        Predicted probabilities for SPAM class (required for AUC-ROC).

    Returns
    -------
    dict[str, float]
        Dictionary with accuracy, precision, recall, f1, and optionally auc_roc.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")

    logger.info(f"Metrics: {metrics}")
    return metrics


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: Optional[list[str]] = None,
) -> go.Figure:
    """
    Return a Plotly annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth labels.
    y_pred : list[int]
        Predicted labels.
    labels : list[str], optional
        Class names. Defaults to ["HAM", "SPAM"].

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    labels = labels or ["HAM", "SPAM"]
    cm = confusion_matrix(y_true, y_pred)
    z_text = [[str(val) for val in row] for row in cm.tolist()]

    fig = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=[f"Predicted {l}" for l in labels],
        y=[f"Actual {l}" for l in labels],
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        font=dict(size=14),
    )
    return fig


def plot_roc_curve(
    y_true: list[int],
    y_prob: list[float],
) -> go.Figure:
    """
    Return a Plotly ROC curve figure.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth labels.
    y_prob : list[float]
        Predicted probabilities for the SPAM class.

    Returns
    -------
    go.Figure
        Plotly figure with AUC annotated.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"DistilBERT (AUC = {auc:.4f})",
                   line=dict(color="#1f77b4", width=2))
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
                   line=dict(color="gray", dash="dash"))
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        font=dict(size=14),
    )
    return fig


def load_test_results(model_dir: Optional[Path] = None) -> dict:
    """
    Load persisted test results from ``test_results.json``.

    Parameters
    ----------
    model_dir : Path, optional
        Directory containing ``test_results.json``. Defaults to ``cfg.model_dir``.

    Returns
    -------
    dict
        Test metrics dict, or empty dict if file not found.
    """
    model_dir = model_dir or cfg.model_dir
    results_path = model_dir / "test_results.json"
    if not results_path.exists():
        logger.warning(f"No test_results.json found at {results_path}")
        return {}
    with results_path.open("r") as f:
        return json.load(f)
