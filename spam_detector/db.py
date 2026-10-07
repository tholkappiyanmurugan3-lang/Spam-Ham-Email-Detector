"""
spam_detector.db
================
SQLite prediction logger.

Every prediction (live or batch) is persisted so users can audit
decisions and monitor model behaviour over time.

Schema
------
predictions
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    created_at  TEXT    ISO-8601 timestamp
    input_hash  TEXT    SHA-256 of cleaned text (dedup key)
    raw_text    TEXT    Original input (first 500 chars)
    label       TEXT    "SPAM" or "HAM"
    label_id    INTEGER 1 or 0
    spam_prob   REAL    P(SPAM) in [0, 1]
    ham_prob    REAL    P(HAM) in [0, 1]
    confidence  REAL    Confidence of predicted class
    source      TEXT    "live" | "batch"

Design decisions
----------------
* SQLite is used (no server required) — adequate for a startup MVP and
  easy to replace with Postgres by swapping the connection string.
* ``input_hash`` allows quick deduplication without storing full text.
* ``with conn:`` context manager is used so writes are auto-committed
  or rolled back on exception.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from spam_detector.config import cfg
from spam_detector.predict import PredictionResult


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database."""
    db_path = db_path or cfg.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Create the ``predictions`` table if it does not already exist.

    Safe to call on every app startup.

    Parameters
    ----------
    db_path : Path, optional
        Path to SQLite file. Defaults to ``cfg.db_path``.
    """
    conn = _get_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                input_hash  TEXT    NOT NULL,
                raw_text    TEXT,
                label       TEXT    NOT NULL,
                label_id    INTEGER NOT NULL,
                spam_prob   REAL    NOT NULL,
                ham_prob    REAL    NOT NULL,
                confidence  REAL    NOT NULL,
                source      TEXT    NOT NULL DEFAULT 'live'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_input_hash ON predictions(input_hash)"
        )
    conn.close()
    logger.info(f"DB initialised at {db_path or cfg.db_path}")


def log_prediction(
    result: PredictionResult,
    source: str = "live",
    db_path: Optional[Path] = None,
) -> int:
    """
    Insert a single prediction record into the database.

    Parameters
    ----------
    result : PredictionResult
        Output of ``Predictor.predict()``.
    source : str
        Origin of the prediction — "live" or "batch".
    db_path : Path, optional
        Path to SQLite file.

    Returns
    -------
    int
        Row ID of the inserted record.
    """
    conn = _get_connection(db_path)
    input_hash = hashlib.sha256(result.cleaned_text.encode()).hexdigest()
    ts = datetime.now(timezone.utc).isoformat()

    with conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (created_at, input_hash, raw_text, label, label_id,
                 spam_prob, ham_prob, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                input_hash,
                result.text[:500],      # Store first 500 chars only
                result.label,
                result.label_id,
                result.spam_prob,
                result.ham_prob,
                result.confidence,
                source,
            ),
        )
    row_id = cursor.lastrowid
    conn.close()
    logger.debug(f"Logged prediction id={row_id} label={result.label} source={source}")
    return row_id


def log_batch(
    results: list[PredictionResult],
    db_path: Optional[Path] = None,
) -> int:
    """
    Bulk-insert a list of predictions (batch mode).

    Parameters
    ----------
    results : list[PredictionResult]
        Outputs of ``Predictor.predict_batch()``.
    db_path : Path, optional
        Path to SQLite file.

    Returns
    -------
    int
        Number of rows inserted.
    """
    conn = _get_connection(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            ts,
            hashlib.sha256(r.cleaned_text.encode()).hexdigest(),
            r.text[:500],
            r.label,
            r.label_id,
            r.spam_prob,
            r.ham_prob,
            r.confidence,
            "batch",
        )
        for r in results
    ]
    with conn:
        conn.executemany(
            """
            INSERT INTO predictions
                (created_at, input_hash, raw_text, label, label_id,
                 spam_prob, ham_prob, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.close()
    logger.info(f"Logged {len(rows)} batch predictions.")
    return len(rows)


def fetch_all_predictions(
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Retrieve logged predictions as a DataFrame.

    Parameters
    ----------
    limit : int, optional
        Maximum number of rows to return (newest first). If None or <= 0, returns all rows.
    db_path : Path, optional
        Path to SQLite file.

    Returns
    -------
    pd.DataFrame
        Rows from the ``predictions`` table, newest first.
    """
    conn = _get_connection(db_path)
    if limit is not None and limit > 0:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC", conn
        )
    conn.close()
    return df



def get_prediction_stats(db_path: Optional[Path] = None) -> dict:
    """
    Return aggregate statistics for the dashboard.

    Returns
    -------
    dict
        Keys: total, spam_count, ham_count, spam_pct, avg_confidence
    """
    conn = _get_connection(db_path)
    cur = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN label='SPAM' THEN 1 ELSE 0 END) AS spam_count,
            SUM(CASE WHEN label='HAM'  THEN 1 ELSE 0 END) AS ham_count,
            AVG(confidence) AS avg_confidence
        FROM predictions
        """
    )
    row = cur.fetchone()
    conn.close()
    total = row["total"] or 0
    spam = row["spam_count"] or 0
    ham = row["ham_count"] or 0
    return {
        "total": total,
        "spam_count": spam,
        "ham_count": ham,
        "spam_pct": (spam / total * 100) if total else 0.0,
        "avg_confidence": row["avg_confidence"] or 0.0,
    }
