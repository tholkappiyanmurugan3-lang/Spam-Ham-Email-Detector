"""
tests/test_db.py
================
Unit tests for the SQLite prediction logger.
"""

import tempfile
from pathlib import Path
import pytest

from spam_detector.db import init_db, log_prediction, log_batch, fetch_all_predictions, get_prediction_stats
from spam_detector.predict import PredictionResult


def make_result(label: str = "SPAM", confidence: float = 0.95) -> PredictionResult:
    """Helper to create a dummy PredictionResult."""
    label_id = 1 if label == "SPAM" else 0
    spam_prob = confidence if label == "SPAM" else 1 - confidence
    ham_prob = 1 - spam_prob
    return PredictionResult(
        text="Test email body",
        cleaned_text="test email body",
        label=label,
        label_id=label_id,
        confidence=confidence,
        spam_prob=spam_prob,
        ham_prob=ham_prob,
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database for each test."""
    db_path = tmp_path / "test_predictions.db"
    init_db(db_path=db_path)
    return db_path


class TestInitDb:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        init_db(db_path=db_path)
        assert db_path.exists()

    def test_idempotent(self, tmp_db):
        """Calling init_db twice should not raise."""
        init_db(db_path=tmp_db)


class TestLogPrediction:
    def test_inserts_row(self, tmp_db):
        result = make_result("SPAM", 0.98)
        row_id = log_prediction(result, source="live", db_path=tmp_db)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_ham_prediction(self, tmp_db):
        result = make_result("HAM", 0.93)
        log_prediction(result, source="live", db_path=tmp_db)
        df = fetch_all_predictions(db_path=tmp_db)
        assert len(df) == 1
        assert df.iloc[0]["label"] == "HAM"

    def test_fetch_returns_correct_data(self, tmp_db):
        result = make_result("SPAM", 0.99)
        log_prediction(result, db_path=tmp_db)
        df = fetch_all_predictions(db_path=tmp_db)
        row = df.iloc[0]
        assert row["label"] == "SPAM"
        assert abs(row["confidence"] - 0.99) < 1e-4


class TestLogBatch:
    def test_bulk_insert(self, tmp_db):
        results = [make_result("SPAM", 0.9), make_result("HAM", 0.85), make_result("SPAM", 0.95)]
        count = log_batch(results, db_path=tmp_db)
        assert count == 3
        df = fetch_all_predictions(db_path=tmp_db)
        assert len(df) == 3

    def test_empty_batch(self, tmp_db):
        count = log_batch([], db_path=tmp_db)
        assert count == 0


class TestGetPredictionStats:
    def test_stats_empty_db(self, tmp_db):
        stats = get_prediction_stats(db_path=tmp_db)
        assert stats["total"] == 0
        assert stats["spam_pct"] == 0.0

    def test_stats_with_data(self, tmp_db):
        log_batch([make_result("SPAM"), make_result("HAM"), make_result("SPAM")], db_path=tmp_db)
        stats = get_prediction_stats(db_path=tmp_db)
        assert stats["total"] == 3
        assert stats["spam_count"] == 2
        assert stats["ham_count"] == 1
        assert abs(stats["spam_pct"] - 66.67) < 0.1
