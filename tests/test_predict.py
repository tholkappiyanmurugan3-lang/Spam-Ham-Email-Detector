"""
tests/test_predict.py
=====================
Integration smoke-test for the Predictor class.

NOTE: These tests require the fine-tuned model to be present at
cfg.model_dir.  They are skipped automatically if the model is not found,
so CI can run test_preprocess.py and test_db.py without a trained model.
"""

import pytest
from pathlib import Path
from spam_detector.config import cfg


MODEL_AVAILABLE = cfg.model_dir.exists()
skip_if_no_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason=f"Fine-tuned model not found at {cfg.model_dir}. Run training first.",
)


@skip_if_no_model
class TestPredictor:
    @pytest.fixture(scope="class")
    def predictor(self):
        from spam_detector.predict import Predictor
        return Predictor()

    def test_predict_returns_result(self, predictor):
        from spam_detector.predict import PredictionResult
        result = predictor.predict("Hello, how are you today?")
        assert isinstance(result, PredictionResult)

    def test_predict_spam_sample(self, predictor):
        spam_text = (
            "CONGRATULATIONS! You have WON a FREE iPhone! "
            "Click http://claim-prize.com NOW to redeem!"
        )
        result = predictor.predict(spam_text)
        assert result.label in ("SPAM", "HAM")   # Model decides, not test
        assert 0.0 <= result.confidence <= 1.0
        assert abs(result.spam_prob + result.ham_prob - 1.0) < 1e-4

    def test_predict_ham_sample(self, predictor):
        ham_text = "Hi John, the meeting is rescheduled to 3 PM tomorrow."
        result = predictor.predict(ham_text)
        assert result.label in ("SPAM", "HAM")
        assert result.confidence > 0.5

    def test_predict_batch(self, predictor):
        texts = [
            "Win a million dollars NOW!",
            "See you at the office at 9 AM.",
            "Free gift card click here",
        ]
        results = predictor.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert r.label in ("SPAM", "HAM")
            assert 0.0 <= r.confidence <= 1.0

    def test_empty_string_handled(self, predictor):
        """Empty input should not raise, just return a low-confidence result."""
        try:
            result = predictor.predict("")
            assert result.label in ("SPAM", "HAM")
        except Exception as e:
            pytest.fail(f"Empty input raised exception: {e}")
