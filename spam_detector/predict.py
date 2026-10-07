"""
spam_detector.predict
=====================
Inference engine — loads the fine-tuned DistilBERT model and runs
predictions on single texts or batches.

Design decisions
----------------
* The ``Predictor`` class is a singleton-style object: the model is
  loaded once at instantiation and reused across calls — important for
  Streamlit where each user interaction re-runs the script.
* Softmax probabilities are returned alongside the hard label so the
  UI can display a confidence bar.
* Batch inference processes texts in chunks to avoid OOM on large CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from spam_detector.config import cfg
from spam_detector.preprocess import clean_email, clean_batch


@dataclass
class PredictionResult:
    """Container for a single prediction."""

    text: str                  # Original (uncleaned) text
    cleaned_text: str          # After preprocessing
    label: str                 # "SPAM" or "HAM"
    label_id: int              # 1 for SPAM, 0 for HAM
    confidence: float          # Probability of the predicted label (0–1)
    spam_prob: float           # P(SPAM)
    ham_prob: float            # P(HAM)


    @property
    def is_spam(self) -> bool:
        return self.label_id == 1

    @property
    def is_uncertain(self) -> bool:
        threshold = cfg.streamlit_cfg.get("confidence_threshold", 0.85)
        return self.confidence < threshold


class Predictor:
    """
    Wraps the fine-tuned DistilBERT model for inference.

    Parameters
    ----------
    model_dir : Path, optional
        Directory containing the saved model + tokenizer.
        Defaults to ``cfg.model_dir``.

    Examples
    --------
    >>> predictor = Predictor()
    >>> result = predictor.predict("Congratulations! You won a free iPhone. Click here.")
    >>> print(result.label, result.confidence)
    SPAM 0.9987
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or cfg.model_dir
        self.device = 0 if torch.cuda.is_available() else -1  # -1 = CPU for pipeline
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model from local disk or Hugging Face Hub."""
        local_model_path = Path(self.model_dir)
        has_local_weights = local_model_path.exists() and (
            (local_model_path / "model.safetensors").exists()
            or (local_model_path / "pytorch_model.bin").exists()
        )

        if has_local_weights:
            source = str(local_model_path)
            logger.info(f"Loading fine-tuned model from local directory: {source}")
        elif cfg.hf_repo:
            source = cfg.hf_repo
            logger.info(f"Local weights not found. Loading model from Hugging Face Hub: {source}")
        else:
            source = str(local_model_path)
            logger.warning(
                f"Model weights not found at {source} and HF_MODEL_REPO not configured. "
                f"Attempting to load from {source}."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.eval()

        # HuggingFace pipeline for convenient inference
        self._pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            top_k=None,           # Return scores for all labels
            truncation=True,
            max_length=cfg.max_length,
        )
        logger.info(f"Model successfully loaded from {source}.")

    def predict(self, text: str, clean: bool = True) -> PredictionResult:
        """
        Predict spam/ham for a single email text.

        Parameters
        ----------
        text : str
            Raw email body (may contain HTML, headers, URLs).
        clean : bool
            If True, run the preprocessing pipeline before inference.

        Returns
        -------
        PredictionResult
            Dataclass with label, confidence, and probabilities.
        """
        cleaned = clean_email(text) if clean else text
        outputs = self._pipeline(cleaned)[0]  # list of {label, score} dicts

        label_scores = {d["label"]: d["score"] for d in outputs}
        spam_prob = label_scores.get("SPAM", label_scores.get("LABEL_1", 0.0))
        ham_prob  = label_scores.get("HAM",  label_scores.get("LABEL_0", 0.0))

        if spam_prob >= ham_prob:
            label, label_id, confidence = "SPAM", 1, spam_prob
        else:
            label, label_id, confidence = "HAM", 0, ham_prob

        return PredictionResult(
            text=text,
            cleaned_text=cleaned,
            label=label,
            label_id=label_id,
            confidence=confidence,
            spam_prob=spam_prob,
            ham_prob=ham_prob,
        )

    def predict_batch(
        self, texts: list[str], clean: bool = True, chunk_size: int = 32
    ) -> list[PredictionResult]:
        """
        Predict spam/ham for a list of emails.

        Parameters
        ----------
        texts : list[str]
            List of raw email bodies.
        clean : bool
            Apply preprocessing pipeline if True.
        chunk_size : int
            Number of texts to process per forward pass (memory control).

        Returns
        -------
        list[PredictionResult]
            One result per input text, in the same order.
        """
        if clean:
            cleaned_texts = clean_batch(texts)
        else:
            cleaned_texts = texts

        results: list[PredictionResult] = []
        for i in range(0, len(cleaned_texts), chunk_size):
            chunk_clean = cleaned_texts[i : i + chunk_size]
            chunk_raw = texts[i : i + chunk_size]
            batch_outputs = self._pipeline(chunk_clean)

            for raw, cleaned, outputs in zip(chunk_raw, chunk_clean, batch_outputs):
                label_scores = {d["label"]: d["score"] for d in outputs}
                spam_prob = label_scores.get("SPAM", label_scores.get("LABEL_1", 0.0))
                ham_prob = label_scores.get("HAM", label_scores.get("LABEL_0", 0.0))

                if spam_prob >= ham_prob:
                    label, label_id, confidence = "SPAM", 1, spam_prob
                else:
                    label, label_id, confidence = "HAM", 0, ham_prob

                results.append(
                    PredictionResult(
                        text=raw,
                        cleaned_text=cleaned,
                        label=label,
                        label_id=label_id,
                        confidence=confidence,
                        spam_prob=spam_prob,
                        ham_prob=ham_prob,
                    )
                )

        logger.info(f"Batch prediction complete: {len(results)} results.")
        return results
