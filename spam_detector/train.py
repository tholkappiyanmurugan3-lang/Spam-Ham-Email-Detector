"""
spam_detector.train
===================
Fine-tune DistilBERT for spam/ham binary classification.

Usage (from project root)::

    python -m spam_detector.train --csv data/raw/spam.csv

Design decisions
----------------
* HuggingFace ``Trainer`` is used over a manual training loop — it
  handles gradient accumulation, mixed precision, checkpointing, and
  logging out of the box.
* ``compute_metrics`` reports F1 (macro) in addition to accuracy so
  performance on the minority (spam) class is never hidden.
* The seed is set globally via ``set_seed`` *before* model init to
  ensure weight initialisation is reproducible.
* Model + tokenizer are saved together so loading at inference time
  requires only a single ``from_pretrained`` call.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from spam_detector.config import cfg
from spam_detector.dataset import load_and_split


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
_accuracy_metric = hf_evaluate.load("accuracy")
_f1_metric = hf_evaluate.load("f1")


def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Compute accuracy and macro-F1 for the HuggingFace Trainer.

    Parameters
    ----------
    eval_pred : EvalPrediction
        Named tuple with ``predictions`` (logits) and ``label_ids``.

    Returns
    -------
    dict[str, float]
        ``{"accuracy": ..., "f1": ...}``
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = _accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = _f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    csv_path: Path,
    output_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    Fine-tune DistilBERT on the provided CSV dataset.

    Parameters
    ----------
    csv_path : Path
        Path to the labelled email CSV.
    output_dir : Path, optional
        Where to save the fine-tuned model. Defaults to ``cfg.model_dir``.
    epochs : int, optional
        Number of epochs (overrides config.yaml if provided).
    batch_size : int, optional
        Batch size (overrides config.yaml if provided).
    max_samples : int, optional
        Truncate dataset to max_samples for fast testing on CPU.
    """
    # --- Setup ---
    set_seed(cfg.seed)
    output_dir = output_dir or cfg.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    # --- Tokenizer & model ---
    logger.info(f"Loading base model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )

    # --- Data ---
    tr_cfg = cfg.training
    train_ds, val_ds, test_ds = load_and_split(
        csv_path=csv_path,
        tokenizer=tokenizer,
        test_size=tr_cfg.get("test_size", 0.15),
        val_size=tr_cfg.get("val_size", 0.10),
        seed=cfg.seed,
    )

    if max_samples is not None and max_samples > 0:
        logger.info(f"Subsampling to {max_samples} training samples for fast execution.")
        train_ds.texts = train_ds.texts[:max_samples]
        train_ds.labels = train_ds.labels[:max_samples]
        val_ds.texts = val_ds.texts[:max(50, int(max_samples * 0.15))]
        val_ds.labels = val_ds.labels[:max(50, int(max_samples * 0.15))]

    num_epochs = epochs if epochs is not None else tr_cfg.get("epochs", 3)
    b_size = batch_size if batch_size is not None else tr_cfg.get("batch_size", 16)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=b_size,
        per_device_eval_batch_size=b_size,
        learning_rate=float(tr_cfg.get("learning_rate", 2e-5)),
        weight_decay=tr_cfg.get("weight_decay", 0.01),
        warmup_steps=tr_cfg.get("warmup_steps", 100),
        eval_strategy=tr_cfg.get("eval_strategy", "epoch"),
        save_strategy=tr_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=tr_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=tr_cfg.get("metric_for_best_model", "f1"),
        logging_dir=str(cfg.logs_dir / "tensorboard"),
        logging_steps=50,
        report_to="none",          # Disable W&B / MLflow unless configured
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # --- Save best model ---
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # --- Evaluate on test set ---
    logger.info("Evaluating on held-out test set...")
    test_results = trainer.evaluate(eval_dataset=test_ds)
    logger.info(f"Test results: {test_results}")

    # Persist test results for the dashboard
    import json
    results_path = output_dir / "test_results.json"
    with results_path.open("w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved to {results_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from typing import Optional

    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for spam detection.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to labelled CSV with 'text' and 'label' (or 'v1','v2') columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the fine-tuned model (default: from config.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config.yaml).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per device (default: from config.yaml).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples to use (useful for fast CPU training/testing).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from (e.g. models/distilbert_spam/checkpoints/checkpoint-487).",
    )
    args = parser.parse_args()
    train(
        csv_path=args.csv,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

