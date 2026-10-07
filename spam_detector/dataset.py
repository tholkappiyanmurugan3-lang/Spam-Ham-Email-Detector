"""
spam_detector.dataset
=====================
PyTorch Dataset and data-loading utilities.

Design decisions
----------------
* Tokenisation is done lazily inside ``__getitem__`` to keep memory
  footprint low; for large corpora you would pre-tokenise and cache.
* Labels are stored as plain Python ints (0 = HAM, 1 = SPAM) so the
  Dataset can be used with scikit-learn scorers as well as PyTorch.
* ``load_and_split`` uses stratified splitting to maintain class balance
  across train / val / test subsets — critical when spam is < 30 % of data.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from spam_detector.config import cfg
from spam_detector.preprocess import clean_batch


class SpamDataset(Dataset):
    """
    PyTorch Dataset for spam/ham classification.

    Parameters
    ----------
    texts : list[str]
        Pre-cleaned email bodies.
    labels : list[int]
        0 for HAM, 1 for SPAM.
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer instance.
    max_length : int
        Maximum token length; inputs are truncated/padded to this.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        assert len(texts) == len(labels), "texts and labels must have the same length."
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_csv_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load a spam/ham CSV dataset.

    Expected columns: ``text`` (str) and ``label`` (0=HAM, 1=SPAM).
    Also accepts the classic SMS spam format with ``v1`` and ``v2`` columns.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``text`` (str) and ``label`` (int).

    Raises
    ------
    ValueError
        If required columns are not found.
    """
    logger.info(f"Loading dataset from {csv_path}")
    # Try reading first with default comma/sniffer, then fallback to tab if only 1 column parsed
    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, sep="\t", header=None, names=["label_str", "text"], encoding="latin-1")
    except Exception:
        df = pd.read_csv(csv_path, sep="\t", header=None, names=["label_str", "text"], encoding="latin-1")

    # Handle SMS Spam Collection format (v1=label string, v2=text) or headerless (label_str, text)
    if "label_str" in df.columns and "text" in df.columns:
        df["label"] = df["label_str"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1, "0": 0, "1": 1})
        df = df[["text", "label"]]
    elif "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label_str", "v2": "text"})
        df["label"] = df["label_str"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1, "0": 0, "1": 1})
        df = df[["text", "label"]]
    elif "text" in df.columns and "label" in df.columns:
        # If label is string 'ham'/'spam'
        if df["label"].dtype == object:
            df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1, "0": 0, "1": 1})
        else:
            df["label"] = df["label"].astype(int)
    else:
        raise ValueError(
            f"CSV must contain ('text','label') or ('v1','v2') columns. "
            f"Found: {list(df.columns)}"
        )

    # Drop rows with missing text or label
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before - after} rows with missing values.")

    logger.info(
        f"Loaded {len(df)} samples | HAM: {(df['label']==0).sum()} | SPAM: {(df['label']==1).sum()}"
    )
    return df.reset_index(drop=True)


def load_and_split(
    csv_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
) -> tuple[SpamDataset, SpamDataset, SpamDataset]:
    """
    Load CSV, clean texts, and return stratified train/val/test Datasets.

    Parameters
    ----------
    csv_path : Path
        Path to the labelled CSV.
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    test_size : float
        Fraction of data held out for testing.
    val_size : float
        Fraction of *remaining* data used for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[SpamDataset, SpamDataset, SpamDataset]
        (train_dataset, val_dataset, test_dataset)
    """
    df = load_csv_dataset(csv_path)
    preproc_cfg = cfg.preprocessing

    logger.info("Cleaning email texts...")
    df["text"] = clean_batch(
        df["text"].tolist(),
        strip_html_flag=preproc_cfg.get("strip_html", True),
        remove_urls_flag=preproc_cfg.get("remove_urls", True),
        remove_headers_flag=preproc_cfg.get("remove_email_headers", True),
        lowercase=preproc_cfg.get("lowercase", True),
        remove_extra_whitespace=preproc_cfg.get("remove_extra_whitespace", True),
    )

    # --- Stratified split: full → train+val | test ---
    X, y = df["text"].tolist(), df["label"].tolist()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # val_size is relative to the trainval portion
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=adjusted_val, stratify=y_trainval, random_state=seed
    )

    max_len = cfg.max_length

    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    return (
        SpamDataset(X_train, y_train, tokenizer, max_len),
        SpamDataset(X_val, y_val, tokenizer, max_len),
        SpamDataset(X_test, y_test, tokenizer, max_len),
    )
