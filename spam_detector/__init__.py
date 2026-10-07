"""
spam_detector
=============
Top-level package for the Spam-Ham Email Detector.

Exposes the main public API so callers can do:
    from spam_detector import Predictor, Preprocessor
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spam-ham-detector")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = ["config", "preprocess", "dataset", "train", "predict", "explain", "evaluate", "db"]
