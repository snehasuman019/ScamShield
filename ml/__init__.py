"""
Machine Learning package for ScamShield.
Implements text preprocessing, rule-based indicators, Logistic Regression,
Random Forest classification, model training, prediction, and evaluation.
"""

from .preprocess import clean_text, extract_structured_features, detect_suspicious_indicators
from .predict import predict_job, load_trained_models
from .evaluate import get_metrics_summary

__all__ = [
    "clean_text",
    "extract_structured_features",
    "detect_suspicious_indicators",
    "predict_job",
    "load_trained_models",
    "get_metrics_summary",
]
