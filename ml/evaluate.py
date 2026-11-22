"""
Evaluation metrics reporting and model explainability utility.
Loads evaluation results, formats comparison tables, and extracts
top informative features from trained pipelines.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ml.predict import load_trained_models
from database.connection import get_db

logger = logging.getLogger("scamshield.evaluate")


def load_metrics_json() -> Optional[Dict[str, Any]]:
    """Load evaluation metrics from models/model_metrics.json."""
    if not config.METRICS_JSON_PATH.exists():
        return None
    try:
        with open(config.METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading metrics JSON: {e}")
        return None


def get_metrics_summary() -> Optional[Dict[str, Any]]:
    """
    Retrieve evaluation metrics from local file or MongoDB Atlas.
    Never returns fabricated metrics.
    """
    metrics = load_metrics_json()
    if metrics:
        return metrics

    # Try MongoDB Atlas collection
    db = get_db()
    if db is not None:
        try:
            docs = list(db[config.COLLECTION_METRICS].find({}, {"_id": 0}).sort("training_timestamp", -1).limit(2))
            if docs:
                return {
                    "source": "MongoDB Atlas",
                    "models_list": docs,
                }
        except Exception:
            pass

    return None


def get_top_fraud_features_logistic(top_n: int = 15) -> List[Tuple[str, float]]:
    """
    Extract words / n-grams with the highest positive coefficients
    from the fitted Logistic Regression pipeline.
    """
    models = load_trained_models()
    if "logistic_regression" not in models:
        return []

    try:
        pipeline = models["logistic_regression"]
        vectorizer = pipeline.named_steps.get("tfidf")
        classifier = pipeline.named_steps.get("classifier")

        if not vectorizer or not classifier:
            return []

        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = classifier.coef_[0]

        # Sort indices by highest positive weight (predictive of fraud)
        top_positive_indices = np.argsort(coefs)[-top_n:][::-1]
        results = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_positive_indices]
        return results
    except Exception as e:
        logger.error(f"Could not extract top features: {e}")
        return []


def get_top_features_random_forest(top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract feature importances from the Random Forest pipeline.
    """
    models = load_trained_models()
    if "random_forest" not in models:
        return []

    try:
        pipeline = models["random_forest"]
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier")

        if not preprocessor or not classifier:
            return []

        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        top_indices = np.argsort(importances)[-top_n:][::-1]
        clean_names = [feature_names[i].split("__")[-1] for i in top_indices]
        return [(clean_names[idx], round(float(importances[i]), 4)) for idx, i in enumerate(top_indices)]
    except Exception as e:
        logger.error(f"Could not extract RF feature importances: {e}")
        return []
