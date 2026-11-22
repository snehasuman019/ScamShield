"""
Prediction and inference engine for ScamShield.
Loads trained scikit-learn pipelines and evaluates new job postings.
Outputs:
- Predicted class (Fraudulent vs Legitimate)
- Predicted probability / confidence
- Transparent rule-based indicators
- Plain-English limitations disclaimer
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import pandas as pd
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ml.preprocess import clean_text, extract_structured_features, detect_suspicious_indicators

logger = logging.getLogger("scamshield.predict")

# In-memory model cache
_cached_models: Dict[str, Any] = {}

DISCLAIMER = (
    "NOTICE: This assessment is an automated risk score produced by statistical machine learning models "
    "trained on historical recruitment data (EMSCAD). It does NOT constitute legal or definitive proof that "
    "a listing is fraudulent or legitimate. Always verify employers through independent official channels, "
    "company registry databases, and official corporate communication domains before sharing personal or financial details."
)


def load_trained_models(force_reload: bool = False) -> Dict[str, Any]:
    """Load both trained pipelines from disk into memory."""
    global _cached_models

    if _cached_models and not force_reload:
        return _cached_models

    models = {}

    if config.LOGISTIC_MODEL_PATH.exists():
        try:
            models["logistic_regression"] = joblib.load(config.LOGISTIC_MODEL_PATH)
        except Exception as e:
            logger.error(f"Error loading Logistic Regression model: {e}")

    if config.RANDOM_FOREST_MODEL_PATH.exists():
        try:
            models["random_forest"] = joblib.load(config.RANDOM_FOREST_MODEL_PATH)
        except Exception as e:
            logger.error(f"Error loading Random Forest model: {e}")

    _cached_models = models
    return _cached_models


def is_model_available(model_key: str) -> bool:
    """Check if the selected model pipeline artifact exists."""
    models = load_trained_models()
    return model_key in models


def predict_job(
    model_key: str,
    job_title: str,
    job_description: str,
    company_name: str = "",
    requirements: str = "",
    structured_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run prediction for a single job listing.
    model_key: 'logistic_regression' or 'random_forest'
    """
    models = load_trained_models()

    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    if model_key not in models:
        raise FileNotFoundError(
            f"Selected model '{model_key}' is not loaded. Please train the models first using 'python ml/train.py'."
        )

    pipeline = models[model_key]
    struct = structured_inputs or {}

    # 1. Rule-based inspection
    has_logo = struct.get("has_company_logo", 1 if company_name else 0)
    has_profile = bool(company_name)
    has_questions = struct.get("has_questions", 0)

    indicators = detect_suspicious_indicators(
        text=f"{job_title} {job_description} {requirements}",
        has_company_logo=has_logo,
        has_company_profile=has_profile,
        has_questions=has_questions,
    )

    # 2. Model Inference
    if model_key == "logistic_regression":
        model_name = "Logistic Regression (TF-IDF NLP)"
        # Prepare combined text
        combined = f"{job_title} {company_name} {job_description} {requirements}"
        clean_combined = clean_text(combined)

        pred_label = int(pipeline.predict([clean_combined])[0])
        probabilities = pipeline.predict_proba([clean_combined])[0]
        prob_fraud = float(probabilities[1])
        prob_legit = float(probabilities[0])

    elif model_key == "random_forest":
        model_name = "Random Forest (Structured Features)"
        # Build single-row DataFrame for structured pipeline
        row_dict = {
            "title": job_title,
            "company_profile": company_name,
            "description": job_description,
            "requirements": requirements,
            "salary_range": struct.get("salary_range", ""),
            "has_company_logo": struct.get("has_company_logo", 0),
            "has_questions": struct.get("has_questions", 0),
            "telecommuting": struct.get("telecommuting", 0),
            "employment_type": struct.get("employment_type", "Unknown"),
            "required_experience": struct.get("required_experience", "Unknown"),
            "required_education": struct.get("required_education", "Unknown"),
        }
        df_single = pd.DataFrame([row_dict])
        X_features = extract_structured_features(df_single)

        pred_label = int(pipeline.predict(X_features)[0])
        probabilities = pipeline.predict_proba(X_features)[0]
        prob_fraud = float(probabilities[1])
        prob_legit = float(probabilities[0])
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    # Determine class and risk level
    predicted_class = "Fraudulent" if pred_label == 1 else "Legitimate"

    # Risk level categorization
    has_high_severity_rule = any(ind["severity"] == "High" for ind in indicators)
    if prob_fraud >= config.RISK_HIGH_THRESHOLD or has_high_severity_rule:
        risk_level = "High"
    elif prob_fraud >= config.RISK_MEDIUM_THRESHOLD:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    confidence = prob_fraud if pred_label == 1 else prob_legit

    return {
        "model_key": model_key,
        "model_name": model_name,
        "model_version": config.MODEL_VERSION,
        "predicted_label": pred_label,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{round(confidence * 100, 1)}%",
        "probability_fraudulent": round(prob_fraud, 4),
        "probability_legitimate": round(prob_legit, 4),
        "risk_level": risk_level,
        "detected_indicators": indicators,
        "indicators_count": len(indicators),
        "limitations_disclaimer": DISCLAIMER,
        "job_title": job_title,
        "structured_inputs": struct,
    }
