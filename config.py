
"""
Configuration settings for ScamShield application.
Loads environment variables and defines application constants,
paths, and database settings.

Supports:
- Local development using .env
- Streamlit Community Cloud using st.secrets
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------
# BASE PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD LOCAL ENVIRONMENT VARIABLES
# ---------------------------------------------------------

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------
# CONFIGURATION READER
# ---------------------------------------------------------

def get_setting(key: str, default: str = "") -> str:
    """
    Read configuration from Streamlit Cloud secrets first.
    Fall back to local environment variables.
    """

    try:
        value = st.secrets.get(key, default)

        if value is not None and str(value).strip():
            return str(value).strip()

    except Exception:
        # Streamlit secrets may not exist during local development
        pass

    return os.getenv(key, default).strip()


# ---------------------------------------------------------
# MONGODB CONFIGURATION
# ---------------------------------------------------------

MONGODB_URI = get_setting("MONGODB_URI")
DB_NAME = get_setting("DB_NAME", "scamshield_db")


# ---------------------------------------------------------
# COLLECTIONS
# ---------------------------------------------------------

COLLECTION_JOBS = "jobs"
COLLECTION_PREDICTIONS = "predictions"
COLLECTION_FEEDBACK = "feedback"
COLLECTION_METRICS = "model_metrics"


# ---------------------------------------------------------
# FILE PATHS FOR DATASETS AND ML ARTIFACTS
# ---------------------------------------------------------

DATASET_CSV_PATH = DATA_DIR / "fake_job_postings.csv"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"

METRICS_JSON_PATH = MODELS_DIR / "model_metrics.json"


# ---------------------------------------------------------
# MODEL METADATA
# ---------------------------------------------------------

MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42


# ---------------------------------------------------------
# RISK CLASSIFICATION THRESHOLDS
# ---------------------------------------------------------

RISK_HIGH_THRESHOLD = 0.70
RISK_MEDIUM_THRESHOLD = 0.40


# ---------------------------------------------------------
# MONGODB CONNECTION TIMEOUT
# ---------------------------------------------------------

# Kept small (2500ms) so Streamlit does not stall if Atlas is offline.
MONGO_TIMEOUT_MS = 2500


# ---------------------------------------------------------
# MONGODB CONFIGURATION VALIDATION
# ---------------------------------------------------------

def is_mongodb_configured() -> bool:
    """
    Check whether a valid MongoDB URI is provided
    and is not a placeholder.
    """

    if not MONGODB_URI:
        return False

    placeholder_keywords = [
        "yourpassword",
        "yourusername",
        "yourcluster",
        "your_mongodb_uri",
        "<username>",
        "<password>",
        "<cluster-url>",
    ]

    uri_lower = MONGODB_URI.lower()

    for keyword in placeholder_keywords:
        if keyword in uri_lower:
            return False

    return (
        MONGODB_URI.startswith("mongodb://")
        or MONGODB_URI.startswith("mongodb+srv://")
    )
