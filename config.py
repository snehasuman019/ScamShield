"""
Configuration settings for ScamShield application.
Loads environment variables and defines application constants, paths, and database settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
DB_NAME = os.getenv("DB_NAME", "scamshield_db").strip()

# Collections
COLLECTION_JOBS = "jobs"
COLLECTION_PREDICTIONS = "predictions"
COLLECTION_FEEDBACK = "feedback"
COLLECTION_METRICS = "model_metrics"

# File paths for datasets and ML artifacts
DATASET_CSV_PATH = DATA_DIR / "fake_job_postings.csv"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
METRICS_JSON_PATH = MODELS_DIR / "model_metrics.json"

# Model metadata
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42

# Risk Classification Thresholds
RISK_HIGH_THRESHOLD = 0.70
RISK_MEDIUM_THRESHOLD = 0.40

# Client connection timeout in milliseconds
# Kept small (2500ms) so Streamlit doesn't stall if Atlas is offline
MONGO_TIMEOUT_MS = 2500


def is_mongodb_configured() -> bool:
    """Check if a valid MongoDB URI is provided and not a placeholder."""
    if not MONGODB_URI:
        return False
    placeholder_keywords = ["yourpassword", "yourusername", "yourcluster", "your_mongodb_uri"]
    for kw in placeholder_keywords:
        if kw in MONGODB_URI.lower():
            return False
    return MONGODB_URI.startswith("mongodb://") or MONGODB_URI.startswith("mongodb+srv://")
