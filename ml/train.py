"""
Model Training & Evaluation Pipeline for ScamShield.
Trains two genuine ML models on the labelled EMSCAD dataset:
1. Model A: Logistic Regression with TF-IDF Vectorization (Text / NLP)
2. Model B: Random Forest Classifier with Structured Tabular Features

Retrieves training data from MongoDB Atlas (PyMongo) when populated,
or falls back cleanly to the local CSV dataset.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project imports
import config
from ml.preprocess import clean_text, extract_structured_features
from database.connection import get_db, check_connection
from database.operations import save_model_evaluation_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scamshield.train")


def load_dataset(source_preference: str = "auto") -> tuple[pd.DataFrame, str]:
    """
    Load dataset from MongoDB Atlas or fallback to local CSV.
    Returns (DataFrame, source_description).
    """
    db = get_db()
    if source_preference in ("auto", "db") and db is not None:
        try:
            job_count = db[config.COLLECTION_JOBS].count_documents({})
            if job_count >= 500:
                logger.info(f"Loading dataset directly from MongoDB Atlas 'jobs' collection ({job_count:,} docs)...")
                cursor = db[config.COLLECTION_JOBS].find({}, {"_id": 0})
                df = pd.DataFrame(list(cursor))
                if len(df) > 0 and "fraudulent" in df.columns:
                    return df, f"MongoDB Atlas ('{config.DB_NAME}.{config.COLLECTION_JOBS}', {len(df):,} records)"
        except Exception as e:
            logger.warning(f"Could not load dataset from MongoDB: {e}")

    # Fallback to CSV
    csv_path = config.DATASET_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}. Please run 'python data/download_dataset.py' first."
        )

    logger.info(f"Loading dataset from local CSV file: {csv_path}...")
    df = pd.read_csv(csv_path)
    return df, f"Local CSV ({csv_path.name}, {len(df):,} records)"


def prepare_data(df: pd.DataFrame, sample_size: int = 0):
    """
    Clean dataset, handle missing values, remove duplicates, and build text/feature matrices.
    """
    logger.info("Cleaning raw dataset and inspecting target distribution...")

    # Drop exact duplicate rows if any
    initial_len = len(df)
    df = df.drop_duplicates(subset=["title", "description"]).copy()
    logger.info(f"Deduplicated dataset: retained {len(df):,} of {initial_len:,} rows.")

    # Target label check
    if "fraudulent" not in df.columns:
        raise ValueError("Missing required ground-truth label 'fraudulent' in dataset!")

    # Clean target
    df["fraudulent"] = df["fraudulent"].fillna(0).astype(int)
    class_counts = df["fraudulent"].value_counts().to_dict()
    logger.info(f"Ground-truth class balance: 0 (Legit)={class_counts.get(0, 0):,}, 1 (Fraud)={class_counts.get(1, 0):,}")

    if sample_size > 0 and sample_size < len(df):
        logger.info(f"Sampling down to {sample_size:,} records for fast execution...")
        df = df.sample(n=sample_size, random_state=config.RANDOM_STATE, stratify=df["fraudulent"])

    # Build combined text for Model A (NLP)
    logger.info("Building normalized combined text corpus for Model A...")
    text_corpus = []
    for _, row in df.iterrows():
        title = str(row.get("title") or "")
        company = str(row.get("company_profile") or "")
        desc = str(row.get("description") or "")
        req = str(row.get("requirements") or "")
        combined = f"{title} {company} {desc} {req}"
        text_corpus.append(clean_text(combined))

    df["combined_text"] = text_corpus

    # Build structured features for Model B (Random Forest)
    logger.info("Extracting multi-attribute tabular features for Model B...")
    structured_features = extract_structured_features(df)

    y = df["fraudulent"].values
    return df, structured_features, y


def train_and_evaluate(source_preference: str = "auto", sample_size: int = 0) -> dict:
    """
    Full training and evaluation pipeline for both models.
    """
    print("\n" + "=" * 70)
    print("      SCAMSHIELD: DUAL MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load data
    df, source_desc = load_dataset(source_preference)
    df, X_structured, y = prepare_data(df, sample_size)

    # 2. Stratified Train / Test Split (80 / 20)
    # Guaranteed identical split index across both models to ensure fair comparison without data leakage
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(f"Dataset split: {len(train_idx):,} training samples, {len(test_idx):,} testing samples.")

    # -------------------------------------------------------------
    # MODEL A: Logistic Regression with TF-IDF Vectorization
    # -------------------------------------------------------------
    logger.info("\n>>> Training Model A: Logistic Regression (TF-IDF NLP)...")
    X_text_train = df["combined_text"].iloc[train_idx]
    X_text_test = df["combined_text"].iloc[test_idx]

    model_a_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
                min_df=2,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",  # Critical for ~4.8% minority fraud class
                max_iter=1000,
                random_state=config.RANDOM_STATE,
                C=1.0,
            ),
        ),
    ])

    model_a_pipeline.fit(X_text_train, y_train)
    y_pred_a = model_a_pipeline.predict(X_text_test)
    y_prob_a = model_a_pipeline.predict_proba(X_text_test)[:, 1]

    metrics_a = {
        "model_name": "Logistic Regression (TF-IDF)",
        "model_type": "LogisticRegression",
        "feature_set": "TF-IDF N-Gram Job Text",
        "accuracy": round(float(accuracy_score(y_test, y_pred_a)), 4),
        "precision": round(float(precision_score(y_test, y_pred_a, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_a, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_a, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_a)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_a).tolist(),
    }

    logger.info(f"Model A Results: F1={metrics_a['f1_score']} | Recall={metrics_a['recall']} | Precision={metrics_a['precision']} | Acc={metrics_a['accuracy']}")

    # -------------------------------------------------------------
    # MODEL B: Random Forest Classifier with Structured Features
    # -------------------------------------------------------------
    logger.info("\n>>> Training Model B: Random Forest (Structured Features)...")
    X_struct_train = X_structured.iloc[train_idx]
    X_struct_test = X_structured.iloc[test_idx]

    categorical_cols = ["employment_type", "required_experience", "required_education"]
    numeric_cols = [
        "has_salary_range",
        "has_company_profile",
        "has_company_logo",
        "has_questions",
        "telecommuting",
        "has_payment_keywords",
        "desc_char_length",
        "desc_word_count",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_cols,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
        ]
    )

    model_b_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    model_b_pipeline.fit(X_struct_train, y_train)
    y_pred_b = model_b_pipeline.predict(X_struct_test)
    y_prob_b = model_b_pipeline.predict_proba(X_struct_test)[:, 1]

    metrics_b = {
        "model_name": "Random Forest (Structured Features)",
        "model_type": "RandomForestClassifier",
        "feature_set": "Metadata & Operational Flags (Salary, Profile, Logo, Questions, Education, Payment Keywords)",
        "accuracy": round(float(accuracy_score(y_test, y_pred_b)), 4),
        "precision": round(float(precision_score(y_test, y_pred_b, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_b, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_b, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_b)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_b).tolist(),
    }

    logger.info(f"Model B Results: F1={metrics_b['f1_score']} | Recall={metrics_b['recall']} | Precision={metrics_b['precision']} | Acc={metrics_b['accuracy']}")

    # -------------------------------------------------------------
    # Save Artifacts & Metadata
    # -------------------------------------------------------------
    training_timestamp = datetime.now(timezone.utc).isoformat()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_a_pipeline, config.LOGISTIC_MODEL_PATH)
    joblib.dump(model_b_pipeline, config.RANDOM_FOREST_MODEL_PATH)
    logger.info(f"Saved fitted pipelines to {config.MODELS_DIR}")

    full_metrics = {
        "training_timestamp": training_timestamp,
        "dataset_source": source_desc,
        "total_samples": len(df),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "target_balance": {
            "legitimate": int((y == 0).sum()),
            "fraudulent": int((y == 1).sum()),
            "fraud_prevalence_pct": round(float((y == 1).mean() * 100), 2),
        },
        "model_version": config.MODEL_VERSION,
        "models": {
            "logistic_regression": metrics_a,
            "random_forest": metrics_b,
        },
        "methodological_notes": (
            "Model A uses linguistic patterns and TF-IDF tokenization of job ad descriptions. "
            "Model B evaluates structured organizational attributes (salary, logo, screening questions, education). "
            "Because each model relies on distinct feature spaces, performance differences reflect the informational value "
            "of text vs. metadata rather than algorithmic superiority alone. Both models utilize class-weighted balancing to "
            "counteract severe real-world class imbalance (~4.8% fraud prevalence)."
        ),
    }

    # Save to local JSON file
    with open(config.METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info(f"Saved metrics metadata to {config.METRICS_JSON_PATH}")

    # Save to MongoDB Atlas collection 'model_metrics' if database is accessible
    db = get_db()
    if db is not None:
        try:
            for m_key, m_val in full_metrics["models"].items():
                record = {
                    "model_key": m_key,
                    "model_name": m_val["model_name"],
                    "version": config.MODEL_VERSION,
                    "accuracy": m_val["accuracy"],
                    "precision": m_val["precision"],
                    "recall": m_val["recall"],
                    "f1_score": m_val["f1_score"],
                    "roc_auc": m_val["roc_auc"],
                    "confusion_matrix": m_val["confusion_matrix"],
                    "feature_set": m_val["feature_set"],
                    "training_timestamp": training_timestamp,
                    "dataset_source": source_desc,
                }
                save_model_evaluation_metrics(record, db=db)
            logger.info("Saved model evaluation metrics to MongoDB Atlas 'model_metrics' collection.")
        except Exception as e:
            logger.warning(f"Could not persist metrics to MongoDB Atlas: {e}")

    print("\n" + "=" * 70)
    print("                TRAINING & EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset Source : {source_desc}")
    print(f"Total Rows     : {len(df):,} | Fraud Prevalence: {full_metrics['target_balance']['fraud_prevalence_pct']}%")
    print("-" * 70)
    print(f"Model A (Logistic Regression TF-IDF):")
    print(f"  Accuracy  : {metrics_a['accuracy']:.4f}")
    print(f"  Precision : {metrics_a['precision']:.4f}")
    print(f"  Recall    : {metrics_a['recall']:.4f}")
    print(f"  F1-Score  : {metrics_a['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics_a['roc_auc']:.4f}")
    print("-" * 70)
    print(f"Model B (Random Forest Structured Features):")
    print(f"  Accuracy  : {metrics_b['accuracy']:.4f}")
    print(f"  Precision : {metrics_b['precision']:.4f}")
    print(f"  Recall    : {metrics_b['recall']:.4f}")
    print(f"  F1-Score  : {metrics_b['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics_b['roc_auc']:.4f}")
    print("=" * 70 + "\n")

    return full_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train ScamShield ML Models on genuine dataset.")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "db", "csv"], help="Data source preference")
    parser.add_argument("--sample_size", type=int, default=0, help="Optional sample size (0 for full dataset)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train_and_evaluate(source_preference=args.source, sample_size=args.sample_size)
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)
