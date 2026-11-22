"""
MongoDB Schema Validation module for ScamShield.
Demonstrates CSE494 syllabus concept: Enforcing document structure in NoSQL
using JSON Schema validation ($jsonSchema) via PyMongo.
"""

import logging
from pymongo.database import Database
from pymongo.errors import PyMongoError
import config

logger = logging.getLogger("scamshield.validation")

JOBS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["title", "description", "fraudulent"],
        "properties": {
            "job_id": {"bsonType": ["string", "int", "long"], "description": "Unique identifier of the job ad"},
            "title": {"bsonType": "string", "description": "Job title is required and must be a string"},
            "company_name": {"bsonType": ["string", "null"], "description": "Company name if known"},
            "description": {"bsonType": "string", "description": "Job description text is required"},
            "fraudulent": {"bsonType": ["int", "long", "double"], "enum": [0, 1], "description": "Ground truth flag (0=Legitimate, 1=Fraudulent)"},
            "employment_type": {"bsonType": ["string", "null"]},
            "required_experience": {"bsonType": ["string", "null"]},
            "required_education": {"bsonType": ["string", "null"]},
            "salary_range": {"bsonType": ["string", "null"]},
            "has_company_logo": {"bsonType": ["int", "long", "double", "null"]},
            "has_questions": {"bsonType": ["int", "long", "double", "null"]},
            "telecommuting": {"bsonType": ["int", "long", "double", "null"]},
            "source": {"bsonType": "string"},
            "created_at": {"bsonType": ["string", "date"]},
        },
    }
}

PREDICTIONS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["prediction_id", "predicted_class", "confidence", "model_name"],
        "properties": {
            "prediction_id": {"bsonType": "string", "description": "Unique UUID for the prediction"},
            "job_title": {"bsonType": "string", "description": "Job title assessed"},
            "job_description_snippet": {"bsonType": "string", "description": "Summary or snippet of job ad"},
            "predicted_class": {"bsonType": "string", "enum": ["Fraudulent", "Legitimate"]},
            "predicted_label": {"bsonType": ["int", "long"], "enum": [0, 1]},
            "confidence": {"bsonType": ["double", "int", "long"], "minimum": 0.0, "maximum": 1.0},
            "risk_level": {"bsonType": "string", "enum": ["Low", "Moderate", "High"]},
            "model_name": {"bsonType": "string"},
            "model_version": {"bsonType": "string"},
            "detected_indicators": {"bsonType": "array"},
            "feedback_status": {"bsonType": ["string", "null"]},
            "created_at": {"bsonType": ["string", "date"]},
        },
    }
}

FEEDBACK_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["feedback_id", "prediction_id", "user_assessment", "timestamp"],
        "properties": {
            "feedback_id": {"bsonType": "string"},
            "prediction_id": {"bsonType": "string"},
            "user_assessment": {"bsonType": "string", "enum": ["Accurate", "Inaccurate", "Uncertain"]},
            "notes": {"bsonType": ["string", "null"]},
            "timestamp": {"bsonType": ["string", "date"]},
        },
    }
}

METRICS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["model_name", "version", "accuracy", "f1_score", "training_timestamp"],
        "properties": {
            "model_name": {"bsonType": "string"},
            "version": {"bsonType": "string"},
            "accuracy": {"bsonType": ["double", "int"]},
            "precision": {"bsonType": ["double", "int"]},
            "recall": {"bsonType": ["double", "int"]},
            "f1_score": {"bsonType": ["double", "int"]},
            "training_timestamp": {"bsonType": ["string", "date"]},
            "dataset_info": {"bsonType": "object"},
        },
    }
}

COLLECTION_VALIDATORS = {
    config.COLLECTION_JOBS: JOBS_VALIDATOR,
    config.COLLECTION_PREDICTIONS: PREDICTIONS_VALIDATOR,
    config.COLLECTION_FEEDBACK: FEEDBACK_VALIDATOR,
    config.COLLECTION_METRICS: METRICS_VALIDATOR,
}


def apply_schema_validation(db: Database) -> dict:
    """
    Apply JSON Schema validation to all ScamShield collections.
    Uses collMod for existing collections or create_collection for new ones.
    Returns status dictionary.
    """
    results = {}
    try:
        existing_collections = db.list_collection_names()
    except Exception as e:
        logger.warning(f"Could not list collections for schema validation: {e}")
        return {"error": str(e)}

    for coll_name, validator in COLLECTION_VALIDATORS.items():
        try:
            if coll_name in existing_collections:
                # Update existing collection with validation rules
                db.command({
                    "collMod": coll_name,
                    "validator": validator,
                    "validationLevel": "moderate",
                    "validationAction": "warn",
                })
                results[coll_name] = "Updated existing schema validator (level: moderate)"
            else:
                # Create collection with validator
                db.create_collection(
                    coll_name,
                    validator=validator,
                    validationLevel="moderate",
                    validationAction="warn",
                )
                results[coll_name] = "Created collection with schema validator"
        except PyMongoError as pe:
            logger.warning(f"Schema validation could not be applied to {coll_name}: {pe}")
            results[coll_name] = f"Skipped or limited permission: {pe.__class__.__name__}"

    return results
