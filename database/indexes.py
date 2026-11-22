"""
MongoDB Index Management and Query Optimization for ScamShield.
Demonstrates CSE494 syllabus concepts: B-Tree Indexing, Compound Indexes,
Unique Constraints, Index Trade-offs, and Query Execution Planning.
"""

import logging
from typing import Dict, Any, List
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.database import Database
from pymongo.errors import PyMongoError
import config

logger = logging.getLogger("scamshield.indexes")

INDEX_DEFINITIONS: Dict[str, List[Dict[str, Any]]] = {
    config.COLLECTION_JOBS: [
        {
            "keys": [("job_id", ASCENDING)],
            "name": "idx_jobs_job_id_unique",
            "unique": True,
            "sparse": True,
            "purpose": "Enforces deduplication on imported jobs; provides O(log N) point lookup by job ID.",
        },
        {
            "keys": [("fraudulent", ASCENDING), ("created_at", DESCENDING)],
            "name": "idx_jobs_fraudulent_created",
            "unique": False,
            "purpose": "Compound index supporting filtered range scans for ground-truth distribution and pagination.",
        },
        {
            "keys": [("title", TEXT), ("company_name", TEXT)],
            "name": "idx_jobs_title_company_text",
            "unique": False,
            "purpose": "Full-text search index for keyword queries on job titles and employer names.",
        },
    ],
    config.COLLECTION_PREDICTIONS: [
        {
            "keys": [("prediction_id", ASCENDING)],
            "name": "idx_pred_prediction_id_unique",
            "unique": True,
            "purpose": "Ensures unique identification and O(1)/O(log N) point lookups for saved predictions.",
        },
        {
            "keys": [("created_at", DESCENDING)],
            "name": "idx_pred_created_at_desc",
            "unique": False,
            "purpose": "Optimizes dashboard feeds and pagination sorted by latest prediction timestamp.",
        },
        {
            "keys": [("predicted_class", ASCENDING), ("created_at", DESCENDING)],
            "name": "idx_pred_class_created_compound",
            "unique": False,
            "purpose": "Compound index serving filtered queries by Fraudulent/Legitimate class without in-memory sorting.",
        },
        {
            "keys": [("model_name", ASCENDING)],
            "name": "idx_pred_model_name",
            "unique": False,
            "purpose": "Accelerates model usage aggregations and filtering by model architecture.",
        },
    ],
    config.COLLECTION_FEEDBACK: [
        {
            "keys": [("prediction_id", ASCENDING)],
            "name": "idx_feedback_prediction_id",
            "unique": False,
            "purpose": "References parent prediction document (foreign key pattern in NoSQL document models).",
        },
        {
            "keys": [("timestamp", DESCENDING)],
            "name": "idx_feedback_timestamp_desc",
            "unique": False,
            "purpose": "Chronological audit trail for user feedback submissions.",
        },
    ],
    config.COLLECTION_METRICS: [
        {
            "keys": [("model_name", ASCENDING), ("training_timestamp", DESCENDING)],
            "name": "idx_metrics_model_time_compound",
            "unique": False,
            "purpose": "Retrieves latest trained model weights, evaluation scores, and hyperparameters.",
        },
    ],
}


def create_all_indexes(db: Database) -> Dict[str, List[str]]:
    """
    Build all planned B-Tree and text indexes in MongoDB Atlas.
    Safe and idempotent (won't crash if an index already exists).
    """
    created_summary = {}

    for coll_name, index_list in INDEX_DEFINITIONS.items():
        collection = db[coll_name]
        created_summary[coll_name] = []
        for idx in index_list:
            keys = idx["keys"]
            name = idx["name"]
            unique = idx.get("unique", False)
            sparse = idx.get("sparse", False)
            try:
                collection.create_index(keys, name=name, unique=unique, sparse=sparse)
                created_summary[coll_name].append(f"{name} (keys={keys}, unique={unique})")
            except PyMongoError as pe:
                logger.warning(f"Index creation notice for {name} on {coll_name}: {pe}")
                created_summary[coll_name].append(f"{name} (already exists or skipped)")

    return created_summary


def get_index_explanations() -> List[Dict[str, str]]:
    """
    Returns structured documentation on index design rationale for CSE494 academic review.
    """
    explanations = []
    for coll_name, index_list in INDEX_DEFINITIONS.items():
        for idx in index_list:
            explanations.append({
                "collection": coll_name,
                "index_name": idx["name"],
                "keys": str(idx["keys"]),
                "unique": "Yes" if idx.get("unique") else "No",
                "purpose": idx["purpose"],
            })
    return explanations
