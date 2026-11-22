"""
PyMongo CRUD operations, Aggregation pipelines, and Query Performance analysis.
Implements working database operations for ScamShield:
- insert_one, insert_many
- find, find_one
- update_one, delete_one
- aggregate (distribution, trends, model stats)
- explain() execution plan inspection
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from pymongo.database import Database
from pymongo.errors import PyMongoError, DuplicateKeyError
import config
from .connection import get_db

logger = logging.getLogger("scamshield.operations")


# -------------------------------------------------------------------------
# Core PyMongo Primitives
# -------------------------------------------------------------------------

def insert_one_doc(collection_name: str, document: Dict[str, Any], db: Optional[Database] = None) -> Optional[Any]:
    """Insert a single document into the specified collection using insert_one()."""
    target_db = db if db is not None else get_db()
    if target_db is None:
        logger.warning(f"Database unavailable for insert_one into {collection_name}")
        return None
    try:
        result = target_db[collection_name].insert_one(document)
        return result.inserted_id
    except PyMongoError as pe:
        logger.error(f"PyMongo insert_one error in {collection_name}: {pe}")
        return None


def insert_many_docs(collection_name: str, documents: List[Dict[str, Any]], ordered: bool = False, db: Optional[Database] = None) -> List[Any]:
    """Insert multiple documents in bulk into the specified collection using insert_many()."""
    if not documents:
        return []
    target_db = db if db is not None else get_db()
    if target_db is None:
        logger.warning(f"Database unavailable for insert_many into {collection_name}")
        return []
    try:
        result = target_db[collection_name].insert_many(documents, ordered=ordered)
        return result.inserted_ids
    except PyMongoError as pe:
        logger.error(f"PyMongo insert_many error in {collection_name}: {pe}")
        return []


def find_docs(
    collection_name: str,
    query: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
    sort: Optional[List[Tuple[str, int]]] = None,
    limit: int = 0,
    skip: int = 0,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Retrieve matching documents from the specified collection using find()."""
    target_db = db if db is not None else get_db()
    if target_db is None:
        return []
    try:
        q = query or {}
        cursor = target_db[collection_name].find(q, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip > 0:
            cursor = cursor.skip(skip)
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)
    except PyMongoError as pe:
        logger.error(f"PyMongo find error in {collection_name}: {pe}")
        return []


def find_one_doc(
    collection_name: str,
    query: Dict[str, Any],
    projection: Optional[Dict[str, Any]] = None,
    db: Optional[Database] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single document matching the query using find_one()."""
    target_db = db if db is not None else get_db()
    if target_db is None:
        return None
    try:
        return target_db[collection_name].find_one(query, projection)
    except PyMongoError as pe:
        logger.error(f"PyMongo find_one error in {collection_name}: {pe}")
        return None


def update_one_doc(
    collection_name: str,
    filter_query: Dict[str, Any],
    update_data: Dict[str, Any],
    upsert: bool = False,
    db: Optional[Database] = None,
) -> bool:
    """Update a single document using update_one() ($set)."""
    target_db = db if db is not None else get_db()
    if target_db is None:
        return False
    try:
        # Wrap in $set if not already an update operator
        if not any(k.startswith("$") for k in update_data.keys()):
            update_body = {"$set": update_data}
        else:
            update_body = update_data

        result = target_db[collection_name].update_one(filter_query, update_body, upsert=upsert)
        return (result.modified_count > 0) or (result.upserted_id is not None)
    except PyMongoError as pe:
        logger.error(f"PyMongo update_one error in {collection_name}: {pe}")
        return False


def delete_one_doc(collection_name: str, filter_query: Dict[str, Any], db: Optional[Database] = None) -> bool:
    """Delete a single document using delete_one()."""
    target_db = db if db is not None else get_db()
    if target_db is None:
        return False
    try:
        result = target_db[collection_name].delete_one(filter_query)
        return result.deleted_count > 0
    except PyMongoError as pe:
        logger.error(f"PyMongo delete_one error in {collection_name}: {pe}")
        return False


def count_docs(collection_name: str, filter_query: Optional[Dict[str, Any]] = None, db: Optional[Database] = None) -> int:
    """Count documents matching filter using count_documents()."""
    target_db = db if db is not None else get_db()
    if target_db is None:
        return 0
    try:
        return target_db[collection_name].count_documents(filter_query or {})
    except PyMongoError as pe:
        logger.error(f"PyMongo count_documents error in {collection_name}: {pe}")
        return 0


def run_aggregation(collection_name: str, pipeline: List[Dict[str, Any]], db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Execute a MongoDB Aggregation Pipeline using aggregate()."""
    target_db = db if db is not None else get_db()
    if target_db is None:
        return []
    try:
        return list(target_db[collection_name].aggregate(pipeline))
    except PyMongoError as pe:
        logger.error(f"PyMongo aggregate error in {collection_name}: {pe}")
        return []


def explain_query_plan(
    collection_name: str,
    query: Dict[str, Any],
    sort: Optional[List[Tuple[str, int]]] = None,
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """
    Execute explain() on a query cursor to demonstrate index performance analysis.
    Returns details on execution stages (e.g. IXSCAN vs COLLSCAN), keys examined, and docs examined.
    """
    target_db = db if db is not None else get_db()
    if target_db is None:
        return {"error": "Database not connected"}
    try:
        cursor = target_db[collection_name].find(query)
        if sort:
            cursor = cursor.sort(sort)
        # Call explain with executionStats verbosity if supported
        try:
            explanation = cursor.explain()
        except Exception:
            explanation = {"fallback": "Explain cursor unsupported by mock or server configuration."}

        # Extract human-readable summary
        summary = {
            "collection": collection_name,
            "query": str(query),
            "sort": str(sort),
            "raw_explanation": explanation,
        }

        # Parse winning plan stage
        query_planner = explanation.get("queryPlanner", {})
        winning_plan = query_planner.get("winningPlan", {})
        summary["winning_plan_stage"] = winning_plan.get("stage", "UNKNOWN")
        summary["index_used"] = winning_plan.get("inputStage", {}).get("indexName", "None (Collection Scan)")

        # Execution stats if available
        exec_stats = explanation.get("executionStats", {})
        if exec_stats:
            summary["n_returned"] = exec_stats.get("nReturned", 0)
            summary["execution_time_ms"] = exec_stats.get("executionTimeMillis", 0)
            summary["total_keys_examined"] = exec_stats.get("totalKeysExamined", 0)
            summary["total_docs_examined"] = exec_stats.get("totalDocsExamined", 0)

        return summary
    except Exception as e:
        logger.error(f"Explain query error: {e}")
        return {"error": str(e)}


# -------------------------------------------------------------------------
# High-Level Business Helpers: Predictions
# -------------------------------------------------------------------------

def save_prediction(
    job_title: str,
    job_description: str,
    predicted_class: str,
    predicted_label: int,
    confidence: float,
    risk_level: str,
    model_name: str,
    model_version: str,
    detected_indicators: List[Dict[str, Any]],
    structured_inputs: Optional[Dict[str, Any]] = None,
    db: Optional[Database] = None,
) -> Optional[str]:
    """Save a job fraud prediction record to the predictions collection."""
    prediction_id = f"PRED-{uuid.uuid4().hex[:10].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    # Truncate description snippet for storage efficiency
    desc_snippet = job_description[:300] + ("..." if len(job_description) > 300 else "")

    document = {
        "prediction_id": prediction_id,
        "job_title": job_title or "Untitled Position",
        "job_description_snippet": desc_snippet,
        "predicted_class": predicted_class,
        "predicted_label": int(predicted_label),
        "confidence": round(float(confidence), 4),
        "risk_level": risk_level,
        "model_name": model_name,
        "model_version": model_version,
        "detected_indicators": detected_indicators,
        "structured_inputs": structured_inputs or {},
        "feedback_status": None,
        "created_at": timestamp,
    }

    inserted_id = insert_one_doc(config.COLLECTION_PREDICTIONS, document, db=db)
    return prediction_id if inserted_id else None


def get_saved_predictions(
    class_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    search_title: Optional[str] = None,
    limit: int = 50,
    skip: int = 0,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Retrieve saved predictions with optional filtering by class, model, or title."""
    query: Dict[str, Any] = {}
    if class_filter and class_filter.lower() != "all":
        query["predicted_class"] = class_filter
    if model_filter and model_filter.lower() != "all":
        query["model_name"] = model_filter
    if search_title:
        query["job_title"] = {"$regex": search_title, "$options": "i"}

    return find_docs(
        config.COLLECTION_PREDICTIONS,
        query=query,
        sort=[("created_at", -1)],
        limit=limit,
        skip=skip,
        db=db,
    )


def save_user_feedback(
    prediction_id: str,
    assessment: str,
    notes: Optional[str] = None,
    db: Optional[Database] = None,
) -> bool:
    """
    Save user feedback in the feedback collection and update the prediction document.
    Does NOT modify model ground-truth data (audit compliance).
    """
    feedback_id = f"FB-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    feedback_doc = {
        "feedback_id": feedback_id,
        "prediction_id": prediction_id,
        "user_assessment": assessment,
        "notes": notes,
        "timestamp": timestamp,
    }

    # Insert into feedback collection
    insert_res = insert_one_doc(config.COLLECTION_FEEDBACK, feedback_doc, db=db)
    if not insert_res:
        return False

    # Update prediction record
    update_one_doc(
        config.COLLECTION_PREDICTIONS,
        filter_query={"prediction_id": prediction_id},
        update_data={"feedback_status": assessment, "feedback_timestamp": timestamp},
        db=db,
    )
    return True


# -------------------------------------------------------------------------
# High-Level Business Helpers: Job Listings CRUD
# -------------------------------------------------------------------------

def create_job_ad(job_data: Dict[str, Any], db: Optional[Database] = None) -> Tuple[bool, str]:
    """Create a new job record in the jobs collection."""
    if "job_id" not in job_data or not job_data["job_id"]:
        job_data["job_id"] = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    if "created_at" not in job_data:
        job_data["created_at"] = datetime.now(timezone.utc).isoformat()

    # Validate essential fields
    if not job_data.get("title") or not job_data.get("description"):
        return False, "Job Title and Description are mandatory."

    inserted_id = insert_one_doc(config.COLLECTION_JOBS, job_data, db=db)
    if inserted_id:
        return True, str(job_data["job_id"])
    return False, "Failed to insert job into database."


def get_job_ads(
    search_term: Optional[str] = None,
    fraud_filter: Optional[int] = None,
    limit: int = 50,
    skip: int = 0,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Retrieve job listings with optional keyword search and fraudulent class filter."""
    query: Dict[str, Any] = {}
    if fraud_filter is not None:
        query["fraudulent"] = int(fraud_filter)
    if search_term:
        query["$or"] = [
            {"title": {"$regex": search_term, "$options": "i"}},
            {"company_name": {"$regex": search_term, "$options": "i"}},
        ]

    return find_docs(
        config.COLLECTION_JOBS,
        query=query,
        sort=[("created_at", -1)],
        limit=limit,
        skip=skip,
        db=db,
    )


def update_job_ad(job_id: Any, updated_fields: Dict[str, Any], db: Optional[Database] = None) -> bool:
    """Update fields of an existing job record."""
    updated_fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    return update_one_doc(
        config.COLLECTION_JOBS,
        filter_query={"job_id": job_id},
        update_data=updated_fields,
        db=db,
    )


def delete_job_ad(job_id: Any, db: Optional[Database] = None) -> bool:
    """Delete a job record by its job_id."""
    return delete_one_doc(
        config.COLLECTION_JOBS,
        filter_query={"job_id": job_id},
        db=db,
    )


# -------------------------------------------------------------------------
# High-Level Business Helpers: Model Metrics
# -------------------------------------------------------------------------

def save_model_evaluation_metrics(metrics_dict: Dict[str, Any], db: Optional[Database] = None) -> bool:
    """Save model evaluation metrics to the model_metrics collection in MongoDB."""
    if "training_timestamp" not in metrics_dict:
        metrics_dict["training_timestamp"] = datetime.now(timezone.utc).isoformat()
    res = insert_one_doc(config.COLLECTION_METRICS, metrics_dict, db=db)
    return res is not None


def get_latest_model_metrics(model_name: Optional[str] = None, db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Retrieve latest metrics for trained models."""
    query = {"model_name": model_name} if model_name else {}
    return find_docs(
        config.COLLECTION_METRICS,
        query=query,
        sort=[("training_timestamp", -1)],
        limit=10,
        db=db,
    )


# -------------------------------------------------------------------------
# Aggregation Pipelines for Analytics
# -------------------------------------------------------------------------

def get_predictions_class_distribution(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """
    MongoDB Aggregation Pipeline: Calculates prediction distribution by class.
    Demonstrates $group, $count, $sort in MongoDB Aggregation Framework.
    """
    pipeline = [
        {"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    return run_aggregation(config.COLLECTION_PREDICTIONS, pipeline, db=db)


def get_predictions_daily_trend(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """
    MongoDB Aggregation Pipeline: Groups predictions by calendar day.
    Demonstrates $substrCP/$dateToString date parsing in aggregation.
    """
    pipeline = [
        {
            "$project": {
                "day": {"$substrCP": ["$created_at", 0, 10]},
                "predicted_class": 1,
            }
        },
        {
            "$group": {
                "_id": {"day": "$day", "class": "$predicted_class"},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.day": 1}},
    ]
    return run_aggregation(config.COLLECTION_PREDICTIONS, pipeline, db=db)


def get_model_usage_aggregation(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """
    MongoDB Aggregation Pipeline: Groups predictions by model used.
    """
    pipeline = [
        {"$group": {"_id": "$model_name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    return run_aggregation(config.COLLECTION_PREDICTIONS, pipeline, db=db)


def get_feedback_breakdown(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """
    MongoDB Aggregation Pipeline: Groups user feedback by assessment category.
    """
    pipeline = [
        {"$group": {"_id": "$user_assessment", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    return run_aggregation(config.COLLECTION_FEEDBACK, pipeline, db=db)


def get_jobs_ground_truth_distribution(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """
    MongoDB Aggregation Pipeline: Calculates class distribution of stored jobs.
    """
    pipeline = [
        {"$group": {"_id": "$fraudulent", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    return run_aggregation(config.COLLECTION_JOBS, pipeline, db=db)


def get_dashboard_counts(db: Optional[Database] = None) -> Dict[str, int]:
    """Retrieve high-level counts for Home Dashboard cards."""
    total_jobs = count_docs(config.COLLECTION_JOBS, {}, db=db)
    total_predictions = count_docs(config.COLLECTION_PREDICTIONS, {}, db=db)
    fraud_predictions = count_docs(config.COLLECTION_PREDICTIONS, {"predicted_class": "Fraudulent"}, db=db)
    legit_predictions = count_docs(config.COLLECTION_PREDICTIONS, {"predicted_class": "Legitimate"}, db=db)

    return {
        "total_jobs": total_jobs,
        "total_predictions": total_predictions,
        "fraud_predictions": fraud_predictions,
        "legit_predictions": legit_predictions,
    }
