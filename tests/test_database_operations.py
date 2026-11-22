"""
Unit tests for PyMongo database operations, CRUD functions, and Aggregation pipelines.
Uses mongomock for an isolated, safe test database that never touches the user's Atlas cluster.
"""

import pytest
import mongomock
import config
from database.operations import (
    insert_one_doc,
    insert_many_docs,
    find_docs,
    find_one_doc,
    update_one_doc,
    delete_one_doc,
    count_docs,
    save_prediction,
    get_saved_predictions,
    save_user_feedback,
    create_job_ad,
    update_job_ad,
    delete_job_ad,
    get_predictions_class_distribution,
    get_model_usage_aggregation,
    explain_query_plan,
)
from database.indexes import create_all_indexes


@pytest.fixture
def mock_db():
    """Provides a fresh isolated in-memory mock database."""
    client = mongomock.MongoClient()
    db = client[config.DB_NAME]
    return db


def test_insert_one_and_find_one(mock_db):
    doc = {"title": "Test Engineer", "department": "QA", "fraudulent": 0}
    inserted_id = insert_one_doc("jobs", doc, db=mock_db)
    assert inserted_id is not None

    found = find_one_doc("jobs", {"title": "Test Engineer"}, db=mock_db)
    assert found is not None
    assert found["department"] == "QA"


def test_insert_many_and_count(mock_db):
    docs = [
        {"job_id": 101, "title": "Job A", "fraudulent": 0},
        {"job_id": 102, "title": "Job B", "fraudulent": 1},
        {"job_id": 103, "title": "Job C", "fraudulent": 0},
    ]
    ids = insert_many_docs("jobs", docs, db=mock_db)
    assert len(ids) == 3

    total = count_docs("jobs", {}, db=mock_db)
    assert total == 3

    fraud_count = count_docs("jobs", {"fraudulent": 1}, db=mock_db)
    assert fraud_count == 1


def test_update_one(mock_db):
    doc = {"job_id": 201, "title": "Junior Dev", "salary_range": None}
    insert_one_doc("jobs", doc, db=mock_db)

    updated = update_one_doc(
        "jobs",
        filter_query={"job_id": 201},
        update_data={"salary_range": "$65,000 - $80,000"},
        db=mock_db,
    )
    assert updated is True

    retrieved = find_one_doc("jobs", {"job_id": 201}, db=mock_db)
    assert retrieved["salary_range"] == "$65,000 - $80,000"


def test_delete_one(mock_db):
    doc = {"job_id": 301, "title": "Temporary Job"}
    insert_one_doc("jobs", doc, db=mock_db)
    assert count_docs("jobs", {"job_id": 301}, db=mock_db) == 1

    deleted = delete_one_doc("jobs", {"job_id": 301}, db=mock_db)
    assert deleted is True
    assert count_docs("jobs", {"job_id": 301}, db=mock_db) == 0


def test_save_and_retrieve_prediction(mock_db):
    pred_id = save_prediction(
        job_title="Remote Assistant",
        job_description="Sample text for testing prediction save",
        predicted_class="Fraudulent",
        predicted_label=1,
        confidence=0.95,
        risk_level="High",
        model_name="Logistic Regression",
        model_version="1.0.0",
        detected_indicators=[{"rule_name": "Upfront Fee", "severity": "High"}],
        db=mock_db,
    )
    assert pred_id is not None
    assert pred_id.startswith("PRED-")

    preds = get_saved_predictions(class_filter="Fraudulent", limit=10, db=mock_db)
    assert len(preds) == 1
    assert preds[0]["prediction_id"] == pred_id
    assert preds[0]["predicted_class"] == "Fraudulent"


def test_save_user_feedback(mock_db):
    pred_id = save_prediction(
        job_title="DevOps Engineer",
        job_description="Cloud infrastructure and CI/CD pipelines",
        predicted_class="Legitimate",
        predicted_label=0,
        confidence=0.91,
        risk_level="Low",
        model_name="Random Forest",
        model_version="1.0.0",
        detected_indicators=[],
        db=mock_db,
    )
    assert pred_id is not None

    fb_success = save_user_feedback(
        prediction_id=pred_id,
        assessment="Accurate",
        notes="Verified directly on official corporate careers site",
        db=mock_db,
    )
    assert fb_success is True

    # Check feedback was recorded in feedback collection
    fb_doc = find_one_doc("feedback", {"prediction_id": pred_id}, db=mock_db)
    assert fb_doc is not None
    assert fb_doc["user_assessment"] == "Accurate"

    # Check prediction document updated with feedback status
    updated_pred = find_one_doc("predictions", {"prediction_id": pred_id}, db=mock_db)
    assert updated_pred["feedback_status"] == "Accurate"


def test_job_crud_helpers(mock_db):
    job_data = {
        "title": "Machine Learning Engineer",
        "company_name": "AI Innovations Lab",
        "description": "Design neural networks and recommendation engines.",
        "fraudulent": 0,
    }
    success, assigned_id = create_job_ad(job_data, db=mock_db)
    assert success is True
    assert assigned_id is not None

    # Update
    up_ok = update_job_ad(assigned_id, {"company_name": "AI Innovations Group"}, db=mock_db)
    assert up_ok is True

    # Delete
    del_ok = delete_job_ad(assigned_id, db=mock_db)
    assert del_ok is True


def test_aggregation_pipelines(mock_db):
    # Insert multiple predictions
    docs = [
        {"prediction_id": "P1", "predicted_class": "Fraudulent", "model_name": "Logistic Regression"},
        {"prediction_id": "P2", "predicted_class": "Fraudulent", "model_name": "Logistic Regression"},
        {"prediction_id": "P3", "predicted_class": "Legitimate", "model_name": "Random Forest"},
    ]
    insert_many_docs("predictions", docs, db=mock_db)

    # Aggregation by class
    dist = get_predictions_class_distribution(db=mock_db)
    dist_map = {item["_id"]: item["count"] for item in dist}
    assert dist_map.get("Fraudulent") == 2
    assert dist_map.get("Legitimate") == 1

    # Aggregation by model
    model_dist = get_model_usage_aggregation(db=mock_db)
    model_map = {item["_id"]: item["count"] for item in model_dist}
    assert model_map.get("Logistic Regression") == 2
    assert model_map.get("Random Forest") == 1


def test_explain_query_structure(mock_db):
    insert_one_doc("predictions", {"prediction_id": "P99", "predicted_class": "Fraudulent"}, db=mock_db)
    res = explain_query_plan("predictions", {"predicted_class": "Fraudulent"}, db=mock_db)
    assert "collection" in res
    assert res["collection"] == "predictions"
    assert "winning_plan_stage" in res
