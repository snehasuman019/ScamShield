"""
Unit tests for ML prediction engine and inference validation.
"""

import pytest
from ml.predict import predict_job, is_model_available


def test_models_are_available():
    assert is_model_available("logistic_regression") is True
    assert is_model_available("random_forest") is True


def test_predict_job_logistic_regression():
    sample_text = (
        "We are hiring a Python Backend Engineer to work on scalable microservices. "
        "Must have experience with Docker, MongoDB, and REST APIs. Full-time position with 401(k)."
    )
    res = predict_job(
        model_key="logistic_regression",
        job_title="Backend Engineer",
        job_description=sample_text,
        company_name="Acme Tech Inc.",
    )

    assert "predicted_class" in res
    assert res["predicted_class"] in ("Fraudulent", "Legitimate")
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["probability_fraudulent"] <= 1.0
    assert 0.0 <= res["probability_legitimate"] <= 1.0
    assert res["risk_level"] in ("Low", "Moderate", "High")
    assert "limitations_disclaimer" in res
    assert "detected_indicators" in res
    assert res["model_key"] == "logistic_regression"


def test_predict_job_random_forest():
    sample_text = (
        "Join our team as a Customer Support Specialist. Provide phone and email support. "
        "High school diploma required. Health and retirement benefits included."
    )
    struct = {
        "has_salary_range": 1,
        "has_company_logo": 1,
        "has_questions": 1,
        "telecommuting": 0,
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "High School or equivalent",
    }
    res = predict_job(
        model_key="random_forest",
        job_title="Support Specialist",
        job_description=sample_text,
        company_name="Enterprise Corp",
        structured_inputs=struct,
    )

    assert res["predicted_class"] in ("Fraudulent", "Legitimate")
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["model_key"] == "random_forest"
    assert res["structured_inputs"]["has_salary_range"] == 1


def test_predict_job_empty_description_raises_error():
    with pytest.raises(ValueError, match="Job description cannot be empty"):
        predict_job(
            model_key="logistic_regression",
            job_title="Analyst",
            job_description="",
        )


def test_predict_job_unsupported_model_raises_error():
    with pytest.raises(Exception):
        predict_job(
            model_key="non_existent_deep_net",
            job_title="Analyst",
            job_description="Sample description text",
        )


def test_predict_job_flags_high_risk_on_scam_pattern():
    scam_text = (
        "URGENT: Earn $5000/week! No interview required! "
        "Pay $100 processing fee via Western Union or CashApp to begin immediately. Telegram @ScamAgent"
    )
    res = predict_job(
        model_key="logistic_regression",
        job_title="Data Entry",
        job_description=scam_text,
    )
    assert res["predicted_class"] == "Fraudulent"
    assert res["risk_level"] == "High"
    assert len(res["detected_indicators"]) > 0
