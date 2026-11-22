"""
Unit tests for text preprocessing, feature extraction, and transparent rule engine.
"""

import pytest
import pandas as pd
from ml.preprocess import (
    clean_text,
    detect_suspicious_indicators,
    extract_structured_features,
    has_payment_indicators,
)


def test_clean_text_normalizes_html_and_urls():
    raw_html = "<p>Join our team! Apply at https://example.com/job &amp; earn <b>$$$</b></p>"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "url_link" in cleaned
    assert cleaned.islower()


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_detect_suspicious_indicators_upfront_fee():
    scam_text = "Applicants must pay a $50 registration fee and processing fee for background screening."
    indicators = detect_suspicious_indicators(scam_text)
    rule_ids = [ind["rule_id"] for ind in indicators]
    assert "RULE-UPFRONT-PAYMENT" in rule_ids
    assert any(ind["severity"] == "High" for ind in indicators)


def test_detect_suspicious_indicators_crypto_and_telegram():
    scam_text = "Salary paid in Bitcoin or USDT. Immediate interview on Telegram @RecruiterMark."
    indicators = detect_suspicious_indicators(scam_text)
    rule_ids = [ind["rule_id"] for ind in indicators]
    assert "RULE-UNCONVENTIONAL-PAYMENT" in rule_ids
    assert "RULE-OFF-PLATFORM-COMM" in rule_ids


def test_detect_suspicious_indicators_legitimate_clean():
    clean_text_sample = (
        "We are looking for a Senior Software Engineer with 5+ years of experience in Python and SQL. "
        "Comprehensive health insurance, 401(k), paid parental leave, and generous PTO."
    )
    indicators = detect_suspicious_indicators(
        clean_text_sample,
        has_company_logo=1,
        has_company_profile=True,
        has_questions=1,
    )
    # Should have no high-severity indicators
    high_sev = [i for i in indicators if i["severity"] == "High"]
    assert len(high_sev) == 0


def test_extract_structured_features():
    df_sample = pd.DataFrame([{
        "title": "Data Analyst",
        "salary_range": "$60,000 - $80,000",
        "company_profile": "A well-established analytics and data consultancy firm.",
        "description": "Analyze datasets and create dashboards using Python and Tableau.",
        "has_company_logo": 1,
        "has_questions": 1,
        "telecommuting": 0,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
    }])

    features = extract_structured_features(df_sample)
    assert features["has_salary_range"].iloc[0] == 1
    assert features["has_company_profile"].iloc[0] == 1
    assert features["has_company_logo"].iloc[0] == 1
    assert features["has_questions"].iloc[0] == 1
    assert features["employment_type"].iloc[0] == "Full-time"
    assert features["desc_word_count"].iloc[0] > 0
    assert features["has_payment_keywords"].iloc[0] == 0


def test_extract_structured_features_missing_values():
    df_empty = pd.DataFrame([{
        "title": "Mystery Job",
        "salary_range": None,
        "company_profile": None,
        "description": None,
    }])
    features = extract_structured_features(df_empty)
    assert features["has_salary_range"].iloc[0] == 0
    assert features["has_company_profile"].iloc[0] == 0
    assert features["has_company_logo"].iloc[0] == 0
    assert features["employment_type"].iloc[0] == "Unknown"
    assert features["desc_char_length"].iloc[0] == 0
