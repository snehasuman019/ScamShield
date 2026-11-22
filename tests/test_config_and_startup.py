"""
Unit tests for configuration verification, placeholder URI detection, and Streamlit page imports.
"""

import pytest
import config


def test_config_defaults():
    assert config.DB_NAME == "scamshield_db"
    assert config.COLLECTION_JOBS == "jobs"
    assert config.COLLECTION_PREDICTIONS == "predictions"
    assert config.COLLECTION_FEEDBACK == "feedback"
    assert config.COLLECTION_METRICS == "model_metrics"
    assert config.RISK_HIGH_THRESHOLD == 0.70
    assert config.RISK_MEDIUM_THRESHOLD == 0.40


def test_is_mongodb_configured_detection():
    # Placeholder strings should return False
    config.MONGODB_URI = "mongodb+srv://yourusername:yourpassword@cluster0.abcde.mongodb.net"
    assert config.is_mongodb_configured() is False

    # Empty string should return False
    config.MONGODB_URI = ""
    assert config.is_mongodb_configured() is False

    # Valid format without placeholder should return True
    config.MONGODB_URI = "mongodb+srv://student_real:SecretPassword99@mycluster.net/scamshield_db"
    assert config.is_mongodb_configured() is True


def test_stream_pages_import_cleanly():
    """Verify that all application scripts and pages compile without SyntaxError or missing module dependencies."""
    import app
    import pages
    assert app is not None
