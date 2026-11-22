"""
Analytics and reporting package for ScamShield.
Provides interactive Plotly visualizations powered by MongoDB Aggregations.
"""

from .reports import (
    create_prediction_distribution_chart,
    create_daily_trends_chart,
    create_model_usage_chart,
    create_feedback_distribution_chart,
    create_ground_truth_distribution_chart,
    create_model_comparison_chart,
    create_confusion_matrix_chart,
    create_feature_importance_chart,
)

__all__ = [
    "create_prediction_distribution_chart",
    "create_daily_trends_chart",
    "create_model_usage_chart",
    "create_feedback_distribution_chart",
    "create_ground_truth_distribution_chart",
    "create_model_comparison_chart",
    "create_confusion_matrix_chart",
    "create_feature_importance_chart",
]
