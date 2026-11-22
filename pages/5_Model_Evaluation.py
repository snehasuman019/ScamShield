"""
Page F: Model Evaluation & Explainability
ScamShield: AI-Powered Fake Job & Internship Detection System

Displays genuine evaluation metrics for trained models:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix heatmaps (True Negatives, False Positives, False Negatives, True Positives)
- Top TF-IDF linguistic indicators and Random Forest feature importances
- Academic discussion on data leakage, class imbalance, and model trade-offs
"""

import streamlit as st
import pandas as pd
import config
from ui_utils import apply_custom_css, render_header, render_sidebar
from ml.evaluate import (
    load_metrics_json,
    get_top_fraud_features_logistic,
    get_top_features_random_forest,
)
from analytics.reports import create_confusion_matrix_chart, create_feature_importance_chart

st.set_page_config(
    page_title="Model Evaluation | ScamShield",
    page_icon="🔬",
    layout="wide",
)

apply_custom_css()
render_sidebar()

render_header(
    title="Model Evaluation & Explainability",
    subtitle="Detailed statistical evaluation and feature importance for genuine trained machine learning pipelines",
    badge="ML Audit",
)

metrics_meta = load_metrics_json()

if not metrics_meta or "models" not in metrics_meta:
    st.warning("⚠️ No trained model metrics found.")
    st.info(
        "To train the models and generate genuine evaluation metrics, run the following terminal command from the project root:\n\n"
        "```bash\npython ml/train.py\n```"
    )
    st.stop()

# Training Dataset & Split Metadata
st.markdown("### 📋 Training Experiment Metadata")
m_info1, m_info2, m_info3, m_info4 = st.columns(4)

with m_info1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Dataset Source</div>
            <div style="color: #F8FAFC; font-weight: 600; font-size: 0.95rem; margin-top: 6px;">{metrics_meta.get('dataset_source', 'EMSCAD')}</div>
            <div class="metric-card-subtitle">{metrics_meta.get('total_samples', 0):,} Total Rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_info2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Train / Test Split</div>
            <div class="metric-card-value">80% / 20%</div>
            <div class="metric-card-subtitle">{metrics_meta.get('train_samples', 0):,} train / {metrics_meta.get('test_samples', 0):,} test</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_info3:
    target_info = metrics_meta.get("target_balance", {})
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Minority Fraud Prevalence</div>
            <div class="metric-card-value" style="color: #FF6B6B;">{target_info.get('fraud_prevalence_pct', 0)}%</div>
            <div class="metric-card-subtitle">{target_info.get('fraudulent', 0):,} fraud / {target_info.get('legitimate', 0):,} legit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_info4:
    timestamp_str = str(metrics_meta.get("training_timestamp", ""))[:19].replace("T", " ")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Training Timestamp</div>
            <div style="color: #2EC4B6; font-weight: 700; font-size: 1.1rem; margin-top: 8px;">{timestamp_str}</div>
            <div class="metric-card-subtitle">Pipeline Version {metrics_meta.get('model_version', '1.0.0')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Side-by-Side Model Evaluation Metrics
models_data = metrics_meta["models"]
m_a = models_data.get("logistic_regression", {})
m_b = models_data.get("random_forest", {})

st.markdown("### ⚖️ Side-by-Side Model Performance Comparison")

comp_table = [
    {"Metric": "Algorithm Family", "Model A: Logistic Regression": "Linear Model (Log-Odds)", "Model B: Random Forest": "Ensemble (Decision Trees)"},
    {"Metric": "Feature Input Space", "Model A: Logistic Regression": m_a.get("feature_set"), "Model B: Random Forest": m_b.get("feature_set")},
    {"Metric": "Accuracy", "Model A: Logistic Regression": f"{m_a.get('accuracy', 0)*100:.2f}%", "Model B: Random Forest": f"{m_b.get('accuracy', 0)*100:.2f}%"},
    {"Metric": "Precision (Positive Predictive Value)", "Model A: Logistic Regression": f"{m_a.get('precision', 0):.4f}", "Model B: Random Forest": f"{m_b.get('precision', 0):.4f}"},
    {"Metric": "Recall (Fraud Detection Rate)", "Model A: Logistic Regression": f"{m_a.get('recall', 0):.4f}", "Model B: Random Forest": f"{m_b.get('recall', 0):.4f}"},
    {"Metric": "F1-Score (Harmonic Mean)", "Model A: Logistic Regression": f"{m_a.get('f1_score', 0):.4f}", "Model B: Random Forest": f"{m_b.get('f1_score', 0):.4f}"},
    {"Metric": "ROC-AUC", "Model A: Logistic Regression": f"{m_a.get('roc_auc', 0):.4f}", "Model B: Random Forest": f"{m_b.get('roc_auc', 0):.4f}"},
]

st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

st.markdown("---")

# Confusion Matrix Heatmaps
st.markdown("### 🔲 Confusion Matrices (Held-Out Test Set)")
st.caption("Inspects True Negatives (correct legit), False Positives (legit flagged as fraud), False Negatives (missed fraud), and True Positives (caught fraud).")

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    fig_cm_a = create_confusion_matrix_chart(m_a.get("confusion_matrix", []), "Model A: Logistic Regression (TF-IDF)")
    st.plotly_chart(fig_cm_a, use_container_width=True)

with cm_col2:
    fig_cm_b = create_confusion_matrix_chart(m_b.get("confusion_matrix", []), "Model B: Random Forest (Structured)")
    st.plotly_chart(fig_cm_b, use_container_width=True)

st.markdown("---")

# Feature Importance and Explainability
st.markdown("### 🔍 Model Explainability & Key Discriminative Indicators")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.markdown("#### Top Fraud Linguistic Features (Logistic Regression)")
    st.caption("Tokens with highest positive regression weights, strongly signaling fraudulent recruitment:")
    top_words = get_top_fraud_features_logistic(top_n=10)
    fig_words = create_feature_importance_chart(top_words, "Top TF-IDF Fraud Tokens")
    st.plotly_chart(fig_words, use_container_width=True)

with feat_col2:
    st.markdown("#### Feature Importances (Random Forest)")
    st.caption("Relative Gini importance across structured metadata attributes:")
    top_struct = get_top_features_random_forest(top_n=10)
    fig_struct = create_feature_importance_chart(top_struct, "Gini Feature Importance")
    st.plotly_chart(fig_struct, use_container_width=True)

st.markdown("---")

# Methodological Discussion for Academic Evaluation
st.markdown("### 🎓 Academic Discussion & Limitations of Model Comparison")
st.markdown(
    """
    <div class="academic-box">
        <h4>1. Asymmetric Feature Spaces</h4>
        <p>
            <b>Model A (Logistic Regression)</b> operates exclusively on natural language text tokens via TF-IDF n-grams (up to 5,000 features). 
            <b>Model B (Random Forest)</b> evaluates structured metadata attributes (salary availability, logo presence, screening questions, employment type). 
            Consequently, performance disparities reflect the <i>informational density of the feature modality</i> rather than purely algorithmic efficiency. 
            In recruitment fraud, textual phrasing (promises of quick riches, wire transfer keywords) often carries stronger discriminative signal than coarse metadata.
        </p>
        <h4 style="margin-top: 10px;">2. Severe Class Imbalance Handling</h4>
        <p>
            With only ~4.3% fraudulent samples, a naive classifier predicting 'Legitimate' for every single posting would trivially achieve ~95.7% accuracy while detecting 0 scams (Recall = 0.0). 
            To prevent this, both models apply <code>class_weight='balanced'</code>, which inversely weights loss penalties according to class frequencies. 
            This prioritizes high <b>Recall</b> ({recall_val:.1f}%) to minimize dangerous False Negatives (missed fraudulent job postings).
        </p>
        <h4 style="margin-top: 10px;">3. Data Leakage Prevention</h4>
        <p>
            To guarantee methodological integrity, the 80/20 train/test split is performed <i>before</i> TF-IDF vocabulary fitting, one-hot encoding, or numeric feature imputation. 
            All scikit-learn transformers are encapsulated inside unified <code>Pipeline</code> objects, guaranteeing that test set statistics never bleed into training phases.
        </p>
    </div>
    """.format(recall_val=m_a.get('recall', 0)*100),
    unsafe_allow_html=True,
)
