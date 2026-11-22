"""
ScamShield: AI-Powered Fake Job & Internship Detection System
Home Dashboard (app.py)

University Course: CSE494 - Intelligent NoSQL Databases
Integrates:
- Streamlit UI
- MongoDB Atlas (PyMongo)
- Genuine Machine Learning Models (Logistic Regression & Random Forest)
- Real-time aggregation pipelines
"""

import streamlit as st
import pandas as pd
import config
from ui_utils import apply_custom_css, render_header, render_sidebar
from database.connection import check_connection, get_db
from database.operations import get_dashboard_counts, get_saved_predictions, get_predictions_class_distribution
from ml.evaluate import load_metrics_json

st.set_page_config(
    page_title="ScamShield | AI Job Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar()

# Page Header
render_header(
    title="ScamShield Dashboard",
    subtitle="AI-Powered Fake Job & Internship Detection System with MongoDB Atlas Integration",
    badge="CSE494 Project",
)

# 1. MongoDB Connection & Status Banner
db = get_db()
is_conn, conn_msg = check_connection()

if is_conn:
    st.markdown(
        f"""
        <div style="background-color: rgba(13, 148, 136, 0.15); border: 1px solid #0D9488; border-radius: 8px; padding: 10px 16px; margin-bottom: 20px;">
            <span style="color: #2EC4B6; font-weight: 600;">✓ MongoDB Atlas Connected</span>
            <span style="color: #94A3B8; margin-left: 10px;">Database: <code>{config.DB_NAME}</code> | Real-time PyMongo Aggregations Enabled</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="background-color: rgba(234, 88, 12, 0.15); border: 1px solid #EA580C; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;">
            <span style="color: #FB923C; font-weight: 600;">ℹ MongoDB Atlas Notice:</span>
            <span style="color: #CBD5E1; margin-left: 6px;">
                Atlas credentials not configured in <code>.env</code>. The application is operating in local/offline mode.
                All Machine Learning classification features function locally, while database history persists when Atlas is connected.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 2. Key Metrics Cards
counts = get_dashboard_counts(db=db)
metrics_meta = load_metrics_json()

# If jobs in DB is 0, check local dataset count for reference
dataset_count = 17880 if config.DATASET_CSV_PATH.exists() else 0
displayed_jobs = counts["total_jobs"] if counts["total_jobs"] > 0 else dataset_count
job_subtext = "Atlas Stored Records" if counts["total_jobs"] > 0 else "Local EMSCAD Dataset"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Job Postings Available</div>
            <div class="metric-card-value">{displayed_jobs:,}</div>
            <div class="metric-card-subtitle">{job_subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Saved Predictions</div>
            <div class="metric-card-value">{counts['total_predictions']:,}</div>
            <div class="metric-card-subtitle">In predictions collection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Flagged Fraudulent</div>
            <div class="metric-card-value" style="color: #FF6B6B;">{counts['fraud_predictions']:,}</div>
            <div class="metric-card-subtitle">Identified high-risk listings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Assessed Legitimate</div>
            <div class="metric-card-value" style="color: #2EC4B6;">{counts['legit_predictions']:,}</div>
            <div class="metric-card-subtitle">Normal recruitment patterns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 3. Model Performance Summary
st.markdown("### 🤖 Trained Machine Learning Models Status")

if metrics_meta and "models" in metrics_meta:
    m_models = metrics_meta["models"]
    m_a = m_models.get("logistic_regression", {})
    m_b = m_models.get("random_forest", {})

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #2A9D8F;">
                <div style="display: flex; justify-content: space-between;">
                    <b style="color: #F8FAFC; font-size: 1.05rem;">Model A: Logistic Regression</b>
                    <span class="badge-legit">NLP Text Classifier</span>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 6px;">Features: TF-IDF N-Grams (Title, Description, Requirements)</div>
                <div style="display: flex; gap: 20px; margin-top: 12px;">
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">Accuracy</span><br><b style="font-size: 1.2rem; color: #F8FAFC;">{m_a.get('accuracy', 0)*100:.1f}%</b></div>
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">F1-Score</span><br><b style="font-size: 1.2rem; color: #2EC4B6;">{m_a.get('f1_score', 0):.3f}</b></div>
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">Recall (Detection)</span><br><b style="font-size: 1.2rem; color: #F8FAFC;">{m_a.get('recall', 0)*100:.1f}%</b></div>
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">ROC-AUC</span><br><b style="font-size: 1.2rem; color: #A8DADC;">{m_a.get('roc_auc', 0):.3f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #457B9D;">
                <div style="display: flex; justify-content: space-between;">
                    <b style="color: #F8FAFC; font-size: 1.05rem;">Model B: Random Forest</b>
                    <span class="badge-neutral">Multi-Attribute Classifier</span>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 6px;">Features: Salary, Logo, Screening, Education, Upfront Keywords</div>
                <div style="display: flex; gap: 20px; margin-top: 12px;">
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">Accuracy</span><br><b style="font-size: 1.2rem; color: #F8FAFC;">{m_b.get('accuracy', 0)*100:.1f}%</b></div>
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">F1-Score</span><br><b style="font-size: 1.2rem; color: #A8DADC;">{m_b.get('f1_score', 0):.3f}</b></div>
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">Recall (Detection)</span><br><b style="font-size: 1.2rem; color: #F8FAFC;">{m_b.get('recall', 0)*100:.1f}%</b></div>
                    <div><span style="color: #94A3B8; font-size: 0.8rem;">ROC-AUC</span><br><b style="font-size: 1.2rem; color: #A8DADC;">{m_b.get('roc_auc', 0):.3f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Model evaluation metrics not yet generated. Run `python ml/train.py` from the terminal to train both models.")

st.markdown("---")

# 4. Recent Prediction History Quick View
st.markdown("### 📋 Recent Analyses from MongoDB")
recent_preds = get_saved_predictions(limit=5, db=db)

if recent_preds:
    pred_rows = []
    for p in recent_preds:
        pred_rows.append({
            "Prediction ID": p.get("prediction_id"),
            "Job Title": p.get("job_title", "Untitled"),
            "Predicted Class": p.get("predicted_class"),
            "Confidence": f"{round(float(p.get('confidence', 0)) * 100, 1)}%",
            "Risk Level": p.get("risk_level", "Unknown"),
            "Model Used": p.get("model_name"),
            "Date": str(p.get("created_at", ""))[:19].replace("T", " "),
        })
    df_recent = pd.DataFrame(pred_rows)
    st.dataframe(df_recent, use_container_width=True, hide_index=True)
else:
    st.info("No predictions recorded yet. Navigate to **Job Scam Detector** in the sidebar to scan a job posting and save your first analysis.")

# 5. Course Syllabus Alignment Box
st.markdown(
    """
    <div class="academic-box">
        <h4>🎓 CSE494: Intelligent NoSQL Databases - Concept Integration</h4>
        <p>
            This system bridges modern NoSQL document databases with genuine machine learning:<br>
            • <b>Flexible BSON Documents</b>: Multi-collection schema for jobs, predictions, feedback, and model metrics.<br>
            • <b>MongoDB Aggregation Framework</b>: <code>$group</code>, <code>$substrCP</code>, and <code>$sort</code> pipelines power real-time analytics.<br>
            • <b>Indexing & <code>explain()</code></b>: Unique B-tree indexes for deduplication and compound indexes evaluated via execution plan analysis.<br>
            • <b>PyMongo Driver Integration</b>: Batched bulk inserts (<code>insert_many</code>), resilient error handling, and query projections.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
