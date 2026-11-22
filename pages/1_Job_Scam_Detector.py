"""
Page B: Job Scam Detector
ScamShield: AI-Powered Fake Job & Internship Detection System

Allows users to input job or internship descriptions, configure structured attributes,
select between Logistic Regression and Random Forest models, inspect transparent red flags,
and persist assessments to MongoDB Atlas.
"""

import streamlit as st
import pandas as pd
import config
from ui_utils import apply_custom_css, render_header, render_sidebar
from database.connection import get_db, check_connection
from database.operations import save_prediction
from ml.predict import predict_job, is_model_available

st.set_page_config(
    page_title="Job Scam Detector | ScamShield",
    page_icon="🛡️",
    layout="wide",
)

apply_custom_css()
render_sidebar()

render_header(
    title="Job & Internship Scam Detector",
    subtitle="Evaluate recruitment advertisements using trained machine learning models and transparent rule heuristics",
    badge="Inference Engine",
)

# Sample Pre-filled Scenarios for Instant Testing
SAMPLE_SCAM = {
    "title": "Remote Data Entry / Administrative Assistant",
    "company": "",
    "description": (
        "URGENT REQUIREMENT: We are seeking motivated individuals for remote data entry and administrative duties. "
        "Earn $4,500 to $6,000 per week working only 2 hours per day from home. No previous experience needed! "
        "Candidates must purchase an initial training software package and background check fee ($75) via CashApp, Zelle, or Western Union. "
        "To start immediately without interview, contact our hiring manager on Telegram: @GlobalCareersDirect."
    ),
    "requirements": "Basic typing skills, internet connection, and instant payment for certification materials.",
    "has_salary": False,
    "has_logo": False,
    "has_questions": False,
    "telecommuting": True,
    "emp_type": "Part-time",
    "experience": "Not Applicable",
    "education": "Unspecified",
}

SAMPLE_LEGIT = {
    "title": "Senior Backend Software Engineer (Python / NoSQL)",
    "company": "CloudScale Systems International",
    "description": (
        "CloudScale Systems is seeking an experienced Backend Engineer to design scalable distributed microservices "
        "and optimize database query performance on MongoDB Atlas. Responsibilities include designing schema validation rules, "
        "monitoring execution stages via explain(), and building RESTful APIs. We offer comprehensive medical, dental, and vision insurance, "
        "competitive 401(k) matching, and annual educational stipends. Candidates will complete a code design review and technical team interview."
    ),
    "requirements": "B.S. in Computer Science or equivalent. 5+ years of Python experience, PyMongo, and distributed systems architecture.",
    "has_salary": True,
    "salary_range": "$125,000 - $155,000",
    "has_logo": True,
    "has_questions": True,
    "telecommuting": True,
    "emp_type": "Full-time",
    "experience": "Mid-Senior level",
    "education": "Bachelor's Degree",
}

# State Management for Form Fields
if "job_title" not in st.session_state:
    st.session_state.job_title = ""
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""
if "job_reqs" not in st.session_state:
    st.session_state.job_reqs = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Quick Presets Buttons
st.markdown("##### ⚡ Quick Presets (Click to Auto-fill Demo Data)")
btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 2])

with btn_c1:
    if st.button("🚨 Load Known Scam Sample", use_container_width=True):
        st.session_state.job_title = SAMPLE_SCAM["title"]
        st.session_state.company_name = SAMPLE_SCAM["company"]
        st.session_state.job_desc = SAMPLE_SCAM["description"]
        st.session_state.job_reqs = SAMPLE_SCAM["requirements"]
        st.session_state.analysis_result = None
        st.rerun()

with btn_c2:
    if st.button("✅ Load Legitimate Job Sample", use_container_width=True):
        st.session_state.job_title = SAMPLE_LEGIT["title"]
        st.session_state.company_name = SAMPLE_LEGIT["company"]
        st.session_state.job_desc = SAMPLE_LEGIT["description"]
        st.session_state.job_reqs = SAMPLE_LEGIT["requirements"]
        st.session_state.analysis_result = None
        st.rerun()

with btn_c3:
    if st.button("🔄 Clear Form", use_container_width=True):
        st.session_state.job_title = ""
        st.session_state.company_name = ""
        st.session_state.job_desc = ""
        st.session_state.job_reqs = ""
        st.session_state.analysis_result = None
        st.rerun()

st.markdown("---")

# Main Input Layout
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📝 Job Posting Content")

    job_title = st.text_input(
        "Job or Internship Title *",
        value=st.session_state.job_title,
        placeholder="e.g. Remote Data Entry Clerk, Frontend Engineering Intern",
    )

    company_name = st.text_input(
        "Hiring Organization / Employer Name (Optional)",
        value=st.session_state.company_name,
        placeholder="e.g. Acme Innovations Corp",
    )

    job_desc = st.text_area(
        "Job Description Text *",
        value=st.session_state.job_desc,
        height=220,
        placeholder="Paste the full job or internship description here...",
    )

    job_reqs = st.text_area(
        "Requirements & Qualifications (Optional)",
        value=st.session_state.job_reqs,
        height=100,
        placeholder="e.g. Required skills, degree prerequisites, years of experience...",
    )

with col_right:
    st.markdown("### ⚙️ Model & Metadata Configuration")

    # Model Selection Control
    model_choice = st.radio(
        "Select Machine Learning Model *",
        options=[
            ("logistic_regression", "Model A: Logistic Regression (TF-IDF NLP Text Vectorizer)"),
            ("random_forest", "Model B: Random Forest (Multi-Attribute & Structured Features)"),
        ],
        format_func=lambda x: x[1],
        help="Model A evaluates textual language and TF-IDF n-grams. Model B evaluates structural metadata (salary, logo, questions).",
    )
    selected_model_key = model_choice[0]

    # Verify model artifact is present
    if not is_model_available(selected_model_key):
        st.error(f"⚠️ Model artifact for '{selected_model_key}' is missing. Please run `python ml/train.py` from your terminal.")

    # Structured features (especially relevant for Random Forest)
    st.markdown("#### Structured Metadata Attributes")
    st.caption("These attributes feed into the Random Forest pipeline and heuristic checks.")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        has_salary = st.checkbox("Salary Disclosed in Posting?", value=bool(st.session_state.get("company_name", False)))
        has_logo = st.checkbox("Company Logo Present?", value=bool(st.session_state.get("company_name", False)))
    with col_meta2:
        has_questions = st.checkbox("Applicant Screening Questions?", value=False)
        is_telecommute = st.checkbox("Remote / Telecommuting Allowed?", value=True)

    emp_type = st.selectbox(
        "Employment Type",
        options=["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"],
        index=0,
    )

    experience_req = st.selectbox(
        "Required Experience Level",
        options=["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship", "Not Applicable", "Unknown"],
        index=0,
    )

    education_req = st.selectbox(
        "Required Education Level",
        options=["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Associate Degree", "Doctorate", "Unspecified", "Unknown"],
        index=0,
    )

st.markdown("---")

# Predict Button Trigger
btn_predict = st.button("🛡️ Run Fraud Risk Assessment", type="primary", use_container_width=True)

if btn_predict:
    if not job_desc.strip():
        st.error("Please provide a job description to perform the analysis.")
    else:
        try:
            structured_inputs = {
                "has_salary_range": 1 if has_salary else 0,
                "has_company_logo": 1 if has_logo else 0,
                "has_questions": 1 if has_questions else 0,
                "telecommuting": 1 if is_telecommute else 0,
                "employment_type": emp_type,
                "required_experience": experience_req,
                "required_education": education_req,
            }

            with st.spinner("Analyzing text patterns, indicators, and metadata..."):
                result = predict_job(
                    model_key=selected_model_key,
                    job_title=job_title,
                    job_description=job_desc,
                    company_name=company_name,
                    requirements=job_reqs,
                    structured_inputs=structured_inputs,
                )
                st.session_state.analysis_result = result
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# Display Analysis Results
if st.session_state.analysis_result:
    res = st.session_state.analysis_result

    st.markdown("## 📊 Assessment Results")

    is_fraud = res["predicted_class"] == "Fraudulent"
    risk_level = res["risk_level"]
    confidence_pct = res["confidence_percentage"]
    prob_fraud = res["probability_fraudulent"] * 100
    prob_legit = res["probability_legitimate"] * 100

    # Risk Banner
    if is_fraud or risk_level == "High":
        banner_color = "#E63946"
        badge_text = "HIGH RISK / POTENTIALLY FRAUDULENT"
        bg_tint = "rgba(230, 57, 70, 0.15)"
        border_col = "#E63946"
    elif risk_level == "Moderate":
        banner_color = "#F4A261"
        badge_text = "MODERATE RISK / EXERCISE CAUTION"
        bg_tint = "rgba(244, 162, 97, 0.15)"
        border_col = "#F4A261"
    else:
        banner_color = "#2A9D8F"
        badge_text = "LOW RISK / RESEMBLES LEGITIMATE POSTING"
        bg_tint = "rgba(42, 157, 143, 0.15)"
        border_col = "#2A9D8F"

    st.markdown(
        f"""
        <div style="background-color: {bg_tint}; border: 2px solid {border_col}; border-radius: 10px; padding: 20px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: {banner_color}; font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em;">{badge_text}</span>
                    <h2 style="color: #F8FAFC; margin: 6px 0 0 0; font-size: 1.8rem;">Predicted Class: {res['predicted_class']}</h2>
                </div>
                <div style="text-align: right;">
                    <span style="color: #94A3B8; font-size: 0.85rem;">Model Confidence</span>
                    <div style="color: #F8FAFC; font-size: 2.2rem; font-weight: 800;">{confidence_pct}</div>
                    <span style="color: #A8DADC; font-size: 0.8rem;">Risk Level: <b>{risk_level}</b></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability Distribution Meter
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown(f"**Fraud Probability**: `{prob_fraud:.1f}%`")
        st.progress(min(prob_fraud / 100.0, 1.0))
    with res_col2:
        st.markdown(f"**Legitimate Probability**: `{prob_legit:.1f}%`")
        st.progress(min(prob_legit / 100.0, 1.0))

    st.markdown("---")

    # Transparent Suspicious Indicators Section
    st.markdown("### 🔍 Transparent Suspicious Indicators Identified")
    indicators = res.get("detected_indicators", [])

    if indicators:
        st.warning(f"Found {len(indicators)} heuristic red flag(s) in this posting:")
        for idx, ind in enumerate(indicators, 1):
            sev_badge = "🔴 High Severity" if ind["severity"] == "High" else "🟡 Medium Severity" if ind["severity"] == "Medium" else "🔵 Informational"
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-left: 4px solid {COLOR_FRAUD if ind['severity']=='High' else '#0D9488'}; padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <b style="color: #F8FAFC;">{idx}. {ind['rule_name']}</b>
                        <span style="font-size: 0.8rem; color: #E2E8F0;">{sev_badge}</span>
                    </div>
                    <div style="color: #CBD5E1; font-size: 0.9rem; margin-top: 4px;">{ind['description']}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 6px;"><b>Matched Sample:</b> <code>{ind.get('matched_sample', '')}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("✓ No explicit suspicious keyword rules or severe heuristic flags were tripped in the posting text.")

    # Plain-English Limitations Disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Assessment Limitations & Verification Guidelines")
    st.info(res["limitations_disclaimer"])

    # Persistence to MongoDB Atlas
    st.markdown("---")
    st.markdown("### 💾 Save Analysis to MongoDB Atlas")
    st.caption("Stores the prediction record in the `predictions` collection for auditing, history tracking, and analytics.")

    db = get_db()
    is_conn, _ = check_connection()

    save_c1, save_c2 = st.columns([1, 2])
    with save_c1:
        if st.button("💾 Save Prediction to MongoDB", type="secondary", use_container_width=True):
            if not is_conn:
                st.error("MongoDB Atlas is not configured. Please set `MONGODB_URI` in `.env` to persist records.")
            else:
                pred_id = save_prediction(
                    job_title=job_title,
                    job_description=job_desc,
                    predicted_class=res["predicted_class"],
                    predicted_label=res["predicted_label"],
                    confidence=res["confidence"],
                    risk_level=res["risk_level"],
                    model_name=res["model_name"],
                    model_version=res["model_version"],
                    detected_indicators=res["detected_indicators"],
                    structured_inputs=res.get("structured_inputs"),
                    db=db,
                )
                if pred_id:
                    st.success(f"✓ Analysis successfully saved to MongoDB collection `predictions`! Identifier: `{pred_id}`")
                else:
                    st.error("Failed to persist prediction to MongoDB. Check database permissions.")
