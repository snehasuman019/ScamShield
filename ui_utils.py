"""
Shared UI Utilities and styling components for ScamShield Streamlit application.
Ensures consistent, responsive dark navy styling with restrained teal accents.
"""

import streamlit as st
import config
from database.connection import check_connection

CUSTOM_CSS = """
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-card-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-card-subtitle {
        color: #0D9488;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Alert / Risk Badges */
    .badge-fraud {
        background-color: rgba(230, 57, 70, 0.2);
        color: #FF6B6B;
        border: 1px solid #E63946;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-legit {
        background-color: rgba(42, 157, 143, 0.2);
        color: #2EC4B6;
        border: 1px solid #2A9D8F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: rgba(69, 123, 157, 0.2);
        color: #A8DADC;
        border: 1px solid #457B9D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Page Header Banner */
    .page-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-left: 4px solid #0D9488;
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 24px;
    }
    .page-header h1 {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .page-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 6px 0 0 0;
    }

    /* Educational Callout */
    .academic-box {
        background-color: #0F172A;
        border: 1px solid #1E3A8A;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .academic-box h4 {
        color: #60A5FA;
        margin: 0 0 6px 0;
        font-size: 0.95rem;
    }
    .academic-box p {
        color: #CBD5E1;
        font-size: 0.85rem;
        margin: 0;
    }
</style>
"""


def apply_custom_css():
    """Apply uniform custom styles to Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge: str = "ScamShield v1.0"):
    """Render a consistent header banner."""
    html_code = f"""
    <div class="page-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>{title}</h1>
            <span class="badge-neutral">{badge}</span>
        </div>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def render_sidebar():
    """Render consistent sidebar information with live MongoDB status."""
    with st.sidebar:
        st.markdown("### 🛡️ **ScamShield**")
        st.caption("AI-Powered Job & Internship Scam Detection")
        st.markdown("---")

        # Live MongoDB Connection Status
        is_conn, msg = check_connection()
        if is_conn:
            st.success("🟢 **MongoDB Atlas**: Connected")
            st.caption(f"Cluster DB: `{config.DB_NAME}`")
        else:
            st.warning("🟠 **MongoDB Atlas**: Offline / Unconfigured")
            st.caption("Operating in Local / Demo Mode")

        st.markdown("---")
        st.markdown("**Academic Course**:")
        st.caption("CSE494: Intelligent NoSQL Databases")
        st.caption("Focus: BSON Modeling, PyMongo, B-Tree Indexes, Aggregations, ML Analytics")

        st.markdown("---")
        st.caption("Created for Pair Programming Demonstration")
