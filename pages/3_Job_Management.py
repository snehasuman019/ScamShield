"""
Page D: Job Records Management (CRUD Operations)
ScamShield: AI-Powered Fake Job & Internship Detection System

Demonstrates core CSE494 syllabus topic: Full CRUD (Create, Read, Update, Delete)
operations on MongoDB Atlas BSON documents via PyMongo.
"""

import streamlit as st
import pandas as pd
import config
from ui_utils import apply_custom_css, render_header, render_sidebar
from database.connection import get_db, check_connection
from database.operations import (
    create_job_ad,
    get_job_ads,
    update_job_ad,
    delete_job_ad,
    count_docs,
)
from import_data import import_dataset

st.set_page_config(
    page_title="Job Records Management | ScamShield",
    page_icon="💼",
    layout="wide",
)

apply_custom_css()
render_sidebar()

render_header(
    title="Job Records Management (CRUD)",
    subtitle="Create, Read, Update, and Delete recruitment documents in MongoDB Atlas",
    badge="PyMongo CRUD",
)

db = get_db()
is_conn, _ = check_connection()

if not is_conn:
    st.error("⚠️ MongoDB Atlas is not connected. CRUD operations require an active MongoDB connection. Configure `MONGODB_URI` in `.env`.")

# Tabbed Interface for CRUD operations
tab_read, tab_create, tab_update, tab_delete, tab_import = st.tabs([
    "📖 1. View & Search Jobs (Read)",
    "➕ 2. Add New Job Ad (Create)",
    "✏️ 3. Edit Existing Job (Update)",
    "🗑️ 4. Remove Job Ad (Delete)",
    "⚡ 5. Dataset Import to MongoDB",
])

# -------------------------------------------------------------------------
# TAB 1: READ / SEARCH
# -------------------------------------------------------------------------
with tab_read:
    st.markdown("### 🔍 Search Stored Job Records")
    r_col1, r_col2, r_col3 = st.columns([3, 2, 1])

    with r_col1:
        search_query = st.text_input("Search Title or Company Name", placeholder="e.g. Software Engineer, Marketing...", key="search_read")
    with r_col2:
        class_filter_opt = st.selectbox("Ground-Truth Filter", ["All", "Legitimate (0)", "Fraudulent (1)"], key="class_read")
    with r_col3:
        page_limit = st.selectbox("Page Limit", [10, 25, 50, 100], index=1, key="limit_read")

    fraud_val = None
    if "Legitimate" in class_filter_opt:
        fraud_val = 0
    elif "Fraudulent" in class_filter_opt:
        fraud_val = 1

    jobs_list = []
    if is_conn:
        jobs_list = get_job_ads(
            search_term=search_query if search_query else None,
            fraud_filter=fraud_val,
            limit=page_limit,
            db=db,
        )

    if not jobs_list:
        st.info("No job listings found in MongoDB matching your query. Switch to the 'Dataset Import' tab to load records from the EMSCAD dataset.")
    else:
        st.markdown(f"**Retrieved {len(jobs_list)} listing(s)**:")
        display_rows = []
        for j in jobs_list:
            display_rows.append({
                "Job ID": j.get("job_id"),
                "Title": j.get("title", "Untitled"),
                "Company": j.get("company_name") or "Unspecified",
                "Employment Type": j.get("employment_type") or "N/A",
                "Ground Truth": "Fraudulent (1)" if j.get("fraudulent") == 1 else "Legitimate (0)",
                "Logo": "Yes" if j.get("has_company_logo") == 1 else "No",
                "Created": str(j.get("created_at", ""))[:10],
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------------
# TAB 2: CREATE
# -------------------------------------------------------------------------
with tab_create:
    st.markdown("### ➕ Create New Job Listing in MongoDB")
    st.caption("Inserts a structured BSON document into the `jobs` collection via `insert_one()`.")

    with st.form("create_job_form", clear_on_submit=True):
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            new_title = st.text_input("Job Title *", placeholder="e.g. Cloud Security Analyst")
            new_company = st.text_input("Company Name", placeholder="e.g. SecureTech Solutions")
            new_salary = st.text_input("Salary Range", placeholder="e.g. $80,000 - $100,000")
            new_emp_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"])
        with c_col2:
            new_exp = st.selectbox("Required Experience", ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship", "Not Applicable"])
            new_edu = st.selectbox("Required Education", ["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Associate Degree", "Doctorate", "Unspecified"])
            new_fraud = st.selectbox("Ground-Truth Label", [(0, "0: Legitimate Posting"), (1, "1: Fraudulent / Scam Posting")], format_func=lambda x: x[1])
            new_logo = st.checkbox("Has Company Logo?", value=True)
            new_questions = st.checkbox("Has Screening Questions?", value=True)

        new_desc = st.text_area("Job Description *", height=150, placeholder="Enter detailed duties and role description...")
        new_reqs = st.text_area("Requirements", height=80, placeholder="Required skills and certifications...")

        btn_submit_create = st.form_submit_button("Insert Job into MongoDB Atlas", type="primary")

        if btn_submit_create:
            if not is_conn:
                st.error("Cannot insert: MongoDB Atlas is not connected.")
            elif not new_title.strip() or not new_desc.strip():
                st.error("Job Title and Description are required fields.")
            else:
                job_doc = {
                    "title": new_title.strip(),
                    "company_name": new_company.strip() or None,
                    "salary_range": new_salary.strip() or None,
                    "employment_type": new_emp_type,
                    "required_experience": new_exp,
                    "required_education": new_edu,
                    "description": new_desc.strip(),
                    "requirements": new_reqs.strip() or None,
                    "has_company_logo": 1 if new_logo else 0,
                    "has_questions": 1 if new_questions else 0,
                    "telecommuting": 0,
                    "fraudulent": new_fraud[0],
                    "source": "Manual Entry (CRUD UI)",
                }
                success, assigned_id = create_job_ad(job_doc, db=db)
                if success:
                    st.success(f"✓ Job document successfully created in MongoDB! Assigned Job ID: `{assigned_id}`")
                else:
                    st.error(f"Failed to insert job: {assigned_id}")


# -------------------------------------------------------------------------
# TAB 3: UPDATE
# -------------------------------------------------------------------------
with tab_update:
    st.markdown("### ✏️ Edit Existing Job Record")
    st.caption("Demonstrates PyMongo `update_one()` with `$set` operator.")

    edit_search = st.text_input("Search Job ID or Title to Edit:", placeholder="e.g. Cloud Security Analyst", key="edit_search_term")
    matching_jobs = []
    if is_conn and edit_search:
        matching_jobs = get_job_ads(search_term=edit_search, limit=10, db=db)

    if matching_jobs:
        job_choices = {str(j.get("job_id")): j for j in matching_jobs}
        target_id = st.selectbox("Select Target Job:", options=list(job_choices.keys()), format_func=lambda x: f"{x} - {job_choices[x].get('title')}")
        target_doc = job_choices[target_id]

        with st.form("edit_job_form"):
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                edit_title = st.text_input("Job Title", value=target_doc.get("title", ""))
                edit_company = st.text_input("Company Name", value=target_doc.get("company_name") or "")
                edit_salary = st.text_input("Salary Range", value=target_doc.get("salary_range") or "")
            with e_col2:
                edit_emp_type = st.text_input("Employment Type", value=target_doc.get("employment_type") or "Full-time")
                edit_exp = st.text_input("Required Experience", value=target_doc.get("required_experience") or "Entry level")
                edit_fraud = st.selectbox(
                    "Ground-Truth Label",
                    [0, 1],
                    index=0 if target_doc.get("fraudulent") == 0 else 1,
                    format_func=lambda x: f"{x} ({'Legitimate' if x==0 else 'Fraudulent'})",
                )

            edit_desc = st.text_area("Job Description", value=target_doc.get("description", ""), height=130)

            btn_update = st.form_submit_button("Save Changes via PyMongo update_one()", type="primary")

            if btn_update:
                if not is_conn:
                    st.error("Cannot update: MongoDB Atlas is offline.")
                else:
                    update_payload = {
                        "title": edit_title,
                        "company_name": edit_company or None,
                        "salary_range": edit_salary or None,
                        "employment_type": edit_emp_type,
                        "required_experience": edit_exp,
                        "description": edit_desc,
                        "fraudulent": edit_fraud,
                    }
                    # Handle both integer and string job_ids
                    job_key = int(target_id) if target_id.isdigit() else target_id
                    ok = update_job_ad(job_key, update_payload, db=db)
                    if ok:
                        st.success(f"✓ Job document `{target_id}` updated successfully in MongoDB!")
                    else:
                        st.error("Update failed or no fields were modified.")
    else:
        st.info("Enter a keyword above to find a job listing to edit.")


# -------------------------------------------------------------------------
# TAB 4: DELETE
# -------------------------------------------------------------------------
with tab_delete:
    st.markdown("### 🗑️ Remove Job Listing from MongoDB")
    st.caption("Demonstrates PyMongo `delete_one()` with mandatory confirmation.")

    del_search = st.text_input("Search Job ID or Title to Delete:", placeholder="e.g. Analyst", key="del_search_term")
    del_jobs = []
    if is_conn and del_search:
        del_jobs = get_job_ads(search_term=del_search, limit=10, db=db)

    if del_jobs:
        del_choices = {str(j.get("job_id")): j for j in del_jobs}
        del_id = st.selectbox("Select Job to Delete:", options=list(del_choices.keys()), format_func=lambda x: f"{x} - {del_choices[x].get('title')}", key="del_select")
        del_doc = del_choices[del_id]

        st.warning(f"You have selected: **{del_doc.get('title')}** (ID: `{del_id}`, Company: {del_doc.get('company_name') or 'N/A'})")

        confirm_checkbox = st.checkbox("⚠️ I confirm that I want to permanently delete this document from MongoDB.", key="confirm_del")

        if st.button("Permanently Delete Document", type="primary"):
            if not confirm_checkbox:
                st.error("Please check the confirmation box before deleting.")
            elif not is_conn:
                st.error("Cannot delete: MongoDB Atlas is offline.")
            else:
                job_key = int(del_id) if del_id.isdigit() else del_id
                deleted = delete_job_ad(job_key, db=db)
                if deleted:
                    st.success(f"✓ Document `{del_id}` successfully removed from MongoDB `jobs` collection.")
                else:
                    st.error("Failed to delete document from database.")
    else:
        st.info("Search for a job listing to initiate the deletion workflow.")


# -------------------------------------------------------------------------
# TAB 5: DATASET IMPORT
# -------------------------------------------------------------------------
with tab_import:
    st.markdown("### ⚡ Import Dataset into MongoDB Atlas")
    st.caption("Imports genuine EMSCAD labelled job records into your MongoDB Atlas cluster.")

    if not is_conn:
        st.warning("MongoDB Atlas is not connected. Configure your connection in `.env` to enable dataset import.")
    else:
        total_current = count_docs(config.COLLECTION_JOBS, {}, db=db)
        st.markdown(f"**Current documents in `jobs` collection**: `{total_current:,}`")

        import_limit = st.select_slider(
            "Select Number of Records to Import:",
            options=[500, 1000, 2500, 5000, 10000, 17880],
            value=1000,
            help="Choose a subset for fast initial testing or import the full dataset.",
        )

        if st.button("🚀 Start Batched PyMongo Import", type="primary"):
            with st.spinner(f"Importing {import_limit:,} records with deduplication into MongoDB Atlas..."):
                res = import_dataset(
                    csv_path=str(config.DATASET_CSV_PATH),
                    batch_size=500,
                    limit=import_limit,
                    setup_db_rules=True,
                )
                if res.get("success"):
                    st.success(
                        f"✓ Import finished! Imported: {res['imported']:,} records | "
                        f"Skipped duplicates: {res['skipped']:,} | Rejected: {res['rejected']:,}"
                    )
                    st.rerun()
                else:
                    st.error(f"Import failed: {res.get('error')}")
