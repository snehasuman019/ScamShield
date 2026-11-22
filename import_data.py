"""
Dataset Import Script for ScamShield.
Imports the cleaned, labelled EMSCAD dataset (fake_job_postings.csv) into MongoDB Atlas.
Demonstrates:
- Safe batch insertions (insert_many)
- Deduplication using unique job_id
- Schema compliance and missing value handling
- Detailed progress and audit reporting
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone
import pandas as pd
from pymongo.errors import PyMongoError, BulkWriteError

import config
from database.connection import get_db, check_connection
from database.indexes import create_all_indexes
from database.validation import apply_schema_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scamshield.import")

REQUIRED_COLUMNS = ["title", "description", "fraudulent"]


def clean_val(val, default=None):
    """Clean pandas NaN or float nulls into JSON/BSON-compatible values."""
    if pd.isna(val) or val is None:
        return default
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return default
    return val


def import_dataset(
    csv_path: str = str(config.DATASET_CSV_PATH),
    batch_size: int = 500,
    limit: int = 0,
    setup_db_rules: bool = True,
) -> dict:
    """
    Import fake_job_postings.csv into MongoDB Atlas 'jobs' collection.
    """
    if not os.path.exists(csv_path):
        logger.error(f"Dataset file not found at: {csv_path}")
        logger.info("Run 'python data/download_dataset.py' to fetch it.")
        return {"success": False, "error": f"File not found: {csv_path}"}

    # Verify MongoDB connection
    is_connected, msg = check_connection()
    if not is_connected:
        logger.error(f"MongoDB connection failed: {msg}")
        return {"success": False, "error": msg}

    db = get_db()
    if db is None:
        return {"success": False, "error": "Unable to obtain MongoDB database reference."}

    logger.info(f"Connected to database '{config.DB_NAME}'. Starting import from {csv_path}...")

    # Optionally set up schema validation and indexes first
    if setup_db_rules:
        logger.info("Ensuring schema validation rules...")
        apply_schema_validation(db)
        logger.info("Building MongoDB indexes for performance and deduplication...")
        create_all_indexes(db)

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} total rows from CSV.")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return {"success": False, "error": str(e)}

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        err = f"CSV missing required columns: {missing_cols}"
        logger.error(err)
        return {"success": False, "error": err}

    if limit > 0:
        df = df.head(limit)
        logger.info(f"Limit applied: processing first {len(df):,} records.")

    # Fetch existing job_ids to prevent duplicate insertions
    existing_job_ids = set()
    try:
        cursor = db[config.COLLECTION_JOBS].find({}, {"job_id": 1, "_id": 0})
        existing_job_ids = {doc["job_id"] for doc in cursor if "job_id" in doc}
        logger.info(f"Found {len(existing_job_ids):,} existing jobs already in MongoDB.")
    except Exception as e:
        logger.warning(f"Could not fetch existing IDs (collection may be empty): {e}")

    imported_count = 0
    skipped_count = 0
    rejected_count = 0

    batch = []
    total_rows = len(df)
    now_iso = datetime.now(timezone.utc).isoformat()

    for idx, row in df.iterrows():
        raw_id = row.get("job_id")
        job_id = int(raw_id) if pd.notna(raw_id) else f"IMPORT-{idx}"

        # Deduplication check
        if job_id in existing_job_ids:
            skipped_count += 1
            continue

        title = clean_val(row.get("title"), "")
        description = clean_val(row.get("description"), "")

        # Reject records missing title or description
        if not title or not description:
            rejected_count += 1
            continue

        # Target label
        fraud_val = clean_val(row.get("fraudulent"), 0)
        try:
            fraudulent = int(fraud_val)
            if fraudulent not in (0, 1):
                fraudulent = 0
        except (ValueError, TypeError):
            fraudulent = 0

        # Construct BSON document
        doc = {
            "job_id": job_id,
            "title": str(title)[:250],
            "company_name": str(clean_val(row.get("company_profile"), ""))[:150] or None,
            "description": str(description),
            "requirements": str(clean_val(row.get("requirements"), "")) or None,
            "benefits": str(clean_val(row.get("benefits"), "")) or None,
            "employment_type": clean_val(row.get("employment_type"), None),
            "required_experience": clean_val(row.get("required_experience"), None),
            "required_education": clean_val(row.get("required_education"), None),
            "salary_range": clean_val(row.get("salary_range"), None),
            "has_company_logo": int(row.get("has_company_logo")) if pd.notna(row.get("has_company_logo")) else 0,
            "has_questions": int(row.get("has_questions")) if pd.notna(row.get("has_questions")) else 0,
            "telecommuting": int(row.get("telecommuting")) if pd.notna(row.get("telecommuting")) else 0,
            "industry": clean_val(row.get("industry"), None),
            "function": clean_val(row.get("function"), None),
            "fraudulent": fraudulent,
            "source": "EMSCAD",
            "created_at": now_iso,
        }

        batch.append(doc)
        existing_job_ids.add(job_id)

        # Execute batched insert
        if len(batch) >= batch_size:
            try:
                db[config.COLLECTION_JOBS].insert_many(batch, ordered=False)
                imported_count += len(batch)
            except BulkWriteError as bwe:
                inserted = bwe.details.get("nInserted", 0)
                imported_count += inserted
                skipped_count += (len(batch) - inserted)
            except PyMongoError as pe:
                logger.error(f"Batch insertion error: {pe}")
                rejected_count += len(batch)

            batch = []
            sys.stdout.write(f"\r[IMPORTING] Processed: {idx + 1:,}/{total_rows:,} | Imported: {imported_count:,} | Skipped: {skipped_count:,}")
            sys.stdout.flush()

    # Flush remaining batch
    if batch:
        try:
            db[config.COLLECTION_JOBS].insert_many(batch, ordered=False)
            imported_count += len(batch)
        except BulkWriteError as bwe:
            inserted = bwe.details.get("nInserted", 0)
            imported_count += inserted
            skipped_count += (len(batch) - inserted)
        except PyMongoError as pe:
            logger.error(f"Final batch error: {pe}")
            rejected_count += len(batch)

    print("\n" + "=" * 60)
    logger.info("=== Dataset Import Completed ===")
    logger.info(f"Total rows inspected: {total_rows:,}")
    logger.info(f"Successfully imported: {imported_count:,}")
    logger.info(f"Skipped (duplicates) : {skipped_count:,}")
    logger.info(f"Rejected (malformed) : {rejected_count:,}")
    print("=" * 60)

    return {
        "success": True,
        "total_rows": total_rows,
        "imported": imported_count,
        "skipped": skipped_count,
        "rejected": rejected_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Import EMSCAD dataset into MongoDB Atlas.")
    parser.add_argument("--csv_path", type=str, default=str(config.DATASET_CSV_PATH), help="Path to fake_job_postings.csv")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for bulk inserts")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit (0 for all)")
    parser.add_argument("--no_db_rules", action="store_true", help="Skip schema validation and index creation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res = import_dataset(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        limit=args.limit,
        setup_db_rules=not args.no_db_rules,
    )
    sys.exit(0 if res.get("success") else 1)
