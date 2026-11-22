"""
Database package for ScamShield.
Manages MongoDB Atlas connections, schema validation, indexes, and PyMongo CRUD operations.
"""

from .connection import get_mongo_client, get_db, check_connection
from .operations import (
    insert_one_doc,
    insert_many_docs,
    find_docs,
    find_one_doc,
    update_one_doc,
    delete_one_doc,
    count_docs,
    run_aggregation,
    explain_query_plan,
)

__all__ = [
    "get_mongo_client",
    "get_db",
    "check_connection",
    "insert_one_doc",
    "insert_many_docs",
    "find_docs",
    "find_one_doc",
    "update_one_doc",
    "delete_one_doc",
    "count_docs",
    "run_aggregation",
    "explain_query_plan",
]
