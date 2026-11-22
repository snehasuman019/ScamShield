"""
MongoDB connection management for ScamShield.
Safely connects to MongoDB Atlas using PyMongo without leaking credentials in logs or UI.
"""

import logging
from typing import Optional, Tuple
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import (
    PyMongoError,
    ServerSelectionTimeoutError,
    ConnectionFailure,
    ConfigurationError,
)
import config

logger = logging.getLogger("scamshield.database")

# Global client cache
_mongo_client: Optional[MongoClient] = None


def get_mongo_client(force_new: bool = False, custom_uri: Optional[str] = None) -> Optional[MongoClient]:
    """
    Get or create a PyMongo MongoClient instance.
    Uses cached client unless force_new is True or custom_uri is provided.
    """
    global _mongo_client

    if _mongo_client is not None and not force_new and custom_uri is None:
        return _mongo_client

    uri = custom_uri or config.MONGODB_URI
    if not uri or not config.is_mongodb_configured():
        return None

    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS,
            connectTimeoutMS=config.MONGO_TIMEOUT_MS,
            socketTimeoutMS=config.MONGO_TIMEOUT_MS,
            appname="ScamShield-App",
        )
        # Verify connection by triggering a server ping
        client.admin.command("ping")
        if custom_uri is None:
            _mongo_client = client
        return client
    except ConfigurationError as ce:
        logger.error("MongoDB configuration error (e.g. invalid URI format)")
        return None
    except (ServerSelectionTimeoutError, ConnectionFailure) as se:
        logger.warning("Could not reach MongoDB Atlas cluster within timeout.")
        return None
    except PyMongoError as pe:
        logger.error(f"Unexpected MongoDB driver error: {pe.__class__.__name__}")
        return None


def get_db(custom_client: Optional[MongoClient] = None) -> Optional[Database]:
    """
    Return the ScamShield MongoDB Database instance ('scamshield_db').
    Returns None if the database connection cannot be established.
    """
    client = custom_client if custom_client is not None else get_mongo_client()
    if client is None:
        return None
    try:
        return client[config.DB_NAME]
    except Exception as e:
        logger.error(f"Error accessing database {config.DB_NAME}: {e}")
        return None


def check_connection(custom_client: Optional[MongoClient] = None) -> Tuple[bool, str]:
    """
    Check if the MongoDB Atlas database is connected and reachable.
    Returns (status: bool, message: str) without exposing the raw connection URI.
    """
    if not config.is_mongodb_configured():
        return (
            False,
            "MongoDB Atlas is not configured. Please set a valid MONGODB_URI in your .env file."
        )

    client = custom_client if custom_client is not None else get_mongo_client()
    if client is None:
        return (
            False,
            "Failed to establish connection with MongoDB Atlas. Check your cluster status and Network Access (IP whitelist)."
        )

    try:
        res = client.admin.command("ping")
        if res.get("ok") == 1.0 or res.get("ok") == 1:
            return (True, f"Connected to MongoDB Atlas successfully! Database: '{config.DB_NAME}'.")
        return (False, "Ping response from MongoDB Atlas was unexpected.")
    except Exception as e:
        return (False, f"Connection failed: {e.__class__.__name__}")


def close_connection():
    """Close the cached PyMongo client connection cleanly."""
    global _mongo_client
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
        _mongo_client = None
