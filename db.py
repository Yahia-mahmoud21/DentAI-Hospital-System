import sqlite3
import os

# Point to the database within this project directory to avoid duplicate DB confusion
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "dental_project_DB.db"))


def connect():
    """Return a sqlite3.Connection with sane defaults and WAL mode enabled for concurrent access."""
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False, isolation_level=None)
    # Apply recommended PRAGMAs
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return conn
