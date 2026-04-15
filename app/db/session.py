import sqlite3
from contextlib import contextmanager
from pathlib import Path
from config.settings import settings
from app.db.models import MIGRATIONS, run_schema_upgrades


def init_wal_mode(conn: sqlite3.Connection) -> None:
    """Enable Write-Ahead Logging mode for better concurrency."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()


def get_db_connection() -> sqlite3.Connection:
    """Create and return a new database connection with WAL mode enabled."""
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_wal_mode(conn)
    return conn


def get_conn() -> sqlite3.Connection:
    """
    Get database connection.

    Note: Returns a fresh connection. Callers should use context managers
    or ensure proper connection cleanup.
    """
    return get_db_connection()


@contextmanager
def get_db_context():
    """
    Context manager for database connections with automatic cleanup.

    Usage:
        with get_db_context() as conn:
            conn.execute("SELECT ...")
            conn.commit()
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db(sqlite_path: str) -> None:
    """Initialize database with schema and WAL mode."""
    # Ensure DB file path directory exists
    p = Path(sqlite_path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    conn.executescript(MIGRATIONS)
    conn.commit()

    # Run schema upgrades for existing databases
    run_schema_upgrades(conn)

    # Enable WAL mode
    init_wal_mode(conn)

    conn.close()


# =============================================================================
# PROACTIVE COOLDOWN PERSISTENCE
# =============================================================================

def get_proactive_cooldown(user_id: int) -> dict:
    """
    Load proactive cooldown state from database.

    Returns:
        {"last_asked_at": ISO string or None, "asks_today": int}
    """
    from datetime import datetime

    with get_db_context() as conn:
        row = conn.execute(
            "SELECT last_asked_at, asks_today, asks_date FROM proactive_cooldown WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if not row:
            return {"last_asked_at": None, "asks_today": 0}

        # Reset asks_today if it's a new day
        today = datetime.now().strftime("%Y-%m-%d")
        if row["asks_date"] != today:
            return {"last_asked_at": row["last_asked_at"], "asks_today": 0}

        return {
            "last_asked_at": row["last_asked_at"],
            "asks_today": row["asks_today"] or 0
        }


def update_proactive_cooldown(user_id: int, last_asked_at: str) -> None:
    """
    Update proactive cooldown after a successful proactive ask.

    Increments asks_today and sets last_asked_at.
    Resets asks_today if it's a new day.
    """
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")

    with get_db_context() as conn:
        # Check current state
        row = conn.execute(
            "SELECT asks_today, asks_date FROM proactive_cooldown WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if row:
            # Reset if new day
            if row["asks_date"] != today:
                asks_today = 1
            else:
                asks_today = (row["asks_today"] or 0) + 1

            conn.execute(
                """UPDATE proactive_cooldown
                   SET last_asked_at = ?, asks_today = ?, asks_date = ?, updated_at = datetime('now')
                   WHERE user_id = ?""",
                (last_asked_at, asks_today, today, user_id)
            )
        else:
            conn.execute(
                """INSERT INTO proactive_cooldown (user_id, last_asked_at, asks_today, asks_date)
                   VALUES (?, ?, 1, ?)""",
                (user_id, last_asked_at, today)
            )

        conn.commit()
