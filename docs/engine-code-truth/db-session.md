# db-session

Source: [app/db/session.py](/D:/Nura/Code/nura_living_memory_code/app/db/session.py)

```text
0001: import sqlite3
0002: from contextlib import contextmanager
0003: from pathlib import Path
0004: from config.settings import settings
0005: from app.db.models import MIGRATIONS, run_schema_upgrades
0006: 
0007: 
0008: def init_wal_mode(conn: sqlite3.Connection) -> None:
0009:     """Enable Write-Ahead Logging mode for better concurrency."""
0010:     conn.execute("PRAGMA journal_mode=WAL;")
0011:     conn.commit()
0012: 
0013: 
0014: def get_db_connection() -> sqlite3.Connection:
0015:     """Create and return a new database connection with WAL mode enabled."""
0016:     conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
0017:     conn.row_factory = sqlite3.Row
0018:     init_wal_mode(conn)
0019:     return conn
0020: 
0021: 
0022: def get_conn() -> sqlite3.Connection:
0023:     """
0024:     Get database connection.
0025: 
0026:     Note: Returns a fresh connection. Callers should use context managers
0027:     or ensure proper connection cleanup.
0028:     """
0029:     return get_db_connection()
0030: 
0031: 
0032: @contextmanager
0033: def get_db_context():
0034:     """
0035:     Context manager for database connections with automatic cleanup.
0036: 
0037:     Usage:
0038:         with get_db_context() as conn:
0039:             conn.execute("SELECT ...")
0040:             conn.commit()
0041:     """
0042:     conn = get_db_connection()
0043:     try:
0044:         yield conn
0045:     finally:
0046:         conn.close()
0047: 
0048: 
0049: def init_db(sqlite_path: str) -> None:
0050:     """Initialize database with schema and WAL mode."""
0051:     # Ensure DB file path directory exists
0052:     p = Path(sqlite_path)
0053:     if p.parent and str(p.parent) != ".":
0054:         p.parent.mkdir(parents=True, exist_ok=True)
0055: 
0056:     conn = sqlite3.connect(sqlite_path, check_same_thread=False)
0057:     conn.executescript(MIGRATIONS)
0058:     conn.commit()
0059: 
0060:     # Run schema upgrades for existing databases
0061:     run_schema_upgrades(conn)
0062: 
0063:     # Enable WAL mode
0064:     init_wal_mode(conn)
0065: 
0066:     conn.close()
0067: 
0068: 
0069: # =============================================================================
0070: # PROACTIVE COOLDOWN PERSISTENCE
0071: # =============================================================================
0072: 
0073: def get_proactive_cooldown(user_id: int) -> dict:
0074:     """
0075:     Load proactive cooldown state from database.
0076: 
0077:     Returns:
0078:         {"last_asked_at": ISO string or None, "asks_today": int}
0079:     """
0080:     from datetime import datetime
0081: 
0082:     with get_db_context() as conn:
0083:         row = conn.execute(
0084:             "SELECT last_asked_at, asks_today, asks_date FROM proactive_cooldown WHERE user_id = ?",
0085:             (user_id,)
0086:         ).fetchone()
0087: 
0088:         if not row:
0089:             return {"last_asked_at": None, "asks_today": 0}
0090: 
0091:         # Reset asks_today if it's a new day
0092:         today = datetime.now().strftime("%Y-%m-%d")
0093:         if row["asks_date"] != today:
0094:             return {"last_asked_at": row["last_asked_at"], "asks_today": 0}
0095: 
0096:         return {
0097:             "last_asked_at": row["last_asked_at"],
0098:             "asks_today": row["asks_today"] or 0
0099:         }
0100: 
0101: 
0102: def update_proactive_cooldown(user_id: int, last_asked_at: str) -> None:
0103:     """
0104:     Update proactive cooldown after a successful proactive ask.
0105: 
0106:     Increments asks_today and sets last_asked_at.
0107:     Resets asks_today if it's a new day.
0108:     """
0109:     from datetime import datetime
0110: 
0111:     today = datetime.now().strftime("%Y-%m-%d")
0112: 
0113:     with get_db_context() as conn:
0114:         # Check current state
0115:         row = conn.execute(
0116:             "SELECT asks_today, asks_date FROM proactive_cooldown WHERE user_id = ?",
0117:             (user_id,)
0118:         ).fetchone()
0119: 
0120:         if row:
0121:             # Reset if new day
0122:             if row["asks_date"] != today:
0123:                 asks_today = 1
0124:             else:
0125:                 asks_today = (row["asks_today"] or 0) + 1
0126: 
0127:             conn.execute(
0128:                 """UPDATE proactive_cooldown
0129:                    SET last_asked_at = ?, asks_today = ?, asks_date = ?, updated_at = datetime('now')
0130:                    WHERE user_id = ?""",
0131:                 (last_asked_at, asks_today, today, user_id)
0132:             )
0133:         else:
0134:             conn.execute(
0135:                 """INSERT INTO proactive_cooldown (user_id, last_asked_at, asks_today, asks_date)
0136:                    VALUES (?, ?, 1, ?)""",
0137:                 (user_id, last_asked_at, today)
0138:             )
0139: 
0140:         conn.commit()
```
