import sqlite3
from datetime import datetime
import json

class ConversationStore:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    metadata TEXT
                )
            """)

    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: dict = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO conversations 
                   (timestamp, session_id, role, content, metadata) 
                   VALUES (?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), session_id, role, content, 
                 json.dumps(metadata or {}))
            )
            return cursor.lastrowid