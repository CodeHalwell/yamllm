import sqlite3
import os
from typing import List, Dict

class ConversationStore:
    def __init__(self, db_path: str = "conversation_history.db"):
        self.db_path = db_path

    def db_exists(self) -> bool:
        """Check if the database file exists"""
        return os.path.exists(self.db_path)

    def create_db(self) -> None:
        """Create the database and messages table if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, role, content)
            )
            conn.commit()
        finally:
            conn.close()

    def get_messages(self, session_id: str = None, limit: int = None) -> List[Dict[str, str]]:
        """Retrieve messages from the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            query = 'SELECT role, content FROM messages'
            params = []
            
            if session_id:
                query += ' WHERE session_id = ?'
                params.append(session_id)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Convert to list of dictionaries and reverse to get chronological order
            messages = [{"role": role, "content": content} for role, content in results]
            return messages[::-1]
        finally:
            conn.close()