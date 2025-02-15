import sqlite3
import os
from typing import List, Dict, Any
import faiss
import numpy as np
import pickle

class ConversationStore:
    def __init__(self, db_path: str = "yamllm/memory/conversation_history.db"):
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

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """Add a message to the database and return its ID"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, role, content)
            )
            message_id = cursor.lastrowid
            conn.commit()
            return message_id
        finally:
            conn.close()

    def get_messages(self, session_id: str = None, limit: int = None) -> List[Dict[str, str]]:
        """Retrieve messages from the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Remove embedding from SELECT statement since it's not in the schema
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
            
            # Update dictionary creation to match selected columns
            messages = [{"role": role, "content": content} 
                    for role, content in results]
            return messages[::-1]
        finally:
            conn.close()

class VectorStore:
    def __init__(self, vector_dim: int = 1536, store_path: str = "yamllm/memory/vector_store"):
        self.vector_dim = vector_dim
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "faiss_index.idx")
        self.metadata_path = os.path.join(store_path, "metadata.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize or load the index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(vector_dim)  # Inner product for cosine similarity
            self.metadata = []  # List to store message metadata

    def add_vector(self, vector: List[float], message_id: int, content: str, role: str) -> None:
        """Add a vector to the index with associated message content
        
        Args:
            vector: The embedding vector
            message_id: The ID of the message
            content: The actual message content
            role: The role of the message sender
        """
        vector_np = np.array([vector]).astype('float32')
        faiss.normalize_L2(vector_np)
        
        self.index.add(vector_np)
        self.metadata.append({
            'id': message_id,
            'content': content,
            'role': role
        })
        self._save_store()

    def _save_store(self) -> None:
        """Save the FAISS index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their metadata
        
        Args:
            query_vector: The query embedding vector
            k: Number of results to return
            
        Returns:
            List of dicts containing message metadata and similarity scores
        """
        query_np = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_np)
        
        distances, indices = self.index.search(query_np, k)
        
        # Return message metadata and similarity scores
        results = [
            {
                **self.metadata[idx],
                'similarity': float(score)
            }
            for idx, score in zip(indices[0], distances[0])
            if idx != -1
        ]
        
        return results