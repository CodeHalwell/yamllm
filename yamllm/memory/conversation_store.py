import sqlite3
import os
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
import pickle

class ConversationStore:
    """
    A class to manage conversation history stored in a SQLite database.

    Attributes:
        db_path (str): The path to the SQLite database file.

    Methods:
        db_exists() -> bool:
            Check if the database file exists.
        create_db() -> None:
            Create the database and messages table if they don't exist.
        add_message(session_id: str, role: str, content: str) -> int:
            Add a message to the database and return its ID.
        get_messages(session_id: str = None, limit: int = None) -> List[Dict[str, str]]:
            Retrieve messages from the database.
        get_session_ids() -> List[str]:
            Retrieve a list of unique session IDs from the database.
        delete_session(session_id: str) -> None:
            Delete all messages associated with a specific session ID.
        delete_database() -> None:
            Delete the entire database file.
        __len__() -> int:
            Returns the total number of messages in the store.
        __str__() -> str:
            Returns a human-readable string representation.
        __repr__() -> str:
            Returns a detailed string representation.
    """
    def __init__(self, db_path: str = "yamllm/memory/conversation_history.db"):
        self.db_path = db_path

    def db_exists(self) -> bool:
        """Check if the database file exists"""
        return os.path.exists(self.db_path)

    def create_db(self) -> None:
        """
        Create the database and messages table if they don't exist.

        This method establishes a connection to the SQLite database specified by
        `self.db_path`. It then creates a table named `messages` with the following
        columns if it does not already exist:
            - id: An integer primary key that auto-increments.
            - session_id: A text field that is not null.
            - role: A text field that is not null.
            - content: A text field that is not null.
            - timestamp: A datetime field that defaults to the current timestamp.

        The connection to the database is closed after the table is created.
        """
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
        """
        Add a message to the database and return its ID.

        Args:
            session_id (str): The ID of the session to which the message belongs.
            role (str): The role of the sender (e.g., 'user', 'assistant').
            content (str): The content of the message.

        Returns:
            int: The ID of the newly added message in the database.
        """
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
        """
        Retrieve messages from the database.
        Args:
            session_id (str, optional): The session ID to filter messages by. Defaults to None.
            limit (int, optional): The maximum number of messages to retrieve. Defaults to None.
        Returns:
            List[Dict[str, str]]: A list of messages, where each message is represented as a dictionary
                                  with 'role' and 'content' keys.
        """
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

    def get_session_ids(self) -> List[str]:
        """
        Retrieve a list of unique session IDs from the database.
        Returns:
            List[str]: A list of unique session IDs.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT session_id FROM messages')
            results = cursor.fetchall()
            return [row[0] for row in results]
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> None:
        """
        Delete all messages associated with a specific session ID.
        Args:
            session_id (str): The ID of the session to delete.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_database(self) -> None:
        """Delete the entire database file."""
        os.remove(self.db_path)


    def __repr__(self) -> str:
        """Returns a detailed string representation of the ConversationStore object."""
        return f"ConversationStore(db_path='{self.db_path}')"
    
    def __str__(self) -> str:
        """Returns a human-readable string representation of the ConversationStore object."""
        session_count = len(self.get_session_ids())
        return f"ConversationStore with {session_count} sessions at {self.db_path}"
    
    def __len__(self) -> int:
        """Returns the total number of messages in the store."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM messages')
            return cursor.fetchone()[0]
        finally:
            conn.close()
        
class VectorStore:
    def __init__(self, vector_dim: int = 1536, store_path: str = "yamllm/memory/vector_store"):
        """
        Initializes the ConversationStore object.
        Args:
            vector_dim (int): The dimensionality of the vectors to be stored. Default is 1536.
            store_path (str): The path to the directory where the vector store and metadata will be saved. Default is "yamllm/memory/vector_store".
        Attributes:
            vector_dim (int): The dimensionality of the vectors to be stored.
            store_path (str): The path to the directory where the vector store and metadata will be saved.
            index_path (str): The path to the FAISS index file.
            metadata_path (str): The path to the metadata file.
            index (faiss.Index): The FAISS index for storing vectors.
            metadata (list): A list to store message metadata.
        The constructor creates the directory if it doesn't exist, and initializes or loads the FAISS index and metadata.
        """
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
        """
        Add a vector to the index with its associated metadata.

        Args:
            vector (List[float]): The embedding vector to be added.
            message_id (int): Unique identifier for the message.
            content (str): The message content.
            role (str): The role of the message sender.

        Note:
            The vector is L2-normalized before being added to the index.
            Updates are automatically saved to disk.
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
        """
        Saves the current state of the conversation store to disk.

        This method writes the FAISS index to the specified index path and 
        serializes the metadata to a file at the specified metadata path.

        Raises:
            IOError: If there is an error writing the index or metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the k most similar vectors in the index.

        Args:
            query_vector (List[float]): The query embedding vector.
            k (int, optional): Number of similar items to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing:
                - id (int): Message ID
                - content (str): Message content
                - role (str): Message role
                - similarity (float): Similarity score
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
    
    def get_vec_and_text(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Retrieve all vectors and their associated metadata.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: A tuple containing:
                - np.ndarray: Array of shape (n, vector_dim) containing all vectors
                - List[Dict[str, Any]]: List of metadata dictionaries for each vector
                  with keys: 'id', 'content', 'role'

        Note:
            Returns empty array and list if the index is empty.
        """
        if self.index.ntotal == 0:
            return np.array([]), []
        
        # Initialize array to store vectors
        vectors = np.empty((self.index.ntotal, self.vector_dim), dtype=np.float32)
        
        # Reconstruct vectors one by one
        for i in range(self.index.ntotal):
            vectors[i] = self.index.reconstruct(i)
        
        return vectors, self.metadata
    
    def __repr__(self) -> str:
        """Returns a detailed string representation of the VectorStore object."""
        return f"VectorStore(vector_dim={self.vector_dim}, store_path='{self.store_path}')"
    
    def __str__(self) -> str:
        """Returns a human-readable string representation of the VectorStore object."""
        vector_count = self.index.ntotal
        return f"VectorStore with {vector_count} vectors of dimension {self.vector_dim}"
    
    def __len__(self) -> int:
        """Returns the total number of vectors in the store."""
        return self.index.ntotal