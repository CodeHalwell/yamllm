"""
Memory management component for YAMLLM.

This module handles conversation memory and vector storage,
extracted from the main LLM class for better separation of concerns.
"""

from typing import Dict, List, Any, Optional
import os
import logging

from yamllm.memory import ConversationStore, VectorStore


class MemoryManager:
    """
    Manages conversation history and vector storage for LLM interactions.
    
    This class encapsulates all memory-related functionality that was
    previously embedded in the main LLM class.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        max_messages: int = 10,
        session_id: Optional[str] = None,
        conversation_db_path: Optional[str] = None,
        vector_index_path: Optional[str] = None,
        vector_store_top_k: int = 5,
        vector_dim: int = 1536,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            enabled: Whether memory is enabled
            max_messages: Maximum messages to keep in context
            session_id: Session identifier for conversations
            conversation_db_path: Path to conversation database
            vector_index_path: Path to vector index
            vector_store_top_k: Number of similar messages to retrieve
            vector_dim: Dimension of embedding vectors
            logger: Logger instance
        """
        self.enabled = enabled
        self.max_messages = max_messages
        self.session_id = session_id
        self.conversation_db_path = conversation_db_path
        self.vector_index_path = vector_index_path or os.path.join("memory", "vector_store", "faiss_index.idx")
        self.vector_store_top_k = vector_store_top_k
        self.vector_dim = vector_dim
        self.logger = logger or logging.getLogger(__name__)
        
        self.conversation_store = None
        self.vector_store = None
        
        if self.enabled:
            self._initialize()
    
    def _initialize(self):
        """Initialize memory and vector stores."""
        try:
            self.conversation_store = ConversationStore(db_path=self.conversation_db_path)
            self.vector_store = VectorStore(
                store_path=os.path.dirname(self.vector_index_path),
                vector_dim=self.vector_dim
            )
            if not self.conversation_store.db_exists():
                self.conversation_store.create_db()
        except Exception as e:
            self.logger.error(f"Failed to initialize memory: {e}")
            self.enabled = False
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history for the current session.
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        if not self.enabled or not self.conversation_store:
            return []
        
        try:
            return self.conversation_store.get_messages(
                session_id=self.session_id,
                limit=limit or self.max_messages
            )
        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def add_message(self, role: str, content: str) -> Optional[int]:
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user', 'assistant', etc.)
            content: Message content
            
        Returns:
            Message ID if successful, None otherwise
        """
        if not self.enabled or not self.conversation_store:
            return None
        
        try:
            return self.conversation_store.add_message(
                session_id=self.session_id,
                role=role,
                content=content
            )
        except Exception as e:
            self.logger.error(f"Error adding message to memory: {e}")
            return None
    
    def add_vector(self, vector: List[float], message_id: int, content: str, role: str):
        """
        Add a vector embedding to the vector store.
        
        Args:
            vector: Embedding vector
            message_id: Associated message ID
            content: Message content
            role: Message role
        """
        if not self.enabled or not self.vector_store:
            return
        
        try:
            self.vector_store.add_vector(
                vector=vector,
                message_id=message_id,
                content=content,
                role=role
            )
        except Exception as e:
            self.logger.error(f"Error adding vector to store: {e}")
    
    def search_similar(self, query_vector: List[float], k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar messages with metadata and similarity scores
        """
        if not self.enabled or not self.vector_store:
            return []
        
        try:
            return self.vector_store.search(
                query_vector=query_vector,
                k=k or self.vector_store_top_k
            )
        except Exception as e:
            self.logger.error(f"Error searching similar messages: {e}")
            return []
    
    def store_interaction(
        self,
        prompt: str,
        response: str,
        prompt_embedding: Optional[List[float]] = None,
        response_embedding: Optional[List[float]] = None
    ):
        """
        Store a complete interaction (prompt and response) in memory.
        
        Args:
            prompt: User prompt
            response: Assistant response
            prompt_embedding: Embedding for the prompt
            response_embedding: Embedding for the response
        """
        if not self.enabled:
            return
        
        try:
            # Store user message
            message_id = self.add_message("user", prompt)
            if message_id and prompt_embedding:
                self.add_vector(prompt_embedding, message_id, prompt, "user")
            
            # Store assistant response
            response_id = self.add_message("assistant", response)
            if response_id and response_embedding:
                self.add_vector(response_embedding, response_id, response, "assistant")
        except Exception as e:
            self.logger.error(f"Error storing interaction: {e}")
    
    def close(self):
        """Clean up resources."""
        if self.conversation_store:
            self.conversation_store.close()
        if self.vector_store:
            self.vector_store.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()