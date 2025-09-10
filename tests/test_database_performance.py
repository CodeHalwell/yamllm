"""
Tests for database performance improvements and indexes.

This module tests the database indexes and performance optimizations
added to the conversation store for improved query performance.
"""

import pytest
import sqlite3
import tempfile
import os
from yamllm.memory.conversation_store import ConversationStore, VectorStore


class TestDatabaseIndexes:
    """Test database index creation and performance."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def conversation_store(self, temp_db_path):
        """Create a conversation store with temporary database."""
        store = ConversationStore(temp_db_path)
        store.create_db()
        return store
    
    def test_database_table_creation(self, conversation_store):
        """Test that the messages table is created correctly."""
        conn = sqlite3.connect(conversation_store.db_path)
        try:
            cursor = conn.cursor()
            
            # Check that the table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='messages'
            """)
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 'messages'
            
            # Check table schema
            cursor.execute("PRAGMA table_info(messages)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            expected_columns = ['id', 'session_id', 'role', 'content', 'timestamp']
            for col in expected_columns:
                assert col in column_names
                
        finally:
            conn.close()
    
    def test_index_creation(self, conversation_store):
        """Test that all required indexes are created."""
        conn = sqlite3.connect(conversation_store.db_path)
        try:
            cursor = conn.cursor()
            
            # Get all indexes
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='messages'
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check that required indexes exist
            expected_indexes = [
                'idx_messages_session_id',
                'idx_messages_session_timestamp',
                'idx_messages_timestamp',
                'idx_messages_role'
            ]
            
            for index in expected_indexes:
                assert index in indexes, f"Index {index} not found"
                
        finally:
            conn.close()
    
    def test_session_id_index_performance(self, conversation_store):
        """Test that session_id index improves query performance."""
        # Add test data
        session_ids = ['session_1', 'session_2', 'session_3'] * 100
        
        for i, session_id in enumerate(session_ids):
            conversation_store.add_message(
                session_id=session_id,
                role='user' if i % 2 == 0 else 'assistant',
                content=f'Message {i}'
            )
        
        conn = sqlite3.connect(conversation_store.db_path)
        try:
            cursor = conn.cursor()
            
            # Query with session_id should use the index
            cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM messages WHERE session_id = ?", ('session_1',))
            plan = cursor.fetchall()
            
            # Check that the query plan mentions using an index
            plan_text = ' '.join([str(row) for row in plan])
            assert 'idx_messages_session_id' in plan_text.lower() or 'index' in plan_text.lower()
            
        finally:
            conn.close()
    
    def test_composite_index_performance(self, conversation_store):
        """Test that composite session_id+timestamp index works correctly."""
        # Add test data with different timestamps
        import time
        
        for i in range(10):
            conversation_store.add_message(
                session_id='test_session',
                role='user',
                content=f'Message {i}'
            )
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        conn = sqlite3.connect(conversation_store.db_path)
        try:
            cursor = conn.cursor()
            
            # Query with session_id and timestamp order should use composite index
            cursor.execute("""
                EXPLAIN QUERY PLAN 
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp DESC
            """, ('test_session',))
            plan = cursor.fetchall()
            
            plan_text = ' '.join([str(row) for row in plan])
            # Should use either the composite index or the session_id index
            assert 'idx_messages_session' in plan_text.lower() or 'index' in plan_text.lower()
            
        finally:
            conn.close()
    
    def test_message_retrieval_performance(self, conversation_store):
        """Test message retrieval performance with indexes."""
        # Add a significant amount of test data
        sessions = [f'session_{i}' for i in range(50)]
        
        for session in sessions:
            for j in range(20):  # 20 messages per session
                conversation_store.add_message(
                    session_id=session,
                    role='user' if j % 2 == 0 else 'assistant',
                    content=f'Message {j} in {session}'
                )
        
        # Test retrieval performance
        import time
        
        start_time = time.time()
        messages = conversation_store.get_messages(session_id='session_25', limit=10)
        end_time = time.time()
        
        # Should be very fast with proper indexing
        assert end_time - start_time < 0.1  # Less than 100ms
        assert len(messages) == 10
        assert all(msg['role'] in ['user', 'assistant'] for msg in messages)
    
    def test_database_size_optimization(self, conversation_store):
        """Test that indexes don't significantly increase database size."""
        # Get initial database size
        initial_size = os.path.getsize(conversation_store.db_path)
        
        # Add data
        for i in range(100):
            conversation_store.add_message(
                session_id=f'session_{i % 10}',
                role='user' if i % 2 == 0 else 'assistant',
                content=f'Test message {i} with some content to make it realistic'
            )
        
        # Get final database size
        final_size = os.path.getsize(conversation_store.db_path)
        
        # Indexes should not more than double the database size
        size_ratio = final_size / max(initial_size, 1)
        assert size_ratio < 3.0  # Reasonable overhead for indexes


class TestConversationStorePerformance:
    """Test performance improvements in conversation store operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def conversation_store(self, temp_db_path):
        """Create a conversation store with temporary database."""
        store = ConversationStore(temp_db_path)
        store.create_db()
        return store
    
    def test_bulk_message_insertion(self, conversation_store):
        """Test performance of bulk message insertion."""
        import time
        
        # Measure time for bulk insertion
        start_time = time.time()
        
        for i in range(1000):
            conversation_store.add_message(
                session_id=f'session_{i % 10}',
                role='user' if i % 2 == 0 else 'assistant',
                content=f'Performance test message {i}'
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 1000 insertions in reasonable time
        assert total_time < 5.0  # Less than 5 seconds
        print(f"Inserted 1000 messages in {total_time:.2f} seconds")
    
    def test_session_retrieval_performance(self, conversation_store):
        """Test performance of session-based message retrieval."""
        # Setup test data
        for i in range(500):
            conversation_store.add_message(
                session_id=f'session_{i % 20}',
                role='user' if i % 2 == 0 else 'assistant',
                content=f'Message {i}'
            )
        
        import time
        
        # Test retrieval performance for specific session
        start_time = time.time()
        messages = conversation_store.get_messages(session_id='session_5', limit=50)
        end_time = time.time()
        
        retrieval_time = end_time - start_time
        assert retrieval_time < 0.1  # Should be very fast with indexes
        assert len(messages) <= 50
        
        print(f"Retrieved session messages in {retrieval_time:.4f} seconds")
    
    def test_session_ids_retrieval_performance(self, conversation_store):
        """Test performance of session IDs retrieval."""
        # Setup test data with many unique sessions
        for i in range(1000):
            conversation_store.add_message(
                session_id=f'unique_session_{i}',
                role='user',
                content=f'Message in session {i}'
            )
        
        import time
        
        # Test session IDs retrieval performance
        start_time = time.time()
        session_ids = conversation_store.get_session_ids()
        end_time = time.time()
        
        retrieval_time = end_time - start_time
        assert retrieval_time < 0.5  # Should be reasonably fast
        assert len(session_ids) == 1000
        
        print(f"Retrieved {len(session_ids)} session IDs in {retrieval_time:.4f} seconds")
    
    def test_database_operations_consistency(self, conversation_store):
        """Test that database operations remain consistent with indexes."""
        # Test add_message
        message_id = conversation_store.add_message(
            session_id='test_session',
            role='user',
            content='Test message'
        )
        assert message_id is not None
        
        # Test get_messages
        messages = conversation_store.get_messages(session_id='test_session')
        assert len(messages) == 1
        assert messages[0]['content'] == 'Test message'
        
        # Test delete_session
        conversation_store.delete_session('test_session')
        messages_after_delete = conversation_store.get_messages(session_id='test_session')
        assert len(messages_after_delete) == 0