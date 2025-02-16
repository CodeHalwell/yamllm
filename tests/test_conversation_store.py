import pytest
from yamllm.memory.conversation_store import ConversationStore, VectorStore

@pytest.fixture
def conversation_store(tmp_path):
    db_path = tmp_path / "test_conversation_history.db"
    store = ConversationStore(db_path=str(db_path))
    store.create_db()
    return store

def test_db_exists(conversation_store):
    assert conversation_store.db_exists() is True

def test_add_message(conversation_store):
    session_id = "session1"
    role = "user"
    content = "Hello, world!"
    message_id = conversation_store.add_message(session_id, role, content)
    assert isinstance(message_id, int)

def test_get_messages(conversation_store):
    session_id = "session1"
    role = "user"
    content = "Hello, world!"
    conversation_store.add_message(session_id, role, content)
    messages = conversation_store.get_messages(session_id=session_id)
    assert len(messages) == 1
    assert messages[0]["role"] == role
    assert messages[0]["content"] == content

@pytest.fixture
def vector_store(tmp_path):
    store_path = tmp_path / "vector_store"
    store = VectorStore(store_path=str(store_path))
    return store

def test_add_vector(vector_store):
    vector = [0.1] * 1536
    message_id = 1
    content = "Hello, world!"
    role = "user"
    vector_store.add_vector(vector, message_id, content, role)
    assert len(vector_store.metadata) == 1
    assert vector_store.metadata[0]["id"] == message_id
    assert vector_store.metadata[0]["content"] == content
    assert vector_store.metadata[0]["role"] == role

def test_search(vector_store):
    vector = [0.1] * 1536
    message_id = 1
    content = "Hello, world!"
    role = "user"
    vector_store.add_vector(vector, message_id, content, role)
    query_vector = [0.1] * 1536
    results = vector_store.search(query_vector, k=1)
    assert len(results) == 1
    assert results[0]["id"] == message_id
    assert results[0]["content"] == content
    assert results[0]["role"] == role
    assert "similarity" in results[0]