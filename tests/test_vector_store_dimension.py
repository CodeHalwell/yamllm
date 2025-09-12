from pathlib import Path

import pytest

from yamllm.memory.conversation_store import VectorStore
from yamllm.core.exceptions import MemoryError


def test_add_vector_dimension_mismatch_raises(tmp_path: Path):
    store = VectorStore(store_path=str(tmp_path), vector_dim=8)
    good = [0.0] * 8
    bad = [0.0] * 7

    store.add_vector(good, 1, "hello", "user")  # should succeed

    with pytest.raises(MemoryError):
        store.add_vector(bad, 2, "world", "assistant")


def test_loading_existing_index_with_wrong_dimension_raises(tmp_path: Path):
    # Create an index at dim=8
    store1 = VectorStore(store_path=str(tmp_path), vector_dim=8)
    store1.add_vector([0.0] * 8, 1, "hello", "user")

    # Attempt to load the same store with a different dimension
    with pytest.raises(MemoryError) as ei:
        VectorStore(store_path=str(tmp_path), vector_dim=16)

    assert "Vector index dimension mismatch" in str(ei.value)
    assert "migrate-index" in str(ei.value)

