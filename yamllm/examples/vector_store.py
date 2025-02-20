from yamllm.memory import VectorStore

vector_store = VectorStore()
vectors, metadata = vector_store.get_vec_and_text()
print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimension: {vectors.shape[1] if len(vectors) > 0 else 0}")
print(f"Number of metadata entries: {len(metadata)}")

print(vectors)
print(metadata)