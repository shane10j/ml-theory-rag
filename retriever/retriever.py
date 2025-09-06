from data.create_embeddings import embed

def query_vector_index(vector_index, query, k, chunks):
    query_embedding = embed([query])
    distances, indices = vector_index.search(query_embedding, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks