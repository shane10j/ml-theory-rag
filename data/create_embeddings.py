from sentence_transformers import SentenceTransformer
import faiss
import tqdm


#Create embeddings from the chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
def embed(chunks):
    texts = [" ".join(chunk) for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=True, show_progress_bar=True)
    return embeddings


#Build Vector Index
def create_vector_index(embeddings):
    D = embeddings.shape[1]
    vector_index = faiss.IndexFlatIP(D)
    vector_index.add(embeddings)
    return vector_index
