# vector_database.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the embedding model and the FAISS vector index.
        We use a lightweight HuggingFace model optimized for fast, local CPU embeddings.
        """
        self.model = SentenceTransformer(model_name)
        # MiniLM-L6-v2 outputs embeddings with 384 dimensions
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize a flat L2 (Euclidean) distance index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunk_map = {} # Maps FAISS integer IDs back to the actual text chunks

    def add_chunks(self, chunks):
        """
        Converts text chunks into mathematical vectors and loads them into the database.
        """
        if not chunks:
            return
            
        # 1. Generate dense vector embeddings for all chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        
        # 2. Add vectors to the FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # 3. Store the mapping so we can retrieve the exact text string later
        for i, chunk in enumerate(chunks):
            self.chunk_map[start_id + i] = chunk

    def search(self, query, top_k=2):
        """
        Searches the vector database for chunks mathematically similar to the query.
        """
        if self.index.ntotal == 0:
            return []
            
        # Embed the user's search query into the exact same vector space
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Perform the mathematical nearest-neighbor search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the actual text chunks based on the returned indices
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 if it can't find enough results
                results.append({
                    "text": self.chunk_map[idx],
                    "distance": float(distances[0][i])
                })
                
        return results


# --- Quick Test ---
if __name__ == "__main__":
    print("🧠 Booting Local Embedding Model (all-MiniLM-L6-v2)...")
    vector_store = VectorStore()
    
    # Simulating the overlapping chunks we generated on Day 1
    dummy_chunks = [
        "The EdgeVision architecture utilizes a decoupled microservice pattern.",
        "The camera client runs entirely on low-power IoT devices using OpenCV.",
        "The inference server requires an NVIDIA GPU with at least 4GB of VRAM.",
        "It runs a 2-billion parameter Vision-Language Model loaded in FP16 precision.",
        "Security protocols require all API endpoints to be secured via JWT authentication."
    ]
    
    print("\n📚 Generating dense embeddings and indexing into FAISS...")
    vector_store.add_chunks(dummy_chunks)
    print(f"Total chunks indexed: {vector_store.index.ntotal}")
    
    # Notice how the query doesn't share exact keywords with the target chunk, 
    # but the semantic meaning is identical.
    user_query = "What hardware do I need for the server?"
    print(f"\n🔍 Executing Vector Search for: '{user_query}'")
    
    results = vector_store.search(user_query, top_k=2)
    
    print("\n🏆 Top Results retrieved purely by mathematical meaning:")
    for idx, res in enumerate(results):
        print(f"Rank {idx+1} (L2 Distance: {res['distance']:.2f}) -> {res['text']}")