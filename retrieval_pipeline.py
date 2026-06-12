# retrieval_pipeline.py
from document_processor import DocumentProcessor
from vector_database import VectorStore

class RAGRetrievalPipeline:
    def __init__(self, chunk_size=500, chunk_overlap=50, model_name='all-MiniLM-L6-v2'):
        """
        Orchestrates the ingestion, embedding, indexing, and retrieval steps.
        """
        self.processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(model_name=model_name)

    def ingest_document(self, raw_text):
        """
        Processes a raw document string, breaks it into overlapping chunks, 
        and updates the FAISS vector database.
        """
        print("📄 Step 1: Chunking raw text document...")
        chunks = self.processor.chunk_text(raw_text)
        print(f"   Generated {len(chunks)} chunks.")
        
        print("🧠 Step 2: Generating dense vector embeddings and indexing in FAISS...")
        self.vector_store.add_chunks(chunks)
        print("✅ Ingestion complete.")

    def retrieve_context(self, query, top_k=2, distance_threshold=1.5):
        """
        Queries the vector index and constructs a clean context block for the LLM.
        Filters out matches that are mathematically too distant (irrelevant noise).
        """
        print(f"🔍 Searching index for query: '{query}'")
        raw_results = self.vector_store.search(query, top_k=top_k)
        
        filtered_chunks = []
        for result in raw_results:
            # In FAISS IndexFlatL2, lower distances indicate higher semantic similarity
            if result['distance'] <= distance_threshold:
                filtered_chunks.append(result['text'])
            else:
                print(f"⚠️ Omitted a chunk due to high distance score ({result['distance']:.2f} > {distance_threshold})")
                
        # Combine the relevant chunks into a structured text prompt block
        structured_context = "\n---\n".join(filtered_chunks)
        return structured_context


# --- Quick Test ---
if __name__ == "__main__":
    # Simulate an internal system handbook
    technical_manual = """
    FairVision-Auditor System Architecture Manual.
    The evaluation architecture consists of an auditing core, a slicing engine, and a mitigation pipeline.
    The metrics module calculates Demographic Parity Difference (DPD) and Disparate Impact (DI).
    
    To run an audit, instantiate the SlicingEvaluator with a trained PyTorch model and a validation DataLoader.
    The evaluator runs a memory-optimized loop using torch.no_grad() to safely aggregate metrics.
    
    The mitigation layer applies Hybrid Augmentation dynamically inside the __getitem__ method of the dataset.
    It forces the network to become invariant to lighting artifacts and color shifts using ColorJitter.
    The target benchmarking objective is to ensure the mitigated model's accuracy on unprivileged slices strictly exceeds the baseline model's accuracy.
    """
    
    print("🚀 Initializing complete RAG Retrieval Pipeline...")
    pipeline = RAGRetrievalPipeline(chunk_size=30, chunk_overlap=5)
    
    # Ingesting the document
    pipeline.ingest_document(technical_manual)
    
    # Test Query 1: Relevant query
    query_1 = "How does the pipeline handle bias or mitigation?"
    print(f"\n--- Running Pipeline for Query 1 ---")
    context_1 = pipeline.retrieve_context(query_1, top_k=2)
    print("\n📦 Structured Context Block for LLM:")
    print(context_1 if context_1 else "[No relevant context found]")
    
    # Test Query 2: Irrelevant query (should be filtered out by thresholding)
    query_2 = "What is the capital of France?"
    print(f"\n--- Running Pipeline for Query 2 ---")
    context_2 = pipeline.retrieve_context(query_2, top_k=1, distance_threshold=1.2)
    print("\n📦 Structured Context Block for LLM:")
    print(context_2 if context_2 else "[No relevant context found]")