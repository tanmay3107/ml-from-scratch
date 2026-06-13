# rag_agent.py
from transformers import pipeline
from retrieval_pipeline import RAGRetrievalPipeline

class LocalRAGAgent:
    def __init__(self, llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initializes the complete RAG system with a local LLM.
        We default to TinyLlama as it is highly optimized for local CPU/GPU inference.
        """
        self.retriever = RAGRetrievalPipeline(chunk_size=50, chunk_overlap=10)
        
        print(f"🤖 Booting Local Generative Engine ({llm_model})...")
        # Initialize the Hugging Face text-generation pipeline
        self.generator = pipeline(
            "text-generation", 
            model=llm_model, 
            device_map="auto" # Automatically maps to GPU if available, else CPU
        )

    def ingest(self, text):
        """Wrapper to feed documents into the vector database."""
        self.retriever.ingest_document(text)

    def ask(self, user_question):
        """
        The core RAG loop: Retrieve context -> Build Prompt -> Generate Answer.
        """
        # 1. Retrieve the mathematically relevant facts from FAISS
        context = self.retriever.retrieve_context(user_question, top_k=2)
        
        if not context:
            return "I'm sorry, I cannot find any relevant information in the provided documents to answer your question."

        # 2. Construct the strict prompt template
        # We explicitly command the model to rely ONLY on the provided context.
        prompt = f"""<|system|>
You are a highly precise technical assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I do not know." Do not hallucinate.
Context:
{context}
</s>
<|user|>
{user_question}
</s>
<|assistant|>
"""
        
        print("🧠 Generating response via Local LLM...")
        # 3. Generate the response
        outputs = self.generator(
            prompt, 
            max_new_tokens=150, 
            temperature=0.1, # Low temperature forces factual, deterministic answers
            do_sample=True,
            return_full_text=False # Only return the newly generated answer, not the prompt
        )
        
        # Extract the text from the Hugging Face output object
        answer = outputs[0]['generated_text'].strip()
        return answer


# --- Quick Test ---
if __name__ == "__main__":
    # A fresh technical document to test against
    server_docs = """
    Debales.ai Infrastructure Deployment Policies.
    All inference servers must be provisioned with Ubuntu 22.04 LTS.
    The primary database cluster runs PostgreSQL 15, and backups are executed asynchronously every 12 hours.
    For local LLM hosting, we utilize the vLLM engine to optimize memory paging and KV cache allocation.
    To prevent out-of-memory (OOM) errors, batch sizes are strictly capped at 16 requests per node.
    """
    
    agent = LocalRAGAgent()
    
    print("\n📚 Loading internal documents into the RAG Agent...")
    agent.ingest(server_docs)
    
    # Let's test the system's ability to reason over the injected context
    question = "How do we prevent out-of-memory errors on the local LLM servers?"
    
    print(f"\n👤 User: {question}")
    answer = agent.ask(question)
    
    print(f"\n🤖 RAG Agent: {answer}")