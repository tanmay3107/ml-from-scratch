# self_attention.py
import numpy as np

def softmax(z, axis=-1):
    """
    Computes numerically stable softmax along a specific axis.
    """
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

class ScaledDotProductAttention:
    def __init__(self, embedding_dim, attention_dim):
        """
        Initializes the Self-Attention mechanism.
        :param embedding_dim: The size of the incoming word vectors.
        :param attention_dim: The projected size for our Q, K, and V vectors (d_k).
        """
        self.d_k = attention_dim
        
        # Weight matrices to project the input embeddings into Q, K, and V spaces
        self.W_Q = np.random.randn(embedding_dim, attention_dim) * np.sqrt(2.0 / embedding_dim)
        self.W_K = np.random.randn(embedding_dim, attention_dim) * np.sqrt(2.0 / embedding_dim)
        self.W_V = np.random.randn(embedding_dim, attention_dim) * np.sqrt(2.0 / embedding_dim)

    def forward(self, X):
        """
        Executes the Self-Attention forward pass.
        X shape: (sequence_length, embedding_dim) -> e.g., 5 words, each represented by a 10D vector
        """
        # 1. Project inputs into Queries, Keys, and Values
        self.Q = np.dot(X, self.W_Q)  # Shape: (seq_len, d_k)
        self.K = np.dot(X, self.W_K)  # Shape: (seq_len, d_k)
        self.V = np.dot(X, self.W_V)  # Shape: (seq_len, d_k)
        
        # 2. Calculate raw attention scores: Q dot K-transpose
        # This gives us a (seq_len, seq_len) matrix showing how every word relates to every other word
        scores = np.dot(self.Q, self.K.T)
        
        # 3. Scale the scores to stabilize gradients
        scaled_scores = scores / np.sqrt(self.d_k)
        
        # 4. Apply Softmax to get probability weights (rows sum to 1.0)
        self.attention_weights = softmax(scaled_scores, axis=-1)
        
        # 5. Multiply weights by Values to get the final output context vectors
        context_output = np.dot(self.attention_weights, self.V)
        
        return context_output, self.attention_weights

# --- Quick Test ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate a sentence of 4 words, where each word is a 6-dimensional embedding
    # Think of this as the sequence: "The", "bank", "of", "the", "river"
    sequence_length = 5
    embedding_dim = 6
    X = np.random.randn(sequence_length, embedding_dim)
    
    print("🧠 Initializing Scaled Dot-Product Attention (Transformer Engine)...")
    attention_layer = ScaledDotProductAttention(embedding_dim=6, attention_dim=8)
    
    print("➡️ Executing Forward Pass...")
    output, weights = attention_layer.forward(X)
    
    print(f"\nOutput Context Tensor Shape: {output.shape} | Expected: (5, 8)")
    
    print("\n🔍 Attention Weights Matrix (5x5):")
    # Rounding for clean visualization
    print(np.round(weights, 2))
    
    print("\nNotice how the rows sum to exactly 1.0. Each row represents a word")
    print("and the columns show what percentage of 'attention' it is paying to the other words in the sequence!")s