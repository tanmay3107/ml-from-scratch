# word2vec.py
import numpy as np

def softmax(z):
    """Numerically stable softmax for probability distribution."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        """
        Initializes the Skip-Gram Neural Network.
        :param vocab_size: Total number of unique words in the corpus.
        :param embedding_dim: The number of dimensions for our dense vectors (e.g., 10).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # W1 is the actual Embedding Matrix we want to extract later!
        # Shape: (vocab_size, embedding_dim)
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # W2 is the context weight matrix 
        # Shape: (embedding_dim, vocab_size)
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.1

    def forward(self, X):
        """
        Forward pass.
        X: One-hot encoded center word of shape (1, vocab_size).
        """
        # Input -> Hidden Layer (This mathematical dot product simply extracts the embedding row!)
        self.h = np.dot(X, self.W1)
        
        # Hidden -> Output Layer (Predicting the context word)
        self.u = np.dot(self.h, self.W2)
        self.y_pred = softmax(self.u)
        
        return self.y_pred

    def backward(self, X, y_true):
        """
        Backpropagation to update the embedding weights.
        y_true: One-hot encoded target context word.
        """
        # 1. Output layer error
        e = self.y_pred - y_true
        
        # 2. Gradients for W2
        dW2 = np.dot(self.h.T, e)
        
        # 3. Gradients for W1 (The Embeddings)
        # Propagate the error backwards through W2 to update W1
        dh = np.dot(e, self.W2.T)
        dW1 = np.dot(X.T, dh)
        
        # 4. Gradient Descent Update
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2

    def extract_embeddings(self):
        """Returns the learned dense vector representation for every word."""
        return self.W1

# --- Data Preparation Helpers ---
def generate_training_data(corpus, window_size=2):
    """
    Generates center-context word pairs via a sliding window.
    """
    vocab = list(set(corpus.split()))
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    words = corpus.split()
    training_data = []
    
    for i, center_word in enumerate(words):
        # Determine the sliding window bounds
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j: # Don't pair the center word with itself
                context_word = words[j]
                training_data.append((word2idx[center_word], word2idx[context_word]))
                
    return training_data, word2idx, vocab

# --- Quick Test ---
if __name__ == "__main__":
    # A focused corpus about algorithms and engineering
    text = "the mlops engineer optimized the deep learning model for real time deployment"
    
    print("📚 Processing corpus and building vocabulary...")
    training_data, word2idx, vocab = generate_training_data(text, window_size=2)
    vocab_size = len(vocab)
    
    print(f"🧠 Initializing Word2Vec Architecture (Vocab Size: {vocab_size}, Embedding Dim: 3)...")
    model = Word2Vec(vocab_size=vocab_size, embedding_dim=3, learning_rate=0.05)
    
    print("🔄 Training neural network to learn spatial context (1000 Epochs)...")
    for epoch in range(1000):
        loss = 0
        for center_idx, context_idx in training_data:
            # Create one-hot vectors
            X = np.zeros((1, vocab_size))
            X[0, center_idx] = 1
            
            y_true = np.zeros((1, vocab_size))
            y_true[0, context_idx] = 1
            
            # Forward and backward pass
            y_pred = model.forward(X)
            model.backward(X, y_true)
            
            # Accumulate cross-entropy loss
            loss -= np.log(y_pred[0, context_idx])
            
        if epoch % 500 == 0:
            print(f"   Epoch {epoch} | Loss: {loss:.4f}")

    # The grand reveal: We extract W1 as our embeddings!
    embeddings = model.extract_embeddings()
    
    print("\n✅ Final Learned Dense Embeddings (First 3 words):")
    for i in range(3):
        word = vocab[i]
        vector = embeddings[i]
        print(f"'{word}': {np.round(vector, 3)}")