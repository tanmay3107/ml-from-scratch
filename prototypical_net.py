# prototypical_net.py
import numpy as np

def euclidean_distance(A, B):
    """
    Computes the squared Euclidean distance matrix between two sets of vectors.
    A: (num_queries, embedding_dim)
    B: (num_prototypes, embedding_dim)
    Returns: (num_queries, num_prototypes)
    """
    # Efficient vectorized distance computation: (a-b)^2 = a^2 - 2ab + b^2
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True).T
    dot_AB = np.dot(A, B.T)
    
    # Clip to prevent negative near-zero floats from numerical instability
    distances = np.maximum(sq_A - 2 * dot_AB + sq_B, 0.0)
    return distances

def softmax(z, axis=-1):
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

class PrototypicalNetwork:
    def __init__(self, n_way, k_shot):
        """
        Initializes the Few-Shot Metric Learning engine.
        :param n_way: Number of classes in the episode.
        :param k_shot: Number of support examples per class.
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.prototypes = None

    def compute_prototypes(self, support_embeddings):
        """
        Calculates the class centroids (prototypes).
        support_embeddings: Shape (n_way * k_shot, embedding_dim).
                            Assumes data is sorted by class!
        """
        embedding_dim = support_embeddings.shape[1]
        
        # Reshape to (n_way, k_shot, embedding_dim)
        support_reshaped = support_embeddings.reshape(self.n_way, self.self.k_shot, embedding_dim)
        
        # Calculate the mean across the k_shot dimension (axis=1)
        self.prototypes = np.mean(support_reshaped, axis=1)
        return self.prototypes

    def forward(self, support_embeddings, query_embeddings):
        """
        Executes an N-way K-shot episodic forward pass.
        Returns the probability distribution for the query set.
        """
        # 1. Build the geometric centers for each class
        self.compute_prototypes(support_embeddings)
        
        # 2. Calculate squared Euclidean distance from queries to prototypes
        distances = euclidean_distance(query_embeddings, self.prototypes)
        
        # 3. Convert distances to probabilities (Negative distance so closer = higher score)
        logits = -distances
        probabilities = softmax(logits)
        
        return probabilities, distances


# --- Quick Test ---
if __name__ == "__main__":
    np.random.seed(42)
    
    N_WAY = 3  # 3 distinct medical conditions
    K_SHOT = 5 # Only 5 reference images per condition!
    EMBEDDING_DIM = 64 # e.g., output from a ResNet-18 backbone
    NUM_QUERIES = 4
    
    print(f"🧠 Initializing Prototypical Network ({N_WAY}-way, {K_SHOT}-shot)...")
    protonet = PrototypicalNetwork(n_way=N_WAY, k_shot=K_SHOT)
    
    # Simulate feature embeddings extracted from a backbone network
    # Support set: 15 examples total (3 classes * 5 shots), sorted by class
    support_set = np.random.randn(N_WAY * K_SHOT, EMBEDDING_DIM)
    
    # Query set: 4 new patients to diagnose
    query_set = np.random.randn(NUM_QUERIES, EMBEDDING_DIM)
    
    print("➡️ Computing class prototypes and executing metric inference...")
    probabilities, distances = protonet.forward(support_set, query_set)
    
    print("\n📊 Extracted Prototypes Shape:", protonet.prototypes.shape)
    
    print("\n🎯 Query Set Predictions:")
    for i in range(NUM_QUERIES):
        predicted_class = np.argmax(probabilities[i])
        confidence = probabilities[i, predicted_class] * 100
        print(f"Patient {i+1} -> Assigned to Class {predicted_class} (Confidence: {confidence:.1f}%)")
        
    print("\n✅ Notice how classification is handled purely by vector mathematics in latent space, completely bypassing traditional dense output layers!")