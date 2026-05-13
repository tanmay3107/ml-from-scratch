# multiclass_mlp.py
import numpy as np

def relu(z):
    """Applies the Rectified Linear Unit activation function element-wise."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Computes the gradient of the ReLU function."""
    return (z > 0).astype(float)

def softmax(z):
    """
    Computes standard Softmax probabilities across final logits.
    Includes numerical stability mapping to prevent float overflow.
    """
    # Shift values by subtracting the maximum row logit
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class MultiClassMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initializes weights using scaled standard normal distributions.
        """
        self.lr = learning_rate
        
        # He/Kaiming Initialization scaling works best for ReLU layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Executes the forward transformation pipeline."""
        # Layer 1: Input -> Hidden (ReLU)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        
        # Layer 2: Hidden -> Output (Softmax logits)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        
        return self.A2

    def backward(self, X, y_onehot):
        """
        Executes backpropagation using cached forward tensor operations.
        y_onehot: Ground truth matrix of shape (n_samples, n_classes).
        """
        m = X.shape[0]
        
        # --- Output Layer Gradients ---
        # Beautifully simplifies to (Predictions - Ground Truth)
        dZ2 = self.A2 - y_onehot
        dW2 = (1.0 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1.0 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # --- Hidden Layer Gradients ---
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        
        dW1 = (1.0 / m) * np.dot(X.T, dZ1)
        db1 = (1.0 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Parameter Updates ---
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y_onehot, epochs=5000):
        """Iterative optimization training loop."""
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y_onehot)
            
            if epoch % 1000 == 0:
                loss = self._categorical_cross_entropy(y_onehot, self.A2)
                # Compute raw training accuracy dynamically
                preds = np.argmax(self.A2, axis=1)
                targets = np.argmax(y_onehot, axis=1)
                acc = np.mean(preds == targets) * 100
                print(f"🔄 Epoch {epoch:4d} | Loss: {loss:.4f} | Training Accuracy: {acc:.1f}%")

    def _categorical_cross_entropy(self, y_true, y_pred):
        """Computes scalar multi-class categorical cross-entropy loss safely."""
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def predict(self, X):
        """Returns discrete class indices with maximum computed probability."""
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)


# --- Quick Test ---
if __name__ == "__main__":
    # Generate a challenging non-linear 3-class spiral dataset
    np.random.seed(42)
    N = 100  # number of points per class
    D = 2    # dimensionality
    K = 3    # number of classes
    
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')
    
    print("📊 Generating 3-Class non-linear spiral distribution...")
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1.0, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
        
    # Convert scalar class targets to One-Hot encoded matrix
    y_onehot = np.eye(K)[y]
    
    print("🧠 Initializing deep Multi-Class Network architecture...")
    # Using 64 hidden units to comfortably map complex spiral boundary layers
    model = MultiClassMLP(input_size=2, hidden_size=64, output_size=3, learning_rate=1.0)
    
    model.fit(X, y_onehot, epochs=5001)
    
    # Evaluate final learned representation boundaries
    final_preds = model.predict(X)
    final_acc = np.mean(final_preds == y) * 100
    print(f"\n✅ Optimization Complete. Final Target Accuracy mapping: {final_acc:.2f}%")