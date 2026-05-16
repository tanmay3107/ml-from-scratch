# adam.py
import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the Adam Optimizer.
        :param learning_rate: The base step size (alpha).
        :param beta1: Decay rate for the first moment estimate (Momentum).
        :param beta2: Decay rate for the second moment estimate (RMSProp).
        :param epsilon: A tiny scalar added to the denominator to prevent division by zero.
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Memory dictionaries to store the moving averages for every parameter
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step counter for bias correction

    def update(self, params, grads):
        """
        Executes a single optimization step using Adam.
        :param params: A dictionary of current weights (e.g., {'W1': matrix, 'b1': matrix})
        :param grads: A dictionary of gradients matching the params (e.g., {'W1': dW1, 'b1': db1})
        :return: An updated params dictionary.
        """
        self.t += 1
        updated_params = {}

        for key in params.keys():
            # Initialize memory state for new parameters dynamically
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # Extract current weight and gradient
            weight = params[key]
            gradient = grads[key]

            # 1. Update biased first moment estimate (Momentum)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradient
            
            # 2. Update biased second raw moment estimate (RMSProp)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(gradient)

            # 3. Compute bias-corrected first moment estimate
            m_corrected = self.m[key] / (1 - np.power(self.beta1, self.t))
            
            # 4. Compute bias-corrected second raw moment estimate
            v_corrected = self.v[key] / (1 - np.power(self.beta2, self.t))

            # 5. Apply the final Adam weight update formula
            updated_weight = weight - self.lr * (m_corrected / (np.sqrt(v_corrected) + self.epsilon))
            
            updated_params[key] = updated_weight

        return updated_params

# --- Quick Test ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate a network layer with random weights
    params = {
        "W1": np.random.randn(3, 3) * 0.1,
        "b1": np.zeros((1, 3))
    }
    
    # Simulate the gradients calculated during backpropagation
    grads = {
        "W1": np.random.randn(3, 3) * 0.5, # High variance gradients
        "b1": np.array([[0.1, -0.2, 0.05]])
    }
    
    print("🧠 Initializing Adam Optimizer...")
    optimizer = AdamOptimizer(learning_rate=0.01)
    
    print("\nInitial W1[0,0]:", params["W1"][0, 0])
    print("Gradient dW1[0,0]:", grads["W1"][0, 0])
    
    # Run a few optimization steps
    for epoch in range(1, 4):
        params = optimizer.update(params, grads)
        print(f"Epoch {epoch} | Updated W1[0,0]: {params['W1'][0, 0]:.6f}")
        
    print("\n✅ Notice how the step sizes adaptively change rather than jumping linearly! Adam is working.")