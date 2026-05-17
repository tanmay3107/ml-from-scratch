# dropout.py
import numpy as np

class Dropout:
    def __init__(self, drop_rate=0.5):
        """
        Initializes the Dropout layer.
        :param drop_rate: The probability of dropping a neuron (setting it to zero).
                          E.g., 0.5 means drop 50% of connections.
        """
        # We store the "keep probability" because the math is cleaner
        self.keep_prob = 1.0 - drop_rate
        self.mask = None
        self.is_training = True

    def forward(self, X):
        """
        Executes the forward pass using Inverted Dropout scaling.
        """
        # If we are testing/evaluating, do absolutely nothing
        if not self.is_training:
            return X
            
        # 1. Generate a binary mask of 0s and 1s based on the keep probability
        # np.random.rand generates numbers between 0 and 1. 
        # Checking if they are < keep_prob gives us our boolean mask.
        self.mask = (np.random.rand(*X.shape) < self.keep_prob)
        
        # 2. Apply the mask and apply Inverted Scaling
        # We divide by keep_prob so the expected value of the activations remains unchanged
        out = (X * self.mask) / self.keep_prob
        
        return out

    def backward(self, dout):
        """
        Executes backpropagation. Only the neurons that survived the forward pass
        are allowed to receive and pass back error gradients.
        """
        # If we are not training, gradients shouldn't be flowing anyway, but for safety:
        if not self.is_training:
            return dout
            
        # The gradient is multiplied by the EXACT same mask used in the forward pass,
        # and scaled by the same inverted dropout factor.
        dX = (dout * self.mask) / self.keep_prob
        
        return dX


# --- Quick Test ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate an intermediate hidden layer output tensor (e.g., 5 samples, 10 neurons)
    hidden_activations = np.ones((5, 10))
    
    print("🧠 Initializing Dropout Layer (Drop Rate: 50%)...")
    dropout_layer = Dropout(drop_rate=0.5)
    
    print("\n--- TRAINING MODE ---")
    dropout_layer.is_training = True
    train_out = dropout_layer.forward(hidden_activations)
    
    print("Forward Pass Output (Notice 0s and scaled 2.0s):")
    print(train_out[0]) # Print first sample's neurons
    
    # Simulate an upstream gradient of 1.0 everywhere
    dout = np.ones((5, 10))
    train_dx = dropout_layer.backward(dout)
    
    print("\nBackward Pass Gradients (Notice gradients only flow where mask is 1):")
    print(train_dx[0])
    
    print("\n--- INFERENCE / EVALUATION MODE ---")
    dropout_layer.is_training = False
    eval_out = dropout_layer.forward(hidden_activations)
    
    print("Forward Pass Output (Should be completely untouched 1.0s):")
    print(eval_out[0])