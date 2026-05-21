# rnn.py
import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initializes the Recurrent Neural Network.
        """
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # 1. Weights for Current Input -> Hidden State
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        
        # 2. Weights for Previous Hidden State -> Current Hidden State (THE MEMORY!)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        
        # 3. Weights for Current Hidden State -> Output
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        Executes the forward pass across a sequence of time steps.
        inputs: A list of sequence arrays (e.g., one-hot encoded words).
        """
        # Dictionary to store the hidden state at each time step
        self.h = {}
        # Initialize the very first memory state (t=-1) as all zeros
        self.h[-1] = np.zeros((self.hidden_size, 1))
        
        self.outputs = {}
        self.inputs = inputs
        
        # Loop through time!
        for t, x_t in enumerate(inputs):
            # The core RNN equation: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            self.h[t] = np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, self.h[t-1]) + self.b_h)
            
            # Compute output for this specific time step
            self.outputs[t] = np.dot(self.W_hy, self.h[t]) + self.b_y
            
        return self.outputs, self.h

    def backward(self, d_outputs):
        """
        Executes Backpropagation Through Time (BPTT).
        d_outputs: A dictionary of error gradients at each time step.
        """
        # Initialize zero-matrices to accumulate our gradients over time
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # This keeps track of the gradient flowing backward from the NEXT time step
        dh_next = np.zeros_like(self.h[0])
        
        # We step backward through time (from the last word to the first word)
        for t in reversed(range(len(self.inputs))):
            # 1. Gradients for Output Weights (Standard deep learning math)
            dW_hy += np.dot(d_outputs[t], self.h[t].T)
            db_y += d_outputs[t]
            
            # 2. Gradient flowing into the hidden state
            # It comes from TWO places: the output at step 't', AND the hidden state at step 't+1'
            dh = np.dot(self.W_hy.T, d_outputs[t]) + dh_next
            
            # 3. Backpropagate through the tanh activation function
            # The derivative of tanh(x) is (1 - tanh(x)^2)
            dtanh = (1 - self.h[t] ** 2) * dh
            
            # 4. Accumulate gradients for the memory and input weights
            db_h += dtanh
            dW_xh += np.dot(dtanh, self.inputs[t].T)
            dW_hh += np.dot(dtanh, self.h[t-1].T)
            
            # 5. Pass the gradient backward to the previous time step
            dh_next = np.dot(self.W_hh.T, dtanh)
            
        # Apply the Gradient Descent update
        # Clipping gradients to prevent the "Exploding Gradient" problem common in RNNs
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)
            
        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.W_hy -= self.lr * dW_hy
        self.b_h -= self.lr * db_h
        self.b_y -= self.lr * db_y

# --- Quick Test ---
if __name__ == "__main__":
    print("🧠 Initializing RNN architecture...")
    # Example: Vocabulary size of 4, hidden memory size of 10, output size of 4
    rnn = RNN(input_size=4, hidden_size=10, output_size=4, learning_rate=0.1)
    
    # Simulate a sequence of 3 words, one-hot encoded
    # e.g., mapping to the sequence: "I", "love", "code"
    sequence = [
        np.array([[1], [0], [0], [0]]),  # Word 1
        np.array([[0], [1], [0], [0]]),  # Word 2
        np.array([[0], [0], [1], [0]])   # Word 3
    ]
    
    print("➡️ Executing Forward Pass through time...")
    outputs, hidden_states = rnn.forward(sequence)
    
    print(f"Time steps processed: {len(outputs)}")
    print(f"Memory state shape at final step: {hidden_states[2].shape}")
    
    # Simulate some dummy loss gradients arriving from our loss function
    d_outputs = {
        0: np.random.randn(4, 1),
        1: np.random.randn(4, 1),
        2: np.random.randn(4, 1)
    }
    
    print("⬅️ Executing Backpropagation Through Time (BPTT)...")
    rnn.backward(d_outputs)
    print("✅ Memory gradients accumulated and clipped successfully.")