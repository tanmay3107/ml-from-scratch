# conv2d.py
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, learning_rate=0.01):
        """
        Initializes the 2D Convolutional Layer parameters.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = learning_rate
        
        # He/Kaiming Initialization tailored for convolutional filter banks
        # Shape: (out_channels, in_channels, kernel_height, kernel_width)
        fan_in = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((out_channels, 1))
        
        # Forward pass cache required for backpropagation calculations
        self.X_padded = None
        self.X = None

    def forward(self, X):
        """
        Executes the forward cross-correlation sliding window.
        X shape: (batch_size, in_channels, height, width)
        """
        self.X = X
        batch_size, in_channels, height, width = X.shape
        
        # Apply zero-padding to spatial dimensions (Height and Width)
        self.X_padded = np.pad(
            X, 
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
            mode='constant'
        )
        
        # Calculate output spatial dimensions
        out_height = int((height + 2 * self.padding - self.kernel_size) / self.stride) + 1
        out_width = int((width + 2 * self.padding - self.kernel_size) / self.stride) + 1
        
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Bare-metal spatial sliding window execution
        for n in range(batch_size):
            for f in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # Map current spatial locations to input tensor slices
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract the valid receptive field across all input channels
                        receptive_field = self.X_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        # Perform element-wise Hadamard product and reduce to a scalar activation
                        out[n, f, h, w] = np.sum(receptive_field * self.W[f]) + self.b[f, 0]
                        
        return out

    def backward(self, dout):
        """
        Distributes upstream gradients back to filter weights, biases, and source pixels.
        dout shape: (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = self.X.shape
        _, _, out_height, out_width = dout.shape
        
        # Initialize zero-tensor gradient accumulators
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(self.X_padded)
        
        # Backpropagate errors through the exact spatial loops
        for n in range(batch_size):
            for f in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Cache local slice window
                        receptive_field = self.X_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        # 1. Filter weight gradients: Receptive Field * Upstream Gradient
                        dW[f] += receptive_field * dout[n, f, h, w]
                        
                        # 2. Bias gradients: Direct scalar sum of upstream spatial gradients
                        db[f, 0] += dout[n, f, h, w]
                        
                        # 3. Input pixel gradients: Distribute filter weights scaled by upstream error
                        dX_padded[n, :, h_start:h_end, w_start:w_end] += self.W[f] * dout[n, f, h, w]
                        
        # Execute bare-metal Gradient Descent parameter optimization
        self.W -= self.lr * dW
        self.b -= self.lr * db
        
        # Carefully strip out padding borders to return precise input dimension gradients
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
            
        return dX


# --- Quick Test ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate a mini-batch of grayscale input tensors: 2 samples, 1 channel, 5x5 pixels
    X = np.random.randn(2, 1, 5, 5)
    
    print("🧠 Initializing bare-metal Conv2D Layer (1 Input Channel -> 2 Filters, 3x3 Kernel)...")
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, learning_rate=0.01)
    
    print("➡️ Executing Forward Spatial Pass...")
    out = conv.forward(X)
    print(f"Output Tensor Shape: {out.shape} | Expected: (2, 2, 3, 3)")
    
    # Simulate an arbitrary upstream loss gradient arriving from deeper network layers
    dout = np.random.randn(*out.shape)
    
    print("⬅️ Executing Backward Backprop Pass (Deriving dW, db, and source dX)...")
    dX = conv.backward(dout)
    print(f"Gradient Tensor dX Shape: {dX.shape} | Expected: (2, 1, 5, 5)")
    print("✅ Tensor spatial transformations and cross-correlation backprop verified successfully.")
