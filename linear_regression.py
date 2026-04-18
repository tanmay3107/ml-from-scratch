# linear_regression.py
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the Linear Regression model.
        :param learning_rate: Step size for gradient descent
        :param n_iterations: Number of times to loop through the dataset
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the model using Gradient Descent.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent Loop
        for _ in range(self.n_iterations):
            # Calculate predictions: y = X*w + b
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicts target values for new data.
        """
        return np.dot(X, self.weights) + self.bias

# --- Quick Test ---
if __name__ == "__main__":
    # Generate some fake data: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])

    print("🧠 Training Linear Regression from scratch...")
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    print(f"✅ Learned Weights: {model.weights[0]:.4f} (Expected ~2.0)")
    print(f"✅ Learned Bias: {model.bias:.4f} (Expected ~1.0)")
    
    # Test a new prediction
    X_test = np.array([[6]])
    pred = model.predict(X_test)
    print(f"🎯 Prediction for X=6: {pred[0]:.4f} (Expected ~13.0)")