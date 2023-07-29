# train_model.py
import numpy as np
import joblib

# Define a simple neural network class
class SimpleNeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(4, 3)  # Random initial weights for 4 input features and 3 output classes

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict(self, X):
        return self.softmax(np.dot(X, self.weights))

# Load the Iris dataset
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Normalize the input data (optional, but recommended for neural networks)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Train the model
model = SimpleNeuralNetwork()
# Train the model using a simple gradient descent algorithm (for demonstration purposes)
for _ in range(1000):
    predictions = model.predict(X)
    error = predictions - np.eye(3)[y]  # One-hot encoding
    gradient = np.dot(X.T, error)
    model.weights -= 0.1 * gradient  # Learning rate: 0.1

# Save the model to a file
joblib.dump(model, 'model.h5')
