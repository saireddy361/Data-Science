import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Define the Linear Regression model (scratch implementation)
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
    
    def fit(self, X, y):
        m = len(X)
        X_b = np.c_[np.ones((m, 1)), X]  # Add x0 = 1 for the bias term
        self.theta = np.random.randn(X_b.shape[1])  # Random initialization
        for _ in range(self.n_iterations):
            gradients = 2/m * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients
    
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]  # Add x0 = 1 for the bias term
        return X_b.dot(self.theta)

# Generate synthetic data for Linear Regression
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100)

# Initialize and train the model
model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predictions, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")
