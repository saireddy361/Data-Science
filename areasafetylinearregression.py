import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Load the dataset
file_path = 'Area Safety Prediction.csv'
data = pd.read_csv(file_path)

# Preprocess the data
X = data.drop(columns=['area', 'outcome', 'class'])
y = data['outcome']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Plot the results
plt.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Outcome')
plt.ylabel('Predicted Outcome')
plt.title('Linear Regression on Area Safety Prediction')
plt.legend()
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")
