import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
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

# Load the Lift Maintenance dataset
data = pd.read_csv('C:/Users/Sai Baba Reddy/data science/LiftMaintainancePrediction.csv')

# Display basic information about the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Check for missing values and fill them if any
data.fillna(data.mean(), inplace=True)

# Preprocess the dataset
# Replace 'TEMPERATURE', 'CAPACITY', and 'OUTCOMES' with your actual feature and target column names
X = data[['TEMPERATURE', 'CAPACITY']].values  # Replace with the relevant feature column(s)
y = data['OUTCOMES'].values  # Replace with the target column

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and train the model
model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Predict and evaluate the model
predictions = model.predict(X)

# Plot the results
plt.scatter(X[:, 0], y, color='blue', label='Actual')  # Assuming TEMPERATURE as the feature to plot
plt.plot(X[:, 0], predictions, color='red', label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")
