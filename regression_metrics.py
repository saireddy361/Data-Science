import numpy as np

# Define Evaluation Metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example Data (Simulated)
y = np.array([3, -0.5, 2, 7])
predictions = np.array([2.5, 0.0, 2, 8])

# Calculate Metrics
mse = mean_squared_error(y, predictions)
rmse = root_mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Print Results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
