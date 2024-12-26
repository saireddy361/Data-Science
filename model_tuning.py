import itertools
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def k_fold_cross_validation(X, y, k):
    # This function performs k-fold cross-validation and returns the indices of each fold.
    fold_size = len(X) // k
    folds = [list(range(i * fold_size, (i + 1) * fold_size)) for i in range(k)]
    return folds

def evaluate_metric(y_true, y_pred):
    # Example evaluation metric: Mean Squared Error (use your desired metric)
    return np.mean((y_true - y_pred) ** 2)

def grid_search(X, y, model_class, param_grid, k=5):
    folds = k_fold_cross_validation(X, y, k)
    param_combinations = list(itertools.product(*param_grid.values()))
    best_score = float('inf')
    best_params = None

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        scores = []
        
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Handle normalization (using StandardScaler)
            normalize = param_dict.get('normalize', False)  # Use get() to handle missing 'normalize'
            if 'normalize' in param_dict:
                del param_dict['normalize']  # Remove normalize from param_dict if it exists
            if normalize:  # Add normalization only if it's True
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Now we can create the model without the 'normalize' parameter
            model = model_class(**param_dict)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = evaluate_metric(y_test, predictions)  # Define your evaluation metric
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = param_dict.copy()

    return best_params, best_score

# Example Usage
# Generate a simple regression dataset for demonstration
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# Define a parameter grid for the model (remove 'normalize' from LinearRegression)
param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False]  # Handle normalization separately
}

# Perform grid search using Linear Regression
best_params, best_score = grid_search(X, y, LinearRegression, param_grid)

print("Best Parameters:", best_params)
print("Best Score:", best_score)
