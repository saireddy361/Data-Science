import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def k_fold_cross_validation(X, y, k):
    # Implement k-fold splitting logic here, returning k subsets for cross-validation
    folds = np.array_split(np.random.permutation(len(X)), k)
    return folds

def evaluate_metric(y_true, y_pred):
    # Use MSE as the evaluation metric
    return mean_squared_error(y_true, y_pred)

def random_search(X, y, model_class, param_distributions, n_iter=10, k=5):
    folds = k_fold_cross_validation(X, y, k)
    best_score = -np.inf
    best_params = None

    param_keys = list(param_distributions.keys())
    
    for _ in range(n_iter):
        param_dict = {key: random.choice(param_distributions[key]) for key in param_keys}
        print(f"Testing parameters: {param_dict}")  # Debugging line to show parameters
        scores = []
        
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Normalize data if normalize=True in the parameters
            if param_dict['normalize']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Create and fit the model with the parameters
            model = model_class(fit_intercept=param_dict['fit_intercept'])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = evaluate_metric(y_test, predictions)  # Evaluate using metric
            scores.append(score)
        
        avg_score = np.mean(scores)
        print(f"Avg Score: {avg_score}")  # Debugging line to show the average score

        if avg_score > best_score:
            best_score = avg_score
            best_params = param_dict
            print(f"New best score: {best_score} with params: {best_params}")  # Debugging line for the best score

    return best_params, best_score

# Example usage with California Housing dataset:
# Load California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

param_distributions = {
     'fit_intercept': [True, False],
     'normalize': [True, False]  # Handle normalization manually
}

# Call random search
best_params, best_score = random_search(X, y, LinearRegression, param_distributions, n_iter=20, k=5)

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
