import numpy as np

def k_fold_cross_validation(X, y, k=5):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds

# Example Usage:
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

folds = k_fold_cross_validation(X, y, k=5)
for i in range(5):
    test_idx = folds[i]
    train_idx = np.hstack([folds[j] for j in range(5) if j != i])
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Example: print train and test data for fold i
    print(f"Fold {i+1}:")
    print("Training Data:", X_train)
    print("Testing Data:", X_test)
    print("Training Labels:", y_train)
    print("Testing Labels:", y_test)
