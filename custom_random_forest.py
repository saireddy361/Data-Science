import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees=10, max_depth=3, bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y): 
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=self.bootstrap)
        return X[indices], y[indices]

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(tree_predictions.mean(axis=0))

# Sample data definition
# X = features (e.g., random data for 10 samples and 3 features)
# y = labels (e.g., binary classification)
X = np.random.rand(10, 3)  # 10 samples, 3 features
y = np.random.randint(0, 2, size=10)  # 10 binary labels (0 or 1)

# Example Usage
forest = RandomForest(n_trees=10, max_depth=3)
forest.fit(X, y)
predictions = forest.predict(X)

# Output the predictions
print("Predictions:", predictions)
