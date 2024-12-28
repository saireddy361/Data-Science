import numpy as np
from sklearn.tree import DecisionTreeRegressor  # Using DecisionTreeRegressor for regression

class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        residuals = y
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Example Usage
# Sample data definition
X = np.random.rand(10, 3)  # 10 samples, 3 features
y = np.random.rand(10)  # 10 continuous target values

model = GradientBoosting(n_estimators=10, learning_rate=0.1)
model.fit(X, y)
predictions = model.predict(X)

# Output predictions
print("Predictions:", predictions)
