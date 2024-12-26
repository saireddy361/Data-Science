import numpy as np 
import pandas as pd 

class LinearRegressionScratch: 
    def __init__(self, learning_rate=0.01, n_iterations=1000): 
        self.lr = learning_rate 
        self.n_iter = n_iterations 
        self.weights = None 
        self.bias = None 

    def fit(self, X, y): 
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features) 
        self.bias = 0 
        for _ in range(self.n_iter): 
            y_pred = np.dot(X, self.weights) + self.bias 
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) 
            db = (1/n_samples) * np.sum(y_pred - y) 
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db 

    def predict(self, X): 
        return np.dot(X, self.weights) + self.bias

# Assuming you have a dataset loaded into pandas DataFrame `df`
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 3, 4, 5, 6],
    'target': [5, 7, 9, 11, 13]
})

X = df[['feature1', 'feature2']].values
y = df['target'].values

model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)

print("Predictions:", predictions)
