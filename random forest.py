import math
import random
from collections import Counter

# Helper functions (same as Decision Tree)
def calculate_entropy(data):
    total = len(data)
    if total == 0:
        return 0
    counts = {}
    for label in data:
        counts[label] = counts.get(label, 0) + 1
    entropy = 0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy

def split_data(dataset, feature_index):
    splits = {}
    for row in dataset:
        key = row[feature_index]
        if key not in splits:
            splits[key] = []
        splits[key].append(row)
    return splits

def calculate_information_gain(dataset, feature_index, target_index):
    total_entropy = calculate_entropy([row[target_index] for row in dataset])
    splits = split_data(dataset, feature_index)
    total_samples = len(dataset)
    
    weighted_entropy = 0
    for subset in splits.values():
        prob = len(subset) / total_samples
        subset_entropy = calculate_entropy([row[target_index] for row in subset])
        weighted_entropy += prob * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Decision Tree Classifier (same as before)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, dataset, features, target_index):
        self.tree = self._build_tree(dataset, features, target_index, depth=0)

    def _build_tree(self, dataset, features, target_index, depth):
        target_values = [row[target_index] for row in dataset]
        if len(set(target_values)) == 1:
            return target_values[0]
        if not features or (self.max_depth is not None and depth >= self.max_depth):
            return max(set(target_values), key=target_values.count)

        best_feature_index = -1
        best_gain = -float('inf')
        for i in range(len(features)):
            gain = calculate_information_gain(dataset, i, target_index)
            if gain > best_gain:
                best_gain = gain
                best_feature_index = i

        if best_gain == 0:
            return max(set(target_values), key=target_values.count)

        best_feature = features[best_feature_index]
        splits = split_data(dataset, best_feature_index)
        subtree = {}
        remaining_features = features[:best_feature_index] + features[best_feature_index + 1:]

        for value, subset in splits.items():
            subtree[value] = self._build_tree(subset, remaining_features, target_index, depth + 1)

        return {best_feature: subtree}

    def predict(self, row):
        node = self.tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]
            feature_index = features.index(feature)  # Fix: Find index of the feature
            value = row[feature_index]  # Access row by index, not name
            node = node[feature].get(value, None)
            if node is None:
                return None
        return node

# Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, dataset, features, target_index):
        self.trees = []
        for _ in range(self.n_trees):
            sample = self._bootstrap_sample(dataset)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample, features, target_index)
            self.trees.append(tree)

    def _bootstrap_sample(self, dataset):
        n_samples = self.sample_size or len(dataset)
        return [random.choice(dataset) for _ in range(n_samples)]

    def predict(self, row):
        predictions = [tree.predict(row) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]

# Example Usage
dataset = [
    ['Sunny', 'Hot', 'High', 'No'],
    ['Sunny', 'Hot', 'High', 'No'],
    ['Overcast', 'Hot', 'High', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Yes'],
    ['Sunny', 'Mild', 'High', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Yes'],
    ['Rainy', 'Mild', 'Normal', 'Yes']
]

features = ['Outlook', 'Temperature', 'Humidity']
target_index = 3

forest = RandomForest(n_trees=5, max_depth=3)
forest.fit(dataset, features, target_index)
print(forest.predict(['Sunny', 'Cool', 'High']))
