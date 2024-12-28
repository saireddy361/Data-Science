import pandas as pd

# Load the CSV file
file_path = 'c:/Users/Sai Baba Reddy/data science/LiftMaintainancePrediction.csv'
data = pd.read_csv(file_path)

from sklearn.cluster import 

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Select features for clustering
X = data[['TEMPERATURE', 'CAPACITY', 'SPEED', 'DOOR_OPENING_TIME', 'p', 'q', 'r']]

# Scale the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Apply KMeans with optimal K (assume 3 based on elbow plot)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.xlabel('TEMPERATURE (scaled)')
plt.ylabel('CAPACITY (scaled)')
plt.title('Lift Maintenance Clustering')
plt.colorbar(label='Cluster')
plt.show()

# Display a sample of the clustered data
print(data.head())
