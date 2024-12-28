import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
file_path = 'Area Safety Prediction.csv'
data = pd.read_csv(file_path)

# Preprocess the data - drop non-essential columns
X = data.drop(columns=['area', 'outcome', 'class'])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.xlabel('Sex Ratio (scaled)')
plt.ylabel('R Cases (scaled)')
plt.title('Area Safety Clustering')
plt.colorbar(label='Cluster')
plt.show()

# Save clustered data to CSV
data.to_csv('Clustered_Area_Safety.csv', index=False)

print("Clustering complete. Results saved to 'Clustered_Area_Safety.csv'")
