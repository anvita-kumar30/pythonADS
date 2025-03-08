import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Load dataset from the uploaded CSV file
df = pd.read_csv("Iris.csv")

# Drop the 'Id' column if it exists (since it's not a feature)
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Extract features (assuming the last column is the target/class label)
X = df.iloc[:, :-1].values  # All columns except the last one

### Distance-Based Outlier Detection (K-NN) ###
k = 5  # Number of neighbors
nbrs = NearestNeighbors(n_neighbors=k)
nbrs.fit(X)

# Compute distances to k-th nearest neighbor
distances, _ = nbrs.kneighbors(X)
k_distances = distances[:, k-1]  # Take the k-th nearest neighbor distance

# Plot k-distance graph
plt.figure(figsize=(8, 5))
plt.plot(np.sort(k_distances))
plt.xlabel("Data Points sorted by distance")
plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
plt.title("K-Distance Graph for Outlier Detection")
plt.show()

# Identify outliers based on threshold (e.g., top 5% highest distances)
threshold = np.percentile(k_distances, 95)  # Top 5% considered as outliers
outlier_indices_knn = np.where(k_distances > threshold)[0]
print("Outliers detected using K-NN:", outlier_indices_knn)

### Density-Based Outlier Detection (DBSCAN) ###
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust 'eps' as needed
labels = dbscan.fit_predict(X)

# Outliers are labeled as -1 in DBSCAN
outlier_indices_dbscan = np.where(labels == -1)[0]
print("Outliers detected using DBSCAN:", outlier_indices_dbscan)

# Visualize DBSCAN clustering
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker="o", edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Outlier Detection")
plt.colorbar(label="Cluster Label")
plt.show()
