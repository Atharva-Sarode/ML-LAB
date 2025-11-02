# Graph-Based Clustering (Spectral Clustering) with Vibrant Colors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import SpectralClustering

# Set vibrant style
sns.set_theme(style="whitegrid", palette="bright")

# Generate a predefined dataset (non-linear clusters)
X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)
# You can also try: X, y_true = make_circles(n_samples=300, noise=0.05, factor=0.5)

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
y_pred = spectral.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_pred, s=80, cmap='Set2', edgecolor='k')
plt.title("Spectral Clustering", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
