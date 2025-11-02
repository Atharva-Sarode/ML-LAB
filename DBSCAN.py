# DBSCAN Clustering Example with Vibrant Colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Set vibrant style
sns.set_theme(style="whitegrid", palette="bright")

# Generate a predefined dataset (you can change to make_circles, make_blobs, etc.)
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Plot the clusters
plt.figure(figsize=(10,6))
palette = sns.color_palette("bright", np.unique(clusters).max() + 1)
colors = [palette[x] if x != -1 else (0.5,0.5,0.5) for x in clusters]  # gray for noise

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=colors, s=50, edgecolor='k')
plt.title("DBSCAN Clustering", fontsize=16)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.grid(True)
plt.show()
