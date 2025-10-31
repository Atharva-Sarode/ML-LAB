# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Set vibrant theme
sns.set_theme(style="whitegrid", palette="bright")

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------- PCA -----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
for target, color in zip([0,1,2], sns.color_palette("bright", 3)):
    plt.scatter(X_pca[y==target,0], X_pca[y==target,1], label=target_names[target], s=100)
plt.title("PCA of Iris Dataset", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# ----------------- SVD -----------------
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
for target, color in zip([0,1,2], sns.color_palette("bright", 3)):
    plt.scatter(X_svd[y==target,0], X_svd[y==target,1], label=target_names[target], s=100)
plt.title("SVD of Iris Dataset", fontsize=14)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.legend()
plt.show()

# ----------------- LDA -----------------
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

plt.figure(figsize=(6,5))
for target, color in zip([0,1,2], sns.color_palette("bright", 3)):
    plt.scatter(X_lda[y==target,0], X_lda[y==target,1], label=target_names[target], s=100)
plt.title("LDA of Iris Dataset", fontsize=14)
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.legend()
plt.show()
