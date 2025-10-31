# Multivariate Linear Regression with Visualization (Diabetes dataset, different features)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Set vibrant theme
sns.set_theme(style="whitegrid", palette="bright")

# Load Diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Select two different features for visualization
X_sel = X[["bp", "s1"]]   # bp = average blood pressure, s1 = blood serum measurement

# Fit Linear Regression Model
model = LinearRegression()
model.fit(X_sel, y)

# Predictions
y_pred = model.predict(X_sel)

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot of actual data
ax.scatter(X_sel["bp"], X_sel["s1"], y,
           color="orange", label="Actual Data", alpha=0.6)

# Create grid for regression plane
x_surf, y_surf = np.meshgrid(
    np.linspace(X_sel["bp"].min(), X_sel["bp"].max(), 30),
    np.linspace(X_sel["s1"].min(), X_sel["s1"].max(), 30)
)
z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf

# Plot regression plane
ax.plot_surface(x_surf, y_surf, z_surf, color="cyan", alpha=0.4, edgecolor="none")

# Labels
ax.set_xlabel("Blood Pressure (bp)", fontsize=12, labelpad=10)
ax.set_ylabel("Serum S1", fontsize=12, labelpad=10)
ax.set_zlabel("Disease Progression", fontsize=12, labelpad=10)
ax.set_title("Multivariate Linear Regression (3D)", fontsize=15, pad=20)

# Legend
ax.legend()
plt.show()

