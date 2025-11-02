import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load diabetes dataset (predefined & complex)
diabetes = datasets.load_diabetes()

# Use one feature (e.g., BMI - feature index 2)
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Performance metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot with vibrant colours
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color="#FF1493", alpha=0.6, edgecolor="black", label="Actual Data")
plt.plot(X_test, y_pred, color="#00FA9A", linewidth=3, label="Regression Line")
plt.title("Simple Linear Regression on Diabetes Dataset", fontsize=15, color="#8A2BE2")
plt.xlabel("BMI Feature", fontsize=12, color="#1E90FF")
plt.ylabel("Disease Progression", fontsize=12, color="#1E90FF")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
