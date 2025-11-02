# Logistic Regression with Sigmoid Curve + Data Points (Digits Dataset)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load Digits dataset (1797 samples, 64 features)
digits = load_digits()
X = pd.DataFrame(digits.data)
y = (digits.target == 0).astype(int)   # Binary classification: "Digit 0" vs "Not 0"

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=5000, solver='lbfgs')
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

# Performance
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---- Logistic (Sigmoid) Curve with Data Points ----
def sigmoid(z):
    """The Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

# Generate smooth sigmoid curve
z = np.linspace(-10, 10, 200)
sig = sigmoid(z)

plt.figure(figsize=(9,6))
plt.plot(z, sig, color="crimson", linewidth=2, label="Sigmoid Curve")

# Overlay test data points
# Get the linear scores (z = w·x + b) for the test set
linear_scores = model.decision_function(X_test_scaled) 
plt.scatter(
    linear_scores, y_prob, 
    c=y_test,                     # Color by the true label (Actual: Not 0 or 0)
    cmap="coolwarm",              # Color map for the two classes
    edgecolor="k", s=70, alpha=0.8, 
    label="Test Data Points"
)

# Threshold reference lines
plt.axhline(y=0.5, color="gold", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")
plt.axvline(x=0, color="royalblue", linestyle="--", linewidth=1.5, label="z = 0 (Boundary)")

plt.title("Logistic Regression: Sigmoid Curve with Digits Dataset", fontsize=15, color="darkred")
plt.xlabel("Linear Function (z = w·x + b)", fontsize=12, color="blue")
plt.ylabel("Predicted Probability of Class = 1 (Digit 0)", fontsize=12, color="green")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()