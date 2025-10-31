# Decision Boundary Visualization for Bagging vs Boosting on make_moons dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# =========================
# 1. Load Dataset (make_moons)
# =========================
X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 2. Define Models
# =========================
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=30,
    random_state=42
)

boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

bagging.fit(X_train, y_train)
boosting.fit(X_train, y_train)

# =========================
# 3. Plot Decision Boundaries
# =========================
def plot_decision_boundary(model, X, y, title, ax):
    # mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # vibrant background
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm",
                    edgecolor="k", s=40, ax=ax, legend=False)
    ax.set_title(title, fontsize=12, weight="bold")

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_decision_boundary(bagging, X_test, y_test, "Decision Boundary - Bagging", axes[0])
plot_decision_boundary(boosting, X_test, y_test, "Decision Boundary - Boosting", axes[1])

plt.tight_layout()
plt.show()
