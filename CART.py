# CART: Smaller Classification and Regression Trees with Vibrant Colors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# Set vibrant style
sns.set_theme(style="whitegrid", palette="bright")

# =========================
# Classification Tree (Iris)
# =========================
print("=== Smaller Classification Tree (Iris) ===")

iris = load_iris()
X_cls = iris.data[:, :2]  # first two features for 2D plot
y_cls = iris.target

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

# Smaller CART classifier
clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=42)
clf.fit(X_train_cls, y_train_cls)

y_pred_cls = clf.predict(X_test_cls)

print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_cls))
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))

plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names[:2], class_names=iris.target_names, filled=True, rounded=True)
plt.title(" CART Classification Tree")
plt.show()



# Decision boundaries
x_min, x_max = X_cls[:, 0].min() - 1, X_cls[:, 0].max() + 1
y_min, y_max = X_cls[:, 1].min() - 1, X_cls[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap="Set2")
sns.scatterplot(x=X_cls[:, 0], y=X_cls[:, 1], hue=iris.target_names[y_cls], palette="bright", s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Smaller CART Classification Decision Boundary")
plt.show()


# =========================
# Regression Tree (Diabetes)
# =========================
print("\n=== Smaller Regression Tree (Diabetes) ===")

diabetes = load_diabetes()
X_reg = diabetes.data[:, :1]  # first feature for 2D plot
y_reg = diabetes.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Smaller CART regression
reg = DecisionTreeRegressor(max_depth=2, min_samples_leaf=10, random_state=42)
reg.fit(X_train_reg, y_train_reg)

y_pred_reg = reg.predict(X_test_reg)

print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))

plt.figure(figsize=(12,8))
plot_tree(reg, feature_names=diabetes.feature_names[:1], filled=True, rounded=True)
plt.title(" CART Regression Tree")
plt.show()

# Predicted vs actual
plt.figure(figsize=(10,6))
plt.scatter(X_test_reg, y_test_reg, color='blue', label='Actual', s=60)
plt.scatter(X_test_reg, y_pred_reg, color='red', label='Predicted', s=60)
plt.xlabel(diabetes.feature_names[0])
plt.ylabel("Target")
plt.title(" CART Regression Predictions")
plt.legend()
plt.show()
