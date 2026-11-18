"""Lecture 2 example: Linear Regression शुरू से (सिर्फ़ NumPy).

घर के size और bedrooms से कीमत का अंदाज़ा। तीन तरीकों से theta निकालते हैं:
Batch Gradient Descent, Stochastic Gradient Descent, और Normal Equations।
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- 1. Data बनाएँ: price = 50 + 0.12*size + 15*bedrooms + noise ----------
m = 200
size = rng.uniform(500, 3500, m)                 # square feet
bedrooms = rng.integers(1, 6, m).astype(float)
price = 50 + 0.12 * size + 15 * bedrooms + rng.normal(0, 25, m)   # हज़ार dollars में

# ---------- 2. Train / test split ----------
idx = rng.permutation(m)
train, test = idx[:160], idx[160:]

# ---------- 3. Feature scaling (mean 0, std 1), सिर्फ़ training data के statistics से ----------
features = np.c_[size, bedrooms]
mu, sigma = features[train].mean(axis=0), features[train].std(axis=0)


def design_matrix(f):
    """Scale करें और x0 = 1 वाला column जोड़ें।"""
    return np.c_[np.ones(len(f)), (f - mu) / sigma]


X_train, y_train = design_matrix(features[train]), price[train]
X_test, y_test = design_matrix(features[test]), price[test]


def cost(theta, X, y):
    """J(theta) = (1/2m) * sum (h(x) - y)^2"""
    return 0.5 * np.mean((X @ theta - y) ** 2)


# ---------- 4. Batch Gradient Descent ----------
def batch_gd(X, y, alpha=0.1, iters=300):
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(iters):
        gradient = X.T @ (X @ theta - y) / len(y)    # पूरे data का gradient
        theta -= alpha * gradient
        history.append(cost(theta, X, y))
    return theta, history


# ---------- 5. Stochastic Gradient Descent (LMS rule, एक बार में एक example) ----------
def sgd(X, y, alpha=0.01, epochs=20):
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            error = y[i] - X[i] @ theta
            theta += alpha * error * X[i]            # theta_j := theta_j + alpha * error * x_j
        history.append(cost(theta, X, y))
    return theta, history


# ---------- 6. Normal Equations: X^T X theta = X^T y ----------
def normal_equations(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


theta_gd, hist_gd = batch_gd(X_train, y_train)
theta_sgd, hist_sgd = sgd(X_train, y_train)
theta_ne = normal_equations(X_train, y_train)


def r2(theta, X, y):
    return 1 - np.sum((y - X @ theta) ** 2) / np.sum((y - y.mean()) ** 2)


print("Method             theta0   theta1   theta2   train J    test R^2")
for name, th in [("Batch GD", theta_gd), ("SGD", theta_sgd), ("Normal Equations", theta_ne)]:
    print(f"{name:<17} {th[0]:8.2f} {th[1]:8.2f} {th[2]:8.2f} {cost(th, X_train, y_train):9.2f} "
          f"{r2(th, X_test, y_test):10.4f}")

# Scaled theta को वापस असली units में बदलें ताकि मतलब समझ आए
slope = theta_ne[1:] / sigma
intercept = theta_ne[0] - np.sum(theta_ne[1:] * mu / sigma)
print(f"\nअसली units में: price = {intercept:.1f} + {slope[0]:.3f}*size + {slope[1]:.1f}*bedrooms")
print("(data बनाते समय सच यह था: price = 50 + 0.120*size + 15*bedrooms)")

new_house = np.array([[2000.0, 3.0]])
print(f"\n2000 sq ft, 3 bedrooms वाले घर का prediction: {(design_matrix(new_house) @ theta_ne)[0]:.1f} हज़ार")

# ---------- 7. Learning rate का असर ----------
print("\nअलग-अलग alpha पर 50 iterations के बाद J:")
for alpha in (0.001, 0.01, 0.1, 1.0, 2.1):
    _, h = batch_gd(X_train, y_train, alpha=alpha, iters=50)
    note = "  <- diverge!" if h[-1] > h[0] else ""
    print(f"  alpha = {alpha:<6} J = {h[-1]:.4g}{note}")

# ---------- 8. Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(hist_gd, label="Batch GD (per iteration)")
axes[0].plot(np.linspace(0, len(hist_gd), len(hist_sgd)), hist_sgd, "o-", ms=3, label="SGD (per epoch, rescaled)")
axes[0].axhline(cost(theta_ne, X_train, y_train), color="gray", ls="--", label="Normal Equations")
axes[0].set(yscale="log", xlabel="iteration", ylabel="J(theta)", title="Cost during training")
axes[0].legend()
pred = X_test @ theta_ne
axes[1].scatter(y_test, pred, s=18)
lims = [y_test.min(), y_test.max()]
axes[1].plot(lims, lims, "r--", label="perfect prediction")
axes[1].set(xlabel="actual price", ylabel="predicted price", title="Test set: predicted vs actual")
axes[1].legend()
fig.tight_layout()
fig.savefig("result.png", dpi=130)
