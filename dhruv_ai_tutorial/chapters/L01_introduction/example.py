"""Lecture 1 example: Machine Learning की शब्दावली, चलते हुए code में (सिर्फ़ NumPy).

तीन छोटे प्रयोग, तीनों synthetic data पर:
  A. Supervised Learning, regression: polynomial model, underfitting और overfitting
  B. Supervised Learning, classification: k-Nearest Neighbours (k-NN) शुरू से
  C. Unsupervised Learning, clustering: k-means शुरू से
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)


def train_test_split(X, y, test_fraction=0.3):
    """Data को shuffle करके दो हिस्सों में बाँटें: training set और test set।"""
    idx = rng.permutation(len(y))
    n_test = int(round(test_fraction * len(y)))
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]


# ---------- A. Regression: y एक continuous संख्या है ----------
# सच्चा function sin(2*pi*x) है, पर model को यह नहीं पता: उसे सिर्फ़ noisy examples मिलते हैं।
x = rng.uniform(0, 1, 40)                                  # feature
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 40)        # target = pattern + noise
x_tr, y_tr, x_te, y_te = train_test_split(x, y)


def poly_features(x, degree):
    """x से features [1, x, x^2, ..., x^degree] बनाएँ।"""
    return np.vander(x, degree + 1, increasing=True)


def fit(x, y, degree):
    """Training: वह theta ढूँढें जो squared-error cost J(theta) को minimize करे।"""
    return np.linalg.lstsq(poly_features(x, degree), y, rcond=None)[0]


def mse(theta, x, y):
    """Prediction h(x) निकालें और औसत squared error नापें।"""
    return np.mean((poly_features(x, len(theta) - 1) @ theta - y) ** 2)


print("A. Regression (polynomial model): model की complexity बढ़ाने का असर")
print("   degree  parameters  train error  test error")
models = {}
for degree in (1, 3, 9, 15):
    theta = models[degree] = fit(x_tr, y_tr, degree)
    print(f"   {degree:>6}  {degree + 1:>10}  {mse(theta, x_tr, y_tr):11.4f}  {mse(theta, x_te, y_te):10.4f}")
print("   degree 1: दोनों errors बड़े (underfitting)।  degree 15: train error सबसे छोटा,")
print("   पर test error फिर बढ़ गया (overfitting)।  degree 3 सबसे अच्छा generalize करती है।")

# ---------- B. Classification: y एक category है (0 या 1) ----------
m = 200
labels = rng.integers(0, 2, m)
centers = np.array([[0.0, 0.0], [2.0, 2.0]])
points = centers[labels] + rng.normal(0, 1.1, (m, 2))     # दो features, दो आपस में मिले हुए बादल
P_tr, l_tr, P_te, l_te = train_test_split(points, labels)


def knn_predict(P_train, l_train, P_new, k):
    """हर नए point के k सबसे पास वाले training points देखें और बहुमत का label दें।"""
    dist = np.linalg.norm(P_new[:, None, :] - P_train[None, :, :], axis=2)
    nearest = np.argsort(dist, axis=1)[:, :k]
    return (l_train[nearest].mean(axis=1) > 0.5).astype(int)


print("\nB. Classification (k-NN): 140 training examples, 60 test examples")
print("   k    train accuracy  test accuracy")
for k in (1, 5, 25):
    acc_tr = np.mean(knn_predict(P_tr, l_tr, P_tr, k) == l_tr)
    acc_te = np.mean(knn_predict(P_tr, l_tr, P_te, k) == l_te)
    print(f"   {k:<4} {acc_tr:14.3f}  {acc_te:13.3f}")
print("   k = 1 training set को रट लेता है (100%), पर नए data पर सबसे कमज़ोर है।")

# ---------- C. Clustering: कोई label नहीं, सिर्फ़ x ----------
true_centers = np.array([[0.0, 0.0], [4.0, 0.5], [2.0, 3.5]])
data = np.vstack([c + rng.normal(0, 0.7, (60, 2)) for c in true_centers])   # labels फेंक दिए गए


def kmeans(X, k, iters=20):
    mu = X[rng.choice(len(X), k, replace=False)].copy()    # शुरुआत: k random points को centroid मानें
    history = []
    for _ in range(iters):
        c = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=2).argmin(axis=1)   # 1. सबसे पास का centroid
        history.append(np.sum((X - mu[c]) ** 2))           # J(c, mu), centroids हिलाने से पहले
        new_mu = np.array([X[c == j].mean(axis=0) for j in range(k)])               # 2. centroid = mean
        if np.allclose(new_mu, mu):
            break
        mu = new_mu
    return mu, c, history


mu, cluster, J_history = kmeans(data, 3)
print("\nC. Clustering (k-means, k = 3): हर iteration पर J(c, mu)")
print("   " + "  ".join(f"{J:.1f}" for J in J_history))
print("   मिले centroids   :", "  ".join(f"({a:5.2f}, {b:5.2f})" for a, b in mu[np.argsort(mu[:, 0])]))
print("   असली centers     :", "  ".join(f"({a:5.2f}, {b:5.2f})" for a, b in true_centers[np.argsort(true_centers[:, 0])]))

# ---------- D. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
grid = np.linspace(0, 1, 300)
axes[0].scatter(x_tr, y_tr, s=18, color="black", label="training set")
axes[0].scatter(x_te, y_te, s=18, color="tab:orange", label="test set")
for degree, style in ((1, ":"), (3, "-"), (15, "--")):
    axes[0].plot(grid, poly_features(grid, degree) @ models[degree], style, label=f"degree {degree}")
axes[0].set(ylim=(-2, 2), xlabel="x (feature)", ylabel="y (target)", title="A. Regression: under / good / overfit")
axes[0].legend(fontsize=8)

g1, g2 = np.meshgrid(np.linspace(-3, 5, 120), np.linspace(-3, 5, 120))
zz = knn_predict(P_tr, l_tr, np.c_[g1.ravel(), g2.ravel()], 25).reshape(g1.shape)
axes[1].contourf(g1, g2, zz, levels=[-0.5, 0.5, 1.5], colors=["#fecaca", "#bfdbfe"], alpha=0.6)
axes[1].scatter(*P_te.T, c=np.where(l_te == 1, "tab:blue", "tab:red"), s=18, edgecolor="white", linewidth=0.4)
axes[1].set(xlabel="x1", ylabel="x2", title="B. Classification: k-NN (k=25), test points")

axes[2].scatter(*data.T, c=cluster, cmap="viridis", s=14)
axes[2].scatter(*mu.T, marker="*", s=260, color="red", edgecolor="black", label="centroids")
axes[2].set(xlabel="x1", ylabel="x2", title="C. Clustering: k-means")
axes[2].legend()
fig.tight_layout()
fig.savefig("result.png", dpi=130)
