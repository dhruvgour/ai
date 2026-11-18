"""Lecture 4 example: Softmax Regression और Perceptron शुरू से (सिर्फ़ NumPy).

भाग A: sklearn के 3-class Iris dataset पर Cross-Entropy loss के साथ Softmax Regression।
भाग B: Iris के linearly separable 2-class subset (setosa बनाम versicolor) पर Perceptron,
       और overlapping subset (versicolor बनाम virginica) के साथ उसकी तुलना।
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

rng = np.random.default_rng(42)
iris = load_iris()
names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# ---------- 1. Data तैयार करें: train (120) और test (30) split ----------
idx = rng.permutation(len(iris.target))
tr, te = idx[:120], idx[120:]


def prep_design(raw_X, train_idx, test_idx):
    """Training statistics से scale करें और x0 = 1 जोड़ें।"""
    mu, sd = raw_X[train_idx].mean(axis=0), raw_X[train_idx].std(axis=0)
    X_tr = np.c_[np.ones(len(train_idx)), (raw_X[train_idx] - mu) / sd]
    X_te = np.c_[np.ones(len(test_idx)), (raw_X[test_idx] - mu) / sd]
    return X_tr, X_te, mu, sd


# 2 petal features (visualisation के लिए) और सभी 4 features दोनों तैयार रखें
X_tr2, X_te2, mu2, sd2 = prep_design(iris.data[:, 2:4], tr, te)
X_tr4, X_te4, _, _ = prep_design(iris.data, tr, te)
y_tr, y_te = iris.target[tr], iris.target[te]


# ---------- 2. भाग A: Softmax Regression और Cross-Entropy ----------
def softmax(Z):
    """Overflow से बचने के लिए पहले हर row का max घटाएँ।"""
    Zs = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Zs)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def cross_entropy(P, y):
    """J(Theta) = -(1/m) * sum log P(y = y^(i) | x^(i))"""
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def fit_softmax(X, y, k=3, alpha=0.5, iters=350):
    Theta = np.zeros((X.shape[1], k))
    Y_oh = np.eye(k)[y]
    ce_hist, acc_hist = [], []
    for _ in range(iters):
        P = softmax(X @ Theta)
        ce_hist.append(cross_entropy(P, y))
        acc_hist.append(float(np.mean(P.argmax(axis=1) == y)))
        grad = (X.T @ (P - Y_oh)) / len(y)  # (1/m) sum (p_c - 1{y=c}) x
        Theta -= alpha * grad
    return Theta, ce_hist, acc_hist


Theta2, ce_hist, acc_hist = fit_softmax(X_tr2, y_tr)
Theta4, ce_hist4, _ = fit_softmax(X_tr4, y_tr)

print("=== भाग A: 3-Class Iris पर Softmax Regression ===")
print("iter    train CE loss    train accuracy")
for step in (0, 10, 50, 150, 349):
    print(f"{step:4d}    {ce_hist[step]:13.4f}    {acc_hist[step] * 100:13.1f}%")

P_te2 = softmax(X_te2 @ Theta2)
P_te4 = softmax(X_te4 @ Theta4)
print(f"\n2 petal features -> train acc: {acc_hist[-1]*100:.1f}%,  test acc: {np.mean(P_te2.argmax(1)==y_te)*100:.1f}%,  test CE: {cross_entropy(P_te2, y_te):.4f}")
print(f"सभी 4 features   -> train acc: {np.mean(softmax(X_tr4@Theta4).argmax(1)==y_tr)*100:.1f}%,  test acc: {np.mean(P_te4.argmax(1)==y_te)*100:.1f}%,  test CE: {cross_entropy(P_te4, y_te):.4f}")

print("\nतीनों classes के एक-एक test फूल पर Softmax probabilities [setosa, versicolor, virginica]:")
sample_ids = [int(np.where(y_te == c)[0][0]) for c in range(3)]
for i in sample_ids:
    probs = P_te4[i]
    pred_name, true_name = names[probs.argmax()], names[y_te[i]]
    print(f"  test #{i:2d}: p = [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}] (sum={probs.sum():.2f}) -> pred: {pred_name:<10} (सच: {true_name})")


# ---------- 3. भाग B: Perceptron (threshold नियम) ----------
def fit_perceptron(X, y, alpha=1.0, max_epochs=25):
    theta = np.zeros(X.shape[1])
    mistakes_per_epoch = []
    for _ in range(max_epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            pred = int(xi @ theta >= 0.0)  # hard threshold g(z) = 1{z >= 0}
            if pred != yi:
                theta += alpha * (yi - pred) * xi
                mistakes += 1
        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:
            break
    return theta, mistakes_per_epoch


# Separable subset: setosa (0) बनाम versicolor (1), petal length व petal width
sep_mask = iris.target < 2
X_sep = np.c_[np.ones(sep_mask.sum()), iris.data[sep_mask, 2:4]]
y_sep = iris.target[sep_mask]
perm_s = rng.permutation(len(y_sep))
th_perc, err_sep = fit_perceptron(X_sep[perm_s], y_sep[perm_s])

# Overlapping subset: versicolor (0) बनाम virginica (1)
ov_mask = iris.target >= 1
X_ov = np.c_[np.ones(ov_mask.sum()), iris.data[ov_mask, 2:4]]
y_ov = (iris.target[ov_mask] == 2).astype(int)
perm_o = rng.permutation(len(y_ov))
_, err_ov = fit_perceptron(X_ov[perm_o], y_ov[perm_o], max_epochs=15)

print("\n=== भाग B: 2-Class Subsets पर Perceptron ===")
print(f"Setosa vs Versicolor (separable): प्रति epoch गलतियाँ = {err_sep}")
print(f"  -> {len(err_sep)} epochs में शून्य गलतियों पर रुक गया, accuracy = {np.mean((X_sep @ th_perc >= 0) == y_sep)*100:.1f}%")
print(f"Versicolor vs Virginica (overlap): 15 epochs में गलतियाँ = {err_ov}")
print("  -> Overlapping data पर Perceptron कभी 0 गलतियों पर शांत नहीं होता।")

# ---------- 4. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

axes[0].plot(ce_hist, lw=2, color="#2563eb", label="2 petal features")
axes[0].plot(ce_hist4, lw=2, ls="--", color="#dc2626", label="all 4 features")
axes[0].set(xlabel="GD iteration", ylabel="Cross-Entropy J(Theta)", title="Softmax Cross-Entropy Curve")
axes[0].legend(fontsize=8.5)

gx, gy = np.meshgrid(np.linspace(0.8, 7.2, 180), np.linspace(0.0, 2.7, 180))
G_std = np.c_[np.ones(gx.size), (np.c_[gx.ravel(), gy.ravel()] - mu2) / sd2]
Z_cls = softmax(G_std @ Theta2).argmax(axis=1).reshape(gx.shape)
axes[1].contourf(gx, gy, Z_cls, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#dbeafe", "#fee2e2", "#dcfce7"], alpha=0.85)
for c, col in enumerate(["#2563eb", "#dc2626", "#16a34a"]):
    m_c = iris.target == c
    axes[1].scatter(iris.data[m_c, 2], iris.data[m_c, 3], s=18, color=col, label=names[c], edgecolor="white", lw=0.4)
axes[1].set(xlabel="petal length (cm)", ylabel="petal width (cm)", title="Softmax 3-Class Regions")
axes[1].legend(fontsize=8, loc="upper left")

for c, col, lab in [(0, "#2563eb", "setosa (y=0)"), (1, "#dc2626", "versicolor (y=1)")]:
    m_c = y_sep == c
    axes[2].scatter(X_sep[m_c, 1], X_sep[m_c, 2], s=22, color=col, label=lab)
xs = np.linspace(1.0, 5.2, 100)
axes[2].plot(xs, -(th_perc[0] + th_perc[1] * xs) / th_perc[2], "k-", lw=2, label="Perceptron boundary")
axes[2].set(xlim=(0.8, 5.4), ylim=(-0.1, 1.9), xlabel="petal length (cm)", ylabel="petal width (cm)",
            title="Perceptron (Separable Subset)")
axes[2].legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig("result.png", dpi=130)
