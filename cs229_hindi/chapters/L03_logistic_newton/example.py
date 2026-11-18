"""Lecture 3 example: Logistic Regression (Gradient Ascent बनाम Newton) और LWR (सिर्फ़ NumPy).

भाग 1: दो exam scores से admission (0/1) का prediction। एक ही data पर Batch Gradient Ascent
और Newton's Method (Fisher Scoring / IRLS) चलाकर हर iteration पर log-likelihood देखते हैं।
भाग 2: एक मुड़े हुए 1D function पर Locally Weighted Regression (LWR) और bandwidth tau का असर।
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def log_likelihood(theta, X, y):
    """Overflow-safe Bernoulli log-likelihood: sum(y*z - log(1 + exp(z)))."""
    z = X @ theta
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


# ==================== 1. Logistic Regression: Data और Setup ====================
m = 200
s1 = rng.normal(65, 12, m)                          # exam 1 score
s2 = 0.85 * s1 + rng.normal(12, 5.5, m)             # exam 2 (मज़बूत correlation -> तंग ridge)
logits_true = -13.5 + 0.11 * s1 + 0.10 * s2
y_all = (rng.uniform(size=m) < sigmoid(logits_true)).astype(float)

idx = rng.permutation(m)
tr, te = idx[:160], idx[160:]
feats = np.c_[s1, s2]
mu, sd = feats[tr].mean(axis=0), feats[tr].std(axis=0)

X_tr = np.c_[np.ones(len(tr)), (feats[tr] - mu) / sd]
X_te = np.c_[np.ones(len(te)), (feats[te] - mu) / sd]
y_tr, y_te = y_all[tr], y_all[te]


def gradient_ascent(X, y, alpha=0.005, iters=200):
    theta = np.zeros(X.shape[1])
    hist = [log_likelihood(theta, X, y)]
    for _ in range(iters):
        theta = theta + alpha * (X.T @ (y - sigmoid(X @ theta)))
        hist.append(log_likelihood(theta, X, y))
    return theta, np.array(hist)


def newtons_method(X, y, iters=7):
    theta = np.zeros(X.shape[1])
    hist = [log_likelihood(theta, X, y)]
    for _ in range(iters):
        h = sigmoid(X @ theta)
        grad = X.T @ (y - h)
        d = h * (1.0 - h)                           # Fisher / IRLS diagonal weights
        neg_H = X.T @ (d[:, None] * X)              # -H = X^T D X (positive definite)
        theta = theta + np.linalg.solve(neg_H, grad)
        hist.append(log_likelihood(theta, X, y))
    return theta, np.array(hist)


th_ga, ll_ga = gradient_ascent(X_tr, y_tr)
th_nt, ll_nt = newtons_method(X_tr, y_tr)
ll_star = ll_nt[-1]

print("=== Logistic Regression: Newton's Method (प्रति iteration) ===")
print("iter    log-likelihood     gap (l* - l)")
for k, val in enumerate(ll_nt):
    print(f"{k:4d}    {val:14.8f}    {max(ll_star - val, 0.0):12.2e}")

print("\n=== Logistic Regression: Gradient Ascent (चुनिंदा iterations) ===")
print("iter    log-likelihood     gap (l* - l)")
for k in (0, 1, 2, 5, 15, 50, 100, 200):
    print(f"{k:4d}    {ll_ga[k]:14.8f}    {max(ll_star - ll_ga[k], 0.0):12.2e}")

acc_tr = np.mean((sigmoid(X_tr @ th_nt) >= 0.5) == y_tr) * 100
acc_te = np.mean((sigmoid(X_te @ th_nt) >= 0.5) == y_te) * 100
print(f"\nNewton theta : { np.round(th_nt, 5) }")
print(f"Ascent theta : { np.round(th_ga, 5) }")
print(f"Accuracy     : train = {acc_tr:.1f}%,  test = {acc_te:.1f}%")


# ==================== 2. Locally Weighted Regression (LWR) ====================
def f_true(x):
    return np.sin(1.4 * x) + 0.35 * np.cos(3.7 * x)


x_lwr = np.sort(rng.uniform(-3.0, 3.0, 65))
y_lwr = f_true(x_lwr) + rng.normal(0.0, 0.26, len(x_lwr))
x_grid = np.linspace(-2.85, 2.85, 160)


def lwr_predict(x_train, y_train, queries, tau):
    """हर query x के लिए (X^T W X) theta = X^T W y हल करें।"""
    X_des = np.c_[np.ones_like(x_train), x_train]
    preds = np.empty_like(queries)
    for idx_q, q in enumerate(queries):
        w = np.exp(-((x_train - q) ** 2) / (2.0 * tau ** 2))
        A = X_des.T @ (w[:, None] * X_des) + 1e-10 * np.eye(2)
        b = X_des.T @ (w * y_train)
        th_local = np.linalg.solve(A, b)
        preds[idx_q] = th_local[0] + th_local[1] * q
    return preds


print("\n=== Locally Weighted Regression: अलग-अलग bandwidth tau पर MSE ===")
fits = {}
for tau, tag in [(3.0, "underfit (लगभग सीधी रेखा)"), (0.8, "थोड़ा smooth"),
                 (0.28, "संतुलित local fit"), (0.06, "overfit (शोर के पीछे)")]:
    fits[tau] = lwr_predict(x_lwr, y_lwr, x_grid, tau)
    mse = np.mean((fits[tau] - f_true(x_grid)) ** 2)
    print(f"  tau = {tau:<5}  MSE vs true curve = {mse:.4f}   [{tag}]")

# ==================== 3. Plots ====================
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

axes[0].semilogy(np.maximum(ll_star - ll_ga, 1e-15), color="#2563eb", lw=1.8, label="Gradient Ascent")
axes[0].semilogy(np.maximum(ll_star - ll_nt, 1e-15), "o-", color="#dc2626", lw=1.8, ms=4.5, label="Newton's Method")
axes[0].set(xlabel="iteration", ylabel=r"$\ell(\theta^*) - \ell(\theta)$", title="Convergence to MLE")
axes[0].legend()

ax1 = axes[1]
ax1.scatter(X_tr[y_tr == 0, 1], X_tr[y_tr == 0, 2], s=18, c="#dc2626", alpha=0.75, label="y = 0")
ax1.scatter(X_tr[y_tr == 1, 1], X_tr[y_tr == 1, 2], s=18, c="#2563eb", alpha=0.75, label="y = 1")
xs = np.linspace(-2.4, 2.4, 100)
ax1.plot(xs, -(th_nt[0] + th_nt[1] * xs) / th_nt[2], "k-", lw=1.8, label="h = 0.5 boundary")
ax1.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel="scaled exam 1", ylabel="scaled exam 2",
        title="Logistic Decision Boundary")
ax1.legend(loc="lower left", fontsize=8.5)

ax2 = axes[2]
ax2.scatter(x_lwr, y_lwr, s=14, color="#6b7280", alpha=0.65, label="noisy data")
ax2.plot(x_grid, f_true(x_grid), "k--", lw=1.2, label="true curve")
for tau, col in [(3.0, "#2563eb"), (0.28, "#16a34a"), (0.06, "#ea580c")]:
    ax2.plot(x_grid, fits[tau], color=col, lw=1.7, label=rf"$\tau = {tau}$")
ax2.set(ylim=(-2.0, 2.0), xlabel="x", ylabel="y", title=r"LWR across bandwidth $\tau$")
ax2.legend(loc="lower left", fontsize=8.2)

fig.tight_layout()
fig.savefig("result.png", dpi=130)
