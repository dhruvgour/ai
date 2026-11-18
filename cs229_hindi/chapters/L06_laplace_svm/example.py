"""Lecture 6 example: Multinomial Naive Bayes (Laplace smoothing) और Hard-Margin Linear SVM.

भाग A: हाथ से लिखे छोटे corpus पर Multinomial Event Model, बिना smoothing और Laplace smoothing के साथ।
भाग B: Linearly separable 2D data पर scipy.optimize SLSQP से Hard-Margin SVM (w, b, margin, support vectors)।
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------- 1. भाग A: Multinomial Naive Bayes (बिना smoothing बनाम Laplace smoothing) ----------
spam_docs = [
    "win free cash win cash now",
    "free offer win prize now",
    "cash prize free free offer",
    "win cheap cash offer now",
    "free prize win money",
]
ham_docs = [
    "team meeting at noon",
    "project report before meeting",
    "lunch at noon with team",
    "review project report today",
    "free for team lunch today",
]
vocab = sorted({w for doc in spam_docs + ham_docs for w in doc.split()})
word_to_id = {w: k for k, w in enumerate(vocab)}
V = len(vocab)


def word_counts(docs):
    """हर शब्द सारे documents में कुल कितनी बार आया (multinomial event model)।"""
    counts = np.zeros(V)
    for doc in docs:
        for w in doc.split():
            if w in word_to_id:                  # vocabulary से बाहर के शब्द छोड़ दें
                counts[word_to_id[w]] += 1
    return counts


c_spam, c_ham = word_counts(spam_docs), word_counts(ham_docs)
n_spam, n_ham = c_spam.sum(), c_ham.sum()
phi_y = len(spam_docs) / (len(spam_docs) + len(ham_docs))

# MLE (smoothing = 0) और Laplace smoothing (numerator में +1, denominator में +|V|)
phi_spam_mle, phi_ham_mle = c_spam / n_spam, c_ham / n_ham
phi_spam_lap = (c_spam + 1.0) / (n_spam + V)
phi_ham_lap = (c_ham + 1.0) / (n_ham + V)


def predict_mle(doc):
    ids = [word_to_id[w] for w in doc.split() if w in word_to_id]
    p1 = phi_y * np.prod(phi_spam_mle[ids])
    p0 = (1.0 - phi_y) * np.prod(phi_ham_mle[ids])
    with np.errstate(invalid="ignore"):
        return p1 / (p1 + p0)


def predict_laplace(doc):
    """Underflow से बचने के लिए log-probabilities जोड़ें।"""
    ids = [word_to_id[w] for w in doc.split() if w in word_to_id]
    log1 = np.log(phi_y) + np.sum(np.log(phi_spam_lap[ids]))
    log0 = np.log(1.0 - phi_y) + np.sum(np.log(phi_ham_lap[ids]))
    m = max(log1, log0)
    return float(np.exp(log1 - m) / (np.exp(log1 - m) + np.exp(log0 - m)))


print(f"भाग A: Multinomial Naive Bayes (|V| = {V}, spam words = {int(n_spam)}, ham words = {int(n_ham)})")
print(f"  Laplace probabilities का जोड़: spam = {phi_spam_lap.sum():.4f}, ham = {phi_ham_lap.sum():.4f}")
print("  word       c(spam) c(ham)   MLE(y=1) MLE(y=0)   Lap(y=1) Lap(y=0)")
for w in ("free", "win", "cash", "team", "meeting"):
    k = word_to_id[w]
    print(f"  {w:<10} {int(c_spam[k]):6d} {int(c_ham[k]):6d}   {phi_spam_mle[k]:8.3f} {phi_ham_mle[k]:8.3f}"
          f"   {phi_spam_lap[k]:8.3f} {phi_ham_lap[k]:8.3f}")

print("\n  Test संदेश                           MLE P(spam|x)   Laplace P(spam|x)")
for msg in ["win free cash cash", "team project report", "free team lunch", "win prize at meeting"]:
    print(f"  {msg:<35} {predict_mle(msg):13.4f}   {predict_laplace(msg):17.4f}")

# ---------- 2. भाग B: Hard-Margin Linear SVM (scipy.optimize SLSQP) ----------
rng = np.random.default_rng(11)
X_neg = rng.normal(loc=[-1.2, -1.2], scale=0.52, size=(12, 2))
X_pos = rng.normal(loc=[1.2, 1.2], scale=0.52, size=(12, 2))
X = np.vstack([X_neg, X_pos])
y = np.r_[-np.ones(len(X_neg)), np.ones(len(X_pos))]


def solve_hard_margin_svm(X_data, y_data):
    """Primal QP: min_{w,b} 0.5*||w||^2  s.t.  y^(i)*(w^T x^(i) + b) - 1 >= 0."""
    res = minimize(
        lambda u: 0.5 * np.dot(u[:2], u[:2]),
        np.zeros(3),
        jac=lambda u: np.array([u[0], u[1], 0.0]),
        constraints={"type": "ineq", "fun": lambda u: y_data * (X_data @ u[:2] + u[2]) - 1.0,
                     "jac": lambda u: np.c_[y_data[:, None] * X_data, y_data]},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    return res.x[:2], float(res.x[2])


w, b = solve_hard_margin_svm(X, y)
Q = (y[:, None] * X) @ (y[:, None] * X).T
dual = minimize(
    lambda a: 0.5 * a @ Q @ a - np.sum(a), np.zeros(len(y)), jac=lambda a: Q @ a - 1.0,
    bounds=[(0.0, None)] * len(y), constraints={"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y},
    method="SLSQP", options={"ftol": 1e-12, "maxiter": 300},
)
alpha = np.maximum(0.0, dual.x)
func_margins = y * (X @ w + b)
gamma = 1.0 / np.linalg.norm(w)
sv_idx = np.where(func_margins <= 1.0 + 1e-5)[0]

# एक non-support point को हिलाकर जाँचें कि hyperplane बदलता है या नहीं
interior_id = int(np.argmax(func_margins))
X_moved = X.copy()
X_moved[interior_id] += np.array([0.8, 0.7])
w_moved, b_moved = solve_hard_margin_svm(X_moved, y)

print("\nभाग B: Hard-Margin Linear SVM (SLSQP)")
print(f"  w = [{w[0]:.4f}, {w[1]:.4f}],  b = {b:.4f},  ||w|| = {np.linalg.norm(w):.4f}")
print(f"  Geometric margin gamma = 1/||w|| = {gamma:.4f}  (slab width 2*gamma = {2 * gamma:.4f})")
print(f"  Strong duality check: primal p* = {0.5 * (w @ w):.6f},  dual d* = {-dual.fun:.6f}")
print(f"  Support vectors ({len(sv_idx)} of {len(y)} points, जहाँ y*(w^T x + b) = 1 और alpha_i > 0):")
for idx in sv_idx:
    print(f"    idx={idx:2d}  x=({X[idx, 0]:6.3f}, {X[idx, 1]:6.3f})  y={int(y[idx]):+d}"
          f"  func_margin={func_margins[idx]:.4f}  alpha={alpha[idx]:.4f}")
print(f"  Non-SV point (idx={interior_id}) को हिलाने पर ||w_new - w|| = {np.linalg.norm(w_moved - w):.2e}")

# ---------- 3. Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
words_show = ["free", "win", "cash", "offer", "team", "meeting", "report", "lunch"]
ids_show = [word_to_id[wd] for wd in words_show]
xpos = np.arange(len(words_show))
axes[0].bar(xpos - 0.18, phi_spam_mle[ids_show], width=0.34, color="#fca5a5", label="Spam MLE (zero possible)")
axes[0].bar(xpos + 0.18, phi_spam_lap[ids_show], width=0.34, color="#dc2626", label="Spam Laplace (+1 / +|V|)")
axes[0].set_xticks(xpos, words_show, rotation=28)
axes[0].set(ylabel=r"$\phi_{k \mid y=1}$", title="Multinomial NB: Laplace lifts zeros")
axes[0].legend(fontsize=8)

ax = axes[1]
ax.scatter(*X[y == -1].T, color="tab:blue", s=34, label="y = -1")
ax.scatter(*X[y == 1].T, color="tab:red", s=34, label="y = +1")
ax.scatter(*X[sv_idx].T, s=135, facecolors="none", edgecolors="goldenrod", linewidths=2.0, label="Support vectors")
xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
y_mid = -(w[0] * xs + b) / w[1]
y_up, y_dn = -(w[0] * xs + b - 1.0) / w[1], -(w[0] * xs + b + 1.0) / w[1]
ax.fill_between(xs, y_dn, y_up, color="teal", alpha=0.12, label=rf"Margin slab ($2\gamma$ = {2 * gamma:.2f})")
ax.plot(xs, y_mid, "k-", lw=2.0, label=r"$w^T x + b = 0$")
ax.plot(xs, y_up, "r--", lw=1.1)
ax.plot(xs, y_dn, "b--", lw=1.1)
ax.set(xlim=(xs[0], xs[-1]), ylim=(X[:, 1].min() - 0.6, X[:, 1].max() + 0.6),
       xlabel="x1", ylabel="x2", title="Hard-Margin SVM via SLSQP", aspect="equal")
ax.legend(fontsize=7.8, loc="lower right")
fig.tight_layout()
fig.savefig("result.png", dpi=130)
