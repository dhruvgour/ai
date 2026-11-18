"""Lecture 7 example: Simplified SMO और RBF बनाम Linear Kernel शुरू से (NumPy).

Noisy two-moons dataset पर soft-margin SVM dual को simplified SMO से हल करते हैं,
Linear और RBF kernel की तुलना करते हैं, और sklearn SVC से मिलान जाँचते हैं।
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

rng = np.random.default_rng(42)

# ---------- 1. Two-moons data और standardization (सिर्फ़ train statistics से) ----------
X_raw, y01 = make_moons(n_samples=220, noise=0.20, random_state=42)
y_all = np.where(y01 == 1, 1.0, -1.0)             # SVM के लिए labels {-1, +1}
perm = rng.permutation(len(y_all))
tr, te = perm[:160], perm[160:]
mu, std = X_raw[tr].mean(axis=0), X_raw[tr].std(axis=0)
X_train, y_train = (X_raw[tr] - mu) / std, y_all[tr]
X_test, y_test = (X_raw[te] - mu) / std, y_all[te]


# ---------- 2. Kernels और Dual Objective ----------
def linear_kernel(X, Z):
    return X @ Z.T


def rbf_kernel(X, Z, sigma=0.7):
    sq_dist = np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    return np.exp(-sq_dist / (2.0 * sigma**2))


def dual_objective(alpha, y, K):
    ay = alpha * y
    return float(np.sum(alpha) - 0.5 * (ay @ K @ ay))


# ---------- 3. Simplified Platt SMO शुरू से ----------
def fit_smo(X, y, kernel_fn, C=2.0, tol=1e-3, max_passes=20, max_steps=180, seed=0):
    """Box constraint 0 <= alpha_i <= C और sum(alpha_i y_i) = 0 के साथ dual maximize करें।"""
    local_rng = np.random.default_rng(seed)
    K = kernel_fn(X, X)
    m = len(y)
    alpha, b, history = np.zeros(m), 0.0, []
    passes, steps = 0, 0
    while passes < max_passes and steps < max_steps:
        changed = 0
        for i in range(m):
            Ei = (alpha * y) @ K[:, i] + b - y[i]
            if not ((y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0.0)):
                continue
            if local_rng.random() < 0.8:
                diffs = np.abs(Ei - ((alpha * y) @ K + b - y))
                diffs[i] = -1.0
                j = int(np.argmax(diffs))
            else:
                j = int(local_rng.integers(m - 1))
                j = j + 1 if j >= i else j
            Ej = (alpha * y) @ K[:, j] + b - y[j]
            ai_old, aj_old = alpha[i], alpha[j]
            L = max(0.0, aj_old - ai_old) if y[i] != y[j] else max(0.0, ai_old + aj_old - C)
            H = min(C, C + aj_old - ai_old) if y[i] != y[j] else min(C, ai_old + aj_old)
            eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
            if H - L < 1e-10 or eta <= 1e-12:
                continue
            aj_new = np.clip(aj_old + y[j] * (Ei - Ej) / eta, L, H)
            if abs(aj_new - aj_old) < 1e-6:
                continue
            ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
            b1 = b - Ei - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
            b2 = b - Ej - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]
            b = b1 if 0.0 < ai_new < C else (b2 if 0.0 < aj_new < C else 0.5 * (b1 + b2))
            alpha[i], alpha[j] = ai_new, aj_new
            changed += 1
        steps += 1
        history.append(dual_objective(alpha, y, K))
        passes = passes + 1 if changed == 0 else 0
    return {"X": X, "y": y, "alpha": alpha, "b": b, "kernel": kernel_fn, "history": history}


def predict_score(mod, Z):
    return (mod["alpha"] * mod["y"]) @ mod["kernel"](mod["X"], Z) + mod["b"]


def acc_pct(y_true, scores):
    return 100.0 * np.mean(np.where(scores >= 0.0, 1.0, -1.0) == y_true)


# ---------- 4. Linear बनाम RBF SMO, और sklearn SVC से मिलान ----------
C_val, sigma_val = 2.0, 0.70
rbf_fn = lambda A, B: rbf_kernel(A, B, sigma=sigma_val)
lin_mod = fit_smo(X_train, y_train, linear_kernel, C=C_val, seed=1)
rbf_mod = fit_smo(X_train, y_train, rbf_fn, C=C_val, seed=2)
svc = SVC(C=C_val, kernel="rbf", gamma=1.0 / (2.0 * sigma_val**2), tol=1e-4).fit(X_train, y_train)

print(f"Dataset: two-moons (train={len(y_train)}, test={len(y_test)}), C={C_val}, RBF sigma={sigma_val}\n")
print("Model                 Dual W(a)   Train Acc   Test Acc   Total SVs   Free / Bounded")
for name, mod in [("Linear SMO (scratch)", lin_mod), ("RBF SMO (scratch)", rbf_mod)]:
    al = mod["alpha"]
    sv, free, bnd = al > 1e-4, (al > 1e-4) & (al < C_val - 1e-4), al >= C_val - 1e-4
    print(f"{name:<21} {mod['history'][-1]:9.3f}     {acc_pct(y_train, predict_score(mod, X_train)):6.1f}%"
          f"    {acc_pct(y_test, predict_score(mod, X_test)):6.1f}%      {sv.sum():4d}       {free.sum():3d} / {bnd.sum():<3d}")
print(f"{'sklearn SVC (RBF)':<21} {'(ref)':>9}     {100*svc.score(X_train, y_train):6.1f}%"
      f"    {100*svc.score(X_test, y_test):6.1f}%      {len(svc.support_):4d}        -")
agree = 100.0 * np.mean(np.sign(predict_score(rbf_mod, X_test)) == svc.predict(X_test))
print(f"\nRBF SMO और sklearn test predictions में सहमति: {agree:.1f}%,  sum(alpha_i y_i) = {rbf_mod['alpha'] @ y_train:+.2e}")

print("\nRBF bandwidth sigma बदलने पर प्रभाव (C = 2.0):")
for sig in (2.80, 0.70, 0.08):
    m_s = fit_smo(X_train, y_train, lambda A, B, s=sig: rbf_kernel(A, B, s), C=C_val, seed=3)
    print(f"  sigma = {sig:<4.2f} -> train = {acc_pct(y_train, predict_score(m_s, X_train)):5.1f}%, "
          f"test = {acc_pct(y_test, predict_score(m_s, X_test)):5.1f}%, SVs = {int(np.sum(m_s['alpha'] > 1e-4))}/{len(y_train)}")

# ---------- 5. Decision boundaries और SMO convergence का चित्र ----------
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
gx, gy = np.meshgrid(np.linspace(-2.4, 2.4, 140), np.linspace(-2.3, 2.3, 140))
grid_pts = np.c_[gx.ravel(), gy.ravel()]
for ax, mod, ttl in [(axes[0], lin_mod, "Linear Kernel SMO"), (axes[1], rbf_mod, "RBF Kernel SMO (sigma=0.7)")]:
    zz = predict_score(mod, grid_pts).reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=[-50, 0, 50], colors=["#dbeafe", "#fee2e2"], alpha=0.65)
    ax.contour(gx, gy, zz, levels=[-1, 0, 1], colors=["#2563eb", "#111827", "#dc2626"],
               linestyles=["--", "-", "--"], linewidths=[1.0, 1.8, 1.0])
    ax.scatter(X_train[y_train < 0, 0], X_train[y_train < 0, 1], c="#2563eb", s=16, label="y = -1")
    ax.scatter(X_train[y_train > 0, 0], X_train[y_train > 0, 1], c="#dc2626", s=16, label="y = +1")
    sv = mod["alpha"] > 1e-4
    ax.scatter(X_train[sv, 0], X_train[sv, 1], facecolors="none", edgecolors="#111827", s=52, lw=1.0, label="SV")
    ax.set(title=f"{ttl} ({sv.sum()} SVs)", xlabel="x1 (scaled)", ylabel="x2 (scaled)", xlim=(-2.4, 2.4), ylim=(-2.3, 2.3))
axes[0].legend(loc="upper right", fontsize=8)
axes[2].plot(rbf_mod["history"], color="#ea580c", lw=2.0, label="RBF SMO W(alpha)")
axes[2].plot(lin_mod["history"], color="#2563eb", lw=1.6, ls="--", label="Linear SMO W(alpha)")
axes[2].set(title="Dual Objective across SMO sweeps", xlabel="SMO sweep", ylabel="W(alpha)")
axes[2].legend(fontsize=8.5)
axes[2].grid(alpha=0.25)
fig.tight_layout()
fig.savefig("result.png", dpi=130)
