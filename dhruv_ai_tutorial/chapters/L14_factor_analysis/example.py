"""Lecture 14 example: Factor Analysis को EM से fit करना, शुरू से (सिर्फ़ NumPy).

Data एक जाने-पहचाने model से बनता है: x = mu + Lambda z + epsilon, जहाँ z सिर्फ़ k = 3 dimension का है
पर x के n = 30 dimension हैं। Training examples सिर्फ़ m = 20 हैं, यानी m < n।
दिखाते हैं: (a) साधारण sample covariance singular है, (b) EM में log-likelihood कभी नहीं घटता,
(c) Factor Analysis का Sigma = Lambda Lambda^T + Psi invertible है और सच के काफ़ी पास है।
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis      # सिर्फ़ अंत में तुलना के लिए

rng = np.random.default_rng(14)

# ---------- 1. सच्चा model और उससे data ----------
n, k, m = 30, 3, 20
Lambda_true = rng.normal(0, 1.0, (n, k))               # factor loading matrix (n x k)
Psi_true = rng.uniform(0.2, 0.8, n)                    # diagonal noise variances
mu_true = rng.normal(0, 2.0, n)
Sigma_true = Lambda_true @ Lambda_true.T + np.diag(Psi_true)


def sample(count):
    z = rng.normal(size=(count, k))                    # z ~ N(0, I)
    eps = rng.normal(size=(count, n)) * np.sqrt(Psi_true)   # epsilon ~ N(0, Psi)
    return mu_true + z @ Lambda_true.T + eps           # x = mu + Lambda z + epsilon


X, X_test = sample(m), sample(2000)


def avg_loglik(X, mu, Sigma):
    """(1/m) sum_i log N(x_i; mu, Sigma)."""
    d = X - mu
    maha = np.einsum("ij,ij->i", d, np.linalg.solve(Sigma, d.T).T)
    return -0.5 * np.mean(maha + np.linalg.slogdet(Sigma)[1] + X.shape[1] * np.log(2 * np.pi))


# ---------- 2. Factor Analysis का EM ----------
def factor_analysis(X, k, iters=300):
    m, n = X.shape
    mu = X.mean(axis=0)                                # mu का M-step हर बार यही देता है
    Xc = X - mu
    S = Xc.T @ Xc / m                                  # sample covariance
    Lambda = np.random.default_rng(0).normal(0, 0.5, (n, k))
    Psi = np.diag(S).copy()
    history = []
    for _ in range(iters):
        Sigma = Lambda @ Lambda.T + np.diag(Psi)       # x का marginal covariance
        history.append(avg_loglik(X, mu, Sigma))
        # E-step: z | x ~ N(mu_z, Sigma_z)
        B = Lambda.T @ np.linalg.inv(Sigma)            # Lambda^T (Lambda Lambda^T + Psi)^-1
        Mz = Xc @ B.T                                  # row i = mu_{z|x_i}
        Sz = np.eye(k) - B @ Lambda                    # Sigma_{z|x}, हर i के लिए एक ही
        # M-step: ध्यान दें कि E[z z^T] = mu_z mu_z^T + Sigma_z चाहिए, सिर्फ़ mu_z नहीं
        Ezz = Mz.T @ Mz / m + Sz                       # (1/m) sum_i E[z z^T]
        Cxz = Xc.T @ Mz / m                            # (1/m) sum_i (x_i - mu) mu_z^T
        Lambda = Cxz @ np.linalg.inv(Ezz)
        Phi = S - Cxz @ Lambda.T - Lambda @ Cxz.T + Lambda @ Ezz @ Lambda.T
        Psi = np.diag(Phi).copy()                      # Psi में सिर्फ़ diagonal रखते हैं
    return mu, Lambda, Psi, np.array(history)


mu, Lambda, Psi, history = factor_analysis(X, k)
Sigma_fa = Lambda @ Lambda.T + np.diag(Psi)

# ---------- 3. Sample covariance singular है ----------
S = np.cov(X.T, bias=True)
eig_S, eig_fa = np.linalg.eigvalsh(S), np.linalg.eigvalsh(Sigma_fa)
print(f"n = {n} dimensions, m = {m} examples, k = {k} factors")
print(f"Sample covariance:   rank = {np.linalg.matrix_rank(S)} (n = {n} चाहिए),  "
      f"सबसे छोटा eigenvalue = {eig_S.min():.1e}")
print(f"Factor Analysis:     rank = {np.linalg.matrix_rank(Sigma_fa)},  सबसे छोटा eigenvalue = {eig_fa.min():.3f}")
print(f"Parameters: full Sigma में {n * (n + 1) // 2}, Factor Analysis (Lambda + Psi) में {n * k + n}")

# ---------- 4. EM का log-likelihood ----------
print("\nEM: iteration और average log-likelihood (training data पर)")
for it in (0, 1, 2, 5, 10, 20, 50, 100, 200, 299):
    print(f"  {it:3d}   {history[it]:9.4f}")
print(f"  log-likelihood कभी घटा? {'हाँ' if np.any(np.diff(history) < -1e-10) else 'नहीं'}")

# ---------- 5. अलग-अलग covariance models की तुलना ----------
def rel_error(Sigma):
    return np.linalg.norm(Sigma - Sigma_true) / np.linalg.norm(Sigma_true)


def test_ll(Sigma):
    return avg_loglik(X_test, mu, Sigma)


sk = FactorAnalysis(n_components=k, tol=1e-6, max_iter=5000).fit(X)
models = [("Full (sample cov)", S), ("Diagonal", np.diag(np.diag(S))),
          ("Spherical sigma^2 I", np.eye(n) * np.trace(S) / n), ("Factor Analysis (EM)", Sigma_fa),
          ("sklearn FactorAnalysis", sk.get_covariance()), ("True Sigma", Sigma_true)]
print("\nModel                     error vs true Sigma    test log-likelihood")
for name, Sigma in models:
    singular = np.linalg.eigvalsh(Sigma).min() < 1e-10
    ll = "  undefined (singular)" if singular else f"{test_ll(Sigma):12.2f}"
    print(f"{name:<25} {rel_error(Sigma):12.3f}        {ll}")
print(f"\nहमारा EM बनाम sklearn, training log-likelihood: {history[-1]:.4f} बनाम {sk.score(X):.4f}")

# ---------- 6. Plots ----------
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
axes[0].plot(history, lw=2)
axes[0].set(xlabel="EM iteration", ylabel="average log-likelihood", title="EM never goes down", xscale="symlog")
vmax = np.abs(Sigma_true).max()
for ax, Sigma, title in [(axes[1], Sigma_true, "True covariance"), (axes[2], S, f"Sample covariance (m={m})"),
                         (axes[3], Sigma_fa, "Factor Analysis")]:
    ax.imshow(Sigma, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set(title=title, xticks=[], yticks=[])
fig.tight_layout()
fig.savefig("result.png", dpi=130)
