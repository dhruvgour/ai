"""Lecture 13 example: K-means और Mixture of Gaussians का EM, दोनों शुरू से (सिर्फ़ NumPy).

चार 2D blobs जिनके covariance अलग-अलग हैं (एक लंबा-पतला, एक छोटा-गोल, दो फैले हुए)।
Labels हम algorithm को नहीं दिखाते, सिर्फ़ अंत में जाँच के लिए इस्तेमाल करते हैं।
दिखाते हैं: (a) k-means में distortion J कभी नहीं बढ़ता, (b) अलग initialization पर अलग J,
(c) EM में log-likelihood कभी नहीं घटता, (d) EM की soft assignments k-means से बेहतर clusters देती हैं।
"""
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(13)

# ---------- 1. Data: mixture of 4 Gaussians से sample करें ----------
k = 4
true_phi = np.array([0.35, 0.25, 0.20, 0.20])
true_mu = np.array([[0.0, 0.0], [3.0, 2.2], [0.5, 5.0], [7.0, 5.0]])
true_Sigma = np.array([[[4.0, 1.7], [1.7, 0.9]],      # लंबा-पतला, तिरछा
                       [[0.15, 0.0], [0.0, 0.15]],    # छोटा और गोल
                       [[1.6, -0.5], [-0.5, 0.6]],    # फैला हुआ
                       [[0.5, 0.0], [0.0, 1.2]]])     # खड़ा ellipse, बाकियों से दूर
m = 600
z_true = rng.choice(k, size=m, p=true_phi)             # latent z^(i), जो असल में छिपा रहता है
X = np.array([rng.multivariate_normal(true_mu[j], true_Sigma[j]) for j in z_true])


# ---------- 2. K-means ----------
def kmeans(X, init_mu):
    """Returns (c, mu, history). history में हर आधे step के बाद का distortion J है।"""
    mu, c, history = init_mu.copy(), None, []
    while True:
        dist2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)       # (m, k)
        new_c = dist2.argmin(axis=1)                   # step (i): सबसे पास वाला centroid
        if c is not None and np.all(new_c == c):
            return c, mu, history                      # कुछ नहीं बदला: converge
        c = new_c
        history.append(dist2[np.arange(len(X)), c].sum())                 # assign के बाद J
        mu = np.array([X[c == j].mean(axis=0) if np.any(c == j) else mu[j] for j in range(len(mu))])
        history.append(((X - mu[c]) ** 2).sum())       # step (ii): centroid को mean पर ले जाने के बाद J


# ---------- 3. EM for Mixture of Gaussians ----------
def log_gaussian(X, mu, Sigma):
    """हर row x के लिए log N(x; mu, Sigma)."""
    d = X - mu
    maha = np.einsum("ij,ij->i", d @ np.linalg.inv(Sigma), d)
    return -0.5 * (maha + np.log(np.linalg.det(Sigma)) + X.shape[1] * np.log(2 * np.pi))


def em_mixture(X, init_mu, iters=60):
    m, n = X.shape
    k = len(init_mu)
    phi, mu = np.full(k, 1.0 / k), init_mu.copy()
    Sigma = np.array([np.cov(X.T) for _ in range(k)])  # शुरुआत: सबका Sigma = पूरे data का covariance
    history = []
    for _ in range(iters):
        # E-step: w[i, j] = p(z = j | x_i), Bayes rule से (log में, ताकि underflow न हो)
        log_joint = np.log(phi) + np.column_stack([log_gaussian(X, mu[j], Sigma[j]) for j in range(k)])
        top = log_joint.max(axis=1, keepdims=True)
        log_px = top + np.log(np.exp(log_joint - top).sum(axis=1, keepdims=True))   # log p(x_i)
        w = np.exp(log_joint - log_px)
        history.append(log_px.sum())                   # मौजूदा parameters पर log-likelihood
        # M-step: weighted counts, weighted means, weighted covariances
        Nj = w.sum(axis=0)
        phi = Nj / m
        mu = (w.T @ X) / Nj[:, None]
        for j in range(k):
            d = X - mu[j]
            Sigma[j] = (w[:, j, None] * d).T @ d / Nj[j]
    return w, phi, mu, Sigma, history


def relabel(c):
    """Cluster के नाम मनमाने हैं, इसलिए वह permutation चुनें जो true z से सबसे ज़्यादा मिले।"""
    best = max(permutations(range(k)), key=lambda p: np.mean(np.array(p)[c] == z_true))
    return np.array(best)[c]


def accuracy(c):
    return np.mean(relabel(c) == z_true)


# ---------- 4. K-means को 8 अलग random initializations से चलाएँ ----------
runs = [kmeans(X, X[np.random.default_rng(seed).choice(m, k, replace=False)]) for seed in range(8)]
print("K-means, 8 random restarts (final distortion J):")
print("  " + "  ".join(f"{h[-1]:7.1f}" for _, _, h in runs))
c_km, mu_km, hist_km = min(runs, key=lambda r: r[2][-1])           # सबसे छोटा J वाला run रखें
_, _, hist_bad = max(runs, key=lambda r: r[2][-1])
print(f"  सबसे अच्छा J = {hist_km[-1]:.1f}, सबसे खराब J = {hist_bad[-1]:.1f} (local optimum)")
print("\nसबसे अच्छे run में J, हर आधे step के बाद (assign, move, assign, ...):")
print("  " + " ".join(f"{j:.1f}" for j in hist_km[:10]))
print(f"  J कभी बढ़ा? {'हाँ' if np.any(np.diff(hist_km) > 1e-9) else 'नहीं'}")

# ---------- 5. EM चलाएँ, k-means के centroids से शुरू करके ----------
w, phi, mu, Sigma, hist_em = em_mixture(X, mu_km)
c_em = w.argmax(axis=1)
print("\nEM: iteration और log-likelihood")
for it in (0, 1, 2, 3, 5, 10, 20, 40, 59):
    print(f"  {it:3d}   {hist_em[it]:10.3f}")
print(f"  log-likelihood कभी घटा? {'हाँ' if np.any(np.diff(hist_em) < -1e-9) else 'नहीं'}")

# ---------- 6. नतीजों की तुलना ----------
order = [int(np.argmin(((mu - t) ** 2).sum(axis=1))) for t in true_mu]   # EM cluster <-> true cluster
print("\ncluster   true phi   EM phi    true mu          EM mu")
for t, j in enumerate(order):
    print(f"   {t}       {true_phi[t]:.2f}      {phi[j]:.3f}    ({true_mu[t, 0]:4.1f}, {true_mu[t, 1]:4.1f})"
          f"     ({mu[j, 0]:5.2f}, {mu[j, 1]:5.2f})")
print(f"\nसही cluster में गए points: K-means {100 * accuracy(c_km):.1f}%,  EM {100 * accuracy(c_em):.1f}%")
unsure = np.sum(w.max(axis=1) < 0.9)
print(f"EM की soft assignment: {unsure} points ऐसे हैं जिनका सबसे बड़ा w भी 0.9 से कम है (overlap वाले)")

# ---------- 7. Plots ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()
colors = np.array(["#2563eb", "#dc2626", "#16a34a", "#ea580c"])
for ax, labels, title in [(axes[0], z_true, "True clusters"),
                          (axes[1], c_km, f"K-means ({100 * accuracy(c_km):.0f}% correct)"),
                          (axes[2], c_em, f"EM, mixture of Gaussians ({100 * accuracy(c_em):.0f}%)")]:
    ax.scatter(X[:, 0], X[:, 1], c=colors[relabel(labels)], s=7)
    ax.set(title=title, xlabel="x1", ylabel="x2")
axes[1].scatter(mu_km[:, 0], mu_km[:, 1], marker="X", s=120, c="black")
angle = np.linspace(0, 2 * np.pi, 100)
circle = np.c_[np.cos(angle), np.sin(angle)]
for j in range(k):                                     # हर Gaussian का 2-sigma ellipse
    ellipse = mu[j] + 2 * circle @ np.linalg.cholesky(Sigma[j]).T
    axes[2].plot(ellipse[:, 0], ellipse[:, 1], "k-", lw=1.2)
axes[3].plot(hist_em, "o-", ms=3)
axes[3].set(title="EM: log-likelihood never decreases", xlabel="iteration", ylabel="log-likelihood")
fig.tight_layout()
fig.savefig("result.png", dpi=130)
