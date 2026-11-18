"""Lecture 15 example: SVD से PCA और Sigmoid-CDF से ICA (सिर्फ़ NumPy).

1. Handwritten digits (load_digits, 64 dimensions) पर SVD से PCA: variance retained बनाम k
   और reconstruction error की जाँच।
2. Laplace-distributed (super-Gaussian) स्वतंत्र sources को mixing matrix A से मिलाकर
   CS229 के sigmoid-CDF stochastic gradient ascent से वापस अलग करना।
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

rng = np.random.default_rng(42)

# ==================== भाग 1: load_digits पर SVD से PCA ====================
digits = load_digits()
X_raw = digits.data.astype(float)                  # (1797, 64), हर उदाहरण 8x8 तस्वीर
m_dig, n_dig = X_raw.shape

# Preprocessing: mean शून्य करें (सभी pixels एक ही 0..16 पैमाने पर हैं)
mu = X_raw.mean(axis=0)
X_c = X_raw - mu

# Thin SVD: X_c = P S V^T, जहाँ V के columns ही Sigma = (1/m) X_c^T X_c के eigenvectors हैं
P, svals, Vt = np.linalg.svd(X_c, full_matrices=False)
eigvals = (svals ** 2) / m_dig                     # lambda_j = s_j^2 / m
total_var = eigvals.sum()
cum_var = np.cumsum(eigvals) / total_var

# जाँचें कि Sigma के सीधे eigenvalues और s_j^2 / m बराबर हैं
Sigma = (X_c.T @ X_c) / m_dig
eig_direct = np.linalg.eigvalsh(Sigma)[::-1]
print(f"PCA on load_digits: m = {m_dig}, n = {n_dig}, total variance = {total_var:.2f}")
print(f"SVD बनाम eig(Sigma) अधिकतम अंतर: {np.max(np.abs(eigvals - eig_direct)):.2e}\n")

print("  k   variance retained (%)   reconstruction MSE   (retained + MSE)")
for k in (1, 2, 5, 10, 20, 29, 41, 50):
    Uk = Vt[:k].T                                  # (64, k) top-k principal directions
    Y = X_c @ Uk                                   # (m, k) compressed coordinates
    X_rec = Y @ Uk.T                               # (m, 64) reconstruction
    mse = np.mean(np.sum((X_c - X_rec) ** 2, axis=1))
    kept = eigvals[:k].sum()
    print(f"{k:3d}          {100 * cum_var[k - 1]:6.2f}%              {mse:8.2f}          {kept + mse:8.2f}")

for target in (0.90, 0.95, 0.99):
    k_min = int(np.searchsorted(cum_var, target) + 1)
    print(f"{int(100 * target)}% variance के लिए सबसे छोटा k = {k_min} (64 से घटकर {k_min} features)")

# ==================== भाग 2: Laplace Sources पर Sigmoid-CDF ICA ====================
m_ica, n_ica = 4000, 4
# Super-Gaussian मूल sources: Laplace(0, 1), जिनकी tails Gaussian से भारी हैं
S_true = rng.laplace(loc=0.0, scale=1.0, size=(m_ica, n_ica))
S_true -= S_true.mean(axis=0)

# अज्ञात mixing matrix A से माइक्रोफ़ोन मिश्रण x^(i) = A s^(i)
A = np.array([[1.0,  0.6, -0.4,  0.3],
              [-0.5, 1.1,  0.5, -0.2],
              [0.3, -0.4,  1.0,  0.6],
              [0.4,  0.2, -0.5,  0.9]])
X_mix = S_true @ A.T
X_mix -= X_mix.mean(axis=0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# CS229 Stochastic Gradient Ascent: W := W + alpha * ((1 - 2g(Wx)) x^T + (W^T)^-1)
W = rng.normal(0.0, 0.4, size=(n_ica, n_ica)) + np.eye(n_ica)
alpha = 0.004
for epoch in range(25):
    for i in rng.permutation(m_ica):
        x = X_mix[i]
        g = sigmoid(W @ x)
        W += alpha * (np.outer(1.0 - 2.0 * g, x) + np.linalg.inv(W.T))
    alpha *= 0.96

S_hat = X_mix @ W.T                                # अनुमानित स्वतंत्र sources

# Absolute correlation matrix: rows = recovered s_hat_i, cols = true s_j
corr_abs = np.abs(np.corrcoef(S_hat.T, S_true.T)[:n_ica, n_ica:])
print("\nICA (Laplace super-Gaussian sources): |corr(s_hat_i, s_j)| matrix:")
header = "         " + "  ".join(f"true s_{j + 1}" for j in range(n_ica)) + "   row max"
print(header)
for i in range(n_ica):
    row_str = "  ".join(f"{corr_abs[i, j]:8.4f}" for j in range(n_ica))
    print(f"s_hat_{i + 1}  {row_str}   {corr_abs[i].max():7.4f}")

# ==================== भाग 3: परिणामों का चित्र ====================
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))

# (a) PCA cumulative variance retained बनाम k
ks = np.arange(1, n_dig + 1)
axes[0, 0].plot(ks, 100 * cum_var, color="#2563eb", lw=2)
for pct, col in ((90, "#ca8a04"), (95, "#16a34a"), (99, "#dc2626")):
    k_star = int(np.searchsorted(cum_var, pct / 100.0) + 1)
    axes[0, 0].axhline(pct, color=col, ls="--", lw=1, label=f"{pct}% at k={k_star}")
    axes[0, 0].axvline(k_star, color=col, ls=":", lw=1)
axes[0, 0].set(xlabel="Number of components k", ylabel="Variance retained (%)",
               title="PCA on load_digits: Variance vs k")
axes[0, 0].legend(fontsize=8.5)

# (b) एक digit का अलग-अलग k से reconstruction
sample_idx = 0
panels = []
for k in (2, 8, 29, 64):
    rec = mu + (X_c[sample_idx] @ Vt[:k].T) @ Vt[:k]
    panels.append(rec.reshape(8, 8))
strip = np.hstack([np.pad(p, ((0, 0), (0, 1)), constant_values=16.0) for p in panels])
axes[0, 1].imshow(strip, cmap="gray_r", vmin=0, vmax=16)
axes[0, 1].set_xticks([3.5, 12.5, 21.5, 30.5])
axes[0, 1].set_xticklabels(["k=2", "k=8", "k=29 (95%)", "k=64 (orig)"])
axes[0, 1].set_yticks([])
axes[0, 1].set_title("Digit Reconstruction Across k")

# (c) पहले true Laplace source, mixed signal, और matched unmixed signal की तुलना
match_row = int(np.argmax(corr_abs[:, 0]))
sign_fix = np.sign(np.corrcoef(S_hat[:, match_row], S_true[:, 0])[0, 1])
sl = slice(0, 120)
axes[1, 0].plot(S_true[sl, 0] / S_true[:, 0].std(), label="True Laplace s_1", lw=1.6, color="#111827")
axes[1, 0].plot(X_mix[sl, 0] / X_mix[:, 0].std() + 4.2, label="Mixed mic x_1 (shifted)", lw=1.1, color="#dc2626")
axes[1, 0].plot(sign_fix * S_hat[sl, match_row] / S_hat[:, match_row].std() - 4.2,
                label=f"Recovered s_hat_{match_row + 1} (shifted)", lw=1.4, color="#16a34a")
axes[1, 0].set(xlabel="Sample index", ylabel="Normalized amplitude", title="ICA Cocktail Party Unmixing")
axes[1, 0].legend(fontsize=8, loc="upper right")

# (d) Absolute correlation matrix heatmap
im = axes[1, 1].imshow(corr_abs, cmap="Blues", vmin=0.0, vmax=1.0)
for r in range(n_ica):
    for c in range(n_ica):
        col_txt = "white" if corr_abs[r, c] > 0.55 else "#111827"
        axes[1, 1].text(c, r, f"{corr_abs[r, c]:.3f}", ha="center", va="center", color=col_txt, fontsize=9.5)
axes[1, 1].set_xticks(range(n_ica), [f"s_{j + 1}" for j in range(n_ica)])
axes[1, 1].set_yticks(range(n_ica), [f"s_hat_{i + 1}" for i in range(n_ica)])
axes[1, 1].set_title("|Correlation| (Recovered vs True Sources)")
fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

fig.tight_layout()
fig.savefig("result.png", dpi=130)
