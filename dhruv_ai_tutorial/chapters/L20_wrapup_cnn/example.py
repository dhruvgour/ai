"""Lecture 20 example: NumPy से 2D Convolution, Max Pooling और छोटा CNN (from-scratch backprop).

1) load_digits के एक अंक पर हाथ से बने vertical और horizontal edge filters तथा 2x2 Max Pooling चलाते हैं।
2) फिर एक छोटा CNN (Conv 3x3 -> ReLU -> 2x2 MaxPool -> Softmax) पूरे from-scratch backpropagation से
   सिखाते हैं (लगभग 5 सेकंड में, 20 s की सीमा से काफ़ी कम) और कच्चे pixels पर सिखाए Softmax से तुलना करते हैं।
"""
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

rng = np.random.default_rng(20)


# ---------- 1. From-scratch 2D Convolution (s=1) और 2x2 Max Pooling ----------
def conv2d_forward(X, K, b, pad=1):
    """X: (B, H, W), filters K: (C, f, f), bias b: (C,) -> Z: (B, C, H_out, W_out)."""
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(Xp, K.shape[1:], axis=(1, 2))
    return np.einsum("bhwuv,cuv->bchw", win, K) + b[:, None, None], (X.shape, win, K, pad)


def conv2d_backward(dZ, cache, need_dx=False):
    """Chain rule से dK, db और (चाहिए तो) 180-degree पलटे filter से सटीक dX।"""
    (_, H, W), win, K, pad = cache
    dK = np.einsum("bchw,bhwuv->cuv", dZ, win)
    db = dZ.sum(axis=(0, 2, 3))
    if not need_dx:
        return None, dK, db
    f = K.shape[1]
    dZ_pad = np.pad(dZ, ((0, 0), (0, 0), (f - 1, f - 1), (f - 1, f - 1)))
    w_in = sliding_window_view(dZ_pad, (f, f), axis=(2, 3))
    dXp = np.einsum("bchwuv,cuv->bhw", w_in, K[:, ::-1, ::-1])
    return dXp[:, pad:pad + H, pad:pad + W], dK, db


def maxpool2x2_forward(A):
    """A: (B, C, H, W) -> P: (B, C, H/2, W/2), विजेता pixel का index बचाते हुए।"""
    B, C, H, W = A.shape
    Af = A.reshape(B, C, H // 2, 2, W // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, H // 2, W // 2, 4)
    return Af.max(axis=-1), (A.shape, Af.argmax(axis=-1))


def maxpool2x2_backward(dP, cache):
    """Gradient केवल forward pass के argmax विजेता स्थान को मिलता है।"""
    (B, C, H, W), idx = cache
    dAf = np.zeros((B, C, H // 2, W // 2, 4))
    np.put_along_axis(dAf, idx[..., None], dP[..., None], axis=-1)
    return dAf.reshape(B, C, H // 2, W // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------- 2. Dataset और Edge Filters का प्रदर्शन ----------
digits = load_digits()
X_all, y_all = digits.images / 16.0, digits.target
perm = rng.permutation(len(y_all))
X_tr, y_tr = X_all[perm[:1000]], y_all[perm[:1000]]
X_te, y_te = X_all[perm[1000:1500]], y_all[perm[1000:1500]]

k_vert = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
sample = X_te[0:1]
edges, _ = conv2d_forward(sample, np.stack([k_vert, k_vert.T]), np.zeros(2), pad=1)
pooled_edges, _ = maxpool2x2_forward(np.maximum(0.0, edges))
print(f"Output size सूत्र जाँच: n=8, p=1, f=3, s=1 -> conv {edges.shape[2]}x{edges.shape[3]}, "
      f"2x2 pool -> {pooled_edges.shape[2]}x{pooled_edges.shape[3]}")
print(f"Sample digit '{y_te[0]}': max vertical-edge = {edges[0, 0].max():.2f}, "
      f"max horizontal-edge = {edges[0, 1].max():.2f}")

# ---------- 3. Tiny CNN: Forward, Backprop और Finite-difference जाँच ----------
K0, b0 = rng.normal(0, 0.25, (8, 3, 3)), np.zeros(8)
W0, c0 = rng.normal(0, 0.15, (8 * 4 * 4, 10)), np.zeros(10)


def cnn_step(Xb, yb, K, b, W, c, reg=1e-4, backward=True, need_dx=False):
    Z, c_conv = conv2d_forward(Xb, K, b, pad=1)
    P, c_pool = maxpool2x2_forward(np.maximum(0.0, Z))
    flat = P.reshape(len(Xb), -1)
    probs = softmax(flat @ W + c)
    loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12)) + 0.5 * reg * (np.sum(K**2) + np.sum(W**2))
    preds = probs.argmax(axis=1)
    if not backward:
        return loss, preds, None
    dlogits = probs.copy()
    dlogits[np.arange(len(yb)), yb] -= 1.0
    dlogits /= len(yb)
    dW, dc = flat.T @ dlogits + reg * W, dlogits.sum(axis=0)
    dZ = maxpool2x2_backward((dlogits @ W.T).reshape(P.shape), c_pool) * (Z > 0)
    dX, dK, db = conv2d_backward(dZ, c_conv, need_dx=need_dx)
    return loss, preds, (dX, dK + reg * K, db, dW, dc)


_, _, (dX_a, dK_a, *_ ) = cnn_step(X_tr[:6], y_tr[:6], K0, b0, W0, c0, need_dx=True)
eps = 1e-5
Kp, Km = K0.copy(), K0.copy(); Kp[0, 1, 2] += eps; Km[0, 1, 2] -= eps
Xp, Xm = X_tr[:6].copy(), X_tr[:6].copy(); Xp[0, 3, 4] += eps; Xm[0, 3, 4] -= eps
gK = (cnn_step(X_tr[:6], y_tr[:6], Kp, b0, W0, c0, backward=False)[0]
      - cnn_step(X_tr[:6], y_tr[:6], Km, b0, W0, c0, backward=False)[0]) / (2 * eps)
gX = (cnn_step(Xp, y_tr[:6], K0, b0, W0, c0, backward=False)[0]
      - cnn_step(Xm, y_tr[:6], K0, b0, W0, c0, backward=False)[0]) / (2 * eps)
print(f"Gradient check: |dK_err| = {abs(dK_a[0, 1, 2] - gK):.2e}, |dX_err| = {abs(dX_a[0, 3, 4] - gX):.2e}\n")

# ---------- 4. Training: Tiny CNN बनाम Raw-Pixel Softmax ----------
t0 = time.time()
K, b, W, c = K0.copy(), b0.copy(), W0.copy(), c0.copy()
Ws, bs = np.zeros((64, 10)), np.zeros(10)
cnn_acc_hist, sm_acc_hist = [], []
print("Epoch   CNN loss   CNN test acc   Softmax test acc")
for ep in range(1, 81):
    for idx in np.array_split(rng.permutation(len(X_tr)), 20):
        Xb, yb = X_tr[idx], y_tr[idx]
        _, _, (_, dK, db, dW, dc) = cnn_step(Xb, yb, K, b, W, c)
        K -= 0.35 * dK; b -= 0.35 * db; W -= 0.35 * dW; c -= 0.35 * dc
        Xf = Xb.reshape(len(Xb), -1)
        ps = softmax(Xf @ Ws + bs); ps[np.arange(len(yb)), yb] -= 1.0
        Ws -= 0.5 * (Xf.T @ ps / len(yb) + 1e-4 * Ws); bs -= 0.5 * ps.mean(axis=0)
    tr_loss, _, _ = cnn_step(X_tr, y_tr, K, b, W, c, backward=False)
    _, te_pred, _ = cnn_step(X_te, y_te, K, b, W, c, backward=False)
    sm_pred = (X_te.reshape(len(X_te), -1) @ Ws + bs).argmax(axis=1)
    cnn_acc_hist.append(np.mean(te_pred == y_te) * 100)
    sm_acc_hist.append(np.mean(sm_pred == y_te) * 100)
    if ep in (1, 10, 25, 50, 80):
        print(f"{ep:5d}   {tr_loss:8.4f}      {cnn_acc_hist[-1]:7.2f}%           {sm_acc_hist[-1]:7.2f}%")

X_shift = np.pad(X_te, ((0, 0), (0, 0), (1, 0)))[:, :, :8]
cnn_sh = np.mean(cnn_step(X_shift, y_te, K, b, W, c, backward=False)[1] == y_te) * 100
sm_sh = np.mean((X_shift.reshape(len(X_shift), -1) @ Ws + bs).argmax(axis=1) == y_te) * 100
print(f"\nTraining समय: {time.time() - t0:.2f} s | CNN params: {K.size + b.size + W.size + c.size}, Softmax params: {Ws.size + bs.size}")
print(f"Clean test accuracy  -> Tiny CNN: {cnn_acc_hist[-1]:.2f}%  बनाम  Raw Softmax: {sm_acc_hist[-1]:.2f}%")
print(f"1-px shifted test    -> Tiny CNN: {cnn_sh:.2f}%  बनाम  Raw Softmax: {sm_sh:.2f}% (pooling + sharing का लाभ)")

# ---------- 5. Plots ----------
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
axes[0].imshow(sample[0], cmap="gray_r"); axes[0].set_title(f"Input 8x8 ('{y_te[0]}')"); axes[0].axis("off")
axes[1].imshow(edges[0, 0], cmap="coolwarm"); axes[1].set_title("Vertical-edge conv"); axes[1].axis("off")
axes[2].imshow(pooled_edges[0, 0], cmap="viridis"); axes[2].set_title("ReLU + 2x2 MaxPool"); axes[2].axis("off")
axes[3].plot(cnn_acc_hist, lw=2, label=f"Tiny CNN ({cnn_acc_hist[-1]:.1f}%)")
axes[3].plot(sm_acc_hist, lw=2, ls="--", label=f"Raw Softmax ({sm_acc_hist[-1]:.1f}%)")
axes[3].set(xlabel="epoch", ylabel="test accuracy (%)", title="CNN vs Softmax on digits")
axes[3].legend(loc="lower right"); axes[3].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("result.png", dpi=130)
