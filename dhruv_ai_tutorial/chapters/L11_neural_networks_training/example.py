"""Lecture 11 example: Neural Network Training शुरू से (सिर्फ़ NumPy).

sklearn digits (8x8 हाथ से लिखे अंक, 10 classes) पर 2-layer MLP:
1) Central finite differences से backpropagation का gradient check,
2) Zero init (symmetry trap) बनाम He initialization,
3) Mini-batch SGD, Momentum और Adam की तुलना,
4) छोटे subset पर L2 weight decay और inverted dropout से train-test gap घटाना।
Notation: X का हर column एक example है, यानी X का shape (64, m) है।
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

rng = np.random.default_rng(42)
digits = load_digits()
perm = rng.permutation(len(digits.data))
X_all, y_all = digits.data[perm].T.astype(float), digits.target[perm]
Y_all = np.eye(10)[y_all].T                             # one-hot (10, 1797)

# Train (1200) / Test (597) split और सिर्फ़ train statistics से input normalization
mu = X_all[:, :1200].mean(axis=1, keepdims=True)
std = X_all[:, :1200].std(axis=1, keepdims=True) + 1e-8
X_tr, Y_tr, y_tr = (X_all[:, :1200] - mu) / std, Y_all[:, :1200], y_all[:1200]
X_te, Y_te, y_te = (X_all[:, 1200:] - mu) / std, Y_all[:, 1200:], y_all[1200:]


def softmax(Z):
    e = np.exp(Z - Z.max(axis=0, keepdims=True))        # overflow से बचाव
    return e / e.sum(axis=0, keepdims=True)


def init_params(n_in, n_h, n_out, seed=0, zero=False, act="relu"):
    if zero:
        return {"W1": np.zeros((n_h, n_in)), "b1": np.zeros((n_h, 1)),
                "W2": np.zeros((n_out, n_h)), "b2": np.zeros((n_out, 1))}
    r = np.random.default_rng(seed)
    s1 = np.sqrt(2.0 / n_in) if act == "relu" else np.sqrt(1.0 / n_in)
    return {"W1": r.normal(0, s1, (n_h, n_in)), "b1": np.zeros((n_h, 1)),
            "W2": r.normal(0, np.sqrt(1.0 / n_h), (n_out, n_h)), "b2": np.zeros((n_out, 1))}


def forward_backward(X, Y, p, lam=0.0, keep_prob=1.0, r=None, act="relu", grad=True):
    """Forward pass (inverted dropout समेत) और chain rule से backpropagation."""
    m = X.shape[1]
    Z1 = p["W1"] @ X + p["b1"]                          # (n_h, m)
    A1 = np.maximum(0.0, Z1) if act == "relu" else np.tanh(Z1)
    mask = ((r.random(A1.shape) < keep_prob) / keep_prob) if keep_prob < 1.0 else None
    if mask is not None:                                # सिर्फ़ training में inverted dropout
        A1 = A1 * mask
    A2 = softmax(p["W2"] @ A1 + p["b2"])                # (10, m)
    ce = -np.sum(Y * np.log(np.clip(A2, 1e-12, 1.0))) / m
    reg = (lam / (2.0 * m)) * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2))
    if not grad:
        return ce + reg, A2
    dZ2 = (A2 - Y) / m                                  # delta^[2] / m, shape (10, m)
    dW2, db2 = dZ2 @ A1.T + (lam / m) * p["W2"], dZ2.sum(axis=1, keepdims=True)
    dA1 = (p["W2"].T @ dZ2) * (mask if mask is not None else 1.0)
    gp = (Z1 > 0).astype(float) if act == "relu" else (1.0 - np.tanh(Z1) ** 2)
    dZ1 = dA1 * gp                                      # delta^[1] / m, shape (n_h, m)
    dW1, db1 = dZ1 @ X.T + (lam / m) * p["W1"], dZ1.sum(axis=1, keepdims=True)
    return ce + reg, A2, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ---------- 1. Gradient checking: backprop बनाम central finite differences ----------
Xg, Yg = X_tr[:, :12], Y_tr[:, :12]
pg = init_params(64, 5, 10, seed=3, act="tanh")
_, _, ana = forward_backward(Xg, Yg, pg, lam=0.2, act="tanh")
eps, ana_vec, num_vec = 1e-6, [], []
for k in ("W1", "b1", "W2", "b2"):
    for idx in np.ndindex(pg[k].shape):
        orig = pg[k][idx]
        pg[k][idx] = orig + eps
        jp, _ = forward_backward(Xg, Yg, pg, lam=0.2, act="tanh", grad=False)
        pg[k][idx] = orig - eps
        jm, _ = forward_backward(Xg, Yg, pg, lam=0.2, act="tanh", grad=False)
        pg[k][idx] = orig
        ana_vec.append(ana[k][idx])
        num_vec.append((jp - jm) / (2.0 * eps))
ana_vec, num_vec = np.array(ana_vec), np.array(num_vec)
rel = np.linalg.norm(ana_vec - num_vec) / (np.linalg.norm(ana_vec) + np.linalg.norm(num_vec))
print(f"1) Gradient check (385 params, lam=0.2): max |diff| = {np.max(np.abs(ana_vec - num_vec)):.2e}, rel_err = {rel:.2e}")


def train_mlp(X, Y, n_h=64, opt="adam", lr=0.01, epochs=45, batch=32, lam=0.0, keep=1.0, zero=False, seed=1):
    p = init_params(X.shape[0], n_h, Y.shape[0], seed=seed, zero=zero)
    m_st = {k: np.zeros_like(v) for k, v in p.items()}
    v_st = {k: np.zeros_like(v) for k, v in p.items()}
    r, m, step, losses = np.random.default_rng(seed + 10), X.shape[1], 0, []
    for _ in range(epochs):
        order = r.permutation(m)
        for s in range(0, m, batch):
            _, _, g = forward_backward(X[:, order[s:s + batch]], Y[:, order[s:s + batch]], p, lam=lam, keep_prob=keep, r=r)
            step += 1
            for k in p:
                if opt == "sgd":
                    p[k] -= lr * g[k]
                elif opt == "momentum":
                    m_st[k] = 0.9 * m_st[k] + 0.1 * g[k]
                    p[k] -= lr * m_st[k]
                else:                                   # Adam (bias correction समेत)
                    m_st[k] = 0.9 * m_st[k] + 0.1 * g[k]
                    v_st[k] = 0.999 * v_st[k] + 0.001 * (g[k] ** 2)
                    p[k] -= lr * (m_st[k] / (1 - 0.9 ** step)) / (np.sqrt(v_st[k] / (1 - 0.999 ** step)) + 1e-8)
        losses.append(forward_backward(X, Y, p, grad=False)[0])
    return p, losses


def evaluate(X, Y, y, p):
    loss, probs = forward_backward(X, Y, p, grad=False)  # test समय पर dropout बंद
    return loss, 100.0 * np.mean(probs.argmax(axis=0) == y)


# ---------- 2. Symmetry problem बनाम He initialization ----------
p_zero, _ = train_mlp(X_tr, Y_tr, opt="adam", lr=0.003, epochs=25, zero=True)
p_he, _ = train_mlp(X_tr, Y_tr, opt="adam", lr=0.003, epochs=25, zero=False)
print(f"\n2) Init (25 epochs Adam): Zero init test acc = {evaluate(X_te, Y_te, y_te, p_zero)[1]:.1f}% "
      f"(symmetry trap!) | He init = {evaluate(X_te, Y_te, y_te, p_he)[1]:.1f}%")

# ---------- 3. Optimizers: Mini-batch SGD vs Momentum vs Adam ----------
print("\n3) Optimizers (64->64->10 ReLU MLP, B=32, 45 epochs):")
print("Optimizer        lr     train loss   train acc   test acc")
opt_curves = {}
for label, opt, lr in [("Mini-batch SGD", "sgd", 0.05), ("Momentum", "momentum", 0.15), ("Adam", "adam", 0.003)]:
    p_opt, hist = train_mlp(X_tr, Y_tr, opt=opt, lr=lr, epochs=45, seed=7)
    opt_curves[label] = hist
    tr_l, tr_a = evaluate(X_tr, Y_tr, y_tr, p_opt)
    print(f"{label:<15} {lr:<7.3f} {tr_l:10.4f} {tr_a:10.2f}% {evaluate(X_te, Y_te, y_te, p_opt)[1]:9.2f}%")

# ---------- 4. Regularization: छोटे subset (m=180) पर train-test gap ----------
Xs, Ys, ys = X_tr[:, :180], Y_tr[:, :180], y_tr[:180]
print("\n4) Regularization (m=180 train, 128 hidden units, 90 epochs Adam):")
print("Setting                 train acc   test acc   acc gap   test CE")
reg_bars = []
for name, lam, keep in [("No regularization", 0.0, 1.0), ("L2 (lambda=0.15)", 0.15, 1.0), ("Inverted Dropout p=0.6", 0.0, 0.6)]:
    pr, _ = train_mlp(Xs, Ys, n_h=128, opt="adam", lr=0.003, epochs=90, batch=30, lam=lam, keep=keep, seed=5)
    _, tr_a = evaluate(Xs, Ys, ys, pr)
    te_ce, te_a = evaluate(X_te, Y_te, y_te, pr)
    reg_bars.append((name, tr_a, te_a))
    print(f"{name:<23} {tr_a:8.2f}% {te_a:9.2f}% {tr_a - te_a:8.2f}% {te_ce:9.4f}")

# ---------- 5. Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for label, hist in opt_curves.items():
    axes[0].plot(range(1, len(hist) + 1), hist, lw=2, label=label)
axes[0].set(yscale="log", xlabel="epoch", ylabel="cross-entropy loss", title="Digits MLP: SGD vs Momentum vs Adam")
axes[0].legend()
xp = np.arange(len(reg_bars))
axes[1].bar(xp - 0.17, [r[1] for r in reg_bars], 0.34, label="train acc (%)", color="#2563eb")
axes[1].bar(xp + 0.17, [r[2] for r in reg_bars], 0.34, label="test acc (%)", color="#16a34a")
axes[1].set_xticks(xp, ["None", "L2 decay", "Dropout"])
axes[1].set(ylim=(80, 103), ylabel="accuracy (%)", title="Regularization shrinks train-test gap (m=180)")
axes[1].legend(loc="lower right")
fig.tight_layout()
fig.savefig("result.png", dpi=130)
