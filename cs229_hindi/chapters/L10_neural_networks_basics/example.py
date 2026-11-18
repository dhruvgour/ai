"""Lecture 10 example: 2-layer Neural Network शुरू से (सिर्फ़ NumPy).

दो problems जिन्हें कोई सीधी line हल नहीं कर सकती: XOR और two-moons। दोनों पर
Logistic Regression (एक अकेला neuron) और एक hidden layer वाला network चलाते हैं,
और फिर देखते हैं कि hidden units ने क्या सीखा।
Notation theory जैसा ही है: X का हर column एक example है, यानी X का shape (n, m) है।
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------- 1. Data ----------
X_xor = np.array([[0.0, 0.0, 1.0, 1.0],
                  [0.0, 1.0, 0.0, 1.0]])               # shape (2, 4): हर column एक point
Y_xor = np.array([[0.0, 1.0, 1.0, 0.0]])               # shape (1, 4)

moons_x, moons_y = make_moons(1000, noise=0.2, random_state=0)
X_train, Y_train = moons_x[:500].T, moons_y[:500][None, :].astype(float)
X_test, Y_test = moons_x[500:].T, moons_y[500:][None, :].astype(float)


# ---------- 2. Forward propagation (vectorized: सारे examples एक साथ) ----------
def forward(X, p):
    Z1 = p["W1"] @ X + p["b1"]          # (n1, m); b1 का shape (n1, 1) है, broadcasting उसे हर column में जोड़ता है
    A1 = np.tanh(Z1)                    # hidden layer
    Z2 = p["W2"] @ A1 + p["b2"]         # (1, m)
    A2 = sigmoid(Z2)                    # output: P(y = 1 | x)
    return A1, A2


def log_loss(A2, Y):
    A2 = np.clip(A2, 1e-12, 1 - 1e-12)                 # log(0) से बचाव
    return -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))


def accuracy(A2, Y):
    return np.mean((A2 > 0.5) == Y)


# ---------- 3. Training: Gradient Descent ----------
def train_network(X, Y, n_hidden, alpha, iters, seed):
    rng = np.random.default_rng(seed)
    n, m = X.shape
    p = {"W1": rng.normal(0, 1, (n_hidden, n)), "b1": np.zeros((n_hidden, 1)),   # छोटे random weights
         "W2": rng.normal(0, 1, (1, n_hidden)), "b2": np.zeros((1, 1))}
    history = []
    for _ in range(iters):
        A1, A2 = forward(X, p)
        history.append(log_loss(A2, Y))
        # Gradients: ये formulas अगले lecture (backpropagation) में derive होंगे।
        dZ2 = (A2 - Y) / m                             # output layer की गलती
        dZ1 = (p["W2"].T @ dZ2) * (1 - A1 ** 2)        # वही गलती hidden layer तक पीछे भेजी गई
        grads = {"W2": dZ2 @ A1.T, "b2": dZ2.sum(axis=1, keepdims=True),
                 "W1": dZ1 @ X.T, "b1": dZ1.sum(axis=1, keepdims=True)}
        for k in p:
            p[k] -= alpha * grads[k]
    return p, history


def train_logistic(X, Y, alpha, iters):
    """Logistic Regression = बिना hidden layer का network: a = sigmoid(w x + b)."""
    n, m = X.shape
    w, b = np.zeros((1, n)), 0.0
    for _ in range(iters):
        dZ = (sigmoid(w @ X + b) - Y) / m
        w, b = w - alpha * dZ @ X.T, b - alpha * dZ.sum()
    return w, b


# ---------- 4. XOR ----------
w, b = train_logistic(X_xor, Y_xor, alpha=1.0, iters=5000)
p_xor, _ = train_network(X_xor, Y_xor, n_hidden=2, alpha=1.0, iters=5000, seed=2)
H_xor, out_xor = forward(X_xor, p_xor)
print("XOR: Logistic Regression का output :", np.round(sigmoid(w @ X_xor + b), 2).ravel())
print("XOR: 2 hidden units वाला network   :", np.round(out_xor, 2).ravel())
print("\nXOR में hidden layer ने हर point को कहाँ भेजा:")
print(" x1  x2   y     a1[1]   a1[2]   output")
for i in range(4):
    print(f"  {X_xor[0, i]:.0f}   {X_xor[1, i]:.0f}   {Y_xor[0, i]:.0f}   {H_xor[0, i]:7.2f} {H_xor[1, i]:7.2f} "
          f"{out_xor[0, i]:8.2f}")

# हर random शुरुआत सफल नहीं होती: cost non-convex है (अगला lecture)। ज़्यादा units हों तो रास्ता आसान होता है।
print("\n20 अलग random शुरुआतों में से कितनी बार XOR हल हुआ (loss < 0.05):")
for n_hidden in (2, 3, 8):
    solved = sum(train_network(X_xor, Y_xor, n_hidden, 1.0, 5000, seed)[1][-1] < 0.05 for seed in range(20))
    print(f"  hidden units = {n_hidden}: {solved} / 20")

# ---------- 5. Two moons: hidden units बढ़ाकर देखें ----------
w, b = train_logistic(X_train, Y_train, alpha=0.5, iters=3000)
print("\nTwo moons (500 train, 500 test)")
print("Model                  parameters   train acc   test acc")
print(f"{'Logistic Regression':<22} {3:10d} {accuracy(sigmoid(w @ X_train + b), Y_train):11.3f} "
      f"{accuracy(sigmoid(w @ X_test + b), Y_test):10.3f}")
nets = {}
for n_hidden in (1, 2, 3, 8):
    nets[n_hidden] = train_network(X_train, Y_train, n_hidden, alpha=0.5, iters=6000, seed=0)
    p = nets[n_hidden][0]
    n_params = sum(v.size for v in p.values())         # (n1*2 + n1) + (1*n1 + 1)
    print(f"{'NN, hidden units = ' + str(n_hidden):<22} {n_params:10d} "
          f"{accuracy(forward(X_train, p)[1], Y_train):11.3f} {accuracy(forward(X_test, p)[1], Y_test):10.3f}")
p_moons, history = nets[8]

# ---------- 6. Vectorization क्यों: वही forward pass, loop से और matrix से ----------
big = np.tile(X_train, (1, 60))                         # 30,000 examples
t0 = time.perf_counter()
loop_out = [forward(big[:, [i]], p_moons)[1][0, 0] for i in range(big.shape[1])]
t1 = time.perf_counter()
vec_out = forward(big, p_moons)[1]
t2 = time.perf_counter()
print(f"\n30,000 examples का forward pass: loop {t1 - t0:.2f} s, vectorized {t2 - t1:.4f} s "
      f"(लगभग {(t1 - t0) / (t2 - t1):.0f} गुना तेज़)")
print("दोनों के जवाबों में सबसे बड़ा फ़र्क:", f"{np.max(np.abs(np.array(loop_out) - vec_out)):.1e}")

# ---------- 7. Plots ----------
g1, g2 = np.meshgrid(np.linspace(-1.8, 2.8, 200), np.linspace(-1.4, 1.9, 200))
G = np.vstack([g1.ravel(), g2.ravel()])
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
surfaces = [("Logistic Regression", sigmoid(w @ G + b)), ("2-layer network, 8 tanh units", forward(G, p_moons)[1])]
for ax, (title, prob) in zip(axes, surfaces):
    ax.contourf(g1, g2, prob.reshape(g1.shape), levels=20, cmap="coolwarm", alpha=0.6)
    ax.contour(g1, g2, prob.reshape(g1.shape), levels=[0.5], colors="k")
    ax.scatter(X_test[0], X_test[1], c=Y_test.ravel(), cmap="coolwarm", s=14, edgecolor="k", linewidth=0.3)
    ax.set(title=title + " (test points)", xlabel="x1", ylabel="x2")
Z1 = p_moons["W1"] @ G + p_moons["b1"]
for j in range(Z1.shape[0]):                            # हर hidden unit की अपनी सीधी line: z1_j = 0
    axes[1].contour(g1, g2, Z1[j].reshape(g1.shape), levels=[0], colors="gray", linewidths=0.6, linestyles="--")
axes[2].scatter(H_xor[0], H_xor[1], c=Y_xor.ravel(), cmap="coolwarm", s=160, edgecolor="k")
for i in range(4):
    axes[2].annotate(f"x=({X_xor[0, i]:.0f},{X_xor[1, i]:.0f})", (H_xor[0, i], H_xor[1, i]),
                     textcoords="offset points", xytext=(8, 8))
h1 = np.linspace(-1.2, 1.2, 50)
W2, b2 = p_xor["W2"].ravel(), p_xor["b2"].item()
axes[2].plot(h1, -(W2[0] * h1 + b2) / W2[1], "k-", label="output layer boundary")
axes[2].set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xlabel="hidden unit 1", ylabel="hidden unit 2",
            title="XOR seen from the hidden layer")
axes[2].legend(loc="lower left")
fig.tight_layout()
fig.savefig("result.png", dpi=130)
