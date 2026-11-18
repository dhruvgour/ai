"""Lecture 11: Neural Networks: Training: 3D animations."""
import numpy as np

from lib.anim import C, animation, ease, grid, lerp, surface


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _relu(z):
    return np.maximum(0.0, z)


# ------------------------------------------------------------------ 1. ravine optimizers

def _ravine_J(w1, w2):
    w1, w2 = np.asarray(w1, float), np.asarray(w2, float)
    u = w2 - 0.18 * w1**2 + 0.35
    return 0.14 * (w1 - 1.7) ** 2 + 2.35 * u**2


def _ravine_grad(w):
    w1, w2 = float(w[0]), float(w[1])
    u = w2 - 0.18 * w1**2 + 0.35
    g1 = 0.28 * (w1 - 1.7) + 4.70 * u * (-0.36 * w1)
    g2 = 4.70 * u
    return np.array([g1, g2])


@animation("SGD vs Momentum vs Adam in a ravine",
           caption=r"""संकरी और मुड़ी हुई घाटी (ravine) में खड़ी दीवारों की ढलान बहुत तेज़ है, जबकि तली की ढलान धीमी है।
           **लाल** गेंद (सादा SGD) हर कदम पर दीवारों के बीच zig-zag करती है और तली में धीरे रेंगती है।
           **नीली** गेंद (Momentum) पिछले gradients का औसत रखकर दीवारों के उछाल को दबा देती है और घाटी के
           घुमाव के साथ तेज़ बहती है। **हरी** गेंद (Adam) हर दिशा के step size को अपने आप संतुलित करके
           सबसे सीधे और जल्दी minimum (हरा तारा) तक पहुँच जाती है।""",
           frames=44, keys=(0.12, 0.45, 1.0))
def ravine_optimizers(s, n):
    xlim, ylim, zmax = (-2.2, 2.3), (-1.3, 1.7), 8.5
    ax = s.ax3d(labels=(r"$w_1$", r"$w_2$", r"$J(w)$"), elev=33, azim=-62,
                xlim=xlim, ylim=ylim, zlim=(0, zmax))
    _, X, Y, Z = surface(ax, lambda a, b: np.minimum(_ravine_J(a, b), zmax),
                         xlim, ylim, n=36, cmap="viridis", alpha=0.38)
    ax.contour(X, Y, Z, levels=np.geomspace(0.08, 7.0, 10), zdir="z", offset=0,
               cmap="viridis", linewidths=0.7, alpha=0.75)
    goal = (1.7, 0.18 * 1.7**2 - 0.35)
    ax.scatter(goal[0], goal[1], 0.0, marker="*", s=120, color=C["green"], depthshade=False)

    start = np.array([-1.85, 1.30])
    steps = n - 1
    # 1) Plain SGD
    w = start.copy()
    p_sgd = [w.copy()]
    for _ in range(steps):
        w = w - 0.35 * _ravine_grad(w)
        p_sgd.append(w.copy())
    # 2) Momentum
    w, v = start.copy(), np.zeros(2)
    p_mom = [w.copy()]
    for _ in range(steps):
        v = 0.84 * v + 0.16 * _ravine_grad(w)
        w = w - 0.95 * v
        p_mom.append(w.copy())
    # 3) Adam with bias correction
    w, m1, m2 = start.copy(), np.zeros(2), np.zeros(2)
    p_adam = [w.copy()]
    for t in range(1, steps + 1):
        g = _ravine_grad(w)
        m1 = 0.85 * m1 + 0.15 * g
        m2 = 0.99 * m2 + 0.01 * (g * g)
        mh = m1 / (1.0 - 0.85**t)
        vh = m2 / (1.0 - 0.99**t)
        w = w - 0.20 * mh / (np.sqrt(vh) + 1e-8)
        p_adam.append(w.copy())

    runs = [(np.array(p_sgd), C["red"], "SGD"),
            (np.array(p_mom), C["blue"], "Momentum"),
            (np.array(p_adam), C["green"], "Adam")]
    for _, col, name in runs:
        ax.plot([], [], [], color=col, lw=2, label=name)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        vals = []
        for path, col, _ in runs:
            q = path[: i + 1]
            zz = _ravine_J(q[:, 0], q[:, 1])
            vals.append(zz[-1])
            s.add(ax.plot(q[:, 0], q[:, 1], zz + 0.08, color=col, lw=1.7))
            s.add(ax.plot(q[:, 0], q[:, 1], 0 * zz, color=col, lw=0.9, ls="--", alpha=0.7))
            s.add(ax.scatter(q[-1:, 0], q[-1:, 1], zz[-1:] + 0.08, color=col, s=48,
                             depthshade=False, edgecolor="white"))
        s.status(f"step {i}  J: SGD {vals[0]:.2f} | Mom {vals[1]:.2f} | Adam {vals[2]:.2f}")
        s.spin(i, n, sweep=42)

    return update


# ------------------------------------------------------------------ 2. non-convex landscape

_WELLS = [
    (1.15, 0.95, -2.15, 0.72),    # deep global minimum
    (-1.20, 0.90, -1.30, 0.60),   # higher local minimum A
    (-0.15, -1.25, -1.15, 0.62),  # higher local minimum B
    (0.0, 0.0, 0.55, 0.50),       # central bump creating saddle ridges
]


def _nc_J(w1, w2):
    w1, w2 = np.asarray(w1, float), np.asarray(w2, float)
    out = 0.22 * (w1**2 + w2**2) + 2.35
    for cx, cy, amp, scale in _WELLS:
        out = out + amp * np.exp(-((w1 - cx) ** 2 + (w2 - cy) ** 2) / scale)
    return out


def _nc_grad(w):
    w1, w2 = float(w[0]), float(w[1])
    g1, g2 = 0.44 * w1, 0.44 * w2
    for cx, cy, amp, scale in _WELLS:
        e = amp * np.exp(-((w1 - cx) ** 2 + (w2 - cy) ** 2) / scale)
        g1 += e * (-2.0 * (w1 - cx) / scale)
        g2 += e * (-2.0 * (w2 - cy) / scale)
    return np.array([g1, g2])


@animation("Non-convex loss: start sets minimum",
           caption=r"""Neural network का cost $J(w)$ एक अकेला convex कटोरा नहीं होता; इसमें कई गहरे-उथले गड्ढे
           (local minima) और बीच की सपाट धारें (saddle regions) होती हैं। यहाँ चार अलग-अलग शुरुआती बिंदुओं से
           Gradient Descent चलाया गया है। हरी और बैंगनी गेंदें गहरे global minimum में उतर जाती हैं, जबकि
           नारंगी और लाल गेंदें पास के उथले local minima में फँसकर रुक जाती हैं। शुरुआत कहाँ हुई, इससे अंतिम
           नतीजा बदल सकता है।""",
           frames=42, keys=(0.05, 0.45, 1.0))
def nonconvex_starts(s, n):
    lim = 2.3
    ax = s.ax3d(labels=(r"$w_1$", r"$w_2$", r"$J(w)$"), elev=35, azim=-55,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(0.4, 4.2))
    _, X, Y, Z = surface(ax, _nc_J, (-lim, lim), (-lim, lim), n=36, cmap="coolwarm", alpha=0.42)
    ax.contour(X, Y, Z, levels=np.linspace(0.75, 3.6, 11), zdir="z", offset=0.4,
               cmap="coolwarm", linewidths=0.7, alpha=0.75)

    starts = [
        (np.array([1.95, 1.85]), C["green"], "start 1 (global)"),
        (np.array([0.32, 0.08]), C["purple"], "start 2 (saddle)"),
        (np.array([-1.95, 1.80]), C["orange"], "start 3 (local A)"),
        (np.array([-1.35, -1.95]), C["red"], "start 4 (local B)"),
    ]
    paths = []
    for w0, col, label in starts:
        w = w0.copy()
        tr = [w.copy()]
        for _ in range(n - 1):
            w = w - 0.14 * _nc_grad(w)
            tr.append(w.copy())
        paths.append((np.array(tr), col))
        ax.plot([], [], [], color=col, lw=2, label=label)
    gbest = paths[0][0][-1]
    ax.scatter(gbest[0], gbest[1], 0.42, marker="*", s=115, color=C["green"], depthshade=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=8.3)

    def update(i):
        s.clear()
        js = []
        for path, col in paths:
            q = path[: i + 1]
            zz = _nc_J(q[:, 0], q[:, 1])
            js.append(zz[-1])
            s.add(ax.plot(q[:, 0], q[:, 1], zz + 0.04, color=col, lw=1.7))
            s.add(ax.plot(q[:, 0], q[:, 1], np.full_like(zz, 0.41), color=col, lw=0.8, ls="--"))
            s.add(ax.scatter(q[-1:, 0], q[-1:, 1], zz[-1:] + 0.04, color=col, s=46,
                             depthshade=False, edgecolor="white"))
        s.status(f"step {i}   global J={js[0]:.2f}  local J={js[2]:.2f}, {js[3]:.2f}")
        s.spin(i, n, sweep=44)

    return update


# ------------------------------------------------------------------ 3. symmetry problem

_SX = np.linspace(-2.0, 2.0, 41)
_SY = np.exp(-1.4 * _SX**2)                     # symmetric bump that needs two different hidden units


def _sym_J(w1, w2):
    w1, w2 = np.asarray(w1, float), np.asarray(w2, float)
    h = _sig(w1[..., None] * _SX + 1.3) + _sig(w2[..., None] * _SX + 1.3) - 0.82
    return 0.5 * np.mean((h - _SY) ** 2, axis=-1)


def _sym_grad(w):
    s1 = _sig(w[0] * _SX + 1.3)
    s2 = _sig(w[1] * _SX + 1.3)
    err = (s1 + s2 - 0.82) - _SY
    g1 = np.mean(err * s1 * (1.0 - s1) * _SX)
    g2 = np.mean(err * s2 * (1.0 - s2) * _SX)
    return np.array([g1, g2])


@animation("Symmetry trap vs random init",
           caption=r"""दो hidden units के weights $w_1, w_2$ के ऊपर असली neural loss $J(w_1, w_2)$ दिखाया गया है।
           काली टूटी रेखा $w_1 = w_2$ की समरूप धार है, जहाँ दोनों neurons बिल्कुल जुड़वाँ हैं। **लाल** गेंद बराबर
           weights ($w_1 = w_2$) से शुरू होती है: दोनों gradients हर कदम पर बराबर रहते हैं, इसलिए वह इसी धार पर
           फँसी रहती है और कभी गहरी घाटी में नहीं उतर पाती। **हरी** और **नीली** गेंदें छोटे random अंतर से
           शुरू होकर समरूपता तोड़ देती हैं और दोनों अलग काम सीखकर $J \approx 0$ वाले minima तक पहुँच जाती हैं।""",
           frames=42, keys=(0.05, 0.45, 1.0))
def symmetry_trap(s, n):
    lim = 4.2
    ax = s.ax3d(labels=(r"$w_1$", r"$w_2$", r"$J(w_1, w_2)$"), elev=34, azim=-48,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(0, 0.16))
    _, X, Y, Z = surface(ax, _sym_J, (-lim, lim), (-lim, lim), n=36, cmap="viridis", alpha=0.40)
    ax.contour(X, Y, Z, levels=np.linspace(0.008, 0.13, 9), zdir="z", offset=0,
               cmap="viridis", linewidths=0.7, alpha=0.75)
    diag = np.linspace(-lim, lim, 70)
    ax.plot(diag, diag, _sym_J(diag, diag) + 0.002, color=C["dark"], lw=1.3, ls="--",
            label=r"symmetry $w_1 = w_2$")
    ax.plot(diag, diag, 0 * diag, color=C["dark"], lw=0.9, ls=":")

    inits = [
        (np.array([-3.2, -3.2]), C["red"], "equal init (stuck)"),
        (np.array([0.35, -0.28]), C["green"], "random init A"),
        (np.array([-0.28, 0.35]), C["blue"], "random init B"),
    ]
    runs = []
    sub = 5
    for w0, col, label in inits:
        w = w0.copy()
        tr = [w.copy()]
        for _ in range(n - 1):
            for _ in range(sub):
                w = w - 2.6 * _sym_grad(w)
            tr.append(w.copy())
        runs.append((np.array(tr), col))
        ax.plot([], [], [], color=col, lw=2, label=label)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=8.3)

    def update(i):
        s.clear()
        vals = []
        for path, col in runs:
            q = path[: i + 1]
            zz = _sym_J(q[:, 0], q[:, 1])
            vals.append(zz[-1])
            s.add(ax.plot(q[:, 0], q[:, 1], zz + 0.003, color=col, lw=1.9))
            s.add(ax.scatter(q[-1:, 0], q[-1:, 1], zz[-1:] + 0.003, color=col, s=50,
                             depthshade=False, edgecolor="white"))
        s.status(f"step {i}  equal J={vals[0]:.3f} | rand J={vals[1]:.3f}")
        s.spin(i, n, sweep=42)

    return update


# ------------------------------------------------------------------ 4. vanishing gradients 3D bars

def _layer_grad_norms(kind: str, seed: int = 7):
    """Run real forward + backward pass on a 6-layer MLP and return ||dW^[l]||_F per layer."""
    rng = np.random.default_rng(seed)
    sizes = [16, 16, 16, 16, 16, 16, 1]
    L = len(sizes) - 1
    m = 64
    X = rng.normal(0, 1.0, (sizes[0], m))
    Y = (X[0:1] * X[1:2] > 0).astype(float)
    Ws, bs = [], []
    for l in range(L):
        scale = np.sqrt(2.0 / sizes[l]) if (kind == "relu" and l < L - 1) else np.sqrt(1.0 / sizes[l])
        Ws.append(rng.normal(0, scale, (sizes[l + 1], sizes[l])))
        bs.append(np.zeros((sizes[l + 1], 1)))
    As, Zs = [X], []
    for l in range(L):
        Z = Ws[l] @ As[-1] + bs[l]
        Zs.append(Z)
        if l == L - 1:
            As.append(_sig(Z))
        elif kind == "sigmoid":
            As.append(_sig(Z))
        else:
            As.append(_relu(Z))
    dZ = (As[-1] - Y) / m
    norms = [0.0] * L
    for l in range(L - 1, -1, -1):
        dW = dZ @ As[l].T
        norms[l] = float(np.linalg.norm(dW))
        if l > 0:
            da = Ws[l].T @ dZ
            if kind == "sigmoid":
                dZ = da * As[l] * (1.0 - As[l])
            else:
                dZ = da * (Zs[l - 1] > 0)
    norms = np.array(norms)
    return norms / norms[-1]                              # relative to output layer l=6


@animation("Vanishing gradients: Sigmoid vs ReLU",
           caption=r"""6-layer गहरे network में backpropagation आख़िरी layer $l = 6$ से पहली layer $l = 1$ की ओर
           चलता है। खंभों की ऊँचाई हर layer के सापेक्ष gradient आकार $\|\nabla_{W^{[l]}} J\|_F$ को दिखाती है।
           **लाल** पंक्ति (Sigmoid) में हर कदम पर $g'(z) \le 0.25$ से गुणा होने के कारण खंभे तेज़ी से सिकुड़ते
           जाते हैं और $l = 1, 2$ तक लगभग गायब हो जाते हैं। **हरी** पंक्ति (He initialization के साथ ReLU) में
           सक्रिय units पर $g'(z) = 1$ रहता है, इसलिए शुरुआती layers तक भी मज़बूत संकेत पहुँचता है।""",
           frames=42, keys=(0.18, 0.55, 1.0))
def vanishing_bars(s, n):
    g_sig = _layer_grad_norms("sigmoid", seed=12)
    g_relu = _layer_grad_norms("relu", seed=12)
    L = len(g_sig)
    zmax = max(g_sig.max(), g_relu.max()) * 1.15
    ax = s.ax3d(labels=("layer $l$", "", r"rel. $\Vert dW^{[l]}\Vert_F$"), elev=27, azim=-56,
                xlim=(0.4, 6.6), ylim=(-0.6, 1.6), zlim=(0, zmax))
    ax.set_xticks(range(1, L + 1))
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Sigmoid", "ReLU"])

    ax.bar([], [], color=C["red"], label="Sigmoid (Xavier)")
    ax.bar([], [], color=C["green"], label="ReLU (He)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    per = 5                                              # 5 frames per backward hop from l=6 down to l=1

    def update(i):
        s.clear()
        active_k = min(L - 1, i // per)                  # 0 -> layer 6, ..., 5 -> layer 1
        cur_layer = L - active_k
        for k in range(active_k + 1):
            l_idx = L - 1 - k                            # 5 down to 0
            grow = 1.0 if k < active_k else ease(((i % per) + 1) / per)
            if i >= L * per:
                grow = 1.0
            hs = max(0.008, g_sig[l_idx] * grow)
            hr = max(0.008, g_relu[l_idx] * grow)
            x0 = (l_idx + 1) - 0.24
            s.add(ax.bar3d(x0, -0.22, 0.0, 0.48, 0.42, hs, color=C["red"], alpha=0.86, shade=True))
            s.add(ax.bar3d(x0, 0.78, 0.0, 0.48, 0.42, hr, color=C["green"], alpha=0.86, shade=True))
        if i < L * per and cur_layer < L:
            s.add(ax.plot([cur_layer + 1, cur_layer], [0.0, 0.0], [0.01, 0.01], color=C["orange"], lw=2.5))
            s.add(ax.plot([cur_layer + 1, cur_layer], [1.0, 1.0], [0.01, 0.01], color=C["orange"], lw=2.5))
        li = cur_layer - 1
        s.status(f"layer {cur_layer}: Sigmoid={g_sig[li]:.4f} | ReLU={g_relu[li]:.2f}")
        s.spin(i, n, sweep=34)

    return update


# ------------------------------------------------------------------ 5. L2 smoothing overfit surface

def _fit_reg_surfaces():
    """Train a 2-layer tanh MLP across L2 strengths lambda on noisy 2D regression data."""
    rng = np.random.default_rng(21)
    m = 26
    X = rng.uniform(-1.9, 1.9, (2, m))

    def f_true(a, b):
        return 0.85 * np.sin(1.25 * a) * np.cos(1.15 * b)

    Y = f_true(X[0], X[1])[None, :] + rng.normal(0, 0.46, (1, m))
    # add two sharp noisy outliers to provoke clear unregularized spikes
    Y[0, 3] += 0.95
    Y[0, 11] -= 0.90

    G1, G2 = grid((-2.0, 2.0), (-2.0, 2.0), 34)
    G = np.vstack([G1.ravel(), G2.ravel()])
    T_true = f_true(G1, G2)

    n_h = 20
    W1_0 = rng.normal(0, 2.1, (n_h, 2))
    b1_0 = rng.normal(0, 1.4, (n_h, 1))
    W2_0 = rng.normal(0, 0.6, (1, n_h))
    b2_0 = np.zeros((1, 1))

    lams = np.array([0.0, 1e-4, 4e-4, 1.5e-3, 5e-3, 1.5e-2, 4.5e-2, 0.12])
    preds, stats = [], []
    for lam in lams:
        W1, b1, W2, b2 = W1_0.copy(), b1_0.copy(), W2_0.copy(), b2_0.copy()
        p = [W1, b1, W2, b2]
        m1 = [np.zeros_like(u) for u in p]
        m2 = [np.zeros_like(u) for u in p]
        for t in range(1, 951):
            A1 = np.tanh(p[0] @ X + p[1])
            A2 = p[2] @ A1 + p[3]
            dZ2 = (A2 - Y) / m
            dW2 = dZ2 @ A1.T + (lam / m) * p[2]
            db2 = dZ2.sum(axis=1, keepdims=True)
            dZ1 = (p[2].T @ dZ2) * (1.0 - A1**2)
            dW1 = dZ1 @ X.T + (lam / m) * p[0]
            db1 = dZ1.sum(axis=1, keepdims=True)
            for k, g in enumerate((dW1, db1, dW2, db2)):
                m1[k] = 0.9 * m1[k] + 0.1 * g
                m2[k] = 0.999 * m2[k] + 0.001 * (g * g)
                mh = m1[k] / (1.0 - 0.9**t)
                vh = m2[k] / (1.0 - 0.999**t)
                p[k] -= 0.025 * mh / (np.sqrt(vh) + 1e-8)
        tr_pred = p[2] @ np.tanh(p[0] @ X + p[1]) + p[3]
        g_pred = (p[2] @ np.tanh(p[0] @ G + p[1]) + p[3]).reshape(G1.shape)
        tr_mse = float(np.mean((tr_pred - Y) ** 2))
        te_mse = float(np.mean((g_pred - T_true) ** 2))
        preds.append(g_pred)
        stats.append((lam, tr_mse, te_mse))
    return X, Y, G1, G2, T_true, preds, stats


@animation("L2 weight decay smooths overfitting",
           caption=r"""काले बिंदु शोर भरे (noisy) training examples हैं और पतली स्लेटी जाली असली चिकनी function है।
           शुरू में $\lambda = 0$ (बिना regularization) पर 20 hidden units वाला network शोर को भी रट लेता है:
           surface में तीखे पहाड़-गड्ढे बन जाते हैं और test MSE ऊँचा रहता है। जैसे-जैसे $L_2$ weight decay
           $\lambda$ बढ़ता है, बड़े weights सिकुड़ते हैं, तीखे मोड़ पिघलकर चिकने हो जाते हैं, और surface असली
           जाली के करीब आकर test error को घटा देती है।""",
           frames=40, keys=(0.0, 0.5, 1.0))
def l2_smoothing(s, n):
    X, Y, G1, G2, T_true, preds, stats = _fit_reg_surfaces()
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$h(x)$"), elev=28, azim=-58,
                xlim=(-2, 2), ylim=(-2, 2), zlim=(-2.0, 2.0))
    ax.plot_wireframe(G1, G2, T_true, rstride=3, cstride=3, color=C["gray"], linewidth=0.5, alpha=0.55)
    ax.scatter(X[0], X[1], np.clip(Y[0], -1.95, 1.95), color=C["dark"], s=20, depthshade=False)
    K = len(preds) - 1

    def update(i):
        s.clear()
        pos = (i / (n - 1)) * K
        k = min(K - 1, int(pos))
        t = ease(pos - k)
        S = np.clip(lerp(preds[k], preds[k + 1], t), -2.0, 2.0)
        lam = lerp(stats[k][0], stats[k + 1][0], t)
        tr = lerp(stats[k][1], stats[k + 1][1], t)
        te = lerp(stats[k][2], stats[k + 1][2], t)
        s.add(ax.plot_surface(G1, G2, S, cmap="coolwarm", alpha=0.47, linewidth=0,
                              rstride=1, cstride=1, vmin=-1.3, vmax=1.3))
        s.status(rf"$\lambda$={lam:.3f}   train MSE={tr:.2f}   test MSE={te:.2f}")
        s.spin(i, n, sweep=38)

    return update


# ------------------------------------------------------------------ 6. mini-batch size noise

@animation("Mini-batch size B and path noise",
           caption=r"""एक ही dataset ($m = 64$) और एक ही शुरुआती बिंदु से तीन अलग batch sizes के साथ उतरने के
           रास्ते दिखाए गए हैं। **नारंगी** रास्ता ($B = 1$, शुद्ध SGD) हर कदम पर एक अकेले example का शोर झेलता है,
           इसलिए खूब लड़खड़ाता है। **नीला** रास्ता ($B = 8$, mini-batch) में औसत लेने से शोर काफ़ी कम हो जाता
           है, पर वह तेज़ और लचीला बना रहता है। **हरा** रास्ता ($B = 64$, full batch) बिल्कुल चिकना है क्योंकि
           हर कदम पूरे data का सटीक gradient लेता है।""",
           frames=44, keys=(0.12, 0.5, 1.0))
def minibatch_noise(s, n):
    rng = np.random.default_rng(9)
    m = 64
    x1 = rng.normal(0.6, 1.0, m)
    x2 = rng.normal(-0.4, 0.85, m)
    A = np.c_[x1, x2]
    y = 1.4 * x1 + 1.8 * x2 + rng.normal(0, 0.75, m)

    def J(t1, t2):
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        err = t1[..., None] * x1 + t2[..., None] * x2 - y
        return 0.5 * np.mean(err**2, axis=-1)

    xlim, ylim, zmax = (-1.6, 3.2), (-1.4, 3.4), 9.0
    ax = s.ax3d(labels=(r"$\theta_1$", r"$\theta_2$", r"$J(\theta)$"), elev=33, azim=-60,
                xlim=xlim, ylim=ylim, zlim=(0, zmax))
    _, X, Y, Z = surface(ax, lambda a, b: np.minimum(J(a, b), zmax), xlim, ylim,
                         n=36, cmap="viridis", alpha=0.33)
    ax.contour(X, Y, Z, levels=np.linspace(0.35, 7.5, 10), zdir="z", offset=0,
               cmap="viridis", linewidths=0.7, alpha=0.75)
    best = np.linalg.solve(A.T @ A, A.T @ y)
    ax.scatter(best[0], best[1], 0.0, marker="*", s=120, color=C["green"], depthshade=False)

    start = np.array([-1.25, -0.95])
    configs = [(1, 0.16, C["orange"], "B = 1 (SGD)"),
               (8, 0.22, C["blue"], "B = 8 (mini-batch)"),
               (64, 0.26, C["green"], "B = 64 (full batch)")]
    runs = []
    for B, lr, col, label in configs:
        r_loc = np.random.default_rng(100 + B)
        th = start.copy()
        tr = [th.copy()]
        for _ in range(n - 1):
            idx = np.arange(m) if B == m else r_loc.choice(m, size=B, replace=False)
            g = A[idx].T @ (A[idx] @ th - y[idx]) / B
            th = th - lr * g
            tr.append(th.copy())
        runs.append((np.array(tr), col))
        ax.plot([], [], [], color=col, lw=2, label=label)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        vals = []
        for path, col in runs:
            q = path[: i + 1]
            zz = J(q[:, 0], q[:, 1])
            vals.append(zz[-1])
            s.add(ax.plot(q[:, 0], q[:, 1], zz + 0.08, color=col, lw=1.5))
            s.add(ax.plot(q[:, 0], q[:, 1], 0 * zz, color=col, lw=0.8, ls="--", alpha=0.7))
            s.add(ax.scatter(q[-1:, 0], q[-1:, 1], zz[-1:] + 0.08, color=col, s=44,
                             depthshade=False, edgecolor="white"))
        s.status(f"step {i}  J: B=1 {vals[0]:.2f} | B=8 {vals[1]:.2f} | B=64 {vals[2]:.2f}")
        s.spin(i, n, sweep=42)

    return update
