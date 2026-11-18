"""Lecture 10: Neural Networks: Basics: 3D animations."""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, animation, ease, grid, lerp


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z):
    return np.maximum(0.0, z)


def _train(X, Y, n_h, alpha, iters, seed, keep=()):
    """Tiny 2-layer network (tanh hidden units, sigmoid output) trained with batch GD.

    X is (n, m) with one example per column, Y is (1, m). Returns the final
    parameters and a dict of parameter snapshots at the iterations listed in ``keep``.
    """
    rng = np.random.default_rng(seed)
    n, m = X.shape
    W1, b1 = rng.normal(0, 1.0, (n_h, n)), np.zeros((n_h, 1))
    W2, b2 = rng.normal(0, 1.0, (1, n_h)), np.zeros((1, 1))
    keep, snaps = set(keep), {}
    for it in range(iters + 1):
        A1 = np.tanh(W1 @ X + b1)
        A2 = _sig(W2 @ A1 + b2)
        if it in keep:
            snaps[it] = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        dZ2 = (A2 - Y) / m
        dZ1 = (W2.T @ dZ2) * (1 - A1**2)
        W2 -= alpha * dZ2 @ A1.T
        b2 -= alpha * dZ2.sum(axis=1, keepdims=True)
        W1 -= alpha * dZ1 @ X.T
        b1 -= alpha * dZ1.sum(axis=1, keepdims=True)
    return (W1, b1, W2, b2), snaps


def _line_in_square(w, b, lim, k=60):
    """Points of the line w.x + b = 0 that fall inside the square [-lim, lim]^2."""
    w = np.asarray(w, float)
    p0 = -b * w / (w @ w)
    d = np.array([-w[1], w[0]]) / np.linalg.norm(w)
    t = np.linspace(-3 * lim, 3 * lim, k)
    pts = p0 + t[:, None] * d
    pts[np.any(np.abs(pts) > lim, axis=1)] = np.nan
    return pts


def _plane_in_box(w, b, lo, hi):
    """Polygon (ordered vertices) where the plane w.p + b = 0 cuts the cube [lo, hi]^3."""
    corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)], float)
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:              # i-j is an edge of the cube
                fi, fj = corners[i] @ w + b, corners[j] @ w + b
                if fi * fj < 0:
                    pts.append(corners[i] + fi / (fi - fj) * (corners[j] - corners[i]))
    pts = np.array(pts)
    c = pts.mean(axis=0)
    u = np.cross(w, [1.0, 0.31, 0.17])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return pts[np.argsort(np.arctan2((pts - c) @ v, (pts - c) @ u))]


# ------------------------------------------------------------------ 1. one neuron

@animation("One neuron: changing w and b",
           caption=r"""ज़मीन पर दो inputs $x_1, x_2$ हैं और ऊँचाई एक sigmoid neuron का output
           $a = g(w_1 x_1 + w_2 x_2 + b)$ है। काला तीर $w$ है और काली रेखा वह boundary है जहाँ $a = 0.5$ है।
           पहले $w$ का size बढ़ता है: ढलान तीखी होकर सीढ़ी जैसी बन जाती है। फिर $w$ की दिशा घूमती है: पूरी
           surface उसी के साथ घूम जाती है, क्योंकि output हमेशा $w$ की दिशा में बढ़ता है। अंत में $b$ बदलता है:
           boundary बिना घूमे आगे-पीछे खिसकती है। एक neuron बस इतना ही कर सकता है: एक सीधी boundary।""",
           frames=45, keys=(0.0, 0.5, 1.0))
def one_neuron(s, n):
    lim = 3.0
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$a$"), elev=30, azim=-70,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(0, 1))
    X, Y = grid((-lim, lim), (-lim, lim), 30)
    third = n // 3

    def params(i):
        if i < third:                                   # phase 1: |w| grows
            t = ease(i / (third - 1))
            return "sharpen", lerp(0.35, 3.0, t), np.deg2rad(20.0), 0.0
        if i < 2 * third:                               # phase 2: direction of w turns
            t = ease((i - third) / (third - 1))
            return "turn", 3.0, np.deg2rad(lerp(20.0, 110.0, t)), 0.0
        t = (i - 2 * third) / (n - 2 * third - 1)       # phase 3: b slides the boundary
        return "shift", 3.0, np.deg2rad(110.0), 4.5 * np.sin(1.5 * np.pi * t)

    def update(i):
        s.clear()
        name, size, ang, b = params(i)
        w = size * np.array([np.cos(ang), np.sin(ang)])
        A = _sig(w[0] * X + w[1] * Y + b)
        s.add(ax.plot_surface(X, Y, A, cmap="coolwarm", alpha=0.6, linewidth=0, rstride=1, cstride=1,
                              vmin=0, vmax=1))
        line = _line_in_square(w, b, lim)
        s.add(ax.plot(line[:, 0], line[:, 1], 0 * line[:, 0], color=C["dark"], lw=1.2, ls="--"))
        s.add(ax.plot(line[:, 0], line[:, 1], 0 * line[:, 0] + 0.5, color=C["dark"], lw=2.0))
        tip = (0.8 + 0.45 * size) * w / size
        s.add(ax.quiver(0, 0, 0, tip[0], tip[1], 0, color=C["dark"], linewidth=2.0,
                        arrow_length_ratio=0.25))
        s.status(f"{name}:  w = ({w[0]:.1f}, {w[1]:.1f})   b = {b:.1f}")
        s.spin(i, n, sweep=30)

    return update


# ------------------------------------------------------------------ 2. activations

@animation("Sigmoid, tanh, ReLU on one plane",
           caption=r"""स्लेटी जाली pre-activation $z = w^T x + b$ है: $x_1, x_2$ के ऊपर एक सपाट plane। रंगीन surface
           activation $a = g(z)$ है। शुरू में $g(z) = z$ है, इसलिए दोनों एक ही हैं। फिर वही plane बारी-बारी से
           तीन functions से गुज़रता है: **sigmoid** उसे $0$ और $1$ के बीच दबा देता है, **tanh** उसे $-1$ और $1$
           के बीच रखता है और zero के दोनों तरफ़ बराबर फैलाता है, और **ReLU** negative हिस्से को काटकर सपाट कर देता
           है पर positive हिस्से को छूता भी नहीं। ध्यान दें कि sigmoid और tanh दोनों सिरों पर सपाट हो जाते हैं
           (saturation), जबकि ReLU की ढलान positive तरफ़ बनी रहती है।""",
           frames=44, keys=(0.3, 0.58, 1.0))
def activations(s, n):
    lim = 2.0
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$a = g(z)$"), elev=13, azim=-74,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-3, 3))
    ax.set_box_aspect((1, 1, 1.05), zoom=0.84)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    X, Y = grid((-lim, lim), (-lim, lim), 28)

    def pre(a, b):
        return 1.25 * a + 0.25 * b                       # the pre-activation plane, between -3 and 3

    Z = pre(X, Y)
    ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color=C["gray"], linewidth=0.5, alpha=0.55)
    xs = np.linspace(-lim, lim, 80)
    for level in (-1.0, 0.0, 1.0):                       # reference heights on the front wall
        ax.plot(xs, 0 * xs - lim, 0 * xs + level, color=C["gray"], lw=0.7, ls=":")
    funcs = [("identity: a = z", lambda z: z), ("sigmoid: a in (0, 1)", _sig),
             ("tanh: a in (-1, 1)", np.tanh), ("ReLU: a = max(0, z)", _relu)]
    shapes = [(name, f(Z)) for name, f in funcs]
    edges = [f(pre(xs, -lim)) for _, f in funcs]         # profile of each shape along the front edge
    # timeline in frames: hold 4, then three times (morph 8, hold 4), plus a longer final hold
    marks = [0, 4, 12, 16, 24, 28, 36]

    def update(i):
        s.clear()
        k = min(2, max(0, (i - 4) // 12))                # which morph we are in or have finished
        t = ease((i - marks[1 + 2 * k]) / 8.0) if i >= 4 else 0.0
        A = lerp(shapes[k][1], shapes[k + 1][1], t)
        s.add(ax.plot_surface(X, Y, A, cmap="coolwarm", alpha=0.7, linewidth=0, rstride=1, cstride=1,
                              vmin=-1.3, vmax=1.3))
        s.add(ax.plot(xs, 0 * xs - lim, lerp(edges[k], edges[k + 1], t), color=C["dark"], lw=2.2))
        if t <= 0.0:
            s.status(shapes[k][0])
        elif t >= 1.0:
            s.status(shapes[k + 1][0])
        else:
            s.status(f"{shapes[k][0].split(':')[0]}  ->  {shapes[k + 1][0].split(':')[0]}")
        s.spin(i, n, sweep=30)

    return update


# ------------------------------------------------------------------ 3. forward pass

@animation("Forward pass, layer by layer",
           caption=r"""यहाँ एक छोटा network 3D में रखा है: एक axis पर layer का number ($0$ यानी input $x$), दूसरे पर
           उस layer के units, और ऊँचाई हर unit का activation $a^{[l]}_j$ है। फ़र्श की रेखाएँ weights हैं: नीली
           positive, लाल negative, और मोटी रेखा मतलब बड़ा weight। Forward propagation बाएँ से दाएँ चलता है: पहले
           पिछली layer के activations weights के रास्ते आगे जाते हैं ($z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$), फिर
           activation function लगता है और उस layer के खंभे उठते हैं। जो ReLU unit zero पर रह गया, उसका
           $z$ negative था। कोई layer तब तक नहीं निकल सकती जब तक उससे पहले वाली पूरी न हो जाए।""",
           frames=42, keys=(0.3, 0.62, 1.0))
def forward_pass(s, n):
    sizes = [3, 5, 4, 1]
    rng = np.random.default_rng(1388)                   # a seed where most units are active
    Ws = [rng.normal(0, 1.0, (sizes[l + 1], sizes[l])) for l in range(3)]
    bs = [rng.normal(0.1, 0.2, sizes[l + 1]) for l in range(3)]
    acts = [np.array([0.9, 0.35, 0.65])]
    for l in range(3):
        z = Ws[l] @ acts[-1] + bs[l]
        acts.append(_relu(z) if l < 2 else _sig(z))
    top = max(a.max() for a in acts) * 1.15
    ys = [np.arange(k) - (k - 1) / 2 for k in sizes]
    ax = s.ax3d(labels=("layer $l$", "unit $j$", "activation"), elev=30, azim=-68,
                xlim=(-0.2, 3.2), ylim=(-2.3, 2.3), zlim=(0, top))
    ax.set_box_aspect((1.5, 1.25, 0.8), zoom=0.95)
    ax.set_xticks(range(4))
    ax.set_yticks([])
    wmax = max(np.abs(W).max() for W in Ws)
    colors = [C["dark"], C["teal"], C["purple"], C["orange"]]
    edges = []                                           # per layer: list of (y_from, y_to, weight)
    for l in range(3):
        edges.append([(ys[l][k], ys[l + 1][j], Ws[l][j, k])
                      for j in range(sizes[l + 1]) for k in range(sizes[l])])
        for y0, y1, w in edges[l]:
            ax.plot([l, l + 1], [y0, y1], [0, 0], color=C["blue"] if w > 0 else C["red"],
                    lw=0.3 + 1.8 * abs(w) / wmax, alpha=0.3)
    for l in range(4):
        ax.scatter(np.full(sizes[l], l), ys[l], 0, color=colors[l], s=10, depthshade=False)
    names = ["input x", "ReLU", "ReLU", "sigmoid"]
    per = 11                                             # frames per layer: 5 travel + 6 rise

    def update(i):
        s.clear()
        stage, f = (0, 0) if i < 4 else (min(3, 1 + (i - 4) // per), (i - 4) % per)
        if i >= 4 + 3 * per:
            stage, f = 3, per
        for l in range(stage + 1):
            if l == 0:
                h = acts[0] * (ease((i + 1) / 4.0) if stage == 0 else 1.0)
            else:
                h = acts[l] * (1.0 if l < stage else ease((f - 4) / 6.0))
            nan = np.full(sizes[l], np.nan)
            s.add(ax.plot(np.c_[np.full(sizes[l], l), np.full(sizes[l], l), nan].ravel(),
                          np.c_[ys[l], ys[l], nan].ravel(), np.c_[0 * h, h, nan].ravel(),
                          color=colors[l], lw=3.0))
            s.add(ax.scatter(np.full(sizes[l], l), ys[l], h, color=colors[l], s=42,
                             depthshade=False, edgecolor="white"))
        if stage == 0:
            s.status(r"layer 0:  $a^{[0]} = x$")
        elif f < 5 and i < 4 + 3 * per:                  # signals travel along the weights
            t = (f + 1) / 5.0
            l = stage - 1
            for y0, y1, w in edges[l]:
                s.add(ax.plot([l, l + 1], [y0, y1], [0, 0], color=C["blue"] if w > 0 else C["red"],
                              lw=0.3 + 1.8 * abs(w) / wmax, alpha=0.85))
            e = np.array(edges[l])
            s.add(ax.scatter(np.full(len(e), l + t), lerp(e[:, 0], e[:, 1], t), 0 * e[:, 0],
                             color=C["yellow"], s=9, depthshade=False))
            s.status(rf"layer {stage}:  $z^{{[{stage}]}} = W^{{[{stage}]}} a^{{[{l}]}} + b^{{[{stage}]}}$")
        else:
            s.status(rf"layer {stage}:  $a^{{[{stage}]}}$ = {names[stage]}$(z^{{[{stage}]}})$")
        if i >= 4 + 3 * per:
            s.status(rf"output:  $h(x) = a^{{[3]}}$ = sigmoid$(z^{{[3]}})$ = {acts[3][0]:.2f}")
        s.spin(i, n, sweep=28)

    return update


# ------------------------------------------------------------------ 4. XOR in hidden space

@animation("XOR: bent until a plane splits it",
           caption=r"""चार XOR points पहले input वाली सपाट चादर पर हैं: लाल points ($y = 1$) एक diagonal पर और नीले
           ($y = 0$) दूसरे पर, इसलिए कोई सीधी रेखा इन्हें अलग नहीं कर सकती। फिर तीन tanh hidden units वाली trained
           layer हर point को नई जगह $a^{[1]} = g(W^{[1]} x + b^{[1]})$ पर भेजती है। तीनों axes अब तीन hidden units
           के activations हैं, और चादर उसी map से मुड़ जाती है। मुड़ने के बाद दोनों लाल points एक तरफ़ और दोनों
           नीले दूसरी तरफ़ आ जाते हैं, और output layer का एक ही हरा plane $W^{[2]} a^{[1]} + b^{[2]} = 0$ उन्हें
           अलग कर देता है। Hidden layer boundary को टेढ़ा नहीं करती, वह data को मोड़ती है।""",
           frames=42, keys=(0.0, 0.5, 1.0))
def xor_hidden(s, n):
    X = np.array([[0.0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0.0, 1, 1, 0]])
    (W1, b1, W2, b2), _ = _train(X, Y, 3, 0.8, 500, seed=22)
    # the camera ends up looking almost along the separating plane, so both sides are easy to tell apart
    ax = s.ax3d(labels=(r"$a_1$", r"$a_2$", r"$a_3$"), elev=24, azim=135,
                xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), ticks=False)
    G1, G2 = grid((-0.25, 1.25), (-0.25, 1.25), 15)

    def embed(x1, x2, t):
        """Blend from the flat input sheet (x1, x2, 0) to the hidden activations."""
        flat = np.stack([x1, x2, 0 * x1])
        hid = np.tanh(np.einsum("jk,k...->j...", W1, np.stack([x1, x2])) +
                      b1.reshape(3, *([1] * x1.ndim)))
        return lerp(flat, hid, t)

    poly = _plane_in_box(W2.ravel(), b2.item(), -1.0, 1.0)
    cols = [C["red"] if y else C["blue"] for y in Y.ravel()]
    ax.scatter([], [], [], color=C["red"], s=40, label="y = 1")
    ax.scatter([], [], [], color=C["blue"], s=40, label="y = 0")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    bend_end, plane_start = int(0.6 * n), int(0.66 * n)

    def update(i):
        s.clear()
        t = ease(max(0, i - 3) / (bend_end - 3))
        S = embed(G1, G2, t)
        s.add(ax.plot_wireframe(S[0], S[1], S[2], color=C["gray"], linewidth=0.6, alpha=0.7))
        P = embed(X[0], X[1], t)
        s.add(ax.scatter(P[0], P[1], P[2], color=cols, s=70, depthshade=False, edgecolor="white"))
        if i >= plane_start:
            a = 0.4 * ease((i - plane_start) / 6.0)
            s.add(ax.add_collection3d(Poly3DCollection([poly], facecolor=C["green"], alpha=a,
                                                       edgecolor=C["green"], linewidth=0.8)))
            s.status("hidden space: one plane separates XOR")
        elif t <= 0:
            s.status("input space: no line separates XOR")
        else:
            s.status(r"hidden layer bends the sheet:  $a = g(Wx + b)$")
        s.spin(i, n, sweep=40)

    return update


# ------------------------------------------------------------------ 5. universal approximation

@animation("Adding ReLU units one at a time",
           caption=r"""काली जाली वह target function है जिसे सीखना है: $x_1, x_2$ के ऊपर एक पहाड़ी और एक गड्ढा। रंगीन
           surface एक hidden layer वाले ReLU network का output है। हर frame में एक नया hidden unit जुड़ता है।
           हर unit surface में एक सीधा **मोड़** डालता है, जो फ़र्श पर एक नारंगी रेखा के रूप में दिखता है। एक-दो
           units से सिर्फ़ तिरछा plane या एक मोड़ बनता है। जैसे-जैसे units बढ़ते हैं, सपाट टुकड़े छोटे होते जाते
           हैं, surface target से चिपकती जाती है और error गिरता जाता है। यही universal approximation का idea है:
           काफ़ी units हों तो एक hidden layer किसी भी continuous surface के पास पहुँच सकती है।""",
           frames=40, keys=(0.08, 0.3, 1.0))
def relu_units(s, n):
    lim = 2.0

    def target(a, b):
        return (1.3 * np.exp(-((a - 0.7) ** 2 + (b - 0.5) ** 2) / 0.9)
                - 1.0 * np.exp(-((a + 0.9) ** 2 + (b + 0.8) ** 2) / 0.7))

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$h(x)$"), elev=30, azim=-58,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-1.5, 1.6))
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    X, Y = grid((-lim, lim), (-lim, lim), 34)
    T = target(X, Y)
    ax.plot_wireframe(X, Y, T, rstride=3, cstride=3, color=C["dark"], linewidth=0.45, alpha=0.55)
    # a pool of random ReLU units; each step greedily adds the one that removes the most error,
    # then the output weights are refitted by least squares
    rng = np.random.default_rng(2)
    ang = rng.uniform(0, 2 * np.pi, 400)
    Wp = np.c_[np.cos(ang), np.sin(ang)]
    bp = rng.uniform(-2.2, 2.2, 400)
    F = _relu(X.ravel()[:, None] * Wp[:, 0] + Y.ravel()[:, None] * Wp[:, 1] + bp)
    Fn = F / (np.linalg.norm(F, axis=0) + 1e-9)
    y = T.ravel()
    kmax = 34
    chosen, fits = [], [np.full_like(y, y.mean())]
    for _ in range(kmax):
        j = int(np.argmax(np.abs(Fn.T @ (y - fits[-1]))))
        chosen.append(j)
        B = np.c_[np.ones_like(y), F[:, chosen]]
        fits.append(B @ np.linalg.lstsq(B, y, rcond=None)[0])
    lines = [_line_in_square(Wp[j], bp[j], lim) for j in chosen]

    def update(i):
        s.clear()
        k = min(kmax, i)
        A = np.clip(fits[k].reshape(X.shape), -1.5, 1.6)
        s.add(ax.plot_surface(X, Y, A, cmap="viridis", alpha=0.6, linewidth=0, rstride=1, cstride=1,
                              vmin=-1.1, vmax=1.3))
        for line in lines[:k]:
            s.add(ax.plot(line[:, 0], line[:, 1], 0 * line[:, 0] - 1.5, color=C["orange"], lw=0.7,
                          alpha=0.8))
        rms = np.sqrt(np.mean((fits[k] - y) ** 2))
        s.status(f"hidden ReLU units: {k}    RMS error = {rms:.3f}")
        s.spin(i, n, sweep=40)

    return update


# ------------------------------------------------------------------ 6. training on circles

@animation("A 2-layer network learns circles",
           caption=r"""ज़मीन पर features $x_1, x_2$ हैं। लाल points ($y = 1$) बीच में हैं और ऊँचाई $1$ पर रखे हैं, नीले
           points ($y = 0$) उन्हें घेरे हुए हैं और ऊँचाई $0$ पर हैं। Surface network का output $a^{[2]}$ है, यानी
           $P(y = 1 \mid x)$ का अंदाज़ा। शुरू में weights random हैं, इसलिए surface का data से कोई मेल नहीं है। Training के साथ hidden units की boundaries
           अपनी जगह लेती हैं और बीच में एक टीला उठता है, जो लाल points तक पहुँचता है और नीलों के ऊपर नीचे रहता
           है। जहाँ surface $0.5$ को काटती है (काली रेखा), वही decision boundary है: एक बंद घेरा, जो कोई अकेला
           neuron नहीं बना सकता। Weights कैसे बदले गए, यह अगले lecture का backpropagation बताएगा।""",
           frames=42, keys=(0.0, 0.45, 1.0))
def learn_circles(s, n):
    rng = np.random.default_rng(5)
    m = 70
    r = np.r_[rng.uniform(0, 0.75, m // 2), rng.uniform(1.35, 2.0, m // 2)]
    th = rng.uniform(0, 2 * np.pi, m)
    X = np.vstack([r * np.cos(th), r * np.sin(th)])
    Y = (np.arange(m) < m // 2).astype(float)[None, :]
    total = 3000
    its = sorted({int(round(total * (i / (n - 1)) ** 2.6)) for i in range(n)})
    _, snaps = _train(X, Y, 6, 0.6, total, seed=11, keep=its)
    lim = 2.2
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$a^{[2]}$"), elev=30, azim=-58,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(0, 1))
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    G1, G2 = grid((-lim, lim), (-lim, lim), 34)
    G = np.vstack([G1.ravel(), G2.ravel()])
    pos = Y.ravel() == 1
    ax.scatter(X[0, pos], X[1, pos], 1.0, color=C["red"], s=14, depthshade=False, label="y = 1")
    ax.scatter(X[0, ~pos], X[1, ~pos], 0.0, color=C["blue"], s=14, depthshade=False, label="y = 0")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def out(p, D):
        W1, b1, W2, b2 = p
        return _sig(W2 @ np.tanh(W1 @ D + b1) + b2)

    def update(i):
        s.clear()
        it = int(round(total * (i / (n - 1)) ** 2.6))
        p = snaps[it]
        A = out(p, G).reshape(G1.shape)
        s.add(ax.plot_surface(G1, G2, A, cmap="coolwarm", alpha=0.5, linewidth=0, rstride=1, cstride=1,
                              vmin=0, vmax=1))
        if A.min() < 0.5 < A.max():
            cs = ax.contour(G1, G2, A, levels=[0.5], colors=[C["dark"]], linewidths=1.6)
            s.add(cs)
        a = out(p, X)
        loss = -np.mean(Y * np.log(a + 1e-12) + (1 - Y) * np.log(1 - a + 1e-12))
        s.status(f"iteration {it}    loss = {loss:.3f}    acc = {np.mean((a > 0.5) == Y):.2f}")
        s.spin(i, n, sweep=40)

    return update
