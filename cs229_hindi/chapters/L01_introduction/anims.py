"""Lecture 1: Introduction and basic concepts: 3D animations."""
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, CLASS_COLORS, animation, ease, grid, lerp


def _steps(n, last):
    """n increasing iteration counts from 0 to ``last``, dense at the start (training slows down)."""
    return np.round(np.geomspace(1, last + 1, n) - 1).astype(int)


def _plane_poly(w, b, lo, hi):
    """Corners of the polygon where the plane w.x + b = 0 cuts the cube [lo, hi]^3 (or None)."""
    pts = []
    for a in range(3):                         # the 4 cube edges parallel to axis a
        j, k = [d for d in range(3) if d != a]
        if abs(w[a]) < 1e-9:
            continue
        for vj in (lo, hi):
            for vk in (lo, hi):
                t = -(b + w[j] * vj + w[k] * vk) / w[a]
                if lo <= t <= hi:
                    p = np.empty(3)
                    p[a], p[j], p[k] = t, vj, vk
                    pts.append(p)
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    nrm = w / np.linalg.norm(w)
    u = np.cross(nrm, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(nrm, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    d = pts - pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(d @ v, d @ u))]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- 1. regression

@animation("Regression: learn a surface",
           caption=r"""यह **Supervised Learning (regression)** है: हर काला बिंदु एक training example है, ज़मीन पर features
           $x_1, x_2$ और ऊँचाई पर target $y$। रंगीन surface hypothesis $h_\theta(x)$ है, जो $\theta = 0$ पर
           सपाट शुरू होती है। Training के हर step में surface data की ओर मुड़ती है, लाल रेखाएँ (errors) छोटी
           होती हैं और cost $J(\theta)$ गिरता है। अंत में एक **नया** $x$ (नारंगी) आता है जो training set में
           नहीं था, और model surface की ऊँचाई पढ़कर उसका prediction दे देता है। Training और prediction का फ़र्क
           यही है।""",
           frames=44, keys=(0.0, 0.2, 1.0))
def regression_surface(s, n):
    rng = np.random.default_rng(1)
    m = 50
    x1, x2 = rng.uniform(-2, 2, m), rng.uniform(-2, 2, m)

    def feats(a, b):
        return np.stack([np.ones_like(a), a, b, a * a, b * b, a * b], axis=-1)

    true_theta = np.array([1.0, 0.9, 0.2, 0.0, -0.45, 0.35])
    X = feats(x1, x2)
    y = X @ true_theta + rng.normal(0, 0.35, m)
    zlim = (-5.5, 4.5)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y$"), elev=22, azim=-140,
                xlim=(-2, 2), ylim=(-2, 2), zlim=zlim)
    ax.scatter(x1, x2, y, color=C["dark"], s=12, depthshade=False)

    n_train = n - 12
    alpha = 1.0 / np.linalg.eigvalsh(X.T @ X / m).max()
    ks = _steps(n_train, 400)
    thetas, th = {}, np.zeros(6)
    for k in range(ks[-1] + 1):
        thetas[k] = th.copy()
        th = th - alpha * X.T @ (X @ th - y) / m
    G1, G2 = grid((-2, 2), (-2, 2), 22)
    F = feats(G1, G2)
    q = np.array([1.2, -1.3])                                  # a new x, not in the training set
    nan = np.full(m, np.nan)

    def update(i):
        s.clear()
        k = ks[min(i, n_train - 1)]
        th = thetas[k]
        pred = X @ th
        s.add(ax.plot_surface(G1, G2, np.clip(F @ th, *zlim), cmap="viridis", alpha=0.42, linewidth=0,
                              vmin=-4.5, vmax=3.5))
        s.add(ax.plot(np.c_[x1, x1, nan].ravel(), np.c_[x2, x2, nan].ravel(),
                      np.c_[y, pred, nan].ravel(), color=C["red"], lw=0.8, alpha=0.8))
        J = 0.5 * np.mean((pred - y) ** 2)
        if i < n_train:
            s.status(rf"training step {k}    $J(\theta)$ = {J:.3f}")
        else:
            t = ease((i - n_train + 1) / 7)
            hq = float(feats(q[0], q[1]) @ th)
            top = zlim[0] + t * (hq - zlim[0])
            s.add(ax.plot([q[0], q[0]], [q[1], q[1]], [zlim[0], top], color=C["orange"], lw=2.0, ls="--"))
            s.add(ax.scatter([q[0]], [q[1]], [zlim[0]], color=C["orange"], s=40, marker="s",
                             depthshade=False))
            if t >= 1:
                s.add(ax.scatter([q[0]], [q[1]], [hq], color=C["orange"], s=75, edgecolor="white",
                                 depthshade=False))
            s.status(rf"prediction for a new $x$:  $h_\theta(x)$ = {hq:.2f}")
        s.spin(i, n, sweep=22)

    return update


# ---------------------------------------------------------------- 2. classification

@animation("Classification: a separating plane",
           caption=r"""यह **Supervised Learning (classification)** है: तीनों axes features हैं ($x_1, x_2, x_3$) और label
           रंग से दिखाया गया है (नीला class 1, लाल class 0)। हरा plane model की **decision boundary** है: उसके
           एक तरफ़ model "class 1" कहता है, दूसरी तरफ़ "class 0"। शुरू में plane गलत जगह है, इसलिए बहुत से
           बिंदु गलत तरफ़ हैं (उन पर काला घेरा है)। Training के साथ plane घूमकर दोनों बादलों के बीच आ जाता है
           और accuracy बढ़ती जाती है। बादल थोड़े मिले हुए हैं, इसलिए कुछ बिंदु अंत तक गलत तरफ़ रह जाते हैं:
           असली data में 100% accuracy ज़रूरी नहीं।""",
           frames=40, keys=(0.0, 0.3, 1.0))
def classification_plane(s, n):
    rng = np.random.default_rng(5)
    m, lo, hi = 36, -2.6, 2.6
    P = np.r_[rng.normal([1.0, 0.8, 0.9], 0.75, (m, 3)), rng.normal([-1.0, -0.8, -0.9], 0.75, (m, 3))]
    P = np.clip(P, lo + 0.1, hi - 0.1)
    y = np.r_[np.ones(m), np.zeros(m)]
    X = np.c_[np.ones(2 * m), P]
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=20, azim=-72,
                xlim=(lo, hi), ylim=(lo, hi), zlim=(lo, hi), ticks=False)
    ax.scatter(*P[y == 1].T, color=C["blue"], s=16, depthshade=False, label="class 1")
    ax.scatter(*P[y == 0].T, color=C["red"], s=16, depthshade=False, label="class 0")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    ks = _steps(n - 4, 250)
    thetas, th = {}, np.array([0.4, -0.3, 1.0, -1.0])        # a poor starting plane
    for k in range(ks[-1] + 1):
        thetas[k] = th.copy()
        th = th + 0.25 * X.T @ (y - _sigmoid(X @ th)) / len(y)

    def update(i):
        s.clear()
        k = ks[min(i, len(ks) - 1)]
        th = thetas[k]
        poly = _plane_poly(th[1:], th[0], lo, hi)
        if poly is not None:
            s.add(ax.add_collection3d(Poly3DCollection([poly], facecolor=C["green"], alpha=0.28,
                                                       edgecolor=C["green"], linewidth=1.0)))
        wrong = (X @ th >= 0) != (y == 1)
        if wrong.any():
            s.add(ax.scatter(*P[wrong].T, s=70, facecolors="none", edgecolors=C["dark"], linewidths=1.1,
                             depthshade=False))
        s.status(f"training step {k}    accuracy = {100 * (1 - wrong.mean()):.0f}%")
        s.spin(i, n, sweep=35)

    return update


# ---------------------------------------------------------------- 3. clustering

@animation("Clustering: k-means finds groups",
           caption=r"""यह **Unsupervised Learning** है: तीनों axes features हैं और कोई label नहीं है, इसलिए शुरू में सारे
           बिंदु स्लेटी हैं। तीन तारे **centroids** हैं, जिन्हें जान-बूझकर एक ही कोने से शुरू किया गया है।
           k-means दो काम बारी-बारी से दोहराता है: हर बिंदु को सबसे पास वाले centroid का रंग देना, फिर हर
           centroid को अपने रंग के बिंदुओं के mean पर ले जाना। कुछ ही rounds में तारे तीनों झुंडों के बीच जा
           बैठते हैं और रंग बदलना बंद हो जाता है। किसी ने algorithm को नहीं बताया था कि झुंड कहाँ हैं।""",
           frames=42, keys=(0.0, 0.3, 1.0))
def kmeans_clusters(s, n):
    rng = np.random.default_rng(4)
    centers = np.array([[-1.4, -1.2, -1.0], [1.5, -1.0, 1.1], [0.0, 1.5, 0.2]])
    P = np.vstack([rng.normal(c, 0.42, (40, 3)) for c in centers])
    P = np.clip(P, -2.4, 2.4)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=26, azim=-150,
                xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), zlim=(-2.5, 2.5), ticks=False)

    ax.computed_zorder = False
    mus = [np.array([[2.2, 2.2, 2.2], [2.0, 2.2, 1.8], [1.8, 2.0, 2.2]])]      # deliberately bad start
    assigns = []
    for _ in range(12):
        d = ((P[:, None, :] - mus[-1][None]) ** 2).sum(-1)
        c = d.argmin(1)
        assigns.append(c)
        new = np.array([P[c == j].mean(0) if (c == j).any() else mus[-1][j] for j in range(3)])
        if np.allclose(new, mus[-1]):
            break
        mus.append(new)
    rounds = len(mus) - 1                       # number of rounds in which centroids moved
    per = max(2, (n - 4) // rounds)

    def update(i):
        s.clear()
        if i == 0:
            mu, cols, trail = mus[0], [C["gray"]] * len(P), 0
            s.status("unlabeled data, 3 badly placed centroids")
        else:
            r, f = divmod(i - 1, per)
            if r >= rounds:
                r, f, done = rounds - 1, per - 1, True
            else:
                done = False
            c = assigns[r]
            t = ease(f / (per - 1))
            mu = lerp(mus[r], mus[r + 1], t)
            cols = [CLASS_COLORS[j] for j in c]
            trail = r + 1
            J = ((P - mu[c]) ** 2).sum()
            if done:
                s.status(f"converged after {rounds} rounds    J = {J:.1f}")
            else:
                s.status(f"round {r + 1}: {'assign colours' if f == 0 else 'move centroids'}    J = {J:.1f}")
        s.add(ax.scatter(*P.T, c=cols, s=13, depthshade=False, alpha=0.85, zorder=2))
        for j in range(3):
            path = np.array([m_[j] for m_ in mus[:trail]] + [mu[j]])
            s.add(ax.plot(*path.T, color=CLASS_COLORS[j], lw=1.6, alpha=0.9, zorder=3))
            s.add(ax.scatter(*mu[j:j + 1].T, marker="*", s=260, color=CLASS_COLORS[j],
                             edgecolor=C["dark"], linewidths=1.0, depthshade=False, zorder=4))
        s.spin(i, n, sweep=30)

    return update


# ---------------------------------------------------------------- 4. reinforcement learning

@animation("RL: value spreads from the reward",
           caption=r"""यह **Reinforcement Learning** की झलक है: ज़मीन एक $6 \times 6$ grid है जिसका हर खाना एक state है
           (काले खाने दीवारें हैं), और सिर्फ़ सुनहरे तारे वाले खाने पर reward $+1$ मिलता है। हर खंभे की ऊँचाई
           उस state की **value** $V(s)$ है, यानी वहाँ से आगे मिलने वाला कुल discounted reward। हर sweep में
           value goal से एक खाना और पीछे फैलती है, और दूरी के साथ $\gamma$ के हिसाब से घटती जाती है। खंभे बन
           जाने के बाद लाल agent बस हर कदम पर सबसे ऊँचे पड़ोसी खंभे की ओर चलता है और दीवारों का चक्कर काटकर
           goal तक पहुँच जाता है। उसे किसी ने रास्ता नहीं बताया था, सिर्फ़ reward बताया था।""",
           frames=44, keys=(0.1, 0.45, 1.0))
def value_bars(s, n):
    N, gamma = 6, 0.85
    walls = {(1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (1, 1), (5, 2)}
    goal, start = (0, 5), (5, 0)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    free = [(a, b) for a in range(N) for b in range(N) if (a, b) not in walls]

    def nbrs(c):
        out = [(c[0] + dx, c[1] + dy) for dx, dy in moves]
        return [p for p in out if 0 <= p[0] < N and 0 <= p[1] < N and p not in walls]

    V = {c: 0.0 for c in free}
    V[goal] = 1.0                                            # R(goal) = 1, the episode ends there
    sweeps = [dict(V)]
    while True:                                              # value iteration, reward 0 elsewhere
        new = {c: (1.0 if c == goal else gamma * max(V[p] for p in nbrs(c))) for c in free}
        if all(abs(new[c] - V[c]) < 1e-12 for c in free):
            break
        V = new
        sweeps.append(dict(V))
    K = len(sweeps) - 1
    path = [start]
    while path[-1] != goal:                                  # greedy policy: go to the best neighbour
        path.append(max(nbrs(path[-1]), key=lambda p: V[p]))
    n_walk = len(path) + 3
    n_sweep = n - n_walk

    ax = s.ax3d(labels=("grid column", "grid row", r"$V(s)$"), elev=40, azim=-62,
                xlim=(0, N), ylim=(0, N), zlim=(0, 1.05))
    ax.computed_zorder = False
    ax.set_xticks(np.arange(N) + 0.5, [str(k) for k in range(N)])
    ax.set_yticks(np.arange(N) + 0.5, [str(k) for k in range(N)])
    tiles = [[(a, b, 0), (a + 1, b, 0), (a + 1, b + 1, 0), (a, b + 1, 0)] for a, b in walls]
    ax.add_collection3d(Poly3DCollection(tiles, facecolor=C["dark"], alpha=0.85, zorder=1))
    ax.scatter([goal[0] + 0.5], [goal[1] + 0.5], [1.04], marker="*", s=170, color="#f59e0b",
               edgecolor=C["dark"], linewidths=0.7, depthshade=False, zorder=6)
    a, b = start
    ax.plot([a, a + 1, a + 1, a, a], [b, b, b + 1, b + 1, b], [0] * 5, color=C["red"], lw=1.6, zorder=1)
    cmap = matplotlib.colormaps["viridis"]

    def update(i):
        s.clear()
        k = K if i >= n_sweep else int(round(i / (n_sweep - 1) * K))
        Vk = sweeps[k]
        cells = [c for c in free if Vk[c] > 0]
        xs = np.array([c[0] for c in cells]) + 0.1
        ys = np.array([c[1] for c in cells]) + 0.1
        hs = np.array([Vk[c] for c in cells])
        s.add(ax.bar3d(xs, ys, np.zeros_like(hs), 0.8, 0.8, hs, color=cmap(0.15 + 0.8 * hs), alpha=0.9,
                       shade=True, zorder=2))
        step = min(max(i - n_sweep + 1, 0), len(path) - 1)
        p = np.array([(c[0] + 0.5, c[1] + 0.5, Vk[c] + 0.03) for c in path[: step + 1]])
        if step > 0:
            s.add(ax.plot(*p.T, color=C["red"], lw=2.0, zorder=5))
        s.add(ax.scatter(*p[-1:].T, color=C["red"], s=70, edgecolor="white", depthshade=False, zorder=6))
        if i < n_sweep:
            s.status(rf"value iteration sweep {k}    ($\gamma$ = {gamma})")
        else:
            s.status(f"agent follows the highest value: step {step}")
        s.spin(i, n, sweep=30)

    return update


# ---------------------------------------------------------------- 5. generalization

@animation("Underfit, good fit, overfit",
           caption=r"""काले बिंदु **training set** हैं और नारंगी बिंदु **test set**, जिन्हें model ने training के समय नहीं
           देखा। Surface एक polynomial model है जिसकी degree (complexity) बढ़ती जा रही है। Degree 1 पर surface
           एक plane है जो data का मोड़ पकड़ ही नहीं पाता: train error और test error दोनों बड़े हैं
           (**underfitting**), जबकि degree 3 पर दोनों छोटे हैं। Degree और बढ़ाने पर surface हर काले बिंदु को
           छूने के लिए बुरी तरह लहराने लगती है: train error लगभग zero हो जाता है, पर नारंगी रेखाएँ लंबी हो जाती
           हैं और test error उछल जाता है (**overfitting**)। असली कसौटी test error है, train error नहीं।""",
           frames=48, keys=(0.0, 0.42, 1.0))
def fit_complexity(s, n):
    rng = np.random.default_rng(5)
    lim, zlim = (-1.5, 1.5), (-2.6, 2.6)

    def truth(a, b):
        return 1.2 * np.sin(1.4 * a) - 0.55 * b * b + 0.5

    def feats(a, b, d):
        return np.stack([a ** p * b ** q for p in range(d + 1) for q in range(d + 1 - p)], axis=-1)

    def sample(k):
        a, b = rng.uniform(*lim, k), rng.uniform(*lim, k)
        return a, b, truth(a, b) + rng.normal(0, 0.22, k)

    a_tr, b_tr, y_tr = sample(30)
    a_te, b_te, y_te = sample(24)
    G1, G2 = grid(lim, lim, 30)
    degrees = [1, 2, 3, 4, 5, 6]
    fits = []
    for d in degrees:
        th = np.linalg.lstsq(feats(a_tr, b_tr, d), y_tr, rcond=None)[0]
        fits.append((feats(G1, G2, d) @ th, feats(a_tr, b_tr, d) @ th, feats(a_te, b_te, d) @ th))

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y$"), elev=24, azim=-60,
                xlim=lim, ylim=lim, zlim=zlim, ticks=False)
    ax.scatter(a_tr, b_tr, y_tr, color=C["dark"], s=14, depthshade=False, label="training set")
    ax.scatter(a_te, b_te, y_te, color=C["orange"], s=14, depthshade=False, label="test set")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    nan = np.full(len(a_te), np.nan)
    per = n // len(degrees)

    def update(i):
        s.clear()
        j, f = divmod(i, per)
        j = min(j, len(degrees) - 1)
        Z, ptr, pte = fits[j]
        if j > 0 and f < 4:                                   # short morph from the previous degree
            t = ease((f + 1) / 5)
            Z, ptr, pte = (lerp(old, cur, t) for old, cur in zip(fits[j - 1], fits[j]))
        Zm = np.where((Z < zlim[0]) | (Z > zlim[1]), np.nan, Z)
        s.add(ax.plot_surface(G1, G2, Zm, cmap="viridis", alpha=0.42, linewidth=0, vmin=-2.2, vmax=2.2))
        seg = np.clip(pte, *zlim)
        s.add(ax.plot(np.c_[a_te, a_te, nan].ravel(), np.c_[b_te, b_te, nan].ravel(),
                      np.c_[y_te, seg, nan].ravel(), color=C["orange"], lw=1.0, alpha=0.9))
        tr, te = np.mean((ptr - y_tr) ** 2), np.mean((pte - y_te) ** 2)
        s.status(f"degree {degrees[j]}    train error {tr:.2f}    test error {te:.2f}")
        s.spin(i, n, sweep=40)

    return update
