"""Lecture 8: Bias-Variance, Regularization, Model Selection: 3D animations."""
import numpy as np

from lib.anim import C, animation, ease, grid, lerp, surface

_LIM = (-1.0, 1.0)
_ZLIM = (-2.6, 2.6)


def _truth(x1, x2):
    """The unknown target function h*(x1, x2) used by the first two animations."""
    return np.sin(2.4 * x1) + 0.8 * np.cos(2.6 * x2)


def _poly2(x1, x2, d):
    """All monomials x1^a * x2^b with a + b <= d (last axis indexes the features)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    return np.stack([x1**a * x2**b for a in range(d + 1) for b in range(d + 1 - a)], axis=-1)


def _sample(rng, m, noise):
    x1, x2 = rng.uniform(-1, 1, m), rng.uniform(-1, 1, m)
    return x1, x2, _truth(x1, x2) + rng.normal(0, noise, m)


def _sample_grid(rng, k, noise):
    """A random training set with one point in each cell of a k x k grid (inputs stay spread out)."""
    c = (np.arange(k) + 0.5) / k * 2 - 1
    A, B = np.meshgrid(c, c)
    x1 = A.ravel() + rng.uniform(-1, 1, k * k) / k
    x2 = B.ravel() + rng.uniform(-1, 1, k * k) / k
    return x1, x2, _truth(x1, x2) + rng.normal(0, noise, k * k)


def _fit(x1, x2, y, d):
    return np.linalg.lstsq(_poly2(x1, x2, d), y, rcond=None)[0]


def _stems(ax, x1, x2, z0, z1, **kw):
    """Vertical line segments from (x1, x2, z0) to (x1, x2, z1) as a single artist."""
    nan = np.full(len(x1), np.nan)
    return ax.plot(np.c_[x1, x1, nan].ravel(), np.c_[x2, x2, nan].ravel(),
                   np.c_[z0, z1, nan].ravel(), **kw)


@animation("Polynomial degree: underfit to overfit",
           caption=r"""काले बिंदु noisy training data हैं: दो features $x_1, x_2$ और ऊँचाई पर target $y$। रंगीन
           surface उन पर fit किया गया polynomial है, जिसकी degree 1 से 6 तक चढ़ती है। Degree 1 का सपाट plane
           मोड़ों को पकड़ नहीं पाता (underfitting), इसलिए लाल residuals लंबे हैं। Degree 3 के आस-पास surface
           असली आकार पकड़ लेती है। Degree 6 पर surface हर बिंदु से गुज़रने के लिए बुरी तरह मुड़ती है (overfitting):
           status line में training error लगभग zero हो जाता है पर test error उछल जाता है।""",
           frames=42, keys=(0.0, 0.4, 1.0))
def degree_sweep(s, n):
    rng = np.random.default_rng(7)
    x1, x2, y = _sample(rng, 36, 0.25)
    t1, t2, ty = _sample(rng, 2000, 0.25)
    degrees = [1, 2, 3, 4, 5, 6]
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y$"), elev=28, azim=-60,
                xlim=_LIM, ylim=_LIM, zlim=_ZLIM, ticks=False)
    ax.scatter(x1, x2, y, color=C["dark"], s=12, depthshade=False)
    G1, G2 = grid(_LIM, _LIM, 32)
    surfs, preds, errs = [], [], []
    for d in degrees:
        th = _fit(x1, x2, y, d)
        surfs.append(_poly2(G1, G2, d) @ th)
        preds.append(_poly2(x1, x2, d) @ th)
        errs.append((np.mean((preds[-1] - y) ** 2), np.mean((_poly2(t1, t2, d) @ th - ty) ** 2)))

    def update(i):
        s.clear()
        t = i / (n - 1) * (len(degrees) - 1)
        k = min(int(t), len(degrees) - 2)
        u = ease((t - k) / 0.7)                    # morph, then hold each degree for a moment
        Z = np.clip(lerp(surfs[k], surfs[k + 1], u), *_ZLIM)
        p = np.clip(lerp(preds[k], preds[k + 1], u), *_ZLIM)
        s.add(ax.plot_surface(G1, G2, Z, cmap="viridis", vmin=-2.2, vmax=2.2, alpha=0.5, linewidth=0,
                              rstride=1, cstride=1))
        s.add(_stems(ax, x1, x2, y, p, color=C["red"], lw=0.9, alpha=0.85))
        j = k + (u > 0.5)
        s.status(f"degree {degrees[j]}    train MSE = {errs[j][0]:.3f}    test MSE = {errs[j][1]:.2f}")
        s.spin(i, n, sweep=40)

    return update


@animation("Variance: refit on new training sets",
           caption=r"""स्लेटी जाली असली function $h^\star$ है। हर frame में उसी distribution से एक **नया** training set
           लिया जाता है और model दोबारा fit होता है। पहले आधे हिस्से में model सरल है (degree 1, नीला): surface लगभग
           नहीं हिलती, पर जाली से दूर रहती है (high bias, low variance)। दूसरे आधे में model complex है (degree 5,
           लाल): surface हर बार बुरी तरह बदलती है (high variance)। चार खंभों पर जमा होते बिंदु हर fit का prediction
           हैं और काला निशान सच है: नीले बिंदु सटे हुए पर सच से हटे हुए हैं, लाल बिंदु सच के आस-पास दूर-दूर बिखरे हैं।""",
           frames=44, keys=(0.1, 0.49, 1.0))
def variance_flicker(s, n):
    rng = np.random.default_rng(11)
    half = n // 2
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$\hat{h}(x)$"), elev=24, azim=-58,
                xlim=_LIM, ylim=_LIM, zlim=_ZLIM, ticks=False)
    surface(ax, _truth, _LIM, _LIM, n=28, wire=True, alpha=0.55, color=C["dark"])
    probes = np.array([[-0.45, -0.8], [-0.45, 0.8], [0.4, 0.0], [-0.85, 0.0]])
    truth_at = _truth(probes[:, 0], probes[:, 1])
    for (a, b), z in zip(probes, truth_at):
        ax.plot([a, a], [b, b], _ZLIM, color=C["gray"], lw=0.6, alpha=0.7)
    ax.scatter(probes[:, 0], probes[:, 1], truth_at, marker="D", s=34, color=C["dark"], depthshade=False,
               zorder=10)
    G1, G2 = grid(_LIM, _LIM, 24)
    fits = []                                      # (surface, predictions at the probes)
    for i in range(n):
        d = 1 if i < half else 5
        x1, x2, y = _sample_grid(rng, 5, 0.3)
        th = _fit(x1, x2, y, d)
        fits.append((np.clip(_poly2(G1, G2, d) @ th, *_ZLIM),
                     np.clip(_poly2(probes[:, 0], probes[:, 1], d) @ th, *_ZLIM)))
    P = np.array([f[1] for f in fits])
    ax.plot([], [], [], color=C["blue"], lw=2, label="degree 1 fits")
    ax.plot([], [], [], color=C["red"], lw=2, label="degree 5 fits")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        simple = i < half
        col = C["blue"] if simple else C["red"]
        s.add(ax.plot_surface(G1, G2, fits[i][0], color=col, alpha=0.3, linewidth=0, shade=False))
        lo = P[: min(i + 1, half)]                 # every degree-1 prediction so far
        s.add(ax.scatter(np.tile(probes[:, 0] - 0.07, len(lo)), np.tile(probes[:, 1], len(lo)), lo.ravel(),
                         color=C["blue"], s=9, depthshade=False))
        if not simple:
            hi = P[half: i + 1]
            s.add(ax.scatter(np.tile(probes[:, 0] + 0.07, len(hi)), np.tile(probes[:, 1], len(hi)),
                             hi.ravel(), color=C["red"], s=9, depthshade=False))
        cur = lo if simple else hi
        spread = cur.std(axis=0).mean() if len(cur) > 1 else 0.0
        s.status(f"degree {1 if simple else 5}    training set #{len(cur)}    spread of fits = {spread:.2f}")
        s.spin(i, n, sweep=30)

    return update


# ---- 2-parameter quadratic costs for the two regularization animations ----
def _quad(center, H):
    """J(theta) = 1/2 (theta - center)^T H (theta - center), usable on grids."""
    c = np.asarray(center, float)

    def J(a, b):
        da, db = a - c[0], b - c[1]
        return 0.5 * (H[0, 0] * da**2 + 2 * H[0, 1] * da * db + H[1, 1] * db**2)

    return J


@animation(r"L2 penalty pulls $\theta$ to the origin",
           caption=r"""ज़मीन पर दो parameters $\theta_1, \theta_2$ हैं और ऊँचाई cost है। स्लेटी जाली बिना regularization का
           $J(\theta)$ है, जिसका minimum हरे तारे पर है। रंगीन surface regularized cost
           $J(\theta) + \frac{\lambda}{2}\|\theta\|_2^2$ है। जैसे-जैसे $\lambda$ बढ़ता है, origin पर टिका penalty का कटोरा
           surface को हर तरफ़ से ऊपर उठाता है, बस origin पर नहीं। इसलिए minimum (लाल गेंद) हरे तारे से खिंचकर origin की
           ओर सरकता जाता है: $\|\theta\|$ छोटा होता है, पर कभी ठीक zero नहीं होता।""",
           frames=42, keys=(0.0, 0.5, 1.0))
def l2_drag(s, n):
    lim, zmax = (-1.2, 3.2), 10.0
    H = np.array([[1.6, 0.9], [0.9, 1.0]])
    best = np.array([2.3, 1.7])
    J = _quad(best, H)
    ax = s.ax3d(labels=(r"$\theta_1$", r"$\theta_2$", "cost"), elev=32, azim=-64,
                xlim=lim, ylim=lim, zlim=(0, zmax))
    X, Y = grid(lim, lim, 36)
    Z0 = J(X, Y)
    ax.plot_wireframe(X, Y, np.where(Z0 > zmax, np.nan, Z0), rstride=3, cstride=3, color=C["gray"],
                      linewidth=0.5, alpha=0.5)
    ax.contour(X, Y, Z0, levels=np.linspace(0.2, zmax, 9), zdir="z", offset=0, cmap="viridis",
               linewidths=0.6, alpha=0.7)
    ax.scatter(*best, 0, marker="*", s=110, color=C["green"], depthshade=False)
    ax.scatter(0, 0, 0, marker="+", s=90, color=C["dark"], depthshade=False)
    ax.text(0, 0, 0.6, "origin", fontsize=8, ha="center")
    lams = 2.5 * np.linspace(0, 1, n) ** 2
    path = np.array([np.linalg.solve(H + lam * np.eye(2), H @ best) for lam in lams])

    def update(i):
        s.clear()
        lam = lams[i]
        Z = Z0 + 0.5 * lam * (X**2 + Y**2)
        s.add(ax.plot_surface(X, Y, np.where(Z > zmax, np.nan, Z), cmap="viridis", vmin=0, vmax=zmax,
                              alpha=0.45, linewidth=0, rstride=1, cstride=1))
        p = path[: i + 1]
        zmin = J(p[-1, 0], p[-1, 1]) + 0.5 * lam * np.sum(p[-1] ** 2)
        s.add(ax.plot(p[:, 0], p[:, 1], 0 * p[:, 0], color=C["red"], lw=1.6))
        s.add(ax.plot([p[-1, 0]] * 2, [p[-1, 1]] * 2, [0, zmin], color=C["red"], lw=0.9, ls="--"))
        s.add(ax.scatter([p[-1, 0]] * 2, [p[-1, 1]] * 2, [0, zmin + 0.08], color=C["red"], s=50,
                         depthshade=False, edgecolor="white"))
        s.status(rf"$\lambda$ = {lam:.2f}    $\theta$ = ({p[-1, 0]:.2f}, {p[-1, 1]:.2f})"
                 rf"    $\|\theta\|$ = {np.linalg.norm(p[-1]):.2f}")
        s.spin(i, n, sweep=30)

    return update


@animation("L1 vs L2: where the minimum lands",
           caption=r"""यहाँ भी ऊँचाई cost $J(\theta)$ है, पर अब $\theta$ को एक budget के अंदर रहना है। फ़र्श पर नीला diamond L1
           budget $\|\theta\|_1 \le t$ है और नारंगी गोल घेरा L2 budget $\|\theta\|_2 \le t$ है। दोनों की सीमा को surface पर
           उठाकर भी खींचा गया है: budget के अंदर सबसे नीचा बिंदु ही constrained minimum है। जैसे-जैसे $t$ घटता है,
           L1 का बिंदु diamond के किनारे पर फिसलकर **कोने** पर पहुँचता है और वहीं अटक जाता है, यानी $\theta_1$ ठीक zero
           (sparsity)। L2 का बिंदु घेरे पर सरकता रहता है और उसके दोनों coordinates non-zero रहते हैं।""",
           frames=42, keys=(0.0, 0.35, 1.0))
def l1_vs_l2(s, n):
    lim, zmax = (-1.8, 1.8), 3.5
    H = np.array([[1.0, 0.5], [0.5, 1.0]])
    best = np.linalg.solve(H, [1.0, 1.6])          # unconstrained minimum, about (0.27, 1.47)
    J = _quad(best, H)
    ax = s.ax3d(labels=(r"$\theta_1$", r"$\theta_2$", r"$J(\theta)$"), elev=40, azim=-80,
                xlim=lim, ylim=lim, zlim=(0, zmax), ticks=False)
    X, Y = grid(lim, lim, 36)
    Z = J(X, Y)
    ax.plot_surface(X, Y, np.where(Z > zmax, np.nan, Z), cmap="viridis", vmin=0, vmax=zmax, alpha=0.22,
                    linewidth=0, rstride=1, cstride=1)
    ax.contour(X, Y, Z, levels=np.linspace(0.15, zmax, 9), zdir="z", offset=0, cmap="viridis",
               linewidths=0.6, alpha=0.7)
    ax.plot(lim, [0, 0], [0, 0], color=C["gray"], lw=0.8)
    ax.plot([0, 0], lim, [0, 0], color=C["gray"], lw=0.8)
    ang = np.linspace(0, 2 * np.pi, 401)
    circle = np.c_[np.cos(ang), np.sin(ang)]
    diamond = circle / np.abs(circle).sum(axis=1, keepdims=True)
    inside = np.linspace(0, 1, 80)[:, None, None]
    ts = np.linspace(1.75, 0.55, n)
    ax.plot([], [], [], color=C["blue"], lw=2, label=r"L1: $\|\theta\|_1 \leq t$")
    ax.plot([], [], [], color=C["orange"], lw=2, label=r"L2: $\|\theta\|_2 \leq t$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        t = ts[i]
        sol = []
        for shape, col in ((diamond, C["blue"]), (circle, C["orange"])):
            edge = t * shape
            ze = J(edge[:, 0], edge[:, 1])
            cand = (inside * edge).reshape(-1, 2)          # the whole region, not only its edge
            p = cand[np.argmin(J(cand[:, 0], cand[:, 1]))]
            zp = J(*p)
            sol.append(p)
            s.add(ax.plot(edge[:, 0], edge[:, 1], 0 * ze, color=col, lw=1.8))
            s.add(ax.plot(edge[:, 0], edge[:, 1], np.where(ze > zmax, np.nan, ze), color=col, lw=1.2,
                          alpha=0.8))
            s.add(ax.plot([p[0]] * 2, [p[1]] * 2, [0, zp], color=col, lw=0.9, ls="--"))
            s.add(ax.scatter([p[0]] * 2, [p[1]] * 2, [0, zp + 0.05], color=col, s=42, depthshade=False,
                             edgecolor="white"))
        s.status(rf"t = {t:.2f}    L1: ({abs(sol[0][0]):.2f}, {sol[0][1]:.2f})"
                 rf"    L2: ({sol[1][0]:.2f}, {sol[1][1]:.2f})")
        s.spin(i, n, sweep=30)

    return update


def _cv_surfaces():
    """Training / validation MSE of 1-D polynomial ridge over a (degree, log10 lambda) grid.

    Each cell is the median over many random training sets, so the surfaces are smooth.
    """
    rng = np.random.default_rng(5)
    degrees = np.arange(1, 11)
    loglams = np.linspace(-5, 1, 13)
    f = lambda x: np.sin(2.5 * x) + 0.7 * np.cos(3.5 * x)
    xv = np.linspace(-1, 1, 200)
    reps = 60
    tr = np.zeros((reps, len(degrees), len(loglams)))
    va = np.zeros_like(tr)
    for r in range(reps):
        x = rng.uniform(-1, 1, 24)
        y = f(x) + rng.normal(0, 0.3, 24)
        yv = f(xv) + rng.normal(0, 0.3, 200)
        for a, d in enumerate(degrees):
            A, Av = np.vander(x, d + 1, increasing=True), np.vander(xv, d + 1, increasing=True)
            for b, ll in enumerate(loglams):
                th = np.linalg.solve(A.T @ A + 10.0**ll * np.eye(d + 1), A.T @ y)
                tr[r, a, b] = np.mean((A @ th - y) ** 2)
                va[r, a, b] = np.mean((Av @ th - yv) ** 2)
    return degrees, loglams, np.median(tr, axis=0), np.median(va, axis=0)


@animation(r"Model selection over (degree, $\lambda$)",
           caption=r"""ज़मीन पर दो hyperparameters हैं: polynomial की degree और $\log_{10}\lambda$। नीली surface training error
           है और नारंगी surface validation error (कई random training sets पर नापा गया)। पहले marker सबसे छोटे $\lambda$
           पर degree बढ़ाता है: नीला बिंदु लगातार नीचे जाता है, पर लाल बिंदु पहले गिरता है और फिर ऊपर चढ़ जाता है
           (overfitting)। दोनों के बीच की खड़ी रेखा generalization gap है। फिर marker $\lambda$ बढ़ाता है और validation
           error दोबारा गिरता है। अंत में वह validation surface के सबसे निचले बिंदु (हरा तारा) पर रुकता है: model वहीं
           चुना जाता है, वहाँ नहीं जहाँ training error सबसे कम है।""",
           frames=46, keys=(0.14, 0.5, 1.0))
def cv_search(s, n):
    degrees, loglams, tr, va = _cv_surfaces()
    zmax = 0.7
    ax = s.ax3d(labels=("degree", r"$\log_{10}\lambda$", "error"), elev=24, azim=-128,
                xlim=(1, 10), ylim=(-5, 1), zlim=(0, zmax))
    ax.zaxis.labelpad = 7
    D, L = np.meshgrid(degrees, loglams, indexing="ij")
    ax.plot_surface(D, L, np.clip(tr, 0, zmax), color=C["blue"], alpha=0.3, linewidth=0.3,
                    edgecolor=(0.15, 0.39, 0.92, 0.45), shade=False)
    ax.plot_surface(D, L, np.clip(va, 0, zmax), color=C["orange"], alpha=0.3, linewidth=0.3,
                    edgecolor=(0.92, 0.35, 0.05, 0.45), shade=False)
    ax.plot([], [], [], color=C["blue"], lw=2, label="training error")
    ax.plot([], [], [], color=C["orange"], lw=2, label="validation error")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    a_best, b_best = np.unravel_index(np.argmin(va), va.shape)
    # search path in grid-index coordinates: raise the degree, then raise lambda, then go to the best cell
    b_top = int(np.argmin(va[-1]))
    corners = np.array([[0, 0], [len(degrees) - 1, 0], [len(degrees) - 1, b_top], [a_best, b_best]], float)
    stops = np.array([0.0, 0.5, 0.8, 1.0])

    def at(Zs, a, b):
        """Bilinear interpolation of a grid at fractional index (a, b)."""
        a0, b0 = min(int(a), len(degrees) - 2), min(int(b), len(loglams) - 2)
        fa, fb = a - a0, b - b0
        return ((1 - fa) * (1 - fb) * Zs[a0, b0] + fa * (1 - fb) * Zs[a0 + 1, b0]
                + (1 - fa) * fb * Zs[a0, b0 + 1] + fa * fb * Zs[a0 + 1, b0 + 1])

    def position(t):
        k = min(np.searchsorted(stops, t, side="right") - 1, len(stops) - 2)
        return lerp(corners[k], corners[k + 1], (t - stops[k]) / (stops[k + 1] - stops[k]))

    trail = np.array([position(t) for t in np.linspace(0, 1, n)])
    deg = 1 + trail[:, 0]
    ll = loglams[0] + trail[:, 1] * (loglams[1] - loglams[0])
    vtr = np.array([at(tr, a, b) for a, b in trail])
    vva = np.array([at(va, a, b) for a, b in trail])
    ztr, zva = np.clip(vtr, 0, zmax), np.clip(vva, 0, zmax)

    def update(i):
        s.clear()
        k = slice(0, i + 1)
        s.add(ax.plot(deg[k], ll[k], ztr[k], color=C["blue"], lw=2.0))
        s.add(ax.plot(deg[k], ll[k], zva[k], color=C["red"], lw=2.0))
        s.add(ax.plot([deg[i]] * 2, [ll[i]] * 2, [ztr[i], zva[i]], color=C["dark"], lw=1.0, ls="--"))
        s.add(ax.scatter([deg[i]] * 2, [ll[i]] * 2, [ztr[i], zva[i]], color=[C["blue"], C["red"]], s=45,
                         depthshade=False, edgecolor="white"))
        if i == n - 1:
            s.add(ax.scatter([deg[i]], [ll[i]], [zva[i]], marker="*", s=170, color=C["green"],
                             depthshade=False))
        s.status(rf"degree {deg[i]:.0f}  $\log_{{10}}\lambda$ = {ll[i]:.1f}"
                 f"   train {vtr[i]:.3f}   val {vva[i]:.3f}")
        s.spin(i, n, sweep=25)

    return update
