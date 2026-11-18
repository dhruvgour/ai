"""Lecture 13: K-means, Mixture of Gaussians, EM: 3D animations."""
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, CLASS_COLORS, animation, ease, grid, lerp

_COLS = np.array(CLASS_COLORS)


# ---------------------------------------------------------------- helpers

def _kmeans_trace(X, mu0, max_iter=40):
    """Run k-means. cs[t] is the assignment made with centroids mus[t]; mus[t + 1] are the new means."""
    mus, cs = [np.array(mu0, float)], []
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - mus[-1][None]) ** 2).sum(-1)
        c = d2.argmin(1)
        if cs and np.all(c == cs[-1]):
            break
        cs.append(c)
        mus.append(np.array([X[c == j].mean(0) if np.any(c == j) else mus[-1][j]
                             for j in range(len(mu0))]))
    return mus, cs


def _kmeans_state(mus, cs, p):
    """State at continuous time p in [0, len(cs)]: (centroids, assignment, step name, iteration)."""
    t = min(int(p), len(cs) - 1)
    frac = p - t
    if frac < 0.4:
        return mus[t], cs[t], "assign", t + 1
    return lerp(mus[t], mus[t + 1], ease((frac - 0.4) / 0.6)), cs[t], "move", t + 1


def _spokes(pts, ends):
    """NaN-separated coordinates for many segments pts[i] -> ends[i] (one plot call)."""
    nan = np.full(len(pts), np.nan)
    return [np.c_[pts[:, d], ends[:, d], nan].ravel() for d in range(pts.shape[1])]


def _gauss2(P, mu, S):
    """Density N(p; mu, S) for 2D points P of shape (..., 2)."""
    d = P - mu
    q = np.einsum("...i,ij,...j->...", d, np.linalg.inv(S), d)
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(S)))


def _em_trace(X, phi, mu, Sig, iters):
    """EM for a mixture of Gaussians. Returns [(phi, mu, Sigma, w, loglik)] before each M-step."""
    k, out = len(phi), []
    for _ in range(iters + 1):
        joint = np.stack([phi[j] * _gauss2(X, mu[j], Sig[j]) for j in range(k)], axis=1)
        px = joint.sum(1)
        w = joint / px[:, None]                                    # E-step
        out.append((phi.copy(), mu.copy(), Sig.copy(), w, np.log(px).sum()))
        Nj = w.sum(0)                                              # M-step
        phi = Nj / len(X)
        mu = (w.T @ X) / Nj[:, None]
        Sig = np.array([(w[:, j, None] * (X - mu[j])).T @ (X - mu[j]) / Nj[j] for j in range(k)])
    return out


def _blobs(rng, means, covs, counts):
    return np.vstack([rng.multivariate_normal(mu, S, c) for mu, S, c in zip(means, covs, counts)])


# ---------------------------------------------------------------- 1. k-means in 3D

@animation("K-means on a 3D point cloud",
           caption=r"""तीन axes तीन features $x_1, x_2, x_3$ हैं और हर बिंदु एक unlabeled example है। बड़े तारे
           centroids $\mu_j$ हैं, जो शुरू में तीनों एक ही cluster के अंदर हैं। हर iteration में दो काम बारी-बारी से होते हैं:
           पहले हर बिंदु अपने सबसे पास वाले centroid का रंग ले लेता है (assign), फिर हर centroid अपने रंग
           के बिंदुओं के mean पर खिसक जाता है (move)। पतली रेखाएँ बिंदु से उसके centroid तक की दूरी हैं,
           जिनके squares का जोड़ distortion $J$ है। देखिए कि $J$ हर step में घटता है और जब कोई बिंदु
           रंग नहीं बदलता तो algorithm रुक जाता है।""",
           frames=44, keys=(0.0, 0.3, 1.0))
def kmeans_3d(s, n):
    rng = np.random.default_rng(4)
    centers = np.array([[-2.3, -1.2, -1.6], [2.3, -1.6, 0.4], [0.2, 2.2, 1.8]])
    X = np.clip(np.vstack([c + rng.normal(0, 0.7, (32, 3)) for c in centers]), -3.9, 3.9)
    mu0 = X[np.random.default_rng(34).choice(32, 3, replace=False)]   # all three start in one cluster
    mus, cs = _kmeans_trace(X, mu0)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=24, azim=-72,
                xlim=(-4, 4), ylim=(-4, 4), zlim=(-4, 4), ticks=False)
    hold = 5

    def update(i):
        s.clear()
        p = len(cs) * min(1.0, i / (n - 1 - hold))
        mu, c, step, it = _kmeans_state(mus, cs, p)
        if i == 0:
            s.add(ax.scatter(*X.T, color=C["gray"], s=14, depthshade=False))
            s.status("random centroids, points not assigned yet")
        else:
            s.add(ax.scatter(*X.T, c=_COLS[c], s=14, depthshade=False))
            for j in range(3):
                pts = X[c == j]
                s.add(ax.plot(*_spokes(pts, np.repeat(mu[None, j], len(pts), 0)), color=_COLS[j],
                              lw=0.5, alpha=0.35))
            J = ((X - mu[c]) ** 2).sum()
            if p >= len(cs):
                s.status(f"converged after {it} iterations    J = {J:.0f}")
            else:
                s.status(f"iteration {it}: {step}    J = {J:.0f}")
        for j in range(3):
            trail = np.vstack([np.array(mus[: it + (step == "move" and i > 0)])[:, j], mu[j]])
            s.add(ax.plot(*trail.T, color=C["dark"], lw=1.0, ls="--"))
        s.add(ax.scatter(*mu.T, c=_COLS[:3], marker="*", s=230, edgecolor="black", linewidth=0.9,
                         depthshade=False))
        s.spin(i, n, sweep=30)

    return update


# ---------------------------------------------------------------- 2. local optima

@animation("K-means: bad init, local optimum",
           caption=r"""फ़र्श पर 2D data ($x_1, x_2$) के चार साफ़ clusters हैं और तारे centroids हैं। हर बिंदु के ऊपर
           खड़ी डंडी की ऊँचाई उसकी अपने centroid से squared दूरी $\|x^{(i)} - \mu_{c^{(i)}}\|^2$ है, यानी
           सारी डंडियों का जोड़ ही distortion $J$ है। पहले run में initialization खराब है: दो centroids एक ही
           cluster को बाँट लेते हैं और एक centroid को दो clusters सँभालने पड़ते हैं, इसलिए वहाँ डंडियाँ ऊँची
           रह जाती हैं और k-means वहीं अटक जाता है (local optimum)। दूसरे run में अच्छे initialization से हर
           cluster को अपना centroid मिलता है और $J$ बहुत छोटा हो जाता है। इसीलिए k-means कई बार चलाकर
           सबसे छोटे $J$ वाला जवाब रखते हैं।""",
           frames=44, keys=(0.0, 0.44, 1.0))
def kmeans_local_optimum(s, n):
    rng = np.random.default_rng(7)
    centers = np.array([[-3.0, 2.4], [-3.0, -2.4], [3.0, 2.4], [3.0, -2.4]])
    X = np.vstack([c + rng.normal(0, 0.62, (30, 2)) for c in centers])
    runs = []
    for seed in range(40):                         # try many random inits, keep the worst and the best
        mu0 = X[np.random.default_rng(seed).choice(len(X), 4, replace=False)]
        mus, cs = _kmeans_trace(X, mu0)
        runs.append((((X - mus[-1][cs[-1]]) ** 2).sum(), seed, mus, cs))
    runs.sort(key=lambda r: (r[0], r[1]))
    (J_good, _, mus_g, cs_g), (J_bad, _, mus_b, cs_b) = runs[0], runs[-1]
    zmax = 1.08 * ((X - mus_b[-1][cs_b[-1]]) ** 2).sum(1).max()
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", "squared distance"), elev=30, azim=-66,
                xlim=(-5, 5), ylim=(-5, 5), zlim=(0, zmax), ticks=False)
    n_a, hold = int(0.46 * n), 4

    def update(i):
        s.clear()
        if i < n_a:
            mus, cs, frac = mus_b, cs_b, i / (n_a - hold)
        else:
            mus, cs, frac = mus_g, cs_g, (i - n_a) / (n - 1 - n_a - hold)
        mu, c, _, it = _kmeans_state(mus, cs, len(cs) * min(1.0, frac))
        d2 = ((X - mu[c]) ** 2).sum(1)
        top = np.minimum(d2, zmax)
        floor = np.c_[X, np.zeros(len(X))]
        for j in range(4):
            sel = c == j
            s.add(ax.plot(*_spokes(floor[sel], np.c_[X[sel], top[sel]]), color=_COLS[j], lw=0.9, alpha=0.75))
        s.add(ax.scatter(X[:, 0], X[:, 1], 0 * top, c=_COLS[c], s=7, depthshade=False))
        s.add(ax.scatter(X[:, 0], X[:, 1], top, c=_COLS[c], s=9, depthshade=False))
        s.add(ax.scatter(mu[:, 0], mu[:, 1], 0 * mu[:, 0], c=_COLS[:4], marker="*", s=230, edgecolor="black",
                         linewidth=0.9, depthshade=False))
        if i < n_a:
            s.status(f"run 1 (bad init)   iteration {it}   J = {d2.sum():.0f}")
        else:
            s.status(f"run 2 (good init)   J = {d2.sum():.0f}   (run 1: {J_bad:.0f})")
        s.spin(i, n, sweep=30)

    return update


# ---------------------------------------------------------------- 3. mixture density through EM

@animation("EM reshapes the mixture density",
           caption=r"""फ़र्श पर 2D data है और ऊँचाई mixture of Gaussians की density
           $p(x) = \sum_j \phi_j\, \mathcal{N}(x; \mu_j, \Sigma_j)$ है। लाल निशान तीनों means $\mu_j$ हैं। शुरू में
           तीनों Gaussians चौड़े हैं और लगभग एक ही जगह हैं, इसलिए surface एक ही चपटा टीला है। हर EM iteration
           में E-step तय करता है कि कौन सा बिंदु किस Gaussian का कितना है, और M-step उन्हीं weights से
           $\phi_j, \mu_j, \Sigma_j$ दोबारा निकालता है। टीला धीरे-धीरे तीन चोटियों में बँट जाता है, हर चोटी अपने
           cluster के आकार में ढल जाती है, और log-likelihood हर iteration में बढ़ता है।""",
           frames=44, keys=(0.0, 0.35, 1.0))
def em_density(s, n):
    rng = np.random.default_rng(2)
    X = _blobs(rng, [(-2.0, -1.3), (2.1, -0.9), (0.0, 2.2)],
               [[[1.0, 0.6], [0.6, 0.6]], [[0.3, 0.0], [0.0, 0.9]], [[1.3, -0.3], [-0.3, 0.3]]],
               [60, 60, 60])
    mu0 = np.array([[-0.7, -0.5], [0.8, -0.3], [0.0, 0.7]])
    n_it = 24
    trace = _em_trace(X, np.full(3, 1 / 3), mu0, np.array([np.cov(X.T)] * 3), n_it)
    lim = (-4.6, 4.6)
    GX, GY = grid(lim, lim, 38)
    G = np.stack([GX, GY], axis=-1)

    def density(state):
        phi, mu, Sig = state[:3]
        return sum(phi[j] * _gauss2(G, mu[j], Sig[j]) for j in range(3))

    zmax = 1.05 * density(trace[-1]).max()
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$p(x)$"), elev=34, azim=-64, xlim=lim, ylim=lim,
                zlim=(0, zmax), ticks=False)
    ax.scatter(X[:, 0], X[:, 1], 0 * X[:, 0], color=C["dark"], s=5, depthshade=False)
    # early iterations change the most, so show them one by one and speed up later
    its = np.round(n_it * np.linspace(0, 1, n - 4) ** 1.7).astype(int)

    def update(i):
        s.clear()
        it = its[min(i, len(its) - 1)]
        st = trace[it]
        s.add(ax.plot_surface(GX, GY, np.minimum(density(st), zmax), cmap="viridis", alpha=0.5,
                              linewidth=0, rstride=1, cstride=1, vmin=0, vmax=zmax))
        s.add(ax.scatter(st[1][:, 0], st[1][:, 1], 0, color=C["red"], marker="X", s=70,
                         edgecolor="white", depthshade=False))
        s.status(f"EM iteration {it}    log-likelihood = {st[4]:.1f}")
        s.spin(i, n, sweep=28)

    return update


# ---------------------------------------------------------------- 4. soft assignments

@animation("Soft assignments sharpen in EM",
           caption=r"""फ़र्श पर दो overlap करते clusters का 2D data है। हर बिंदु की ऊँचाई उसकी responsibility
           $w_1^{(i)} = p(z^{(i)} = 1 \mid x^{(i)})$ है, यानी EM के हिसाब से वह कितना पहले Gaussian का है
           (ऊँचाई 1 मतलब पूरी तरह पहले का, 0 मतलब पूरी तरह दूसरे का)। फ़र्श पर दोनों Gaussians के ellipses बने
           हैं। शुरू में दोनों Gaussians लगभग एक जैसे हैं, इसलिए सारे बिंदु $0.5$ के आस-पास तैर रहे हैं। जैसे-जैसे EM
           चलता है, ज़्यादातर बिंदु 1 या 0 की ओर चले जाते हैं, पर बीच के overlap वाले बिंदु बीच की ऊँचाई पर ही
           रहते हैं। यही soft assignment है। K-means में हर बिंदु सीधे 0 या 1 पर होता (hard assignment)।""",
           frames=40, keys=(0.0, 0.3, 1.0))
def soft_assignment(s, n):
    rng = np.random.default_rng(11)
    X = _blobs(rng, [(-1.5, -0.2), (1.5, 0.3)], [[[1.0, 0.35], [0.35, 0.6]], [[0.7, -0.2], [-0.2, 1.0]]],
               [70, 70])
    mu0 = np.array([[-0.25, -0.5], [0.3, 0.45]])
    trace = _em_trace(X, np.array([0.5, 0.5]), mu0, np.array([1.3 * np.cov(X.T)] * 2), 40)
    lim = (-4.2, 4.2)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$w_1^{(i)}$"), elev=20, azim=-72, xlim=lim, ylim=lim,
                zlim=(0, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks([0, 0.5, 1])
    ax.zaxis.labelpad = 2
    ax.scatter(X[:, 0], X[:, 1], 0 * X[:, 0], color=C["gray"], s=5, depthshade=False, alpha=0.6)
    ang = np.linspace(0, 2 * np.pi, 60)
    circle = np.c_[np.cos(ang), np.sin(ang)]
    cmap = colormaps["coolwarm"]
    floor = np.c_[X, np.zeros(len(X))]
    its = np.round(40 * np.linspace(0, 1, n - 4) ** 1.5).astype(int)

    def update(i):
        s.clear()
        it = its[min(i, len(its) - 1)]
        _, mu, Sig, w, _ = trace[it]
        w1 = w[:, 0]
        s.add(ax.plot(*_spokes(floor, np.c_[X, w1]), color=C["gray"], lw=0.4, alpha=0.5))
        s.add(ax.scatter(X[:, 0], X[:, 1], w1, c=cmap(w1), s=16, edgecolor="black", linewidth=0.3,
                         depthshade=False))
        for j, col in ((0, cmap(1.0)), (1, cmap(0.0))):
            e = np.clip(mu[j] + 2 * circle @ np.linalg.cholesky(Sig[j]).T, lim[0], lim[1])
            s.add(ax.plot(e[:, 0], e[:, 1], 0 * e[:, 0], color=col, lw=1.6))
        unsure = int(np.sum((w1 > 0.1) & (w1 < 0.9)))
        s.status(rf"EM iteration {it}    points with 0.1 < $w_1$ < 0.9: {unsure}")
        s.spin(i, n, sweep=30)

    return update


# ---------------------------------------------------------------- 5. EM as coordinate ascent

@animation(r"EM: coordinate ascent on J(Q,$\theta$)",
           caption=r"""यहाँ एक छोटा model है: $x \sim \frac{1}{2}\mathcal{N}(-\theta, 1) + \frac{1}{2}\mathcal{N}(\theta, 1)$,
           जिसमें सिर्फ़ एक parameter $\theta$ है। एक axis पर $\theta$ है, दूसरे पर $q$ जो distribution $Q$ को
           तय करता है ($Q_i(z) = p(z \mid x^{(i)}; q)$), और ऊँचाई lower bound $J(Q, \theta)$ है। काली रेखा वह
           जगह है जहाँ $q = \theta$, यानी जहाँ bound tight है और $J = \ell(\theta)$। **नीले** कदम E-step हैं:
           $\theta$ स्थिर, $Q$ बदलकर काली रेखा तक चढ़ना। **नारंगी** कदम M-step हैं: $Q$ स्थिर, $\theta$ बदलकर
           उस दिशा की चोटी तक जाना। दोनों कदम $J$ को बढ़ाते हैं, इसलिए सीढ़ी $\ell$ के maximum (हरा तारा) तक
           चढ़ती है और कभी नीचे नहीं उतरती।""",
           frames=44, keys=(0.12, 0.42, 1.0))
def em_coordinate_ascent(s, n):
    rng = np.random.default_rng(5)
    x = rng.choice([-1.0, 1.0], 150) * 1.25 + rng.normal(0, 1, 150)

    def post(q):                                   # Q_i(z = +1) when Q is the posterior at parameter q
        return np.clip(1 / (1 + np.exp(-2 * np.asarray(q, float)[..., None] * x)), 1e-12, 1 - 1e-12)

    def J(theta, q):                               # lower bound per example, works on grids
        w = post(q)
        th = np.asarray(theta, float)[..., None]
        const = np.log(0.5) - 0.5 * np.log(2 * np.pi)
        lp, lm = const - 0.5 * (x - th) ** 2, const - 0.5 * (x + th) ** 2
        return np.mean(w * (lp - np.log(w)) + (1 - w) * (lm - np.log(1 - w)), axis=-1)

    def m_step(q):
        return float(np.mean((2 * post(q) - 1) * x))

    lim = (0.0, 2.2)
    pts = [(0.25, 1.9)]                            # (theta, q): start with a Q that does not match theta
    for _ in range(8):
        th = pts[-1][0]
        pts.append((th, th))                       # E-step: q := theta
        pts.append((m_step(th), th))               # M-step: theta := argmax J(Q, theta)
    pts = np.array(pts)
    zs = J(pts[:, 0], pts[:, 1])
    GX, GY = grid(lim, lim, 36)
    GZ = J(GX, GY)
    zlim = (GZ.min(), GZ.max() + 0.03 * np.ptp(GZ))
    ax = s.ax3d(labels=(r"$\theta$", r"$q$  (sets $Q$)", r"$J(Q,\theta)$"), elev=36, azim=-74,
                xlim=lim, ylim=lim, zlim=zlim)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_zticklabels([])
    ax.plot_surface(GX, GY, GZ, cmap="viridis", alpha=0.42, linewidth=0, rstride=1, cstride=1)
    d = np.linspace(lim[0], lim[1], 60)
    ax.plot(d, d, J(d, d) + 0.004, color=C["dark"], lw=1.8, label=r"$q=\theta$:  $J=\ell(\theta)$")
    ax.plot(d, d, zlim[0], color=C["dark"], lw=0.8, ls=":")
    best = d[np.argmax(J(d, d))]
    ax.scatter([best], [best], [J(best, best) + 0.01], marker="*", s=150, color=C["green"],
               edgecolor="black", linewidth=0.6, depthshade=False, zorder=10)
    ax.plot([], [], [], color=C["blue"], lw=2, label="E-step")
    ax.plot([], [], [], color=C["orange"], lw=2, label="M-step")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    per = 3                                        # frames per move

    def update(i):
        s.clear()
        prog = min(i / per, len(pts) - 1)
        k, frac = int(prog), prog - int(prog)
        for a in range(min(k + 1, len(pts) - 1)):
            t = 1.0 if a < k else ease(frac)
            if t == 0:
                continue
            seg = pts[a] + np.linspace(0, t, 12)[:, None] * (pts[a + 1] - pts[a])
            col = C["blue"] if a % 2 == 0 else C["orange"]
            s.add(ax.plot(seg[:, 0], seg[:, 1], J(seg[:, 0], seg[:, 1]) + 0.006, color=col, lw=2.2))
            s.add(ax.plot(seg[:, 0], seg[:, 1], zlim[0], color=col, lw=1.0, alpha=0.8))
        cur = pts[k] + (ease(frac) * (pts[k + 1] - pts[k]) if k < len(pts) - 1 else 0)
        zc = float(J(cur[0], cur[1]))
        s.add(ax.scatter([cur[0]], [cur[1]], [zc + 0.008], color=C["red"], s=45, edgecolor="white",
                         depthshade=False))
        name = "start" if i == 0 else ("E-step" if k % 2 == 0 else "M-step")
        if k >= len(pts) - 1:
            name = "converged"
        s.status(rf"{name}    $\theta$ = {cur[0]:.2f}    $J$ = {zc:.3f}")
        s.spin(i, n, sweep=30)

    return update


# ---------------------------------------------------------------- 6. Jensen's inequality

@animation("Jensen's inequality for log",
           caption=r"""Surface एक concave function है: $f(x) = \log x_1 + \log x_2$। Random variable $X$ तीन बिंदुओं
           (फ़र्श पर त्रिकोण के कोने) में से कोई एक बराबर probability से लेता है। नारंगी त्रिकोण surface के उन्हीं
           तीन बिंदुओं को जोड़ने वाला chord है, और उसका बीच का बिंदु $E[f(X)]$ है। उसके ठीक ऊपर surface पर हरा
           बिंदु $f(E[X])$ है। Concave function में chord हमेशा surface के नीचे रहता है, इसलिए
           $f(E[X]) \ge E[f(X)]$, और लाल डंडी यही gap है। जैसे-जैसे $X$ का distribution सिकुड़कर एक ही बिंदु पर
           आता है (variance 0), gap बंद हो जाता है। EM इसी बात से bound को tight बनाता है।""",
           frames=40, keys=(0.0, 0.42, 1.0))
def jensen(s, n):
    def f(a, b):
        return np.log(a) + np.log(b)

    lim = (0.2, 3.3)
    zlim = (-2.2, 2.4)
    center = np.array([1.5, 1.5])
    ang = np.radians([165.0, 285.0, 45.0])
    dirs = np.c_[np.cos(ang), np.sin(ang)]
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$f(x)$"), elev=15, azim=-52, xlim=lim, ylim=lim, zlim=zlim,
                ticks=False)
    GX, GY = grid(lim, lim, 34)
    ax.plot_surface(GX, GY, np.maximum(f(GX, GY), zlim[0]), cmap="viridis", alpha=0.3, linewidth=0, rstride=1, cstride=1)
    fc = f(*center)
    ax.plot([center[0]] * 2, [center[1]] * 2, [zlim[0], fc], color=C["gray"], lw=0.9, ls="--")
    ax.scatter([center[0]], [center[1]], [fc], color=C["green"], s=45, edgecolor="black", linewidth=0.5,
               depthshade=False, label=r"$f(E[X])$", zorder=10)
    ax.scatter([], [], [], color=C["orange"], s=45, edgecolor="black", linewidth=0.5, label=r"$E[f(X)]$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        t = ease((i - 3) / (n - 8))
        r = 1.25 * (1 - t) + 0.02
        P = center + r * dirs
        z = f(P[:, 0], P[:, 1])
        mean_f = z.mean()
        tri = Poly3DCollection([np.c_[P, z]], facecolor=C["orange"], edgecolor=C["orange"], alpha=0.45)
        s.add(ax.add_collection3d(tri))
        loop = np.r_[0, 1, 2, 0]
        s.add(ax.plot(P[loop, 0], P[loop, 1], zlim[0], color=C["dark"], lw=1.0))
        s.add(ax.plot(*_spokes(np.c_[P, np.full(3, zlim[0])], np.c_[P, z]), color=C["dark"], lw=0.7, ls=":"))
        s.add(ax.scatter(P[:, 0], P[:, 1], z, color=C["dark"], s=22, depthshade=False))
        s.add(ax.plot([center[0]] * 2, [center[1]] * 2, [mean_f, fc], color=C["red"], lw=3.0))
        s.add(ax.scatter([center[0]], [center[1]], [mean_f], color=C["orange"], s=45, edgecolor="black",
                         linewidth=0.5, depthshade=False))
        s.status(rf"$f(E[X])$ = {fc:.2f}   $E[f(X)]$ = {mean_f:.2f}   gap = {fc - mean_f:.2f}")
        s.spin(i, n, sweep=25)

    return update
