"""Lecture 3: Locally Weighted Regression, Logistic Regression, Newton's Method: 3D animations."""
import numpy as np

from lib.anim import C, animation, ease, grid, lerp


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _steps(n, last):
    """n increasing iteration counts from 0 to ``last``, dense at the start."""
    return np.round(np.geomspace(1, last + 1, n) - 1).astype(int)


# ================================================================ Locally Weighted Regression

_LIM = (-3.0, 3.0)


def _truth(a, b):
    return np.sin(a) + 0.6 * np.cos(1.2 * b)


def _lwr_data(seed, m, noise):
    rng = np.random.default_rng(seed)
    P = rng.uniform(-2.9, 2.9, (m, 2))
    return P, _truth(P[:, 0], P[:, 1]) + rng.normal(0, noise, m)


def _lwr(Q, P, y, tau):
    """Locally weighted linear regression at every query row of Q.

    Returns (theta, prediction, weights): one theta = (theta0, theta1, theta2) per query point.
    """
    d2 = ((Q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / (2 * tau**2))
    A = np.c_[np.ones(len(P)), P]
    AtWA = np.einsum("qm,mi,mj->qij", W, A, A) + 1e-9 * np.eye(3)
    AtWy = np.einsum("qm,mi,m->qi", W, A, y)
    theta = np.linalg.solve(AtWA, AtWy[..., None])[..., 0]
    return theta, theta[:, 0] + (theta[:, 1:] * Q).sum(1), W


@animation("LWR: a local plane follows the query",
           caption=r"""ज़मीन पर दो features $x_1, x_2$ हैं और ऊँचाई पर target $y$। Data एक मुड़ी हुई surface से आया है,
           जिसे कोई एक plane fit नहीं कर सकता। नारंगी चौकोर query point $x$ है। हर frame में Locally Weighted
           Regression सिर्फ़ उसी $x$ के लिए एक नया plane fit करता है (नीला टुकड़ा), जिसमें पास के बिंदुओं का
           weight $w^{(i)}$ बड़ा है (बड़े, गहरे लाल बिंदु) और दूर के बिंदुओं का लगभग zero। Query के साथ-साथ
           plane भी झुकता और घूमता है, और लाल रेखा उसके predictions का निशान है: सीधे टुकड़ों से एक मुड़ा
           हुआ fit बन जाता है। कीमत यह है कि हर नए $x$ पर fit दोबारा करना पड़ता है।""",
           frames=44, keys=(0.08, 0.5, 1.0))
def lwr_local_plane(s, n):
    P, y = _lwr_data(2, 150, 0.12)
    tau = 0.6
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y$"), elev=30, azim=-62,
                xlim=_LIM, ylim=_LIM, zlim=(-2.2, 2.2), ticks=False)
    ax.scatter(P[:, 0], P[:, 1], y, color=C["gray"], s=5, alpha=0.6, depthshade=False)
    ang = np.linspace(0.25 * np.pi, 2.25 * np.pi, n)
    Q = np.c_[1.9 * np.cos(ang), 1.9 * np.sin(ang)]
    theta, pred, W = _lwr(Q, P, y, tau)
    U, V = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6))

    def update(i):
        s.clear()
        q, th, w = Q[i], theta[i], W[i]
        near = w > 0.03
        s.add(ax.scatter(P[near, 0], P[near, 1], y[near], c=w[near], cmap="Reds", vmin=-0.25, vmax=1.0,
                         s=8 + 45 * w[near], edgecolor=C["dark"], linewidths=0.3, depthshade=False))
        A, B = np.clip(q[0] + U, *_LIM), np.clip(q[1] + V, *_LIM)
        s.add(ax.plot_surface(A, B, th[0] + th[1] * A + th[2] * B, color=C["blue"], alpha=0.45,
                              linewidth=0, shade=False))
        s.add(ax.plot(Q[: i + 1, 0], Q[: i + 1, 1], pred[: i + 1], color=C["red"], lw=2.0))
        s.add(ax.plot([q[0], q[0]], [q[1], q[1]], [-2.2, pred[i]], color=C["orange"], lw=1.3, ls="--"))
        s.add(ax.scatter([q[0]], [q[1]], [-2.2], color=C["orange"], marker="s", s=38, depthshade=False))
        s.add(ax.scatter([q[0]], [q[1]], [pred[i]], color=C["red"], s=50, edgecolor="white",
                         depthshade=False))
        s.status(rf"query $x$ = ({q[0]:.1f}, {q[1]:.1f})    prediction = {pred[i]:.2f}")
        s.spin(i, n, sweep=30)

    return update


@animation(r"LWR: the bandwidth $\tau$",
           caption=r"""वही Locally Weighted Regression, पर अब हर grid point को query मानकर पूरी fitted surface
           बनाई गई है, और bandwidth $\tau$ बड़े से छोटा होता जा रहा है। $\tau$ बड़ा हो तो हर बिंदु का weight
           लगभग बराबर है और surface लगभग एक plane रह जाती है (**underfitting**)। बीच के $\tau$ पर surface data
           का मोड़ पकड़ लेती है और सच्चे function से error सबसे छोटा होता है। $\tau$ बहुत छोटा हो तो हर
           prediction सिर्फ़ दो-चार पड़ोसियों पर टिका होता है, surface noise के पीछे भागकर ऊबड़-खाबड़ हो जाती
           है और error फिर बढ़ जाता है (**overfitting**)।""",
           frames=42, keys=(0.0, 0.62, 1.0))
def lwr_bandwidth(s, n):
    P, y = _lwr_data(5, 90, 0.3)
    zlim = (-2.6, 2.6)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y$"), elev=30, azim=-62,
                xlim=_LIM, ylim=_LIM, zlim=zlim, ticks=False)
    ax.scatter(P[:, 0], P[:, 1], y, color=C["dark"], s=9, depthshade=False)
    G1, G2 = grid(_LIM, _LIM, 26)
    Q = np.c_[G1.ravel(), G2.ravel()]
    true = _truth(G1, G2)
    taus = np.geomspace(4.0, 0.16, n)
    fits = [_lwr(Q, P, y, t)[1].reshape(G1.shape) for t in taus]
    errs = [np.mean((Z - true) ** 2) for Z in fits]
    best = int(np.argmin(errs))

    def update(i):
        s.clear()
        Z = np.clip(fits[i], *zlim)
        s.add(ax.plot_surface(G1, G2, Z, cmap="viridis", alpha=0.5, linewidth=0, vmin=-1.8, vmax=1.8))
        note = "  (best)" if i == best else ""
        s.status(rf"$\tau$ = {taus[i]:.2f}    error vs true function = {errs[i]:.3f}{note}")
        s.spin(i, n, sweep=30)

    return update


# ================================================================ Logistic Regression

@animation("Logistic regression: sigmoid surface",
           caption=r"""ज़मीन पर features $x_1, x_2$ हैं और ऊँचाई एक probability है। हर बिंदु अपने label की ऊँचाई पर
           रखा है: लाल बिंदु $y = 0$ पर, नीले $y = 1$ पर। Surface $h_\theta(x) = g(\theta^T x)$ है, यानी
           model के हिसाब से $y = 1$ होने की probability। $\theta = 0$ पर surface हर जगह $0.5$ है। Gradient
           ascent के साथ वह एक तरफ़ $1$ की ओर उठती है और दूसरी तरफ़ $0$ की ओर गिरती है, और पतली रेखाएँ
           (errors $y^{(i)} - h_\theta(x^{(i)})$) छोटी होती जाती हैं। काली रेखा, जहाँ $h_\theta(x) = 0.5$ है,
           decision boundary है: sigmoid मुड़ा हुआ है, पर boundary सीधी है।""",
           frames=40, keys=(0.0, 0.35, 1.0))
def sigmoid_surface(s, n):
    rng = np.random.default_rng(3)
    m = 35
    P = np.r_[rng.normal([1.1, 0.9], 1.0, (m, 2)), rng.normal([-1.1, -0.9], 1.0, (m, 2))]
    P = np.clip(P, -3.3, 3.3)
    y = np.r_[np.ones(m), np.zeros(m)]
    X = np.c_[np.ones(2 * m), P]
    lim = (-3.5, 3.5)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$h_\theta(x)$"), elev=22, azim=-68,
                xlim=lim, ylim=lim, zlim=(0, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.scatter(P[y == 1, 0], P[y == 1, 1], 1.0, color=C["blue"], s=13, depthshade=False, label="y = 1")
    ax.scatter(P[y == 0, 0], P[y == 0, 1], 0.0, color=C["red"], s=13, depthshade=False, label="y = 0")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    ks = _steps(n, 600)
    thetas, th = {}, np.zeros(3)
    for k in range(ks[-1] + 1):
        thetas[k] = th.copy()
        th = th + 0.02 * X.T @ (y - _sigmoid(X @ th))          # batch gradient ascent on l(theta)
    G1, G2 = grid(lim, lim, 30)
    nan = np.full(2 * m, np.nan)
    cols = np.where(y == 1, C["blue"], C["red"])

    def update(i):
        s.clear()
        k = ks[i]
        th = thetas[k]
        h = _sigmoid(X @ th)
        s.add(ax.plot_surface(G1, G2, _sigmoid(th[0] + th[1] * G1 + th[2] * G2), cmap="coolwarm_r",
                              vmin=0, vmax=1, alpha=0.45, linewidth=0))
        for cls in (0, 1):
            sel = y == cls
            s.add(ax.plot(np.c_[P[sel, 0], P[sel, 0], nan[sel]].ravel(),
                          np.c_[P[sel, 1], P[sel, 1], nan[sel]].ravel(),
                          np.c_[y[sel], h[sel], nan[sel]].ravel(), color=cols[sel][0], lw=0.6, alpha=0.7))
        if abs(th[2]) > 1e-6:                                   # boundary: theta^T x = 0, where h = 0.5
            a = np.linspace(*lim, 60)
            b = -(th[0] + th[1] * a) / th[2]
            ok = (b > lim[0]) & (b < lim[1])
            s.add(ax.plot(a[ok], b[ok], 0.5, color=C["dark"], lw=1.8))
        ll = np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        s.status(rf"iteration {k}    $\ell(\theta)$ = {ll:.2f}")
        s.spin(i, n, sweep=35)

    return update


# ---- shared 2-parameter logistic problem for the log-likelihood animations (no intercept) ----
# x2 is strongly correlated with x1 and unscaled, so the hill l(theta) is a long narrow ridge.
_rng = np.random.default_rng(0)
_m = 80
_x1 = np.clip(_rng.normal(0, 1.0, _m), -2.8, 2.8)
_x2 = np.clip(1.5 * _x1 + 0.8 * _rng.normal(0, 1.0, _m), -4.8, 4.8)
_X = np.c_[_x1, _x2]
_y = (_rng.uniform(size=_m) < _sigmoid(_X @ np.array([3.0, -1.2]))).astype(float)
_T1, _T2 = (-0.6, 4.6), (-3.2, 1.2)
_ZLIM = (-80.0, -40.0)


def _ll(t1, t2):
    """Log-likelihood l(theta1, theta2) of the shared logistic problem (works on grids)."""
    z = np.asarray(t1, float)[..., None] * _x1 + np.asarray(t2, float)[..., None] * _x2
    return np.sum(_y * z - np.logaddexp(0, z), axis=-1)


def _grad_hess(th):
    h = _sigmoid(_X @ th)
    return _X.T @ (_y - h), -(_X.T * (h * (1 - h))) @ _X


def _newton_path(steps):
    th = np.zeros(2)
    path = [th.copy()]
    for _ in range(steps):
        g, H = _grad_hess(th)
        th = th - np.linalg.solve(H, g)
        path.append(th.copy())
    return np.array(path)


_ALPHA = 1.0 / np.linalg.eigvalsh(_X.T @ _X / 4).max()        # 1 / (largest curvature at theta = 0)


def _ga_path(steps):
    th = np.zeros(2)
    path = [th.copy()]
    for _ in range(steps):
        th = th + _ALPHA * _grad_hess(th)[0]
        path.append(th.copy())
    return np.array(path)


_MLE = _newton_path(12)[-1]
_LMAX = float(_ll(*_MLE))


def _hill(s, alpha=0.4):
    ax = s.ax3d(labels=(r"$\theta_1$", r"$\theta_2$", r"$\ell(\theta)$"), elev=30, azim=-140,
                xlim=_T1, ylim=_T2, zlim=_ZLIM)
    ax.zaxis.set_rotate_label(False)                           # keep the z label upright at this azimuth
    ax.set_zlabel(r"$\ell(\theta)$", rotation=90, labelpad=-2)
    G1, G2 = grid(_T1, _T2, 40)
    Z = _ll(G1, G2)
    ax.plot_surface(G1, G2, np.where(Z < _ZLIM[0], np.nan, Z), cmap="viridis", alpha=alpha, linewidth=0,
                    vmin=_ZLIM[0], vmax=_ZLIM[1])
    ax.contour(G1, G2, Z, levels=_LMAX - np.array([32, 22, 14, 8, 4, 1.5, 0.3]), zdir="z",
               offset=_ZLIM[0], cmap="viridis", linewidths=0.7, alpha=0.8)
    ax.scatter([_MLE[0]], [_MLE[1]], [_ZLIM[0]], marker="*", s=110, color=C["green"], depthshade=False)
    return ax


def _gap(value):
    """Short text for the distance l* - l still left to climb."""
    if value >= 0.1:
        return f"{value:.1f}"
    return f"{value:.0e}" if value > 1e-12 else "0"


def _draw_path(s, ax, p, color, lw=1.6, marker=None):
    z = _ll(p[:, 0], p[:, 1]) + 0.4
    s.add(ax.plot(p[:, 0], p[:, 1], z, color=color, lw=lw, marker=marker, ms=3))
    s.add(ax.plot(p[:, 0], p[:, 1], np.full(len(p), _ZLIM[0]), color=color, lw=0.9, ls="--"))
    s.add(ax.scatter(p[-1:, 0], p[-1:, 1], z[-1:], color=color, s=50, edgecolor="white", depthshade=False))


@animation(r"Gradient ascent on $\ell(\theta)$",
           caption=r"""अब parameter space देखिए: ज़मीन पर $\theta_1, \theta_2$ और ऊँचाई पर log-likelihood
           $\ell(\theta)$। Linear Regression का $J$ एक कटोरा था जिसकी तली ढूँढनी थी। यहाँ $\ell$ एक concave
           पहाड़ी है जिसकी चोटी (हरा तारा, maximum likelihood) ढूँढनी है, इसलिए update में plus का निशान है:
           $\theta := \theta + \alpha \nabla_\theta \ell(\theta)$। लाल गेंद $\theta = 0$ से चलती है। पहले कुछ
           कदमों में वह तेज़ चढ़ाई चढ़कर पहाड़ी की लंबी पीठ (ridge) पर पहुँच जाती है, फिर पीठ के साथ-साथ बहुत
           धीरे चोटी की ओर खिसकती है, क्योंकि वहाँ gradient छोटा है। पहाड़ी concave है, इसलिए कोई दूसरी चोटी
           नहीं है जहाँ वह अटक जाए।""",
           frames=44, keys=(0.0, 0.3, 1.0))
def ascent_on_hill(s, n):
    ax = _hill(s)
    ks = _steps(n, 160)
    path = _ga_path(ks[-1])

    def update(i):
        s.clear()
        k = ks[i]
        _draw_path(s, ax, path[: k + 1], C["red"])
        g = _grad_hess(path[k])[0]
        s.status(rf"step {k}    $\ell$ = {_ll(*path[k]):.2f}    "
                 rf"$\|\nabla\ell\|$ = {np.linalg.norm(g):.2f}")
        s.spin(i, n, sweep=32)

    return update


@animation("Newton's method vs gradient ascent",
           caption=r"""उसी पहाड़ी पर, उसी जगह से दो तरीके साथ-साथ चल रहे हैं, और दोनों के लिए गिनती एक ही है: एक
           iteration। **नीला** gradient ascent है, जो सिर्फ़ ढलान देखता है। **लाल** Newton's method है, जो
           Hessian से surface का मोड़ (curvature) भी देखता है: $\theta := \theta - H^{-1} \nabla_\theta \ell$।
           Newton तीन-चार छलाँगों में चोटी पर है, और status line में उसका बचा हुआ फ़ासला $\ell^* - \ell$ हर
           iteration में लगभग square होता जाता है (quadratic convergence)। Gradient ascent को वही दूरी तय करने
           में सैकड़ों iterations लगते हैं। ध्यान रहे कि Newton का एक iteration महँगा है, क्योंकि उसमें
           $H$ बनाकर एक linear system हल करना पड़ता है।""",
           frames=44, keys=(0.0, 0.3, 1.0))
def newton_vs_ascent(s, n):
    ax = _hill(s, alpha=0.3)
    slow = 24                                                  # first 24 frames: 4 frames per iteration
    its = np.r_[np.arange(slow) / 4.0, np.round(np.geomspace(6, 300, n - slow))]
    newton, ga = _newton_path(int(its[-1])), _ga_path(int(its[-1]))
    ax.plot([], [], [], color=C["red"], lw=2, label="Newton's method")
    ax.plot([], [], [], color=C["blue"], lw=2, label="gradient ascent")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def upto(path, t):
        k = int(np.floor(t))
        p = path[: k + 1]
        if t > k:
            p = np.vstack([p, lerp(path[k], path[k + 1], ease(t - k))])
        return p

    def update(i):
        s.clear()
        t = its[i]
        _draw_path(s, ax, upto(ga, t), C["blue"], lw=1.4)
        _draw_path(s, ax, upto(newton, t), C["red"], lw=1.8, marker="o")
        k = int(np.floor(t))
        gap_n, gap_g = max(_LMAX - _ll(*newton[k]), 0.0), max(_LMAX - _ll(*ga[k]), 0.0)
        s.status(rf"iteration {k}    $\ell^* - \ell$:  Newton {_gap(gap_n)},  GA {_gap(gap_g)}")
        s.spin(i, n, sweep=32)

    return update


@animation("Newton: jump to the parabola's top",
           caption=r"""Newton's method इतना तेज़ क्यों है? हर iteration में वह वर्तमान $\theta$ (लाल गेंद) पर
           $\ell$ की ऊँचाई, gradient और Hessian तीनों मिलाकर एक **quadratic surface** (नारंगी जाली) बनाता है
           जो उस जगह असली पहाड़ी से चिपकी होती है। Quadratic की चोटी सीधे formula से मिल जाती है, और वह ठीक
           $\theta - H^{-1} \nabla_\theta \ell$ पर है, इसलिए गेंद सीधे वहीं कूद जाती है। पहली जाली असली पहाड़ी
           से काफ़ी अलग है, पर चोटी के पास आते-आते जाली और पहाड़ी लगभग एक हो जाती हैं। इसीलिए आखिरी छलाँगें
           लगभग बिल्कुल सही पड़ती हैं।""",
           frames=40, keys=(0.1, 0.4, 1.0))
def newton_quadratic(s, n):
    ax = _hill(s, alpha=0.3)
    path = _newton_path(4)
    G1, G2 = grid(_T1, _T2, 33)
    per = n // 4

    def quad(th):
        g, H = _grad_hess(th)
        D1, D2 = G1 - th[0], G2 - th[1]
        return (_ll(*th) + g[0] * D1 + g[1] * D2
                + 0.5 * (H[0, 0] * D1**2 + 2 * H[0, 1] * D1 * D2 + H[1, 1] * D2**2))

    quads = [quad(th) for th in path[:4]]

    def update(i):
        s.clear()
        k, f = divmod(i, per)
        k = min(k, 3)
        grow = ease((f + 1) / 4)                               # the quadratic model appears ...
        move = ease((f - 4) / (per - 6))                       # ... then the ball jumps to its top
        Q = quads[k]
        lo = _ZLIM[1] - grow * (_ZLIM[1] - _ZLIM[0])
        Q = np.where((Q < max(lo, _ZLIM[0])) | (Q > _ZLIM[1]), np.nan, Q)
        s.add(ax.plot_wireframe(G1, G2, Q, color=C["orange"], linewidth=0.8, alpha=0.9,
                                rstride=4, cstride=4))
        cur = lerp(path[k], path[k + 1], move)
        _draw_path(s, ax, np.vstack([path[: k + 1], cur]), C["red"], lw=1.8, marker="o")
        s.add(ax.scatter([path[k + 1, 0]], [path[k + 1, 1]], [quads[k].max() + 0.4], marker="^", s=40,
                         color=C["orange"], edgecolor=C["dark"], linewidths=0.5, depthshade=False))
        s.status(rf"Newton iteration {k + 1}:  $\ell$ = {_ll(*cur):.3f}   ($\ell^*$ = {_LMAX:.3f})")
        s.spin(i, n, sweep=32)

    return update
