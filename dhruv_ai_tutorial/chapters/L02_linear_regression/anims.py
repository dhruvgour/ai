"""Lecture 2: Linear Regression: 3D animations."""
import numpy as np

from lib.anim import C, animation, arrow, plane, surface

# ---- shared 1-feature dataset: y = 1 + 2x + noise, x deliberately not centred ----
_rng = np.random.default_rng(0)
_x = _rng.uniform(0.0, 2.0, 30)
_y = 1.0 + 2.0 * _x + _rng.normal(0, 0.4, 30)
_X = np.c_[np.ones_like(_x), _x]
_m = len(_x)
_H = _X.T @ _X / _m                       # Hessian of J
_LMAX = np.linalg.eigvalsh(_H).max()
_TLIM0, _TLIM1 = (-1.5, 3.5), (-0.5, 4.5)


def _J(t0, t1):
    """Mean squared-error cost J(theta0, theta1) on the shared dataset (works on grids)."""
    t0, t1 = np.asarray(t0, float), np.asarray(t1, float)
    err = t0[..., None] + t1[..., None] * _x - _y
    return 0.5 * np.mean(err**2, axis=-1)


def _gd_path(start, alpha, steps):
    th = np.array(start, float)
    path = [th.copy()]
    for _ in range(steps):
        th = th - alpha * (_X.T @ (_X @ th - _y)) / _m
        path.append(th.copy())
    return np.array(path)


def _bowl(s, alpha=0.45, zmax=9.5):
    ax = s.ax3d(labels=(r"$\theta_0$", r"$\theta_1$", r"$J(\theta)$"), elev=32, azim=-62,
                xlim=_TLIM0, ylim=_TLIM1, zlim=(0, zmax))
    _, X, Y, Z = surface(ax, _J, _TLIM0, _TLIM1, n=36, alpha=alpha)
    ax.contour(X, Y, Z, levels=np.linspace(0.15, zmax, 10), zdir="z", offset=0, cmap="viridis",
               linewidths=0.7, alpha=0.8)
    return ax


@animation("Fitting a plane with GD",
           caption=r"""दो features $x_1, x_2$ और target $y$ वाले data points 3D में बिखरे हैं। नीला plane
           hypothesis $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2$ है। शुरू में $\theta = 0$ है,
           इसलिए plane सपाट है। हर iteration में Gradient Descent plane को थोड़ा घुमाता और उठाता है। लाल
           रेखाएँ residuals (errors) हैं: देखिए कैसे वे छोटी होती जाती हैं और $J(\theta)$ गिरता जाता है।""",
           frames=40)
def plane_fit(s, n):
    rng = np.random.default_rng(1)
    m = 45
    x1, x2 = rng.uniform(-2, 2, m), rng.uniform(-2, 2, m)
    y = 1.0 + 1.5 * x1 - 1.0 * x2 + rng.normal(0, 0.45, m)
    X = np.c_[np.ones(m), x1, x2]
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y$"), elev=26, azim=-165,
                xlim=(-2, 2), ylim=(-2, 2), zlim=(-5, 6))
    ax.scatter(x1, x2, y, color=C["dark"], s=14, depthshade=False)

    thetas = [np.zeros(3)]
    for _ in range(n - 1):
        th = thetas[-1]
        thetas.append(th - 0.13 * X.T @ (X @ th - y) / m)

    def update(i):
        s.clear()
        th = thetas[i]
        pred = X @ th
        s.add(plane(ax, lambda a, b: th[0] + th[1] * a + th[2] * b, (-2, 2), (-2, 2), alpha=0.35))
        nan = np.full(m, np.nan)
        s.add(ax.plot(np.c_[x1, x1, nan].ravel(), np.c_[x2, x2, nan].ravel(),
                      np.c_[y, pred, nan].ravel(), color=C["red"], lw=0.9, alpha=0.8))
        s.status(rf"iteration {i}    $J(\theta)$ = {0.5 * np.mean((pred - y) ** 2):.3f}")
        s.spin(i, n, sweep=40)

    return update


@animation(r"Gradient Descent on J($\theta$)",
           caption=r"""अब हम data की जगह **parameter space** देख रहे हैं: ज़मीन पर $\theta_0$ और $\theta_1$, और
           ऊँचाई $J(\theta)$। Surface एक convex कटोरा है। लाल गेंद वर्तमान $\theta$ है। हर step में वह सबसे
           तेज़ ढलान की दिशा में चलती है: पहले तेज़ी से घाटी में उतरती है, फिर लंबी-पतली घाटी के साथ धीरे-धीरे
           minimum (हरा तारा) की ओर बढ़ती है। नीचे फ़र्श पर contours और रास्ते की परछाईं भी दिख रही है।""",
           frames=44)
def gd_on_bowl(s, n):
    ax = _bowl(s)
    path = _gd_path((-1.0, 0.2), 0.42, n - 1)
    z = _J(path[:, 0], path[:, 1])
    best = np.linalg.solve(_X.T @ _X, _X.T @ _y)
    ax.scatter(*best, 0, marker="*", s=110, color=C["green"], depthshade=False)

    def update(i):
        s.clear()
        p, zz = path[: i + 1], z[: i + 1]
        s.add(ax.plot(p[:, 0], p[:, 1], zz + 0.05, color=C["red"], lw=1.8))
        s.add(ax.plot(p[:, 0], p[:, 1], 0 * zz, color=C["red"], lw=1.0, ls="--"))
        s.add(ax.scatter(p[-1:, 0], p[-1:, 1], zz[-1:] + 0.05, color=C["red"], s=55, depthshade=False,
                         edgecolor="white"))
        s.status(rf"step {i}    $\theta$ = ({p[-1, 0]:.2f}, {p[-1, 1]:.2f})    $J$ = {zz[-1]:.3f}")
        s.spin(i, n, sweep=45)

    return update


@animation(r"Learning rate $\alpha$: small / good / large",
           caption=r"""एक ही जगह से तीन गेंदें छोड़ी गई हैं, बस $\alpha$ अलग है। **नीली** ($\alpha$ छोटा)
           सही दिशा में है पर रेंग रही है। **हरी** ($\alpha$ ठीक) जल्दी तली तक पहुँच जाती है। **लाल**
           ($\alpha$ बड़ा) हर कदम में घाटी के पार कूद जाती है और दीवारों के बीच zig-zag करती रहती है।
           $\alpha$ इससे थोड़ा और बड़ा हो तो यही उछाल हर step में बढ़ेगा और algorithm diverge कर जाएगा।""",
           frames=40)
def learning_rates(s, n):
    ax = _bowl(s, alpha=0.3)
    start = (-1.0, 0.2)
    runs = [("small", 0.04, C["blue"]), ("good", 0.45, C["green"]),
            ("too large", 1.93 / _LMAX, C["red"])]
    paths = [(_gd_path(start, a, n - 1), col, name, a) for name, a, col in runs]
    for _, col, name, a in paths:
        ax.plot([], [], [], color=col, lw=2, label=rf"{name}: $\alpha$={a:.2f}")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        for p, col, _, _ in paths:
            q = p[: i + 1]
            zz = _J(q[:, 0], q[:, 1]) + 0.05
            s.add(ax.plot(q[:, 0], q[:, 1], zz, color=col, lw=1.5, marker="o", ms=2.5))
            s.add(ax.scatter(q[-1:, 0], q[-1:, 1], zz[-1:], color=col, s=45, depthshade=False,
                             edgecolor="white"))
        s.status(f"step {i}")
        s.spin(i, n, sweep=35)

    return update


@animation("Batch GD vs Stochastic GD",
           caption=r"""**नीला** रास्ता Batch Gradient Descent है: हर step में पूरे data का gradient, इसलिए रास्ता
           smooth है। **नारंगी** रास्ता SGD है: हर update सिर्फ़ एक random example से, इसलिए रास्ता लड़खड़ाता
           है। फिर भी वह उसी घाटी में पहुँचता है और अंत में minimum के आस-पास मँडराता रहता है, पूरी तरह
           रुकता नहीं। यहाँ एक frame में Batch का एक step और SGD के चार सस्ते updates दिखाए गए हैं।""",
           frames=44)
def batch_vs_sgd(s, n):
    ax = _bowl(s, alpha=0.3)
    start = (-1.0, 0.2)
    batch = _gd_path(start, 0.42, n - 1)
    rng = np.random.default_rng(3)
    per = 4
    th = np.array(start, float)
    sgd = [th.copy()]
    for _ in range((n - 1) * per):
        j = rng.integers(_m)
        th = th - 0.11 * (_X[j] @ th - _y[j]) * _X[j]
        sgd.append(th.copy())
    sgd = np.array(sgd)
    ax.plot([], [], [], color=C["blue"], lw=2, label="Batch GD")
    ax.plot([], [], [], color=C["orange"], lw=2, label="SGD")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        for p, col, lw in ((batch[: i + 1], C["blue"], 1.8), (sgd[: i * per + 1], C["orange"], 1.0)):
            zz = _J(p[:, 0], p[:, 1]) + 0.05
            s.add(ax.plot(p[:, 0], p[:, 1], zz, color=col, lw=lw))
            s.add(ax.scatter(p[-1:, 0], p[-1:, 1], zz[-1:], color=col, s=45, depthshade=False,
                             edgecolor="white"))
        s.status(f"batch steps: {i}    SGD updates: {i * per}")
        s.spin(i, n, sweep=45)

    return update


@animation("Normal Equations = projection",
           caption=r"""यहाँ axes तीन training examples हैं ($m=3$), इसलिए $\vec{y}$ 3D में एक vector है (काला)।
           नीला plane $X$ के दो columns का span है: $X\theta$ हमेशा इसी plane में रहता है। नारंगी बिंदु
           $X\theta$ plane पर चलते हुए उस जगह पहुँचता है जहाँ लाल residual $\vec{y} - X\theta$ सबसे छोटा है।
           उस पल residual plane के ठीक **perpendicular** हो जाता है: $X^T(\vec{y} - X\theta) = 0$। यही
           Normal Equation है।""",
           frames=40, keys=(0.0, 0.35, 1.0))
def projection(s, n):
    a1, a2 = np.array([1.0, 1.0, 1.0]), np.array([-1.0, 0.2, 1.4])
    A = np.c_[a1, a2]
    y = np.array([0.6, 2.6, 1.0])
    best = np.linalg.solve(A.T @ A, A.T @ y)
    ax = s.ax3d(labels=("example 1", "example 2", "example 3"), elev=20, azim=-50,
                xlim=(-2, 2.5), ylim=(-0.5, 3), zlim=(-0.5, 3), ticks=False)
    S, T = np.meshgrid(np.linspace(-0.4, 2.2, 8), np.linspace(-1.0, 1.4, 8))
    P = S[..., None] * a1 + T[..., None] * a2
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=C["blue"], alpha=0.22, shade=False)
    arrow(ax, (0, 0, 0), y, color=C["dark"], lw=2.2)
    ax.text(*(y * 1.06), r"$\vec{y}$", fontsize=11)
    ax.text(*(1.9 * a1 + 1.25 * a2), "span of X columns", color=C["blue"], fontsize=8, ha="center")
    start = np.array([1.9, -0.8])

    def update(i):
        s.clear()
        t = 1 - (1 - i / (n - 1)) ** 2.2
        th = start + t * (best - start)
        fit = A @ th
        r = y - fit
        done = i == n - 1
        s.add(arrow(ax, (0, 0, 0), fit, color=C["orange"], lw=2.0))
        s.add(ax.text(*(fit * 1.05 - np.array([0, 0, 0.25])), r"$X\theta$", color=C["orange"], fontsize=11))
        s.add(ax.plot(*np.c_[fit, y], color=C["green"] if done else C["red"], lw=2.2))
        g = A.T @ r
        s.status(rf"$\|\vec{{y}} - X\theta\|$ = {np.linalg.norm(r):.3f}     "
                 rf"$X^T(\vec{{y}} - X\theta)$ = ({g[0]:.2f}, {g[1]:.2f})")
        s.spin(i, n, sweep=35)

    return update
