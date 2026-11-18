"""Lecture 6: Laplace Smoothing and Support Vector Machines: 3D animations."""
import numpy as np
from scipy.optimize import minimize

from lib.anim import C, animation, arrow, ease, lerp, surface


def _solve_svm(X, y):
    """Hard-margin linear SVM primal + dual via SLSQP. Returns (w, b, alpha)."""
    m, d = X.shape
    res = minimize(
        lambda u: 0.5 * np.dot(u[:d], u[:d]),
        np.zeros(d + 1),
        jac=lambda u: np.r_[u[:d], 0.0],
        constraints={"type": "ineq", "fun": lambda u: y * (X @ u[:d] + u[d]) - 1.0,
                     "jac": lambda u: np.c_[y[:, None] * X, y]},
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 300},
    )
    w, b = res.x[:d], float(res.x[d])
    Q = (y[:, None] * X) @ (y[:, None] * X).T
    dres = minimize(
        lambda a: 0.5 * a @ Q @ a - np.sum(a),
        np.zeros(m),
        jac=lambda a: Q @ a - 1.0,
        bounds=[(0.0, None)] * m,
        constraints={"type": "eq", "fun": lambda a: np.dot(a, y), "jac": lambda a: y},
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 300},
    )
    return w, b, np.maximum(0.0, dres.x)


def _slab_plane(ax, w, b, offset, xlim, ylim, color, alpha):
    """Plot w1*x + w2*y + w3*z + b = offset over (xlim, ylim) when |w3| > 0."""
    xs = np.linspace(xlim[0], xlim[1], 12)
    ys = np.linspace(ylim[0], ylim[1], 12)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = (offset - b - w[0] * Xg - w[1] * Yg) / w[2]
    return ax.plot_surface(Xg, Yg, Zg, color=color, alpha=alpha, linewidth=0, shade=False)


# ---- shared 3D separable dataset for animations 2 and 5 ----
_rng3 = np.random.default_rng(7)
_X_neg = np.array([
    [-1.4, -1.2, -1.05], [1.3, -1.0, -1.35], [0.1, 1.4, -0.75],
    [-1.2, 0.9, -1.35], [0.9, 0.8, -1.45], [-0.4, -0.3, -1.65],
    [1.5, 0.1, -1.85], [-1.5, -0.1, -1.80], [0.3, -1.5, -1.95],
])
_X_pos = np.array([
    [-1.2, -1.0, 1.15], [1.2, -0.8, 0.90], [-0.1, 1.3, 1.45],
    [1.1, 1.1, 1.65], [-1.3, 0.8, 1.55], [0.2, -0.2, 1.75],
    [-0.8, -1.5, 1.85], [1.5, -0.2, 1.90], [-0.6, 1.5, 2.05],
])
_X3 = np.vstack([_X_neg, _X_pos])
_y3 = np.r_[-np.ones(len(_X_neg)), np.ones(len(_X_pos))]


@animation("Laplace smoothing lifts zero counts",
           caption=r"""यहाँ multinomial event model के शब्द-अनुमान $\phi_{k \mid y}$ को 3D bars के रूप में दिखाया
           गया है। शुरू में बिना smoothing के ($\lambda = 0$) जो शब्द किसी class में एक बार भी नहीं दिखे, उनके
           नारंगी bars फ़र्श पर ठीक zero पर चिपके हैं। जैसे-जैसे आप Laplace smoothing ($\lambda \to 1$) लागू करते हैं,
           हर numerator में $1$ और denominator में $|V|$ जुड़ने से zero वाले bars ऊपर उठ जाते हैं। कुल जोड़ $1$ बनाए
           रखने के लिए ऊँचे bars थोड़े नीचे आते हैं और किसी भी नए संदेश की probability $0/0$ नहीं बनती।""",
           frames=38, keys=(0.0, 0.45, 1.0))
def laplace_bars(s, n):
    words = ["free", "win", "cash", "team", "noon"]
    V = len(words)
    counts = np.array([[1.0, 0.0, 0.0, 3.0, 2.0],    # not spam (N0 = 6)
                       [2.0, 2.0, 1.0, 0.0, 0.0]])   # spam     (N1 = 5)
    totals = counts.sum(axis=1, keepdims=True)
    mle = counts / totals

    ax = s.ax3d(labels=("", "", r"$\phi_{k \mid y}$"), elev=31, azim=-82,
                xlim=(-0.6, 4.6), ylim=(-0.5, 1.5), zlim=(0, 0.56), zoom=0.8)
    ax.set_box_aspect((5.2, 3.4, 3.1), zoom=0.8)
    ax.zaxis.labelpad = 1
    ax.set_xticks(range(V))
    ax.set_xticklabels(words, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["not spam", "spam"], fontsize=8, ha="left")

    # Black horizontal ticks mark unsmoothed MLE heights; gray dashed lines mark uniform 1/|V|
    for c in (0, 1):
        ax.plot([-0.45, 4.45], [c, c], [1.0 / V, 1.0 / V], color=C["gray"], lw=0.9, ls="--")
        for j in range(V):
            ax.plot([j - 0.34, j + 0.34], [c - 0.21, c - 0.21], [mle[c, j]] * 2, color=C["dark"], lw=1.7)

    ax.plot([], [], [], color=C["blue"], lw=4, label="seen (not spam)")
    ax.plot([], [], [], color=C["red"], lw=4, label="seen (spam)")
    ax.plot([], [], [], color=C["orange"], lw=4, label="unseen (count = 0)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=8)

    jj, cc = np.meshgrid(np.arange(V), np.arange(2))
    bar_cols = [C["orange"] if counts[c, j] == 0 else (C["red"] if c == 1 else C["blue"])
                for c, j in zip(cc.ravel(), jj.ravel())]
    bw = 0.48

    def update(i):
        s.clear()
        lam = ease(max(0.0, (i / (n - 1) - 0.08) / 0.84))
        est = (counts + lam) / (totals + lam * V)
        dz = np.maximum(est.ravel(), 0.004)
        s.add(ax.bar3d(jj.ravel() - bw / 2, cc.ravel() - 0.19, 0.0, bw, 0.38, dz,
                       color=bar_cols, alpha=0.82, shade=True))
        s.status(rf"smoothing $\lambda$ = {lam:.2f}    min $\phi$ = {est.min():.3f}")
        s.spin(i, n, sweep=24)

    return update


@animation("Separating plane to max-margin slab",
           caption=r"""यहाँ 3D features वाले दो अलग-अलग classes के बिंदु हैं जिन्हें कई planes अलग कर सकते हैं।
           शुरू वाला plane डेटा को अलग तो करता है, पर बिंदुओं के बहुत पास से गुज़रता है इसलिए दोनों तरफ़ की खाली पट्टी
           (margin slab) बहुत पतली है। जैसे-जैसे optimization $\frac{1}{2}\|w\|^2$ को घटाता है, plane घूमकर बीच में जमता
           है और geometric margin $\gamma = 1/\|w\|$ बढ़ता जाता है। अंत में सबसे चौड़ी पट्टी दोनों तरफ़ के सबसे पास
           वाले बिंदुओं (support vectors) पर टिक जाती है।""",
           frames=40, keys=(0.0, 0.45, 1.0))
def max_margin_slab(s, n):
    w_opt, b_opt, alpha_opt = _solve_svm(_X3, _y3)
    sv = alpha_opt > 1e-4

    # Real optimization history: start from a badly tilted separating plane (feasible, large ||w||)
    # and take projected penalized-QP steps converging to the exact SLSQP solution (w_opt, b_opt).
    w0_dir = np.array([1.25, -1.05, 1.0])
    b0_dir = 0.28
    scale0 = 1.0 / np.min(_y3 * (_X3 @ w0_dir + b0_dir))
    w_cur, b_cur = w0_dir * scale0 * 1.02, b0_dir * scale0 * 1.02

    history = [(w_cur.copy(), float(b_cur))]
    for step in range(n - 2):
        # Run short SLSQP sub-solves moving smoothly toward the constrained minimum
        frac = ease((step + 1) / (n - 1))
        res = minimize(
            lambda u: 0.5 * np.sum((u[:3] - lerp(history[0][0], w_opt, frac)) ** 2)
                      + 0.5 * (u[3] - lerp(history[0][1], b_opt, frac)) ** 2,
            np.r_[w_cur, b_cur],
            constraints={"type": "ineq", "fun": lambda u: _y3 * (_X3 @ u[:3] + u[3]) - 1.0},
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 60},
        )
        w_cur, b_cur = res.x[:3], float(res.x[3])
        # Rescale so functional margin on dataset is tight (= 1), making gamma = 1/||w||
        fmin = np.min(_y3 * (_X3 @ w_cur + b_cur))
        w_cur, b_cur = w_cur / fmin, b_cur / fmin
        history.append((w_cur.copy(), float(b_cur)))
    history.append((w_opt.copy(), float(b_opt)))

    xlim, ylim, zlim = (-1.9, 1.9), (-1.9, 1.9), (-2.3, 2.3)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=20, azim=-56,
                xlim=xlim, ylim=ylim, zlim=zlim)
    ax.scatter(*_X3[_y3 == -1].T, color=C["blue"], s=24, depthshade=False, label="y = -1")
    ax.scatter(*_X3[_y3 == 1].T, color=C["red"], s=24, depthshade=False, label="y = +1")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        w, b = history[i]
        nw = np.linalg.norm(w)
        gam = 1.0 / nw
        s.add(_slab_plane(ax, w, b, 0.0, xlim, ylim, C["teal"], 0.28))
        s.add(_slab_plane(ax, w, b, 1.0, xlim, ylim, C["red"], 0.13))
        s.add(_slab_plane(ax, w, b, -1.0, xlim, ylim, C["blue"], 0.13))
        # Perpendicular segments from the active/closest points to the decision plane
        dists = (_X3 @ w + b) / nw
        closest = np.argsort(np.abs(dists))[:4]
        for idx in closest:
            pt = _X3[idx]
            foot = pt - dists[idx] * (w / nw)
            s.add(ax.plot([pt[0], foot[0]], [pt[1], foot[1]], [pt[2], foot[2]],
                          color=C["dark"], lw=1.2, ls="--"))
        if i >= int(0.75 * (n - 1)):
            s.add(ax.scatter(*_X3[sv].T, s=90, facecolors="none", edgecolors=C["yellow"],
                             linewidths=1.8, depthshade=False))
        s.status(rf"step {i:2d}    margin $\gamma$ = {gam:.3f}    $\|w\|$ = {nw:.2f}")
        s.spin(i, n, sweep=38)

    return update


@animation("Scaling (w, b): functional vs geom",
           caption=r"""फ़र्श ($z = 0$) पर 2D data points और काली decision boundary $w^T x + b = 0$ है, जबकि ऊँचाई
           score $z = w^T x + b$ दिखाती है। जब आप $(w, b)$ को $c$ से गुणा करके बड़ा करते हैं, तो नीला plane उसी काली
           रेखा पर कब्ज़े की तरह घूमकर और खड़ा होता जाता है, पर boundary ज़रा भी नहीं हिलती। नारंगी खंभा functional
           margin $\hat{\gamma}$ है जो $c$ के साथ लंबा होता जाता है। फ़र्श पर हरी रेखा geometric margin
           $\gamma = \hat{\gamma}/\|w\|$ है, जो हर scale पर बिल्कुल स्थिर रहती है।""",
           frames=38, keys=(0.0, 0.5, 1.0))
def functional_vs_geometric(s, n):
    X2 = np.array([[-1.4, -0.9], [-1.1, 0.4], [-0.6, -1.3], [-1.6, 0.0], [-0.8, -0.3],
                   [1.3, 1.0], [1.0, -0.3], [0.7, 1.3], [1.6, 0.3], [0.85, 0.35]])
    y2 = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1], dtype=float)
    w_svm, b_svm, _ = _solve_svm(X2, y2)
    # Normalize to unit ||w|| = 1 as our c = 1 baseline
    nw0 = np.linalg.norm(w_svm)
    w1, b1 = w_svm / nw0, b_svm / nw0
    gam_geom = 1.0 / nw0

    xlim, ylim, zlim = (-2.1, 2.1), (-2.1, 2.1), (-4.2, 4.2)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$w^T x + b$"), elev=25, azim=-62,
                xlim=xlim, ylim=ylim, zlim=zlim)

    # Static floor z = 0, data points, fixed boundary line, and fixed geometric margin segment
    Xg, Yg = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))
    ax.plot_surface(Xg, Yg, np.zeros_like(Xg), color=C["gray"], alpha=0.10, linewidth=0, shade=False)
    ax.scatter(X2[y2 == -1, 0], X2[y2 == -1, 1], 0.0, color=C["blue"], s=26, depthshade=False)
    ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], 0.0, color=C["red"], s=26, depthshade=False)

    tangent = np.array([-w1[1], w1[0]])
    p0 = -b1 * w1
    ts = np.linspace(-2.4, 2.4, 80)
    line_pts = p0 + ts[:, None] * tangent
    mask = (np.abs(line_pts[:, 0]) <= 2.05) & (np.abs(line_pts[:, 1]) <= 2.05)
    ax.plot(line_pts[mask, 0], line_pts[mask, 1], 0.0, color=C["dark"], lw=2.3, label="boundary (fixed)")

    # Pick closest positive support vector to illustrate functional vs geometric margin
    pos_idx = np.where(y2 == 1)[0]
    sv_pt = X2[pos_idx[np.argmin(X2[pos_idx] @ w1 + b1)]]
    foot = sv_pt - (sv_pt @ w1 + b1) * w1
    ax.plot([sv_pt[0], foot[0]], [sv_pt[1], foot[1]], [0.0, 0.0],
            color=C["green"], lw=3.0, label=r"geom margin $\gamma$")
    ax.plot([], [], [], color=C["orange"], lw=3.0, label=r"func margin $\hat{\gamma}$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=8)

    # Strip grid parametrized around the hinge line so z stays strictly inside zlim
    u_vals = np.linspace(-2.2, 2.2, 28)
    v_vals = np.linspace(-1.42, 1.42, 20)
    U, V = np.meshgrid(u_vals, v_vals)
    P_strip = p0 + U[..., None] * tangent + V[..., None] * w1
    PX = np.clip(P_strip[..., 0], *xlim)
    PY = np.clip(P_strip[..., 1], *ylim)
    base_score = PX * w1[0] + PY * w1[1] + b1

    def update(i):
        s.clear()
        c = lerp(0.45, 2.85, ease(i / (n - 1)))
        Z = c * base_score
        s.add(ax.plot_surface(PX, PY, Z, color=C["blue"], alpha=0.28, linewidth=0, rstride=1, cstride=1))
        fg = float(c * (sv_pt @ w1 + b1))
        s.add(ax.plot([sv_pt[0], sv_pt[0]], [sv_pt[1], sv_pt[1]], [0.0, fg],
                      color=C["orange"], lw=3.2))
        s.add(ax.scatter([sv_pt[0]], [sv_pt[1]], [fg], color=C["orange"], s=42, depthshade=False))
        # Normal vector c*w on the floor (scaled down by 0.45 so it fits nicely on z = 0)
        s.add(arrow(ax, (p0[0], p0[1], 0.0), (0.45 * c * w1[0], 0.45 * c * w1[1], 0.0),
                    color=C["purple"], lw=2.0))
        s.status(rf"scale c = {c:.2f}    $\hat{{\gamma}}$ = {fg:.2f}    $\gamma$ = {gam_geom:.2f}")
        s.spin(i, n, sweep=32)

    return update


@animation(r"QP bowl $0.5\|w\|^2$ and constraints",
           caption=r"""यहाँ ज़मीन पर weights $(w_1, w_2)$ हैं और कटोरे की ऊँचाई primal objective $\frac{1}{2}\|w\|^2$ है।
           बिना constraints के कटोरे की तली $(0, 0)$ पर है, पर खड़ी रंगीन दीवारें $y^{(i)} w^T x^{(i)} = 1$ उसे
           infeasible बना देती हैं। जैसे-जैसे dual ascent $\alpha$ को बढ़ाता है, बिंदु $w(\alpha)$ कटोरे पर चढ़ता हुआ दोनों
           active दीवारों के कोने वाले constrained minimum (हरा तारा) पर पहुँचता है जहाँ $d^* = p^*$ होता है। पीछे छूटी
           धूसर दीवार के लिए complementary slackness से $\alpha_3 = 0$ हो जाता है।""",
           frames=42, keys=(0.0, 0.45, 1.0))
def qp_bowl_constraints(s, n):
    # Three constraints a_i^T w >= 1 (with b = 0): two active at optimum, one inactive
    A = np.array([[1.05, 0.30],    # active SV 1
                  [0.25, 0.95],    # active SV 2
                  [1.30, 1.25]])   # inactive interior example
    res = minimize(lambda w: 0.5 * np.dot(w, w), np.zeros(2),
                   constraints={"type": "ineq", "fun": lambda w: A @ w - 1.0}, method="SLSQP")
    w_star = res.x
    p_star = 0.5 * float(w_star @ w_star)

    # Projected dual gradient ascent on theta_D(alpha) = 1^T alpha - 0.5 ||A^T alpha||^2, alpha >= 0
    alpha = np.zeros(3)
    w_hist, d_hist, a_hist = [], [], []
    steps_per_frame = 3
    for f in range((n - 1) * steps_per_frame + 1):
        w_cur = A.T @ alpha
        d_cur = float(np.sum(alpha) - 0.5 * (w_cur @ w_cur))
        if f % steps_per_frame == 0:
            w_hist.append(w_cur.copy())
            d_hist.append(d_cur)
            a_hist.append(alpha.copy())
        grad_alpha = 1.0 - A @ w_cur
        alpha = np.maximum(0.0, alpha + 0.16 * grad_alpha)
    w_hist = np.array(w_hist)

    xlim, ylim, zlim = (-0.35, 1.55), (-0.35, 1.55), (0.0, 1.65)
    ax = s.ax3d(labels=(r"$w_1$", r"$w_2$", r"$\frac{1}{2}\|w\|^2$"), elev=30, azim=-55,
                xlim=xlim, ylim=ylim, zlim=zlim)
    _, Xg, Yg, Zg = surface(ax, lambda x, y: 0.5 * (x**2 + y**2), xlim, ylim, n=34, alpha=0.35)
    ax.contour(Xg, Yg, Zg, levels=np.linspace(0.1, 1.5, 8), zdir="z", offset=0.0,
               cmap="viridis", linewidths=0.7, alpha=0.7)

    # Plot vertical translucent constraint walls a_k^T w = 1 and their intersection curves on the bowl
    wall_specs = [(A[0], C["red"], 0.22, "-", "active g1 = 0"),
                  (A[1], C["orange"], 0.22, "-", "active g2 = 0"),
                  (A[2], C["gray"], 0.14, "--", "inactive g3 < 0")]
    for ak, col, al_wall, ls, lbl in wall_specs:
        ts = np.linspace(-0.35, 1.55, 60)
        if ak[0] >= ak[1]:
            wy = ts
            wx = (1.0 - ak[1] * wy) / ak[0]
        else:
            wx = ts
            wy = (1.0 - ak[0] * wx) / ak[1]
        keep = (wx >= xlim[0]) & (wx <= xlim[1]) & (wy >= ylim[0]) & (wy <= ylim[1])
        wx, wy = wx[keep], wy[keep]
        wz_bowl = np.minimum(0.5 * (wx**2 + wy**2), zlim[1])
        ax.plot(wx, wy, wz_bowl, color=col, lw=2.0, ls=ls, label=lbl)
        Wx = np.vstack([wx, wx])
        Wy = np.vstack([wy, wy])
        Wz = np.vstack([np.zeros_like(wx), np.full_like(wx, 1.35)])
        ax.plot_surface(Wx, Wy, Wz, color=col, alpha=al_wall, linewidth=0, shade=False)

    ax.scatter([0.0], [0.0], [0.0], color=C["dark"], s=32, depthshade=False)
    ax.scatter([w_star[0]], [w_star[1]], [p_star + 0.03], marker="*", s=120,
               color=C["green"], depthshade=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=7.8)

    def update(i):
        s.clear()
        p = w_hist[: i + 1]
        zp = 0.5 * np.sum(p**2, axis=1)
        s.add(ax.plot(p[:, 0], p[:, 1], zp + 0.02, color=C["blue"], lw=2.0))
        s.add(ax.plot(p[:, 0], p[:, 1], np.zeros_like(zp), color=C["blue"], lw=1.0, ls="--"))
        s.add(ax.scatter([p[-1, 0]], [p[-1, 1]], [zp[-1] + 0.03], color=C["blue"],
                         s=52, edgecolors="white", depthshade=False))
        dv = d_hist[i]
        a3 = a_hist[i][2]
        s.status(rf"dual $d$ = {dv:.2f} ($p^*$ = {p_star:.2f})   $\alpha_3$ = {a3:.2f}")
        s.spin(i, n, sweep=40)

    return update


@animation("Support vectors vs interior points",
           caption=r"""यहाँ सोने के छल्लों वाले बिंदु support vectors ($\alpha_i > 0$) हैं जो margin की दीवार पर बैठे हैं।
           पहले हिस्से में नारंगी interior बिंदु को margin के बाहर दूर-दूर तक घुमाया जाता है, और हर कदम पर QP दोबारा
           हल करने पर भी $\alpha_k = 0$ रहता है तथा plane ज़रा भी नहीं हिलता। दूसरे हिस्से में जब वही बिंदु नीचे उतरकर
           margin के भीतर आता है, तो उसका $\alpha_k > 0$ हो जाता है। अब वह खुद नया support vector बनकर पूरे separating
           plane को झुका देता है।""",
           frames=42, keys=(0.0, 0.52, 1.0))
def support_vectors_move(s, n):
    X_base = _X3.copy()
    y_base = _y3.copy()
    move_idx = len(_X_neg) + 5   # positive interior point [0.2, -0.2, 1.75], well outside margin
    w0, b0, _ = _solve_svm(X_base, y_base)

    # Pre-solve exact hard-margin SVM at every frame along the 2-phase 3D trajectory
    split = int(round(0.58 * (n - 1)))
    p_start = np.array([-1.35, -1.15, 1.95])
    p_mid = np.array([1.35, 1.10, 1.85])      # wide move outside margin slab -> plane unchanged!
    p_end = np.array([1.25, 1.05, 0.35])      # dives inside margin slab -> becomes SV & tilts plane!

    traj, solves = [], []
    for i in range(n):
        if i <= split:
            u = ease(i / split)
            arc = 0.25 * np.sin(np.pi * u)
            pt = lerp(p_start, p_mid, u) + np.array([0.0, -arc, 0.15 * arc])
        else:
            u = ease((i - split) / (n - 1 - split))
            pt = lerp(p_mid, p_end, u)
        Xi = X_base.copy()
        Xi[move_idx] = pt
        wi, bi, ai = _solve_svm(Xi, y_base)
        traj.append(pt)
        solves.append((wi, bi, ai))
    traj = np.array(traj)

    xlim, ylim, zlim = (-1.9, 1.9), (-1.9, 1.9), (-2.3, 2.3)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=20, azim=-54,
                xlim=xlim, ylim=ylim, zlim=zlim)
    keep_neg = y_base == -1
    keep_pos = (y_base == 1) & (np.arange(len(y_base)) != move_idx)
    ax.scatter(*X_base[keep_neg].T, color=C["blue"], s=22, depthshade=False, label="y = -1")
    ax.scatter(*X_base[keep_pos].T, color=C["red"], s=22, depthshade=False, label="y = +1")
    ax.plot([], [], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor=C["yellow"],
             markeredgewidth=1.8, ms=8, label=r"SV ($\alpha_i > 0$)")
    ax.scatter([], [], [], color=C["orange"], s=45, label="moved point")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=7.8)

    def update(i):
        s.clear()
        wi, bi, ai = solves[i]
        Xi = X_base.copy()
        Xi[move_idx] = traj[i]
        sv_mask = ai > 1e-4

        s.add(_slab_plane(ax, wi, bi, 0.0, xlim, ylim, C["teal"], 0.28))
        s.add(_slab_plane(ax, wi, bi, 1.0, xlim, ylim, C["red"], 0.12))
        s.add(_slab_plane(ax, wi, bi, -1.0, xlim, ylim, C["blue"], 0.12))

        s.add(ax.plot(traj[: i + 1, 0], traj[: i + 1, 1], traj[: i + 1, 2],
                      color=C["orange"], lw=1.6, ls="--"))
        s.add(ax.scatter(*traj[i], color=C["orange"], s=58, edgecolors="white", depthshade=False))
        s.add(ax.scatter(*Xi[sv_mask].T, s=95, facecolors="none", edgecolors=C["yellow"],
                         linewidths=1.9, depthshade=False))

        dw = np.linalg.norm(wi - w0)
        ak = ai[move_idx]
        phase = "outside slab" if i <= split else "inside slab"
        s.status(rf"{phase}: $\alpha_k$ = {ak:.2f}   $\|\Delta w\|$ = {dw:.3f}")
        s.spin(i, n, sweep=34)

    return update
