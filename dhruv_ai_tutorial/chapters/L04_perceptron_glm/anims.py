"""Lecture 4: Perceptron, Exponential Family & GLMs: 3D animations."""
import numpy as np

from lib.anim import C, animation, ease, grid, surface


# ---------------------------------------------------------------------------
# 1. Perceptron plane jumping on misclassified points
# ---------------------------------------------------------------------------
@animation(
    "Perceptron Plane Jumps on Mistakes",
    caption=r"""ज़मीन ($z=0$) पर नीले बिंदु ($y=0$) और लाल त्रिकोण ($y=1$) दो linearly separable classes हैं।
    झुका हुआ plane वर्तमान linear score $z = \theta^T x$ दिखाता है और काली रेखा decision boundary $\theta^T x = 0$ है।
    Perceptron हर कदम पर केवल गलत वर्गीकृत बिंदु (नारंगी घेरा) को देखकर $\theta := \theta + \alpha(y - h_\theta(x))x$ से अचानक छलाँग लगाता है।
    जैसे ही गलतियाँ शून्य होती हैं, हरा plane स्थिर होकर रुक जाता है।""",
    frames=38,
    keys=(0.0, 0.45, 1.0),
)
def perceptron_jumps(s, n):
    rng = np.random.default_rng(4)
    m_half = 13
    x0 = rng.normal(loc=[-1.05, -0.85], scale=0.48, size=(m_half, 2))
    x1 = rng.normal(loc=[1.05, 0.85], scale=0.48, size=(m_half, 2))
    pts = np.vstack([x0, x1])
    y = np.r_[np.zeros(m_half, dtype=int), np.ones(m_half, dtype=int)]
    X = np.c_[np.ones(len(y)), pts]

    # Run real Perceptron learning from an initially reversed plane
    th = np.array([0.25, -1.25, 0.95], dtype=float)
    alpha = 0.55
    events = []  # (theta, n_mistakes, active_idx_or_None, step_num)
    upd = 0
    for _ in range(15):
        preds_all = (X @ th >= 0).astype(int)
        n_err = int(np.sum(preds_all != y))
        if n_err == 0:
            events.append((th.copy(), 0, None, upd))
            break
        for j in range(len(y)):
            pred_j = int(X[j] @ th >= 0)
            if pred_j != y[j]:
                cur_err = int(np.sum((X @ th >= 0).astype(int) != y))
                events.append((th.copy(), cur_err, j, upd))
                th = th + alpha * (y[j] - pred_j) * X[j]
                upd += 1
                new_err = int(np.sum((X @ th >= 0).astype(int) != y))
                if new_err == 0:
                    break

    # Map events across frames so each mistake spotlight + jump is clearly visible
    total_u = upd
    schedule = []
    n_conv = 6
    n_active = n - n_conv
    for f in range(n_active):
        idx = min(len(events) - 2, int(f * max(1, len(events) - 1) / n_active))
        schedule.append(events[idx])
    for _ in range(n_conv):
        schedule.append(events[-1])

    ax = s.ax3d(
        labels=(r"$x_1$", r"$x_2$", r"$\theta^T x$"),
        elev=25,
        azim=-56,
        xlim=(-2.3, 2.3),
        ylim=(-2.3, 2.3),
        zlim=(-3.8, 3.8),
    )
    Xg, Yg = grid((-2.3, 2.3), (-2.3, 2.3), n=26)
    ax.plot_surface(Xg, Yg, np.zeros_like(Xg), color=C["gray"], alpha=0.14, linewidth=0, shade=False)
    ax.scatter(x0[:, 0], x0[:, 1], np.zeros(m_half), color=C["blue"], s=26, marker="o",
               depthshade=False, label="y = 0")
    ax.scatter(x1[:, 0], x1[:, 1], np.zeros(m_half), color=C["red"], s=30, marker="^",
               depthshade=False, label="y = 1")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        cur_th, n_err, bad_idx, u_cnt = schedule[i]
        done = n_err == 0
        Zp = np.clip(cur_th[0] + cur_th[1] * Xg + cur_th[2] * Yg, -3.8, 3.8)
        col = C["green"] if done else C["purple"]
        s.add(ax.plot_surface(Xg, Yg, Zp, color=col, alpha=0.33, linewidth=0, antialiased=True))

        # Intersection line theta^T x = 0 on z = 0 floor
        ts = np.linspace(-2.3, 2.3, 80)
        if abs(cur_th[2]) >= abs(cur_th[1]) and abs(cur_th[2]) > 1e-6:
            bx, by = ts, -(cur_th[0] + cur_th[1] * ts) / cur_th[2]
        else:
            by, bx = ts, -(cur_th[0] + cur_th[2] * ts) / (cur_th[1] + 1e-9)
        keep = (np.abs(bx) <= 2.3) & (np.abs(by) <= 2.3)
        if np.any(keep):
            s.add(ax.plot(bx[keep], by[keep], np.zeros(keep.sum()), color=C["dark"], lw=2.2))

        # Error stems on misclassified points + spotlight on active mistake
        scores = np.clip(X @ cur_th, -3.8, 3.8)
        wrong = ((X @ cur_th >= 0).astype(int) != y)
        if np.any(wrong):
            nan = np.full(wrong.sum(), np.nan)
            wx, wy, wz = pts[wrong, 0], pts[wrong, 1], scores[wrong]
            s.add(ax.plot(np.c_[wx, wx, nan].ravel(), np.c_[wy, wy, nan].ravel(),
                          np.c_[np.zeros_like(wz), wz, nan].ravel(), color=C["orange"], lw=1.2))
        if bad_idx is not None:
            px, py = pts[bad_idx]
            s.add(ax.scatter([px], [py], [0.0], color=C["orange"], s=115, facecolors="none",
                             linewidths=2.2, depthshade=False))

        s.status(f"update {u_cnt}/{total_u}   mistakes: {n_err}")
        s.spin(i, n, sweep=36)

    return update


# ---------------------------------------------------------------------------
# 2. Exponential-family density over (y, eta)
# ---------------------------------------------------------------------------
@animation(
    r"Exponential Family $p(y;\eta)$",
    caption=r"""यहाँ $(y, \eta)$ तल के ऊपर Exponential Family के Gaussian सदस्य $p(y;\eta) = b(y)\exp(\eta y - a(\eta))$ की 3D density दिखाई गई है।
    नीली टूटी रेखा $\eta = 0$ पर base measure $b(y)$ है।
    जैसे-जैसे लाल काट (slice) $\eta$ को बढ़ाती है, $\exp(\eta y)$ वितरण को दाईं ओर झुकाता है और $a(\eta) = \frac{1}{2}\eta^2$ कुल क्षेत्रफल को $1$ बनाए रखता है।
    शिखर हमेशा हरी रेखा $E[y;\eta] = a'(\eta) = \eta$ पर चलता है।""",
    frames=40,
    keys=(0.0, 0.5, 1.0),
)
def exp_family_density(s, n):
    ylim_v = (-3.4, 4.4)
    etalim = (-1.6, 2.2)

    def dens(yv, etav):
        # b(y) * exp(eta * y - a(eta)) with b(y) = exp(-y^2/2)/sqrt(2*pi), a(eta) = eta^2/2
        b_y = np.exp(-0.5 * yv**2) / np.sqrt(2.0 * np.pi)
        return b_y * np.exp(etav * yv - 0.5 * etav**2)

    ax = s.ax3d(
        labels=(r"$y$", r"$\eta$", r"$p(y;\eta)$"),
        elev=28,
        azim=-55,
        xlim=ylim_v,
        ylim=etalim,
        zlim=(0.0, 0.45),
    )
    surface(ax, dens, ylim_v, etalim, n=36, cmap="viridis", alpha=0.38)

    ys = np.linspace(ylim_v[0], ylim_v[1], 80)
    ax.plot(ys, np.zeros_like(ys), dens(ys, 0.0), color=C["blue"], ls="--", lw=1.6, label=r"$b(y)$ at $\eta=0$")
    etas_line = np.linspace(etalim[0], etalim[1], 50)
    ax.plot(etas_line, etas_line, np.zeros_like(etas_line), color=C["green"], ls="--", lw=1.4,
            label=r"$E[y]=a'(\eta)$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    ys_curtain = np.linspace(ylim_v[0], ylim_v[1], 28)

    def update(i):
        s.clear()
        eta_i = etalim[0] + (etalim[1] - etalim[0]) * (i / (n - 1))
        pz = dens(ys, eta_i)
        s.add(ax.plot(ys, np.full_like(ys, eta_i), pz, color=C["red"], lw=2.4))

        # Curtain drop lines under current conditional density p(y; eta_i)
        pc = dens(ys_curtain, eta_i)
        nan = np.full_like(ys_curtain, np.nan)
        ec = np.full_like(ys_curtain, eta_i)
        s.add(ax.plot(np.c_[ys_curtain, ys_curtain, nan].ravel(), np.c_[ec, ec, nan].ravel(),
                      np.c_[np.zeros_like(pc), pc, nan].ravel(), color=C["red"], lw=0.75, alpha=0.5))

        # Mean E[y] = a'(eta) = eta_i at ridge peak
        mu_i = eta_i
        peak = float(dens(mu_i, eta_i))
        s.add(ax.plot([mu_i, mu_i], [eta_i, eta_i], [0.0, peak], color=C["green"], lw=1.8))
        s.add(ax.scatter([mu_i], [eta_i], [peak], color=C["green"], s=50, edgecolor="white", depthshade=False))

        s.status(rf"$\eta$={eta_i:+.2f}   $a(\eta)$={0.5 * eta_i**2:.2f}   $E[y]$={mu_i:+.2f}")
        s.spin(i, n, sweep=38)

    return update


# ---------------------------------------------------------------------------
# 3. GLM pipeline morphing linear plane to canonical mean surface
# ---------------------------------------------------------------------------
@animation(
    "GLM: Plane Morphs to Mean Surface",
    caption=r"""GLM दो चरणों में काम करता है: पहले features से सपाट linear plane $\eta = \theta^T x$ बनता है (धूसर जाली), फिर canonical response $g(\eta) = a'(\eta)$ उसे मोड़कर mean surface $E[y\mid x]$ बनाता है।
    यहाँ Bernoulli GLM के लिए सपाट plane धीरे-धीरे मोड़ खाकर $[0, 1]$ के बीच बंधी sigmoid surface में बदल रहा है।
    नारंगी रेखाएँ दिखाती हैं कि कैसे बड़े धनात्मक $\eta$ नीचे $1$ पर और ऋणात्मक $\eta$ ऊपर $0$ पर दब जाते हैं।""",
    frames=40,
    keys=(0.0, 0.5, 1.0),
)
def glm_plane_to_mean(s, n):
    rng = np.random.default_rng(7)
    m = 36
    x1 = rng.uniform(-2.0, 2.0, m)
    x2 = rng.uniform(-2.0, 2.0, m)
    X = np.c_[np.ones(m), x1, x2]
    true_th = np.array([0.15, 1.35, -1.05])
    probs = 1.0 / (1.0 + np.exp(-(X @ true_th)))
    y = (rng.random(m) < probs).astype(float)

    # Fit actual Bernoulli GLM (Logistic Regression via Newton steps)
    th = np.zeros(3)
    for _ in range(8):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ th, -20, 20)))
        grad = X.T @ (p - y)
        H = (X * (p * (1.0 - p))[:, None]).T @ X + 1e-4 * np.eye(3)
        th = th - np.linalg.solve(H, grad)

    ax = s.ax3d(
        labels=(r"$x_1$", r"$x_2$", r"$\eta \to g(\eta)$"),
        elev=23,
        azim=-62,
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        zlim=(-3.2, 3.2),
    )
    surface(ax, lambda a, b: np.clip(th[0] + th[1] * a + th[2] * b, -3.2, 3.2),
            (-2.0, 2.0), (-2.0, 2.0), n=22, wire=True, alpha=0.28, color=C["gray"])
    ax.scatter(x1[y == 0], x2[y == 0], y[y == 0], color=C["blue"], s=18, depthshade=False, label="y = 0")
    ax.scatter(x1[y == 1], x2[y == 1], y[y == 1], color=C["red"], s=18, depthshade=False, label="y = 1")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    Xg, Yg = grid((-2.0, 2.0), (-2.0, 2.0), n=32)
    Eta = np.clip(th[0] + th[1] * Xg + th[2] * Yg, -3.2, 3.2)
    Mean = 1.0 / (1.0 + np.exp(-(th[0] + th[1] * Xg + th[2] * Yg)))

    ax_pts, ay_pts = np.meshgrid([-1.4, 0.0, 1.4], [-1.4, 0.0, 1.4])
    ax_pts, ay_pts = ax_pts.ravel(), ay_pts.ravel()
    eta_pts = np.clip(th[0] + th[1] * ax_pts + th[2] * ay_pts, -3.2, 3.2)
    mu_pts = 1.0 / (1.0 + np.exp(-(th[0] + th[1] * ax_pts + th[2] * ay_pts)))

    def update(i):
        s.clear()
        t = float(ease(i / (n - 1)))
        Zt = (1.0 - t) * Eta + t * Mean
        s.add(ax.plot_surface(Xg, Yg, Zt, cmap="coolwarm", alpha=0.44, linewidth=0, antialiased=True))

        zt_pts = (1.0 - t) * eta_pts + t * mu_pts
        nan = np.full_like(ax_pts, np.nan)
        s.add(ax.plot(np.c_[ax_pts, ax_pts, nan].ravel(), np.c_[ay_pts, ay_pts, nan].ravel(),
                      np.c_[eta_pts, zt_pts, nan].ravel(), color=C["orange"], lw=1.2, alpha=0.85))
        s.status(rf"morph $t$ = {t:.2f}:  $(1-t)\eta + t\,g(\eta)$")
        s.spin(i, n, sweep=40)

    return update


# ---------------------------------------------------------------------------
# 4. Poisson regression surface fitting count data
# ---------------------------------------------------------------------------
@animation(
    r"Poisson GLM: $\exp(\theta^T x)$",
    caption=r"""गिनती वाले data $y \in \{0, 1, 2, \dots\}$ (काले बिंदु) पर Poisson GLM fit हो रहा है।
    शुरू में $\theta = 0$ है, इसलिए अनुमानित दर $\hat{\lambda}(x) = \exp(0) = 1$ की सपाट चादर है।
    Gradient Ascent चलते ही surface $\hat{\lambda}(x) = \exp(\theta^T x)$ घातांकीय रूप से ऊपर उठती है, हमेशा धनात्मक रहती है, और residuals छोटे होते जाते हैं।""",
    frames=40,
    keys=(0.0, 0.45, 1.0),
)
def poisson_surface(s, n):
    rng = np.random.default_rng(12)
    m = 44
    x1 = rng.uniform(-1.5, 1.5, m)
    x2 = rng.uniform(-1.5, 1.5, m)
    X = np.c_[np.ones(m), x1, x2]
    lam_true = np.exp(0.75 + 0.72 * x1 + 0.52 * x2)
    y = rng.poisson(lam_true).astype(float)

    # Run real Gradient Ascent on Poisson log-likelihood
    th = np.zeros(3)
    sub = 4
    alpha = 0.045
    thetas = [th.copy()]
    for _ in range(n - 1):
        for _ in range(sub):
            lam = np.exp(np.clip(X @ th, -6, 4))
            th = th + alpha * (X.T @ (y - lam)) / m
        thetas.append(th.copy())

    from scipy.special import gammaln

    log_y_fact = gammaln(y + 1.0)
    zmax = max(13.0, float(y.max()) + 1.0)
    ax = s.ax3d(
        labels=(r"$x_1$", r"$x_2$", r"count $y, \hat{\lambda}$"),
        elev=24,
        azim=-132,
        xlim=(-1.6, 1.6),
        ylim=(-1.6, 1.6),
        zlim=(0.0, zmax),
    )
    ax.scatter(x1, x2, y, color=C["dark"], s=20, depthshade=False, label=r"counts $y^{(i)}$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    Xg, Yg = grid((-1.55, 1.55), (-1.55, 1.55), n=32)

    def update(i):
        s.clear()
        cur_th = thetas[i]
        Zlam = np.clip(np.exp(cur_th[0] + cur_th[1] * Xg + cur_th[2] * Yg), 0.0, zmax)
        s.add(ax.plot_surface(Xg, Yg, Zlam, cmap="YlOrRd", alpha=0.44, linewidth=0, antialiased=True))

        pred = np.exp(X @ cur_th)
        nan = np.full(m, np.nan)
        s.add(ax.plot(np.c_[x1, x1, nan].ravel(), np.c_[x2, x2, nan].ravel(),
                      np.c_[y, np.clip(pred, 0.0, zmax), nan].ravel(), color=C["teal"], lw=0.9, alpha=0.75))
        nll = float(np.mean(pred - y * (X @ cur_th) + log_y_fact))
        s.status(f"step {i * sub}   mean NLL = {nll:.2f}")
        s.spin(i, n, sweep=42)

    return update


# ---------------------------------------------------------------------------
# 5. Three softmax probability surfaces sharpening
# ---------------------------------------------------------------------------
@animation(
    "Softmax: 3 Class Surfaces Sharpen",
    caption=r"""तीन classes (नीली, लाल, हरी) के data पर Softmax Regression Cross-Entropy loss को कम कर रहा है।
    शुरू में $\Theta = 0$ होने से तीनों probability surfaces $P(y=c\mid x) = 1/3$ पर सपाट हैं।
    जैसे-जैसे Gradient Descent आगे बढ़ता है, हर class के क्षेत्र में उसकी surface $1$ की ओर उठती है और बाक़ी दो $0$ की ओर झुकती हैं, जबकि हर बिंदु पर तीनों का जोड़ $1$ रहता है।""",
    frames=42,
    keys=(0.0, 0.25, 1.0),
)
def softmax_surfaces(s, n):
    rng = np.random.default_rng(21)
    m_per = 20
    centers = [(-1.35, -0.95), (1.35, -0.95), (0.0, 1.45)]
    pts = np.vstack([rng.normal(loc=c, scale=0.52, size=(m_per, 2)) for c in centers])
    y = np.repeat([0, 1, 2], m_per)
    m = len(y)
    X = np.c_[np.ones(m), pts]
    Y_oh = np.eye(3)[y]

    def softmax(Z):
        Zs = Z - Z.max(axis=-1, keepdims=True)
        E = np.exp(Zs)
        return E / E.sum(axis=-1, keepdims=True)

    # Run real Gradient Descent on Cross-Entropy loss
    Theta = np.zeros((3, 3))
    sub = 3
    alpha = 0.14
    hist = [Theta.copy()]
    for _ in range(n - 1):
        for _ in range(sub):
            P = softmax(X @ Theta)
            Theta = Theta - alpha * (X.T @ (P - Y_oh)) / m
        hist.append(Theta.copy())

    ax = s.ax3d(
        labels=(r"$x_1$", r"$x_2$", r"$P(y=c\mid x)$"),
        elev=27,
        azim=-60,
        xlim=(-2.5, 2.5),
        ylim=(-2.5, 2.5),
        zlim=(0.0, 1.05),
    )
    cols = [C["blue"], C["red"], C["green"]]
    for c in range(3):
        mask = y == c
        ax.scatter(pts[mask, 0], pts[mask, 1], np.full(mask.sum(), 0.01),
                   color=cols[c], s=16, depthshade=False, label=f"class {c + 1}")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    Xg, Yg = grid((-2.5, 2.5), (-2.5, 2.5), n=30)
    G = np.stack([np.ones_like(Xg), Xg, Yg], axis=-1)

    def update(i):
        s.clear()
        Th = hist[i]
        Pg = softmax(G @ Th)
        for c in range(3):
            s.add(ax.plot_surface(Xg, Yg, Pg[..., c], color=cols[c], alpha=0.33,
                                  linewidth=0, antialiased=True))

        # Analytical 3-way decision rays on floor z = 0.02 once weights become non-zero
        if i > 0:
            A = np.array([
                [Th[1, 0] - Th[1, 1], Th[2, 0] - Th[2, 1]],
                [Th[1, 1] - Th[1, 2], Th[2, 1] - Th[2, 2]],
            ])
            b = np.array([Th[0, 1] - Th[0, 0], Th[0, 2] - Th[0, 1]])
            if abs(np.linalg.det(A)) > 1e-6:
                xc, yc = np.linalg.solve(A, b)
                for c1, c2, c3 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                    d = Th[1:, c1] - Th[1:, c2]
                    ray_dir = np.array([-d[1], d[0]])
                    ray_dir /= np.linalg.norm(ray_dir) + 1e-9
                    # Pick the half-ray where c1 & c2 beat c3
                    test_pt = np.array([1.0, xc + ray_dir[0], yc + ray_dir[1]])
                    if (test_pt @ Th[:, c1]) < (test_pt @ Th[:, c3]):
                        ray_dir = -ray_dir
                    ts = np.linspace(0.0, 4.5, 60)
                    rx, ry = xc + ts * ray_dir[0], yc + ts * ray_dir[1]
                    keep = (np.abs(rx) <= 2.5) & (np.abs(ry) <= 2.5)
                    if np.any(keep):
                        s.add(ax.plot(rx[keep], ry[keep], np.full(keep.sum(), 0.02),
                                      color=C["dark"], lw=1.8))

        P_tr = softmax(X @ Th)
        ce = float(-np.mean(np.log(P_tr[np.arange(m), y] + 1e-12)))
        acc = float(np.mean(P_tr.argmax(axis=1) == y))
        s.status(f"step {i * sub}   CE = {ce:.3f}   acc = {acc * 100:.0f}%")
        s.spin(i, n, sweep=40)

    return update
