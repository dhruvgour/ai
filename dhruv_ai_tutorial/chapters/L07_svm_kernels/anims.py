"""Lecture 7: Support Vector Machines (Dual, Kernels, Soft Margin, SMO): 3D animations."""
import numpy as np

from lib.anim import C, animation, ease, grid, lerp, plane, surface


# ---- lightweight deterministic SMO solver used by the animations ----
def _rbf_gram(X, Z, sigma):
    d2 = np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma**2))


def _smo_solve(K, y, C_val, tol=1e-4, max_passes=120, seed=0):
    """Simplified Platt SMO on precomputed Gram matrix K for labels y in {-1, +1}."""
    rng = np.random.default_rng(seed)
    m = len(y)
    alpha = np.zeros(m)
    b = 0.0
    passes = 0
    total_iters = 0
    while passes < max_passes and total_iters < 2500:
        changed = 0
        for i in range(m):
            Ei = (alpha * y) @ K[:, i] + b - y[i]
            if (y[i] * Ei < -tol and alpha[i] < C_val) or (y[i] * Ei > tol and alpha[i] > 0.0):
                # pick j maximizing |Ei - Ej| most of the time, with occasional random step
                if rng.random() < 0.8:
                    E_all = (alpha * y) @ K + b - y
                    diffs = np.abs(Ei - E_all)
                    diffs[i] = -1.0
                    j = int(np.argmax(diffs))
                else:
                    j = int(rng.integers(m - 1))
                    if j >= i:
                        j += 1
                Ej = (alpha * y) @ K[:, j] + b - y[j]
                ai_old, aj_old = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(C_val, C_val + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - C_val)
                    H = min(C_val, ai_old + aj_old)
                if H - L < 1e-10:
                    continue
                eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                if eta <= 1e-12:
                    continue
                aj_new = np.clip(aj_old + y[j] * (Ei - Ej) / eta, L, H)
                if abs(aj_new - aj_old) < 1e-7:
                    continue
                ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
                b1 = b - Ei - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
                b2 = b - Ej - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]
                if 0.0 < ai_new < C_val:
                    b = b1
                elif 0.0 < aj_new < C_val:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
                alpha[i], alpha[j] = ai_new, aj_new
                changed += 1
        total_iters += 1
        passes = passes + 1 if changed == 0 else 0
    return alpha, b


def _zero_contour_3d(X, Y, Z, level=0.0):
    """Marching-squares segment list (xs, ys) with NaN separators for Z(X, Y) = level."""
    V = Z - level
    xs, ys = [], []
    rows, cols = V.shape
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    for r in range(rows - 1):
        for c in range(cols - 1):
            vals = (V[r, c], V[r, c + 1], V[r + 1, c + 1], V[r + 1, c])
            pts = ((X[r, c], Y[r, c]), (X[r, c + 1], Y[r, c + 1]),
                   (X[r + 1, c + 1], Y[r + 1, c + 1]), (X[r + 1, c], Y[r + 1, c]))
            hits = []
            for a, b_idx in edges:
                va, vb = vals[a], vals[b_idx]
                if (va < 0.0) != (vb < 0.0) or (va == 0.0 and vb != 0.0):
                    t = va / (va - vb)
                    hits.append((pts[a][0] + t * (pts[b_idx][0] - pts[a][0]),
                                 pts[a][1] + t * (pts[b_idx][1] - pts[a][1])))
            if len(hits) >= 2:
                xs.extend([hits[0][0], hits[1][0], np.nan])
                ys.extend([hits[0][1], hits[1][1], np.nan])
    return np.asarray(xs, float), np.asarray(ys, float)


def _line_in_box(w, offset, xlim, ylim):
    """Segment of w[0]*x1 + w[1]*x2 + offset = 0 clipped to xlim x ylim."""
    nrm2 = float(w @ w)
    if nrm2 < 1e-12:
        return np.array([]), np.array([])
    d = np.array([-w[1], w[0]]) / np.sqrt(nrm2)
    p = -offset * w / nrm2
    lo, hi = -20.0, 20.0
    for k, (a, b_lim) in enumerate((xlim, ylim)):
        if abs(d[k]) > 1e-9:
            t1, t2 = sorted(((a - p[k]) / d[k], (b_lim - p[k]) / d[k]))
            lo, hi = max(lo, t1), min(hi, t2)
    if lo > hi:
        return np.array([]), np.array([])
    ts = np.array([lo, hi])
    return p[0] + ts * d[0], p[1] + ts * d[1]


def _stems(ax, x1, x2, z0, z1, **kw):
    nan = np.full(len(x1), np.nan)
    return ax.plot(np.c_[x1, x1, nan].ravel(), np.c_[x2, x2, nan].ravel(),
                   np.c_[z0, z1, nan].ravel(), **kw)


# ================================================================ Animation 1
@animation(r"Feature Lift: $z = x_1^2 + x_2^2$",
           caption=r"""शुरू में अंदर का नीला समूह और बाहर का लाल छल्ला 2D फ़र्श ($z=0$) पर पड़े हैं, जहाँ कोई भी
           सीधी रेखा इन्हें अलग नहीं कर सकती। जैसे-जैसे feature map $\phi(x) = (x_1, x_2, x_1^2 + x_2^2)$ सक्रिय
           होता है, हर बिंदु कटोरे जैसी सतह $z = x_1^2 + x_2^2$ पर ऊपर उठता है। बाहर वाले लाल बिंदु ज़्यादा ऊँचे उठ
           जाते हैं और बीच में एक सपाट हरा separating plane आसानी से दोनों classes के बीच आ जाता है। फ़र्श पर
           उसी plane का कटाव एक गोल वृत्ताकार boundary बनाता है।""",
           frames=40, keys=(0.0, 0.36, 1.0))
def ring_lift(s, n):
    rng = np.random.default_rng(7)
    m_in, m_out = 26, 34
    r_in = rng.uniform(0.15, 0.68, m_in)
    th_in = rng.uniform(0, 2 * np.pi, m_in)
    r_out = rng.uniform(1.12, 1.55, m_out)
    th_out = rng.uniform(0, 2 * np.pi, m_out)
    x_in = np.c_[r_in * np.cos(th_in), r_in * np.sin(th_in)]
    x_out = np.c_[r_out * np.cos(th_out), r_out * np.sin(th_out)]
    X2 = np.r_[x_in, x_out]
    y = np.r_[-np.ones(m_in), np.ones(m_out)]
    z_full = X2[:, 0] ** 2 + X2[:, 1] ** 2

    # Fit actual linear SVM in lifted 3D space phi(x) = (x1, x2, x1^2 + x2^2)
    Phi = np.c_[X2, z_full]
    alpha, b = _smo_solve(Phi @ Phi.T, y, C_val=20.0, seed=1)
    w = (alpha * y) @ Phi
    sv = alpha > 1e-3
    n_sv = int(sv.sum())

    lim = (-1.75, 1.75)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_1^2 + x_2^2$"), elev=22, azim=-56,
                xlim=lim, ylim=lim, zlim=(-0.15, 3.3))
    G1, G2 = grid(lim, lim, 28)
    GZ = G1**2 + G2**2
    # Faint floor shadows of the original 2D points
    ax.scatter(x_in[:, 0], x_in[:, 1], -0.12, color=C["blue"], s=6, alpha=0.22, depthshade=False)
    ax.scatter(x_out[:, 0], x_out[:, 1], -0.12, color=C["red"], s=6, alpha=0.22, depthshade=False)

    # Precompute intersection circle on floor where w1*x1 + w2*x2 + w3*(x1^2+x2^2) + b = 0
    F_floor = w[0] * G1 + w[1] * G2 + w[2] * GZ + b
    cx, cy = _zero_contour_3d(G1, G2, F_floor, 0.0)

    def update(i):
        s.clear()
        tl = ease(i / (0.62 * (n - 1)))
        tp = ease((i - 0.52 * (n - 1)) / (0.36 * (n - 1)))
        zc = tl * z_full
        s.add(ax.plot_wireframe(G1, G2, tl * GZ, rstride=3, cstride=3, color=C["gray"],
                                lw=0.45, alpha=0.35))
        s.add(ax.scatter(x_in[:, 0], x_in[:, 1], zc[:m_in], color=C["blue"], s=18, depthshade=False))
        s.add(ax.scatter(x_out[:, 0], x_out[:, 1], zc[m_in:], color=C["red"], s=18, depthshade=False))
        if tp > 0.02:
            s.add(plane(ax, lambda a, b_g: -(w[0] * a + w[1] * b_g + b) / w[2],
                        lim, lim, color=C["green"], alpha=0.30 * tp, n=10))
            s.add(ax.scatter(X2[sv, 0], X2[sv, 1], zc[sv], facecolors="none", edgecolors=C["dark"],
                             s=58, linewidths=1.3, depthshade=False))
            if len(cx):
                s.add(ax.plot(cx, cy, -0.12, color=C["green"], lw=1.8, alpha=tp))
        s.status(f"lift = {int(round(100 * tl))}%   plane = {'on' if tp > 0.1 else 'off'}   SVs = {n_sv}")
        s.spin(i, n, sweep=42, tilt=-4)

    return update


# ================================================================ Animation 2
@animation("RBF Surface Built Bump by Bump",
           caption=r"""RBF SVM का decision function $f(x) = \sum_i \alpha_i y^{(i)} K(x^{(i)}, x) + b$ असल में
           support vectors पर रखे Gaussian bumps का जोड़ है। यहाँ पहले पूरा SMO चलाकर support vectors निकाले
           गए हैं, और फिर एक-एक करके हर support vector का bump जोड़ा जा रहा है। लाल बिंदु ($y=+1$) ऊपर की ओर
           पहाड़ी उठाते हैं और नीले बिंदु ($y=-1$) नीचे की ओर गड्ढा बनाते हैं। काले घेरे सक्रिय support vectors हैं
           और $z=0$ पर काली रेखा बनती हुई non-linear decision boundary दिखाती है।""",
           frames=40, keys=(0.12, 0.52, 1.0))
def rbf_bumps(s, n):
    rng = np.random.default_rng(14)
    # Clean curved moons so every support vector sits along the margin and builds the S-curve smoothly
    t0 = np.linspace(0.12, np.pi - 0.12, 18)
    c_pos = np.c_[np.cos(t0) - 0.42, np.sin(t0) - 0.24] + rng.normal(0, 0.07, (18, 2))
    c_neg = np.c_[1.0 - np.cos(t0) - 0.42, 0.24 - np.sin(t0)] + rng.normal(0, 0.07, (18, 2))
    X = np.r_[c_pos, c_neg]
    y = np.r_[np.ones(18), -np.ones(18)]
    sigma = 0.72
    K = _rbf_gram(X, X, sigma)
    alpha, b = _smo_solve(K, y, C_val=4.0, seed=2)

    sv_idx = np.where(alpha > 1e-3)[0]
    # Order support vectors from left to right along x1 so the curved surface grows across space
    pos_sv = sv_idx[y[sv_idx] > 0][np.argsort(X[sv_idx[y[sv_idx] > 0], 0])]
    neg_sv = sv_idx[y[sv_idx] < 0][np.argsort(X[sv_idx[y[sv_idx] < 0], 0])]
    order = []
    for k in range(max(len(pos_sv), len(neg_sv))):
        if k < len(pos_sv):
            order.append(int(pos_sv[k]))
        if k < len(neg_sv):
            order.append(int(neg_sv[k]))
    order = np.array(order, int)
    n_sv = len(order)

    xlim, ylim, zlim = (-1.9, 1.9), (-1.45, 1.45), (-2.1, 2.1)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$f(x)$"), elev=27, azim=-58,
                xlim=xlim, ylim=ylim, zlim=zlim)
    plane(ax, lambda a, b_g: 0.0 * a, xlim, ylim, color=C["gray"], alpha=0.12, n=6)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], 0.0, color=C["red"], s=16, depthshade=False)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], 0.0, color=C["blue"], s=16, depthshade=False)

    G1, G2 = grid(xlim, ylim, 34)
    grid_pts = np.c_[G1.ravel(), G2.ravel()]
    K_grid = _rbf_gram(X[order], grid_pts, sigma).reshape(n_sv, *G1.shape)
    K_train = K[order]                                   # (n_sv, m)
    weights = alpha[order] * y[order]

    def update(i):
        s.clear()
        prog = (i / (n - 1)) * n_sv
        full = min(int(prog), n_sv)
        frac = ease(prog - full) if full < n_sv else 0.0
        w_part = np.zeros(n_sv)
        w_part[:full] = weights[:full]
        if full < n_sv:
            w_part[full] = frac * weights[full]
        b_part = b * min(1.0, prog / max(1, n_sv))
        Z = np.tensordot(w_part, K_grid, axes=(0, 0)) + b_part
        Z_clip = np.clip(Z, *zlim)
        s.add(ax.plot_surface(G1, G2, Z_clip, cmap="coolwarm", vmin=-1.7, vmax=1.7,
                              alpha=0.45, linewidth=0, rstride=1, cstride=1))
        cx, cy = _zero_contour_3d(G1, G2, Z, 0.0)
        if len(cx):
            s.add(ax.plot(cx, cy, 0.0, color=C["dark"], lw=2.0))
        shown = full + (1 if full < n_sv and frac > 0.1 else 0)
        if shown > 0:
            act = order[:shown]
            s.add(ax.scatter(X[act, 0], X[act, 1], np.zeros(shown), facecolors="none",
                             edgecolors=C["dark"], s=60, linewidths=1.4, depthshade=False))
            cur = act[-1]
            z_cur = float(w_part[:shown] @ K_train[:shown, cur] + b_part)
            s.add(ax.plot([X[cur, 0]] * 2, [X[cur, 1]] * 2, [0.0, np.clip(z_cur, *zlim)],
                          color=C["orange"], lw=2.2))
        pred_tr = w_part @ K_train + b_part
        acc = 100.0 * np.mean(np.sign(pred_tr + 1e-12) == y) if shown > 0 else 50.0
        s.status(f"SV bumps: {shown}/{n_sv}   train acc = {acc:.0f}%")
        s.spin(i, n, sweep=40)

    return update


# ================================================================ Animation 3
@animation(r"RBF $\sigma$: Underfit to Overfit",
           caption=r"""यहाँ एक ही noisy two-moons dataset पर RBF kernel की चौड़ाई $\sigma$ को बड़े से बहुत छोटे तक
           घटाया जा रहा है। शुरू में बड़ा $\sigma$ बहुत चौड़े bumps बनाता है, इसलिए surface लगभग चपटी रहती है और
           मुड़ी हुई सीमा नहीं पकड़ पाती (underfitting)। बीच के $\sigma \approx 0.65$ पर चिकनी S-आकार की सीमा बनती है
           और test accuracy सबसे ऊपर होती है। अंत में बहुत छोटे $\sigma$ पर हर बिंदु के ऊपर नुकीली सुई बन जाती है:
           लगभग सारे बिंदु support vectors बन जाते हैं और test accuracy गिर जाती है (overfitting)।""",
           frames=38, keys=(0.05, 0.48, 0.97))
def rbf_bandwidth_sweep(s, n):
    def _moons(seed, m_half, noise):
        r = np.random.default_rng(seed)
        a = r.uniform(0, np.pi, m_half)
        b_ang = r.uniform(0, np.pi, m_half)
        p1 = np.c_[np.cos(a) - 0.45, np.sin(a) - 0.22] + r.normal(0, noise, (m_half, 2))
        p2 = np.c_[1.0 - np.cos(b_ang) - 0.45, 0.28 - np.sin(b_ang)] + r.normal(0, noise, (m_half, 2))
        return np.r_[p1, p2], np.r_[np.ones(m_half), -np.ones(m_half)]

    X_tr, y_tr = _moons(21, 22, 0.20)
    X_te, y_te = _moons(99, 120, 0.20)
    xlim, ylim, zlim = (-1.95, 1.95), (-1.55, 1.55), (-2.1, 1.9)
    floor = -1.95

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$f(x)$"), elev=28, azim=-58,
                xlim=xlim, ylim=ylim, zlim=zlim)
    ax.scatter(X_tr[y_tr > 0, 0], X_tr[y_tr > 0, 1], floor, color=C["red"], s=15, depthshade=False)
    ax.scatter(X_tr[y_tr < 0, 0], X_tr[y_tr < 0, 1], floor, color=C["blue"], s=15, depthshade=False)

    G1, G2 = grid(xlim, ylim, 34)
    grid_pts = np.c_[G1.ravel(), G2.ravel()]
    sigmas = np.geomspace(2.9, 0.13, n)
    frames_data = []
    for sig in sigmas:
        K_tr = _rbf_gram(X_tr, X_tr, sig)
        al, b = _smo_solve(K_tr, y_tr, C_val=5.0, seed=3)
        wy = al * y_tr
        Z = (wy @ _rbf_gram(X_tr, grid_pts, sig)).reshape(G1.shape) + b
        tr_acc = 100.0 * np.mean(np.sign(wy @ K_tr + b) == y_tr)
        te_acc = 100.0 * np.mean(np.sign(wy @ _rbf_gram(X_tr, X_te, sig) + b) == y_te)
        sv = al > 1e-3
        frames_data.append((sig, Z, tr_acc, te_acc, sv))

    def update(i):
        s.clear()
        sig, Z, tr_acc, te_acc, sv = frames_data[i]
        s.add(ax.plot_surface(G1, G2, np.clip(Z, -1.6, 1.75), cmap="coolwarm", vmin=-1.4, vmax=1.4,
                              alpha=0.45, linewidth=0, rstride=1, cstride=1))
        cx, cy = _zero_contour_3d(G1, G2, Z, 0.0)
        if len(cx):
            s.add(ax.plot(cx, cy, floor + 0.02, color=C["dark"], lw=1.8))
        s.add(ax.scatter(X_tr[sv, 0], X_tr[sv, 1], np.full(sv.sum(), floor),
                         facecolors="none", edgecolors=C["dark"], s=46, linewidths=1.0, depthshade=False))
        s.status(rf"$\sigma$={sig:.2f}  train={tr_acc:.0f}%  test={te_acc:.0f}%  SV={int(sv.sum())}")
        s.spin(i, n, sweep=36)

    return update


# ================================================================ Animation 4
@animation(r"Soft-Margin $C$ Sweep with Outliers",
           caption=r"""दो साफ समूहों के बीच तीन भटके हुए outliers डाल दिए गए हैं और linear soft-margin SVM में
           $C$ को छोटे से बड़े तक बढ़ाया जा रहा है। छोटे $C$ पर नीला plane $z = w^T x + b$ शांत और कम ढलान
           वाला रहता है: फ़र्श पर दिखती margin की पट्टी चौड़ी रहती है और outliers पर नारंगी slack डंडियाँ $\xi_i > 0$
           बर्दाश्त कर ली जाती हैं। जैसे ही $C$ बहुत बड़ा होता है, गलती से बचने की ज़िद में plane बुरी तरह घूमकर खड़ी
           ढलान ले लेता है और पूरा margin पिचककर पतला हो जाता है।""",
           frames=38, keys=(0.05, 0.50, 0.97))
def soft_margin_c_sweep(s, n):
    rng = np.random.default_rng(31)
    x_neg = rng.normal([-1.15, -0.85], 0.36, (18, 2))
    x_pos = rng.normal([1.15, 0.85], 0.36, (18, 2))
    # Three outliers: separable only at the cost of a badly twisted, tiny-margin hyperplane
    outliers_X = np.array([[0.58, -0.15], [-0.48, 0.25], [0.22, -0.55]])
    outliers_y = np.array([-1.0, 1.0, 1.0])
    X = np.r_[x_neg, x_pos, outliers_X]
    y = np.r_[-np.ones(18), np.ones(18), outliers_y]
    K_lin = X @ X.T

    xlim, ylim, zlim = (-2.2, 2.2), (-2.2, 2.2), (-3.2, 3.0)
    floor = -3.05
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$w^T x + b$"), elev=25, azim=-60,
                xlim=xlim, ylim=ylim, zlim=zlim)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], floor, color=C["blue"], s=16, depthshade=False)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], floor, color=C["red"], s=16, depthshade=False)
    # Mark the 3 injected outliers with dark x-crosses on the floor
    ax.scatter(outliers_X[:, 0], outliers_X[:, 1], floor + 0.03, marker="x", color=C["dark"],
               s=42, linewidths=1.4, depthshade=False)

    C_vals = np.geomspace(0.06, 45.0, n)
    G1, G2 = grid(xlim, ylim, 16)
    sols = []
    for cv in C_vals:
        al, b = _smo_solve(K_lin, y, C_val=float(cv), tol=1e-5, max_passes=180, seed=5)
        w = (al * y) @ X
        sols.append((cv, w, b, al))

    def update(i):
        s.clear()
        cv, w, b, al = sols[i]
        ZP = np.clip(w[0] * G1 + w[1] * G2 + b, *zlim)
        s.add(ax.plot_surface(G1, G2, ZP, color=C["teal"], alpha=0.30, linewidth=0, shade=False))
        # Decision boundary (solid) and +/-1 margin lines (dashed) on the floor
        for off, ls, col, lw in ((b, "-", C["dark"], 2.0),
                                 (b - 1.0, "--", C["red"], 1.1),
                                 (b + 1.0, "--", C["blue"], 1.1)):
            lx, ly = _line_in_box(w, off, xlim, ylim)
            if len(lx):
                s.add(ax.plot(lx, ly, floor + 0.02, color=col, ls=ls, lw=lw))
        # Slack bars xi_i = max(0, 1 - y_i (w^T x_i + b)) rising from floor
        f_val = X @ w + b
        xi = np.maximum(0.0, 1.0 - y * f_val)
        viol = xi > 0.04
        if np.any(viol):
            top = np.minimum(floor + 1.15 * xi[viol], zlim[1] - 0.1)
            s.add(_stems(ax, X[viol, 0], X[viol, 1], np.full(viol.sum(), floor), top,
                         color=C["orange"], lw=2.3))
        bnd = al >= 0.98 * cv
        marg = 2.0 / max(1e-6, float(np.linalg.norm(w)))
        s.status(f"C={cv:.2f}   margin={marg:.2f}   bounded SVs={int(bnd.sum())}")
        s.spin(i, n, sweep=38)

    return update


# ================================================================ Animation 5
@animation(r"SMO Path on Dual $W(\alpha_1,\alpha_2)$",
           caption=r"""यहाँ 3-बिंदु वाले SVM ($y=(+1,+1,-1)$, समता से $\alpha_3=\alpha_1+\alpha_2$) का असली concave
           dual surface $W(\alpha_1,\alpha_2)$ box $[0, C]^2$ के ऊपर दिखाया गया है। हर कदम में SMO एक जोड़ा
           चुनता है: $(1,2)$ चुनने पर बिंदु $45^\circ$ की तिरछी रेखा पर चलता है, और $(1,3)$ या $(2,3)$ चुनने पर अक्ष
           के समांतर। नारंगी मेहराब वर्तमान 1D quadratic टुकड़ा $[L, H]$ है। जब बिना सीमा वाला शिखर वैध क्षेत्र से बाहर
           निकल जाता है, तो update किनारे $L$ या $H$ पर clip हो जाता है और धीरे-धीरे शिखर (हरा तारा) तक पहुँचता है।""",
           frames=42, keys=(0.14, 0.55, 1.0))
def smo_dual_box(s, n):
    # 3-example SVM with y = (+1, +1, -1): equality constraint gives alpha_3 = alpha_1 + alpha_2,
    # so W(alpha_1, alpha_2) is an exact static strictly-concave quadratic over [0, C]^2.
    X = np.array([[0.0, 0.72], [0.78, 0.05], [0.18, 0.18]])
    y = np.array([1.0, 1.0, -1.0])
    C_box = 1.55
    K = _rbf_gram(X, X, sigma=0.75)
    Q = np.outer(y, y) * K

    def W_grid(a1, a2):
        a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
        a3 = a1 + a2
        A = np.stack([a1, a2, a3], axis=-1)
        quad = np.einsum("...i,ij,...j->...", A, Q, A)
        return 2.0 * a3 - 0.5 * quad

    # Run exact step-by-step SMO starting from alpha = (0, 0, 0) on pairs (2,3), (1,2), (1,3)...
    alpha = np.zeros(3)
    pair_schedule = [(1, 2), (0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (0, 1)]
    steps = [(alpha[:2].copy(), (0, 0), alpha[:2].copy(), alpha[:2].copy(), False)]
    for i_idx, j_idx in pair_schedule:
        # Compute exact 1D slice along valid segment [L, H] inside 0 <= alpha_k <= C_box
        old = alpha.copy()
        # Parameterize move by delta added to alpha[j_idx]:
        # alpha_new[j] = old[j] + d, alpha_new[i] = old[i] - y[i]*y[j]*d
        s_ij = -y[i_idx] * y[j_idx]
        d_lo = max(-old[j_idx], (-old[i_idx] / s_ij) if s_ij > 0 else ((C_box - old[i_idx]) / s_ij))
        d_hi = min(C_box - old[j_idx], ((C_box - old[i_idx]) / s_ij) if s_ij > 0 else (-old[i_idx] / s_ij))
        v = np.zeros(3)
        v[j_idx], v[i_idx] = 1.0, s_ij
        eta = float(v @ Q @ v)
        grad_d = float((np.ones(3) - Q @ old) @ v)
        d_unc = grad_d / eta
        d_clip = float(np.clip(d_unc, d_lo, d_hi))
        clipped = abs(d_unc - d_clip) > 1e-4
        end_lo = old.copy()
        end_lo[j_idx] += d_lo
        end_lo[i_idx] += s_ij * d_lo
        end_hi = old.copy()
        end_hi[j_idx] += d_hi
        end_hi[i_idx] += s_ij * d_hi
        alpha[j_idx] += d_clip
        alpha[i_idx] += s_ij * d_clip
        steps.append((alpha[:2].copy(), (i_idx + 1, j_idx + 1), end_lo[:2], end_hi[:2], clipped))

    opt_a = steps[-1][0]
    opt_w = float(W_grid(opt_a[0], opt_a[1]))

    z_floor = -0.35
    z_top = opt_w + 0.35
    ax = s.ax3d(labels=(r"$\alpha_1$", r"$\alpha_2$", r"$W(\alpha)$"), elev=30, azim=-56,
                xlim=(-0.05, C_box + 0.08), ylim=(-0.05, C_box + 0.08), zlim=(z_floor, z_top))
    _, GX, GY, GZ = surface(ax, lambda a, b_g: np.maximum(W_grid(a, b_g), z_floor),
                            (0.0, C_box), (0.0, C_box), n=34, cmap="viridis", alpha=0.42)
    ax.contour(GX, GY, GZ, levels=8, zdir="z", offset=z_floor, cmap="viridis", linewidths=0.65, alpha=0.7)
    # Draw [0, C] x [0, C] box on floor plus alpha_1 + alpha_2 <= C feasible edge
    bx = [0, C_box, C_box, 0, 0]
    by = [0, 0, C_box, C_box, 0]
    ax.plot(bx, by, z_floor + 0.01, color=C["dark"], lw=1.4)
    ax.plot([0, C_box], [C_box, 0], z_floor + 0.01, color=C["purple"], lw=1.1, ls="--")
    ax.scatter([opt_a[0]], [opt_a[1]], [opt_w + 0.03], marker="*", s=115, color=C["green"], depthshade=False)

    n_seg = len(steps) - 1
    pts_all = np.array([st[0] for st in steps])

    def update(i):
        s.clear()
        pos = (i / (n - 1)) * n_seg
        k = min(int(pos), n_seg - 1)
        u = ease(pos - k)
        cur = lerp(pts_all[k], pts_all[k + 1], u)
        _, pair_lbl, p_lo, p_hi, was_clip = steps[k + 1]

        # Active [L, H] segment on floor and its 1D parabolic arch on the dome
        ts = np.linspace(0.0, 1.0, 26)
        seg_xy = p_lo[None, :] + ts[:, None] * (p_hi - p_lo)[None, :]
        seg_z = W_grid(seg_xy[:, 0], seg_xy[:, 1])
        s.add(ax.plot(seg_xy[:, 0], seg_xy[:, 1], z_floor + 0.02, color=C["orange"], lw=2.0))
        s.add(ax.plot(seg_xy[:, 0], seg_xy[:, 1], np.maximum(seg_z, z_floor) + 0.02,
                      color=C["orange"], lw=2.2))

        # Path traced so far on dome and floor
        hist = np.vstack([pts_all[: k + 1], cur])
        hz = W_grid(hist[:, 0], hist[:, 1])
        s.add(ax.plot(hist[:, 0], hist[:, 1], hz + 0.03, color=C["red"], lw=1.9, marker="o", ms=3.2))
        s.add(ax.plot(hist[:, 0], hist[:, 1], np.full(len(hist), z_floor + 0.02),
                      color=C["red"], lw=1.0, ls="--"))
        w_cur = float(hz[-1])
        s.add(ax.scatter([cur[0]], [cur[1]], [w_cur + 0.03], color=C["red"], s=58,
                         edgecolor="white", depthshade=False))
        tag = " [clip]" if was_clip else ""
        s.status(f"step {k + 1}/{n_seg}  pair {pair_lbl}{tag}  W={w_cur:.3f}")
        s.spin(i, n, sweep=44)

    return update
