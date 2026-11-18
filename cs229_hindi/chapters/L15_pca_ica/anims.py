"""Lecture 15: Principal Component Analysis and Independent Component Analysis: 3D animations."""
import numpy as np

from lib.anim import C, animation, arrow, ease, lerp


def _cube_segments(half=1.0):
    """Return (24, 3) endpoints for the 12 edges of [-half, half]^3 separated by NaNs."""
    v = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)], float)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(v[i] - v[j]) > 1e-9) == 1:
                edges.extend([v[i], v[j], [np.nan, np.nan, np.nan]])
    return np.array(edges)


def _apply_segs(segs, M, shift=(0.0, 0.0, 0.0)):
    """Apply 3x3 matrix M to NaN-separated 3D segments and translate by shift."""
    out = segs @ M.T
    out += np.asarray(shift, float)
    return out


def _axis_angle_rot(axis, theta):
    """3D rotation matrix around unit axis by angle theta (Rodrigues formula)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    ux, uy, uz = a
    K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@animation("Sweeping u to Maximize Variance",
           caption=r"""यहाँ 3D में zero-mean data का बादल फैला है और नीला तीर unit vector $u$ है। शुरू में $u$ कम
           फैलाव वाली दिशा में है, इसलिए उस रेखा पर नारंगी projections पास-पास सिमटे हैं और $u^T \Sigma u$ छोटा है।
           जैसे-जैसे $u$ घूमकर पहले principal component (हरी dashed रेखा) की ओर जाता है, नारंगी बिंदु सबसे ज़्यादा फैल
           जाते हैं। अंत में $u$ ठीक top eigenvector $u_1$ पर रुकता है जहाँ projected variance $\lambda_1$ अधिकतम है।""",
           frames=40, keys=(0.0, 0.38, 1.0))
def sweep_first_pc(s, n):
    rng = np.random.default_rng(15)
    m = 68
    raw = rng.normal(size=(m, 3)) * np.array([2.05, 0.78, 0.32])
    R = _axis_angle_rot((0.45, -0.65, 0.60), 0.95)
    X = raw @ R.T
    X -= X.mean(axis=0)
    Sigma = (X.T @ X) / m
    eigvals, eigvecs = np.linalg.eigh(Sigma)   # ascending order
    u3, u2, u1 = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
    if u1[0] < 0:
        u1 = -u1
    lmax = eigvals[2]

    # Projected gradient ascent on u^T Sigma u over the unit sphere
    u = u3 + 0.35 * u2 + 0.05 * u1
    u = u / np.linalg.norm(u)
    us = [u.copy()]
    for _ in range(n - 1):
        u = u + 0.048 * (Sigma @ u)
        u = u / np.linalg.norm(u)
        if u @ u1 < 0:
            u = -u
        us.append(u.copy())

    lim = 3.2
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=23, azim=-56,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=C["dark"], s=14, alpha=0.35, depthshade=False)
    ax.plot([-3.0 * u1[0], 3.0 * u1[0]], [-3.0 * u1[1], 3.0 * u1[1]], [-3.0 * u1[2], 3.0 * u1[2]],
            color=C["green"], ls="--", lw=1.2, alpha=0.75)
    ax.text(*(2.85 * u1), "PC1", color=C["green"], fontsize=9, fontweight="bold")

    def update(i):
        s.clear()
        uk = us[i]
        proj_scalar = X @ uk
        P = proj_scalar[:, None] * uk[None, :]
        var = float(uk @ Sigma @ uk)

        nan = np.full((m, 3), np.nan)
        seg = np.hstack([X, P, nan]).reshape(-1, 3)
        s.add(ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=C["gray"], lw=0.55, alpha=0.38))
        s.add(ax.plot([-3.0 * uk[0], 3.0 * uk[0]], [-3.0 * uk[1], 3.0 * uk[1]], [-3.0 * uk[2], 3.0 * uk[2]],
                      color=C["blue"], lw=1.6))
        s.add(arrow(ax, (0, 0, 0), 2.35 * uk, color=C["blue"], lw=2.3))
        s.add(ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=C["orange"], s=22, depthshade=False))
        s.status(rf"var $u^T\Sigma u$ = {var:.2f}  (max $\lambda_1$ = {lmax:.2f})")
        s.spin(i, n, sweep=36)

    return update


@animation("Collapsing Cloud to Principal Plane",
           caption=r"""यहाँ 3D points को top दो eigenvectors $u_1, u_2$ से बने नीले principal plane ($k=2$) पर उतारा
           जा रहा है। हल्के सलेटी बिंदु मूल $x^{(i)}$ हैं और नीले बिंदु धीरे-धीरे plane पर उनके orthogonal projection
           $\tilde{x}^{(i)} = U_2 U_2^T x^{(i)}$ पर गिरते हैं। लाल रेखाएँ perpendicular residuals
           $x^{(i)} - \tilde{x}^{(i)}$ दिखाती हैं। PCA ठीक ऐसा plane चुनता है जिससे इन लाल रेखाओं की squared
           लंबाइयों का औसत सबसे कम हो।""",
           frames=40, keys=(0.0, 0.5, 1.0))
def collapse_to_plane(s, n):
    rng = np.random.default_rng(21)
    m = 52
    raw = rng.normal(size=(m, 3)) * np.array([1.75, 1.25, 0.68])
    R = _axis_angle_rot((0.55, 0.35, -0.25), 0.72)
    X = raw @ R.T
    X -= X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U2 = Vt[:2].T
    X_hat = X @ U2 @ U2.T
    full_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))

    lim = 3.0
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=22, azim=-52,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    g1, g2 = np.meshgrid(np.linspace(-2.5, 2.5, 10), np.linspace(-2.5, 2.5, 10))
    P_surf = g1[..., None] * U2[:, 0] + g2[..., None] * U2[:, 1]
    ax.plot_surface(P_surf[..., 0], P_surf[..., 1], P_surf[..., 2],
                    color=C["blue"], alpha=0.20, linewidth=0, shade=False)
    arrow(ax, (0, 0, 0), 2.1 * U2[:, 0], color=C["blue"], lw=2.0)
    arrow(ax, (0, 0, 0), 1.8 * U2[:, 1], color=C["teal"], lw=2.0)
    ax.text(*(2.25 * U2[:, 0]), r"$u_1$", color=C["blue"], fontsize=10)
    ax.text(*(1.98 * U2[:, 1]), r"$u_2$", color=C["teal"], fontsize=10)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=C["gray"], s=11, alpha=0.35, depthshade=False)

    nan = np.full((m, 3), np.nan)
    full_segs = np.hstack([X, X_hat, nan]).reshape(-1, 3)

    def update(i):
        s.clear()
        t = float(ease(i / (n - 1)))
        cur = lerp(X, X_hat, t)
        traced = np.hstack([X, cur, nan]).reshape(-1, 3)

        s.add(ax.plot(full_segs[:, 0], full_segs[:, 1], full_segs[:, 2],
                      color=C["red"], lw=0.7, ls=":", alpha=0.45))
        if i > 0:
            s.add(ax.plot(traced[:, 0], traced[:, 1], traced[:, 2],
                          color=C["red"], lw=1.15, alpha=0.85))
        s.add(ax.scatter(cur[:, 0], cur[:, 1], cur[:, 2], color=C["blue"], s=23, depthshade=False))
        s.status(f"projection {int(round(100 * t)):3d}%   plane MSE = {full_mse:.2f}")
        s.spin(i, n, sweep=38)

    return update


@animation("PCA Reconstruction: k = 1, 2, 3",
           caption=r"""एक मुड़े हुए 3D बिंदु-समूह को अलग-अलग $k$ से reconstruct किया गया है। शुरू में $k=1$ पर सारे
           अनुमानित बिंदु सिर्फ़ एक सीधी रेखा ($u_1$) पर हैं और लाल error रेखाएँ लंबी हैं। बीच में $k=2$ लेते ही बिंदु
           principal plane पर फैल जाते हैं और ज़्यादातर variance वापस मिल जाता है। अंत में $k=3$ पर तीसरा component भी
           जुड़ जाता है, reconstruction error शून्य हो जाता है, और मूल 3D बादल पूरा लौट आता है।""",
           frames=42, keys=(0.08, 0.50, 1.0))
def reconstruct_k123(s, n):
    rng = np.random.default_rng(9)
    m = 60
    t_param = np.linspace(-2.1, 2.1, m)
    w_param = rng.uniform(-1.15, 1.15, m)
    z_param = 0.55 * np.sin(1.4 * t_param) + rng.normal(0, 0.22, m)
    raw = np.c_[t_param * 1.15, w_param * 1.05, z_param]
    X = raw @ _axis_angle_rot((0.4, -0.5, 0.7), 0.65).T
    X -= X.mean(axis=0)

    _, svals, Vt = np.linalg.svd(X, full_matrices=False)
    tot_var = float(np.sum(svals**2) / m)
    X1 = X @ Vt[:1].T @ Vt[:1]
    X2 = X @ Vt[:2].T @ Vt[:2]
    X3 = X.copy()
    v1_pct = 100.0 * float(svals[0] ** 2) / np.sum(svals**2)
    v2_pct = 100.0 * float(np.sum(svals[:2] ** 2)) / np.sum(svals**2)

    colors = [C["blue"] if val < -0.6 else (C["teal"] if val < 0.6 else C["orange"]) for val in t_param]
    lim = 3.0
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=24, azim=-58,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=C["gray"], s=11, alpha=0.30, depthshade=False)

    u1 = Vt[0]
    g1, g2 = np.meshgrid(np.linspace(-2.5, 2.5, 9), np.linspace(-2.2, 2.2, 9))
    P_plane = g1[..., None] * Vt[0] + g2[..., None] * Vt[1]
    nan = np.full((m, 3), np.nan)

    def update(i):
        s.clear()
        if i <= 9:
            cur, k_lbl, plane_alpha = X1, "1", 0.0
        elif i <= 17:
            tau = float(ease((i - 9) / 8.0))
            cur, k_lbl, plane_alpha = lerp(X1, X2, tau), "1 -> 2", 0.18 * tau
        elif i <= 25:
            cur, k_lbl, plane_alpha = X2, "2", 0.18
        elif i <= 34:
            tau = float(ease((i - 25) / 9.0))
            cur, k_lbl, plane_alpha = lerp(X2, X3, tau), "2 -> 3", 0.18 * (1.0 - tau)
        else:
            cur, k_lbl, plane_alpha = X3, "3", 0.0

        if i <= 17:
            s.add(ax.plot([-2.8 * u1[0], 2.8 * u1[0]], [-2.8 * u1[1], 2.8 * u1[1]], [-2.8 * u1[2], 2.8 * u1[2]],
                          color=C["blue"], lw=1.4, alpha=0.7))
        if plane_alpha > 0.01:
            s.add(ax.plot_surface(P_plane[..., 0], P_plane[..., 1], P_plane[..., 2],
                                  color=C["blue"], alpha=plane_alpha, linewidth=0, shade=False))

        mse = float(np.mean(np.sum((X - cur) ** 2, axis=1)))
        var_pct = max(0.0, min(100.0, 100.0 * (1.0 - mse / tot_var)))
        if k_lbl == "1":
            var_pct = v1_pct
        elif k_lbl == "2":
            var_pct = v2_pct

        if mse > 1e-4:
            segs = np.hstack([X, cur, nan]).reshape(-1, 3)
            s.add(ax.plot(segs[:, 0], segs[:, 1], segs[:, 2], color=C["red"], lw=0.85, alpha=0.65))
        s.add(ax.scatter(cur[:, 0], cur[:, 1], cur[:, 2], c=colors, s=24, depthshade=False))
        s.status(f"k = {k_lbl:<6} var = {var_pct:5.1f}%   MSE = {mse:.2f}")
        s.spin(i, n, sweep=42)

    return update


@animation("ICA: Cube Mixed & Unmixed by W",
           caption=r"""शुरू में तीन स्वतंत्र super-Gaussian sources एक सीधे घन (cube) के भीतर हैं। फिर अज्ञात mixing
           matrix $A$ उन्हें तिरछे और दबे हुए बक्से ($x = As$) में मिला देता है जहाँ सभी coordinates आपस में जुड़ जाते हैं।
           इसके बाद CS229 का sigmoid-CDF gradient ascent नियम $W$ को step-by-step सीखता है। देखिए कैसे सीखा हुआ
           $W$ तिरछे बक्से को सीधा करके वापस स्वतंत्र अक्षों वाला घन ($\hat{s} = Wx$) बना देता है।""",
           frames=44, keys=(0.0, 0.28, 1.0))
def ica_cube_unmix(s, n):
    rng = np.random.default_rng(12)
    m_train = 1400
    u_raw = rng.uniform(-1.0, 1.0, size=(m_train, 3))
    S_train = np.sign(u_raw) * (np.abs(u_raw) ** 3)     # super-Gaussian, strictly inside [-1, 1]^3
    scale_s = 1.65 / np.max(np.abs(S_train))
    S_train *= scale_s
    half = 1.65

    A = np.array([[1.05, 0.62, -0.42],
                  [-0.52, 0.95, 0.50],
                  [0.44, -0.38, 0.92]])
    X_train = S_train @ A.T

    # Run real CS229 sigmoid-CDF stochastic gradient ascent and store W trajectory
    W = 0.75 * np.eye(3)
    W_snaps = [W.copy()]
    n_unmix = n - 14
    total_epochs = n_unmix - 1
    for _ in range(total_epochs):
        for idx in rng.permutation(m_train):
            x = X_train[idx]
            g = 1.0 / (1.0 + np.exp(-(W @ x)))
            W = W + 0.0045 * (np.outer(1.0 - 2.0 * g, x) + np.linalg.inv(W.T))
        W_snaps.append(W.copy())

    # Pick a crisp subset of points (including outer points near cube faces) for 3D plotting
    radii = np.max(np.abs(S_train), axis=1)
    vis_idx = np.r_[np.argsort(radii)[-95:], rng.choice(m_train - 95, 75, replace=False)]
    S_vis = S_train[vis_idx]
    pt_colors = np.clip((S_vis / half + 1.0) * 0.42 + 0.08, 0.05, 0.92)
    cube_segs = _cube_segments(half=half)

    lim = 2.85
    ax = s.ax3d(labels=("axis 1", "axis 2", "axis 3"), elev=23, azim=-54,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))

    def _row_best_corr(S_est):
        corr = np.abs(np.corrcoef(S_train.T, S_est.T)[:3, 3:])
        return float(np.mean(np.max(corr, axis=1)))

    def update(i):
        s.clear()
        if i <= 10:
            tau = float(ease(i / 10.0))
            M = lerp(np.eye(3), A, tau)
            box_col = C["blue"] if i == 0 else C["orange"]
            pts = S_vis @ M.T
            segs = _apply_segs(cube_segs, M)
            status_txt = "sources s in cube" if i == 0 else f"mixing x = As ({int(round(100 * tau)):3d}%)"
        elif i <= 13:
            M = A
            box_col = C["red"]
            pts = S_vis @ M.T
            segs = _apply_segs(cube_segs, M)
            mc = _row_best_corr(X_train)
            status_txt = f"mixed box x = As   corr = {mc:.2f}"
        else:
            ep = i - 14
            Wk = W_snaps[ep]
            M_raw = Wk @ A
            # Uniform scalar normalisation for display so box stays at half-width 1.65
            corners = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
            max_ext = float(np.max(np.abs(corners @ M_raw.T)))
            M = M_raw * (half / max(max_ext, 1e-6))
            box_col = C["green"] if ep > total_epochs - 6 else C["purple"]
            pts = S_vis @ M.T
            segs = _apply_segs(cube_segs, M)
            mc = _row_best_corr(X_train @ Wk.T)
            status_txt = f"ICA epoch {ep:2d}   max|corr| = {mc:.3f}"

        s.add(ax.plot(segs[:, 0], segs[:, 1], segs[:, 2], color=box_col, lw=1.55, alpha=0.9))
        s.add(ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pt_colors, s=18, depthshade=False))
        s.status(status_txt)
        s.spin(i, n, sweep=34)

    return update


@animation("Gaussian Symmetry vs Cube Cloud",
           caption=r"""बाईं ओर गोलाकार Gaussian बादल $\mathcal{N}(0, I)$ है और दाईं ओर स्वतंत्र non-Gaussian घन (cube)
           बादल। जब दोनों को किसी orthogonal rotation $R(\theta)$ से घुमाया जाता है, तो बाएँ Gaussian गोले का आकार
           बिल्कुल वैसा ही दिखता है क्योंकि $R R^T = I$। इसलिए Gaussian sources में $A$ और $AR$ के बीच फ़र्क करना
           असंभव है। दाएँ घन के कोने और फलक घूमते ही तिरछे हो जाते हैं, जिससे ICA सही अक्ष पहचान लेता है।""",
           frames=40, keys=(0.0, 0.5, 1.0))
def gaussian_vs_cube(s, n):
    rng = np.random.default_rng(33)
    m = 150
    g_pts = rng.normal(0.0, 0.50, size=(m, 3))
    norms = np.linalg.norm(g_pts, axis=1)
    g_pts = g_pts[norms < 1.32][:120]

    c_pts = rng.uniform(-0.98, 0.98, size=(120, 3))
    cube_segs = _cube_segments(half=1.02)

    cL = np.array([-2.15, 0.0, 0.0])
    cR = np.array([2.15, 0.0, 0.0])

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=23, azim=-68,
                xlim=(-3.9, 3.9), ylim=(-2.1, 2.1), zlim=(-2.1, 2.1), ticks=False)

    # Translucent reference sphere around Gaussian cloud
    u_ang, v_ang = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 14))
    rad = 1.12
    sx = cL[0] + rad * np.cos(u_ang) * np.sin(v_ang)
    sy = cL[1] + rad * np.sin(u_ang) * np.sin(v_ang)
    sz = cL[2] + rad * np.cos(v_ang)
    ax.plot_wireframe(sx, sy, sz, color=C["blue"], linewidth=0.45, alpha=0.28)

    ax.text(cL[0], 0.0, 1.62, "Gaussian (ambiguous)", color=C["blue"], fontsize=9, ha="center", fontweight="bold")
    ax.text(cR[0], 0.0, 1.62, "Cube (identifiable)", color=C["orange"], fontsize=9, ha="center", fontweight="bold")

    rot_axis = (0.32, 0.44, 0.84)
    triad = 1.25 * np.eye(3)

    def update(i):
        s.clear()
        deg = 56.0 * ease(i / (n - 1))
        R = _axis_angle_rot(rot_axis, np.deg2rad(deg))

        g_rot = g_pts @ R.T + cL
        c_rot = c_pts @ R.T + cR
        segs_rot = _apply_segs(cube_segs, R, shift=cR)

        s.add(ax.scatter(g_rot[:, 0], g_rot[:, 1], g_rot[:, 2], color=C["blue"], s=14, alpha=0.68, depthshade=False))
        s.add(ax.scatter(c_rot[:, 0], c_rot[:, 1], c_rot[:, 2], color=C["orange"], s=14, alpha=0.72, depthshade=False))
        s.add(ax.plot(segs_rot[:, 0], segs_rot[:, 1], segs_rot[:, 2], color=C["dark"], lw=1.35, alpha=0.85))

        for k_ax, col in enumerate((C["red"], C["green"], C["purple"])):
            vec = triad[k_ax] @ R.T
            s.add(arrow(ax, cL, vec, color=col, lw=1.7))
            s.add(arrow(ax, cR, vec, color=col, lw=1.7))

        s.status(rf"rotation $\theta$ = {deg:4.1f} deg   Gaussian density unchanged")
        s.spin(i, n, sweep=16)

    return update
