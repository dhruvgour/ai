"""Lecture 14: Factor Analysis: 3D animations."""
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, animation, arrow, ease, grid, lerp


# ---------------------------------------------------------------- helpers

def _spokes(pts, ends):
    """NaN-separated coordinates for segments pts[i] -> ends[i] in one plot call."""
    nan = np.full(len(pts), np.nan)
    return [np.c_[pts[:, d], ends[:, d], nan].ravel() for d in range(pts.shape[1])]


def _unit_sphere(nu=26, nv=16):
    """Unit sphere mesh of shape (nv, nu, 3), grid <= 40x40."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    U, V = np.meshgrid(u, v)
    return np.stack([np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V)], axis=-1)


_SPHERE = _unit_sphere(26, 16)
_ANG = np.linspace(0, 2 * np.pi, 72)
_CIRCLE2 = np.c_[np.cos(_ANG), np.sin(_ANG)]


def _ellipsoid_mesh(mu, Sigma, scale=1.85):
    """3D ellipsoid points mu + scale * Sigma^(1/2) * sphere; safe for singular Sigma."""
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.maximum(vals, 0.0)
    radii = np.sqrt(vals)
    pts = mu + scale * (_SPHERE * radii) @ vecs.T
    return pts, vals, vecs


def _avg_ll(X, mu, Sigma):
    """Average Gaussian log-likelihood per example."""
    d = X - mu
    maha = np.einsum("ij,ij->i", d, np.linalg.solve(Sigma, d.T).T)
    return -0.5 * float(np.mean(maha + np.linalg.slogdet(Sigma)[1] + X.shape[1] * np.log(2 * np.pi)))


def _plane_angle_deg(L1, L2):
    """Angle in degrees between two 2D subspaces in R^3 spanned by columns of L1 and L2."""
    n1 = np.cross(L1[:, 0], L1[:, 1])
    n2 = np.cross(L2[:, 0], L2[:, 1])
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))))


# ---------------------------------------------------------------- 1. Generative model

@animation(r"FA: $\mu + \Lambda z + \epsilon$",
           caption=r"""यहाँ Factor Analysis की तीन-चरणीय कहानी ($n=3, k=2$) दिखाई गई है। शुरू में कम-आयामी latent बिंदु
           $z^{(i)} \sim \mathcal{N}(0, I)$ एक सपाट 2D चादर पर हैं। पहले चरण में $\mu + \Lambda z$ इस चादर को खींचकर और
           घुमाकर 3D space के एक तिरछे 2D subspace में पहुँचा देता है, जहाँ नारंगी और हरे तीर $\Lambda$ के दो columns हैं।
           दूसरे चरण में स्वतंत्र diagonal noise $\epsilon^{(i)} \sim \mathcal{N}(0, \Psi)$ हर बिंदु को उस चादर से बाहर धकेलता है
           (लाल रेखाएँ यही विस्थापन दिखाती हैं), जिससे एक चपटा 3D बादल बनता है।""",
           frames=42, keys=(0.0, 0.48, 1.0))
def fa_generative(s, n):
    rng = np.random.default_rng(14)
    m = 56
    Z = np.clip(rng.normal(0, 0.9, (m, 2)), -1.95, 1.95)
    mu = np.array([0.25, -0.2, 0.35])
    Lam = np.array([[1.25, 0.30], [0.35, 1.15], [0.85, -0.75]])
    psi = np.array([0.22, 0.20, 0.52])
    eps = rng.normal(0, 1.0, (m, 3)) * np.sqrt(psi)

    P0 = np.c_[Z[:, 0], Z[:, 1], np.zeros(m)]
    P1 = mu + Z @ Lam.T

    GZ1, GZ2 = grid((-2.1, 2.1), (-2.1, 2.1), 12)
    G_flat = np.stack([GZ1, GZ2, np.zeros_like(GZ1)], axis=-1)
    G_map = mu + GZ1[..., None] * Lam[:, 0] + GZ2[..., None] * Lam[:, 1]

    cols = colormaps["viridis"]((Z[:, 0] + Z[:, 1] + 3.2) / 6.4)
    lim = (-3.8, 3.8)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=23, azim=-58,
                xlim=lim, ylim=lim, zlim=lim, ticks=False)

    mid = int(round(0.48 * (n - 1)))

    def update(i):
        s.clear()
        if i <= mid:
            t1 = ease(i / max(1, mid))
            t2 = 0.0
        else:
            t1 = 1.0
            t2 = ease((i - mid) / max(1, n - 1 - mid))

        G = lerp(G_flat, G_map, t1)
        s.add(ax.plot_surface(G[..., 0], G[..., 1], G[..., 2], color=C["blue"], alpha=0.24,
                              linewidth=0, shade=False))
        s.add(ax.plot_wireframe(G[..., 0], G[..., 1], G[..., 2], rstride=2, cstride=2,
                                color=C["blue"], linewidth=0.55, alpha=0.45))

        c_cur = lerp(np.zeros(3), mu, t1)
        v1 = lerp(np.array([1.0, 0.0, 0.0]), Lam[:, 0], t1)
        v2 = lerp(np.array([0.0, 1.0, 0.0]), Lam[:, 1], t1)
        s.add(arrow(ax, c_cur, 1.5 * v1, color=C["orange"], lw=2.1))
        s.add(arrow(ax, c_cur, 1.5 * v2, color=C["green"], lw=2.1))

        base = lerp(P0, P1, t1)
        pts = base + t2 * eps
        if t2 > 0.01:
            s.add(ax.scatter(*base.T, color=C["blue"], s=8, alpha=0.55, depthshade=False))
            s.add(ax.plot(*_spokes(base, pts), color=C["red"], lw=0.85, alpha=0.75))
        s.add(ax.scatter(*pts.T, c=cols, s=24, edgecolor="black", linewidth=0.35, depthshade=False))

        if i == 0:
            s.status("step 1: sample latent z ~ N(0, I)")
        elif t2 == 0.0:
            s.status(f"step 2: map via mu + Lambda z ({int(round(100 * t1))}%)")
        else:
            s.status(f"step 3: add noise eps ~ N(0, Psi) ({int(round(100 * t2))}%)")
        s.spin(i, n, sweep=36)

    return update


# ---------------------------------------------------------------- 2. Singular covariance collapse

@animation("Singular covariance as m drops",
           caption=r"""यहाँ $n = 3$ dimensions में sample covariance matrix $S$ का $2\sigma$ ellipsoid दिखाया गया है जब
           examples की संख्या $m$ धीरे-धीरे घटती है। जब $m = 22$ है, तो तीनों दिशाओं में फैलाव है, $\mathrm{rank}(S) = 3$ है
           और ellipsoid पूरी तरह फूला हुआ है। जैसे ही $m$ घटकर $3$ ($m = n$) रह जाता है, तीनों बिंदु एक ही 2D समतल में आ
           जाते हैं: सबसे छोटा eigenvalue शून्य हो जाता है और ellipsoid पिचककर चपटी तश्तरी बन जाता है। अंत में $m = 2$ पर
           यह सिमटकर एक 1D सुई रह जाता है और $S^{-1}$ पूरी तरह अपरिभाषित हो जाता है।""",
           frames=42, keys=(0.0, 0.62, 1.0))
def singular_collapse(s, n):
    rng = np.random.default_rng(21)
    # Fix first 3 points to form a clear tilted triangle in R^3, then sample 19 more around them
    base3 = np.array([[-1.9, -1.1, -0.9], [2.0, -0.7, 1.1], [-0.2, 1.9, 0.6]])
    cov0 = np.array([[1.7, 0.65, 0.55], [0.65, 1.35, 0.25], [0.55, 0.25, 0.95]])
    rest = rng.multivariate_normal(np.zeros(3), cov0, 19)
    X = np.clip(np.vstack([base3, rest]), -3.5, 3.5)
    M = len(X)

    # Piecewise schedule for continuous sample count c in [2.0, 22.0]
    # k0=0.0 -> c=22 (rank 3); k1=0.62 -> c=3 (rank 2 pancake); k2=1.0 -> c=2 (rank 1 needle)
    k1_idx = int(round(0.62 * (n - 1)))
    cs = np.empty(n)
    for i in range(n):
        if i <= k1_idx - 2:
            u = i / max(1, k1_idx - 2)
            cs[i] = 22.0 - 19.0 * (u ** 0.68)
        elif i <= k1_idx + 2:
            cs[i] = 3.0
        elif i <= n - 4:
            u = (i - (k1_idx + 2)) / max(1, (n - 4) - (k1_idx + 2))
            cs[i] = 3.0 - 1.0 * ease(u)
        else:
            cs[i] = 2.0

    lim = (-3.8, 3.8)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=24, azim=-62,
                xlim=lim, ylim=lim, zlim=lim, ticks=False)

    def update(i):
        s.clear()
        c = cs[i]
        k_int = int(np.floor(c))
        frac = c - k_int
        w = np.zeros(M)
        w[:k_int] = 1.0
        if k_int < M:
            w[k_int] = frac
        w_sum = w.sum()
        mu = (w[:, None] * X).sum(0) / w_sum
        Xc = X - mu
        S = (Xc * w[:, None]).T @ Xc / w_sum

        mesh, vals, vecs = _ellipsoid_mesh(mu, S, scale=1.8)
        rk = int(np.sum(vals > 1e-5))
        col = C["blue"] if rk == 3 else (C["orange"] if rk == 2 else C["red"])

        if rk >= 2:
            s.add(ax.plot_surface(mesh[..., 0], mesh[..., 1], mesh[..., 2],
                                  color=col, alpha=0.34, linewidth=0, shade=True))
            s.add(ax.plot_wireframe(mesh[..., 0], mesh[..., 1], mesh[..., 2],
                                    rstride=4, cstride=4, color=col, linewidth=0.5, alpha=0.45))
        if rk == 2:
            # Highlight sharp rim of the collapsed 2D disc
            rim = mu + 1.8 * (_CIRCLE2 * np.sqrt(vals[1:])) @ vecs[:, 1:].T
            s.add(ax.plot(*rim.T, color=C["orange"], lw=2.2))
        if rk == 1:
            # Highlight the collapsed 1D needle
            dvec = 1.8 * np.sqrt(vals[2]) * vecs[:, 2]
            seg = np.vstack([mu - dvec, mu + dvec])
            s.add(ax.plot(*seg.T, color=C["red"], lw=3.2))

        for d in range(3):
            length = 1.8 * np.sqrt(vals[d])
            if length > 0.08:
                s.add(arrow(ax, mu, length * vecs[:, d], color=C["dark"], lw=1.5))

        active = int(np.ceil(c - 1e-9))
        if active < M:
            s.add(ax.scatter(*X[active:].T, color=C["gray"], s=10, alpha=0.25, depthshade=False))
        s.add(ax.scatter(*X[:active].T, color=C["dark"], s=28, edgecolor="white",
                         linewidth=0.45, depthshade=False))

        s.status(f"m = {active:2d} (n=3)   rank = {rk}   min eig = {vals[0]:.2f}")
        s.spin(i, n, sweep=38)

    return update


# ---------------------------------------------------------------- 3. Full vs Diagonal vs Spherical

@animation(r"Full, Diagonal, Spherical $\Sigma$",
           caption=r"""तिरछे फैले हुए 3D data पर तीन अलग-अलग covariance models की तुलना देखिए। शुरू में **Full** $\Sigma$
           (नीला, 6 parameters) सभी correlations पकड़कर बादल की तिरछी दिशा में बिल्कुल फिट बैठता है। बीच में जब
           off-diagonal entries शून्य होती हैं, तो **Diagonal** $\Sigma$ (नारंगी, 3 parameters) सीधा होकर coordinate axes के
           समानांतर फैल जाता है और खाली जगह घेर लेता है। अंत में **Spherical** $\sigma^2 I$ (बैंगनी, 1 parameter) तीनों
           दिशाओं की variance भी बराबर कर देता है और गोल गेंद बन जाता है, जिससे log-likelihood और गिर जाता है।""",
           frames=44, keys=(0.0, 0.5, 1.0))
def covariance_restrictions(s, n):
    rng = np.random.default_rng(9)
    m = 95
    L0 = np.array([[1.45, 0.20], [0.95, -0.42], [0.55, 0.18]])
    Z = rng.normal(size=(m, 2))
    X = Z @ L0.T + rng.normal(size=(m, 3)) * np.array([0.32, 0.30, 0.25])
    X -= X.mean(0)
    X = np.clip(X, -3.7, 3.7)

    mu = np.zeros(3)
    S_full = X.T @ X / m
    S_diag = np.diag(np.diag(S_full))
    S_sph = np.eye(3) * (np.trace(S_full) / 3.0)

    lim = (-3.9, 3.9)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=24, azim=-56,
                xlim=lim, ylim=lim, zlim=lim, ticks=False)
    ax.scatter(*X.T, color=C["dark"], s=11, alpha=0.72, depthshade=False)

    # Static floor reference contour of Full Sigma in (x1, x2)
    ref2 = 1.85 * _CIRCLE2 @ np.linalg.cholesky(S_full[:2, :2]).T
    ax.plot(ref2[:, 0], ref2[:, 1], lim[0], color=C["blue"], lw=1.1, ls="--", alpha=0.65,
            label="Full contour (floor)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def state_at(i):
        t = i / max(1, n - 1)
        if t <= 0.14:
            return S_full, C["blue"], "Full (6 params)"
        if t < 0.42:
            u = ease((t - 0.14) / 0.28)
            return lerp(S_full, S_diag, u), C["teal"], "morph: Full -> Diag"
        if t <= 0.58:
            return S_diag, C["orange"], "Diagonal (3 params)"
        if t < 0.86:
            u = ease((t - 0.58) / 0.28)
            return lerp(S_diag, S_sph, u), C["pink"], "morph: Diag -> Sph"
        return S_sph, C["purple"], "Spherical (1 param)"

    def update(i):
        s.clear()
        Sig, col, name = state_at(i)
        mesh, vals, vecs = _ellipsoid_mesh(mu, Sig, scale=1.85)
        s.add(ax.plot_surface(mesh[..., 0], mesh[..., 1], mesh[..., 2],
                              color=col, alpha=0.32, linewidth=0, shade=True))
        s.add(ax.plot_wireframe(mesh[..., 0], mesh[..., 1], mesh[..., 2],
                                rstride=4, cstride=4, color=col, linewidth=0.5, alpha=0.42))
        for d in range(3):
            s.add(arrow(ax, mu, 1.85 * np.sqrt(vals[d]) * vecs[:, d], color=col, lw=1.9))

        c2 = 1.85 * _CIRCLE2 @ np.linalg.cholesky(Sig[:2, :2]).T
        s.add(ax.plot(c2[:, 0], c2[:, 1], lim[0], color=col, lw=1.8))

        ll = _avg_ll(X, mu, Sig)
        s.status(f"{name}   avg log L = {ll:.2f}")
        s.spin(i, n, sweep=34)

    return update


# ---------------------------------------------------------------- 4. Conditional Gaussian slice

@animation(r"Conditional slice $p(x_1 | x_2)$",
           caption=r"""हरा-नीला टीला दो सह-संबंधित variables $(x_1, x_2)$ की 2D joint Gaussian density $p(x_1, x_2)$ है।
           जैसे-जैसे पारदर्शी काट-समतल अलग-अलग $x_2$ पर खिसकता है, नारंगी वक्र टीले का सीधा अनुप्रस्थ काट $p(x_1, x_2)$
           दिखाता है और ऊँचा नीला वक्र normalize करने के बाद बनी conditional density $p(x_1 \mid x_2)$ है। फ़र्श की हरी
           रेखा के साथ नीली घंटी का केंद्र $\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)$ सीधी रेखा में खिसकता है,
           पर घंटी की चौड़ाई और ऊँचाई ($\Sigma_{1|2}$) हर $x_2$ पर बिल्कुल स्थिर रहती है।""",
           frames=41, keys=(0.0, 0.5, 1.0))
def conditional_slice(s, n):
    Sig = np.array([[0.85, 0.62], [0.62, 0.80]])
    invS = np.linalg.inv(Sig)
    norm2 = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(Sig)))
    slope = Sig[0, 1] / Sig[1, 1]
    var_cond = Sig[0, 0] - Sig[0, 1] ** 2 / Sig[1, 1]
    std_cond = np.sqrt(var_cond)

    xlim, ylim, zlim = (-3.0, 3.0), (-2.4, 2.4), (0.0, 0.70)
    GX, GY = grid(xlim, ylim, 38)
    P = np.stack([GX, GY], axis=-1)
    GZ = norm2 * np.exp(-0.5 * np.einsum("...i,ij,...j->...", P, invS, P))

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", "density"), elev=28, azim=-58,
                xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_zticks([0.0, 0.3, 0.6])
    ax.plot_surface(GX, GY, GZ, cmap="viridis", alpha=0.35, linewidth=0, rstride=1, cstride=1)
    ax.contour(GX, GY, GZ, levels=5, zdir="z", offset=0.0, cmap="viridis", linewidths=0.7, alpha=0.7)

    ys_line = np.linspace(ylim[0], ylim[1], 50)
    ax.plot(slope * ys_line, ys_line, 0.0, color=C["green"], lw=1.5, ls="--",
            label=r"$\mu_{1|2}(x_2)$")
    ax.plot([], [], [], color=C["orange"], lw=2.0, label=r"slice $p(x_1, x_2)$")
    ax.plot([], [], [], color=C["blue"], lw=2.2, label=r"$p(x_1 | x_2)$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False)

    xs = np.linspace(xlim[0], xlim[1], 90)
    c_vals = np.linspace(-1.75, 1.75, n)

    def update(i):
        s.clear()
        c = float(c_vals[i])
        yc = np.full_like(xs, c)
        pts = np.c_[xs, yc]
        z_joint = norm2 * np.exp(-0.5 * np.einsum("ij,jk,ik->i", pts, invS, pts))

        m12 = slope * c
        z_cond = np.exp(-0.5 * ((xs - m12) / std_cond) ** 2) / (np.sqrt(2.0 * np.pi) * std_cond)
        peak = 1.0 / (np.sqrt(2.0 * np.pi) * std_cond)

        # Translucent vertical cutting curtain at x2 = c
        curtain = [[(xlim[0], c, 0.0), (xlim[1], c, 0.0),
                    (xlim[1], c, 0.68), (xlim[0], c, 0.68)]]
        s.add(ax.add_collection3d(Poly3DCollection(curtain, facecolor=C["gray"], alpha=0.10,
                                                   edgecolor="none")))

        # Filled joint cross-section + orange curve
        verts_j = [list(zip(xs, yc, z_joint)) + [(xs[-1], c, 0.0), (xs[0], c, 0.0)]]
        s.add(ax.add_collection3d(Poly3DCollection(verts_j, facecolor=C["orange"], alpha=0.28,
                                                   edgecolor="none")))
        s.add(ax.plot(xs, yc, z_joint, color=C["orange"], lw=2.1))

        # Normalized conditional Gaussian curve + mean & constant-width markers
        s.add(ax.plot(xs, yc, z_cond, color=C["blue"], lw=2.4))
        s.add(ax.plot([m12, m12], [c, c], [0.0, peak], color=C["green"], lw=1.4, ls=":"))
        s.add(ax.scatter([m12], [c], [0.0], color=C["green"], s=38, edgecolor="black",
                         linewidth=0.5, depthshade=False))
        h_half = peak * np.exp(-0.5)
        s.add(ax.plot([m12 - std_cond, m12 + std_cond], [c, c], [h_half, h_half],
                      color=C["dark"], lw=2.0))

        s.status(f"x2 = {c:+.2f}   mean = {m12:+.2f}   var = {var_cond:.2f}")
        s.spin(i, n, sweep=24)

    return update


# ---------------------------------------------------------------- 5. EM rotates the latent subspace

@animation("EM rotates the latent subspace",
           caption=r"""यहाँ $n = 3$ observed space में चपटे बादल पर $k = 2$ factors वाला Factor Analysis EM चल रहा है।
           शुरू में नीला 2D subspace $\mu + \Lambda z$ गलत कोण पर खड़ा है, इसलिए बिंदुओं से उनके E-step projections
           $\mu + \Lambda \mu_{z|x}$ तक की लाल दूरियाँ बहुत लंबी हैं। हर EM iteration में E-step हर बिंदु का posterior
           $\mathcal{N}(\mu_{z|x}, \Sigma_{z|x})$ निकालता है और M-step $E[zz^T] = \mu_{z|x}\mu_{z|x}^T + \Sigma_{z|x}$ का
           उपयोग करके $\Lambda$ और $\Psi$ को घुमाता व ढालता है। कुछ ही iterations में subspace ठीक असली समतल पर
           बैठ जाता है और log-likelihood लगातार बढ़ता है।""",
           frames=44, keys=(0.0, 0.32, 1.0))
def em_subspace(s, n):
    rng = np.random.default_rng(14)
    m, k = 90, 2
    L_true = np.array([[1.45, 0.25], [0.30, 1.35], [0.55, -0.45]])
    psi_true = np.array([0.05, 0.07, 0.10])
    Z_true = rng.normal(size=(m, k))
    X = Z_true @ L_true.T + rng.normal(size=(m, 3)) * np.sqrt(psi_true)
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / m

    # Deliberately misoriented initial subspace (steeply tilted away from the pancake)
    Lam = np.array([[0.22, -0.12], [-0.18, 0.24], [1.05, 0.65]])
    psi = np.diag(S).copy()

    n_it = 26
    trace = []
    for _ in range(n_it + 1):
        Sig = Lam @ Lam.T + np.diag(psi)
        B = Lam.T @ np.linalg.inv(Sig)
        Mz = Xc @ B.T
        Sz = np.eye(k) - B @ Lam
        ll = _avg_ll(X, mu, Sig)
        ang = _plane_angle_deg(Lam, L_true)
        trace.append((Lam.copy(), Mz.copy(), ll, ang))

        Ezz = Mz.T @ Mz / m + Sz
        Cxz = Xc.T @ Mz / m
        Lam = Cxz @ np.linalg.inv(Ezz)
        Phi = S - Cxz @ Lam.T - Lam @ Cxz.T + Lam @ Ezz @ Lam.T
        psi = np.maximum(np.diag(Phi), 1e-4)

    GZ1, GZ2 = grid((-2.15, 2.15), (-2.15, 2.15), 10)
    lim = (-3.8, 3.8)
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=22, azim=-62,
                xlim=lim, ylim=lim, zlim=lim, ticks=False)
    ax.scatter(*X.T, color=C["dark"], s=14, depthshade=False)

    # Smooth continuous progression through EM history, slower in early steps where subspace turns fast
    prog = n_it * (np.linspace(0, 1, n - 3) ** 1.85)
    prog = np.r_[prog, np.full(3, float(n_it))]

    def update(i):
        s.clear()
        p = prog[i]
        it = min(int(p), n_it - 1)
        frac = ease(p - it) if p < n_it else 1.0
        L_a, Mz_a, _, _ = trace[it]
        L_b, Mz_b, _, _ = trace[it + 1]
        L_cur = lerp(L_a, L_b, frac)
        Mz_cur = lerp(Mz_a, Mz_b, frac)
        it_disp = int(round(p))
        ll_cur = lerp(trace[it][2], trace[it + 1][2], frac)
        ang_cur = _plane_angle_deg(L_cur, L_true)

        G = mu + GZ1[..., None] * L_cur[:, 0] + GZ2[..., None] * L_cur[:, 1]
        G = np.clip(G, lim[0], lim[1])
        s.add(ax.plot_surface(G[..., 0], G[..., 1], G[..., 2], color=C["blue"], alpha=0.26,
                              linewidth=0, shade=False))
        s.add(ax.plot_wireframe(G[..., 0], G[..., 1], G[..., 2], rstride=2, cstride=2,
                                color=C["blue"], linewidth=0.55, alpha=0.45))

        s.add(arrow(ax, mu, 1.5 * L_cur[:, 0], color=C["orange"], lw=2.1))
        s.add(arrow(ax, mu, 1.5 * L_cur[:, 1], color=C["teal"], lw=2.1))

        proj = np.clip(mu + Mz_cur @ L_cur.T, lim[0], lim[1])
        s.add(ax.plot(*_spokes(X, proj), color=C["red"], lw=0.65, alpha=0.6))
        s.add(ax.scatter(*proj.T, color=C["blue"], s=7, alpha=0.65, depthshade=False))

        s.status(f"iter {it_disp:2d}   angle err {ang_cur:4.1f} deg   ll = {ll_cur:.2f}")
        s.spin(i, n, sweep=34)

    return update
